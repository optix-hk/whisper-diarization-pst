import asyncio
import io
import logging
import os
import re
import subprocess
import tempfile
import threading
import time

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from diarization import DiarizationResult, MSDDDiarizer, SortformerDiarizer
from helpers import (
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
)
from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels
from speaker_store import SpeakerEmbeddingStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPUTE_TYPES = {"cpu": "int8", "cuda": "float16"}

app = FastAPI(title="Whisper Diarization API")


class TranscriptionSegment(BaseModel):
    speaker: str
    start_time: float
    end_time: float
    text: str


class TranscriptionResult(BaseModel):
    segments: list[TranscriptionSegment]
    srt: str | None = None
    language: str
    processing_time_seconds: float


class SpeakerInfo(BaseModel):
    name: str
    sample_count: int
    created_at: str | None = None
    updated_at: str | None = None


class SpeakerList(BaseModel):
    speakers: list[SpeakerInfo]


class RenameRequest(BaseModel):
    new_name: str
    force: bool = False


class Models:
    def __init__(self):
        self.whisper_model = None
        self.whisper_pipeline = None
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None
        self.diarizer_model = None
        self.speaker_db = os.environ.get("SPEAKER_DB", "~/.whisper-diarization/speakers.db")
        self.speaker_persistence = os.environ.get("SPEAKER_PERSISTENCE", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self.shared_store: SpeakerEmbeddingStore | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


models = Models()

whisper_semaphore = asyncio.Semaphore(1)
diarizer_semaphore = asyncio.Semaphore(1)
db_lock = threading.Lock()
background_tasks: set[asyncio.Task] = set()


@app.on_event("startup")
def load_models():
    t0 = time.time()
    whisper_model_name = os.environ.get("WHISPER_MODEL", "medium.en")
    device = models.device
    compute_type = COMPUTE_TYPES[device]

    logger.info(f"Loading Whisper model ({whisper_model_name}) on {device}...")
    models.whisper_model = faster_whisper.WhisperModel(
        whisper_model_name, device=device, compute_type=compute_type
    )
    models.whisper_pipeline = faster_whisper.BatchedInferencePipeline(models.whisper_model)
    logger.info(f"Whisper model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    logger.info("Loading alignment model...")
    models.alignment_model, models.alignment_tokenizer = load_alignment_model(
        device, dtype=torch.float16 if device == "cuda" else torch.float32
    )
    logger.info(f"Alignment model loaded in {time.time() - t1:.1f}s")

    t2 = time.time()
    logger.info("Loading punctuation model...")
    models.punct_model = PunctuationModel(model="kredor/punctuate-all")
    logger.info(f"Punctuation model loaded in {time.time() - t2:.1f}s")

    diarizer_type = os.environ.get("DIARIZER", "msdd")
    t3 = time.time()
    logger.info(f"Loading diarizer ({diarizer_type})...")
    if diarizer_type == "msdd":
        models.diarizer_model = MSDDDiarizer(device=device)
    elif diarizer_type == "sortformer":
        models.diarizer_model = SortformerDiarizer(device=device)
    logger.info(f"Diarizer loaded in {time.time() - t3:.1f}s")

    if models.speaker_persistence:
        models.shared_store = SpeakerEmbeddingStore(models.speaker_db)
        speaker_count = len(models.shared_store.list_speakers())
        logger.info(
            f"Speaker persistence enabled (DB: {models.speaker_db}, "
            f"{speaker_count} stored profiles)"
        )
    else:
        logger.info("Speaker persistence disabled (SPEAKER_PERSISTENCE=0)")

    logger.info(f"All models loaded in {time.time() - t0:.1f}s")


@app.on_event("shutdown")
async def shutdown():
    for task in background_tasks:
        task.cancel()
    if background_tasks:
        await asyncio.wait(background_tasks, timeout=10.0)
    background_tasks.clear()
    if models.shared_store is not None:
        models.shared_store.close()
        models.shared_store = None


@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": models.device,
        "whisper_loaded": models.whisper_model is not None,
        "alignment_loaded": models.alignment_model is not None,
        "punct_loaded": models.punct_model is not None,
        "diarizer_loaded": models.diarizer_model is not None,
        "speaker_persistence": models.speaker_persistence,
        "speaker_db": models.speaker_db,
        "pending_db_updates": len(background_tasks),
    }


@app.get("/speakers", response_model=SpeakerList)
def list_speakers():
    if models.shared_store is None:
        raise HTTPException(status_code=503, detail="Speaker persistence is disabled")
    with db_lock:
        speakers = models.shared_store.list_speakers()
    return SpeakerList(speakers=[SpeakerInfo(**s) for s in speakers])


@app.get("/speakers/{name}", response_model=SpeakerInfo)
def show_speaker(name: str):
    if models.shared_store is None:
        raise HTTPException(status_code=503, detail="Speaker persistence is disabled")
    with db_lock:
        speakers = models.shared_store.list_speakers()
    for s in speakers:
        if s["name"] == name:
            return SpeakerInfo(**s)
    raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")


@app.delete("/speakers/{name}")
def delete_speaker(name: str):
    if models.shared_store is None:
        raise HTTPException(status_code=503, detail="Speaker persistence is disabled")
    with db_lock:
        try:
            models.shared_store.delete_speaker(name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return {"detail": f"Deleted '{name}'"}


@app.post("/speakers/{name}/rename")
def rename_speaker(name: str, request: RenameRequest):
    if models.shared_store is None:
        raise HTTPException(status_code=503, detail="Speaker persistence is disabled")
    if name == request.new_name:
        return {"detail": f"Speaker already named '{name}'"}
    with db_lock:
        try:
            existing = {s["name"] for s in models.shared_store.list_speakers()}
            if request.new_name in existing:
                if not request.force:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Speaker '{request.new_name}' already exists. "
                            f"Use force=true to merge '{name}' into '{request.new_name}'."
                        ),
                    )
                models.shared_store.merge_speakers(name, request.new_name)
                return {"detail": f"Merged '{name}' into '{request.new_name}'"}
            models.shared_store.rename_speaker(name, request.new_name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return {"detail": f"Renamed '{name}' to '{request.new_name}'"}


def _run_whisper_alignment(audio_waveform, language, batch_size, suppress_numerals):
    suppress_tokens = (
        find_numeral_symbol_tokens(models.whisper_model.hf_tokenizer) if suppress_numerals else [-1]
    )
    whisper_language = language.lower() if language else None

    if batch_size > 0:
        transcript_segments, info = models.whisper_pipeline.transcribe(
            audio_waveform,
            whisper_language,
            suppress_tokens=suppress_tokens,
            batch_size=batch_size,
        )
    else:
        transcript_segments, info = models.whisper_model.transcribe(
            audio_waveform,
            whisper_language,
            suppress_tokens=suppress_tokens,
            vad_filter=True,
        )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    detected_language = info.language

    emissions, stride = generate_emissions(
        models.alignment_model,
        torch.from_numpy(audio_waveform)
        .to(models.alignment_model.dtype)
        .to(models.alignment_model.device),
        batch_size=batch_size,
    )

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=langs_to_iso[detected_language],
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        models.alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    return word_timestamps, detected_language


def _run_diarization(audio_waveform):
    return models.diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))


def _build_segments(wsm, speaker_ts, detected_language, include_srt, override_speaker=None):
    if detected_language in punct_model_langs:
        words_list = [x["word"] for x in wsm]
        labeled_words = models.punct_model.predict(words_list)
        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)

    if override_speaker is not None:
        for entry in wsm:
            entry["speaker"] = override_speaker

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    if override_speaker is not None:
        for seg in ssm:
            seg["speaker"] = override_speaker

    result_segments = [
        TranscriptionSegment(
            speaker=s["speaker"],
            start_time=s["start_time"],
            end_time=s["end_time"],
            text=s["text"].strip(),
        )
        for s in ssm
    ]

    srt_text = None
    if include_srt:
        srt_buf = io.StringIO()
        from helpers import write_srt

        write_srt(ssm, srt_buf)
        srt_text = srt_buf.getvalue()

    return TranscriptionResult(
        segments=result_segments,
        srt=srt_text,
        language=detected_language,
        processing_time_seconds=0.0,
    )


@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    batch_size: int = Form(8),
    suppress_numerals: bool = Form(False),
    skip_diarization: bool = Form(False),
    include_srt: bool = Form(True),
    no_persist: bool = Form(False),
    match_threshold: float = Form(0.75),
    speaker_name: str | None = Form(None),
):
    if speaker_name is not None and skip_diarization:
        raise HTTPException(
            status_code=400,
            detail=(
                "speaker_name requires skip_diarization=false "
                "(diarization runs in background to extract embedding)"
            ),
        )

    t_start = time.time()
    loop = asyncio.get_running_loop()

    upload_ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=upload_ext, delete=False) as tmp:
        tmp.write(audio.file.read())
        upload_path = tmp.name

    wav_path = upload_path
    needs_cleanup = [upload_path]

    if upload_ext not in (".wav", ".flac"):
        wav_path = upload_path + ".wav"
        needs_cleanup.append(wav_path)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    upload_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    wav_path,
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            for p in needs_cleanup:
                if os.path.exists(p):
                    os.unlink(p)
            raise HTTPException(
                status_code=500, detail="ffmpeg not found — needed to convert non-WAV audio"
            )
        except subprocess.CalledProcessError as e:
            for p in needs_cleanup:
                if os.path.exists(p):
                    os.unlink(p)
            raise HTTPException(
                status_code=400, detail=f"ffmpeg conversion failed: {e.stderr.decode()[:500]}"
            )

    try:
        audio_waveform = faster_whisper.decode_audio(wav_path)
    except Exception as e:
        for p in needs_cleanup:
            if os.path.exists(p):
                os.unlink(p)
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")

    logger.info(f"Audio decoded: {audio_waveform.shape}, upload took {time.time() - t_start:.1f}s")

    async with whisper_semaphore:
        word_timestamps, detected_language = await loop.run_in_executor(
            None, _run_whisper_alignment, audio_waveform, language, batch_size, suppress_numerals
        )

    if speaker_name is not None and not skip_diarization:
        first_word_start = int(word_timestamps[0]["start"] * 1000)
        last_word_end = int(word_timestamps[-1]["end"] * 1000)
        speaker_ts = [[first_word_start, last_word_end, 0]]
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        result = _build_segments(
            wsm, speaker_ts, detected_language, include_srt, override_speaker=speaker_name
        )
        result.processing_time_seconds = round(time.time() - t_start, 2)

        async def _background_diarize():
            try:
                async with diarizer_semaphore:
                    diarization_result = await loop.run_in_executor(
                        None, _run_diarization, audio_waveform
                    )
                if (
                    isinstance(diarization_result, DiarizationResult)
                    and diarization_result.speaker_embeddings
                ):
                    with db_lock:
                        if models.shared_store is not None:
                            emb = list(diarization_result.speaker_embeddings.values())[0]
                            existing = models.shared_store._get_profile_by_name(speaker_name)
                            if existing is not None:
                                models.shared_store.update_embedding(speaker_name, emb)
                            else:
                                models.shared_store.add_speaker(speaker_name, emb)
            except Exception:
                logger.warning("Background DB update failed", exc_info=True)

        task = asyncio.create_task(_background_diarize())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        for p in needs_cleanup:
            if os.path.exists(p):
                os.unlink(p)

        elapsed = time.time() - t_start
        logger.info(f"Transcription (mode 3) completed in {elapsed:.1f}s")
        return result

    if skip_diarization or models.diarizer_model is None:
        first_word_start = int(word_timestamps[0]["start"] * 1000)
        last_word_end = int(word_timestamps[-1]["end"] * 1000)
        speaker_ts = [[first_word_start, last_word_end, 0]]
        diarization_result = None
    else:
        async with diarizer_semaphore:
            diarization_result = await loop.run_in_executor(None, _run_diarization, audio_waveform)
        if isinstance(diarization_result, DiarizationResult):
            speaker_ts = diarization_result.speaker_ts
        else:
            speaker_ts = diarization_result

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if (
        models.speaker_persistence
        and not skip_diarization
        and not no_persist
        and isinstance(diarization_result, DiarizationResult)
        and diarization_result.speaker_embeddings
    ):
        with db_lock:
            if models.shared_store is not None:
                pd = PersistentSpeakerDiarizer(
                    store=models.shared_store,
                    min_threshold=match_threshold,
                    interactive=False,
                )
                label_map = pd.resolve_speakers(diarization_result, word_speaker_mapping=wsm)
                speaker_ts = apply_persistent_labels(speaker_ts, label_map)
                wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    result = _build_segments(wsm, speaker_ts, detected_language, include_srt)
    result.processing_time_seconds = round(time.time() - t_start, 2)

    for p in needs_cleanup:
        if os.path.exists(p):
            os.unlink(p)

    elapsed = time.time() - t_start
    logger.info(f"Transcription completed in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
