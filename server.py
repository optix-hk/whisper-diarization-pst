import io
import logging
import os
import re
import subprocess
import tempfile
import time

import faster_whisper
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

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


class Models:
    def __init__(self):
        self.whisper_model = None
        self.whisper_pipeline = None
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None
        self.diarizer_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


models = Models()


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
    if os.environ.get("SKIP_DIARIZATION", "").lower() in ("1", "true", "yes"):
        logger.info("Diarization disabled (SKIP_DIARIZATION=1)")
    else:
        t3 = time.time()
        logger.info(f"Loading diarizer ({diarizer_type})...")
        if diarizer_type == "msdd":
            models.diarizer_model = MSDDDiarizer(device=device)
        elif diarizer_type == "sortformer":
            models.diarizer_model = SortformerDiarizer(device=device)
        logger.info(f"Diarizer loaded in {time.time() - t3:.1f}s")

    logger.info(f"All models loaded in {time.time() - t0:.1f}s")


@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": models.device,
        "whisper_loaded": models.whisper_model is not None,
        "alignment_loaded": models.alignment_model is not None,
        "punct_loaded": models.punct_model is not None,
        "diarizer_loaded": models.diarizer_model is not None,
    }


@app.post("/transcribe", response_model=TranscriptionResult)
def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    batch_size: int = Form(8),
    suppress_numerals: bool = Form(False),
    skip_diarization: bool = Form(False),
    include_srt: bool = Form(True),
    no_persist: bool = Form(True),
    speaker_db: str = Form("~/.whisper-diarization/speakers.db"),
    match_threshold: float = Form(0.75),
):
    t_start = time.time()

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
                ["ffmpeg", "-y", "-i", upload_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            for p in needs_cleanup:
                if os.path.exists(p):
                    os.unlink(p)
            raise HTTPException(status_code=500, detail="ffmpeg not found — needed to convert non-WAV audio")
        except subprocess.CalledProcessError as e:
            for p in needs_cleanup:
                if os.path.exists(p):
                    os.unlink(p)
            raise HTTPException(status_code=400, detail=f"ffmpeg conversion failed: {e.stderr.decode()[:500]}")

    try:
        audio_waveform = faster_whisper.decode_audio(wav_path)
    except Exception as e:
        for p in needs_cleanup:
            if os.path.exists(p):
                os.unlink(p)
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")

    logger.info(f"Audio decoded: {audio_waveform.shape}, upload took {time.time() - t_start:.1f}s")

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
        torch.from_numpy(audio_waveform).to(models.alignment_model.dtype).to(models.alignment_model.device),
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

    if skip_diarization or models.diarizer_model is None:
        first_word_start = int(word_timestamps[0]["start"] * 1000)
        last_word_end = int(word_timestamps[-1]["end"] * 1000)
        speaker_ts = [[first_word_start, last_word_end, 0]]
        diarization_result = None
    else:
        diarization_result = models.diarizer_model.diarize(
            torch.from_numpy(audio_waveform).unsqueeze(0)
        )
        if isinstance(diarization_result, DiarizationResult):
            speaker_ts = diarization_result.speaker_ts
        else:
            speaker_ts = diarization_result

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if not skip_diarization and not no_persist and isinstance(diarization_result, DiarizationResult) and diarization_result.speaker_embeddings:
        store = SpeakerEmbeddingStore(speaker_db)
        pd = PersistentSpeakerDiarizer(
            store=store,
            min_threshold=match_threshold,
            interactive=False,
        )
        label_map = pd.resolve_speakers(diarization_result, word_speaker_mapping=wsm)
        speaker_ts = apply_persistent_labels(speaker_ts, label_map)
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        store.close()

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
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

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

    for p in needs_cleanup:
        if os.path.exists(p):
            os.unlink(p)

    elapsed = time.time() - t_start
    logger.info(f"Transcription completed in {elapsed:.1f}s")

    return TranscriptionResult(
        segments=result_segments,
        srt=srt_text,
        language=detected_language,
        processing_time_seconds=round(elapsed, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
