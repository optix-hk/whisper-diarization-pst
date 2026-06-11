import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from diarization.msdd.msdd import DiarizationResult
from speaker_matcher import match_speakers
from speaker_store import SpeakerEmbeddingStore


def apply_persistent_labels(
    speaker_ts: List[Tuple[int, int, int]], label_map: Dict[int, str]
) -> List[Tuple[int, int, str]]:
    return [
        (start, end, label_map.get(spk_id, spk_id))
        for start, end, spk_id in speaker_ts
    ]


def _get_sample_sentence(
    word_speaker_mapping: List[dict], speaker_id: int
) -> str:
    words = []
    for entry in word_speaker_mapping:
        if entry["speaker"] == speaker_id:
            words.append(entry["word"])
            if entry["word"] and entry["word"][-1] in ".?!":
                break
        elif words:
            break
    return " ".join(words) if words else ""


def _format_ms(ms: int) -> str:
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}s"


class PersistentSpeakerDiarizer:
    def __init__(
        self,
        store: SpeakerEmbeddingStore,
        min_threshold: float = 0.6,
        interactive: bool = False,
    ):
        self._store = store
        self._min_threshold = min_threshold
        self._interactive = interactive

    def resolve_speakers(
        self,
        result: DiarizationResult,
        word_speaker_mapping: Optional[List[dict]] = None,
    ) -> Dict[int, str]:
        matches = match_speakers(
            result.speaker_embeddings, self._store, self._min_threshold
        )

        label_map: Dict[int, str] = {}
        new_speakers: Dict[int, np.ndarray] = {}

        for spk_id, (name, score) in matches.items():
            if name is not None:
                label_map[spk_id] = name
                self._store.update_embedding(name, result.speaker_embeddings[spk_id])
            else:
                new_speakers[spk_id] = result.speaker_embeddings[spk_id]

        if new_speakers:
            existing_names = {s["name"] for s in self._store.list_speakers()}
            next_num = 0
            for spk_id in sorted(new_speakers.keys()):
                if self._interactive:
                    sample = ""
                    if word_speaker_mapping is not None:
                        sample = _get_sample_sentence(word_speaker_mapping, spk_id)
                    spk_segments = [
                        (s, e) for s, e, sp in result.speaker_ts if sp == spk_id
                    ]
                    if spk_segments:
                        first_seg = spk_segments[0]
                        timestamp = f"{_format_ms(first_seg[0])} - {_format_ms(first_seg[1])}"
                    else:
                        timestamp = "unknown"
                    default_name = self._next_available_name(existing_names, next_num)
                    prompt_parts = [f'New speaker detected ({timestamp}):']
                    if sample:
                        prompt_parts.append(f'  "{sample}"')
                    prompt_parts.append(f"Enter name [default: {default_name}]: ")
                    prompt = "\n".join(prompt_parts)
                    if not sys.stdin.isatty():
                        chosen = default_name
                    else:
                        response = input(prompt).strip()
                        chosen = response if response else default_name
                else:
                    chosen = self._next_available_name(existing_names, next_num)

                label_map[spk_id] = chosen
                existing_names.add(chosen)
                next_num += 1
                self._store.add_speaker(chosen, new_speakers[spk_id])

        return label_map

    def _next_available_name(self, existing_names: set, start: int) -> str:
        n = start
        while f"Speaker {n}" in existing_names:
            n += 1
        return f"Speaker {n}"
