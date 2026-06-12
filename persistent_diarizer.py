import sys
from typing import Dict, List, Optional, Set, Tuple

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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def merge_new_speakers(
    new_speakers: Dict[int, np.ndarray], merge_threshold: float = 0.85
) -> Dict[int, int]:
    spk_ids = sorted(new_speakers.keys())
    merge_map: Dict[int, int] = {spk_id: spk_id for spk_id in spk_ids}

    for i, id_a in enumerate(spk_ids):
        for id_b in spk_ids[i + 1 :]:
            if merge_map[id_b] != id_b:
                continue
            sim = _cosine_similarity(new_speakers[id_a], new_speakers[id_b])
            if sim >= merge_threshold:
                merge_map[id_b] = id_a

    return merge_map


class PersistentSpeakerDiarizer:
    def __init__(
        self,
        store: SpeakerEmbeddingStore,
        min_threshold: float = 0.75,
        interactive: bool = False,
        merge_threshold: float = 0.85,
    ):
        self._store = store
        self._min_threshold = min_threshold
        self._interactive = interactive
        self._merge_threshold = merge_threshold

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
            merge_map = merge_new_speakers(new_speakers, self._merge_threshold)
            merged_label_map = self._assign_labels(
                new_speakers, merge_map, result, word_speaker_mapping
            )
            label_map.update(merged_label_map)

        return label_map

    def _assign_labels(
        self,
        new_speakers: Dict[int, np.ndarray],
        merge_map: Dict[int, int],
        result: DiarizationResult,
        word_speaker_mapping: Optional[List[dict]],
    ) -> Dict[int, str]:
        existing_names = {s["name"] for s in self._store.list_speakers()}
        assigned: Dict[int, str] = {}
        next_num = 0

        for spk_id in sorted(new_speakers.keys()):
            canonical = merge_map[spk_id]
            if canonical in assigned:
                assigned[spk_id] = assigned[canonical]
                continue

            if self._interactive:
                chosen = self._interactive_prompt(
                    spk_id, existing_names, next_num, result, word_speaker_mapping
                )
            else:
                chosen = self._next_available_name(existing_names, next_num)

            assigned[canonical] = chosen
            assigned[spk_id] = chosen
            existing_names.add(chosen)
            next_num += 1

            if chosen in {s["name"] for s in self._store.list_speakers()}:
                self._store.update_embedding(chosen, new_speakers[spk_id])
            else:
                self._store.add_speaker(chosen, new_speakers[spk_id])

            for other_id in new_speakers:
                if other_id != spk_id and merge_map.get(other_id) == canonical:
                    self._store.update_embedding(chosen, new_speakers[other_id])

        return assigned

    def _interactive_prompt(
        self,
        spk_id: int,
        existing_names: Set[str],
        next_num: int,
        result: DiarizationResult,
        word_speaker_mapping: Optional[List[dict]],
    ) -> str:
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

        stored_names = {s["name"] for s in self._store.list_speakers()}

        while True:
            prompt_parts = [f'New speaker detected ({timestamp}):']
            if sample:
                prompt_parts.append(f'  "{sample}"')
            prompt_parts.append(f"Enter name [default: {default_name}]: ")
            prompt = "\n".join(prompt_parts)

            if not sys.stdin.isatty():
                return default_name

            response = input(prompt).strip()
            chosen = response if response else default_name

            if chosen in stored_names:
                merge_prompt = (
                    f"  '{chosen}' already exists. "
                    f"Merge with existing speaker? [y/n]: "
                )
                merge_response = input(merge_prompt).strip().lower()
                if merge_response.startswith("y"):
                    return chosen
            else:
                return chosen

    def _next_available_name(self, existing_names: set, start: int) -> str:
        n = start
        while f"Speaker {n}" in existing_names:
            n += 1
        return f"Speaker {n}"
