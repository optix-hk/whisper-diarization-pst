import sys
from collections import Counter
from typing import Dict, List, Optional, Set

import numpy as np

from speaker_matcher import match_speakers
from speaker_store import SpeakerEmbeddingStore


def apply_persistent_labels(
    speaker_ts: list[tuple[int, int, int]],
    label_map: dict[int, str],
) -> list[tuple[int, int, str]]:
    return [
        (start, end, label_map.get(i, str(spk_id)))
        for i, (start, end, spk_id) in enumerate(speaker_ts)
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
        speaker_ts: list[tuple[int, int, int]],
        segment_embeddings: list[np.ndarray | None],
        word_speaker_mapping: list[dict] | None = None,
    ) -> dict[int, str]:
        matches = match_speakers(
            segment_embeddings, self._store, self._min_threshold
        )

        label_map: dict[int, str] = {}
        unmatched_indices: list[int] = []
        for seg_idx, (name, score) in matches.items():
            if name is not None:
                label_map[seg_idx] = name
            else:
                unmatched_indices.append(seg_idx)

        still_unmatched: list[int] = []
        for seg_idx in unmatched_indices:
            spk_id = speaker_ts[seg_idx][2]
            cluster_labels = [
                label_map[i]
                for i, (_, _, sid) in enumerate(speaker_ts)
                if sid == spk_id and i in label_map
            ]
            if cluster_labels:
                label_map[seg_idx] = Counter(cluster_labels).most_common(1)[0][0]
            else:
                still_unmatched.append(seg_idx)

        new_groups: dict[int, list[int]] = {}
        for seg_idx in still_unmatched:
            spk_id = speaker_ts[seg_idx][2]
            new_groups.setdefault(spk_id, []).append(seg_idx)

        new_speaker_embeddings: dict[int, np.ndarray] = {}
        for spk_id, seg_indices in new_groups.items():
            embs = [
                segment_embeddings[i]
                for i in seg_indices
                if segment_embeddings[i] is not None
            ]
            if embs:
                new_speaker_embeddings[spk_id] = np.mean(embs, axis=0)
            else:
                new_speaker_embeddings[spk_id] = np.zeros(192, dtype=np.float32)

        merge_map = merge_new_speakers(new_speaker_embeddings, self._merge_threshold)

        existing_names = {s["name"] for s in self._store.list_speakers()}
        assigned: dict[int, str] = {}
        next_num = 0

        for spk_id in sorted(new_groups.keys()):
            canonical = merge_map[spk_id]
            if canonical in assigned:
                label = assigned[canonical]
                for seg_idx in new_groups[spk_id]:
                    label_map[seg_idx] = label
                continue

            if self._interactive:
                chosen = self._interactive_prompt(
                    spk_id, existing_names, next_num, speaker_ts, word_speaker_mapping
                )
            else:
                chosen = self._next_available_name(existing_names, next_num)

            assigned[canonical] = chosen
            assigned[spk_id] = chosen
            existing_names.add(chosen)
            next_num += 1

            for seg_idx in new_groups[spk_id]:
                label_map[seg_idx] = chosen

            if chosen in {s["name"] for s in self._store.list_speakers()}:
                self._store.update_embedding(chosen, new_speaker_embeddings[spk_id])
            else:
                self._store.add_speaker(chosen, new_speaker_embeddings[spk_id])

            for other_id in new_groups:
                if other_id != spk_id and merge_map.get(other_id) == canonical:
                    self._store.update_embedding(chosen, new_speaker_embeddings[other_id])

        matched_by_name: dict[str, list[np.ndarray]] = {}
        for seg_idx, (name, _) in matches.items():
            if name is not None and segment_embeddings[seg_idx] is not None:
                matched_by_name.setdefault(name, []).append(segment_embeddings[seg_idx])

        for name, embs in matched_by_name.items():
            mean_emb = np.mean(embs, axis=0)
            self._store.update_embedding(name, mean_emb)

        return label_map

    def _interactive_prompt(
        self,
        spk_id: int,
        existing_names: Set[str],
        next_num: int,
        speaker_ts: list[tuple[int, int, int]],
        word_speaker_mapping: Optional[List[dict]],
    ) -> str:
        sample = ""
        if word_speaker_mapping is not None:
            sample = _get_sample_sentence(word_speaker_mapping, spk_id)
        spk_segments = [
            (s, e) for s, e, sp in speaker_ts if sp == spk_id
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
