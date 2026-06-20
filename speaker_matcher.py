from typing import List, Optional

import numpy as np

from speaker_store import SpeakerEmbeddingStore


def match_speakers(
    segment_embeddings: List[Optional[np.ndarray]],
    store: SpeakerEmbeddingStore,
    min_threshold: float = 0.75,
) -> dict[int, tuple[str | None, float]]:
    profiles = store.get_all_profiles()
    if not profiles:
        return {i: (None, 0.0) for i in range(len(segment_embeddings))}

    stored_vectors = []
    stored_names = []
    for p in profiles:
        norm = p.embedding / (np.linalg.norm(p.embedding) + 1e-8)
        stored_vectors.append(norm)
        stored_names.append(p.name)
    stored_matrix = np.stack(stored_vectors)

    matches: dict[int, tuple[str | None, float]] = {}
    for seg_idx, emb in enumerate(segment_embeddings):
        if emb is None:
            matches[seg_idx] = (None, 0.0)
            continue
        query_norm = emb / (np.linalg.norm(emb) + 1e-8)
        scores = stored_matrix @ query_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= min_threshold:
            matches[seg_idx] = (stored_names[best_idx], best_score)
        else:
            matches[seg_idx] = (None, best_score)

    return matches
