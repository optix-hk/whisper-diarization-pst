from typing import Dict, Optional, Tuple

import numpy as np

from speaker_store import SpeakerEmbeddingStore


def match_speakers(
    speaker_embeddings: Dict[int, np.ndarray],
    store: SpeakerEmbeddingStore,
    min_threshold: float = 0.6,
) -> Dict[int, Tuple[Optional[str], float]]:
    profiles = store.get_all_profiles()
    if not profiles:
        return {spk_id: (None, 0.0) for spk_id in speaker_embeddings}

    stored_vectors = []
    stored_names = []
    for p in profiles:
        norm = p.embedding / (np.linalg.norm(p.embedding) + 1e-8)
        stored_vectors.append(norm)
        stored_names.append(p.name)
    stored_matrix = np.stack(stored_vectors)

    raw_matches: Dict[int, Tuple[Optional[str], float]] = {}
    for spk_id, emb in speaker_embeddings.items():
        query_norm = emb / (np.linalg.norm(emb) + 1e-8)
        scores = stored_matrix @ query_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= min_threshold:
            raw_matches[spk_id] = (stored_names[best_idx], best_score)
        else:
            raw_matches[spk_id] = (None, best_score)

    claimed: Dict[str, int] = {}
    for spk_id, (name, score) in raw_matches.items():
        if name is not None:
            if name not in claimed:
                claimed[name] = spk_id
            else:
                prev_id = claimed[name]
                prev_score = raw_matches[prev_id][1]
                if score > prev_score:
                    raw_matches[prev_id] = (None, raw_matches[prev_id][1])
                    claimed[name] = spk_id
                else:
                    raw_matches[spk_id] = (None, score)

    return raw_matches
