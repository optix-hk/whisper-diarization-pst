import numpy as np
import pytest

from speaker_matcher import match_speakers
from speaker_store import SpeakerEmbeddingStore


@pytest.fixture
def store():
    s = SpeakerEmbeddingStore(":memory:")
    yield s
    s.close()


def _make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(192).astype(np.float32)
    return emb / np.linalg.norm(emb)


def test_match_speakers_all_known(store):
    emb_a = _make_embedding(0)
    emb_b = _make_embedding(1)
    store.add_speaker("Alice", emb_a)
    store.add_speaker("Bob", emb_b)
    query_a = emb_a + _make_embedding(100) * 0.05
    query_a /= np.linalg.norm(query_a)
    query_b = emb_b + _make_embedding(101) * 0.05
    query_b /= np.linalg.norm(query_b)
    speaker_embeddings = {0: query_a, 1: query_b}
    result = match_speakers(speaker_embeddings, store, min_threshold=0.6)
    assert result[0][0] == "Alice"
    assert result[0][1] > 0.6
    assert result[1][0] == "Bob"
    assert result[1][1] > 0.6


def test_match_speakers_no_match(store):
    store.add_speaker("Alice", _make_embedding(0))
    unrelated = _make_embedding(99)
    result = match_speakers({0: unrelated}, store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] < 0.6


def test_match_speakers_empty_store(store):
    emb = _make_embedding(0)
    result = match_speakers({0: emb}, store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] == 0.0


def test_match_speakers_collision_resolved_by_score(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query_close = emb_a + _make_embedding(100) * 0.05
    query_close /= np.linalg.norm(query_close)
    query_far = emb_a + _make_embedding(200) * 0.3
    query_far /= np.linalg.norm(query_far)
    speaker_embeddings = {0: query_close, 1: query_far}
    result = match_speakers(speaker_embeddings, store, min_threshold=0.6)
    assert result[0][0] == "Alice"
    assert result[1][0] is None


def test_match_speakers_multiple_stored(store):
    embs = {f"Speaker_{i}": _make_embedding(i) for i in range(5)}
    for name, emb in embs.items():
        store.add_speaker(name, emb)
    query = embs["Speaker_3"] + _make_embedding(300) * 0.05
    query /= np.linalg.norm(query)
    result = match_speakers({0: query}, store, min_threshold=0.6)
    assert result[0][0] == "Speaker_3"
