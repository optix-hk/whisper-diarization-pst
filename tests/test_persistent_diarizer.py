import numpy as np
import pytest

from diarization.msdd.msdd import DiarizationResult
from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels
from speaker_store import SpeakerEmbeddingStore


def _make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(192).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def store():
    s = SpeakerEmbeddingStore(":memory:")
    yield s
    s.close()


def test_resolve_speakers_auto_label(store):
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0), (1000, 2000, 1)],
        speaker_embeddings={0: _make_embedding(0), 1: _make_embedding(1)},
    )
    label_map = pd.resolve_speakers(result)
    assert label_map[0] == "Speaker 0"
    assert label_map[1] == "Speaker 1"
    speakers = store.list_speakers()
    assert len(speakers) == 2


def test_resolve_speakers_matches_known(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query = emb_a + _make_embedding(100) * 0.05
    query /= np.linalg.norm(query)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0)],
        speaker_embeddings={0: query},
    )
    label_map = pd.resolve_speakers(result)
    assert label_map[0] == "Alice"


def test_resolve_speakers_updates_known_embedding(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query = emb_a + _make_embedding(100) * 0.05
    query /= np.linalg.norm(query)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0)],
        speaker_embeddings={0: query},
    )
    pd.resolve_speakers(result)
    profiles = store.get_all_profiles()
    assert profiles[0].sample_count == 2


def test_resolve_speakers_mixed_known_and_new(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query_a = emb_a + _make_embedding(100) * 0.05
    query_a /= np.linalg.norm(query_a)
    query_b = _make_embedding(1)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0), (1000, 2000, 1)],
        speaker_embeddings={0: query_a, 1: query_b},
    )
    label_map = pd.resolve_speakers(result)
    assert label_map[0] == "Alice"
    assert label_map[1] == "Speaker 0"


def test_apply_persistent_labels():
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1)]
    label_map = {0: "Alice", 1: "Bob"}
    result = apply_persistent_labels(speaker_ts, label_map)
    assert result == [(0, 1000, "Alice"), (1000, 2000, "Bob")]


def test_apply_persistent_labels_partial_map():
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1), (2000, 3000, 2)]
    label_map = {0: "Alice", 2: "Carol"}
    result = apply_persistent_labels(speaker_ts, label_map)
    assert result[0] == (0, 1000, "Alice")
    assert result[1] == (1000, 2000, 1)
    assert result[2] == (2000, 3000, "Carol")
