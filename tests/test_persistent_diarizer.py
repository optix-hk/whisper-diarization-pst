from unittest.mock import patch

import numpy as np
import pytest

from persistent_diarizer import (
    PersistentSpeakerDiarizer,
    apply_persistent_labels,
    merge_new_speakers,
)
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
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1)]
    segment_embeddings = [_make_embedding(0), _make_embedding(1)]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Speaker 0"
    assert label_map[1] == "Speaker 1"
    speakers = store.list_speakers()
    assert len(speakers) == 2


def test_resolve_speakers_matches_known(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query = emb_a + _make_embedding(100) * 0.05
    query /= np.linalg.norm(query)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    speaker_ts = [(0, 1000, 0)]
    segment_embeddings = [query]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Alice"


def test_resolve_speakers_updates_known_embedding(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query = emb_a + _make_embedding(100) * 0.05
    query /= np.linalg.norm(query)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    speaker_ts = [(0, 1000, 0)]
    segment_embeddings = [query]
    pd.resolve_speakers(speaker_ts, segment_embeddings)
    profiles = store.get_all_profiles()
    assert profiles[0].sample_count == 2


def test_resolve_speakers_mixed_known_and_new(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query_a = emb_a + _make_embedding(100) * 0.05
    query_a /= np.linalg.norm(query_a)
    query_b = _make_embedding(1)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1)]
    segment_embeddings = [query_a, query_b]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Alice"
    assert label_map[1] == "Speaker 0"


def test_resolve_speakers_per_segment_correction(store):
    """Segment in MSDD cluster 1 matches stored 'Alice' — corrects MSDD cluster assignment."""
    emb_alice = _make_embedding(0)
    emb_bob = _make_embedding(1)
    store.add_speaker("Alice", emb_alice)
    store.add_speaker("Bob", emb_bob)
    query_alice = emb_alice + _make_embedding(100) * 0.05
    query_alice /= np.linalg.norm(query_alice)
    query_bob = emb_bob + _make_embedding(101) * 0.05
    query_bob /= np.linalg.norm(query_bob)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    # MSDD put both segments in cluster 0, but seg 1 is actually Bob
    speaker_ts = [(0, 1000, 0), (1000, 2000, 0)]
    segment_embeddings = [query_alice, query_bob]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Alice"
    assert label_map[1] == "Bob"


def test_resolve_speakers_cluster_fallback_for_unmatched(store):
    """Unmatched segment inherits majority label from same-MSDD-cluster segments."""
    emb_alice = _make_embedding(0)
    store.add_speaker("Alice", emb_alice)
    query_alice = emb_alice + _make_embedding(100) * 0.05
    query_alice /= np.linalg.norm(query_alice)
    unrelated = _make_embedding(99)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    # Two segments in cluster 0: seg0 matches Alice, seg1 doesn't match
    speaker_ts = [(0, 1000, 0), (1000, 2000, 0)]
    segment_embeddings = [query_alice, unrelated]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Alice"
    assert label_map[1] == "Alice"  # inherited from cluster majority


def test_resolve_speakers_entire_cluster_unmatched_grouped_as_new(store):
    """All segments in a cluster unmatched → grouped as one new speaker."""
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    speaker_ts = [(0, 1000, 0), (1000, 2000, 0), (2000, 3000, 1)]
    segment_embeddings = [_make_embedding(0), _make_embedding(0), _make_embedding(1)]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    # Cluster 0 → "Speaker 0", cluster 1 → "Speaker 1"
    assert label_map[0] == "Speaker 0"
    assert label_map[1] == "Speaker 0"
    assert label_map[2] == "Speaker 1"
    speakers = store.list_speakers()
    assert len(speakers) == 2


def test_resolve_speakers_none_embedding_falls_back_to_cluster(store):
    """None embedding (extraction failure) is treated as unmatched → cluster fallback."""
    emb_alice = _make_embedding(0)
    store.add_speaker("Alice", emb_alice)
    query_alice = emb_alice + _make_embedding(100) * 0.05
    query_alice /= np.linalg.norm(query_alice)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=False)
    speaker_ts = [(0, 1000, 0), (1000, 2000, 0)]
    segment_embeddings = [query_alice, None]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Alice"
    assert label_map[1] == "Alice"  # inherited via cluster fallback


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
    assert result[1] == (1000, 2000, "1")
    assert result[2] == (2000, 3000, "Carol")


def test_merge_new_speakers_no_merge():
    embs = {0: _make_embedding(0), 1: _make_embedding(1)}
    merge_map = merge_new_speakers(embs, merge_threshold=0.85)
    assert merge_map == {0: 0, 1: 1}


def test_merge_new_speakers_similar_merged():
    base = _make_embedding(0)
    similar = base + _make_embedding(100) * 0.05
    similar /= np.linalg.norm(similar)
    embs = {0: base, 1: similar}
    merge_map = merge_new_speakers(embs, merge_threshold=0.85)
    assert merge_map[0] == 0
    assert merge_map[1] == 0


def test_merge_new_speakers_three_with_one_similar():
    base = _make_embedding(0)
    similar = base + _make_embedding(100) * 0.05
    similar /= np.linalg.norm(similar)
    different = _make_embedding(1)
    embs = {0: base, 1: similar, 2: different}
    merge_map = merge_new_speakers(embs, merge_threshold=0.85)
    assert merge_map[0] == 0
    assert merge_map[1] == 0
    assert merge_map[2] == 2


def test_resolve_speakers_merged_new_speakers_get_same_label(store):
    base = _make_embedding(0)
    similar = base + _make_embedding(100) * 0.05
    similar /= np.linalg.norm(similar)
    pd = PersistentSpeakerDiarizer(
        store, min_threshold=0.75, interactive=False, merge_threshold=0.85
    )
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1)]
    segment_embeddings = [base, similar]
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == label_map[1]
    speakers = store.list_speakers()
    assert len(speakers) == 1


def test_resolve_speakers_merge_updates_existing_on_duplicate_name(store):
    emb_alice = _make_embedding(0)
    store.add_speaker("Alice", emb_alice)
    new_emb = _make_embedding(1)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.75, interactive=True)
    speaker_ts = [(0, 1000, 0)]
    segment_embeddings = [new_emb]
    with (
        patch("persistent_diarizer.input", side_effect=["Alice", "y"]),
        patch("persistent_diarizer.sys.stdin") as mock_stdin,
    ):
        mock_stdin.isatty.return_value = True
        label_map = pd.resolve_speakers(speaker_ts, segment_embeddings)
    assert label_map[0] == "Alice"
    profile = store.get_all_profiles()[0]
    assert profile.name == "Alice"
    assert profile.sample_count == 2
