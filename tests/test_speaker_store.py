import threading

import numpy as np
import pytest

from speaker_store import SpeakerEmbeddingStore


@pytest.fixture
def store():
    s = SpeakerEmbeddingStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def sample_embedding():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(192).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def test_add_speaker(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    profiles = store.get_all_profiles()
    assert len(profiles) == 1
    assert profiles[0].name == "Alice"
    assert np.allclose(profiles[0].embedding, sample_embedding)
    assert profiles[0].sample_count == 1


def test_add_speaker_duplicate_name_raises(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    with pytest.raises(ValueError, match="already exists"):
        store.add_speaker("Alice", sample_embedding)


def test_get_all_profiles_empty(store):
    assert store.get_all_profiles() == []


def test_find_best_match_hit(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    query = (
        sample_embedding + np.random.default_rng(0).standard_normal(192).astype(np.float32) * 0.1
    )
    query /= np.linalg.norm(query)
    name, score = store.find_best_match(query, min_threshold=0.5)
    assert name == "Alice"
    assert score > 0.5


def test_find_best_match_miss(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    rng = np.random.default_rng(99)
    unrelated = rng.standard_normal(192).astype(np.float32)
    unrelated /= np.linalg.norm(unrelated)
    name, score = store.find_best_match(unrelated, min_threshold=0.6)
    assert name is None
    assert score < 0.6


def test_find_best_match_empty_store(store, sample_embedding):
    name, score = store.find_best_match(sample_embedding, min_threshold=0.6)
    assert name is None
    assert score == 0.0


def test_find_best_match_multiple_profiles(store, sample_embedding):
    rng = np.random.default_rng(42)
    emb_a = rng.standard_normal(192).astype(np.float32)
    emb_a /= np.linalg.norm(emb_a)
    emb_b = rng.standard_normal(192).astype(np.float32)
    emb_b /= np.linalg.norm(emb_b)
    store.add_speaker("Alice", emb_a)
    store.add_speaker("Bob", emb_b)
    query = emb_a + rng.standard_normal(192).astype(np.float32) * 0.05
    query /= np.linalg.norm(query)
    name, score = store.find_best_match(query, min_threshold=0.6)
    assert name == "Alice"


def test_update_embedding_running_average(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    new_emb = (
        sample_embedding + np.random.default_rng(1).standard_normal(192).astype(np.float32) * 0.1
    )
    new_emb /= np.linalg.norm(new_emb)
    store.update_embedding("Alice", new_emb)
    profiles = store.get_all_profiles()
    expected = (sample_embedding * 1 + new_emb * 1) / 2
    assert np.allclose(profiles[0].embedding, expected)
    assert profiles[0].sample_count == 2


def test_update_embedding_nonexistent_raises(store, sample_embedding):
    with pytest.raises(ValueError, match="not found"):
        store.update_embedding("Alice", sample_embedding)


def test_rename_speaker(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    store.rename_speaker("Alice", "Alicia")
    profiles = store.get_all_profiles()
    assert profiles[0].name == "Alicia"


def test_rename_speaker_nonexistent_raises(store, sample_embedding):
    with pytest.raises(ValueError, match="not found"):
        store.rename_speaker("Alice", "Alicia")


def test_rename_speaker_to_existing_name_raises(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    rng = np.random.default_rng(7)
    bob_emb = rng.standard_normal(192).astype(np.float32)
    bob_emb /= np.linalg.norm(bob_emb)
    store.add_speaker("Bob", bob_emb)
    with pytest.raises(ValueError, match="already exists"):
        store.rename_speaker("Alice", "Bob")


def test_delete_speaker(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    store.delete_speaker("Alice")
    assert store.get_all_profiles() == []


def test_delete_speaker_nonexistent_raises(store, sample_embedding):
    with pytest.raises(ValueError, match="not found"):
        store.delete_speaker("Alice")


def test_list_speakers(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    speakers = store.list_speakers()
    assert len(speakers) == 1
    assert speakers[0]["name"] == "Alice"
    assert "created_at" in speakers[0]
    assert "updated_at" in speakers[0]


def test_list_speakers_multiple(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    rng = np.random.default_rng(7)
    bob_emb = rng.standard_normal(192).astype(np.float32)
    bob_emb /= np.linalg.norm(bob_emb)
    store.add_speaker("Bob", bob_emb)
    speakers = store.list_speakers()
    assert len(speakers) == 2
    names = {s["name"] for s in speakers}
    assert names == {"Alice", "Bob"}


def test_update_embedding_multiple_times(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    rng = np.random.default_rng(5)
    for i in range(3):
        emb = rng.standard_normal(192).astype(np.float32)
        emb /= np.linalg.norm(emb)
        store.update_embedding("Alice", emb)
    profiles = store.get_all_profiles()
    assert profiles[0].sample_count == 4


def test_find_best_match_after_update(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    rng = np.random.default_rng(3)
    new_emb = rng.standard_normal(192).astype(np.float32)
    new_emb /= np.linalg.norm(new_emb)
    store.update_embedding("Alice", new_emb)
    profiles = store.get_all_profiles()
    query = profiles[0].embedding + rng.standard_normal(192).astype(np.float32) * 0.05
    query /= np.linalg.norm(query)
    name, score = store.find_best_match(query, min_threshold=0.5)
    assert name == "Alice"
    assert score > 0.5


def test_context_manager(sample_embedding):
    with SpeakerEmbeddingStore(":memory:") as store:
        store.add_speaker("Alice", sample_embedding)
        profiles = store.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "Alice"


def test_add_speaker_wrong_dimension_raises(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    wrong_dim = np.random.default_rng(0).standard_normal(128).astype(np.float32)
    with pytest.raises(ValueError, match="dimension must be 192"):
        store.add_speaker("Bob", wrong_dim)


def test_merge_speakers(store, sample_embedding):
    rng = np.random.default_rng(7)
    bob_emb = rng.standard_normal(192).astype(np.float32)
    bob_emb /= np.linalg.norm(bob_emb)
    store.add_speaker("Alice", sample_embedding)
    store.add_speaker("Bob", bob_emb)
    store.merge_speakers("Bob", "Alice")
    speakers = store.list_speakers()
    assert len(speakers) == 1
    assert speakers[0]["name"] == "Alice"
    assert speakers[0]["sample_count"] == 2


def test_merge_speakers_source_not_found(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    with pytest.raises(ValueError, match="not found"):
        store.merge_speakers("Bob", "Alice")


def test_merge_speakers_target_not_found(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    with pytest.raises(ValueError, match="not found"):
        store.merge_speakers("Alice", "Bob")


def test_rename_speaker_existing_name_suggests_merge(store, sample_embedding):
    rng = np.random.default_rng(7)
    bob_emb = rng.standard_normal(192).astype(np.float32)
    bob_emb /= np.linalg.norm(bob_emb)
    store.add_speaker("Alice", sample_embedding)
    store.add_speaker("Bob", bob_emb)
    with pytest.raises(ValueError, match="Use merge"):
        store.rename_speaker("Bob", "Alice")


def test_update_embedding_concurrent_no_lost_updates(store, sample_embedding):
    store.add_speaker("Alice", sample_embedding)
    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def update():
        barrier.wait()
        store.update_embedding("Alice", sample_embedding)

    threads = [threading.Thread(target=update) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    profiles = store.get_all_profiles()
    assert profiles[0].sample_count == 1 + num_threads


def test_add_speaker_concurrent_no_duplicate_error(store, sample_embedding):
    num_threads = 10
    barrier = threading.Barrier(num_threads)
    errors = []

    def add(i):
        barrier.wait()
        try:
            store.add_speaker(f"Speaker_{i}", sample_embedding)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(store.list_speakers()) == num_threads
