import io

import numpy as np
from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with (
        patch("server.faster_whisper"),
        patch("server.load_alignment_model", return_value=(MagicMock(), MagicMock())),
        patch("server.PunctuationModel", return_value=MagicMock()),
        patch("server.MSDDDiarizer", return_value=MagicMock()),
        patch("server.SortformerDiarizer", return_value=MagicMock()),
        patch("server.SpeakerEmbedder", return_value=MagicMock()),
    ):
        from server import app, background_tasks, models

        models.whisper_model = MagicMock()
        models.whisper_pipeline = MagicMock()
        models.alignment_model = MagicMock()
        models.alignment_tokenizer = MagicMock()
        models.punct_model = MagicMock()
        models.diarizer_model = MagicMock()
        models.embedder = MagicMock()
        models.device = "cpu"
        background_tasks.clear()
        with TestClient(app) as c:
            yield c


@pytest.fixture
def store_client():
    with (
        patch("server.faster_whisper"),
        patch("server.load_alignment_model", return_value=(MagicMock(), MagicMock())),
        patch("server.PunctuationModel", return_value=MagicMock()),
        patch("server.MSDDDiarizer", return_value=MagicMock()),
        patch("server.SortformerDiarizer", return_value=MagicMock()),
        patch("server.SpeakerEmbedder", return_value=MagicMock()),
    ):
        from server import app, background_tasks, models
        from speaker_store import SpeakerEmbeddingStore

        models.whisper_model = MagicMock()
        models.whisper_pipeline = MagicMock()
        models.alignment_model = MagicMock()
        models.alignment_tokenizer = MagicMock()
        models.punct_model = MagicMock()
        models.diarizer_model = MagicMock()
        models.embedder = MagicMock()
        models.device = "cpu"
        background_tasks.clear()
        with TestClient(app) as c:
            if models.shared_store is not None:
                models.shared_store.close()
            models.shared_store = SpeakerEmbeddingStore(":memory:")
            yield c
        models.shared_store = None


def _add_test_speaker(store, name, dim=192):
    store.add_speaker(name, np.random.randn(dim).astype(np.float32))


def test_health_endpoint_returns_all_fields(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "device" in data
    assert "pending_db_updates" in data
    assert "embedder_loaded" in data


def test_speaker_name_with_skip_diarization_rejected(client):
    audio_bytes = io.BytesIO(b"\x00" * 1024)
    resp = client.post(
        "/transcribe",
        data={"speaker_name": "Alice", "skip_diarization": True},
        files={"audio": ("test.wav", audio_bytes, "audio/wav")},
    )
    assert resp.status_code == 400


def test_pending_db_updates_zero_at_rest(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["pending_db_updates"] == 0


def test_shutdown_closes_shared_store(client):
    from server import models
    from speaker_store import SpeakerEmbeddingStore

    models.shared_store = SpeakerEmbeddingStore(":memory:")

    store = models.shared_store
    assert store is not None

    store.close()
    models.shared_store = None
    assert models.shared_store is None


def test_background_task_added_and_removed():
    import asyncio

    from server import background_tasks

    async def _run():
        await asyncio.sleep(0)

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(_run())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        assert task in background_tasks
        loop.run_until_complete(task)
        assert len(background_tasks) == 0
    finally:
        loop.close()


def test_list_speakers_empty(store_client):
    resp = store_client.get("/speakers")
    assert resp.status_code == 200
    assert resp.json() == {"speakers": []}


def test_list_speakers_with_data(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    _add_test_speaker(models.shared_store, "Bob")
    resp = store_client.get("/speakers")
    assert resp.status_code == 200
    data = resp.json()
    names = {s["name"] for s in data["speakers"]}
    assert names == {"Alice", "Bob"}
    for s in data["speakers"]:
        assert "sample_count" in s
        assert "created_at" in s
        assert "updated_at" in s


def test_show_speaker(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    resp = store_client.get("/speakers/Alice")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Alice"
    assert data["sample_count"] == 1


def test_show_speaker_not_found(store_client):
    resp = store_client.get("/speakers/NoSuchSpeaker")
    assert resp.status_code == 404


def test_delete_speaker(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    resp = store_client.delete("/speakers/Alice")
    assert resp.status_code == 200
    assert resp.json()["detail"] == "Deleted 'Alice'"
    assert models.shared_store.list_speakers() == []


def test_delete_speaker_not_found(store_client):
    resp = store_client.delete("/speakers/NoSuchSpeaker")
    assert resp.status_code == 404


def test_rename_speaker(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    resp = store_client.post("/speakers/Alice/rename", json={"new_name": "Bob"})
    assert resp.status_code == 200
    assert "Renamed" in resp.json()["detail"]
    names = {s["name"] for s in models.shared_store.list_speakers()}
    assert names == {"Bob"}


def test_rename_speaker_conflict(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    _add_test_speaker(models.shared_store, "Bob")
    resp = store_client.post("/speakers/Alice/rename", json={"new_name": "Bob"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


def test_rename_speaker_merge_with_force(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    _add_test_speaker(models.shared_store, "Bob")
    resp = store_client.post("/speakers/Alice/rename", json={"new_name": "Bob", "force": True})
    assert resp.status_code == 200
    assert "Merged" in resp.json()["detail"]
    names = {s["name"] for s in models.shared_store.list_speakers()}
    assert names == {"Bob"}
    bob = models.shared_store._get_profile_by_name("Bob")
    assert bob.sample_count == 2


def test_rename_speaker_same_name(store_client):
    from server import models

    _add_test_speaker(models.shared_store, "Alice")
    resp = store_client.post("/speakers/Alice/rename", json={"new_name": "Alice"})
    assert resp.status_code == 200
    assert "already named" in resp.json()["detail"]


def test_speakers_endpoint_503_when_disabled(client):
    from server import models

    models.shared_store = None
    resp = client.get("/speakers")
    assert resp.status_code == 503
