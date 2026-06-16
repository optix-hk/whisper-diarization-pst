import io

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
    ):
        from server import app, background_tasks, models

        models.whisper_model = MagicMock()
        models.whisper_pipeline = MagicMock()
        models.alignment_model = MagicMock()
        models.alignment_tokenizer = MagicMock()
        models.punct_model = MagicMock()
        models.diarizer_model = MagicMock()
        models.device = "cpu"
        background_tasks.clear()
        with TestClient(app) as c:
            yield c


def test_health_endpoint_returns_all_fields(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "device" in data
    assert "pending_db_updates" in data


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
