import io

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client():
    with patch("server.faster_whisper"), \
         patch("server.load_alignment_model", return_value=(MagicMock(), MagicMock())), \
         patch("server.PunctuationModel", return_value=MagicMock()), \
         patch("server.MSDDDiarizer", return_value=MagicMock()), \
         patch("server.SortformerDiarizer", return_value=MagicMock()):
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
