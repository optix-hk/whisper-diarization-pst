from unittest.mock import MagicMock

import numpy as np
import torch

from speaker_embedder import SpeakerEmbedder


def _make_embedder():
    """Create a SpeakerEmbedder with a mock model (bypasses __init__)."""
    embedder = SpeakerEmbedder.__new__(SpeakerEmbedder)
    embedder.device = "cpu"
    embedder.min_duration_samples = int(0.5 * 16000)

    def mock_forward(signal, length=None):
        batch_size = signal.shape[0]
        emb = torch.randn(batch_size, 192)
        return MagicMock(), emb

    embedder.model = MagicMock()
    embedder.model.forward = mock_forward
    return embedder


def test_embed_segment_returns_192_dim():
    embedder = _make_embedder()
    audio = torch.randn(16000 * 5)
    emb = embedder.embed_segment(audio, 1000, 2000)
    assert emb.shape == (192,)
    assert emb.dtype == np.float32


def test_embed_segment_silence_pads_short_segment():
    embedder = _make_embedder()
    audio = torch.randn(16000 * 5)
    captured = []

    original = embedder.model.forward

    def capturing(signal, length=None):
        captured.append(signal.shape[1])
        return original(signal, length)

    embedder.model.forward = capturing

    emb = embedder.embed_segment(audio, 1000, 1100)  # 0.1s segment
    assert emb.shape == (192,)
    assert captured[0] >= embedder.min_duration_samples


def test_embed_segment_slices_correctly():
    embedder = _make_embedder()
    audio = torch.ones(16000 * 5)
    captured = []

    original = embedder.model.forward

    def capturing(signal, length=None):
        captured.append(signal.clone())
        return original(signal, length)

    embedder.model.forward = capturing

    embedder.embed_segment(audio, 1000, 2000)  # 1s-2s = 16000 samples
    assert captured[0].shape[1] == 16000
    assert (captured[0] == 1.0).all()


def test_embed_segments_aligned_with_speaker_ts():
    embedder = _make_embedder()
    audio = torch.randn(16000 * 10)
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1), (2000, 3000, 0)]
    embeddings = embedder.embed_segments(audio, speaker_ts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert emb.shape == (192,)


def test_embed_segments_returns_none_on_failure():
    embedder = _make_embedder()
    embedder.model.forward = MagicMock(side_effect=RuntimeError("GPU OOM"))
    audio = torch.randn(16000 * 5)
    speaker_ts = [(0, 1000, 0)]
    embeddings = embedder.embed_segments(audio, speaker_ts)
    assert embeddings == [None]


def test_embed_segments_partial_failure_returns_none_for_failed_only():
    embedder = _make_embedder()
    audio = torch.randn(16000 * 10)
    speaker_ts = [(0, 1000, 0), (1000, 2000, 1), (2000, 3000, 0)]
    call_count = [0]
    original = embedder.model.forward

    def fail_on_second(signal, length=None):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("fail")
        return original(signal, length)

    embedder.model.forward = fail_on_second
    embeddings = embedder.embed_segments(audio, speaker_ts)
    assert len(embeddings) == 3
    assert embeddings[0] is not None
    assert embeddings[1] is None
    assert embeddings[2] is not None
