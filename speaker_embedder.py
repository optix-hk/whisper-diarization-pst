import logging

import numpy as np
import torch

from nemo.collections.asr.models import EncDecSpeakerLabelModel

logger = logging.getLogger(__name__)


class SpeakerEmbedder:
    def __init__(self, device: str | torch.device):
        self.model = EncDecSpeakerLabelModel.from_pretrained("titanet_large")
        self.model.to(device).eval()
        self.device = device
        self.min_duration_samples = int(0.5 * 16000)

    def embed_segment(
        self, audio: torch.Tensor, start_ms: int, end_ms: int
    ) -> np.ndarray:
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        start_sample = int(start_ms * 16)
        end_sample = int(end_ms * 16)
        segment = audio[start_sample:end_sample]
        if segment.shape[0] < self.min_duration_samples:
            pad = self.min_duration_samples - segment.shape[0]
            segment = torch.nn.functional.pad(segment, (0, pad))
        with torch.no_grad():
            _, emb = self.model.forward(
                segment.unsqueeze(0).to(self.device),
                torch.tensor([segment.shape[0]], device=self.device),
            )
        return emb.squeeze(0).cpu().numpy()

    def embed_segments(
        self, audio: torch.Tensor, speaker_ts: list
    ) -> list[np.ndarray | None]:
        results = []
        for start, end, _ in speaker_ts:
            try:
                results.append(self.embed_segment(audio, start, end))
            except Exception:
                logger.warning(
                    "Embedding extraction failed for segment %d-%d ms",
                    start,
                    end,
                    exc_info=True,
                )
                results.append(None)
        return results
