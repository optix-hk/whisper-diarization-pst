import sys

from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np


def _make_mock_module(name, attrs=None):
    mod = ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
    speaker_embeddings: Dict[int, np.ndarray]


torch_mock = _make_mock_module(
    "torch",
    {
        "device": MagicMock,
        "Tensor": MagicMock,
        "float16": "float16",
        "float32": "float32",
        "cuda": _make_mock_module("torch.cuda", {"is_available": MagicMock(return_value=False)}),
        "from_numpy": MagicMock,
    },
)
sys.modules["torch"] = torch_mock

for mod_name in [
    "omegaconf",
    "nemo",
    "nemo.collections",
    "nemo.collections.asr",
    "nemo.collections.asr.models",
    "nemo.collections.asr.models.msdd_models",
    "nemo.collections.asr.parts",
    "nemo.collections.asr.parts.utils",
    "nemo.collections.asr.parts.utils.speaker_utils",
    "nemo.collections.asr.parts.mixins",
    "nemo.collections.asr.parts.mixins.diarization",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _make_mock_module(mod_name)

sys.modules["nemo.collections.asr.models"].NeuralDiarizer = MagicMock
sys.modules["nemo.collections.asr.models"].SortformerEncLabelModel = MagicMock
sys.modules["nemo.collections.asr.models.msdd_models"].NeuralDiarizer = MagicMock
sys.modules["nemo.collections.asr.parts.utils.speaker_utils"].rttm_to_labels = MagicMock
sys.modules["nemo.collections.asr.parts.mixins.diarization"].DiarizeConfig = MagicMock
sys.modules["omegaconf"].OmegaConf = MagicMock

for mod_name, attrs in [
    (
        "faster_whisper",
        {
            "WhisperModel": MagicMock,
            "BatchedInferencePipeline": MagicMock,
            "decode_audio": MagicMock,
        },
    ),
    (
        "ctc_forced_aligner",
        {
            "generate_emissions": MagicMock,
            "get_alignments": MagicMock,
            "get_spans": MagicMock,
            "load_alignment_model": MagicMock,
            "postprocess_results": MagicMock,
            "preprocess_text": MagicMock,
        },
    ),
    ("deepmultilingualpunctuation", {"PunctuationModel": MagicMock}),
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _make_mock_module(mod_name, attrs)
