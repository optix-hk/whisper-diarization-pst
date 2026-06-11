import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Tuple

import numpy as np
from unittest.mock import MagicMock


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


torch_mock = _make_mock_module("torch", {"device": MagicMock, "Tensor": MagicMock})
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
