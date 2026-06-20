# Per-Segment Speaker Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cluster-level speaker embedding matching with per-segment verification using a standalone TitaNet model, correcting individual segment misassignments from MSDD clustering.

**Architecture:** A new `SpeakerEmbedder` component loads `titanet_large` via NeMo's public API and extracts one 192-dim embedding per MSDD segment directly from the audio waveform. Each segment is matched against the SQLite speaker DB independently. Matched segments receive their per-segment label (correcting MSDD cluster errors). Unmatched segments fall back to the majority label of their MSDD cluster neighbors. Mode 3 is simplified to use the embedder directly instead of running full MSDD in the background.

**Tech Stack:** Python, NeMo (EncDecSpeakerLabelModel/titanet_large), PyTorch, SQLite, FastAPI, pytest

**Spec:** `docs/superpowers/specs/2026-06-20-per-segment-speaker-verification-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `speaker_embedder.py` | Create | `SpeakerEmbedder` class — loads titanet_large, extracts per-segment embeddings |
| `tests/test_speaker_embedder.py` | Create | Tests for SpeakerEmbedder (silence padding, slicing, None handling) |
| `diarization/msdd/msdd.py` | Modify | Remove `speaker_embeddings` from `DiarizationResult`, remove `_extract_embeddings()` |
| `speaker_matcher.py` | Modify | Refactor `match_speakers` for per-segment list input/output, remove collision resolution |
| `persistent_diarizer.py` | Modify | Refactor `resolve_speakers` for per-segment embeddings + cluster fallback, update `apply_persistent_labels` |
| `diarize_parallel.py` | Modify | Remove `speaker_embeddings` from subprocess dict, use SpeakerEmbedder in main process |
| `diarize.py` | Modify | Use SpeakerEmbedder + new `resolve_speakers` signature |
| `server.py` | Modify | Add `embedder_semaphore`, load SpeakerEmbedder, update Mode 2, simplify Mode 3, update `/health` |
| `tests/conftest.py` | Modify | Update mock `DiarizationResult`, add `EncDecSpeakerLabelModel` mock |
| `tests/test_speaker_matcher.py` | Modify | Update for per-segment input/output |
| `tests/test_persistent_diarizer.py` | Modify | Update for new signatures, add cluster fallback tests |
| `tests/test_server.py` | Modify | Update fixtures and tests for new flows |

---

### Task 1: Create SpeakerEmbedder

**Files:**
- Create: `speaker_embedder.py`
- Create: `tests/test_speaker_embedder.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add EncDecSpeakerLabelModel mock to conftest.py**

In `tests/conftest.py`, after line 55 (`sys.modules["nemo.collections.asr.models"].SortformerEncLabelModel = MagicMock`), add:

```python
sys.modules["nemo.collections.asr.models"].EncDecSpeakerLabelModel = MagicMock
```

- [ ] **Step 2: Write failing tests for SpeakerEmbedder**

Create `tests/test_speaker_embedder.py`:

```python
import numpy as np
import torch
from unittest.mock import MagicMock

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_speaker_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'speaker_embedder'`

- [ ] **Step 4: Create speaker_embedder.py**

Create `speaker_embedder.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_speaker_embedder.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS (all existing tests still pass)

- [ ] **Step 7: Commit**

```bash
git add speaker_embedder.py tests/test_speaker_embedder.py tests/conftest.py
git commit -m "feat: add SpeakerEmbedder with titanet_large for per-segment embedding extraction"
```

---

### Task 2: Refactor Persistence Layer (match_speakers, PersistentSpeakerDiarizer, DiarizationResult)

This task refactors all coupled persistence components together because they depend on each other's signatures.

**Files:**
- Modify: `tests/conftest.py`
- Modify: `diarization/msdd/msdd.py`
- Modify: `speaker_matcher.py`
- Modify: `persistent_diarizer.py`
- Modify: `diarize_parallel.py` (subprocess function only)
- Modify: `tests/test_speaker_matcher.py`
- Modify: `tests/test_persistent_diarizer.py`

- [ ] **Step 1: Update conftest.py mock DiarizationResult**

In `tests/conftest.py`, replace the mock `DiarizationResult` dataclass (lines 19-22):

```python
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
    speaker_embeddings: Dict[int, np.ndarray]
```

with:

```python
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
```

- [ ] **Step 2: Update DiarizationResult and MSDDDiarizer in msdd.py**

In `diarization/msdd/msdd.py`:

1. Change the import on line 7 from:
```python
from typing import Dict, List, Tuple, Union
```
to:
```python
from typing import List, Tuple, Union
```

2. Remove the `import numpy as np` on line 9 (no longer used after removing `_extract_embeddings`).

3. Replace the `DiarizationResult` dataclass (lines 17-20):
```python
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
    speaker_embeddings: Dict[int, np.ndarray]
```
with:
```python
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
```

4. Remove the entire `_extract_embeddings` method (lines 27-42).

5. In the `diarize` method, remove lines 96-97:
```python
            valid_speaker_ids = {sp for _, _, sp in labels}
            speaker_embeddings = self._extract_embeddings(valid_speaker_ids)
```

6. Change the return statement on line 99 from:
```python
        return DiarizationResult(speaker_ts=labels, speaker_embeddings=speaker_embeddings)
```
to:
```python
        return DiarizationResult(speaker_ts=labels)
```

- [ ] **Step 3: Update diarize_parallel.py subprocess function**

In `diarize_parallel.py`, change line 41 from:
```python
    queue.put({"speaker_ts": result.speaker_ts, "speaker_embeddings": result.speaker_embeddings})
```
to:
```python
    queue.put({"speaker_ts": result.speaker_ts})
```

- [ ] **Step 4: Rewrite speaker_matcher.py**

Replace the entire contents of `speaker_matcher.py` with:

```python
from typing import List, Optional, Tuple

import numpy as np

from speaker_store import SpeakerEmbeddingStore


def match_speakers(
    segment_embeddings: List[Optional[np.ndarray]],
    store: SpeakerEmbeddingStore,
    min_threshold: float = 0.75,
) -> dict[int, tuple[str | None, float]]:
    profiles = store.get_all_profiles()
    if not profiles:
        return {i: (None, 0.0) for i in range(len(segment_embeddings))}

    stored_vectors = []
    stored_names = []
    for p in profiles:
        norm = p.embedding / (np.linalg.norm(p.embedding) + 1e-8)
        stored_vectors.append(norm)
        stored_names.append(p.name)
    stored_matrix = np.stack(stored_vectors)

    matches: dict[int, tuple[str | None, float]] = {}
    for seg_idx, emb in enumerate(segment_embeddings):
        if emb is None:
            matches[seg_idx] = (None, 0.0)
            continue
        query_norm = emb / (np.linalg.norm(emb) + 1e-8)
        scores = stored_matrix @ query_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= min_threshold:
            matches[seg_idx] = (stored_names[best_idx], best_score)
        else:
            matches[seg_idx] = (None, best_score)

    return matches
```

- [ ] **Step 5: Rewrite persistent_diarizer.py**

Replace the entire contents of `persistent_diarizer.py` with:

```python
import sys
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from speaker_matcher import match_speakers
from speaker_store import SpeakerEmbeddingStore


def apply_persistent_labels(
    speaker_ts: list[tuple[int, int, int]],
    label_map: dict[int, str],
) -> list[tuple[int, int, str]]:
    return [
        (start, end, label_map.get(i, str(spk_id)))
        for i, (start, end, spk_id) in enumerate(speaker_ts)
    ]


def _get_sample_sentence(
    word_speaker_mapping: List[dict], speaker_id: int
) -> str:
    words = []
    for entry in word_speaker_mapping:
        if entry["speaker"] == speaker_id:
            words.append(entry["word"])
            if entry["word"] and entry["word"][-1] in ".?!":
                break
        elif words:
            break
    return " ".join(words) if words else ""


def _format_ms(ms: int) -> str:
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:05.2f}s"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def merge_new_speakers(
    new_speakers: Dict[int, np.ndarray], merge_threshold: float = 0.85
) -> Dict[int, int]:
    spk_ids = sorted(new_speakers.keys())
    merge_map: Dict[int, int] = {spk_id: spk_id for spk_id in spk_ids}

    for i, id_a in enumerate(spk_ids):
        for id_b in spk_ids[i + 1 :]:
            if merge_map[id_b] != id_b:
                continue
            sim = _cosine_similarity(new_speakers[id_a], new_speakers[id_b])
            if sim >= merge_threshold:
                merge_map[id_b] = id_a

    return merge_map


class PersistentSpeakerDiarizer:
    def __init__(
        self,
        store: SpeakerEmbeddingStore,
        min_threshold: float = 0.75,
        interactive: bool = False,
        merge_threshold: float = 0.85,
    ):
        self._store = store
        self._min_threshold = min_threshold
        self._interactive = interactive
        self._merge_threshold = merge_threshold

    def resolve_speakers(
        self,
        speaker_ts: list[tuple[int, int, int]],
        segment_embeddings: list[np.ndarray | None],
        word_speaker_mapping: list[dict] | None = None,
    ) -> dict[int, str]:
        matches = match_speakers(
            segment_embeddings, self._store, self._min_threshold
        )

        label_map: dict[int, str] = {}
        unmatched_indices: list[int] = []
        for seg_idx, (name, score) in matches.items():
            if name is not None:
                label_map[seg_idx] = name
            else:
                unmatched_indices.append(seg_idx)

        still_unmatched: list[int] = []
        for seg_idx in unmatched_indices:
            spk_id = speaker_ts[seg_idx][2]
            cluster_labels = [
                label_map[i]
                for i, (_, _, sid) in enumerate(speaker_ts)
                if sid == spk_id and i in label_map
            ]
            if cluster_labels:
                label_map[seg_idx] = Counter(cluster_labels).most_common(1)[0][0]
            else:
                still_unmatched.append(seg_idx)

        new_groups: dict[int, list[int]] = {}
        for seg_idx in still_unmatched:
            spk_id = speaker_ts[seg_idx][2]
            new_groups.setdefault(spk_id, []).append(seg_idx)

        new_speaker_embeddings: dict[int, np.ndarray] = {}
        for spk_id, seg_indices in new_groups.items():
            embs = [
                segment_embeddings[i]
                for i in seg_indices
                if segment_embeddings[i] is not None
            ]
            if embs:
                new_speaker_embeddings[spk_id] = np.mean(embs, axis=0)
            else:
                new_speaker_embeddings[spk_id] = np.zeros(192, dtype=np.float32)

        merge_map = merge_new_speakers(new_speaker_embeddings, self._merge_threshold)

        existing_names = {s["name"] for s in self._store.list_speakers()}
        assigned: dict[int, str] = {}
        next_num = 0

        for spk_id in sorted(new_groups.keys()):
            canonical = merge_map[spk_id]
            if canonical in assigned:
                label = assigned[canonical]
                for seg_idx in new_groups[spk_id]:
                    label_map[seg_idx] = label
                continue

            if self._interactive:
                chosen = self._interactive_prompt(
                    spk_id, existing_names, next_num, speaker_ts, word_speaker_mapping
                )
            else:
                chosen = self._next_available_name(existing_names, next_num)

            assigned[canonical] = chosen
            assigned[spk_id] = chosen
            existing_names.add(chosen)
            next_num += 1

            for seg_idx in new_groups[spk_id]:
                label_map[seg_idx] = chosen

            if chosen in {s["name"] for s in self._store.list_speakers()}:
                self._store.update_embedding(chosen, new_speaker_embeddings[spk_id])
            else:
                self._store.add_speaker(chosen, new_speaker_embeddings[spk_id])

            for other_id in new_groups:
                if other_id != spk_id and merge_map.get(other_id) == canonical:
                    self._store.update_embedding(chosen, new_speaker_embeddings[other_id])

        matched_by_name: dict[str, list[np.ndarray]] = {}
        for seg_idx, (name, _) in matches.items():
            if name is not None and segment_embeddings[seg_idx] is not None:
                matched_by_name.setdefault(name, []).append(segment_embeddings[seg_idx])

        for name, embs in matched_by_name.items():
            mean_emb = np.mean(embs, axis=0)
            self._store.update_embedding(name, mean_emb)

        return label_map

    def _interactive_prompt(
        self,
        spk_id: int,
        existing_names: Set[str],
        next_num: int,
        speaker_ts: list[tuple[int, int, int]],
        word_speaker_mapping: Optional[List[dict]],
    ) -> str:
        sample = ""
        if word_speaker_mapping is not None:
            sample = _get_sample_sentence(word_speaker_mapping, spk_id)
        spk_segments = [
            (s, e) for s, e, sp in speaker_ts if sp == spk_id
        ]
        if spk_segments:
            first_seg = spk_segments[0]
            timestamp = f"{_format_ms(first_seg[0])} - {_format_ms(first_seg[1])}"
        else:
            timestamp = "unknown"
        default_name = self._next_available_name(existing_names, next_num)

        stored_names = {s["name"] for s in self._store.list_speakers()}

        while True:
            prompt_parts = [f'New speaker detected ({timestamp}):']
            if sample:
                prompt_parts.append(f'  "{sample}"')
            prompt_parts.append(f"Enter name [default: {default_name}]: ")
            prompt = "\n".join(prompt_parts)

            if not sys.stdin.isatty():
                return default_name

            response = input(prompt).strip()
            chosen = response if response else default_name

            if chosen in stored_names:
                merge_prompt = (
                    f"  '{chosen}' already exists. "
                    f"Merge with existing speaker? [y/n]: "
                )
                merge_response = input(merge_prompt).strip().lower()
                if merge_response.startswith("y"):
                    return chosen
            else:
                return chosen

    def _next_available_name(self, existing_names: set, start: int) -> str:
        n = start
        while f"Speaker {n}" in existing_names:
            n += 1
        return f"Speaker {n}"
```

- [ ] **Step 6: Rewrite test_speaker_matcher.py**

Replace the entire contents of `tests/test_speaker_matcher.py` with:

```python
import numpy as np
import pytest

from speaker_matcher import match_speakers
from speaker_store import SpeakerEmbeddingStore


@pytest.fixture
def store():
    s = SpeakerEmbeddingStore(":memory:")
    yield s
    s.close()


def _make_embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(192).astype(np.float32)
    return emb / np.linalg.norm(emb)


def test_match_speakers_all_known(store):
    emb_a = _make_embedding(0)
    emb_b = _make_embedding(1)
    store.add_speaker("Alice", emb_a)
    store.add_speaker("Bob", emb_b)
    query_a = emb_a + _make_embedding(100) * 0.05
    query_a /= np.linalg.norm(query_a)
    query_b = emb_b + _make_embedding(101) * 0.05
    query_b /= np.linalg.norm(query_b)
    result = match_speakers([query_a, query_b], store, min_threshold=0.6)
    assert result[0][0] == "Alice"
    assert result[0][1] > 0.6
    assert result[1][0] == "Bob"
    assert result[1][1] > 0.6


def test_match_speakers_no_match(store):
    store.add_speaker("Alice", _make_embedding(0))
    unrelated = _make_embedding(99)
    result = match_speakers([unrelated], store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] < 0.6


def test_match_speakers_empty_store(store):
    emb = _make_embedding(0)
    result = match_speakers([emb], store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] == 0.0


def test_match_speakers_multiple_match_same_stored(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query1 = emb_a + _make_embedding(100) * 0.05
    query1 /= np.linalg.norm(query1)
    query2 = emb_a + _make_embedding(101) * 0.05
    query2 /= np.linalg.norm(query2)
    result = match_speakers([query1, query2], store, min_threshold=0.6)
    assert result[0][0] == "Alice"
    assert result[1][0] == "Alice"


def test_match_speakers_none_embedding(store):
    store.add_speaker("Alice", _make_embedding(0))
    result = match_speakers([None], store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] == 0.0


def test_match_speakers_multiple_stored(store):
    embs = {f"Speaker_{i}": _make_embedding(i) for i in range(5)}
    for name, emb in embs.items():
        store.add_speaker(name, emb)
    query = embs["Speaker_3"] + _make_embedding(300) * 0.05
    query /= np.linalg.norm(query)
    result = match_speakers([query], store, min_threshold=0.6)
    assert result[0][0] == "Speaker_3"
```

- [ ] **Step 7: Rewrite test_persistent_diarizer.py**

Replace the entire contents of `tests/test_persistent_diarizer.py` with:

```python
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
    assert result[1] == (1000, 2000, 1)
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
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests across all files)

- [ ] **Step 9: Commit**

```bash
git add tests/conftest.py diarization/msdd/msdd.py speaker_matcher.py persistent_diarizer.py diarize_parallel.py tests/test_speaker_matcher.py tests/test_persistent_diarizer.py
git commit -m "feat: refactor persistence layer for per-segment embedding verification"
```

---

### Task 3: Update diarize.py CLI Integration

**Files:**
- Modify: `diarize.py`

- [ ] **Step 1: Add SpeakerEmbedder import**

In `diarize.py`, add this import after line 21 (`from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels`):

```python
from speaker_embedder import SpeakerEmbedder
```

- [ ] **Step 2: Replace the persistence block**

In `diarize.py`, replace lines 253-263:

```python
if not args.skip_diarization and not args.no_persist and isinstance(diarization_result, DiarizationResult) and diarization_result.speaker_embeddings:
    store = SpeakerEmbeddingStore(args.speaker_db)
    pd = PersistentSpeakerDiarizer(
        store=store,
        min_threshold=args.match_threshold,
        interactive=args.interactive,
    )
    label_map = pd.resolve_speakers(diarization_result, word_speaker_mapping=wsm)
    speaker_ts = apply_persistent_labels(speaker_ts, label_map)
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    store.close()
```

with:

```python
if not args.skip_diarization and not args.no_persist and isinstance(diarization_result, DiarizationResult):
    embedder = SpeakerEmbedder(device=args.device)
    segment_embeddings = embedder.embed_segments(
        torch.from_numpy(audio_waveform), speaker_ts
    )
    del embedder
    torch.cuda.empty_cache()

    store = SpeakerEmbeddingStore(args.speaker_db)
    pd = PersistentSpeakerDiarizer(
        store=store,
        min_threshold=args.match_threshold,
        interactive=args.interactive,
    )
    label_map = pd.resolve_speakers(speaker_ts, segment_embeddings, word_speaker_mapping=wsm)
    speaker_ts = apply_persistent_labels(speaker_ts, label_map)
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    store.close()
```

- [ ] **Step 3: Run tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS (diarize.py is not directly tested, but imports must not break)

- [ ] **Step 4: Commit**

```bash
git add diarize.py
git commit -m "feat: integrate SpeakerEmbedder into diarize.py for per-segment verification"
```

---

### Task 4: Update diarize_parallel.py CLI Integration

**Files:**
- Modify: `diarize_parallel.py`

- [ ] **Step 1: Add SpeakerEmbedder import**

In `diarize_parallel.py`, add this import after line 21 (`from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels`):

```python
from speaker_embedder import SpeakerEmbedder
```

- [ ] **Step 2: Update the main-process persistence block**

In `diarize_parallel.py`, replace lines 266-279:

```python
    if not args.skip_diarization and not args.no_persist and speaker_embeddings:
        store = SpeakerEmbeddingStore(args.speaker_db)
        diarization_result = DiarizationResult(
            speaker_ts=speaker_ts, speaker_embeddings=speaker_embeddings
        )
        pd = PersistentSpeakerDiarizer(
            store=store,
            min_threshold=args.match_threshold,
            interactive=args.interactive,
        )
        label_map = pd.resolve_speakers(diarization_result, word_speaker_mapping=wsm)
        speaker_ts = apply_persistent_labels(speaker_ts, label_map)
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        store.close()
```

with:

```python
    if not args.skip_diarization and not args.no_persist:
        embedder = SpeakerEmbedder(device=args.device)
        segment_embeddings = embedder.embed_segments(
            torch.from_numpy(audio_waveform), speaker_ts
        )
        del embedder
        torch.cuda.empty_cache()

        store = SpeakerEmbeddingStore(args.speaker_db)
        pd = PersistentSpeakerDiarizer(
            store=store,
            min_threshold=args.match_threshold,
            interactive=args.interactive,
        )
        label_map = pd.resolve_speakers(speaker_ts, segment_embeddings, word_speaker_mapping=wsm)
        speaker_ts = apply_persistent_labels(speaker_ts, label_map)
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
        store.close()
```

- [ ] **Step 3: Remove unused speaker_embeddings variable**

In `diarize_parallel.py`, in the `else` block starting at line 255, remove the `speaker_embeddings` variable. Change:

```python
    else:
        nemo_process.join()
        if results_queue.empty():
            raise RuntimeError("Diarization process did not return any results.")

        diarization_dict = results_queue.get_nowait()
        speaker_ts = diarization_dict["speaker_ts"]
        speaker_embeddings = diarization_dict.get("speaker_embeddings", {})
```

to:

```python
    else:
        nemo_process.join()
        if results_queue.empty():
            raise RuntimeError("Diarization process did not return any results.")

        diarization_dict = results_queue.get_nowait()
        speaker_ts = diarization_dict["speaker_ts"]
```

- [ ] **Step 4: Remove unused DiarizationResult import if no longer needed**

In `diarize_parallel.py`, check if `DiarizationResult` is still used anywhere in the file. If it is not, change line 20 from:

```python
from diarization import DiarizationResult, MSDDDiarizer
```
to:
```python
from diarization import MSDDDiarizer
```

- [ ] **Step 5: Run tests to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add diarize_parallel.py
git commit -m "feat: integrate SpeakerEmbedder into diarize_parallel.py for per-segment verification"
```

---

### Task 5: Server Startup, Semaphores, and /health

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Add SpeakerEmbedder import and embedder_semaphore**

In `server.py`, add this import after line 35 (`from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels`):

```python
from speaker_embedder import SpeakerEmbedder
```

Add the new semaphore after line 97 (`diarizer_semaphore = asyncio.Semaphore(1)`):

```python
embedder_semaphore = asyncio.Semaphore(1)
```

- [ ] **Step 2: Add embedder field to Models class**

In `server.py`, in the `Models.__init__` method, add after line 83 (`self.diarizer_model = None`):

```python
        self.embedder: SpeakerEmbedder | None = None
```

- [ ] **Step 3: Load SpeakerEmbedder at startup**

In `server.py`, in the `load_models` function, after line 135 (`logger.info(f"Diarizer loaded in {time.time() - t3:.1f}s")`), add:

```python
    if models.speaker_persistence:
        t4 = time.time()
        logger.info("Loading speaker embedder (titanet_large)...")
        try:
            models.embedder = SpeakerEmbedder(device=device)
            logger.info(f"Speaker embedder loaded in {time.time() - t4:.1f}s")
        except Exception:
            logger.warning("Failed to load speaker embedder", exc_info=True)
```

- [ ] **Step 4: Update /health endpoint**

In `server.py`, in the `health` function, add `embedder_loaded` to the returned dict after line 170 (`"diarizer_loaded": models.diarizer_model is not None,`):

```python
        "embedder_loaded": models.embedder is not None,
```

- [ ] **Step 5: Update test fixtures to mock SpeakerEmbedder**

In `tests/test_server.py`, in both the `client` and `store_client` fixtures, add this patch inside the `with` block, after `patch("server.SortformerDiarizer", return_value=MagicMock())`:

```python
        patch("server.SpeakerEmbedder", return_value=MagicMock()),
```

Also add `models.embedder = MagicMock()` after `models.diarizer_model = MagicMock()` in both fixtures.

- [ ] **Step 6: Update /health test**

In `tests/test_server.py`, update `test_health_endpoint_returns_all_fields` to also check for the new field:

```python
def test_health_endpoint_returns_all_fields(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "device" in data
    assert "pending_db_updates" in data
    assert "embedder_loaded" in data
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add SpeakerEmbedder to server startup with embedder_semaphore and /health"
```

---

### Task 6: Server Mode 2 and Mode 3 Flow Updates

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server.py`

- [ ] **Step 1: Add _run_embedding helper function**

In `server.py`, after the `_run_diarization` function (line 294), add:

```python
def _run_embedding(audio_waveform, speaker_ts):
    return models.embedder.embed_segments(
        torch.from_numpy(audio_waveform), speaker_ts
    )


def _run_embed_clip(audio_waveform, start_ms, end_ms):
    return models.embedder.embed_segment(
        torch.from_numpy(audio_waveform), start_ms, end_ms
    )
```

- [ ] **Step 2: Update Mode 3 flow (simplified — no background MSDD)**

In `server.py`, replace the Mode 3 background task (lines 448-467):

```python
        async def _background_diarize():
            try:
                async with diarizer_semaphore:
                    diarization_result = await loop.run_in_executor(
                        None, _run_diarization, audio_waveform
                    )
                if (
                    isinstance(diarization_result, DiarizationResult)
                    and diarization_result.speaker_embeddings
                ):
                    with db_lock:
                        if models.shared_store is not None:
                            emb = list(diarization_result.speaker_embeddings.values())[0]
                            existing = models.shared_store._get_profile_by_name(speaker_name)
                            if existing is not None:
                                models.shared_store.update_embedding(speaker_name, emb)
                            else:
                                models.shared_store.add_speaker(speaker_name, emb)
            except Exception:
                logger.warning("Background DB update failed", exc_info=True)
```

with:

```python
        async def _background_embed():
            try:
                async with embedder_semaphore:
                    embedding = await loop.run_in_executor(
                        None, _run_embed_clip, audio_waveform,
                        first_word_start, last_word_end
                    )
                with db_lock:
                    if models.shared_store is not None:
                        existing = models.shared_store._get_profile_by_name(speaker_name)
                        if existing is not None:
                            models.shared_store.update_embedding(speaker_name, embedding)
                        else:
                            models.shared_store.add_speaker(speaker_name, embedding)
            except Exception:
                logger.warning("Background DB update failed", exc_info=True)
```

- [ ] **Step 3: Update the task creation for Mode 3**

In `server.py`, change line 469 from:

```python
        task = asyncio.create_task(_background_diarize())
```

to:

```python
        task = asyncio.create_task(_background_embed())
```

- [ ] **Step 4: Update validation message for speaker_name + skip_diarization**

In `server.py`, change lines 370-374 from:

```python
            detail=(
                "speaker_name requires skip_diarization=false "
                "(diarization runs in background to extract embedding)"
            ),
```

to:

```python
            detail=(
                "speaker_name requires skip_diarization=false "
                "(speaker embedding is extracted in background)"
            ),
```

- [ ] **Step 5: Update Mode 2 flow with embedder step**

In `server.py`, replace lines 488-512:

```python
        async with diarizer_semaphore:
            diarization_result = await loop.run_in_executor(None, _run_diarization, audio_waveform)
        if isinstance(diarization_result, DiarizationResult):
            speaker_ts = diarization_result.speaker_ts
        else:
            speaker_ts = diarization_result

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if (
        models.speaker_persistence
        and not skip_diarization
        and not no_persist
        and isinstance(diarization_result, DiarizationResult)
        and diarization_result.speaker_embeddings
    ):
        with db_lock:
            if models.shared_store is not None:
                pd = PersistentSpeakerDiarizer(
                    store=models.shared_store,
                    min_threshold=match_threshold,
                    interactive=False,
                )
                label_map = pd.resolve_speakers(diarization_result, word_speaker_mapping=wsm)
                speaker_ts = apply_persistent_labels(speaker_ts, label_map)
                wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

with:

```python
        async with diarizer_semaphore:
            diarization_result = await loop.run_in_executor(None, _run_diarization, audio_waveform)
        if isinstance(diarization_result, DiarizationResult):
            speaker_ts = diarization_result.speaker_ts
        else:
            speaker_ts = diarization_result

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if (
        models.speaker_persistence
        and not skip_diarization
        and not no_persist
        and models.embedder is not None
        and isinstance(diarization_result, DiarizationResult)
    ):
        async with embedder_semaphore:
            segment_embeddings = await loop.run_in_executor(
                None, _run_embedding, audio_waveform, speaker_ts
            )
        with db_lock:
            if models.shared_store is not None:
                pd = PersistentSpeakerDiarizer(
                    store=models.shared_store,
                    min_threshold=match_threshold,
                    interactive=False,
                )
                label_map = pd.resolve_speakers(speaker_ts, segment_embeddings, word_speaker_mapping=wsm)
                speaker_ts = apply_persistent_labels(speaker_ts, label_map)
                wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

- [ ] **Step 6: Remove DiarizationResult import if no longer needed in server.py**

Check if `DiarizationResult` is still referenced anywhere in `server.py`. The `isinstance(diarization_result, DiarizationResult)` check still uses it, so keep the import. No change needed.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests across all files)

- [ ] **Step 9: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: update server Mode 2 with per-segment embedder, simplify Mode 3"
```

---

## Self-Review Notes

**Spec coverage check:**
- SpeakerEmbedder (new component) → Task 1 ✓
- DiarizationResult removes speaker_embeddings → Task 2 ✓
- MSDDDiarizer._extract_embeddings removed → Task 2 ✓
- match_speakers refactored for per-segment → Task 2 ✓
- PersistentSpeakerDiarizer.resolve_speakers refactored → Task 2 ✓
- apply_persistent_labels keyed by segment index → Task 2 ✓
- Cluster fallback logic → Task 2 ✓
- DB update with mean of matched segments → Task 2 ✓
- CLI diarize.py integration → Task 3 ✓
- CLI diarize_parallel.py integration → Task 4 ✓
- Server startup + embedder_semaphore → Task 5 ✓
- Server /health updated → Task 5 ✓
- Server Mode 2 flow → Task 6 ✓
- Server Mode 3 simplified → Task 6 ✓
- Testing changes (conftest, test_speaker_matcher, test_persistent_diarizer, test_server) → Tasks 1, 2, 5, 6 ✓
- Error handling (embedder load failure, per-segment failure, None embeddings) → Tasks 1, 5, 6 ✓

**Type consistency check:**
- `embed_segment` returns `np.ndarray` — consistent across speaker_embedder.py, test_speaker_embedder.py, server.py `_run_embed_clip`
- `embed_segments` returns `list[np.ndarray | None]` — consistent across speaker_embedder.py, test_speaker_embedder.py, server.py `_run_embedding`, persistent_diarizer.py
- `match_speakers` takes `List[Optional[np.ndarray]]` → `dict[int, tuple[str | None, float]]` — consistent across speaker_matcher.py, test_speaker_matcher.py, persistent_diarizer.py
- `resolve_speakers` takes `(speaker_ts, segment_embeddings, word_speaker_mapping)` → `dict[int, str]` — consistent across persistent_diarizer.py, test_persistent_diarizer.py, diarize.py, diarize_parallel.py, server.py
- `apply_persistent_labels` takes `(speaker_ts, dict[int, str])` → `list[tuple[int, int, str]]` — consistent across persistent_diarizer.py, test_persistent_diarizer.py, diarize.py, diarize_parallel.py, server.py
- `DiarizationResult` has only `speaker_ts` — consistent across msdd.py, conftest.py
