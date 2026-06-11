# Persistent Speaker Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent speaker recognition to the whisper-diarization pipeline by extracting TitaNet embeddings from NeMo's MSDD diarizer, storing them in SQLite, and matching speakers across runs.

**Architecture:** Post-diarization embedding extraction and matching. After MSDD diarization completes, per-speaker mean embeddings are extracted from NeMo's internal state, compared against a persistent SQLite database, and speakers are relabeled with persistent names. The diarization pipeline itself is untouched.

**Tech Stack:** SQLite (via Python stdlib `sqlite3`), NumPy (already a transitive dependency), PyTorch (already used), NeMo TitaNet embeddings (192-dim float32 vectors).

---

## File Structure

| File | Responsibility |
|---|---|
| `speaker_store.py` (new) | `SpeakerEmbeddingStore` — SQLite CRUD for speaker profiles |
| `speaker_matcher.py` (new) | `match_speakers()` — pure-function cosine similarity matching |
| `persistent_diarizer.py` (new) | `PersistentSpeakerDiarizer` orchestrator + `DiarizationResult` dataclass + `apply_persistent_labels()` |
| `speakerctl.py` (new) | CLI tool for profile management (`list`, `rename`, `delete`, `show`) |
| `diarization/msdd/msdd.py` (modify) | Return `DiarizationResult` with embeddings extracted from `emb_sess_test_dict` |
| `diarization/__init__.py` (modify) | Export `DiarizationResult` |
| `helpers.py` (modify) | Update `get_sentences_speaker_mapping` to handle persistent string labels |
| `diarize.py` (modify) | Add CLI args, integrate persistent speaker matching between Stage 4 and Stage 5 |
| `diarize_parallel.py` (modify) | Same integration as `diarize.py`, update `diarize_parallel()` to pass `DiarizationResult` |
| `tests/test_speaker_store.py` (new) | Unit tests for `SpeakerEmbeddingStore` |
| `tests/test_speaker_matcher.py` (new) | Unit tests for `SpeakerMatcher` |
| `tests/test_persistent_diarizer.py` (new) | Unit tests for `PersistentSpeakerDiarizer` |

---

### Task 1: SpeakerEmbeddingStore — SQLite Storage

**Files:**
- Create: `speaker_store.py`
- Create: `tests/test_speaker_store.py`

- [ ] **Step 1: Write failing tests for SpeakerEmbeddingStore**

Create `tests/test_speaker_store.py`:

```python
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
    query = sample_embedding + np.random.default_rng(0).standard_normal(192).astype(np.float32) * 0.1
    query /= np.linalg.norm(query)
    name, score = store.find_best_match(query, min_threshold=0.6)
    assert name == "Alice"
    assert score > 0.6


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
    new_emb = sample_embedding + np.random.default_rng(1).standard_normal(192).astype(np.float32) * 0.1
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_speaker_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'speaker_store'`

- [ ] **Step 3: Implement SpeakerEmbeddingStore**

Create `speaker_store.py`:

```python
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpeakerProfile:
    id: int
    name: str
    embedding: np.ndarray
    sample_count: int


class SpeakerEmbeddingStore:
    def __init__(self, db_path: str = "~/.whisper-diarization/speakers.db"):
        resolved = Path(db_path).expanduser()
        if str(resolved) != ":memory:":
            resolved.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(resolved))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                sample_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    def add_speaker(self, name: str, embedding: np.ndarray):
        blob = self._serialize_embedding(embedding)
        try:
            self._conn.execute(
                "INSERT INTO speakers (name, embedding) VALUES (?, ?)",
                (name, blob),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Speaker '{name}' already exists")

    def get_all_profiles(self) -> List[SpeakerProfile]:
        rows = self._conn.execute("SELECT * FROM speakers").fetchall()
        return [
            SpeakerProfile(
                id=row["id"],
                name=row["name"],
                embedding=self._deserialize_embedding(row["embedding"]),
                sample_count=row["sample_count"],
            )
            for row in rows
        ]

    def find_best_match(
        self, embedding: np.ndarray, min_threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        profiles = self.get_all_profiles()
        if not profiles:
            return None, 0.0
        best_name = None
        best_score = -1.0
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        for profile in profiles:
            stored_norm = profile.embedding / (np.linalg.norm(profile.embedding) + 1e-8)
            score = float(np.dot(query_norm, stored_norm))
            if score > best_score:
                best_score = score
                best_name = profile.name
        if best_score < min_threshold:
            return None, best_score
        return best_name, best_score

    def update_embedding(self, name: str, new_embedding: np.ndarray):
        profiles = self.get_all_profiles()
        profile = next((p for p in profiles if p.name == name), None)
        if profile is None:
            raise ValueError(f"Speaker '{name}' not found")
        updated = (profile.embedding * profile.sample_count + new_embedding) / (
            profile.sample_count + 1
        )
        blob = self._serialize_embedding(updated)
        self._conn.execute(
            "UPDATE speakers SET embedding = ?, sample_count = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
            (blob, profile.sample_count + 1, name),
        )
        self._conn.commit()

    def rename_speaker(self, old_name: str, new_name: str):
        cursor = self._conn.execute(
            "UPDATE speakers SET name = ? WHERE name = ?", (new_name, old_name)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Speaker '{old_name}' not found")
        self._conn.commit()

    def delete_speaker(self, name: str):
        cursor = self._conn.execute("DELETE FROM speakers WHERE name = ?", (name,))
        if cursor.rowcount == 0:
            raise ValueError(f"Speaker '{name}' not found")
        self._conn.commit()

    def list_speakers(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT name, sample_count, created_at, updated_at FROM speakers"
        ).fetchall()
        return [
            {
                "name": row["name"],
                "sample_count": row["sample_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def close(self):
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_speaker_store.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add speaker_store.py tests/test_speaker_store.py
git commit -m "feat: add SpeakerEmbeddingStore with SQLite-backed speaker profiles"
```

---

### Task 2: SpeakerMatcher — Cosine Similarity Matching

**Files:**
- Create: `speaker_matcher.py`
- Create: `tests/test_speaker_matcher.py`

- [ ] **Step 1: Write failing tests for match_speakers**

Create `tests/test_speaker_matcher.py`:

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
    speaker_embeddings = {0: query_a, 1: query_b}
    result = match_speakers(speaker_embeddings, store, min_threshold=0.6)
    assert result[0][0] == "Alice"
    assert result[0][1] > 0.6
    assert result[1][0] == "Bob"
    assert result[1][1] > 0.6


def test_match_speakers_no_match(store):
    store.add_speaker("Alice", _make_embedding(0))
    unrelated = _make_embedding(99)
    result = match_speakers({0: unrelated}, store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] < 0.6


def test_match_speakers_empty_store(store):
    emb = _make_embedding(0)
    result = match_speakers({0: emb}, store, min_threshold=0.6)
    assert result[0][0] is None
    assert result[0][1] == 0.0


def test_match_speakers_collision_resolved_by_score(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query_close = emb_a + _make_embedding(100) * 0.05
    query_close /= np.linalg.norm(query_close)
    query_far = emb_a + _make_embedding(200) * 0.3
    query_far /= np.linalg.norm(query_far)
    speaker_embeddings = {0: query_close, 1: query_far}
    result = match_speakers(speaker_embeddings, store, min_threshold=0.6)
    assert result[0][0] == "Alice"
    assert result[1][0] is None


def test_match_speakers_multiple_stored(store):
    embs = {f"Speaker_{i}": _make_embedding(i) for i in range(5)}
    for name, emb in embs.items():
        store.add_speaker(name, emb)
    query = embs["Speaker_3"] + _make_embedding(300) * 0.05
    query /= np.linalg.norm(query)
    result = match_speakers({0: query}, store, min_threshold=0.6)
    assert result[0][0] == "Speaker_3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_speaker_matcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'speaker_matcher'`

- [ ] **Step 3: Implement match_speakers**

Create `speaker_matcher.py`:

```python
from typing import Dict, Optional, Tuple

import numpy as np

from speaker_store import SpeakerEmbeddingStore


def match_speakers(
    speaker_embeddings: Dict[int, np.ndarray],
    store: SpeakerEmbeddingStore,
    min_threshold: float = 0.6,
) -> Dict[int, Tuple[Optional[str], float]]:
    profiles = store.get_all_profiles()
    if not profiles:
        return {spk_id: (None, 0.0) for spk_id in speaker_embeddings}

    stored_vectors = []
    stored_names = []
    for p in profiles:
        norm = p.embedding / (np.linalg.norm(p.embedding) + 1e-8)
        stored_vectors.append(norm)
        stored_names.append(p.name)
    stored_matrix = np.stack(stored_vectors)

    raw_matches: Dict[int, Tuple[Optional[str], float]] = {}
    for spk_id, emb in speaker_embeddings.items():
        query_norm = emb / (np.linalg.norm(emb) + 1e-8)
        scores = stored_matrix @ query_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= min_threshold:
            raw_matches[spk_id] = (stored_names[best_idx], best_score)
        else:
            raw_matches[spk_id] = (None, best_score)

    claimed: Dict[str, int] = {}
    for spk_id, (name, score) in raw_matches.items():
        if name is not None:
            if name not in claimed:
                claimed[name] = spk_id
            else:
                prev_id = claimed[name]
                prev_score = raw_matches[prev_id][1]
                if score > prev_score:
                    raw_matches[prev_id] = (None, raw_matches[prev_id][1])
                    claimed[name] = spk_id
                else:
                    raw_matches[spk_id] = (None, score)

    return raw_matches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_speaker_matcher.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add speaker_matcher.py tests/test_speaker_matcher.py
git commit -m "feat: add SpeakerMatcher with cosine similarity matching"
```

---

### Task 3: DiarizationResult and Embedding Extraction from MSDD

**Files:**
- Modify: `diarization/msdd/msdd.py`
- Modify: `diarization/__init__.py`

- [ ] **Step 1: Add DiarizationResult dataclass to msdd.py**

At the top of `diarization/msdd/msdd.py`, add the import for `dataclass` and the new dataclass definition. Add after the existing imports (after line 12):

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
    speaker_embeddings: Dict[int, np.ndarray]
```

- [ ] **Step 2: Modify MSDDDiarizer.diarize() to extract embeddings and return DiarizationResult**

Replace the body of `MSDDDiarizer.diarize()` (lines 19-71) with:

```python
    def diarize(self, audio: torch.Tensor) -> DiarizationResult:
        with tempfile.TemporaryDirectory() as temp_path:
            pcm = (audio.cpu().numpy() * 32768).clip(-32768, 32767).astype("int16")
            with wave.open(os.path.join(temp_path, "mono_file.wav"), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm.tobytes())

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": os.path.join(temp_path, "mono_file.wav"),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
            }

            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            self.model._initialize_configs(
                manifest_path=manifest_path,
                max_speakers=8,
                num_speakers=None,
                tmpdir=temp_path,
                batch_size=24,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = temp_path
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()

            pred_labels_clus = rttm_to_labels(
                os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            )

            labels = []
            for label in pred_labels_clus:
                start, end, speaker = label.split()
                start, end = float(start), float(end)
                start, end = int(start * 1000), int(end * 1000)
                labels.append((start, end, int(speaker.split("_")[1])))

            labels = sorted(labels, key=lambda x: x[0])

            speaker_embeddings = self._extract_embeddings()

        return DiarizationResult(speaker_ts=labels, speaker_embeddings=speaker_embeddings)
```

Add the `_extract_embeddings` method to `MSDDDiarizer` class (before `create_config()`):

```python
    def _extract_embeddings(self) -> Dict[int, np.ndarray]:
        speaker_embeddings = {}
        try:
            emb_sess = self.model.clustering_embedding.emb_sess_test_dict
            if not emb_sess:
                return speaker_embeddings
            base_scale = list(emb_sess.keys())[0]
            for uniq_name, data in emb_sess[base_scale].items():
                avg_embs = data["avg_embs"]
                num_speakers = avg_embs.shape[1]
                for spk_idx in range(num_speakers):
                    emb = avg_embs[:, spk_idx].cpu().numpy()
                    speaker_embeddings[spk_idx] = emb
        except (AttributeError, KeyError, IndexError):
            pass
        return speaker_embeddings
```

- [ ] **Step 3: Update diarization/__init__.py to export DiarizationResult**

Replace the contents of `diarization/__init__.py` with:

```python
from .msdd.msdd import DiarizationResult, MSDDDiarizer
from .sortformer.sortformer import SortformerDiarizer

__all__ = ["DiarizationResult", "MSDDDiarizer", "SortformerDiarizer"]
```

- [ ] **Step 4: Verify the module can be imported**

Run: `python -c "from diarization import DiarizationResult, MSDDDiarizer; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add diarization/msdd/msdd.py diarization/__init__.py
git commit -m "feat: extract speaker embeddings from MSDD, return DiarizationResult"
```

---

### Task 4: PersistentSpeakerDiarizer — Orchestrator

**Files:**
- Create: `persistent_diarizer.py`
- Create: `tests/test_persistent_diarizer.py`

- [ ] **Step 1: Write failing tests for PersistentSpeakerDiarizer**

Create `tests/test_persistent_diarizer.py`:

```python
import numpy as np
import pytest

from diarization.msdd.msdd import DiarizationResult
from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels
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
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0), (1000, 2000, 1)],
        speaker_embeddings={0: _make_embedding(0), 1: _make_embedding(1)},
    )
    label_map = pd.resolve_speakers(result)
    assert label_map[0] == "Speaker 0"
    assert label_map[1] == "Speaker 1"
    speakers = store.list_speakers()
    assert len(speakers) == 2


def test_resolve_speakers_matches_known(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query = emb_a + _make_embedding(100) * 0.05
    query /= np.linalg.norm(query)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0)],
        speaker_embeddings={0: query},
    )
    label_map = pd.resolve_speakers(result)
    assert label_map[0] == "Alice"


def test_resolve_speakers_updates_known_embedding(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query = emb_a + _make_embedding(100) * 0.05
    query /= np.linalg.norm(query)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0)],
        speaker_embeddings={0: query},
    )
    pd.resolve_speakers(result)
    profiles = store.get_all_profiles()
    assert profiles[0].sample_count == 2


def test_resolve_speakers_mixed_known_and_new(store):
    emb_a = _make_embedding(0)
    store.add_speaker("Alice", emb_a)
    query_a = emb_a + _make_embedding(100) * 0.05
    query_a /= np.linalg.norm(query_a)
    query_b = _make_embedding(1)
    pd = PersistentSpeakerDiarizer(store, min_threshold=0.6, interactive=False)
    result = DiarizationResult(
        speaker_ts=[(0, 1000, 0), (1000, 2000, 1)],
        speaker_embeddings={0: query_a, 1: query_b},
    )
    label_map = pd.resolve_speakers(result)
    assert label_map[0] == "Alice"
    assert label_map[1] == "Speaker 0"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_persistent_diarizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'persistent_diarizer'`

- [ ] **Step 3: Implement PersistentSpeakerDiarizer and apply_persistent_labels**

Create `persistent_diarizer.py`:

```python
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from diarization.msdd.msdd import DiarizationResult
from speaker_matcher import match_speakers
from speaker_store import SpeakerEmbeddingStore


def apply_persistent_labels(
    speaker_ts: List[Tuple[int, int, int]], label_map: Dict[int, str]
) -> List[Tuple[int, int, str]]:
    return [
        (start, end, label_map.get(spk_id, spk_id))
        for start, end, spk_id in speaker_ts
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


class PersistentSpeakerDiarizer:
    def __init__(
        self,
        store: SpeakerEmbeddingStore,
        min_threshold: float = 0.6,
        interactive: bool = False,
    ):
        self._store = store
        self._min_threshold = min_threshold
        self._interactive = interactive

    def resolve_speakers(
        self,
        result: DiarizationResult,
        word_speaker_mapping: Optional[List[dict]] = None,
    ) -> Dict[int, str]:
        matches = match_speakers(
            result.speaker_embeddings, self._store, self._min_threshold
        )

        label_map: Dict[int, str] = {}
        new_speakers: Dict[int, np.ndarray] = {}

        for spk_id, (name, score) in matches.items():
            if name is not None:
                label_map[spk_id] = name
                self._store.update_embedding(name, result.speaker_embeddings[spk_id])
            else:
                new_speakers[spk_id] = result.speaker_embeddings[spk_id]

        if new_speakers:
            existing_names = {s["name"] for s in self._store.list_speakers()}
            next_num = 0
            for spk_id in sorted(new_speakers.keys()):
                if self._interactive:
                    sample = ""
                    if word_speaker_mapping is not None:
                        sample = _get_sample_sentence(word_speaker_mapping, spk_id)
                    spk_segments = [
                        (s, e) for s, e, sp in result.speaker_ts if sp == spk_id
                    ]
                    if spk_segments:
                        first_seg = spk_segments[0]
                        timestamp = f"{_format_ms(first_seg[0])} - {_format_ms(first_seg[1])}"
                    else:
                        timestamp = "unknown"
                    default_name = self._next_available_name(existing_names, next_num)
                    prompt_parts = [f'New speaker detected ({timestamp}):']
                    if sample:
                        prompt_parts.append(f'  "{sample}"')
                    prompt_parts.append(f"Enter name [default: {default_name}]: ")
                    prompt = "\n".join(prompt_parts)
                    if not sys.stdin.isatty():
                        chosen = default_name
                    else:
                        response = input(prompt).strip()
                        chosen = response if response else default_name
                else:
                    chosen = self._next_available_name(existing_names, next_num)

                label_map[spk_id] = chosen
                existing_names.add(chosen)
                next_num += 1
                self._store.add_speaker(chosen, new_speakers[spk_id])

        return label_map

    def _next_available_name(self, existing_names: set, start: int) -> str:
        n = start
        while f"Speaker {n}" in existing_names:
            n += 1
        return f"Speaker {n}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_persistent_diarizer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add persistent_diarizer.py tests/test_persistent_diarizer.py
git commit -m "feat: add PersistentSpeakerDiarizer orchestrator and apply_persistent_labels"
```

---

### Task 5: Update helpers.py for Persistent String Labels

**Files:**
- Modify: `helpers.py`

- [ ] **Step 1: Update get_sentences_speaker_mapping to handle string speaker labels**

In `helpers.py`, the function `get_sentences_speaker_mapping` (line 360) formats speakers as `f"Speaker {spk}"`. When `spk` is already a persistent string label (e.g., `"Alice"`), it should use it directly.

Replace line 366:

```python
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}
```

with:

```python
    snt = {"speaker": spk if isinstance(spk, str) else f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}
```

Replace lines 373-378:

```python
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
```

with:

```python
            snt = {
                "speaker": spk if isinstance(spk, str) else f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
```

- [ ] **Step 2: Verify helpers.py is syntactically correct**

Run: `python -c "from helpers import get_sentences_speaker_mapping; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add helpers.py
git commit -m "feat: update get_sentences_speaker_mapping to handle persistent string labels"
```

---

### Task 6: Integrate Persistent Speaker Matching into diarize.py

**Files:**
- Modify: `diarize.py`

- [ ] **Step 1: Add new CLI arguments**

Add these new argument definitions after line 94 (after the `--diarizer` argument):

```python
parser.add_argument(
    "--speaker-db",
    default="~/.whisper-diarization/speakers.db",
    help="Path to speaker profile database for persistent speaker recognition",
)

parser.add_argument(
    "--match-threshold",
    type=float,
    default=0.6,
    help="Minimum cosine similarity threshold to match a known speaker (default: 0.6)",
)

parser.add_argument(
    "--interactive",
    action="store_true",
    default=False,
    help="Prompt for new speaker names instead of auto-labeling",
)

parser.add_argument(
    "--no-persist",
    action="store_true",
    default=False,
    help="Disable persistent speaker matching (use original behavior)",
)
```

- [ ] **Step 2: Update the diarization section to use DiarizationResult**

Add these imports at the top of `diarize.py`. Add after line 31 (after the existing imports from helpers):

```python
from diarization import DiarizationResult
from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels
from speaker_store import SpeakerEmbeddingStore
```

- [ ] **Step 3: Update the diarization + matching section (lines 187-201)**

Replace lines 187-201:

```python
if args.diarizer == "msdd":
    from diarization import MSDDDiarizer

    diarizer_model = MSDDDiarizer(device=args.device)

elif args.diarizer == "sortformer":
    from diarization import SortformerDiarizer

    diarizer_model = SortformerDiarizer(device=args.device)

speaker_ts = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
del diarizer_model
torch.cuda.empty_cache()

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

with:

```python
if args.diarizer == "msdd":
    from diarization import MSDDDiarizer

    diarizer_model = MSDDDiarizer(device=args.device)

elif args.diarizer == "sortformer":
    from diarization import SortformerDiarizer

    diarizer_model = SortformerDiarizer(device=args.device)

diarization_result = diarizer_model.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))

if isinstance(diarization_result, DiarizationResult):
    speaker_ts = diarization_result.speaker_ts
else:
    speaker_ts = diarization_result

del diarizer_model
torch.cuda.empty_cache()

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if not args.no_persist and isinstance(diarization_result, DiarizationResult) and diarization_result.speaker_embeddings:
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

Note: This re-runs `get_words_speaker_mapping` after applying persistent labels because the original `wsm` was computed with integer speaker IDs. The second call uses the string labels.

- [ ] **Step 4: Verify diarize.py is syntactically correct**

Run: `python -c "import ast; ast.parse(open('diarize.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add diarize.py
git commit -m "feat: integrate persistent speaker matching into diarize.py"
```

---

### Task 7: Integrate Persistent Speaker Matching into diarize_parallel.py

**Files:**
- Modify: `diarize_parallel.py`

- [ ] **Step 1: Add new CLI arguments**

Add these new argument definitions after line 108 (after the `--diarizer` argument):

```python
    parser.add_argument(
        "--speaker-db",
        default="~/.whisper-diarization/speakers.db",
        help="Path to speaker profile database for persistent speaker recognition",
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.6,
        help="Minimum cosine similarity threshold to match a known speaker (default: 0.6)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Prompt for new speaker names instead of auto-labeling",
    )

    parser.add_argument(
        "--no-persist",
        action="store_true",
        default=False,
        help="Disable persistent speaker matching (use original behavior)",
    )
```

- [ ] **Step 2: Update diarize_parallel function to pass DiarizationResult through queue**

Replace the `diarize_parallel` function (lines 36-39):

```python
def diarize_parallel(audio: torch.Tensor, device, queue: mp.Queue):
    model = MSDDDiarizer(device=device)
    result = model.diarize(audio)
    queue.put(result)
```

with:

```python
def diarize_parallel(audio: torch.Tensor, device, queue: mp.Queue):
    model = MSDDDiarizer(device=device)
    result = model.diarize(audio)
    queue.put({"speaker_ts": result.speaker_ts, "speaker_embeddings": result.speaker_embeddings})
```

- [ ] **Step 3: Update the main block to handle DiarizationResult and persistent matching**

Add imports after line 33 (after existing helpers imports):

```python
from diarization import DiarizationResult
from persistent_diarizer import PersistentSpeakerDiarizer, apply_persistent_labels
from speaker_store import SpeakerEmbeddingStore
```

Replace lines 213-219:

```python
    nemo_process.join()
    if results_queue.empty():
        raise RuntimeError("Diarization process did not return any results.")

    speaker_ts = results_queue.get_nowait()

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

with:

```python
    nemo_process.join()
    if results_queue.empty():
        raise RuntimeError("Diarization process did not return any results.")

    diarization_dict = results_queue.get_nowait()
    speaker_ts = diarization_dict["speaker_ts"]
    speaker_embeddings = diarization_dict.get("speaker_embeddings", {})

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if not args.no_persist and speaker_embeddings:
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

- [ ] **Step 4: Verify diarize_parallel.py is syntactically correct**

Run: `python -c "import ast; ast.parse(open('diarize_parallel.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add diarize_parallel.py
git commit -m "feat: integrate persistent speaker matching into diarize_parallel.py"
```

---

### Task 8: speakerctl.py — CLI Profile Management Tool

**Files:**
- Create: `speakerctl.py`

- [ ] **Step 1: Implement speakerctl.py**

Create `speakerctl.py`:

```python
import argparse
import sys

from speaker_store import SpeakerEmbeddingStore


def cmd_list(args):
    store = SpeakerEmbeddingStore(args.db)
    speakers = store.list_speakers()
    if not speakers:
        print("No speakers in database.")
        store.close()
        return
    for s in speakers:
        print(f"  {s['name']}  (samples: {s['sample_count']}, created: {s['created_at']}, updated: {s['updated_at']})")
    store.close()


def cmd_rename(args):
    store = SpeakerEmbeddingStore(args.db)
    store.rename_speaker(args.old_name, args.new_name)
    print(f"Renamed '{args.old_name}' to '{args.new_name}'")
    store.close()


def cmd_delete(args):
    store = SpeakerEmbeddingStore(args.db)
    store.delete_speaker(args.name)
    print(f"Deleted '{args.name}'")
    store.close()


def cmd_show(args):
    store = SpeakerEmbeddingStore(args.db)
    speakers = store.list_speakers()
    for s in speakers:
        if s["name"] == args.name:
            print(f"Name: {s['name']}")
            print(f"Sample count: {s['sample_count']}")
            print(f"Created: {s['created_at']}")
            print(f"Updated: {s['updated_at']}")
            store.close()
            return
    print(f"Speaker '{args.name}' not found.", file=sys.stderr)
    store.close()
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage persistent speaker profiles for whisper-diarization"
    )
    parser.add_argument(
        "--db",
        default="~/.whisper-diarization/speakers.db",
        help="Path to speaker profile database (default: ~/.whisper-diarization/speakers.db)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all stored speakers")

    rename_parser = subparsers.add_parser("rename", help="Rename a speaker")
    rename_parser.add_argument("old_name", help="Current speaker name")
    rename_parser.add_argument("new_name", help="New speaker name")

    delete_parser = subparsers.add_parser("delete", help="Delete a speaker profile")
    delete_parser.add_argument("name", help="Speaker name to delete")

    show_parser = subparsers.add_parser("show", help="Show speaker details")
    show_parser.add_argument("name", help="Speaker name to show")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "rename":
        cmd_rename(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "show":
        cmd_show(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify speakerctl.py works**

Run: `python speakerctl.py --db /tmp/test_speakers.db list`
Expected: `No speakers in database.`

Then run: `python speakerctl.py --db /tmp/test_speakers.db list` (again, verifying no errors)
Then clean up: `rm /tmp/test_speakers.db`

- [ ] **Step 3: Commit**

```bash
git add speakerctl.py
git commit -m "feat: add speakerctl.py CLI tool for speaker profile management"
```

---

### Task 9: Run Full Test Suite and Verify

**Files:**
- None (verification only)

- [ ] **Step 1: Run all unit tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (12 + 5 + 6 = 23 tests)

- [ ] **Step 2: Verify all modified scripts are syntactically valid**

Run: `python -c "import ast; [ast.parse(open(f).read()) for f in ['diarize.py', 'diarize_parallel.py', 'helpers.py', 'speaker_store.py', 'speaker_matcher.py', 'persistent_diarizer.py', 'speakerctl.py', 'diarization/msdd/msdd.py', 'diarization/__init__.py']]; print('All files OK')"`
Expected: `All files OK`

- [ ] **Step 3: Final commit (if any lint/fixup needed)**

Only commit if there are uncommitted changes from verification fixes.
