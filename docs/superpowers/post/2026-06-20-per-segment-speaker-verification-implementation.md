# Per-Segment Speaker Verification Implementation

> **Spec:** `docs/superpowers/specs/2026-06-20-per-segment-speaker-verification-design.md`
> **Plan:** `docs/superpowers/plans/2026-06-20-per-segment-speaker-verification.md`
> **Commits:** 8 (`724832b..ec7c78f`)
> **Tests:** 71 passing (was 58)

## Overview

The persistent speaker diarization system matched speakers at the **cluster level** — one mean embedding per MSDD speaker ID, all segments in a cluster getting the same label. This couldn't correct per-segment errors: if MSDD assigned jenny's segment to foster's cluster, the cluster mean matched "foster" and jenny's segment was mislabeled.

The fix replaces cluster-level matching with **per-segment verification**. A new `SpeakerEmbedder` component loads `titanet_large` via NeMo's public API and extracts one 192-dim embedding per individual segment. Each segment is matched against the DB independently. Unmatched segments fall back to the majority label of their MSDD cluster neighbors. Mode 3 is simplified to use the embedder directly instead of running full MSDD in the background.

## Commits

| # | SHA | Message | What Changed |
|---|-----|---------|--------------|
| 1 | `e9f1510` | feat: add SpeakerEmbedder with titanet_large | New `speaker_embedder.py` + 6 tests + conftest mock |
| 2 | `360ed24` | feat: refactor persistence layer | `match_speakers`, `PersistentSpeakerDiarizer`, `DiarizationResult` refactored; 21 tests rewritten |
| 3 | `c7678a5` | feat: integrate SpeakerEmbedder into diarize.py | CLI script uses embedder + new `resolve_speakers` |
| 4 | `5bdb2bd` | feat: integrate SpeakerEmbedder into diarize_parallel.py | Parallel CLI uses embedder in main process; subprocess dict simplified |
| 5 | `94c5c16` | feat: add SpeakerEmbedder to server startup | `embedder_semaphore`, `models.embedder`, `/health` field |
| 6 | `83b27cb` | feat: update server Mode 2 + simplify Mode 3 | Mode 2 adds embedder step; Mode 3 drops background MSDD |
| 7 | `da54c38` | fix: remove stale speaker_embeddings assignment | Dead variable in `diarize_parallel.py` skip_diarization branch |
| 8 | `ec7c78f` | test: add Mode 2/3 server integration tests | 2 new server tests, None embedder guard, BRANCH_CONTEXT.md update |

## Architecture Change

### Before: Cluster-Level Matching

```
MSDD → speaker_ts [(start, end, spk_id), ...]
     → _extract_embeddings() → {spk_id: cluster_mean_emb}    ← NeMo internal hack
     → match_speakers({spk_id: emb}) → {spk_id: name}        ← one label per cluster
     → apply_persistent_labels(speaker_ts, {spk_id: name})   ← all segments in cluster get same label
```

### After: Per-Segment Matching

```
MSDD → speaker_ts [(start, end, spk_id), ...]
     → SpeakerEmbedder.embed_segments(audio, speaker_ts)     ← standalone titanet_large
       → [emb_0, emb_1, ...] (one per segment, None on failure)
     → match_speakers([emb_0, ...]) → {seg_idx: (name, score)}  ← independent per segment
     → cluster fallback for unmatched segments               ← inherit majority from same-MSDD-cluster
     → apply_persistent_labels(speaker_ts, {seg_idx: name})  ← each segment gets own label
```

### Server Semaphore Change

| Before | After |
|--------|-------|
| `whisper_semaphore` | `whisper_semaphore` |
| `diarizer_semaphore` | `diarizer_semaphore` |
| — | `embedder_semaphore` (NEW) |

Three independent GPU models can run concurrently. Mode 3's background task uses `embedder_semaphore` instead of `diarizer_semaphore`.

## Key Decisions & Deviations from Plan

| Planned | Actual | Reason |
|---------|--------|--------|
| Add one `EncDecSpeakerLabelModel` mock line to conftest | Also enhanced torch mock (copy non-overridden attrs from real torch) | Existing torch mock lacked `randn`, `nn.functional.pad`, `tensor`, `no_grad` — tests couldn't run |
| `test_apply_persistent_labels_partial_map` asserts `(1000, 2000, 1)` (int) | Changed to `(1000, 2000, "1")` (str) | `apply_persistent_labels` returns `str(spk_id)` as fallback; int was a copy-paste leftover from old test |
| Plan specified 6 tasks | Added 2 extra commits (stale cleanup, server tests + docs) | Final review found stale `speaker_embeddings = {}` and missing Mode 2/3 server integration tests required by spec |
| No explicit None embedder guard in Mode 3 | Added early return in `_background_embed` if `models.embedder is None` | Prevents `AttributeError` noise when embedder fails to load at startup |
| Plan didn't mention unused import cleanup | Removed `Tuple`, `Dict`, `numpy` imports left behind by refactoring | Avoid introducing new F401 lint violations |

## Interface Changes

### `DiarizationResult` (diarization/msdd/msdd.py)

```python
# Before
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
    speaker_embeddings: Dict[int, np.ndarray]    # REMOVED

# After
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
```

`MSDDDiarizer._extract_embeddings()` method removed entirely. No more dependency on NeMo's undocumented `emb_sess_test_dict`.

### `match_speakers` (speaker_matcher.py)

```python
# Before
def match_speakers(speaker_embeddings: Dict[int, np.ndarray], store, min_threshold=0.6)
    -> Dict[int, Tuple[Optional[str], float]]          # keyed by spk_id

# After
def match_speakers(segment_embeddings: List[Optional[np.ndarray]], store, min_threshold=0.75)
    -> dict[int, tuple[str | None, float]]              # keyed by segment index
```

Collision resolution removed — multiple segments matching the same stored speaker is expected. `None` embeddings return `(None, 0.0)`.

### `PersistentSpeakerDiarizer.resolve_speakers` (persistent_diarizer.py)

```python
# Before
def resolve_speakers(self, result: DiarizationResult, word_speaker_mapping=None)
    -> Dict[int, str]                                   # {spk_id: name}

# After
def resolve_speakers(self, speaker_ts, segment_embeddings, word_speaker_mapping=None)
    -> dict[int, str]                                   # {seg_idx: name}
```

New logic: per-segment matching → cluster fallback for unmatched → new speaker grouping by MSDD ID → DB update with mean of matched segment embeddings.

### `apply_persistent_labels` (persistent_diarizer.py)

```python
# Before: keyed by speaker ID
def apply_persistent_labels(speaker_ts, label_map: Dict[int, str])  # {spk_id: name}

# After: keyed by segment index
def apply_persistent_labels(speaker_ts, label_map: dict[int, str])  # {seg_idx: name}
```

### New: `SpeakerEmbedder` (speaker_embedder.py)

```python
class SpeakerEmbedder:
    def __init__(self, device: str | torch.device)
    def embed_segment(self, audio: torch.Tensor, start_ms: int, end_ms: int) -> np.ndarray
    def embed_segments(self, audio: torch.Tensor, speaker_ts: list) -> list[np.ndarray | None]
```

Loads `titanet_large` via `EncDecSpeakerLabelModel.from_pretrained()`. Silence-pads segments < 0.5s. Returns `None` for failed segments.

## Test Coverage

| File | Tests | Key Coverage |
|------|-------|--------------|
| `test_speaker_embedder.py` (NEW) | 6 | 192-dim output, silence padding, segment slicing, None on failure, partial failure |
| `test_speaker_matcher.py` | 6 | All known, no match, empty store, multiple match same stored, None embedding, multiple stored |
| `test_persistent_diarizer.py` | 15 | Auto label, known match, embedding update, mixed, **per-segment correction**, **cluster fallback**, **entire cluster unmatched**, **None embedding fallback**, label application, merge |
| `test_server.py` | 18 | Health fields, speaker_name+skip rejection, pending updates, shutdown, background tasks, speaker CRUD, **Mode 3 embedder flow**, **Mode 2 persistent labels** |
| Others | 26 | speaker_store (26 tests, unchanged) |
| **Total** | **71** | Was 58 |

**Key new tests** verifying the core behavioral change:
- `test_resolve_speakers_per_segment_correction` — segment in wrong MSDD cluster gets correct label from its own embedding
- `test_resolve_speakers_cluster_fallback_for_unmatched` — unmatched segment inherits majority label from same-cluster segments
- `test_resolve_speakers_none_embedding_falls_back_to_cluster` — failed embedding extraction treated as unmatched → cluster fallback
- `test_mode3_updates_db_with_embedder` — Mode 3 uses embedder directly, no MSDD call, DB updated
- `test_mode2_persistent_labels_in_response` — Mode 2 embedder → matching → persistent labels in response

## Behavioral Changes

| Aspect | Before | After |
|--------|--------|-------|
| Matching granularity | Per-cluster (one label for all segments in MSDD cluster) | Per-segment (each segment matched independently) |
| Embedding source | NeMo internal `emb_sess_test_dict` (undocumented) | `SpeakerEmbedder` via `EncDecSpeakerLabelModel.from_pretrained()` (public API) |
| Embedding extraction | Cluster mean (one per speaker ID) | Per-segment (one per segment, silence-padded to 0.5s min) |
| Unmatched segments | Entire cluster auto-labeled as new speaker | Cluster fallback: inherit majority label from same-MSDD-cluster matched segments |
| DB update | Single cluster mean per speaker | Mean of all matched segment embeddings per speaker |
| Mode 3 background | Full MSDD diarization to extract one embedding (~10s) | Direct `embed_segment` call (~0.1s) |
| Mode 3 semaphore | `diarizer_semaphore` (contends with Mode 2) | `embedder_semaphore` (independent) |
| Collision resolution | Two clusters matching same stored profile → higher score wins | Removed — multiple segments matching same stored speaker is expected |
| `/health` endpoint | No embedder status | Reports `embedder_loaded` |
| GPU models at startup | Whisper + alignment + punctuation + MSDD | + SpeakerEmbedder (titanet_large, ~300MB) |

## Known Considerations

- `merge_new_speakers` has a pre-existing transitive chain bug: if A↔B and B↔C are above threshold but A↔C is not, C gets a different canonical than A and B. Not introduced by this change; 3-line fix with a `_find_root` helper.
- `SpeakerEmbeddingStore.update_embedding` always adds +1 to `sample_count` regardless of how many segments contributed to the mean. Pre-existing API limitation; a `weight` parameter could address this.
- CLI scripts (`diarize.py`, `diarize_parallel.py`) don't gracefully handle embedder load failure (no try/except). Server handles this gracefully. Fail-fast may be acceptable for CLI.
- End-to-end testing with real GPU + NeMo needed to verify diarization quality improvement on actual audio.
