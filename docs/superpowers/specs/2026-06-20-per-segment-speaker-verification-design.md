# Per-Segment Speaker Verification Design

## Problem

The current persistent speaker diarization system extracts **one cluster-mean embedding per speaker ID** from NeMo's internal `emb_sess_test_dict` after MSDD diarization completes. This embedding is matched against the SQLite speaker database, and all segments in the same MSDD cluster receive the same persistent label.

This architecture cannot correct per-segment diarization errors. If MSDD assigns a segment to the wrong cluster (e.g., jenny's speech segment is clustered with foster's segments), the cluster-mean embedding is dominated by the majority speaker, matches the wrong stored profile, and all segments in that cluster — including the misassigned one — get the wrong label. The carefully built DB profiles are only used to name clusters *en masse*, not to verify individual segments.

**Concrete failure case**: A 27-second conversation between jenny and foster was diarized after building DB profiles for both speakers (3 clips each via Mode 3). MSDD merged adjacent utterances from different speakers into single segments and assigned jenny's segment to foster's cluster. The cluster-mean matching labeled jenny's segment as "foster" because the cluster was dominated by foster's voice.

## Solution

Replace cluster-level embedding matching with **per-segment embedding verification**. After MSDD produces `speaker_ts` (segments with integer speaker IDs), a standalone TitaNet model extracts one embedding per individual segment directly from the audio waveform. Each segment is matched against the DB independently. Segments that don't match above threshold fall back to the majority label of their MSDD cluster neighbors.

This converts the persistence layer from a passive cluster-renamer into an active per-segment corrector. Even when MSDD assigns a segment to the wrong cluster, the per-segment embedding will correctly identify the actual speaker and label the segment accordingly.

## Approach

**Per-segment matching with cluster fallback** (Approach A from brainstorming). A separate `SpeakerEmbedder` component loads `titanet_large` via NeMo's public API (`EncDecSpeakerLabelModel.from_pretrained`). After MSDD diarization, the embedder extracts one 192-dim embedding per segment. Each is matched against stored profiles independently. Matched segments receive their per-segment label (correcting MSDD errors). Unmatched segments inherit the majority label from other segments in the same MSDD cluster. If an entire cluster is unmatched, its segments are grouped as a new speaker.

Alternative approaches considered:
- **Strict per-segment matching** (no cluster fallback) — rejected because short segments with bad embeddings would create spurious new speakers, fragmenting the transcript
- **Two-pass verify-and-correct** (keep cluster-level matching, add per-segment override) — rejected because it retains redundant cluster-level embedding extraction and the fragile `emb_sess_test_dict` dependency, and two matching systems are harder to maintain

## Scope

**Both CLI and server.** All entry points (`diarize.py`, `diarize_parallel.py`, `server.py`) receive the same per-segment verification, following the existing pattern where persistence features are uniformly applied.

**MSDD backend only.** The SpeakerEmbedder uses TitaNet, which is the same embedding model used by MSDD internally. Sortformer is an end-to-end model without exposed embeddings; per-segment verification could work with Sortformer if a SpeakerEmbedder is loaded alongside it, but that is out of scope for this change.

**Mode 3 simplified.** Currently Mode 3 runs full MSDD diarization in the background just to extract one cluster-mean embedding. With the new SpeakerEmbedder, Mode 3 embeds the clip directly — no MSDD inference needed, no `diarizer_semaphore` contention. MSDD remains loaded at startup for Mode 2 requests.

## Architecture

### Updated Pipeline

```
Stage 1-4: Unchanged (Source Sep → Transcription → Forced Alignment → Diarization)
Stage 4.5 (REPLACED): Per-Segment Speaker Verification
  ├── SpeakerEmbedder extracts per-segment embeddings from audio waveform
  ├── Match each segment embedding against DB independently
  ├── Matched segments → per-segment label (corrects MSDD cluster errors)
  ├── Unmatched segments → inherit majority label from same-MSDD-cluster segments
  ├── If entire cluster unmatched → group as new speaker (one label per MSDD ID)
  └── Update DB with mean of matched segment embeddings per speaker
Stage 5-7: Unchanged (Word mapping, punctuation, output)
```

### Key Architectural Changes

- `DiarizationResult` loses `speaker_embeddings` field — cluster-level means no longer extracted
- `MSDDDiarizer._extract_embeddings()` removed — no more dependency on NeMo's undocumented `emb_sess_test_dict`
- New `SpeakerEmbedder` component using NeMo's public API
- `match_speakers` and `PersistentSpeakerDiarizer` refactored from cluster-keyed to segment-keyed
- `apply_persistent_labels` keyed by segment index instead of speaker ID
- Mode 3 simplified: uses `SpeakerEmbedder` directly, no background MSDD

## Components

### New: `SpeakerEmbedder` (`speaker_embedder.py`)

Loads `titanet_large` via NeMo's public API and extracts embeddings from audio segments.

```python
class SpeakerEmbedder:
    def __init__(self, device: str | torch.device):
        self.model = EncDecSpeakerLabelModel.from_pretrained('titanet_large')
        self.model.to(device).eval()
        self.device = device
        self.min_duration_samples = int(0.5 * 16000)  # 0.5s at 16kHz

    def embed_segment(self, audio: torch.Tensor, start_ms: int, end_ms: int) -> np.ndarray:
        """Extract 192-dim embedding for one segment. Silence-pads to 0.5s minimum."""
        start_sample = int(start_ms * 16)
        end_sample = int(end_ms * 16)
        segment = audio[start_sample:end_sample]
        if segment.shape[0] < self.min_duration_samples:
            pad = self.min_duration_samples - segment.shape[0]
            segment = torch.nn.functional.pad(segment, (0, pad))
        with torch.no_grad():
            _, emb = self.model.forward(segment.unsqueeze(0).to(self.device))
        return emb.squeeze(0).cpu().numpy()

    def embed_segments(self, audio: torch.Tensor, speaker_ts: list) -> list[np.ndarray | None]:
        """Extract embeddings for all segments. Returns list aligned with speaker_ts.
        None entries indicate per-segment extraction failures (treated as unmatched)."""
        results = []
        for start, end, _ in speaker_ts:
            try:
                results.append(self.embed_segment(audio, start, end))
            except Exception:
                results.append(None)
        return results
```

**Behaviors:**
- 192-dim float32 output, same embedding space as TitaNet embeddings already stored in the DB
- Segments shorter than 0.5s are silence-padded (zero-padded) to 0.5s — always, no tiered fallback
- `embed_segments` returns a list indexed by segment position in `speaker_ts`; `None` entries for failed segments are treated as unmatched by downstream matching
- Uses NeMo's public `EncDecSpeakerLabelModel.from_pretrained()` API — no internal attribute hacking

### Modified: `DiarizationResult` and `MSDDDiarizer` (`diarization/msdd/msdd.py`)

```python
@dataclass
class DiarizationResult:
    speaker_ts: List[Tuple[int, int, int]]
    # speaker_embeddings field REMOVED
```

- `_extract_embeddings()` method removed entirely
- `diarize()` returns `DiarizationResult(speaker_ts=labels)` — no embedding extraction
- Removes dependency on `model.clustering_embedding.emb_sess_test_dict`

### Modified: `match_speakers` (`speaker_matcher.py`)

Input changes from `{spk_id: embedding}` to `List[np.ndarray]` (per-segment). Output changes from `{spk_id: (name, score)}` to `{seg_idx: (name, score)}`.

```python
def match_speakers(
    segment_embeddings: list[np.ndarray | None],
    store: SpeakerEmbeddingStore,
    min_threshold: float = 0.75,
) -> dict[int, tuple[str | None, float]]:
    """Match each segment embedding independently against stored profiles.
    None entries (failed extraction) are treated as unmatched with score 0.0.
    No collision resolution needed — multiple segments matching the same
    stored speaker is the expected normal case."""
```

- Collision resolution logic removed — multiple segments matching "jenny" is normal, not a conflict
- Each segment independently gets its best match above threshold
- `None` embeddings (extraction failures) return `(None, 0.0)` — handled by cluster fallback

### Modified: `PersistentSpeakerDiarizer` (`persistent_diarizer.py`)

`resolve_speakers` changes from taking `DiarizationResult` to taking `speaker_ts` + `segment_embeddings`:

```python
def resolve_speakers(
    self,
    speaker_ts: list[tuple[int, int, int]],
    segment_embeddings: list[np.ndarray | None],
    word_speaker_mapping: list[dict] | None = None,
) -> dict[int, str]:  # {seg_idx: label}
```

**Logic:**
1. `match_speakers(segment_embeddings, store, threshold)` → `{seg_idx: (name, score)}`
2. Matched segments → per-segment label (corrects MSDD errors)
3. Unmatched segments → cluster fallback (see below)
4. New speaker assignment for fully unmatched clusters
5. DB update: mean of matched segment embeddings per speaker
6. Returns `{seg_idx: persistent_label}`

### Modified: `apply_persistent_labels` (`persistent_diarizer.py`)

Keyed by segment index instead of speaker ID:

```python
def apply_persistent_labels(
    speaker_ts: list[tuple[int, int, int]],
    label_map: dict[int, str],  # {segment_index: label}
) -> list[tuple[int, int, str]]:
    return [
        (start, end, label_map.get(i, str(spk_id)))
        for i, (start, end, spk_id) in enumerate(speaker_ts)
    ]
```

## Data Flow

### Mode 2 (Meeting Diarization)

```
1. Whisper + alignment → word_timestamps
2. MSDD diarization → speaker_ts [(start, end, spk_id), ...]
3. SpeakerEmbedder.embed_segments(audio, speaker_ts) → [emb_0, emb_1, ...]
4. PersistentSpeakerDiarizer.resolve_speakers(speaker_ts, segment_embeddings, store)
   → {seg_idx: label}
5. apply_persistent_labels(speaker_ts, label_map) → relabeled speaker_ts
6. get_words_speaker_mapping(word_timestamps, relabeled_speaker_ts)
7. _build_segments(wsm, relabeled_speaker_ts, ...)
```

Steps 2-3 use GPU models (MSDD then SpeakerEmbedder). Step 4 uses the DB under `db_lock`.

### Mode 3 (Known Speaker — Simplified)

```
1. Whisper + alignment → word_timestamps
2. Respond immediately: speaker_name as all segment labels
3. Background task:
   a. SpeakerEmbedder.embed_segment(audio, first_word_start, last_word_end) → embedding
   b. DB update: store.update_embedding(speaker_name, emb) or store.add_speaker(speaker_name, emb)
```

No MSDD inference, no `diarizer_semaphore`. The background task only needs the SpeakerEmbedder (fast forward pass on one clip) and a DB write. This is ~10x faster than the current Mode 3 background which runs full MSDD diarization.

### Cluster Fallback Logic

For unmatched segments (per-segment match score < threshold), inherit from MSDD cluster neighbors:

```python
def _resolve_unmatched_segments(
    speaker_ts: list[tuple[int, int, int]],
    matches: dict[int, tuple[str | None, float]],
) -> dict[int, str | None]:
    """For each unmatched segment, inherit majority label from same-MSDD-cluster segments."""
    from collections import Counter
    resolved = {}
    for seg_idx, (name, _) in matches.items():
        if name is not None:
            continue  # already matched

        spk_id = speaker_ts[seg_idx][2]
        # Collect matched labels from segments with the same MSDD speaker ID
        cluster_labels = [
            matches[i][0] for i, (_, _, sid) in enumerate(speaker_ts)
            if sid == spk_id and matches.get(i, (None,))[0] is not None
        ]
        if cluster_labels:
            # Inherit the most common matched label in this cluster
            resolved[seg_idx] = Counter(cluster_labels).most_common(1)[0][0]
        else:
            # Entire cluster unmatched — will be grouped as new speaker
            resolved[seg_idx] = None
    return resolved
```

**Example with jenny/foster**: If MSDD puts jenny-2 in cluster 1 (foster's cluster), but jenny-2's per-segment embedding matches "jenny" above threshold, it gets labeled "jenny" directly — regardless of its MSDD cluster assignment. If a short segment in cluster 1 has a bad embedding and doesn't match, it inherits "foster" (the majority of cluster 1's matched segments). If an entire cluster has no matches, all its segments are grouped as one new speaker.

### New Speaker Assignment

For clusters where all segments are unmatched:
- Group by MSDD speaker ID
- Each group gets one new label: `"Speaker N"` (next available integer) or interactive prompt (CLI only)
- All segments in the group receive the same label
- Mean of the group's segment embeddings is stored in DB as the new speaker's profile

### DB Update Strategy

After all segments are labeled:
- **Matched speakers** (e.g., "jenny"): collect all segment embeddings labeled "jenny", compute mean, `store.update_embedding("jenny", mean)` (running average)
- **New speakers** (e.g., "Speaker 2"): collect group's segment embeddings, compute mean, `store.add_speaker("Speaker 2", mean)`

One DB update per speaker per run, using the mean of their per-segment embeddings. Consistent with the current running-average approach but using per-segment embeddings instead of cluster means.

## Server Integration

### Startup

`SpeakerEmbedder` loaded alongside other models in `Models.__init__` and the startup handler:

```python
class Models:
    def __init__(self):
        # ... existing fields ...
        self.embedder: SpeakerEmbedder | None = None  # NEW
```

At startup (after Whisper, alignment, punctuation, diarizer are loaded):
```python
models.embedder = SpeakerEmbedder(models.device)
```

### Semaphore Strategy

Three independent GPU models, each with its own semaphore:

| Semaphore | Protects | Concurrency |
|-----------|----------|-------------|
| `whisper_semaphore` (existing) | Whisper + alignment | Serializes within Whisper |
| `diarizer_semaphore` (existing) | MSDD diarization | Serializes within MSDD |
| `embedder_semaphore` (NEW) | SpeakerEmbedder | Serializes within embedder |

All three can run concurrently with each other (different models on GPU). This allows Mode 3's background embedding to run while a Mode 2 MSDD diarization is in progress.

### Server Mode 2 Flow (with semaphores)

```python
async with diarizer_semaphore:
    diarization_result = await loop.run_in_executor(None, _run_diarization, audio_waveform)
speaker_ts = diarization_result.speaker_ts

async with embedder_semaphore:
    segment_embeddings = await loop.run_in_executor(
        None, models.embedder.embed_segments, audio_waveform, speaker_ts
    )

with db_lock:
    if models.shared_store is not None:
        pd = PersistentSpeakerDiarizer(store=models.shared_store, min_threshold=match_threshold)
        label_map = pd.resolve_speakers(speaker_ts, segment_embeddings, wsm)
        speaker_ts = apply_persistent_labels(speaker_ts, label_map)
        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
```

### Server Mode 3 Flow (simplified)

```python
# Respond immediately (unchanged)
result = _build_segments(wsm, speaker_ts, ..., override_speaker=speaker_name)

# Background task — NO MSDD, just embedder + DB
async def _background_embed():
    async with embedder_semaphore:
        embedding = await loop.run_in_executor(
            None, models.embedder.embed_segment,
            audio_waveform, first_word_start, last_word_end
        )
    with db_lock:
        if models.shared_store is not None:
            existing = models.shared_store._get_profile_by_name(speaker_name)
            if existing is not None:
                models.shared_store.update_embedding(speaker_name, embedding)
            else:
                models.shared_store.add_speaker(speaker_name, embedding)
```

## CLI Integration

### `diarize.py`

After diarization completes:
1. Instantiate `SpeakerEmbedder(device)`
2. `embed_segments(audio, speaker_ts)` → per-segment embeddings
3. `PersistentSpeakerDiarizer.resolve_speakers(speaker_ts, segment_embeddings, store)` → label map
4. `apply_persistent_labels` → relabel
5. Re-run `get_words_speaker_mapping` with relabeled `speaker_ts`
6. Clean up embedder model (`del embedder; torch.cuda.empty_cache()`)

### `diarize_parallel.py`

The SpeakerEmbedder runs in the **main process** (after the MSDD subprocess returns `speaker_ts`). The embedder is not part of the subprocess — it loads in the main process after MSDD completes. The `DiarizationResult` dict from the subprocess no longer contains `speaker_embeddings`; it only contains `speaker_ts`.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| SpeakerEmbedder fails to load at startup | Server starts without embedder. Per-segment verification disabled. `resolve_speakers` skipped, speakers retain MSDD integer IDs. Logged as warning. `/health` reports `embedder_loaded: false`. |
| Embedding extraction fails for one segment | That segment gets `None` match → cluster fallback. Other segments unaffected. Logged as warning. |
| Embedding extraction fails for all segments | All segments unmatched → grouped by MSDD speaker ID → auto-labeled "Speaker N". DB not updated. |
| DB is empty | All segments unmatched → grouped by MSDD speaker ID → auto-labeled. New speakers added to DB. |
| `--skip-diarization` | No `speaker_ts` segments to embed. Persistence disabled (same as current). |
| `--no-persist` | SpeakerEmbedder not loaded. Speakers retain MSDD integer IDs (same as current). |

## Testing

### Modified Tests

- `tests/conftest.py` — Remove `speaker_embeddings` from mock `DiarizationResult`. Add mock for `EncDecSpeakerLabelModel`.
- `tests/test_speaker_matcher.py` — Input from `{spk_id: emb}` → `List[emb]`. Output from `{spk_id: ...}` → `{seg_idx: ...}`. Remove collision resolution tests (no longer needed — multiple segments matching the same stored speaker is expected).
- `tests/test_persistent_diarizer.py` — Use `speaker_ts` + `segment_embeddings` instead of `DiarizationResult`. Add cluster fallback tests. Update existing tests for new `resolve_speakers` signature.
- `tests/test_server.py` — Update Mode 2 (embedder step) and Mode 3 (no background MSDD) flow tests. Update `/health` to check `embedder_loaded`.

### New Tests

- `tests/test_speaker_embedder.py` — Silence padding for short segments, segment slicing correctness, output dimension (192-dim float32), list alignment with `speaker_ts`.
- Cluster fallback tests — unmatched segment inherits from same-cluster majority; entire cluster unmatched → grouped as new speaker; mixed matched/unmatched in same cluster; all segments match (no fallback needed).
- Mode 3 simplified flow tests — no MSDD call in background, embedder called directly, DB updated with clip embedding, `embedder_semaphore` used.

## Files Changed

| File | Change |
|------|--------|
| `speaker_embedder.py` (NEW) | `SpeakerEmbedder` class with `embed_segment` and `embed_segments` |
| `diarization/msdd/msdd.py` | Remove `speaker_embeddings` from `DiarizationResult`, remove `_extract_embeddings()` |
| `diarization/__init__.py` | No change (still exports `DiarizationResult`) |
| `speaker_matcher.py` | Refactor `match_speakers` for per-segment input/output, remove collision resolution |
| `persistent_diarizer.py` | Refactor `resolve_speakers` for per-segment embeddings, add cluster fallback, update `apply_persistent_labels` |
| `server.py` | Add `embedder_semaphore`, load `SpeakerEmbedder` at startup, update Mode 2 flow, simplify Mode 3 flow, update `/health` |
| `diarize.py` | Instantiate `SpeakerEmbedder`, call `embed_segments`, use new `resolve_speakers` signature |
| `diarize_parallel.py` | Same as `diarize.py`; embedder runs in main process after subprocess returns |
| `tests/conftest.py` | Update mock `DiarizationResult`, add `EncDecSpeakerLabelModel` mock |
| `tests/test_speaker_matcher.py` | Update for per-segment input/output |
| `tests/test_persistent_diarizer.py` | Update for new signatures, add cluster fallback tests |
| `tests/test_server.py` | Update Mode 2/Mode 3 flow tests, `/health` |
| `tests/test_speaker_embedder.py` (NEW) | New test file for `SpeakerEmbedder` |

## Known Considerations

- The `SpeakerEmbedder` loads `titanet_large` via NeMo's public API, adding ~300MB GPU memory and ~5s startup time. This is the same model MSDD uses internally, so the embedding space is identical to what's already stored in the DB — no migration needed.
- Per-segment embedding extraction adds GPU inference time proportional to the number of segments. For typical meetings (50-200 segments), this is ~0.5-2s total. The embedder runs inside `embedder_semaphore`, concurrent with Whisper and MSDD.
- The `_extract_embeddings()` removal breaks backward compatibility with any code that accesses `DiarizationResult.speaker_embeddings`. The only consumers are the persistence layer (being refactored) and tests (being updated).
- CLI scripts (`diarize.py`, `diarize_parallel.py`) load the embedder transiently (instantiate, use, delete). This adds ~5s model loading time to CLI runs. The server avoids this by preloading at startup.
- The cluster fallback assumes MSDD's clustering is *partially* correct — if most segments in a cluster are correctly assigned, the minority of bad-embedding segments inherit the correct label. If MSDD's clustering is completely wrong for a cluster, the fallback may propagate the wrong label to unmatched segments. This is an acceptable tradeoff: the per-segment matching still corrects matched segments, and unmatched segments are edge cases.
