# Branch Context: diarization-persistence

## Goal

Add persistent speaker recognition to the whisper-diarization pipeline. Instead of discarding speaker embeddings after each run, they are stored in SQLite and cross-referenced in all subsequent diarizations. When a speaker's embedding matches a stored profile (cosine similarity above threshold), the persistent label is used instead of an arbitrary integer. Users can label new speakers interactively or via a CLI management tool.

## Design Spec

`docs/superpowers/specs/2026-06-11-persistent-speaker-diarization-design.md`

## Implementation Plan

`docs/superpowers/plans/2026-06-11-persistent-speaker-diarization.md`

## Architecture

Post-diarization embedding extraction and matching (Approach 1 from brainstorming). After MSDD diarization completes, per-speaker mean embeddings are extracted from NeMo's internal state (`model.clustering_embedding.emb_sess_test_dict`), compared against a persistent SQLite database, and speakers are relabeled with persistent names. The diarization pipeline itself is untouched — this is a pure post-processing layer.

MSDD backend only. Sortformer is an end-to-end model without exposed embeddings; adding an embedding model alongside it would negate its lighter footprint.

## Pipeline Change

```
Stage 1-4: Unchanged (Source Sep → Transcription → Forced Alignment → Diarization)
Stage 4.5 (NEW): Speaker Embedding Matching
  ├── Extract per-speaker mean embeddings from NeMo's internal state
  ├── Compare each mean embedding against stored profiles in SQLite
  ├── Match speakers above minimum cosine similarity threshold
  ├── For unmatched speakers: auto-assign label or prompt interactively
  └── Update stored profiles with matched/new embeddings
Stage 5-7: Unchanged (Word mapping, punctuation, output)
```

## New Components

### speaker_store.py — SpeakerEmbeddingStore

SQLite-backed storage for speaker profiles. Single `speakers` table with columns: `id`, `name` (UNIQUE), `embedding` (BLOB, 192-dim float32 → 768 bytes), `sample_count`, `created_at`, `updated_at`.

Key methods:
- `add_speaker(name, embedding)` — inserts with dimension validation
- `find_best_match(embedding, min_threshold=0.6)` — cosine similarity against all stored profiles, returns `(name, score)` or `(None, score)`
- `update_embedding(name, new_embedding)` — running average: `(old_mean * old_count + new) / (old_count + 1)`
- `rename_speaker(old_name, new_name)`, `delete_speaker(name)`, `list_speakers()`
- Context manager support (`with` statement)
- Embedding dimension validation (first speaker sets the dimension, subsequent must match)

Default DB path: `~/.whisper-diarization/speakers.db` (configurable via `--speaker-db`).

### speaker_matcher.py — match_speakers()

Pure function. Takes a dict of `{speaker_id: embedding}`, loads stored profiles from the store, computes cosine similarity via vectorized matrix multiply, returns `{speaker_id: (matched_name_or_None, score)}`.

Collision resolution: if two new speakers both match the same stored profile, the higher-score one wins and the other is treated as new.

### persistent_diarizer.py — PersistentSpeakerDiarizer

Orchestrator that ties together `SpeakerMatcher` and `SpeakerEmbeddingStore`. `resolve_speakers(result, word_speaker_mapping)`:

1. Calls `match_speakers()` to find known speakers
2. Updates matched speakers' embeddings in the store (running average)
3. For new speakers: auto-assigns `"Speaker N"` (next available integer not in store), or prompts interactively if `--interactive` is set
4. Interactive prompts include the first sentence spoken by the new speaker and segment timestamps
5. Adds new speakers to the store
6. Returns `{speaker_id: persistent_label}` mapping

Also contains `apply_persistent_labels(speaker_ts, label_map)` which converts integer speaker IDs to persistent string labels.

### speakerctl.py — CLI Profile Management

Standalone CLI tool with subcommands: `list`, `rename <old> <new>`, `delete <name>`, `show <name>`.

### server.py — FastAPI Service with Preloaded Models

Long-running HTTP service that loads all models once at startup, eliminating the ~20s per-request import and model-loading overhead of the CLI scripts. Endpoints:

- `GET /health` — returns model loading status and device info
- `POST /transcribe` — accepts audio file upload (WAV, MP3, OGG, etc.) and returns JSON with segments, SRT, language, and processing time

Key features:
- All models (Whisper, alignment, punctuation, diarizer) preloaded into GPU memory at startup
- Non-WAV formats auto-converted to 16kHz mono WAV via ffmpeg server-side
- `include_srt` form parameter to omit SRT from response (reduces payload size)
- `skip_diarization` form parameter to skip diarizer inference
- Persistent speaker matching supported via `no_persist` and `speaker_db` parameters
- Interactive mode not available in server context (no stdin)
- Configured via environment variables: `WHISPER_MODEL` (default: medium.en), `SKIP_DIARIZATION`, `DIARIZER`

Startup command:
```bash
WHISPER_MODEL=tiny.en SKIP_DIARIZATION=1 python server.py
```

## Modified Files

### diarization/msdd/msdd.py

- Added `DiarizationResult` dataclass with `speaker_ts` and `speaker_embeddings` fields
- Added `_extract_embeddings()` method that reads `model.clustering_embedding.emb_sess_test_dict` (NeMo's cluster-average embeddings per speaker, shape `(192, num_speakers)`) before the temp directory is cleaned up
- `diarize()` now returns `DiarizationResult` instead of a plain list

### diarization/__init__.py

- Exports `DiarizationResult` alongside `MSDDDiarizer` and `SortformerDiarizer`

### helpers.py

- `get_sentences_speaker_mapping()`: when `spk` is already a string (persistent label like "Alice"), uses it directly; when it's an integer, formats as `"Speaker {spk}"`

### diarize.py

- Added CLI args: `--speaker-db`, `--match-threshold` (default 0.6), `--interactive`, `--no-persist`, `--skip-diarization`
- `--skip-diarization` skips the NeMo diarizer entirely, creates a single dummy `speaker_ts` spanning the full audio, and disables persistent matching
- After diarization, if `--skip-diarization` is not set, `--no-persist` is not set, and embeddings are available, runs `PersistentSpeakerDiarizer.resolve_speakers()` and reapplies `get_words_speaker_mapping()` with persistent labels
- Backward compatible: `isinstance(diarization_result, DiarizationResult)` check

### diarize_parallel.py

- Same CLI args and integration as `diarize.py`, including `--skip-diarization`
- When `--skip-diarization` is set, the NeMo subprocess is not spawned at all
- `diarize_parallel()` function serializes `DiarizationResult` to a plain dict before putting it in `mp.Queue` (dataclasses may not be picklable across process boundaries)
- Main block reconstructs `DiarizationResult` from dict for persistent matching

## Tests

31 tests across 3 test files, all passing:

- `tests/conftest.py` — mocks NeMo/torch/omegaconf so tests run without GPU dependencies
- `tests/test_speaker_store.py` — 20 tests: CRUD operations, cosine similarity matching, running average, dimension validation, context manager, error handling
- `tests/test_speaker_matcher.py` — 5 tests: all-known, no-match, empty store, collision resolution, multiple stored profiles
- `tests/test_persistent_diarizer.py` — 6 tests: auto-labeling, known speaker matching, embedding updates, mixed known/new, label application, partial label maps

## CLI Usage

```bash
# Run with persistent speaker recognition (default)
python diarize.py -a audio.wav

# Interactive mode — prompts for new speaker names
python diarize.py -a audio.wav --interactive

# Disable persistence (original behavior)
python diarize.py -a audio.wav --no-persist

# Skip diarization entirely (transcription only, no speaker labels)
python diarize.py -a audio.wav --skip-diarization

# Skip diarization + source separation for maximum speed
python diarize.py -a audio.wav --skip-diarization --no-stem --whisper-model turbo

# Custom threshold and DB path
python diarize.py -a audio.wav --match-threshold 0.7 --speaker-db ~/my-speakers.db

# Manage speaker profiles
python speakerctl.py list
python speakerctl.py rename "Speaker 0" "Alice"
python speakerctl.py delete "Speaker 1"
python speakerctl.py show "Alice"
```

## Commits

```
162a12b feat: add --skip-diarization flag and FastAPI server with preloaded models
e691c66 feat: add speakerctl.py CLI tool for speaker profile management
9b41422 feat: integrate persistent speaker matching into diarize_parallel.py
16e0238 feat: integrate persistent speaker matching into diarize.py
7d77603 feat: update get_sentences_speaker_mapping to handle persistent string labels
e4481d7 feat: add PersistentSpeakerDiarizer orchestrator and apply_persistent_labels
82300a1 feat: extract speaker embeddings from MSDD, return DiarizationResult
df78176 feat: add SpeakerMatcher with cosine similarity matching
9812d21 fix: address code quality issues in SpeakerEmbeddingStore
cf4725f feat: add SpeakerEmbeddingStore with SQLite-backed speaker profiles
```

## Known Considerations

- The default cosine similarity threshold of 0.6 may need empirical tuning for real-world audio; it's configurable via `--match-threshold`
- Running average for embeddings means stored embeddings drift from unit norm over time; `find_best_match` re-normalizes before comparison, so matching remains correct
- The embedding extraction relies on NeMo's internal `emb_sess_test_dict` attribute, which is an implementation detail not part of NeMo's public API — it could change between NeMo versions
- `--interactive` mode in `diarize_parallel.py` works but prompts appear after both Whisper and NeMo have finished (due to parallel execution)
- End-to-end testing requires a GPU with NeMo installed; unit tests run without those dependencies thanks to the conftest.py mocks
- `server.py` runs the transcribe endpoint synchronously on a single worker; concurrent requests are not supported without adding a task queue or multiple uvicorn workers
- The server does not support `--interactive` mode (no stdin in HTTP context); new speakers are always auto-labeled
