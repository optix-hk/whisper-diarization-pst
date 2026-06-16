# Server Concurrency Design

## Problem

`server.py` currently has no concurrency safety. Three issues under concurrent load:

1. **GPU model inference is not thread-safe** — CTranslate2, PyTorch, and NeMo cannot run concurrent inference on the same model objects from multiple threads. Two simultaneous requests calling `whisper_pipeline.transcribe()` will corrupt GPU state or crash.
2. **SQLite TOCTOU race** — `update_embedding()` reads the current embedding and sample_count in Python, computes a running average, then writes it back. Two concurrent requests can read the same stale state, each computing an update that overwrites the other.
3. **No task scheduling** — every request immediately starts GPU inference with no queuing or backpressure.

## Three API Modes

| Mode | `skip_diarization` | `speaker_name` | Use case | Needs | Response priority |
|------|-------------------|----------------|----------|-------|--------------------|
| 1. Transcribe only | `true` | — | Service testing, misc | Whisper + alignment | Fast |
| 2. Meeting diarization | `false` | — | Meeting transcript, who said what | Whisper + alignment + diarization + DB match | Medium |
| 3. Known speaker | `false` | `"Alice"` | STT button press, update user embedding | Whisper + alignment (sync), then diarization + DB update (async) | **Fastest** |

Mode 3 responds with the transcript immediately after Whisper+alignment, then runs diarization and DB update in the background. The caller provides `speaker_name` — the response uses it for all segment labels. The background diarization extracts the embedding and updates the DB under the provided name.

## Architecture

### Two Semaphores

Whisper+alignment and NeMo diarization use different GPU models, different VRAM allocations, and have no shared writable state. They can run concurrently. Two semaphores allow this:

```
whisper_semaphore = asyncio.Semaphore(1)   # serializes Whisper + forced alignment
diarizer_semaphore = asyncio.Semaphore(1)  # serializes NeMo diarization
```

Concurrency matrix:

| | Mode 1 (Whisper) | Mode 2 (Whisper+NeMo) | Mode 3 sync (Whisper) | Mode 3 async (NeMo) |
|---|---|---|---|---|
| Mode 1 (Whisper) | Blocked | Blocked | Blocked | **Concurrent** |
| Mode 2 (Whisper) | Blocked | Blocked | Blocked | **Concurrent** |
| Mode 2 (NeMo) | **Concurrent** | Blocked | **Concurrent** | Blocked |
| Mode 3 async (NeMo) | **Concurrent** | Blocked | **Concurrent** | Blocked |

Example: Mode 3's background NeMo diarization does NOT block a Mode 1 Whisper request. They run simultaneously on GPU.

### Request Flow

```
Mode 1 (transcribe only):
  acquire whisper_semaphore
  run Whisper + alignment  ──► transcript ready
  release whisper_semaphore
  return response

Mode 2 (meeting diarization):
  acquire whisper_semaphore
  run Whisper + alignment  ──► transcript ready
  release whisper_semaphore
  acquire diarizer_semaphore
  run NeMo diarization     ──► speaker_ts + embeddings
  release diarizer_semaphore
  run DB match + update (under diarizer lock)
  return response

Mode 3 (known speaker):
  acquire whisper_semaphore
  run Whisper + alignment  ──► transcript ready
  release whisper_semaphore
  return response immediately (speaker_name used for labels)
  [background task:]
    acquire diarizer_semaphore
    run NeMo diarization   ──► embeddings
    release diarizer_semaphore
    update DB with speaker_name + embedding
      (if name exists → update_embedding with running average;
       if name is new → add_speaker)
```

### Async Endpoint with Thread Executor

The `/transcribe` endpoint becomes `async def`. GPU inference calls (synchronous PyTorch/CTranslate2) are wrapped in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking the event loop. The semaphore is acquired in async context (not inside the executor), so queuing happens in the event loop.

### SQLite Safety

**Single shared `SpeakerEmbeddingStore` instance** with a `threading.Lock`:

- The server creates one `SpeakerEmbeddingStore` at startup and holds it for the process lifetime
- A `threading.Lock` wraps all DB read/write operations
- `update_embedding()` and `add_speaker()` use `BEGIN IMMEDIATE` transactions to acquire a write lock at transaction start, preventing the TOCTOU race where two threads read the same stale state before either writes

The lock is held only for the duration of the DB operation (milliseconds), not during GPU inference.

### Background Task Management for Mode 3

Mode 3 uses `asyncio.create_task()` to run diarization + DB update after responding. A `set()` of active background tasks is maintained for:

- **Shutdown**: on SIGINT/SIGTERM, cancel all background tasks and wait (with timeout) for them to finish
- **Monitoring**: the `/health` endpoint reports `pending_db_updates` count

Background task failure handling:
- Log the error (warning level)
- Do NOT affect the already-sent response
- The embedding update is simply lost — next encounter with the same speaker will create a fresh DB entry or update from the existing one

### Shutdown Sequence

1. Stop accepting new requests (uvicorn graceful shutdown)
2. Cancel all pending background tasks
3. Wait up to 10 seconds for in-flight background tasks to complete
4. Close the shared DB connection

## Modified Files

### server.py

- Convert `transcribe()` to `async def`
- Add `whisper_semaphore` and `diarizer_semaphore` (module-level)
- Add `speaker_name: str | None = Form(None)` parameter
- Add `db_lock = threading.Lock()` and shared `SpeakerEmbeddingStore` instance
- Mode logic:
  - `skip_diarization=true` → mode 1: acquire whisper_semaphore, run Whisper+alignment, respond
  - `skip_diarization=false`, `speaker_name=None` → mode 2: acquire both semaphores sequentially, run full pipeline, respond
  - `skip_diarization=false`, `speaker_name="Alice"` → mode 3: acquire whisper_semaphore, run Whisper+alignment, respond, then `asyncio.create_task` for diarization+DB
- GPU calls wrapped in `run_in_executor`
- Track background tasks in a `set()` for shutdown
- Register shutdown handler

### speaker_store.py

- `update_embedding()`: wrap in `BEGIN IMMEDIATE` transaction
- `add_speaker()`: wrap in `BEGIN IMMEDIATE` transaction
- All other write methods (`rename_speaker`, `delete_speaker`, `merge_speakers`): same treatment
- No external API changes — the server-level `threading.Lock` handles cross-request serialization

## API Changes

### New parameter: `speaker_name`

```
POST /transcribe
  audio: file (required)
  skip_diarization: bool (default: false)
  speaker_name: str | None (default: None)   ← NEW
  language: str | None
  batch_size: int
  suppress_numerals: bool
  include_srt: bool
  no_persist: bool
  match_threshold: float
```

When `speaker_name` is provided:
- `skip_diarization` must be `false` (diarization runs in background to get embedding)
- Response segments all have `speaker: <speaker_name>`
- Background diarization + DB update happens after response

### Updated health endpoint

```
GET /health
  {
    "status": "ready",
    "device": "cuda",
    "whisper_loaded": true,
    "alignment_loaded": true,
    "punct_loaded": true,
    "diarizer_loaded": true,
    "speaker_persistence": true,
    "speaker_db": "~/.whisper-diarization/speakers.db",
    "pending_db_updates": 0    ← NEW
  }
```

## Stress Testing Plan

1. **Concurrent mode 1 requests**: send 5 transcribe-only requests simultaneously. All should queue behind whisper_semaphore, complete without error, and return correct transcripts.

2. **Concurrent mode 1 + mode 3**: send a mode 1 request and a mode 3 request simultaneously. Mode 1's Whisper and mode 3's background NeMo should run concurrently (not blocked by each other). Verify mode 3 responds as fast as mode 1.

3. **SQLite race condition**: send 10 mode 3 requests with the same `speaker_name` simultaneously. After all background tasks complete, verify the speaker's `sample_count` equals 10 + previous count, and no updates were lost.

4. **Mode 3 background failure**: mock NeMo to raise an exception. Verify the mode 3 response is already sent with the transcript, and the error is logged.

5. **Shutdown with pending tasks**: send a mode 3 request, then immediately SIGTERM the server. Verify graceful shutdown waits for the background task (up to timeout).
