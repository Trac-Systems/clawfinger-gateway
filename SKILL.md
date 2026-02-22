# Local Voice Gateway — Installation & Operations Skill

> **Platform**: macOS and Linux only. Not compatible with Windows. Primarily tested on macOS (Apple Silicon).

## What This Is

A local voice gateway that handles phone calls for the Clawfinger Android app. It runs the full ASR → LLM → TTS pipeline on Apple Silicon using MLX models. The phone connects via ADB reverse port forwarding — no ngrok, no remote servers.

## Security: localhost-only

The gateway MUST only bind to `127.0.0.1`. Never use `0.0.0.0` or any network-facing address. This applies to everything: the gateway API, control center UI, mlx_audio sidecar, and any future services. No exceptions. The phone reaches the gateway via ADB reverse port forwarding — there is no reason to expose anything to any network, not even the local LAN.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+ (tested on 3.13.5)
- ADB (Android Debug Bridge) for phone connection
- ~4GB disk for models, ~500MB for venv

## Complete Installation

### Step 1: Create venv and install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Install Kokoro TTS dependencies

`mlx-audio` does not declare all Kokoro TTS runtime deps. These must be installed manually:

```bash
pip install 'misaki==0.7.0' num2words spacy phonemizer mecab-python3 unidic-lite webrtcvad 'setuptools<81'
```

**Critical version pins:**
| Package | Pin | Reason |
|---------|-----|--------|
| `misaki` | `==0.7.0` | 0.7.4 crashes with `NoneType` phonemes bug; 0.6.x missing `MToken` class needed by mlx_audio 0.3.x |
| `setuptools` | `<81` | `webrtcvad` imports `pkg_resources` which was removed in setuptools 82+ |

### Step 3: Download models

All models are cached in `.models/` via `HF_HOME`. Download them before first run to avoid cold-start delays:

```bash
export HF_HOME="$(pwd)/.models"
source .venv/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
for repo in [
    'mlx-community/parakeet-tdt-0.6b-v2',
    'mlx-community/Kokoro-82M-bf16',
    'mlx-community/Qwen2.5-1.5B-Instruct-4bit',
]:
    print(f'Downloading {repo}...')
    snapshot_download(repo)
    print(f'  Done: {repo}')
"
```

**Models:**
| Model | Purpose | Size | Notes |
|-------|---------|------|-------|
| `mlx-community/parakeet-tdt-0.6b-v2` | ASR (speech-to-text) | 2.3 GB | Do NOT use `whisper-small-mlx` — broken processor with mlx_audio 0.3.x |
| `mlx-community/Kokoro-82M-bf16` | TTS (text-to-speech) | 375 MB | Voice: `af_heart`, speed: 1.2 |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | LLM (conversation) | 852 MB | Loaded in-process via mlx-lm |

### Step 4: Configure

Edit `config.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8996,
  "bearer_token": "localdev",
  "mlx_audio_base": "http://127.0.0.1:8765",
  "stt_model": "mlx-community/parakeet-tdt-0.6b-v2",
  "stt_language": "en",
  "tts_model": "mlx-community/Kokoro-82M-bf16",
  "tts_voice": "af_heart",
  "tts_speed": 1.2,
  "llm_backend": "mlx_local",
  "llm_local_model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "llm_remote_base_url": "",
  "llm_remote_api_key": "",
  "llm_remote_model": "",
  "llm_max_tokens": 400,
  "llm_temperature": 0.2,
  "llm_system_prompt": "You are a concise, friendly real-time voice assistant. Respond in plain English with 2-3 short sentences and no markdown.",
  "max_history_turns": 8
}
```

**Key settings:**
- `bearer_token`: Must be non-empty. Phone profile must have the same value or profile parsing fails silently.
- `llm_backend`: `"mlx_local"` for in-process MLX, or `"openai_remote"` to use an external OpenAI-compatible endpoint.
- `tts_speed`: 1.2 is natural cadence for Kokoro. Lower = slower speech.
- All settings can be overridden via env vars: `GATEWAY_PORT=9000`, `GATEWAY_BEARER_TOKEN=xyz`, etc.

## Running

### Start

```bash
bin/start.sh
```

This starts two processes:
1. **mlx_audio server** on `127.0.0.1:8765` — handles ASR and TTS model inference
2. **Gateway** (FastAPI/uvicorn) on `127.0.0.1:8996` — phone API, control center UI, agent interface

The script waits for mlx_audio to be healthy before starting the gateway. LLM is preloaded at gateway startup.

### Stop

```bash
bin/stop.sh
```

Or kill individually:
```bash
kill $(cat tmp/gateway.pid)
kill $(cat tmp/mlx_audio.pid)
```

### Manual start (for debugging)

```bash
source .venv/bin/activate
export HF_HOME="$(pwd)/.models"

# Terminal 1: mlx_audio
python -m mlx_audio.server --host 127.0.0.1 --port 8765

# Terminal 2: gateway
python -m uvicorn app:app --host 127.0.0.1 --port 8996 --log-level info
```

## Pre-warming Models

First request to each model has extra latency (model loading into GPU memory). Pre-warm after startup:

```bash
# Warm TTS (Kokoro)
curl -s -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Kokoro-82M-bf16","input":"Warm up.","voice":"af_heart","speed":1.2,"response_format":"wav"}' \
  -o /dev/null

# Warm ASR (Parakeet) — needs a WAV file
python3 -c "
import wave, tempfile
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir='tmp') as f:
    w = wave.open(f, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b'\x00\x00' * 16000); w.close(); print(f.name)
" | xargs -I{} curl -s -X POST http://127.0.0.1:8765/v1/audio/transcriptions \
  -F "file=@{}" -F "model=mlx-community/parakeet-tdt-0.6b-v2" -F "language=en" -o /dev/null
```

The LLM is automatically pre-warmed at gateway startup when `llm_backend` is `mlx_local`.

## Phone Connection

### 1. Set up ADB reverse forwarding

```bash
adb reverse tcp:8996 tcp:8996
```

This maps `127.0.0.1:8996` on the phone to `127.0.0.1:8996` on this machine.

### 2. Push the phone profile

The PhoneBridge app reads its profile from the **internal app data directory**, NOT from `/sdcard/`:

```bash
# Push to staging location
adb push profiles/pixel10pro-blazer-profile-v1.json /data/local/tmp/profile.json

# Copy into app's internal data via run-as
adb shell "run-as com.tracsystems.phonebridge cp /data/local/tmp/profile.json files/profiles/profile.json"
```

**Profile gateway section must be:**
```json
"gateway": {
  "base_url": "http://127.0.0.1:8996",
  "bearer": "localdev"
}
```

The `bearer` field MUST be non-empty. If blank, the app's profile parser returns null (line 5410 of `GatewayCallAssistantService.kt`) and the call assistant won't start. The value must match `bearer_token` in `config.json`.

### 3. Restart the phone app

```bash
adb shell am force-stop com.tracsystems.phonebridge
adb shell am start -n com.tracsystems.phonebridge/.MainActivity
```

### 4. Verify

```bash
# Verify profile on phone
adb shell "run-as com.tracsystems.phonebridge cat files/profiles/profile.json" | python3 -c "
import sys, json; print(json.dumps(json.load(sys.stdin)['gateway'], indent=2))"

# Verify gateway health
curl -s -H "Authorization: Bearer localdev" http://127.0.0.1:8996/health | python3 -m json.tool

# Verify ADB reverse
adb reverse --list
```

## API Endpoints

### Phone API (bearer auth required)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check — mlx_audio + LLM status |
| `POST` | `/api/asr` | ASR only — multipart `audio` file → `{"transcript": "..."}` |
| `POST` | `/api/turn` | Full voice turn — see below |
| `POST` | `/api/session/new` | Create session → `{"session_id": "..."}` |
| `POST` | `/api/session/reset` | Reset session history |

### `/api/turn` — Main voice turn

**Request**: `multipart/form-data` with `Authorization: Bearer <token>` header

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | yes | WAV audio from caller |
| `session_id` | string | no | Session identifier (auto-generated if empty) |
| `reset_session` | string | no | `"true"` to clear session history |
| `transcript_hint` | string | no | Local ASR result — used as fallback if server ASR is empty |
| `skip_asr` | string | no | `"true"` to skip server ASR, use `transcript_hint` directly |
| `forced_reply` | string | no | Skip ASR+LLM, TTS this text directly (greeting, max duration goodbye) |

**Response**:
```json
{
  "ok": true,
  "session_id": "abc123",
  "transcript": "what caller said",
  "reply": "assistant response",
  "audio_base64": "<base64 WAV>",
  "metrics": {
    "asr_ms": 450.2,
    "llm_ms": 355.8,
    "tts_ms": 630.4,
    "total_ms": 1436.4,
    "llm_model": "local/mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  }
}
```

The phone reads `audio_base64` first, falls back to `audio_wav_base64`, then `audioBase64`.

### Control Center (no auth)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Control center SPA UI |
| `GET` | `/api/status` | System status (uptime, call count, model health, config) |
| `GET` | `/api/sessions` | List persisted sessions |
| `GET` | `/api/sessions/{id}` | Full session detail with turn-by-turn transcript |
| `POST` | `/api/config` | Hot-reload config from disk |
| `POST` | `/api/call/inject` | Inject TTS message — `{"text": "...", "session_id": "..."}` |
| `POST` | `/api/call/dial` | Dial outbound call — `{"number": "+49..."}` |
| `WS` | `/ws/events` | Real-time event stream for UI |

### Instructions

Three-layer instruction system controlling the LLM system prompt:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/instructions` | Returns `{base, sessions: {sid: text}}` |
| `POST` | `/api/instructions` | Set base (global) instruction — `{"text": "..."}` |
| `POST` | `/api/instructions/{sid}` | Set per-session instruction override |
| `POST` | `/api/instructions/{sid}/turn` | Set one-shot per-turn supplement (consumed after next turn) |
| `DELETE` | `/api/instructions/{sid}` | Clear per-session instruction |

**Prompt assembly** (`instruction_store.build_system_prompt`):
1. Session instruction wins over base; base falls back to `config.json`'s `llm_system_prompt` if unset
2. Per-turn supplement is appended with `\n\n` separator and consumed after one use

### Dial

`POST /api/call/dial` sends an ADB broadcast to the phone's `CallCommandReceiver`:
```bash
adb shell am broadcast \
  -a com.tracsystems.phonebridge.CALL_COMMAND \
  -n com.tracsystems.phonebridge/.CallCommandReceiver \
  --es type dial --es number "+49..."
```
The explicit component flag (`-n`) is required — Android 14+ silently drops implicit broadcasts to exported receivers.

### Agent Interface

| Method | Path | Purpose |
|--------|------|---------|
| `WS` | `/api/agent/ws` | Agent WebSocket — receives all events, full bidirectional control |
| `POST` | `/api/agent/inject` | REST inject — `{"text": "...", "session_id": "..."}` |
| `GET` | `/api/agent/sessions` | List active session IDs |

**Agent WebSocket protocol** — agent sends JSON messages:

| Message | Fields | Description |
|---------|--------|-------------|
| `takeover` | `session_id` | Take over LLM for a session (gateway forwards transcripts, agent replies) |
| `release` | `session_id` | Release LLM control back to local LLM |
| `inject` | `text`, `session_id` | Inject TTS message into call |
| `set_instructions` | `instructions`, `session_id`, `scope` | Set instructions; scope: `global`, `session`, or `turn` |
| `dial` | `number` | Dial outbound call |
| `ping` | — | Heartbeat |

**Ack messages**: `takeover.ack`, `release.ack`, `set_instructions.ack`, `dial.ack`, `pong`.

During takeover, gateway sends `{"type": "turn.request", "session_id": "...", "transcript": "..."}` and expects `{"reply": "..."}` within 30s.

Agent receives all event bus events: `turn.started`, `turn.transcript`, `turn.reply`, `turn.complete`, `turn.error`, `agent.connected`, `agent.disconnected`, `agent.inject`, `agent.takeover`, `agent.release`, `instructions.updated`, `call.dial`, `status.update`.

## File Structure

```
gateway/
├── app.py                  # FastAPI application — all endpoints
├── config.py               # config.json loader + env var overrides
├── voice_pipeline.py       # ASR/TTS via mlx_audio HTTP
├── llm_backend.py          # LLM abstraction (local MLX / remote OpenAI)
├── session_store.py        # In-memory session history + JSON persistence
├── event_bus.py            # Async pub/sub → UI + agent WebSockets
├── agent_interface.py      # Agent WS protocol (takeover/inject/release)
├── instruction_store.py    # In-memory instruction layers (base/session/turn)
├── static/index.html       # Control center SPA (vanilla HTML/JS/CSS)
├── config.json             # Runtime configuration
├── requirements.txt        # Base Python deps
├── bin/start.sh            # Start mlx_audio + gateway
├── bin/stop.sh             # Stop both
├── .gitignore              # Excludes .venv/, .models/, sessions/, tmp/
├── .venv/                  # Python virtual environment (not tracked)
├── .models/                # HuggingFace model cache (not tracked)
├── sessions/               # Persisted session JSON logs (not tracked)
└── tmp/                    # Temp audio files + PID files (not tracked)
```

## Verification Checklist

After installation, verify each component:

```bash
source .venv/bin/activate
export HF_HOME="$(pwd)/.models"

# 1. mlx_audio server responds
curl -sf http://127.0.0.1:8765/v1/models && echo "OK: mlx_audio"

# 2. ASR works
curl -s -X POST http://127.0.0.1:8765/v1/audio/transcriptions \
  -F "file=@tmp/tmpipgkgp6y.wav" \
  -F "model=mlx-community/parakeet-tdt-0.6b-v2" \
  -F "language=en" && echo " OK: ASR"

# 3. TTS works
curl -s -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Kokoro-82M-bf16","input":"Hello.","voice":"af_heart","speed":1.2,"response_format":"wav"}' \
  -o /dev/null -w "OK: TTS (%{size_download} bytes)\n"

# 4. Gateway health
curl -s -H "Authorization: Bearer localdev" http://127.0.0.1:8996/health | python3 -m json.tool

# 5. Full turn (forced_reply)
curl -s -X POST http://127.0.0.1:8996/api/turn \
  -H "Authorization: Bearer localdev" \
  -F "audio=@tmp/tmpipgkgp6y.wav" \
  -F "forced_reply=Hello, this is a test." | python3 -c "
import sys, json; d=json.load(sys.stdin); d.pop('audio_base64',None); print(json.dumps(d, indent=2))"

# 6. ADB reverse active
adb reverse --list | grep 8996

# 7. Phone profile correct
adb shell "run-as com.tracsystems.phonebridge cat files/profiles/profile.json" | python3 -c "
import sys, json; g=json.load(sys.stdin)['gateway']; print(g); assert g['base_url']=='http://127.0.0.1:8996'; assert g['bearer']"
```

## Troubleshooting

### TTS returns 500
Check `tmp/mlx_audio.log`. Common causes:
- **`No module named 'misaki'`**: `pip install 'misaki==0.7.0'`
- **`No module named 'num2words'`**: `pip install num2words`
- **`No module named 'spacy'`**: `pip install spacy`
- **`No module named 'phonemizer'`**: `pip install phonemizer`
- **`No module named 'pkg_resources'`**: `pip install 'setuptools<81'`
- **`NoneType phonemes` crash**: Wrong misaki version — must be `==0.7.0`

### ASR returns 500 / "peer closed connection"
- If using `whisper-small-mlx`: switch to `parakeet-tdt-0.6b-v2` (Whisper processor bug with mlx_audio 0.3.x)
- Check `tmp/mlx_audio.log` for the actual error

### Phone doesn't hit gateway
1. Check `adb reverse --list` — must show `tcp:8996 tcp:8996`
2. Check profile at correct path: `adb shell "run-as com.tracsystems.phonebridge cat files/profiles/profile.json"`
3. Profile `bearer` must be non-empty
4. Force-stop and restart the app after profile changes
5. Check `tmp/gateway.log` for incoming requests — if only `/api/status` polling, phone isn't connecting

### First turn slow / times out
Pre-warm models after startup. First TTS call loads Kokoro (~3s), first ASR call loads Parakeet (~2s). LLM is pre-warmed automatically.

### Config changes not taking effect
Call `POST /api/config` to hot-reload from `config.json`, or restart the gateway. mlx_audio model changes require mlx_audio restart.

## Tested Environment

- macOS 15.6, Apple M4 Max
- Python 3.13.5
- mlx-audio 0.3.1, mlx-lm 0.30.5, mlx 0.30.6
- misaki 0.7.0, spacy 3.8.11, phonemizer 3.3.0
- fastapi 0.130.0, uvicorn 0.41.0, httpx 0.28.1
