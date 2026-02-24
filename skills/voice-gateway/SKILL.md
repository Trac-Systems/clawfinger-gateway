---
name: local-voice-gateway
description: Local voice gateway for Clawfinger. Use this skill when installing, configuring, running, or troubleshooting the FastAPI gateway (ASR/LLM/TTS), its API endpoints, agent WebSocket protocol, instruction system, or control center UI. Runs on macOS and Linux.
---

# Local Voice Gateway — Installation & Operations Skill

> **Platform**: macOS and Linux. Not compatible with Windows.

> **Repo**: [Trac-Systems/clawfinger-gateway](https://github.com/Trac-Systems/clawfinger-gateway/)
> **Companion repo**: [Trac-Systems/clawfinger-app](https://github.com/Trac-Systems/clawfinger-app/) — the Android phone app. Set up the phone FIRST (root, install app), then come here for the gateway. Ignore this if you are an agent that just wants to use an already installed & running gateway/app stack!

## Important — OpenClaw Agents

**If you are an OpenClaw agent with the Clawfinger plugin loaded, DO NOT use the raw API endpoints or WebSocket directly.** Use the `clawfinger_*` tools instead (e.g. `clawfinger_status`, `clawfinger_sessions`, `clawfinger_dial`, `clawfinger_takeover`). The plugin maintains a persistent WebSocket bridge automatically — you never need to connect to WebSockets yourself. See the **clawfinger** skill for the tool reference.

If you are an agent without the plugin that only needs to use a gateway, then skip all installation instructions and focus on the REST API endpoints below.

## What This Is

A local voice gateway that handles phone calls for the Clawfinger Android app. It runs the full ASR → LLM → TTS pipeline locally — no cloud, no remote servers. The phone connects via ADB reverse port forwarding.

The gateway Python code (FastAPI) is fully cross-platform. It talks to the ASR/TTS sidecar and LLM backend via standard HTTP APIs. The default inference stack uses MLX on Apple Silicon, but on Linux you swap in compatible backends — see "Linux Setup" below.

## Overall setup order

The gateway is step 4 in the full setup flow. Complete the phone side first:

1. **Host setup**: Install ADB/fastboot (see [clawfinger-app SKILL.md](https://github.com/Trac-Systems/clawfinger-app/))
2. **Root the phone**: Follow the device-specific root skill in clawfinger-app
3. **Install the app**: Build and install the Clawfinger APK
4. **Set up the gateway** (you are here): Install, configure, start, and warm up
5. **Push a profile and connect**: Push a device profile, set up ADB reverse, verify
6. **Tune endpoints**: Capture/playback endpoint training via the voice bridge skill in clawfinger-app

## Security: localhost-only

The gateway MUST only bind to `127.0.0.1`. Never use `0.0.0.0` or any network-facing address. This applies to everything: the gateway API, control center UI, mlx_audio sidecar, and any future services. No exceptions. The phone reaches the gateway via ADB reverse port forwarding — there is no reason to expose anything to any network, not even the local LAN.

## Prerequisites

- Python 3.12+ (tested on 3.13.5, 3.13.12)
- ADB (Android Debug Bridge) for phone connection
- **ffmpeg** — required by mlx_audio for audio format handling (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux)
- ~4GB disk for models, ~500MB for venv
- **macOS (Apple Silicon)**: MLX-based inference out of the box — no extra setup
- **Linux**: Requires separate ASR/TTS server and LLM server — see "Linux Setup" below

## Complete Installation (macOS / Apple Silicon)

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

### Step 2b: Install Piper TTS (German)

```bash
pip install piper-tts flask pathvalidate
```

**Critical version pins:**
| Package | Pin | Reason |
|---------|-----|--------|
| `misaki` | `==0.7.0` | 0.7.4 crashes with `NoneType` phonemes bug; 0.6.x missing `MToken` class needed by mlx_audio 0.3.x |
| `setuptools` | `<81` | `webrtcvad` imports `pkg_resources` which was removed in setuptools 82+ |

### Step 3: Download models

All models are cached in `.models/` via `HF_HOME`. Download them before first run to avoid cold-start delays.

**Speed up downloads**: Set `HF_TOKEN` to a HuggingFace access token for higher rate limits and faster parallel downloads. Without a token, downloads are throttled and can be very slow (~4 GB of models total). Create a token at https://huggingface.co/settings/tokens (read access is sufficient).

```bash
export HF_HOME="$(pwd)/.models"
export HF_TOKEN="hf_YOUR_TOKEN_HERE"  # optional but recommended — much faster downloads
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

# Download Piper German TTS model (ONNX — not a HuggingFace model)
mkdir -p voices
curl -L -o voices/de_DE-thorsten-high.onnx \
  'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx?download=true'
curl -L -o voices/de_DE-thorsten-high.onnx.json \
  'https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json?download=true'
```

**Models:**
| Model | Purpose | Size | Notes |
|-------|---------|------|-------|
| `mlx-community/parakeet-tdt-0.6b-v2` | ASR (speech-to-text) | 2.3 GB | Do NOT use `whisper-small-mlx` — broken processor with mlx_audio 0.3.x |
| `mlx-community/Kokoro-82M-bf16` | TTS (text-to-speech) | 375 MB | Voice: `af_heart`, speed: 1.2 |
| `de_DE-thorsten-high` (Piper) | TTS (German) | 109 MB | ONNX, Piper HTTP sidecar on :5123 |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | LLM (conversation) | 852 MB | Loaded in-process via mlx-lm |

### Step 4: Configure

Copy the example config and edit as needed:

```bash
cp config.example.json config.json
```

`config.example.json` ships with working defaults for local MLX inference. Edit `config.json` to change your bearer token or point to a remote LLM.

**Key settings:**
- `bearer_token`: Must be non-empty. Phone profile must have the same value or profile parsing fails silently.
- `llm_model`: Model name — loaded locally via MLX when `llm_base_url` is empty, or sent as `model` field to a remote OpenAI-compatible endpoint.
- `llm_base_url`: Empty = local MLX. Set to an OpenAI-compatible base URL (e.g. `http://localhost:11434/v1`) for remote inference.
- `llm_api_key`: Bearer token for remote endpoint (empty = no auth).
- `llm_top_p`, `llm_top_k`, `llm_repeat_penalty`, `llm_stop`: Generation parameters — adjustable at runtime via the control center LLM Settings panel or `POST /api/config/llm`.
- `tts_voice`: Kokoro voice ID (e.g. `af_heart`, `am_michael`). See `GET /api/config/tts` for the full list of available voices grouped by category.
- `tts_speed`: 1.2 is natural cadence for Kokoro. Lower = slower speech.
- `tts_lang`: TTS language — `en` (Kokoro) or `de` (Piper German). Default: `en`. **Control-center-only** — agents cannot change this.
- `piper_base`: Piper HTTP server URL. Default: `http://127.0.0.1:5123`.
- `piper_voice`: Piper voice model name. Default: `thorsten-high`. Options: `thorsten-high`, `thorsten-medium`, `thorsten-low`, `karlsson-low`, `pavoque-low`, `eva_k-x_low`, `kerstin-low`, `ramona-low`, `thorsten_emotional-medium`.
- `piper_speaker`: Speaker ID for multi-speaker models (e.g. thorsten_emotional emotions: amused=0, angry=1, disgusted=2, drunk=3, neutral=4, sleepy=5, surprised=6, whisper=7). Default: `0`.
- `piper_length_scale`: Piper speech rate (lower = faster). Default: `1.0`.
- `piper_noise_scale`: Piper expressiveness/variation. Default: `0.667`.
- `piper_noise_w`: Piper phoneme duration variation. Default: `0.8`.
- `piper_sentence_silence`: Silence between sentences in seconds. Default: `0.2`.
- `llm_top_p_enabled`, `llm_top_k_enabled`: Boolean flags to enable/disable sending `top_p` / `top_k` to the model. Default: both `true`. Useful when remote APIs don't support certain params.
- `llm_context_tokens`: Total context window size in tokens. 0 = no token-based limit (use `max_history_turns` only). When set, history compaction also respects this budget.
- All config changes made via the control center or API are saved to `config.json` automatically and take effect immediately. LLM model changes are hot-loaded on the next turn — no restart needed.
- All settings can be overridden via env vars: `GATEWAY_PORT=9000`, `GATEWAY_BEARER_TOKEN=xyz`, etc.

**Call policy settings** (gateway is the single source of truth — phone fetches these at each call start):

| Setting | Default | Description |
|---------|---------|-------------|
| `call_auto_answer` | `true` | Auto-answer incoming calls |
| `call_auto_answer_delay_ms` | `500` | Delay before answering (ms) |
| `caller_allowlist` | `[]` | Phone numbers allowed to call. Empty = allow all. |
| `caller_blocklist` | `[]` | Phone numbers always blocked. Checked even when allowlist is empty. |
| `unknown_callers_allowed` | `true` | Accept calls with hidden/unavailable caller ID |
| `keep_history` | `false` | Persist conversation history per caller number across calls. When enabled, returning callers resume with their previous conversation context. When disabled, any saved histories are deleted on the next call from each number. Unknown/hidden callers always start fresh. |
| `greeting_incoming` | `"Hello, I am {owner}'s assistant..."` | Greeting for incoming calls. `{owner}` replaced with `greeting_owner`. |
| `greeting_outgoing` | `"Hello, this is {owner}'s assistant calling. Please wait for the beep before speaking."` | Greeting for outgoing calls. `{owner}` replaced with `greeting_owner`. |
| `greeting_owner` | `"the owner"` | Name substituted into `{owner}` placeholders |
| `max_duration_sec` | `300` | Max call duration in seconds |
| `max_duration_message` | `"I'm sorry, but we have reached..."` | TTS message played when max duration is reached |

**Security settings** (passphrase authentication):

| Setting | Default | Description |
|---------|---------|-------------|
| `auth_passphrase` | `""` | Voice passphrase. Empty = disabled. When set, caller must speak the passphrase before the AI engages. |
| `auth_reject_message` | `"I'm sorry, I can't help you right now. Goodbye."` | Message played on auth failure or caller rejection |
| `auth_max_attempts` | `3` | Max passphrase attempts before rejection. 0 = unlimited. |

All call policy and security settings are configurable at runtime via the control center UI or `POST /api/config/call`.

## Linux Setup

The gateway Python code runs on Linux without modification. The only macOS-specific components are the default inference backends (`mlx_audio` for ASR/TTS, `mlx-lm` for LLM). On Linux, replace them with compatible servers:

### ASR/TTS sidecar replacement

The gateway talks to the ASR/TTS sidecar via standard OpenAI-compatible HTTP endpoints. Any server implementing these works:

- `POST /v1/audio/transcriptions` — ASR (Whisper-compatible)
- `POST /v1/audio/speech` — TTS
- `GET /v1/models` — health check

Set `mlx_audio_base` in `config.json` to point at the replacement server (default: `http://127.0.0.1:8765`).

**Compatible Linux alternatives:**
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) — OpenAI-compatible ASR with CUDA
- [openedai-speech](https://github.com/matatonic/openedai-speech) — OpenAI-compatible TTS
- Any OpenAI-compatible speech API that supports the endpoints above

### LLM replacement

Set `llm_base_url` in `config.json` to a local LLM server. The gateway sends standard OpenAI chat completion requests:

```json
{
  "llm_base_url": "http://127.0.0.1:11434/v1",
  "llm_model": "qwen2.5:1.5b",
  "llm_api_key": ""
}
```

**Compatible Linux alternatives:**
- [Ollama](https://ollama.com) — `http://localhost:11434/v1`
- [vLLM](https://github.com/vllm-project/vllm) — OpenAI-compatible server with CUDA
- [llama.cpp server](https://github.com/ggerganov/llama.cpp) — CPU or CUDA
- Any OpenAI-compatible chat completion endpoint

### Linux installation steps

```bash
# 1. Create venv and install base deps (same as macOS)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Skip mlx-audio and mlx-lm (Apple Silicon only)
#    Instead, start your ASR/TTS server and LLM server separately

# 3. Configure
cp config.example.json config.json
# Edit config.json:
#   - Set mlx_audio_base to your ASR/TTS server URL
#   - Set llm_base_url to your LLM server URL
#   - Set llm_model to the model name your LLM server expects

# 4. Start gateway only (no bin/start.sh — that starts mlx_audio)
source .venv/bin/activate
python app.py
```

## Running — Starting and Stopping the Gateway

> **IMPORTANT FOR AGENTS**: The gateway consists of TWO processes that must both be running for calls to work: the **mlx_audio sidecar** (ASR/TTS inference) and the **gateway** (FastAPI API + control center). If either is not running, calls will fail silently — the phone picks up but produces no audio. When the user asks to start the gateway or reports call issues, **always check if both processes are running** (`lsof -i :8996 -i :8765 | grep LISTEN`). If not, tell the user they can start it manually with `bin/start.sh` and should run it in the background to keep it up while they work.

### Start (recommended)

```bash
cd /path/to/gateway
bin/start.sh
```

This starts all processes in the foreground:
1. **mlx_audio server** on `127.0.0.1:8765` — handles ASR and Kokoro TTS model inference
2. **Piper TTS server** on `127.0.0.1:5123` — handles German TTS (if voice model present)
3. **Gateway** (FastAPI/uvicorn) on `127.0.0.1:8996` — phone API, control center UI, agent interface

The script waits for mlx_audio to be healthy before starting the gateway. LLM is preloaded at gateway startup. Ctrl+C stops both.

**To run in the background** (recommended so it stays up while you work):

```bash
cd /path/to/gateway
nohup bin/start.sh > /tmp/gateway-all.log 2>&1 &
echo "Gateway starting in background, logs at /tmp/gateway-all.log"
```

Or if the agent is starting it programmatically:
```bash
cd /path/to/gateway && bin/start.sh > /tmp/gateway-all.log 2>&1 &
```

### Stop

```bash
cd /path/to/gateway
bin/stop.sh
```

Or kill individually:
```bash
kill $(cat tmp/gateway.pid) 2>/dev/null
kill $(cat tmp/mlx_audio.pid) 2>/dev/null
```

Or by port:
```bash
lsof -ti :8996 | xargs kill 2>/dev/null   # gateway
lsof -ti :8765 | xargs kill 2>/dev/null   # mlx_audio
```

### Checking status

```bash
# Are all processes listening?
lsof -i :8996 -i :8765 -i :5123 | grep LISTEN

# Full health check
curl -s http://127.0.0.1:8996/api/status | python3 -m json.tool
```

In the status response, check:
- `mlx_audio.ok` should be `true` — if `false` or missing, mlx_audio is not running
- `piper.ok` should be `true` when German TTS is needed
- `llm.loaded` should be `true` — if `false`, LLM failed to load

### Manual start (for debugging)

Use this when you need separate log output for each process:

```bash
source .venv/bin/activate
export HF_HOME="$(pwd)/.models"

# Terminal 1: mlx_audio
python -m mlx_audio.server --host 127.0.0.1 --port 8765

# Terminal 2: gateway
python app.py
```

### Restarting after code changes

The gateway serves `static/index.html` from disk on each request, but Python code changes require a restart:

```bash
cd /path/to/gateway
bin/stop.sh && bin/start.sh > /tmp/gateway-all.log 2>&1 &
```

> **Browser cache warning**: After restarting the gateway, the control center UI may still show old HTML from the browser cache. Always hard-refresh (Cmd+Shift+R) after a gateway restart.

## Pre-warming Models

First request to each model has extra latency due to model loading into GPU memory. **Without warming, the first phone call will fail** — the greeting TTS takes too long and the caller hears silence. `bin/start.sh` does NOT auto-warm models, so you must warm them after startup.

**Typical first-load times** (measured on Apple M4 Max, will vary by hardware):
| Model | First load | Subsequent calls |
|-------|-----------|-----------------|
| TTS (Kokoro) | ~2s | ~0.5s |
| ASR (Parakeet) | ~90s | ~0.5s |
| LLM (Qwen) | auto-warmed at startup | ~0.3s |

```bash
# Warm TTS (Kokoro) — fast, do this first
curl -s -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Kokoro-82M-bf16","input":"Warm up.","voice":"af_heart","speed":1.2,"response_format":"wav"}' \
  -o /dev/null

# Warm ASR (Parakeet) — slow (~90s), run in background
(python3 -c "
import wave, tempfile, os
os.makedirs('tmp', exist_ok=True)
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir='tmp') as f:
    w = wave.open(f, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b'\x00\x00' * 16000); w.close(); print(f.name)
" | xargs -I{} curl -s -X POST http://127.0.0.1:8765/v1/audio/transcriptions \
  -F "file=@{}" -F "model=mlx-community/parakeet-tdt-0.6b-v2" -F "language=en" -o /dev/null) &
echo "ASR warming in background (~90s)..."
```

The LLM is automatically pre-warmed at gateway startup when running in local MLX mode (i.e. `llm_base_url` is empty). Calls will work for greetings and forced replies immediately after TTS warmup; ASR warmup is only needed before the first real voice turn.

Piper does not need explicit warming — the ONNX model loads in ~350ms on first request. However, the first German TTS request will have this minor startup delay.

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
| `POST` | `/api/session/end` | Mark session as ended — `{"session_id": "..."}` → `{"ok": true}`. Publishes `session.ended` event. |

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
| `caller_number` | string | no | Caller's phone number (sent by phone app for filtering) |
| `call_direction` | string | no | `"incoming"` or `"outgoing"` |

**Turn flow:**
1. **Caller filtering** — if `caller_number` is provided, checked against `caller_blocklist`, `caller_allowlist`, and `unknown_callers_allowed`. Rejected callers receive `{"ok": false, "rejected": true, "reason": "..."}` with TTS rejection audio.
2. **Forced reply** — if `forced_reply` is set, skips ASR + LLM, goes straight to TTS.
3. **ASR** — transcribes audio (or uses `transcript_hint` if `skip_asr` is true).
4. **Passphrase gate** — if `auth_passphrase` is configured and session is not yet authenticated, the transcript is fuzzy-matched against the passphrase (case-insensitive, punctuation-stripped, substring match). On match, session is marked authenticated. On failure, attempt count is incremented; after `auth_max_attempts` failures, returns rejection message with `"hangup": true`.
5. **LLM** — generates response.
6. **TTS** — synthesizes reply audio.

**Response**:
```json
{
  "ok": true,
  "session_id": "abc123",
  "transcript": "what caller said",
  "reply": "assistant response",
  "audio_base64": "<base64 WAV>",
  "hangup": false,
  "rejected": false,
  "metrics": {
    "asr_ms": 450.2,
    "llm_ms": 355.8,
    "tts_ms": 630.4,
    "total_ms": 1436.4,
    "llm_model": "local/mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  }
}
```

**Special response fields:**
- `rejected: true` — caller was blocked by allowlist/blocklist/unknown policy. App should play audio and hang up.
- `hangup: true` — gateway requests call termination (e.g., max auth attempts exceeded). App should play audio and hang up.

The phone reads `audio_base64` first, falls back to `audio_wav_base64`, then `audioBase64`.

### Control Center (no auth)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Control center SPA UI |
| `GET` | `/api/status` | System status (uptime, call count, model health, config) |
| `GET` | `/api/sessions` | List persisted sessions |
| `GET` | `/api/sessions/{id}` | Full session detail with turn-by-turn transcript |
| `POST` | `/api/config` | Hot-reload config from disk |
| `GET` | `/api/config/tts` | Current TTS settings + available voices |
| `POST` | `/api/config/tts` | Update voice, speed |
| `POST` | `/api/tts/preview` | Preview TTS voice with sample audio |
| `GET` | `/api/config/llm` | Current LLM generation params |
| `POST` | `/api/config/llm` | Hot-update LLM params — `{"temperature": 0.5, "top_p": 0.9, ...}` |
| `GET` | `/api/config/call` | Current call policy + security settings |
| `POST` | `/api/config/call` | Update call policy + security settings |
| `POST` | `/api/call/inject` | Inject TTS message — `{"text": "...", "session_id": "..."}` |
| `POST` | `/api/call/dial` | Dial outbound call — `{"number": "+49..."}` |
| `POST` | `/api/call/hangup` | Force hang up active call — `{"session_id": "..."}` (optional). Sends ADB hangup broadcast and ends gateway session. |
| `GET` | `/api/caller-history` | List saved caller histories (number, total_calls, last_call_at) |
| `DELETE` | `/api/caller-history/{number}` | Delete saved history for a caller number |
| `WS` | `/ws/events` | Real-time event stream for UI |

### TTS Config API

**`GET /api/config/tts`** — Returns current TTS settings. Response shape depends on active language.

When `lang: "en"` (Kokoro):
```json
{
  "lang": "en",
  "model": "mlx-community/Kokoro-82M-bf16",
  "voice": "af_heart",
  "speed": 1.2,
  "voices": {
    "American Female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "American Male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"],
    "British Female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "British Male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
  }
}
```

When `lang: "de"` (Piper):
```json
{
  "lang": "de",
  "model": "mlx-community/Kokoro-82M-bf16",
  "piper_voice": "thorsten-high",
  "piper_speaker": 0,
  "piper_length_scale": 1.0,
  "piper_noise_scale": 0.667,
  "piper_noise_w": 0.8,
  "piper_sentence_silence": 0.2,
  "voices": {
    "Male": ["thorsten-high", "thorsten-medium", "thorsten-low", "karlsson-low", "pavoque-low"],
    "Female": ["eva_k-x_low", "kerstin-low", "ramona-low"],
    "Emotional": ["thorsten_emotional-medium"]
  },
  "emotions": {"amused": 0, "angry": 1, "disgusted": 2, "drunk": 3, "neutral": 4, "sleepy": 5, "surprised": 6, "whisper": 7}
}
```

**`POST /api/config/tts`** — Updates TTS settings. Accepts:

| Field | Type | Description |
|-------|------|-------------|
| `lang` | string | `"en"` or `"de"` — switches TTS engine. **Control-center-only.** |
| `voice` / `tts_voice` | string | Kokoro voice ID (English mode) |
| `speed` / `tts_speed` | float | Kokoro speed (English mode) |
| `piper_voice` | string | Piper voice model name (German mode) |
| `piper_speaker` | int | Piper speaker ID (German mode) |
| `piper_length_scale` | float | Piper speech rate (German mode) |
| `piper_noise_scale` | float | Piper expressiveness (German mode) |
| `piper_noise_w` | float | Piper duration variation (German mode) |
| `piper_sentence_silence` | float | Piper sentence pause in seconds (German mode) |

Returns the full TTS config response (same shape as GET). Saves to `config.json`. Publishes `config.tts_updated` event.

**`POST /api/tts/preview`** — Synthesize a sample phrase. Routes to Kokoro (English) or Piper (German) based on current `tts_lang`.

```bash
curl -X POST http://127.0.0.1:8996/api/tts/preview \
  -H "Content-Type: application/json" \
  -d '{"voice": "am_michael", "speed": 1.0, "text": "Hello, how are you?"}'
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `text` | no | `"Hello, this is a voice preview."` (EN) / `"Hallo, das ist eine Sprachvorschau."` (DE) | Text to synthesize |
| `voice` | no | current `tts_voice` | Voice ID to preview (English only) |
| `speed` | no | current `tts_speed` | Speed to preview (English only) |

Does not affect the current config — use `POST /api/config/tts` to apply.

### LLM Config API

**`GET /api/config/llm`** — Returns all LLM generation parameters.

```json
{
  "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "base_url": "",
  "has_api_key": false,
  "max_tokens": 400,
  "context_tokens": 0,
  "max_history_turns": 8,
  "temperature": 0.2,
  "top_p": 1.0,
  "top_p_enabled": true,
  "top_k": 0,
  "top_k_enabled": true,
  "repeat_penalty": 1.0,
  "stop": [],
  "is_local": true
}
```

| Field | Description |
|-------|-------------|
| `model` | Model name (loaded locally via MLX when `base_url` is empty, or sent to remote endpoint) |
| `base_url` | Empty = local MLX. Set to an OpenAI-compatible base URL for remote inference. |
| `has_api_key` | `true` if `llm_api_key` is set (the actual key is never returned) |
| `max_tokens` | Max output tokens per LLM turn |
| `context_tokens` | Total context window budget in tokens. `0` = no token-based limit (use `max_history_turns` only). |
| `max_history_turns` | Max conversation turns to keep in verbatim history before compaction triggers |
| `temperature` | Sampling temperature |
| `top_p` | Nucleus sampling threshold (only sent to model if `top_p_enabled` is `true` AND value < 1.0) |
| `top_p_enabled` | Whether `top_p` is sent to the model. Disable for APIs that don't support it. |
| `top_k` | Top-K sampling (only sent to local MLX model if `top_k_enabled` is `true` AND value > 0) |
| `top_k_enabled` | Whether `top_k` is sent to the model. OpenAI API doesn't support `top_k`. |
| `repeat_penalty` | Repetition penalty (local MLX only) |
| `stop` | Stop sequences |
| `is_local` | `true` if using local MLX inference (`base_url` empty), `false` if remote |

**`POST /api/config/llm`** — Updates LLM settings. Accepts any subset of fields. Both short aliases and full config keys work:

```bash
curl -X POST http://127.0.0.1:8996/api/config/llm \
  -H "Content-Type: application/json" \
  -d '{"temperature": 0.5, "top_p": 0.9, "top_k_enabled": false, "context_tokens": 4096}'
```

| Field | Aliases | Type | Description |
|-------|---------|------|-------------|
| `model` | `llm_model` | string | Model name |
| `base_url` | `llm_base_url` | string | Remote endpoint URL (empty = local) |
| `api_key` | `llm_api_key` | string | Bearer token for remote endpoint |
| `max_tokens` | `llm_max_tokens` | int | Max output tokens |
| `temperature` | `llm_temperature` | float | Sampling temperature |
| `top_p` | `llm_top_p` | float | Nucleus sampling threshold |
| `top_p_enabled` | `llm_top_p_enabled` | bool | Send `top_p` to model |
| `top_k` | `llm_top_k` | int | Top-K sampling |
| `top_k_enabled` | `llm_top_k_enabled` | bool | Send `top_k` to model |
| `repeat_penalty` | `llm_repeat_penalty` | float | Repetition penalty |
| `stop` | `llm_stop` | list | Stop sequences |
| `context_tokens` | `llm_context_tokens` | int | Context window budget |
| `max_history_turns` | — | int | Max verbatim history turns |

Returns the full LLM config response (same shape as GET). Saves to `config.json`. Publishes `config.llm_updated` event. Model changes are hot-loaded on the next turn — no restart needed.

### Call Policy API

**`GET /api/config/call`** — Returns all call policy settings. The phone app fetches this at each call start.

```json
{
  "auto_answer": true,
  "auto_answer_delay_ms": 500,
  "caller_allowlist": [],
  "caller_blocklist": [],
  "unknown_callers_allowed": true,
  "greeting_incoming": "Hello, I am Markus's assistant...",
  "greeting_outgoing": "Hello, this is Markus's assistant calling.",
  "greeting_incoming_template": "Hello, I am {owner}'s assistant...",
  "greeting_outgoing_template": "Hello, this is {owner}'s assistant calling. Please wait for the beep before speaking.",
  "greeting_owner": "Markus",
  "max_duration_sec": 300,
  "max_duration_message": "...",
  "auth_required": true,
  "auth_reject_message": "...",
  "auth_max_attempts": 3,
  "keep_history": false
}
```

Note: `auth_passphrase` is **never returned** — only `auth_required` (bool). The phone doesn't need the passphrase; the gateway handles verification internally. The `greeting_*_template` fields contain the raw `{owner}` placeholders for editing; the `greeting_incoming`/`greeting_outgoing` fields have `{owner}` already resolved.

**`POST /api/config/call`** — Updates call settings. Accepts any subset of fields:

```bash
curl -X POST http://127.0.0.1:8996/api/config/call \
  -H "Content-Type: application/json" \
  -d '{"greeting_incoming": "Hi!", "max_duration_sec": 600, "auth_passphrase": "blue harvest"}'
```

Publishes `config.call_updated` event to all UI/agent subscribers.

### Passphrase Authentication

When `auth_passphrase` is set (non-empty), callers must speak the passphrase before the AI assistant engages:

1. **Greeting plays normally** — `forced_reply` turns bypass the passphrase gate.
2. **First real turn** — the caller's transcript is fuzzy-matched against the passphrase:
   - Case-insensitive: `"Blue Harvest"` matches `"blue harvest"`
   - Punctuation stripped: `"blue, harvest!"` matches `"blue harvest"`
   - Substring match: `"the passphrase is blue harvest"` matches `"blue harvest"`
3. **On match** — session is marked authenticated, reply: "Authentication successful. How can I help you?"
4. **On failure** — attempt counter incremented:
   - Under `auth_max_attempts`: reply: "That's not correct. Please try again."
   - At `auth_max_attempts` (default 3): reply: `auth_reject_message` + `hangup: true` → phone disconnects.
   - `auth_max_attempts: 0` = unlimited retries.
5. **Once authenticated**, the session stays authenticated for all subsequent turns.

### Instructions

Three-layer instruction system controlling the LLM system prompt:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/instructions` | Returns `{base, sessions: {sid: text}}` |
| `POST` | `/api/instructions` | Set base (global) instruction — `{"text": "..."}` |
| `POST` | `/api/instructions/{sid}` | Set per-session instruction override |
| `POST` | `/api/instructions/{sid}/turn` | Set one-shot per-turn supplement (consumed after next turn) |
| `DELETE` | `/api/instructions/{sid}` | Clear per-session instruction |

**Prompt assembly** (per turn):
1. **System instruction** (from config or `/api/instructions`) — never compacted, always in full. Session instruction wins over base; base falls back to `config.json`'s `llm_system_prompt` if unset. Per-turn supplement is appended with `\n\n` separator and consumed after one use.
2. **Summary of earlier conversation** (if history was compacted by LLM summarization)
3. **Recent conversation history** (verbatim user/assistant pairs)
4. **Agent injected knowledge** (if any — see Agent Context Injection below)
5. **Current user transcript**

### Conversation History Compaction

When conversation history exceeds `max_history_turns`, older messages are summarized by the LLM and replaced with a compact summary. The summary is injected as a system message before the recent verbatim history. This preserves key facts (caller identity, decisions, commitments) while keeping the context window within budget. The summary is updated each time compaction triggers. If `llm_context_tokens` is set, the token budget is also enforced by shrinking the number of kept messages.

### Agent Context Injection

Agents can inject knowledge into a session that persists until cleared. Injected knowledge appears as a system message right before the current user turn, so the LLM has it fresh. Each `inject_context` call **replaces** the previous knowledge for that session (one slot per session). Use the REST or WebSocket endpoints below.

**`GET /api/agent/context/{session_id}`** — Read the current injected knowledge for a session.

```json
{
  "session_id": "abc123",
  "knowledge": "Caller is John Smith, account #12345. Balance: €1,234.56.",
  "has_knowledge": true
}
```

`has_knowledge` is `false` and `knowledge` is `""` when nothing is injected.

**`POST /api/agent/context/{session_id}`** — Inject or replace knowledge for a session.

```bash
curl -X POST http://127.0.0.1:8996/api/agent/context/abc123 \
  -H "Content-Type: application/json" \
  -d '{"context": "Caller is John Smith, account #12345. Balance: €1,234.56."}'
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `context` | yes | string | Knowledge text to inject. Sanitized via `safe_text()`. |

Returns `{"ok": true, "session_id": "abc123"}`. Returns HTTP 400 if `context` is empty. Publishes `agent.context_injected` event.

Each POST **replaces** the previous knowledge — there is one knowledge slot per session. To update, just POST again with the new text.

**`DELETE /api/agent/context/{session_id}`** — Clear injected knowledge for a session.

```bash
curl -X DELETE http://127.0.0.1:8996/api/agent/context/abc123
```

Returns `{"ok": true, "session_id": "abc123"}`. Publishes `agent.context_cleared` event. Knowledge is also automatically cleared when a session is reset.

### Agent Call State API

**`GET /api/agent/sessions`** — List all active session IDs.

Returns a JSON array of session ID strings:
```json
["abc123", "def456"]
```

**`GET /api/agent/call/{sid}`** — Full call state for a session. Useful for agents to understand context before taking over.

```json
{
  "session_id": "abc123",
  "history": [
    {"role": "user", "content": "Hi, I need help with my account."},
    {"role": "assistant", "content": "Sure! What's your account number?"}
  ],
  "turn_count": 1,
  "instructions": {
    "base": "You are a concise, friendly real-time voice assistant.",
    "session": "",
    "pending_turn": ""
  },
  "agent_takeover": false,
  "created_at": null
}
```

| Field | Description |
|-------|-------------|
| `history` | Full conversation history (recent verbatim messages) |
| `turn_count` | Number of completed turns |
| `instructions` | Current instruction layers (base, session override, pending one-shot turn supplement) |
| `agent_takeover` | `true` if an agent has taken over LLM control for this session |
| `created_at` | Session creation timestamp (if persisted) or `null` |

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
| `GET` | `/api/agent/call/{sid}` | Full call state — history, instructions, turn count, takeover status |
| `GET` | `/api/agent/context/{sid}` | Read injected agent knowledge |
| `POST` | `/api/agent/context/{sid}` | Inject/replace agent knowledge — `{"context": "..."}` |
| `DELETE` | `/api/agent/context/{sid}` | Clear agent knowledge |

**Agent WebSocket protocol** — agent sends JSON messages:

| Message | Fields | Description |
|---------|--------|-------------|
| `takeover` | `session_id` | Take over LLM for a session (gateway forwards transcripts, agent replies) |
| `release` | `session_id` | Release LLM control back to local LLM |
| `inject` | `text`, `session_id` | Inject TTS message into call |
| `set_instructions` | `instructions`, `session_id`, `scope` | Set instructions; scope: `global`, `session`, or `turn` |
| `set_call_config` | `config` | Update call policy settings (non-security subset only) |
| `dial` | `number` | Dial outbound call |
| `hangup` | `session_id` (optional) | Force hang up the active call and end gateway session |
| `get_call_state` | `session_id` | Query full call state (history, instructions, takeover status) |
| `inject_context` | `session_id`, `context` | Inject/replace agent knowledge for a session |
| `clear_context` | `session_id` | Clear injected agent knowledge |
| `end_session` | `session_id` | Mark a session as ended (hung up) |
| `ping` | — | Heartbeat |

**Ack/response messages**: `takeover.ack`, `release.ack`, `set_instructions.ack`, `set_call_config.ack`, `dial.ack`, `hangup.ack`, `call_state`, `inject_context.ack`, `clear_context.ack`, `end_session.ack`, `pong`.

**`set_call_config`** — agents can adjust greetings and call parameters but **NOT security settings**:

Allowed keys: `greeting_incoming`, `greeting_outgoing`, `greeting_owner`, `max_duration_sec`, `max_duration_message`, `call_auto_answer`, `call_auto_answer_delay_ms`, `keep_history`, `tts_voice`, `tts_speed`.

**Not allowed from agents**: `tts_lang` (language switching is control-center-only), `piper_*` settings, `auth_*`, `caller_allowlist`, `caller_blocklist`, `unknown_callers_allowed`.

### Takeover turn protocol (`request_id` correlation)

During takeover, the gateway sends a `turn.request` with a unique `request_id`:

```json
{
  "type": "turn.request",
  "session_id": "abc123",
  "transcript": "what the caller said",
  "request_id": "a1b2c3d4..."
}
```

The agent MUST reply within 30 seconds with the `request_id` echoed back:

```json
{
  "reply": "the agent's response text",
  "request_id": "a1b2c3d4..."
}
```

The `request_id` field is required for reliable reply correlation. The gateway's WebSocket uses a single-reader pattern — only the `agent_ws` loop reads from the socket. The `/api/turn` endpoint posts a pending request and awaits a Future that the loop resolves when the matching `request_id` arrives. This eliminates the previous race condition where two readers competed for the same WebSocket.

If the agent does not include `request_id` in its reply, the reply cannot be correlated and the turn will time out, falling back to local LLM.

### Session locking

The gateway uses per-session asyncio locks to coordinate concurrent access to session state. The lock is held during:
- LLM prompt assembly and generation in `/api/turn`
- Context injection (`inject_context` via REST and WS)
- Context clearing (`clear_context` via REST and WS)

This prevents races where an `inject_context` arrives mid-prompt-assembly, ensuring the LLM always sees a consistent snapshot of session state.

Agent receives all event bus events: `turn.started`, `turn.transcript`, `turn.reply`, `turn.complete`, `turn.error`, `turn.caller_rejected`, `turn.authenticated`, `turn.auth_failed`, `agent.connected`, `agent.disconnected`, `agent.inject`, `agent.takeover`, `agent.release`, `agent.context_injected`, `agent.context_cleared`, `instructions.updated`, `config.tts_updated`, `config.llm_updated`, `config.call_updated`, `call.dial`, `call.hangup`, `status.update`. The `turn.complete` event includes `transcript`, `reply`, and `session_id` alongside `metrics`.

## File Structure

```
gateway/
├── app.py                  # FastAPI application — all endpoints
├── config.py               # config.json loader + env var overrides
├── voice_pipeline.py       # ASR/TTS via mlx_audio HTTP
├── llm_backend.py          # LLM abstraction (local MLX / remote OpenAI)
├── session_store.py        # In-memory session history + caller info + auth state
├── event_bus.py            # Async pub/sub → UI + agent WebSockets
├── agent_interface.py      # Agent WS protocol (takeover/inject/release)
├── instruction_store.py    # In-memory instruction layers (base/session/turn)
├── static/index.html       # Control center SPA (vanilla HTML/JS/CSS)
├── config.example.json     # Example config with defaults (copy to config.json)
├── config.json             # Runtime configuration (gitignored)
├── README.md               # Project overview and API reference
├── LICENSE.md              # MIT license
├── SKILL.md                # Full installation, ops, and API docs
├── requirements.txt        # Base Python deps
├── bin/start.sh            # Start mlx_audio + gateway
├── bin/stop.sh             # Stop both
├── voices/                 # Piper TTS voice models (.onnx) — gitignored except .gitkeep
├── .gitignore              # Excludes .venv/, .models/, sessions/, tmp/
├── .venv/                  # Python virtual environment (not tracked)
├── .models/                # HuggingFace model cache (not tracked)
├── sessions/               # Persisted session JSON logs (not tracked)
├── caller_history/         # Per-caller conversation history (not tracked)
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

# 3b. Piper TTS works (German)
curl -s -X POST http://127.0.0.1:5123/ -d "Hallo, das ist ein Test." -o /dev/null -w "OK: Piper (%{size_download} bytes)\n"

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

### First turn slow / times out / greeting fails
Pre-warm models after startup. First TTS call loads Kokoro (~2s), first ASR call loads Parakeet (~90s on M4 Max). LLM is pre-warmed automatically. Without warming, the greeting turn can take long enough that the caller hangs up before hearing anything. Always warm models after starting services — `bin/start.sh` handles this automatically, but if starting manually, run the warmup commands from the "Pre-warming Models" section above.

### Piper TTS returns error / German doesn't work
- Check Piper is running: `lsof -i :5123 | grep LISTEN`
- Check model file exists: `ls -la voices/de_DE-thorsten-high.onnx`
- Missing Flask: `pip install flask` (Piper HTTP server requires it)
- Missing pathvalidate: `pip install pathvalidate`
- If Piper didn't start, check `PIPER_MODEL` path in start.sh matches voice file location

### Config changes not taking effect
Config changes made via the control center or API are saved to `config.json` automatically and take effect immediately. LLM model changes are hot-loaded on the next turn — no restart needed. Call `POST /api/config` to force a full reload from disk. mlx_audio model changes require mlx_audio restart.

## Tested Environment

**macOS (primary):**
- macOS 15.6, Apple M4 Max
- Python 3.13.5
- mlx-audio 0.3.1, mlx-lm 0.30.5, mlx 0.30.6
- misaki 0.7.0, spacy 3.8.11, phonemizer 3.3.0
- fastapi 0.130.0, uvicorn 0.41.0, httpx 0.28.1

**Linux**: The gateway Python code (FastAPI, voice_pipeline, llm_backend) is cross-platform and runs on Linux. Requires compatible ASR/TTS and LLM servers — see "Linux Setup" section above.
