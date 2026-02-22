# Clawfinger Gateway

A local voice gateway that runs the full **ASR → LLM → TTS** pipeline on Apple Silicon using MLX models. Designed for the Clawfinger Android app — the phone connects via ADB reverse port forwarding, keeping everything on localhost with zero cloud dependencies.

## How It Works

1. Phone sends audio over HTTP to the gateway
2. **ASR** (Parakeet via mlx_audio) transcribes the caller's speech
3. **LLM** (Qwen 1.5B via mlx-lm, or any OpenAI-compatible endpoint) generates a reply
4. **TTS** (Kokoro via mlx_audio) synthesizes the reply to speech
5. Audio is returned to the phone as base64 WAV

The gateway also provides:

- **Control Center UI** — web dashboard for live call monitoring, instruction editing, LLM parameter tuning, and session logs
- **Agent Interface** — WebSocket protocol for external agents to observe calls, take over LLM generation, inject TTS messages, and query call state
- **Instruction System** — three-layer prompt management (global / per-session / per-turn) with automatic compaction for long instruction chains
- **Outbound Dialing** — trigger calls on the phone via ADB broadcast

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- ADB for phone connection
- ~4 GB disk for models

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install 'misaki==0.7.0' num2words spacy phonemizer mecab-python3 unidic-lite webrtcvad 'setuptools<81'

# Configure
cp config.example.json config.json

# Start
bin/start.sh
```

The control center is at `http://127.0.0.1:8996`. See [SKILL.md](SKILL.md) for full installation, configuration, API documentation, and troubleshooting.

## API Overview

### Phone API (bearer auth required)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health check — mlx_audio + LLM status |
| `POST /api/asr` | ASR only — audio file in, transcript out |
| `POST /api/turn` | Full voice turn — ASR → LLM → TTS (audio in, audio out) |
| `POST /api/session/new` | Create a new session |
| `POST /api/session/reset` | Reset session history |

### Control Center (no auth)

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Control center web UI |
| `GET /api/status` | System status (uptime, calls, model health, config) |
| `GET /api/sessions` | List persisted sessions |
| `GET /api/sessions/{id}` | Session detail with turn-by-turn transcript |
| `POST /api/config` | Hot-reload config from disk |
| `GET/POST /api/config/llm` | View/update LLM generation parameters at runtime |
| `POST /api/call/inject` | Inject TTS message into event stream |
| `POST /api/call/dial` | Dial outbound call via ADB |
| `WS /ws/events` | Real-time event stream for UI |

### Instructions

| Endpoint | Purpose |
|----------|---------|
| `GET /api/instructions` | Get current instructions (base + per-session) |
| `POST /api/instructions` | Set base (global) instruction |
| `POST /api/instructions/{sid}` | Set per-session instruction |
| `POST /api/instructions/{sid}/turn` | Set one-shot per-turn supplement |
| `DELETE /api/instructions/{sid}` | Clear per-session instruction |

### Agent Interface

| Endpoint | Purpose |
|----------|---------|
| `WS /api/agent/ws` | Agent WebSocket (takeover, inject, observe, query state) |
| `POST /api/agent/inject` | REST inject — TTS message into call |
| `GET /api/agent/sessions` | List active session IDs |
| `GET /api/agent/call/{sid}` | Query call state (history, instructions, metadata) |

## License

[MIT](LICENSE.md)
