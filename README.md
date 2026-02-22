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

| Endpoint | Purpose |
|----------|---------|
| `POST /api/turn` | Full voice turn (audio in, audio out) |
| `GET/POST /api/config/llm` | View/update LLM generation parameters at runtime |
| `GET/POST /api/instructions` | Manage system prompt layers |
| `WS /api/agent/ws` | Agent WebSocket (takeover, inject, observe) |
| `GET /api/agent/call/{sid}` | Query call state (history, instructions, metadata) |
| `POST /api/call/dial` | Dial outbound call via ADB |

## License

[MIT](LICENSE.md)
