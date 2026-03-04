# Clawfinger Gateway

A local voice gateway that runs the full **ASR → LLM → TTS** pipeline on Apple Silicon using MLX models. Handles two endpoint types — **phone calls** and **robot control** — with zero cloud dependencies.

## Architecture

```
                                        ┌── Intercom P2P ──► Jetson (hardware robot)
Phone (Android) ──ADB──► Gateway (:8996)┤
                             │          └── WebSocket ──────► Isaac Sim (sim adapter)
                    ┌────────┼────────┐
                    ▼        ▼        ▼
              mlx_audio   mlx-vlm   Piper
              ASR+TTS     LLM/VLM   DE TTS
              (:8765)               (:5123)
```

- **Phone endpoint**: Caller audio → ASR → LLM → TTS → audio back. The phone connects via `adb reverse tcp:8996`.
- **Robot endpoint**: Voice commands → LLM decomposes into actions → dispatched to robot via [Intercom](https://github.com/Trac-Systems/intercom/) P2P (Hyperswarm/HyperDHT) for hardware or WebSocket for simulation. Camera/mic perception, spatial memory, multi-step project orchestration.
- **Agent interface**: External agents (OpenClaw, custom) observe sessions, take over LLM, inject context, control both phone and robot.
- **Control center**: Browser UI at `:8996` for live management of everything.

## Model Stack

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| ASR | Parakeet TDT 0.6B | ~600MB | Speech-to-text (via mlx_audio) |
| LLM/VLM | Qwen3.5-4B 4-bit | 2.9GB | Text reasoning + multimodal vision (via mlx-vlm) |
| TTS (EN) | Kokoro 82M | 375MB | English speech synthesis (via mlx_audio) |
| TTS (DE) | Piper Thorsten | 109MB | German speech synthesis (ONNX sidecar) |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- ~4 GB disk for models
- ADB for phone connection (optional)

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
bin/start.sh
```

Control center: `http://127.0.0.1:8996`

## API Overview

### Phone

| Endpoint | Purpose |
|----------|---------|
| `POST /api/turn` | Full voice turn — ASR → LLM → TTS |
| `POST /api/asr` | ASR only — transcript out |
| `POST /api/call/dial` | Dial outbound call via ADB |
| `POST /api/call/hangup` | Hang up active call via ADB |
| `POST /api/call/inject` | Inject TTS into active call |
| `GET/POST /api/config/call` | Call policy (auto-answer, greetings, filtering) |
| `GET /api/caller-history` | Caller history |

### Robot

| Endpoint | Purpose |
|----------|---------|
| `GET/POST /api/config/robot` | Robot config (model, transport, wake word, safety) |
| `POST /api/robot/project/start` | Start a voice-driven project |
| `GET /api/robot/project` | Current project status with step progress |
| `POST /api/robot/project/cancel` | Cancel active project |
| `GET /api/robot/skills` | List available skill packages |
| `GET /api/robot/skills/{name}/{topic}` | Read skill knowledge (.md content) |
| `POST /api/robot/camera/snapshot` | Capture camera frame |
| `POST /api/robot/camera/describe` | VLM scene description |
| `GET /api/robot/camera/stream` | MJPEG video stream |
| `GET /api/robot/perception` | List cameras and mics |
| `WS /ws/robot` | Robot transport WebSocket (sim adapter / hardware) |

### Spatial Memory

| Endpoint | Purpose |
|----------|---------|
| `GET/POST /api/robot/memory/persons` | List / teach persons |
| `GET/POST /api/robot/memory/objects` | List / teach objects |
| `GET/POST /api/robot/memory/rooms` | List / define rooms |
| `POST /api/robot/memory/observations` | Record observation |
| `POST /api/robot/memory/query` | Natural language query with time filters |
| `POST /api/robot/memory/last_seen` | Most recent observation of entity |
| `GET /api/robot/memory/stats` | DB statistics with temporal range |

### Agent Interface

| Endpoint | Purpose |
|----------|---------|
| `WS /api/agent/ws` | Agent WebSocket — phone + robot control |
| `POST /api/agent/inject` | Inject TTS message |
| `GET /api/agent/sessions` | List active sessions |
| `GET /api/agent/call/{sid}` | Full call state |
| `GET/POST/DELETE /api/agent/context/{sid}` | Agent knowledge injection |
| `POST /api/agent/takeover` | Take over session LLM |
| `POST /api/agent/release` | Release LLM control |

### System

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health check |
| `GET /api/status` | System status (uptime, models, sessions) |
| `GET/POST /api/config/tts` | TTS voice, speed, language |
| `GET/POST /api/config/llm` | LLM model and generation params |
| `GET/POST /api/instructions` | Global LLM instructions |
| `POST /api/instructions/{sid}` | Per-session instructions |
| `POST /api/instructions/{sid}/turn` | One-shot turn supplement |
| `WS /ws/events` | Real-time event stream for UI |

## OpenClaw Integration

The gateway ships with an [OpenClaw](https://openclaw.dev) plugin at `openclaw/clawfinger/` providing 35+ tools for phone call control, robot commands, spatial memory, and perception — all accessible from OpenClaw agents and slash commands.

## Documentation

- [Phone Gateway](skills/phone-gateway/SKILL.md) — phone API, call policy, agent protocol
- [Robot Gateway](skills/robot-gateway/SKILL.md) — robot API, transport, perception, spatial memory
- [Control Center](skills/control-center/SKILL.md) — browser UI panels and features
- [Agent Takeover](skills/agent-takeover/SKILL.md) — takeover lifecycle and turn protocol
- [OpenClaw Plugin](skills/openclaw-clawfinger/SKILL.md) — all plugin tools and slash commands
- [Voice Gateway](skills/voice-gateway/SKILL.md) — installation, configuration, architecture

## License

[MIT](LICENSE.md)
