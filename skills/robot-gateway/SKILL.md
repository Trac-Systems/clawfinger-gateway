---
name: robot-gateway
description: Clawfinger robot endpoint — connecting humanoid robots (starting with Unitree G1) to the gateway via Intercom P2P transport. Covers robot config, capability registry, command dispatch, and troubleshooting.
metadata:
  openclaw:
    emoji: "\U0001F916"
    skillKey: robot-gateway
---

# Robot Gateway — Clawfinger Robot Endpoint

> **This skill covers the robot endpoint.** For phone endpoint, see [`phone-gateway`](../phone-gateway/SKILL.md).

## Overview

Robot endpoint for the Clawfinger gateway. Connects to humanoid robots (starting with Unitree G1) via Intercom P2P transport. The gateway acts as the "brain" (LLM reasoning, task decomposition) while the robot's onboard computer (Jetson Orin) acts as the "body" (RL policies, motor control, sensors).

## Supported Models

| Model | ID | Capabilities |
|-------|-----|-------------|
| Unitree G1 (EDU, Pro) | `unitree_g1` | locomotion, posture, manipulation, gesture, vision, audio, dexterous_hands |

## Architecture

```
Mac Mini (gateway)                          Robot (Jetson Orin) OR Isaac Sim adapter (RTX)
  app.py                                      Intercom peer / WebSocket client
  endpoints/robot/                            unitree_sdk2_python / isaac_backend.py
  Intercom bridge sidecar  <-- P2P -->        Motor control / sensors / sim physics
  Qwen3.5-4B VLM (mlx-vlm)                   YOLO-World + CLIP (on-device)
```

- **Gateway** = System 2 brain (LLM reasoning, task decomposition, visual confirmation)
- **Robot/Sim** = System 1 body (RL policies, motor control, sensors, local detection)
- **Transport**: Intercom P2P (hardware) or WebSocket (sim adapter)
- **Two transports**: `robot.transport = "intercom"` for real robot, `"websocket"` for sim adapter

### Intercom Topology (gateway-centric hub)

The Mac Mini runs the Intercom Pear process as a child of the gateway. The Python gateway connects to its SC-Bridge WebSocket on `:49222` to send/receive P2P messages. Each device (Jetson, etc.) runs a lightweight Intercom peer that initiates the connection to the gateway's peer via `intercom_key`.

```
Mac Mini (gateway)
  app.py → IntercomBridge (WS client)
         ↕ ws://127.0.0.1:49222
  IntercomProcess (Pear child process)
         ↕ P2P (HyperDHT)
Jetson (robot)
  Intercom peer → unitree_sdk2_python
```

Why gateway-centric:
1. One place to manage all P2P channels (not N bridge sidecars on N devices)
2. Gateway is always-on; devices may reboot, move, lose connectivity
3. Phone relay routes through Mac Mini — devices connect to one stable hub
4. Consistent with separation of concerns: gateway orchestrates, devices execute

### Intercom Setup (one-time)

#### Step 1: Install on Mac Mini

```bash
cd gateway
bin/intercom-setup.sh
```

- Checks Node 22.x + Pear runtime
- Clones Intercom into `gateway/intercom/` (gitignored)
- `npm install` inside it
- Runs Intercom briefly to trigger ED25519 keypair generation
- Prints the gateway's public key hex

#### Step 2: Install on Jetson

Same script on Jetson side (different peer-store-name: `robot`).

#### Step 3: Exchange keys (admin carries these manually)

Admin copies:
- Gateway pubkey → Jetson config (as `inviterKeys`)
- Robot pubkey → Mac Mini config (as `robot.intercom_key`)

#### Step 4: Pair

```bash
cd gateway
bin/intercom-pair.sh <robot-pubkey-hex>
```

- Creates channel `clawfinger-robot-g1`
- Generates SC-Bridge token, saves to `tmp/.sc_bridge_token`
- Writes pairing data to `intercom/stores/gateway/pairing.json`
- Updates `config.json` (`robot.intercom_key` and `robot.intercom_channel`)

After pairing, gateway startup handles everything automatically when `robot.enabled: true`.

### Transport Architecture

```
gateway/transport/
  intercom_manager.py    # Spawns/monitors Pear process (auto-restart on crash)
  intercom_bridge.py     # WS client to SC-Bridge on :49222
```

**IntercomProcess**: Spawns `pear run ./intercom/ --sc-bridge 1 ...` as a child process. Auto-restarts with exponential backoff on crash. Writes PID to `tmp/intercom.pid`.

**IntercomBridge**: WebSocket client to SC-Bridge. Handles:
- Auth + channel join on connect
- Heartbeat sending every `heartbeat_interval`s
- Disconnect detection with `disconnect_timeout` + `disconnect_debounce`
- Request/response correlation via `_req_id`/`_resp_id`
- Auto-reconnect with exponential backoff (1s → 60s)

### Message Protocol (on the private channel)

**Gateway → Robot:**
```json
{"type": "robot_command", "command": "walk", "params": {"speed": 0.3}, "_req_id": 42}
{"type": "robot_command", "command": "stop"}
{"type": "heartbeat", "ts": 1709337600.123}
```

**Robot → Gateway:**
```json
{"type": "robot_response", "_resp_id": 42, "ok": true, "detail": "walking"}
{"type": "heartbeat_ack", "ts": 1709337600.123, "battery": 85, "pose": "standing"}
{"type": "robot_event", "event": "task_completed", "task_id": "abc"}
```

Fire-and-forget commands (`stop`, `safety_stop`) don't use `_req_id` — they're sent without waiting for a response.

### Verifying Pairing

```bash
curl -s http://127.0.0.1:8996/api/config/robot | jq '{
  enabled, connected, transport, transport_connected, intercom_running, intercom_channel
}'
```

## Configuration

Robot config lives in the `robot` section of `config.json`:

### Shared robot config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `robot.enabled` | bool | `false` | Is robot endpoint active |
| `robot.model` | str | `"unitree_g1"` | Robot model ID |
| `robot.transport` | str | `"intercom"` | Transport type |
| `robot.intercom_key` | str | `""` | Intercom P2P public key for pairing |
| `robot.sc_bridge_port` | int | `49222` | SC-Bridge WebSocket port |
| `robot.intercom_channel` | str | `"clawfinger-robot-g1"` | Private channel name |
| `robot.pear_path` | str | `""` | Path to pear binary (auto-discovered if empty) |
| `robot.heartbeat_interval` | int | `5` | Seconds between heartbeats |
| `robot.disconnect_timeout` | int | `15` | Seconds before declaring disconnected |
| `robot.disconnect_debounce` | int | `3` | Seconds before confirming disconnect |
| `robot.safety_stop_on_disconnect` | bool | `true` | Enable safety stop on disconnect |
| `robot.offline_mode` | str | `"complete_task"` | `"complete_task"` or `"safety_stop"` |
| `robot.voice` | str | `"am_adam"` | TTS voice for robot speaker |
| `robot.voice_lang` | str | `"en"` | TTS language for robot |

### G1-specific config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `robot.unitree_g1.jetson_ip` | str | `"192.168.123.164"` | Jetson Orin IP |
| `robot.unitree_g1.locomotion_ip` | str | `"192.168.123.161"` | Locomotion computer IP |
| `robot.unitree_g1.dds_domain` | int | `0` | CycloneDDS domain ID |
| `robot.unitree_g1.max_speed` | float | `0.5` | Max walk speed m/s (safety limit) |
| `robot.unitree_g1.enable_hands` | bool | `true` | Enable dexterous hand control |
| `robot.unitree_g1.enable_low_level` | bool | `false` | Enable low-level joint control (dangerous) |
| `robot.unitree_g1.wifi_networks` | list | `[]` | Pre-configured WiFi SSIDs |

### Perception config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `robot.camera.width` | int | `640` | Default snapshot/stream width |
| `robot.camera.height` | int | `480` | Default snapshot/stream height |
| `robot.camera.quality` | int | `50` | JPEG quality (1-100) |
| `robot.camera.stream_fps` | int | `5` | Default video stream FPS |
| `robot.audio_monitor.sample_rate` | int | `16000` | Audio sample rate (Hz) |
| `robot.audio_monitor.channels` | int | `1` | Audio channels |
| `robot.audio_monitor.chunk_ms` | int | `100` | Audio chunk duration (ms) |

### Intercom transport limits

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `robot.intercom_limits.max_message_kb` | int | `256` | Max message size (KB). Raised for video frame streaming. |
| `robot.intercom_limits.pow` | int | `0` | Proof-of-work difficulty (0 = disabled for private robot channel) |
| `robot.intercom_limits.rate_limit` | int | `0` | Messages/sec limit (0 = unlimited) |

## API Endpoints

### Config

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/config/robot` | Read robot config (shared + model-specific defaults + capabilities) |
| `POST` | `/api/config/robot` | Update robot config (security keys blocked from agents) |

### Commands (via Agent WS)

Robot commands are sent through the `/api/agent/ws` WebSocket:

| Send | Fields | Description |
|------|--------|-------------|
| `robot_command` | `command: {type, params}` | Send command to robot via Intercom |
| `robot_status` | — | Get robot config + connection state |

| Receive | Fields | Description |
|---------|--------|-------------|
| `robot.command.ack` | `ok`, `detail`, `error` | Command result |
| `robot.status` | config + connection info | Robot state |

### Future REST endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/robot/task` | Submit async task |
| `GET` | `/api/robot/task/{id}` | Check task status |
| `DELETE` | `/api/robot/task/{id}` | Cancel task |

## Capability Registry

Each robot model registers its capabilities. The gateway dispatches commands through the capability interface — it doesn't need to know the specific model. The control center robot tab adapts panels based on reported capabilities.

**G1 capabilities**: `locomotion`, `posture`, `manipulation`, `gesture`, `vision`, `audio`, `dexterous_hands`

Commands are validated against capabilities before dispatch. If a robot doesn't support a capability (e.g., a wheeled robot can't do `gesture`), the command returns an error.

## Networking

Intercom P2P (HyperDHT) handles connectivity regardless of network topology. Once paired via `intercom_key`, the channel reconnects automatically.

### Local operation (same LAN)
- Robot on Ethernet (192.168.123.x) or WiFi
- Low latency, full bandwidth
- Control center shows: "Local"

### Remote operation (mobile hotspot)
- Owner places mobile hotspot near robot
- Jetson connects to pre-configured WiFi SSID
- Intercom hole-punches through NAT to Mac Mini
- Higher latency, bandwidth-adaptive
- Control center shows: "Remote" + latency indicator

### WiFi pre-configuration
- Pre-add hotspot SSIDs on Jetson via nmcli (or from control center)
- `robot.unitree_g1.wifi_networks` stores known SSIDs
- Jetson auto-connects when Ethernet unavailable

### Connection quality & degradation
- `disconnect_debounce` (default 3s) prevents flapping on lossy mobile connections
- `offline_mode`: `"complete_task"` = finish current task then idle, `"safety_stop"` = stop immediately
- Jetson-side bandwidth adaptation (not gateway-side)

## G1-Specific Notes

- Network: 192.168.123.0/24 (Jetson .164, locomotion .161)
- SSH: `unitree@192.168.123.164`
- WiFi disabled by default — enable via rfkill
- Python SDK: `unitree_sdk2_python` (CycloneDDS 0.10.2)
- Mic not in SDK — use ALSA directly
- Camera via pyrealsense2, not DDS

## Isaac Sim Integration (Development & Testing)

The sim adapter bridges Isaac Sim (Omniverse) to the gateway, enabling full robot development and testing without physical hardware. The sim adapter runs on an RTX GPU machine and connects to the gateway via WebSocket.

### Architecture

```
Mac Mini (gateway :8996)                 RTX Machine (sim adapter)
  app.py                                   adapter.py
  robot controller                         isaac_backend.py (physics, locomotion)
  VLM confirmation                         detection.py (YOLO-World)
  spatial memory                           clip_embedder.py (CLIP observations)
         <-- WebSocket (reverse tunnel) -->
```

### Setup on RTX Machine

**Prerequisites**: NVIDIA RTX GPU, Ubuntu, CUDA 12+, conda

```bash
# 1. Create conda environment
conda create -n g1sim python=3.11
conda activate g1sim

# 2. Install Isaac Sim + Isaac Lab
pip install isaacsim==5.1.0 isaaclab==0.54.3

# 3. Accept EULA (required for headless operation)
export OMNI_KIT_ACCEPT_EULA=Y
echo "yes" > $(python -c "import isaacsim; print(isaacsim.__path__[0])")/kit/EULA_ACCEPTED

# 4. Install adapter dependencies
pip install websockets torch onnxruntime-gpu ultralytics open-clip-torch

# 5. Install Unitree SDK (for joint configs)
cd /path/to/unitree_sdk2_python && pip install -e .

# 6. Clone/copy sim-adapter files from gateway
# Files: adapter.py, isaac_backend.py, detection.py, clip_embedder.py
# Assets: assets/g1_locomotion.onnx (locomotion policy)
```

### Environment Variables

```bash
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=Y
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export UNITREE_SIM_PATH=/path/to/unitree_sim_isaaclab
export PROJECT_ROOT=$UNITREE_SIM_PATH
```

### Connecting to Gateway

The sim adapter connects to the gateway via WebSocket. For remote machines (e.g., ngrok tunneled), use a reverse SSH tunnel:

```bash
# On RTX machine: create reverse tunnel so adapter can reach gateway
ssh -N -R 18996:127.0.0.1:8996 user@gateway-host &

# Launch adapter (headless, with cameras)
xvfb-run -a python adapter.py \
  --headless --enable_cameras \
  --gateway ws://127.0.0.1:18996/ws/robot \
  --bearer localdev \
  --obs-interval 5.0
```

### Gateway Configuration for Sim

Set `robot.transport` to `"websocket"` in `config.json`:

```json
{
  "robot": {
    "enabled": true,
    "transport": "websocket"
  }
}
```

The gateway automatically accepts WebSocket connections on `/ws/robot` when transport is `"websocket"`.

### Sim Adapter Components

| File | Purpose |
|------|---------|
| `sim-adapter/adapter.py` | Main adapter: WS client, command handler, camera/audio streaming |
| `sim-adapter/isaac_backend.py` | Isaac Sim physics: locomotion controller, env stepping, camera capture |
| `sim-adapter/detection.py` | YOLO-World open-vocabulary object detection on sim camera frames |
| `sim-adapter/clip_embedder.py` | CLIP ViT-B-32 embedding for spatial memory observations |
| `sim-adapter/assets/g1_locomotion.onnx` | Trained velocity locomotion policy (337KB, bidirectional) |

### Locomotion Policy

The ONNX policy runs at 50Hz in the sim adapter's main thread:

- **Input**: [1, 123] — angular velocity, gravity, command [vx, vy, vyaw], joint positions/velocities, phase clock
- **Output**: [1, 37] — joint position targets for all 37 DOF
- **Training range**: `vx ∈ [-0.5, 1.0]`, `vy ∈ [-0.5, 0.5]`, `vyaw ∈ [-1.0, 1.0]`
- **Forward + backward walking**, turning, lateral movement

### Training New Policies

```bash
# Train velocity locomotion in Isaac Lab
cd /path/to/IsaacLab
xvfb-run -a python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 \
  --num_envs 8192 \
  --headless \
  --max_iterations 2000

# Export to ONNX (after training completes)
python export_onnx.py \
  /path/to/logs/rsl_rl/g1_flat/<timestamp>/model_1999.pt \
  sim-adapter/assets/g1_locomotion.onnx
```

### Threading Model (Critical)

Isaac Sim requires specific threading:
- **Main thread**: `env.step()` / physics stepping (Kit kernel requirement)
- **Background thread**: asyncio event loop (WebSocket, camera streaming, command handling)
- **Never**: `run_in_executor()` for sim stepping — deadlocks

### Troubleshooting (Sim)

| Symptom | Fix |
|---------|-----|
| `vkCreateInstance` error | Need `xvfb-run` — no display attached |
| Adapter connects but no robot movement | Check `g1_locomotion.onnx` exists in `assets/` |
| Camera frames blank | Ensure `--enable_cameras` flag passed |
| WebSocket disconnects | Check reverse tunnel is active; verify bearer token |
| `RuntimeError: Cannot run the event loop` | Physics must run in main thread, not asyncio |

---

## LLM (Qwen3.5-4B Multimodal VLM)

The gateway uses a single multimodal VLM for both phone and robot: **Qwen3.5-4B** at 4-bit quantization via `mlx-vlm`.

### Model Details

| Property | Value |
|----------|-------|
| Model | [`TracNetwork/Qwen3.5-4B-4bit-mlx`](https://huggingface.co/TracNetwork/Qwen3.5-4B-4bit-mlx) |
| Source | `Qwen/Qwen3.5-4B` (4-bit quantized via `mlx_vlm.convert`) |
| Size | ~2.9 GB on disk |
| Architecture | Gated Delta Networks + sparse MoE |
| Modalities | Text + images + video |
| Context window | 262K tokens |
| Inference | `mlx-vlm` on Apple Silicon (MPS) |

### Configuration

Global LLM config applies to both phone and robot:

```json
{
  "llm": {
    "model": ".models/Qwen3.5-4B-4bit",
    "multimodal": true,
    "max_tokens": 80,
    "temperature": 0.25
  }
}
```

Robot can override specific LLM settings via `robot.llm`:

```json
{
  "robot": {
    "llm": {
      "max_tokens": 200,
      "temperature": 0.3,
      "system_prompt": "You are Robert, a helpful robot assistant..."
    }
  }
}
```

The `config.robot_llm()` function merges `robot.llm` over global `llm` — robot overrides take precedence, unset keys fall back to global.

### Multimodal Usage

When `llm.multimodal: true`, the VLM backend handles both text-only and image+text inputs:

- **Text-only** (phone calls): Same as before, VLM handles plain text
- **Image+text** (robot vision): Messages include `image_url` content blocks with base64 JPEG data
- **Visual confirmation**: Cropped detection images sent to VLM for YES/NO/UNSURE classification

### Key Technical Notes

- **Thinking mode**: Must use `processor.tokenizer.apply_chat_template(enable_thinking=False)` — `mlx_vlm`'s own `apply_chat_template` doesn't properly pass this kwarg
- **GenerationResult**: Both `mlx_lm` and `mlx_vlm` `generate()` may return `GenerationResult` objects, not strings. Use `getattr(result, "text", None)`
- **Download**: `huggingface_hub.snapshot_download('TracNetwork/Qwen3.5-4B-4bit-mlx', local_dir='.models/Qwen3.5-4B-4bit')`
- **Convert from source**: `python -m mlx_vlm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 4 --mlx-path .models/Qwen3.5-4B-4bit`

---

## Visual Confirmation Pipeline (YOLO-World + VLM)

When the robot needs to find or identify objects, a two-stage pipeline runs:

### Flow

```
1. Gateway sends detect_request → Robot/Sim
2. Robot/Sim runs YOLO-World (open-vocabulary detection) on camera frame
3. Returns detections: [{bbox, confidence, class, cropped_b64}, ...]
4. Gateway sends each crop to VLM: "Is this a '{target}'? YES/NO/UNSURE"
5. VLM responds:
   - YES → object confirmed, proceed with action
   - NO  → skip this detection, try next
   - UNSURE → ask user via voice: "I found something, is this what you want?"
6. User confirms/denies via voice → robot acts accordingly
```

### Intercom Protocol

| Direction | Type | Fields |
|-----------|------|--------|
| Gateway → Robot/Sim | `detect_request` | `classes: ["keys", "phone"]`, `confidence: 0.3`, `max_detections: 5` |
| Robot/Sim → Gateway | `detection_result` | `detections: [{class, confidence, bbox, cropped_b64}]` |

### Controller Actions

- `detect_object`: Runs full pipeline — YOLO detect → VLM confirm → report results
- `confirm_visual`: Direct VLM image Q&A — send image + question, get answer

---

## CLIP Observation Pipeline

The robot/sim continuously embeds camera frames with CLIP and sends 512-dim embeddings to the gateway for spatial memory storage.

### Flow

```
Robot/Sim: camera frame → CLIP ViT-B-32 → 512-dim embedding
  → Intercom: {type: "observation", embedding: [512 floats], metadata: {...}}
Gateway: stores embedding + metadata directly in ChromaDB (no CLIP on gateway at runtime)
```

### Embedding Split

| Location | CLIP Runtime | Purpose | Rate |
|----------|-------------|---------|------|
| Jetson (real robot) | TensorRT, CUDA 11.4 | Runtime observations | 5 fps |
| RTX machine (sim) | PyTorch, CUDA 12 | Runtime observations | configurable via `--obs-interval` |
| Mac Mini (gateway) | open_clip, MPS | Teaching (reference photos) + text queries | On-demand only |

### Observation Metadata Schema

Each observation stores 7 query dimensions:

```python
{
    "entity_type": "person",         # person | object | scene
    "entity_id": "abc123",           # matched entity ID
    "entity_name": "Alex",           # matched entity name
    "room": "kitchen",               # room zone
    "description": "Person near counter",
    "labels": "person,counter",      # detection labels
    "world_x": 2.3, "world_y": 1.4, "world_z": 0.9,  # object position (meters)
    "robot_x": 1.2, "robot_y": 3.4, "robot_theta": 0.5,  # robot pose when observed
    "timestamp": 1709500000.0,
    "source": "head_rgb",
    "depth_available": false,
}
```

### Robot Voice (German TTS)

The gateway synthesizes robot speech audio and sends WAV bytes to the robot:

- `robot.voice_lang = "en"` → Kokoro TTS (English)
- `robot.voice_lang = "de"` → Piper TTS (German, thorsten-high voice)

The robot just plays audio — no TTS engine needed on-device.

```
User speaks → Robot mic → Gateway ASR → LLM → Gateway TTS → WAV bytes → Robot speaker
```

The `voice_pipeline.synthesize_robot()` function routes by `robot.voice_lang` and sends base64 WAV via the `robot_speak` message type.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Robot shows "Not connected" | Intercom transport not running or key mismatch | Check `robot.intercom_key` matches device; verify Intercom process is running (`intercom_running: true` in `/api/config/robot`) |
| Commands return "not connected" | No active robot connection | Enable robot endpoint (`robot.enabled: true`); check Intercom bridge |
| `transport_connected: false` | SC-Bridge WS connection failed | Check Intercom process is running; check `sc_bridge_port` setting |
| `intercom_running: false` | Pear binary not found or Intercom not installed | Run `bin/intercom-setup.sh`; check `robot.pear_path` config |
| SC-Bridge connection refused | Port mismatch or Intercom crashed | Check `tmp/intercom.pid`; Intercom auto-restarts with exponential backoff |
| Pairing fails | Missing keypair or wrong pubkey | Re-run `bin/intercom-setup.sh` to regenerate keypair; verify key hex |
| Capability error on command | Command requires capability robot doesn't have | Check `GET /api/config/robot` for capabilities list |
| Config update rejected | Security key (`intercom_key`, `enable_low_level`) sent by agent | Security-sensitive keys can only be set from control center, not agents |

---

## Voice I/O (via Intercom)

### Protocol

| Message type | Direction | Fields | Description |
|---|---|---|---|
| `robot_voice_input` | Robot → Gateway | `transcript`, `source`, `wake_word_detected` | Voice input from robot mic |
| `robot_speak` | Gateway → Robot | `text`, `voice`, `speed` | TTS playback on robot speaker |

### Wake Word Gating

The robot only reacts to speech directed at it. Configurable per deployment:

- `robot.wake_word` (default: `"Robert"`) — the robot's name
- `robot.wake_phrases` (default: `["hey {name}", "hi {name}", "ok {name}", "listen {name}", "{name}"]`) — activation phrase patterns
- `robot.wake_word_timeout` (default: `30`) — seconds before conversational mode expires

**Conversational mode** (wake word not required) activates when:
- A project is running
- Robot asked a question and is waiting for reply
- Last interaction within `wake_word_timeout` seconds

**Safety override**: `"{wake_word} stop"` always works regardless of mode.

### Voice Flow

```
User speaks → Robot mic (Jetson) → OpenWakeWord → ASR → Intercom P2P
    → Gateway controller → check_wake_word() → LLM → Intercom → robot speaker
```

---

## Robot Skills

### Directory Structure

```
skills/robot/
  household_objects/
    skill.json              # execution_mode: "slow"
    common.md               # knowledge topic
  precision_grasp/
    skill.json              # execution_mode: "fast", status: "coming_soon"
```

### skill.json Format

**Slow-path** (LLM knowledge):
```json
{
  "name": "household_objects",
  "description": "Common household objects — visual descriptions, typical locations, search strategies",
  "execution_mode": "slow",
  "topics": ["common"],
  "required_capabilities": ["vision", "locomotion"]
}
```

**Fast-path** (trained policy, coming soon):
```json
{
  "name": "precision_grasp",
  "description": "Trained RL policy for precision object grasping",
  "execution_mode": "fast",
  "status": "coming_soon",
  "required_capabilities": ["manipulation", "vision"],
  "policy_format": "onnx",
  "notes": "Runs entirely on Jetson. No LLM in the loop."
}
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/robot/skills` | List all robot skills (slow + fast) |
| `GET` | `/api/robot/skills/{name}/{topic}` | Get topic content for a slow-path skill |

---

## Projects (Voice-Driven Task Execution)

A project is a multi-step autonomous execution triggered by voice. The user says something like "Robert, find my keys" and the LLM decomposes it into steps using primitives and knowledge.

### LLM Output Format

```json
{
  "say": "I'll look around for your keys.",
  "gesture": "nod",
  "do": {"action": "look"},
  "continue": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `say` | string/null | Text spoken on robot speaker via Intercom |
| `gesture` | string/null | Gesture command: nod, shake_head, wave, point, thumbs_up |
| `do` | object/null | Robot primitive to execute |
| `continue` | bool | true = auto-continue; false = wait for user |

### Autonomous Loop Lifecycle

1. User speaks → LLM outputs first step with `continue: true`
2. Background loop executes actions, feeds results to LLM
3. User can interrupt via voice at any time (pauses loop)
4. LLM outputs `done` or `abort` → project ends

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/robot/project` | Current project state or `{status: "idle"}` |
| `POST` | `/api/robot/project/cancel` | Cancel running project |

### Event Bus Events

| Event | When |
|-------|------|
| `robot.project.started` | LLM initiates a project from voice |
| `robot.project.step` | Each autonomous action dispatched |
| `robot.project.voice_interrupt` | User spoke during execution |
| `robot.project.completed` | LLM outputs done |
| `robot.project.failed` | Timeout, disconnect, abort, max steps |
| `robot.project.cancelled` | User said "stop" or API cancel |
| `robot.project.ask_user` | Robot asks user a question |
| `robot.skill.loaded` | Skill topic loaded into context |

---

## Robot Takeover (Agent Control)

Agents (OpenClaw) can take full control of the robot endpoint. Text-only: agents send/receive text, gateway bridges to voice via Intercom.

### Protocol

| Message | Direction | Description |
|---|---|---|
| `robot_takeover` | Agent → Gateway | Agent takes control |
| `robot_takeover.ack` | Gateway → Agent | Confirmed |
| `robot_turn.request` | Gateway → Agent | User spoke — transcript forwarded |
| reply with `request_id` | Agent → Gateway | Text reply (spoken on robot) + optional commands |
| `robot_release` | Agent → Gateway | Agent releases control |
| `robot_release.ack` | Gateway → Agent | Confirmed |

**Safety override**: `"{wake_word} stop"` always bypasses agent takeover.

---

## Perception (Camera + Audio)

REST endpoints for accessing robot cameras and microphones. Camera sources and mic sources are defined in the robot model defaults (e.g., G1 has `head_rgb`, `head_depth`; mic `mic_array`).

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/robot/perception` | List available cameras + mics from model defaults |
| `POST` | `/api/robot/camera/snapshot` | Capture single frame |
| `POST` | `/api/robot/camera/describe` | VLM scene description |
| `GET` | `/api/robot/camera/stream?source=...&token=...` | MJPEG video stream |
| `POST` | `/api/robot/camera/stream/start` | Start video stream |
| `POST` | `/api/robot/camera/stream/stop` | Stop video stream |
| `POST` | `/api/robot/audio/monitor/start` | Start mic audio monitoring |
| `POST` | `/api/robot/audio/monitor/stop` | Stop mic audio monitoring |

**Snapshot** body: `{source?, width?, height?, quality?}` — returns JPEG image data.

**Describe** body: `{source?, prompt?}` — captures a frame and runs it through the VLM. Returns text description.

**Stream start** body: `{source?, fps?, width?, height?, quality?}` — begins MJPEG streaming. Connect to the `GET` stream endpoint with the returned token.

**Stream stop** body: `{source?}` — stops active video stream.

**Audio monitor start** body: `{source?, sample_rate?, channels?, chunk_ms?}` — starts mic capture on robot.

**Audio monitor stop** body: `{source?}` — stops mic capture.

### Intercom Protocol (Perception)

**Gateway → Robot:**

| Message type | Pattern | Fields |
|---|---|---|
| `camera_snapshot` | req/resp | `source`, `width`, `height`, `quality` |
| `camera_stream_start` | fire-and-forget | `source`, `fps`, `width`, `height`, `quality` |
| `camera_stream_stop` | fire-and-forget | `source` |
| `camera_describe` | req/resp | `source`, `prompt` |
| `audio_monitor_start` | fire-and-forget | `source`, `sample_rate`, `channels`, `chunk_ms` |
| `audio_monitor_stop` | fire-and-forget | `source` |

**Robot → Gateway:**

| Message type | Fields | Description |
|---|---|---|
| `camera_frame` | `source`, `image_base64`, `width`, `height`, `seq`, `ts` | Single camera frame (response to snapshot or stream chunk) |
| `audio_chunk` | `source`, `audio_base64`, `sample_rate`, `channels`, `seq`, `ts` | Mic audio chunk during monitoring |

---

## Spatial Memory

The gateway includes a persistent spatial memory subsystem for the robot. It stores what the robot has seen, where objects and persons were last observed, and the layout of known rooms/zones.

### REST Endpoints (19 total)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/robot/memory/stats` | DB statistics: entity counts, observation counts, DB size |
| `GET` | `/api/robot/memory/persons` | List all known persons |
| `POST` | `/api/robot/memory/persons` | Teach a person (name, description, optional photos) |
| `GET` | `/api/robot/memory/persons/{id}` | Get person record by ID |
| `PUT` | `/api/robot/memory/persons/{id}` | Update person record |
| `DELETE` | `/api/robot/memory/persons/{id}` | Delete person record |
| `GET` | `/api/robot/memory/objects` | List all known objects |
| `POST` | `/api/robot/memory/objects` | Teach an object (name, description, optional photos) |
| `GET` | `/api/robot/memory/objects/{id}` | Get object record by ID |
| `PUT` | `/api/robot/memory/objects/{id}` | Update object record |
| `DELETE` | `/api/robot/memory/objects/{id}` | Delete object record |
| `GET` | `/api/robot/memory/rooms` | List all known rooms/zones |
| `POST` | `/api/robot/memory/rooms` | Define a room/zone (name, description) |
| `GET` | `/api/robot/memory/rooms/{id}` | Get room record by ID |
| `PUT` | `/api/robot/memory/rooms/{id}` | Update room record |
| `DELETE` | `/api/robot/memory/rooms/{id}` | Delete room record |
| `POST` | `/api/robot/memory/observations` | Record a new observation (entity seen at location/time) |
| `GET` | `/api/robot/memory/observations` | List observations with optional filters (entity_id, room, type, since) |
| `POST` | `/api/robot/memory/query` | Natural language query with filters (room, type, time range) |

### Query Types

| Query type | Body | Description |
|------------|------|-------------|
| `text` | `{"type": "text", "text": "where are my keys?"}` | Natural language search — returns matching entities + last seen location |
| `entity` | `{"type": "entity", "entity_id": "..."}` | All observations for a specific entity |
| `room` | `{"type": "room", "room": "kitchen"}` | All entities last seen in a room |
| `recent` | `{"type": "recent", "since": "2026-03-01T00:00:00Z"}` | All observations since timestamp |

### Time-Aware Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/robot/memory/last_seen` | Most recent observation matching `entity_name`, `entity_type`, and/or `room`. Returns `time_ago` field. |
| `POST` | `/api/robot/memory/query` | Accepts `time_filter` with natural expressions: `"last hour"`, `"today"`, `"yesterday"`, `"last 3 days"` |
| `GET` | `/api/robot/memory/stats` | Now includes `oldest_ago`, `newest_ago`, `time_span` temporal range |

All query results include a `time_ago` field (e.g. "5 min ago", "yesterday") on every observation that has a `timestamp`.

### Usage Notes

- The robot writes observations automatically as it navigates and uses VLM scene descriptions
- Agents can also write observations directly via the REST API
- Reference photos are stored as base64 in the DB and used for visual re-identification on the robot
- The `clawfinger_memory_*` OpenClaw tools wrap these endpoints — see [openclaw-clawfinger/SKILL.md](../openclaw-clawfinger/SKILL.md)

---

## Time Awareness

The robot system is fully time-aware. The `time_utils.py` module provides all time functions used across the system.

### Robot System Prompt

The robot LLM sees current time and project elapsed time in every prompt:

```
## Current State
Current time: Tuesday, 2026-03-04 14:32 CET (afternoon)
Project elapsed: 1m 12s
Current project: Find my keys

Step 1/3: ✓ Walk to kitchen [18s]
Step 2/3: → Search counter [running 12s]
Step 3/3: ○ Report findings
```

Step durations show how long each completed step took and how long the active step has been running.

### Spatial Memory Temporal Features

- **`time_ago`** on all query results: `"5 min ago"`, `"yesterday"`, `"3 days ago"`
- **`POST /api/robot/memory/last_seen`**: Most recent observation for an entity
- **`time_filter`** on `POST /api/robot/memory/query`: Natural expressions like `"last hour"`, `"today"`, `"yesterday"`, `"last 3 days"`
- **Enhanced stats**: `oldest_ago`, `newest_ago`, `time_span` in `GET /api/robot/memory/stats`

### Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `timezone` | string | `"Europe/Berlin"` | Timezone for time display and time-of-day calculation |

---

## Safety Limits

Gateway-side command validator runs before every command dispatch:

- **Speed clamping**: `robot.safety.max_speed` (default: 0.3 m/s)
- **Force clamping**: `robot.safety.max_grasp_force` (default: 0.8)
- **Reach envelope**: `robot.safety.max_reach_m` (default: 0.6m)
- **Command blocking**: `robot.safety.blocked_commands` — blocked types list

The G1's own hardware safety (emergency stop, joint limits, LiDAR, self-collision avoidance) is NOT duplicated.

### Config

```json
{
  "robot": {
    "wake_word": "Robert",
    "wake_phrases": ["hey {name}", "hi {name}", "ok {name}", "listen {name}", "{name}"],
    "wake_word_timeout": 30,
    "wake_word_model": "",
    "voice_speed": 1.0,
    "safety": {
      "max_speed": 0.3,
      "max_grasp_force": 0.8,
      "max_reach_m": 0.6,
      "blocked_commands": []
    }
  }
}
