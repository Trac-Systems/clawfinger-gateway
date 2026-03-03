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
Mac Mini (gateway)                          Robot (Jetson Orin)
  app.py                                      Intercom peer
  endpoints/robot/                            unitree_sdk2_python
  Intercom bridge sidecar  <-- P2P -->        Motor control / sensors
```

- **Gateway** = System 2 brain (LLM reasoning, task decomposition)
- **Robot** = System 1 body (RL policies, motor control, sensors)
- **Transport**: Intercom P2P — see https://github.com/Trac-Systems/intercom/

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

REST endpoints for accessing robot cameras and microphones. Camera sources and mic sources are defined in the robot model defaults (e.g., G1 has `head_rgb`, `head_depth`, `head_stereo`; mic `head_mic`).

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
