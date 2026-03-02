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

The Mac Mini runs the Intercom bridge sidecar as a local process that exposes an HTTP/WS API for the Python gateway to send/receive P2P messages. Each device (Jetson, etc.) runs a lightweight Intercom peer that initiates the connection to the gateway's peer via `intercom_key`.

```
Mac Mini (gateway)
  └── Intercom peer (hub)          <- runs alongside gateway process
        ├── <- Robot 1 connects in  <- Jetson initiates connection
        ├── <- Robot 2 connects in
        └── <- Glasses connects in  (future)
```

Why gateway-centric:
1. One place to manage all P2P channels (not N bridge sidecars on N devices)
2. Gateway is always-on; devices may reboot, move, lose connectivity
3. Phone relay routes through Mac Mini — devices connect to one stable hub
4. Consistent with separation of concerns: gateway orchestrates, devices execute

## Configuration

Robot config lives in the `robot` section of `config.json`:

### Shared robot config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `robot.enabled` | bool | `false` | Is robot endpoint active |
| `robot.model` | str | `"unitree_g1"` | Robot model ID |
| `robot.transport` | str | `"intercom"` | Transport type |
| `robot.intercom_key` | str | `""` | Intercom P2P public key for pairing |
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

### Future (when Intercom transport is built)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/robot/status` | Robot status (battery, pose, connectivity, current task) |
| `POST` | `/api/robot/command` | Send command to robot |
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
| Robot shows "Not connected" | Intercom transport not running or key mismatch | Check `robot.intercom_key` matches device; verify Intercom bridge sidecar is running |
| Commands return "not connected" | No active robot connection | Enable robot endpoint (`robot.enabled: true`); check Intercom bridge |
| Capability error on command | Command requires capability robot doesn't have | Check `GET /api/config/robot` for capabilities list |
| Config update rejected | Security key (`intercom_key`, `enable_low_level`) sent by agent | Security-sensitive keys can only be set from control center, not agents |
