---
name: control-center
description: Clawfinger gateway control center UI — browser-based dashboard for managing calls, TTS/LLM config, robot control, and spatial memory. Served by the gateway on the root path.
metadata:
  openclaw:
    emoji: "\U0001F5A5"
    skillKey: control-center
---

# Control Center — Gateway Browser UI

The Clawfinger gateway serves a browser-based control center at `http://127.0.0.1:8996/`. It provides real-time management of all gateway subsystems without needing the CLI or raw API calls.

## Tabs

Two tabs at the top:

- **Phone**: Live call monitoring, call control, instructions, session log, agent panel, call policy, security, TTS settings, LLM settings, system status
- **Robot**: Robot connection, robot status, task control, movement control, perception, robot audio, sensor readout, robot settings, robot LLM, spatial memory

## Phone Tab Panels

- **Live Monitor**: Real-time call transcript + turn count badge
- **Call Control**: Dial, hangup, inject TTS, call status display
- **Instructions**: Set global/session/turn LLM instructions
- **Session Log**: Past sessions with history viewer
- **Agent Panel**: Agent takeover status + inject button
- **Call Policy**: Auto-answer, greetings, max duration, caller filtering
- **Security**: Passphrase auth, allowlist/blocklist, unknown caller policy
- **TTS Settings**: Voice selector, speed, language (EN/DE), Piper params when German
- **LLM Settings**: Model, temperature, max tokens, system prompt
- **System Status**: Gateway health, uptime, active sessions, LLM/ASR status

## Robot Tab Panels

### Robot Connection Panel

Shows transport type (WebSocket/Intercom), connection status badge, and connected robot model/capabilities.

### Robot Status Panel

Displays current robot state: battery, pose, active project, error messages.

### Task Control Panel

Voice-driven task input — type a natural language command (e.g. "walk to the kitchen") and the robot executes it as a project. Shows inline project progress with step tracking when a structured plan is active. Badge shows step count or "running" status.

### Movement Control Panel

Manual movement buttons: forward, backward, left turn, right turn, stop. Speed slider. Direct dispatch via robot_command.

### Robot Settings Panel

Configures robot behavior, voice, wake word, and safety limits:

- **Transport**: WebSocket (sim adapter) or Intercom P2P (hardware)
- **Voice**: Voice selector, language (EN/DE), speed slider
- **Wake Word**: Name input, timeout setting
- **Safety**: Max speed, grasp force, reach limits, safety stop on disconnect
- **G1-Specific**: Jetson IP, DDS domain, hand control toggle

### Robot LLM Panel

Overrides global LLM settings for the robot endpoint:

- **Model override**: Use a different model for robot reasoning
- **Multimodal toggle**: Enable/disable image+video input
- **Max tokens**: Override response length for robot
- **Temperature**: Override sampling temperature
- **System prompt**: Robot-specific system prompt

Empty fields fall back to global LLM settings. See `config.robot_llm()` in config.py.

### Perception Panel

Camera and audio controls (requires robot/sim connection):

- Camera source selector, snapshot, describe (VLM), video stream
- Audio monitor with amplitude visualization
- Sensor readouts: IMU (roll/pitch/yaw), grippers

### Spatial Memory Panel

The **Memory** sub-section provides a UI for the spatial memory subsystem.

### Panels

**Stats panel**
- Total entity counts: persons, objects, rooms
- Total observation count
- DB size on disk
- Last observation timestamp

**Persons panel**
- Table of all known persons: name, description, photo thumbnails, last seen location + time
- Add person: name, description, optional reference photo upload
- Edit / delete person records

**Objects panel**
- Table of all known objects: name, description, photo thumbnails, last seen location + time
- Add object: name, description, optional reference photo upload
- Edit / delete object records

**Rooms panel**
- Table of defined rooms/zones: name, description
- Add room: name, description
- Edit / delete room records

**Query panel**
- Natural language search box: type a question (e.g. "where are my keys?") and get back matching entities with last-seen location and timestamp
- Filter by type (person / object / room) and time range
- Results shown as cards with entity details and observation history

### Time Display

The control center uses relative timestamps throughout:

- **`relativeTime(unixTs)`** JS helper mirrors `time_utils.relative()` — "5 min ago", "yesterday", "3 days ago"
- **Memory query results** show relative time instead of raw timestamps (hover for full date)
- **Memory stats** display temporal range: oldest and newest observation times
- Full date/time available on hover via HTML `title` attribute

### Notes

- Reference photos are stored in the DB and used for visual re-identification on the robot
- The Memory tab is always accessible regardless of robot connection state — you can teach the robot about persons and objects before it arrives
- Security-sensitive settings (intercom keys, low-level control) are only editable from the control center, not from agents or the REST API
- For REST API access to spatial memory, see [robot-gateway/SKILL.md](../robot-gateway/SKILL.md#spatial-memory)
- For OpenClaw plugin tools (`clawfinger_memory_*`), see [openclaw-clawfinger/SKILL.md](../openclaw-clawfinger/SKILL.md#spatial-memory-tools)
