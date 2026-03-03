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

- **Status**: Gateway health, active sessions, LLM/ASR status, uptime
- **Call**: Active call sessions, conversation history, inject TTS, takeover controls
- **Config**: TTS voice/language, LLM model/parameters, call policy (greeting, auto-answer, blocklist)
- **Robot**: Robot connection state, command dispatch, project status, perception panel
- **Memory** *(Spatial Memory Panel — see below)*

## Spatial Memory Panel

The **Memory** tab in the Robot section provides a UI for the spatial memory subsystem.

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

### Notes

- Reference photos are stored in the DB and used for visual re-identification on the robot
- The Memory tab is always accessible regardless of robot connection state — you can teach the robot about persons and objects before it arrives
- Security-sensitive settings (intercom keys, low-level control) are only editable from the control center, not from agents or the REST API
- For REST API access to spatial memory (19 endpoints), see [robot-gateway/SKILL.md](../robot-gateway/SKILL.md#spatial-memory)
- For OpenClaw plugin tools (`clawfinger_memory_*`), see [openclaw-clawfinger/SKILL.md](../openclaw-clawfinger/SKILL.md#spatial-memory-tools)
