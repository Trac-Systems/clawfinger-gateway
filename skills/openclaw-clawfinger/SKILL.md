---
name: openclaw-clawfinger
description: OpenClaw plugin for the Clawfinger voice gateway — real-time call takeover, TTS injection, context injection, and live observation via the agent WebSocket bridge. Use this skill to understand the plugin tools and workflows.
metadata:
  openclaw:
    emoji: "\U0001F4DE"
    skillKey: openclaw-clawfinger
    requires:
      - plugin:clawfinger
---

# OpenClaw Clawfinger Plugin

OpenClaw plugin that bridges to the Clawfinger voice gateway. Gives OpenClaw agents full control over active phone calls: take over the LLM, inject speech, push context, observe transcripts, and manage call policy.

> **Plugin location**: `gateway/openclaw/clawfinger/`
> **Install**: Add the plugin path to `~/.openclaw/openclaw.json` (see Installation below)

## Installation

Add to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "load": {
      "paths": ["/path/to/gateway/openclaw/clawfinger"]
    },
    "entries": {
      "clawfinger": {
        "enabled": true,
        "config": {
          "gatewayUrl": "http://127.0.0.1:8996",
          "bearerToken": "localdev"
        }
      }
    }
  }
}
```

Then install the plugin's dependencies and restart:

```bash
cd /path/to/gateway/openclaw/clawfinger
npm install
openclaw gateway restart
```

## Available Tools

### Status and Observation

| Tool | Description |
|------|-------------|
| `clawfinger_status` | Gateway health, active sessions, bridge connection status |
| `clawfinger_sessions` | List active call session IDs |
| `clawfinger_call_state` | Full call state for a session: conversation history, instructions, takeover status |

### Call Control

| Tool | Description |
|------|-------------|
| `clawfinger_dial` | Dial an outbound phone call (phone must be connected via ADB) |
| `clawfinger_hangup` | Force hang up the active phone call via ADB and end the gateway session |
| `clawfinger_inject` | Inject a TTS message into the active call — text is synthesized and played to the caller |
| `clawfinger_takeover` | Take over LLM control for a session — then use turn_wait/turn_reply to handle turns |
| `clawfinger_turn_wait` | Wait for the next caller turn during takeover (returns transcript + request_id) |
| `clawfinger_turn_reply` | Send your reply text for a takeover turn (requires request_id from turn_wait) |
| `clawfinger_release` | Release LLM control back to the local gateway LLM |
| `clawfinger_session_end` | Mark a call session as ended (hung up) — moves it from active to ended state |

### Context and Instructions

| Tool | Description |
|------|-------------|
| `clawfinger_context_set` | Inject knowledge into a session — the LLM sees this before each user turn. Replaces existing context. |
| `clawfinger_context_clear` | Clear injected knowledge from a session |
| `clawfinger_instructions_set` | Set LLM system instructions. Scope: `global`, `session`, or `turn` (one-shot). |

### Configuration

| Tool | Description |
|------|-------------|
| `clawfinger_call_config_get` | Read call policy: auto-answer, greetings, caller filtering, max duration, auth |
| `clawfinger_call_config_set` | Update call policy settings (pass only fields to change). Allowed fields: `greeting_incoming`, `greeting_outgoing`, `greeting_owner`, `max_duration_sec`, `max_duration_message`, `call_auto_answer`, `call_auto_answer_delay_ms`, `keep_history`, `tts_voice`, `tts_speed`. **Not allowed**: `tts_lang` (language is control-center-only), `piper_*` settings. |

## WS Bridge

The plugin maintains a persistent WebSocket connection to the gateway at `/api/agent/ws`. The bridge:

- Auto-reconnects with exponential backoff (1s -> 30s max)
- Sends ping heartbeats every 15s
- Receives all gateway events (`turn.*`, `agent.*`, `config.*`, etc.)
- Handles `request_id` correlation for takeover turn replies

The bridge starts automatically when the plugin loads and stops when it unloads.

## WS Event Envelope Format

**All events from the gateway use a nested envelope.** The top-level JSON has `type`, `timestamp`, `session_id`, and a `data` object containing the event-specific fields:

```json
{
  "type": "turn.transcript",
  "timestamp": 1708700000.123,
  "session_id": "abc123def456",
  "data": {
    "transcript": "what the caller actually said"
  }
}
```

**Common event payloads** (fields inside `data`):

| Event | `data` fields |
|-------|---------------|
| `turn.started` | `session_id` |
| `turn.transcript` | `transcript` |
| `turn.reply` | `reply` |
| `turn.complete` | `metrics`, `transcript`, `reply`, `model` |
| `turn.request` | `session_id`, `transcript`, `request_id` (takeover only) |
| `turn.stale` | `session_id`, `reason` |
| `turn.error` | `error` |
| `turn.authenticated` | `session_id` |
| `turn.auth_failed` | `session_id`, `attempt` |
| `turn.caller_rejected` | `number`, `reason` |
| `agent.connected` | *(empty)* |
| `config.updated` | `key`, `value` |

**Important:** Always read event-specific fields from `event.data`, not from the top level. For example, to get the transcript text: `event.data.transcript`, **not** `event.transcript`.

## Takeover Lifecycle

1. **Observe** — Use `clawfinger_status` and `clawfinger_sessions` to see active calls.
2. **Inspect** — Use `clawfinger_call_state` to read conversation history and current instructions.
3. **Prepare** — Optionally use `clawfinger_context_set` to inject knowledge the LLM should have.
4. **Take over** — Call `clawfinger_takeover` with the session ID. The gateway routes caller transcripts to you instead of the local LLM.
5. **Respond** — When you receive a `turn.request` event with a transcript, the bridge sends your reply back with `request_id` correlation.
6. **Release** — Call `clawfinger_release` to hand control back to the local LLM.

During takeover, if you fail to reply within 30 seconds, the gateway falls back to the local LLM for that turn.

## Turn Protocol (request_id correlation)

During takeover, the gateway sends:

```json
{
  "type": "turn.request",
  "session_id": "abc123",
  "transcript": "what the caller said",
  "request_id": "a1b2c3d4..."
}
```

The plugin replies with the `request_id` echoed back:

```json
{
  "reply": "the agent's response text",
  "request_id": "a1b2c3d4..."
}
```

This single-reader pattern eliminates WebSocket race conditions. The gateway's WS loop is the sole reader; `/api/turn` posts pending requests and awaits Futures resolved by the loop.

## Example Workflows

### Monitor and inject context

```
1. clawfinger_status          -> check gateway is healthy
2. clawfinger_sessions        -> get active session IDs
3. clawfinger_call_state      -> read conversation history
4. clawfinger_context_set     -> push relevant knowledge
```

### Full call takeover

```
1. clawfinger_sessions        -> find the active session
2. clawfinger_takeover        -> take LLM control
3. clawfinger_turn_wait       -> blocks until caller speaks, returns transcript + request_id
4. clawfinger_turn_reply      -> send your response with the request_id
   (repeat 3-4 for each turn)
5. clawfinger_release         -> hand back to local LLM
```

### Outbound call with greeting

```
1. clawfinger_instructions_set  -> set instructions for the call
2. clawfinger_dial              -> dial the number
3. clawfinger_sessions          -> find the new session
4. clawfinger_context_set       -> push context for the LLM
```

## Slash Command

All gateway operations are also available as direct `/clawfinger` subcommands that bypass the LLM:

| Command | Description |
|---------|-------------|
| `/clawfinger` | Show help with all subcommands |
| `/clawfinger status` | Gateway health, bridge connection, sessions, uptime, LLM status |
| `/clawfinger sessions` | List active session IDs |
| `/clawfinger state <session_id>` | Full call state: history, instructions, takeover status |
| `/clawfinger dial <number>` | Dial outbound call (e.g. `+49123456789`) |
| `/clawfinger hangup [session_id]` | Force hang up the active call and end gateway session |
| `/clawfinger inject <text>` | Inject TTS into active call (uses first session) |
| `/clawfinger inject <session_id> <text>` | Inject TTS into a specific session |
| `/clawfinger takeover <session_id>` | Take over LLM control for a session |
| `/clawfinger release <session_id>` | Release LLM control back to local LLM |
| `/clawfinger context get <session_id>` | Read injected knowledge for a session |
| `/clawfinger context set <session_id> <text>` | Inject/replace knowledge for a session |
| `/clawfinger context clear <session_id>` | Clear injected knowledge |
| `/clawfinger config call` | Show call policy settings (auto-answer, greetings, filtering) |
| `/clawfinger config tts` | Show TTS settings (voice, speed, language, Piper params if German) |
| `/clawfinger config llm` | Show LLM model and generation params |
| `/clawfinger config robot` | Show robot config (model, capabilities, connection) |
| `/clawfinger instructions <text>` | Set global LLM system instructions |
| `/clawfinger instructions <session_id> <text>` | Set per-session LLM instructions |
| `/clawfinger end <session_id>` | Mark a session as ended (hung up) |

## Plugin Architecture

```
gateway/openclaw/clawfinger/
  package.json              # Plugin manifest
  openclaw.plugin.json      # Config schema + UI hints
  src/
    index.ts                # Entry: registers service + tools + command
    ws-bridge.ts            # Persistent WS with reconnect + heartbeat
    gateway-client.ts       # REST client for all gateway endpoints
```

## Robot Tools

These tools interact with the robot endpoint via Intercom P2P transport. `clawfinger_robot_status` works always (shows transport state even when disconnected). Command tools require `robot.enabled: true` and an active robot connection.

| Tool | Description |
|------|-------------|
| `clawfinger_robot_status` | Robot status: connection state, transport info, model, capabilities. Works even when robot is disconnected. |
| `clawfinger_robot_command` | Send a command to the connected robot via Intercom transport. Supports all G1 capabilities (walk, stop, look, speak, etc.). |
| `clawfinger_robot_config_get` | Read robot config (shared + model-specific) |
| `clawfinger_robot_config_set` | Update robot config (security keys blocked from agents) |

### Future robot tools

| Tool | Capability | Description |
|------|-----------|-------------|
| `clawfinger_robot_walk` | `locomotion` | Walk: direction, distance, speed |
| `clawfinger_robot_turn` | `locomotion` | Turn: angle (degrees) |
| `clawfinger_robot_stand` / `_sit` / `_stop` | `posture` | Posture commands |
| `clawfinger_robot_pick_up` | `manipulation` | Pick up object by description |
| `clawfinger_robot_place` | `manipulation` | Place object at target location |
| `clawfinger_robot_hand_over` | `manipulation` | Extend hand, wait for take |
| `clawfinger_robot_gesture` | `gesture` | Wave, point, nod, etc. |
| `clawfinger_robot_look` | `vision` | Describe scene (camera + VLM) |
| `clawfinger_robot_snapshot` | `vision` | Return camera frame |
| `clawfinger_robot_speak` | `audio` | TTS on robot speaker |
| `clawfinger_robot_listen` | `audio` | Mic capture + ASR |
| `clawfinger_robot_task` | (any) | Submit compound task (async, validated against capabilities) |
| `clawfinger_robot_task_status` | (any) | Check task progress |
| `clawfinger_robot_task_cancel` | (any) | Cancel running task |

## Cross-Endpoint Workflows

The plugin shares a single WS bridge for both phone and robot commands. An agent can interleave phone and robot tool calls freely.

### Phone agent commands robot (between turns)

```
1. clawfinger_dial +49123456789              (phone)
2. clawfinger_takeover <sid>                 (phone)
3. clawfinger_turn_wait                      (phone — caller speaks)
4. clawfinger_robot_command walk {speed: 0.3} (robot — dispatched via Intercom)
5. clawfinger_turn_reply "I've sent the robot to get your package"
6. clawfinger_robot_status                   (check transport state between turns)
```

### Monitor robot while on call

```
1. clawfinger_turn_wait                      (phone — waiting for caller)
2. (caller speaks)
3. clawfinger_robot_status                   (robot — check connection, transport state)
4. clawfinger_turn_reply "Let me check..."   (phone — respond to caller)
```

### Transport-aware status checks

```
1. clawfinger_robot_status                   → shows connected/disconnected + transport state
2. clawfinger_robot_command stop             → sends stop via Intercom (fire-and-forget)
3. clawfinger_robot_command walk {speed: 0.2} → waits for robot response (request/response)
```

## Gateway API Reference

For full API documentation, endpoint details, and config options, see the [Phone Gateway skill](../phone-gateway/SKILL.md). For robot endpoint details, see the [Robot Gateway skill](../robot-gateway/SKILL.md).

---

## Robot Skill & Project Tools

| Tool | Description |
|------|-------------|
| `clawfinger_robot_skill_list` | List available robot skills (slow + fast path) |
| `clawfinger_robot_skill_topic` | Read a skill topic's knowledge content |
| `clawfinger_robot_project_status` | Current project execution state |
| `clawfinger_robot_project_cancel` | Cancel running robot project |

## Robot Takeover Tools

| Tool | Description |
|------|-------------|
| `clawfinger_robot_takeover` | Take full control of robot (voice + commands) |
| `clawfinger_robot_turn_wait` | Wait for user to speak to robot during takeover |
| `clawfinger_robot_turn_reply` | Send text reply (spoken on robot) + optional commands |
| `clawfinger_robot_release` | Release robot control back to local LLM |

### Robot Takeover Workflow

```
1. clawfinger_robot_takeover              → take control
2. clawfinger_robot_turn_wait             → user speaks to robot
3. clawfinger_robot_turn_reply            → agent text spoken on robot + optional commands
   (repeat 2-3)
4. clawfinger_robot_release               → hand back to local LLM
```

## Robot Perception Tools

| Tool | Description |
|------|-------------|
| `clawfinger_robot_snapshot` | Capture a camera frame from the robot. Optional params: `source`, `width`, `height`, `quality`. Returns image data. |
| `clawfinger_robot_describe` | VLM scene description from robot camera. Optional params: `source`, `prompt`. Returns text description of what the camera sees. |

### Perception Slash Commands

```
/clawfinger robot perception              — list available cameras + mics
/clawfinger robot snapshot [source]       — capture camera frame
/clawfinger robot describe [source] [prompt] — VLM scene description
/clawfinger robot stream start|stop [source] — video stream control
/clawfinger robot audio start|stop [source]  — audio monitoring control
```

### Robot Slash Commands

```
/clawfinger robot skills              — list robot skills
/clawfinger robot skill <name> <topic> — read skill topic
/clawfinger robot project             — current project status
/clawfinger robot project cancel      — cancel running project
/clawfinger robot takeover            — take control of robot
/clawfinger robot release             — release robot control
```

### Robot Events

| Event | When |
|-------|------|
| `robot.project.started` | Project initiated from voice |
| `robot.project.step` | Each autonomous action |
| `robot.project.completed` | Project done |
| `robot.project.failed` | Timeout, disconnect, abort |
| `robot.project.cancelled` | User "stop" or API cancel |
| `robot.skill.loaded` | Skill topic loaded into context |
