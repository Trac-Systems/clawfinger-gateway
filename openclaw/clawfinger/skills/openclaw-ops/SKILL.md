---
name: openclaw-ops
description: Operational runbooks for OpenClaw skill-only automation of the Clawfinger voice gateway — scheduled health checks, webhook triggers, scripted REST operations. No plugin required.
metadata:
  openclaw:
    emoji: "\U0001F6E0"
    skillKey: openclaw-ops
---

# OpenClaw Ops — Gateway Automation Without a Plugin

Operational runbooks for automating the Clawfinger voice gateway using only OpenClaw skills and REST calls. No plugin installation required — just `curl` commands and OpenClaw's built-in scheduling/webhook features.

Use this when you want quick automation (health monitoring, scheduled policy changes, event-driven actions) without installing the full Clawfinger OpenClaw plugin.

> For full plugin-based automation with real-time WS bridge, takeover, and tool calling, see [OpenClaw Plugin](../openclaw-clawfinger/SKILL.md).
> For phone API reference, see [Phone Gateway](../phone-gateway/SKILL.md). For robot API reference, see [Robot Gateway](../robot-gateway/SKILL.md).

## Prerequisites

- Clawfinger gateway running and reachable (default: `http://127.0.0.1:8996`)
- Bearer token configured (default: `localdev`)
- `curl` and `jq` available on the host

All examples below use these variables:

```bash
GW="http://127.0.0.1:8996"
TOKEN="localdev"
AUTH="-H 'Authorization: Bearer $TOKEN'"
```

## Health Monitoring

### Quick health check

```bash
curl -sf "$GW/api/status" | jq '{
  healthy: .mlx_audio.ok,
  llm_loaded: .llm.loaded,
  sessions: .active_sessions,
  uptime_min: (.uptime_s / 60 | floor),
  agents: (.agents | length)
}'
```

### Cron-based heartbeat (every 5 minutes)

```bash
# Add to crontab: */5 * * * * /path/to/gateway-health.sh
#!/bin/bash
GW="http://127.0.0.1:8996"
STATUS=$(curl -sf "$GW/api/status")
if [ $? -ne 0 ]; then
  echo "$(date): Gateway unreachable" >> /tmp/gateway-health.log
  exit 1
fi
MLX_OK=$(echo "$STATUS" | jq -r '.mlx_audio.ok')
LLM_OK=$(echo "$STATUS" | jq -r '.llm.loaded')
if [ "$MLX_OK" != "true" ] || [ "$LLM_OK" != "true" ]; then
  echo "$(date): Gateway degraded — mlx=$MLX_OK llm=$LLM_OK" >> /tmp/gateway-health.log
fi
```

## Call Operations

### Dial an outbound call

```bash
curl -s -X POST "$GW/api/call/dial" \
  -H "Content-Type: application/json" \
  -d '{"number": "+491234567890"}' | jq .
```

### Inject a TTS message into an active call

```bash
# Get active sessions first
SESSIONS=$(curl -s "$GW/api/agent/sessions")
SID=$(echo "$SESSIONS" | jq -r '.[0]')

# Inject speech
curl -s -X POST "$GW/api/agent/inject" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Please hold, I am looking that up for you.\", \"session_id\": \"$SID\"}" | jq .
```

### Get full call state

```bash
curl -s "$GW/api/agent/call/$SID" | jq '{
  session: .session_id,
  turns: .turn_count,
  takeover: .agent_takeover,
  last_messages: [.history[-2:][] | "\(.role): \(.content[0:80])"]
}'
```

## Context Injection

### Push knowledge into a session

```bash
curl -s -X POST "$GW/api/agent/context/$SID" \
  -H "Content-Type: application/json" \
  -d '{"context": "Caller is John Smith, account #12345. Balance: EUR 1,234.56. Last payment: 2026-02-15."}' | jq .
```

### Read current context

```bash
curl -s "$GW/api/agent/context/$SID" | jq .
```

### Clear context

```bash
curl -s -X DELETE "$GW/api/agent/context/$SID" | jq .
```

## Policy Management

### Read current call policy

```bash
curl -s "$GW/api/config/call" | jq .
```

### Update greetings

```bash
curl -s -X POST "$GW/api/config/call" \
  -H "Content-Type: application/json" \
  -d '{
    "greeting_incoming": "Hello, I am {owner}'\''s assistant. How can I help you?",
    "greeting_owner": "Markus"
  }' | jq .
```

### Enable passphrase authentication

```bash
curl -s -X POST "$GW/api/config/call" \
  -H "Content-Type: application/json" \
  -d '{
    "auth_passphrase": "blue harvest",
    "auth_max_attempts": 3,
    "auth_reject_message": "Access denied. Goodbye."
  }' | jq .
```

### Block a caller

```bash
# Read current blocklist
BLOCKLIST=$(curl -s "$GW/api/config/call" | jq -c '.caller_blocklist')

# Add a number
curl -s -X POST "$GW/api/config/call" \
  -H "Content-Type: application/json" \
  -d "{\"caller_blocklist\": $(echo "$BLOCKLIST" | jq '. + [\"+491111111111\"]')}" | jq .caller_blocklist
```

### Scheduled policy change (business hours)

```bash
# Enable auto-answer during business hours, disable at night
# Cron: 0 8 * * 1-5  /path/to/business-hours-on.sh
# Cron: 0 18 * * 1-5 /path/to/business-hours-off.sh

# business-hours-on.sh
curl -s -X POST "$GW/api/config/call" \
  -H "Content-Type: application/json" \
  -d '{"call_auto_answer": true, "unknown_callers_allowed": true}'

# business-hours-off.sh
curl -s -X POST "$GW/api/config/call" \
  -H "Content-Type: application/json" \
  -d '{"call_auto_answer": false, "unknown_callers_allowed": false}'
```

## TTS and LLM Configuration

### Change TTS voice

```bash
curl -s -X POST "$GW/api/config/tts" \
  -H "Content-Type: application/json" \
  -d '{"voice": "am_michael", "speed": 1.0}' | jq .
```

### List available voices

```bash
curl -s "$GW/api/config/tts" | jq '.voices'
```

### Switch to German TTS (Piper Thorsten)

```bash
# Switch language to German — routes TTS through Piper sidecar on :5123
curl -s -X POST "$GW/api/config/tts" \
  -H "Content-Type: application/json" \
  -d '{"lang": "de"}' | jq .

# Configure Piper voice parameters
curl -s -X POST "$GW/api/config/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "lang": "de",
    "piper_voice": "thorsten-high",
    "piper_length_scale": 1.0,
    "piper_noise_scale": 0.667,
    "piper_noise_w": 0.8,
    "piper_sentence_silence": 0.2
  }' | jq .
```

Available Piper voices: `thorsten-high`, `thorsten-medium`, `thorsten-low`, `karlsson-low`, `pavoque-low`, `eva_k-x_low`, `kerstin-low`, `ramona-low`, `thorsten_emotional-medium`.

### Switch back to English TTS (Kokoro)

```bash
curl -s -X POST "$GW/api/config/tts" \
  -H "Content-Type: application/json" \
  -d '{"lang": "en", "voice": "am_adam", "speed": 1.2}' | jq .
```

### Adjust LLM parameters

```bash
curl -s -X POST "$GW/api/config/llm" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 0.3, "max_tokens": 300}' | jq .
```

### Switch to remote LLM

```bash
curl -s -X POST "$GW/api/config/llm" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-..."
  }' | jq .
```

## Instructions Management

### Set global instructions

```bash
curl -s -X POST "$GW/api/instructions" \
  -H "Content-Type: application/json" \
  -d '{"text": "You are a concise, professional real-time voice assistant. Keep responses under 2 sentences."}' | jq .
```

### Set per-session instructions

```bash
curl -s -X POST "$GW/api/instructions/$SID" \
  -H "Content-Type: application/json" \
  -d '{"text": "This caller is a VIP customer. Be extra helpful and patient."}' | jq .
```

### Set one-shot turn instruction

```bash
curl -s -X POST "$GW/api/instructions/$SID/turn" \
  -H "Content-Type: application/json" \
  -d '{"text": "End this turn by asking the caller to confirm their email address."}' | jq .
```

## Session Management

### List all sessions

```bash
curl -s "$GW/api/sessions" | jq '.[] | {session_id, turn_count, created_at}'
```

### Reset a session

```bash
curl -s -X POST "$GW/api/session/reset" \
  -H "Authorization: Bearer $TOKEN" \
  -d "session_id=$SID" | jq .
```

## Robot Operations

### Read robot config + transport status

```bash
curl -s "$GW/api/config/robot" | jq '{
  enabled: .enabled,
  model: .model,
  connected: .connected,
  capabilities: .capabilities,
  transport: .transport,
  transport_connected: .transport_connected,
  intercom_running: .intercom_running,
  intercom_channel: .intercom_channel
}'
```

### Update robot config

```bash
curl -s -X POST "$GW/api/config/robot" \
  -H "Content-Type: application/json" \
  -d '{"voice": "am_adam", "heartbeat_interval": 10}' | jq .
```

**Security keys** (`intercom_key`, `safety_stop_on_disconnect`, `enable_low_level`) cannot be set via the API from agents — only from the control center.

### Check Intercom process health

```bash
curl -s "$GW/api/config/robot" | jq '{
  transport: .transport,
  transport_connected: .transport_connected,
  intercom_running: .intercom_running,
  channel: .intercom_channel,
  connected: .connected
}'
```

## Robot Perception

### List available cameras and mics

```bash
curl -s -H "Authorization: Bearer $TOKEN" "$GW/api/robot/perception" | jq .
```

### Capture a camera snapshot

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/camera/snapshot" \
  -d '{"source": "head_rgb", "width": 640, "height": 480}' | jq .
```

### VLM scene description

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/camera/describe" \
  -d '{"source": "head_rgb", "prompt": "What objects are on the table?"}' | jq .
```

### Start/stop video stream

```bash
# Start stream
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/camera/stream/start" \
  -d '{"source": "head_rgb", "fps": 10, "width": 320, "height": 240}' | jq .

# View MJPEG stream (e.g., in browser or ffplay)
# GET $GW/api/robot/camera/stream?source=head_rgb&token=$TOKEN

# Stop stream
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/camera/stream/stop" \
  -d '{"source": "head_rgb"}' | jq .
```

### Start/stop audio monitoring

```bash
# Start mic monitoring
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/audio/monitor/start" \
  -d '{"source": "mic_array", "sample_rate": 16000, "channels": 1}' | jq .

# Stop mic monitoring
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/audio/monitor/stop" \
  -d '{"source": "mic_array"}' | jq .
```

## Combining with OpenClaw Skills

These REST operations can be composed into OpenClaw skills for more complex automation:

- **Pre-call setup**: Set instructions + dial + inject context in sequence
- **Post-call cleanup**: Read call state, save transcript, reset session
- **Escalation**: Monitor sessions, inject context when keywords detected, take over via plugin if needed
- **Reporting**: Periodic session listing + call state extraction for analytics
- **Robot monitoring**: Periodic robot status checks, battery alerts, disconnect detection

For real-time event-driven automation (reacting to `turn.complete`, `agent.takeover`, `robot.connected`, etc.), use the [OpenClaw Clawfinger Plugin](../openclaw-clawfinger/SKILL.md) which provides a persistent WebSocket bridge.

---

## Spatial Memory Operations

```bash
# List known persons
curl -s -H "Authorization: Bearer $TOKEN" "$GW/api/robot/memory/persons" | jq .

# Teach a person
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/memory/persons" \
  -d '{"name": "Markus", "description": "Owner"}' | jq .

# Query spatial memory
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  "$GW/api/robot/memory/query" \
  -d '{"type": "text", "text": "where are my keys?"}' | jq .

# Memory stats
curl -s -H "Authorization: Bearer $TOKEN" "$GW/api/robot/memory/stats" | jq .
```

## Robot Skill & Project Operations

```bash
# List robot skills
curl -s -H "Authorization: Bearer $TOKEN" "$GW/api/robot/skills" | jq '.[] | {name, execution_mode, description}'

# Read a skill topic
curl -s -H "Authorization: Bearer $TOKEN" "$GW/api/robot/skills/household_objects/common"

# Check project status
curl -s -H "Authorization: Bearer $TOKEN" "$GW/api/robot/project" | jq '{status, description, step, max_steps}'

# Cancel running project
curl -s -X POST -H "Authorization: Bearer $TOKEN" "$GW/api/robot/project/cancel" | jq .
```
