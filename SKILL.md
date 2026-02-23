---
name: clawfinger-gateway
description: Clawfinger voice gateway — local ASR/LLM/TTS pipeline for AI-assisted phone calls. Use this skill for gateway installation, API usage, agent integration, or OpenClaw plugin setup. Runs on macOS and Linux.
---

# Clawfinger Gateway

> **Repo**: [Trac-Systems/clawfinger-gateway](https://github.com/Trac-Systems/clawfinger-gateway/)
> **Companion repo**: [Trac-Systems/clawfinger-app](https://github.com/Trac-Systems/clawfinger-app/)

Local voice gateway: ASR -> LLM -> TTS pipeline for AI-assisted phone calls. No cloud, no remote servers.

## Skills

### [Voice Gateway](skills/voice-gateway/SKILL.md)
Installation, configuration, startup, API endpoints, agent WebSocket protocol, instruction system, control center UI.

### [OpenClaw Plugin](skills/openclaw-clawfinger/SKILL.md)
OpenClaw plugin for real-time call takeover, injection, context, and observation via the agent WebSocket. Gives OpenClaw agents full access to gateway streams beyond REST.

### [OpenClaw Ops](skills/openclaw-ops/SKILL.md)
Operational runbooks for OpenClaw skill-only automation: scheduled checks, webhook triggers, scripted REST operations (dial, inject, policy updates) — no plugin required.
