---
name: voice-gateway
description: Clawfinger voice/phone endpoint overview — installation, configuration, API reference, and operations for the FastAPI gateway running the ASR/LLM/TTS pipeline. Cross-references robot endpoint and spatial memory subsystem.
metadata:
  openclaw:
    emoji: "\U0001F399"
    skillKey: voice-gateway
---

# Voice Gateway — Clawfinger Phone Endpoint Overview

> **Full phone endpoint documentation**: see [`phone-gateway`](../phone-gateway/SKILL.md).
> **Robot endpoint documentation**: see [`robot-gateway`](../robot-gateway/SKILL.md).

The Clawfinger gateway is a local FastAPI server that handles the full ASR → LLM → TTS pipeline for Android phone calls and robot voice I/O. No cloud services required.

## Subsystems

- **Phone endpoint**: Call handling, agent takeover, TTS injection, session management
- **Robot endpoint**: Humanoid robot control via Intercom P2P transport
- **Spatial memory**: Persistent DB of known persons, objects, and rooms — see note below

## Spatial Memory

The gateway includes a spatial memory subsystem that stores what the robot has observed: persons it knows, objects it has seen, rooms/zones it has mapped, and timestamped observation records. This allows the robot (and agents) to answer questions like "where did I last see my keys?" or "who was in the kitchen this morning?"

For the full spatial memory API reference (19 REST endpoints, query types, and OpenClaw tools), see [robot-gateway/SKILL.md](../robot-gateway/SKILL.md#spatial-memory).

For OpenClaw plugin tools (`clawfinger_memory_*`) and slash commands, see [openclaw-clawfinger/SKILL.md](../openclaw-clawfinger/SKILL.md#spatial-memory-tools).

## Time Awareness

Time context (current time, time-of-day) is injected into all LLM system prompts for both phone and robot endpoints. Phone prompts also include call duration. Robot prompts include project elapsed time and step durations.

All temporal data comes from `time_utils.py` using the configured `timezone` setting (default: `"Europe/Berlin"`).

## Quick Reference

| Component | Skill |
|-----------|-------|
| Phone/voice API | [phone-gateway](../phone-gateway/SKILL.md) |
| Robot API + spatial memory | [robot-gateway](../robot-gateway/SKILL.md) |
| Agent takeover lifecycle | [agent-takeover](../agent-takeover/SKILL.md) |
| OpenClaw plugin tools | [openclaw-clawfinger](../openclaw-clawfinger/SKILL.md) |
| REST automation runbooks | [openclaw-ops](../openclaw-ops/SKILL.md) |
