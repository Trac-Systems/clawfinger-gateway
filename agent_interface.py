"""Agent WebSocket + REST interface for OpenClaw integration."""

from __future__ import annotations

import time
from typing import Any

from fastapi import WebSocket

from event_bus import bus

# Connected agents: {ws: {"name": ..., "connected_at": ..., "takeover_sessions": set()}}
_AGENTS: dict[WebSocket, dict[str, Any]] = {}

# Sessions where an agent has taken over LLM: {session_id: agent_ws}
_TAKEOVER: dict[str, WebSocket] = {}


async def agent_connect(ws: WebSocket) -> None:
    _AGENTS[ws] = {
        "name": "agent",
        "connected_at": time.time(),
        "takeover_sessions": set(),
    }
    await bus.subscribe(ws)
    await bus.publish("agent.connected", {"agent_count": len(_AGENTS)})


async def agent_disconnect(ws: WebSocket) -> None:
    info = _AGENTS.pop(ws, None)
    await bus.unsubscribe(ws)
    # Release any takeovers this agent held
    if info:
        for sid in list(info.get("takeover_sessions", set())):
            _TAKEOVER.pop(sid, None)
    await bus.publish("agent.disconnected", {"agent_count": len(_AGENTS)})


def get_takeover_agent(session_id: str) -> WebSocket | None:
    """If an agent has taken over this session's LLM, return its WebSocket."""
    ws = _TAKEOVER.get(session_id)
    if ws and ws in _AGENTS:
        return ws
    # Clean up stale takeover
    _TAKEOVER.pop(session_id, None)
    return None


async def takeover(ws: WebSocket, session_id: str) -> bool:
    if ws not in _AGENTS:
        return False
    _TAKEOVER[session_id] = ws
    _AGENTS[ws]["takeover_sessions"].add(session_id)
    await bus.publish("agent.takeover", {"session_id": session_id}, session_id=session_id)
    return True


async def release(ws: WebSocket, session_id: str) -> bool:
    if _TAKEOVER.get(session_id) is not ws:
        return False
    _TAKEOVER.pop(session_id, None)
    if ws in _AGENTS:
        _AGENTS[ws]["takeover_sessions"].discard(session_id)
    await bus.publish("agent.release", {"session_id": session_id}, session_id=session_id)
    return True


def list_agents() -> list[dict[str, Any]]:
    return [
        {
            "name": info["name"],
            "connected_at": info["connected_at"],
            "takeover_sessions": list(info["takeover_sessions"]),
        }
        for info in _AGENTS.values()
    ]
