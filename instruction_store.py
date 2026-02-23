"""In-memory instruction storage with base / session / turn layers."""

from __future__ import annotations

import config

_BASE: str = ""
_SESSION: dict[str, str] = {}
_TURN: dict[str, str] = {}
_AGENT_KNOWLEDGE: dict[str, str] = {}


def get_base() -> str:
    return _BASE or config.get("llm_system_prompt", "")


def set_base(text: str) -> None:
    global _BASE
    _BASE = text


def get_session(sid: str) -> str:
    return _SESSION.get(sid, "")


def set_session(sid: str, text: str) -> None:
    _SESSION[sid] = text


def clear_session(sid: str) -> None:
    _SESSION.pop(sid, None)


def get_turn(sid: str) -> str:
    return _TURN.get(sid, "")


def set_turn(sid: str, text: str) -> None:
    _TURN[sid] = text


def pop_turn(sid: str) -> str:
    return _TURN.pop(sid, "")


def build_system_prompt(sid: str) -> str:
    base = get_session(sid) or get_base()
    turn_extra = pop_turn(sid)
    if turn_extra:
        return base + "\n\n" + turn_extra
    return base


# ---------------------------------------------------------------------------
# Agent knowledge injection
# ---------------------------------------------------------------------------

def set_agent_knowledge(sid: str, text: str) -> None:
    _AGENT_KNOWLEDGE[sid] = text


def get_agent_knowledge(sid: str) -> str:
    return _AGENT_KNOWLEDGE.get(sid, "")


def clear_agent_knowledge(sid: str) -> None:
    _AGENT_KNOWLEDGE.pop(sid, None)


def snapshot() -> dict:
    return {
        "base": get_base(),
        "sessions": dict(_SESSION),
    }
