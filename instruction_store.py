"""In-memory instruction storage with base / session / turn layers.

ALL instruction state is strictly per-session.  There is no shared global
mutable state — the only cross-session fallback is the *immutable* config
value ``llm_system_prompt`` which is set at gateway startup and never
modified at runtime.
"""

from __future__ import annotations

import config
import session_store
import time_utils

_SESSION: dict[str, str] = {}
_TURN: dict[str, str] = {}
_AGENT_KNOWLEDGE: dict[str, str] = {}


def get_base() -> str:
    """Return the immutable default system prompt from config (never mutated at runtime)."""
    return config.get("llm.system_prompt", "")


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
        base = base + "\n\n" + turn_extra

    # Inject time context + call duration
    time_line = time_utils.time_context_line()
    duration_line = ""
    meta = session_store.get_session_detail(sid)
    if meta:
        created = meta.get("created_at")
        if created:
            elapsed = time_utils.now_unix() - created
            duration_line = f"\nCall duration so far: {time_utils.duration(elapsed)}"
    base += f"\n\n[{time_line}{duration_line}]"
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


def clear_all_for_session(sid: str) -> None:
    """Remove ALL instruction state for a session (call on session end/reset)."""
    _SESSION.pop(sid, None)
    _TURN.pop(sid, None)
    _AGENT_KNOWLEDGE.pop(sid, None)


def snapshot() -> dict:
    return {
        "base": get_base(),
        "sessions": dict(_SESSION),
    }
