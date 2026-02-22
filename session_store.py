"""Session history management with disk persistence."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import config

_SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory conversation history: {session_id: [{"role": ..., "content": ...}, ...]}
_HISTORY: dict[str, list[dict[str, str]]] = {}

# Session metadata for control center: {session_id: {...}}
_META: dict[str, dict[str, Any]] = {}

# Caller info per session: {session_id: {"number": str, "direction": str}}
_CALLER_INFO: dict[str, dict[str, str]] = {}

# Passphrase auth state: {session_id: {"validated": bool, "attempts": int}}
_AUTH_STATE: dict[str, dict[str, Any]] = {}


def get_or_create(session_id: str | None = None) -> str:
    if not session_id:
        session_id = uuid.uuid4().hex
    _HISTORY.setdefault(session_id, [])
    if session_id not in _META:
        _META[session_id] = {
            "session_id": session_id,
            "created_at": time.time(),
            "turns": [],
        }
    return session_id


def reset(session_id: str) -> str:
    _HISTORY.pop(session_id, None)
    _META.pop(session_id, None)
    _CALLER_INFO.pop(session_id, None)
    _AUTH_STATE.pop(session_id, None)
    return get_or_create(session_id)


def get_history(session_id: str) -> list[dict[str, str]]:
    return _HISTORY.get(session_id, [])


def append(session_id: str, role: str, content: str) -> None:
    _HISTORY.setdefault(session_id, []).append({"role": role, "content": content})
    trim(session_id)


def trim(session_id: str) -> None:
    history = _HISTORY.get(session_id)
    if not history:
        return
    max_turns = config.get("max_history_turns", 8)
    keep = max(1, max_turns) * 2
    if len(history) > keep:
        del history[: len(history) - keep]


def record_turn(session_id: str, turn_data: dict[str, Any]) -> None:
    """Record a completed turn for session log persistence."""
    meta = _META.get(session_id)
    if meta is None:
        get_or_create(session_id)
        meta = _META[session_id]
    meta["turns"].append({**turn_data, "timestamp": time.time()})


def save_session(session_id: str) -> None:
    """Persist session to disk as JSON."""
    meta = _META.get(session_id)
    if not meta:
        return
    path = _SESSIONS_DIR / f"{session_id}.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def list_sessions() -> list[dict[str, Any]]:
    """List all persisted sessions (summaries)."""
    sessions = []
    for path in sorted(_SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", path.stem),
                "created_at": data.get("created_at"),
                "turn_count": len(data.get("turns", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return sessions


def get_session_detail(session_id: str) -> dict[str, Any] | None:
    """Get full session detail — check memory first, then disk."""
    if session_id in _META:
        return _META[session_id]
    path = _SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def active_sessions() -> dict[str, dict[str, Any]]:
    """Return all current in-memory session metadata."""
    return dict(_META)


# ---------------------------------------------------------------------------
# Caller info
# ---------------------------------------------------------------------------

def set_caller_info(session_id: str, number: str, direction: str) -> None:
    _CALLER_INFO[session_id] = {"number": number, "direction": direction}


def get_caller_info(session_id: str) -> dict[str, str]:
    return _CALLER_INFO.get(session_id, {"number": "", "direction": ""})


# ---------------------------------------------------------------------------
# Passphrase auth state
# ---------------------------------------------------------------------------

def is_authenticated(session_id: str) -> bool:
    state = _AUTH_STATE.get(session_id)
    return state is not None and state.get("validated", False)


def mark_authenticated(session_id: str) -> None:
    _AUTH_STATE.setdefault(session_id, {"validated": False, "attempts": 0})
    _AUTH_STATE[session_id]["validated"] = True


def record_auth_attempt(session_id: str) -> int:
    """Record a failed passphrase attempt. Returns total attempt count."""
    state = _AUTH_STATE.setdefault(session_id, {"validated": False, "attempts": 0})
    state["attempts"] += 1
    return state["attempts"]


def clear_auth_state(session_id: str) -> None:
    _AUTH_STATE.pop(session_id, None)
