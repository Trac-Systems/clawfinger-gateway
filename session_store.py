"""Session history management with disk persistence."""

from __future__ import annotations

import asyncio
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

# Compacted conversation summaries: {session_id: "Summary text..."}
_SUMMARY: dict[str, str] = {}

# Passphrase auth state: {session_id: {"validated": bool, "attempts": int}}
_AUTH_STATE: dict[str, dict[str, Any]] = {}

# Per-session asyncio locks for concurrent access coordination
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}


def get_lock(session_id: str) -> asyncio.Lock:
    """Return a per-session asyncio lock, creating if needed."""
    if session_id not in _SESSION_LOCKS:
        _SESSION_LOCKS[session_id] = asyncio.Lock()
    return _SESSION_LOCKS[session_id]


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
    _SUMMARY.pop(session_id, None)
    _SESSION_LOCKS.pop(session_id, None)
    # Also clean up agent knowledge for this session
    import instruction_store
    instruction_store.clear_agent_knowledge(session_id)
    return get_or_create(session_id)


def get_history(session_id: str) -> list[dict[str, str]]:
    return _HISTORY.get(session_id, [])


def append(session_id: str, role: str, content: str) -> None:
    _HISTORY.setdefault(session_id, []).append({"role": role, "content": content})
    compact(session_id)


def compact(session_id: str) -> None:
    """Compact conversation history: summarize oldest messages, keep recent ones."""
    history = _HISTORY.get(session_id)
    if not history:
        return

    max_turns = config.get("max_history_turns", 8)
    keep = max(1, max_turns) * 2  # messages to keep verbatim

    # Also check token budget if configured
    context_limit = config.get("llm_context_tokens", 0)
    if context_limit > 0:
        reserve = config.get("llm_max_tokens", 400) + 300  # output + system prompt headroom
        budget = context_limit - reserve
        while keep > 2:
            recent = history[-keep:] if len(history) >= keep else history
            total_chars = sum(len(m["content"]) for m in recent)
            if total_chars / 4 <= budget:
                break
            keep -= 2

    if len(history) <= keep:
        return  # nothing to compact

    # Split: old messages to summarize, recent to keep verbatim
    to_summarize = history[: len(history) - keep]
    recent = history[len(history) - keep :]

    # Build text to summarize (include prior summary if exists)
    existing_summary = _SUMMARY.get(session_id, "")
    summary_input_parts = []
    if existing_summary:
        summary_input_parts.append(f"Previous summary:\n{existing_summary}")
    for msg in to_summarize:
        role = msg["role"]
        summary_input_parts.append(f"{role}: {msg['content']}")
    summary_input = "\n".join(summary_input_parts)

    # Use LLM to summarize
    import llm_backend
    messages = [
        {"role": "system", "content": (
            "Summarize this phone conversation history into a concise paragraph. "
            "Preserve: caller identity, key facts mentioned, decisions made, "
            "questions asked, and any commitments. "
            "Drop: filler, greetings, repetition. "
            "Output only the summary, nothing else."
        )},
        {"role": "user", "content": summary_input},
    ]
    try:
        summary_text, _, _ = llm_backend.generate(messages)
        _SUMMARY[session_id] = summary_text
    except Exception:
        # If summarization fails, fall back to simple truncation
        _SUMMARY.pop(session_id, None)

    # Replace history with just the recent messages
    _HISTORY[session_id] = list(recent)


def get_summary(session_id: str) -> str:
    """Get the compacted summary for a session (empty if none)."""
    return _SUMMARY.get(session_id, "")


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
