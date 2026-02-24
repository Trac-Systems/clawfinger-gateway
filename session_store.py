"""Session history management with disk persistence."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import re

import config

_SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_CALLER_HISTORY_DIR = Path(__file__).resolve().parent / "caller_history"
_CALLER_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

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

# Ended sessions: {session_id: ended_at_timestamp}
_ENDED: dict[str, float] = {}

# Last activity timestamp per session (updated on each turn)
_LAST_ACTIVITY: dict[str, float] = {}

# Per-session asyncio locks for concurrent access coordination
_SESSION_LOCKS: dict[str, asyncio.Lock] = {}

# Pending TTS inject queue: {session_id: [{"text": str, "audio_base64": str}, ...]}
_INJECT_QUEUE: dict[str, list[dict[str, str]]] = {}

# Stale session TTL in seconds (no activity → auto-end)
_SESSION_TTL = 300  # 5 minutes

# Session generation counter — bumped on reset to invalidate in-flight turns
_GENERATION: dict[str, int] = {}


def get_lock(session_id: str) -> asyncio.Lock:
    """Return a per-session asyncio lock, creating if needed."""
    if session_id not in _SESSION_LOCKS:
        _SESSION_LOCKS[session_id] = asyncio.Lock()
    return _SESSION_LOCKS[session_id]


def get_generation(session_id: str) -> int:
    """Return current generation counter for a session."""
    return _GENERATION.get(session_id, 0)


def bump_generation(session_id: str) -> int:
    """Increment generation counter (invalidates in-flight turns). Returns new value."""
    _GENERATION[session_id] = _GENERATION.get(session_id, 0) + 1
    return _GENERATION[session_id]


def queue_inject(session_id: str, text: str, audio_base64: str) -> None:
    """Queue a TTS inject for delivery on next /api/turn poll."""
    _INJECT_QUEUE.setdefault(session_id, []).append({
        "text": text,
        "audio_base64": audio_base64,
    })


def drain_inject(session_id: str) -> dict[str, str] | None:
    """Pop the next pending inject for a session, or None if empty."""
    q = _INJECT_QUEUE.get(session_id)
    if q:
        return q.pop(0)
    return None


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
    bump_generation(session_id)  # invalidate any in-flight turns
    # Save caller history before wiping state (phone may never call end_session)
    if config.get("keep_history", False):
        caller = _CALLER_INFO.get(session_id, {})
        number = caller.get("number", "")
        if number:
            history = _HISTORY.get(session_id, [])
            summary = _SUMMARY.get(session_id, "")
            if history:
                save_caller_history(number, history, summary)
    _HISTORY.pop(session_id, None)
    _META.pop(session_id, None)
    _CALLER_INFO.pop(session_id, None)
    _AUTH_STATE.pop(session_id, None)
    _SUMMARY.pop(session_id, None)
    _SESSION_LOCKS.pop(session_id, None)
    _ENDED.pop(session_id, None)
    _LAST_ACTIVITY.pop(session_id, None)
    _INJECT_QUEUE.pop(session_id, None)
    # Clean up ALL instruction state for this session
    import instruction_store
    instruction_store.clear_all_for_session(session_id)
    return get_or_create(session_id)


def get_history(session_id: str) -> list[dict[str, str]]:
    return _HISTORY.get(session_id, [])


def append(session_id: str, role: str, content: str) -> None:
    _HISTORY.setdefault(session_id, []).append({"role": role, "content": content})


def compact(session_id: str) -> None:
    """Compact conversation history: summarize oldest messages, keep recent ones.

    Called once per turn (after both user+assistant are appended), NOT per append.
    This avoids double-compaction and ensures we always summarize complete pairs.
    """
    history = _HISTORY.get(session_id)
    if not history:
        return

    max_turns = config.get("max_history_turns", 8)
    keep = max(1, max_turns) * 2  # messages to keep verbatim

    # Token budget: use explicit config, or auto-detect from loaded model
    context_limit = config.get("llm_context_tokens", 0)
    if context_limit <= 0:
        import llm_backend
        context_limit = llm_backend.get_context_window()
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
        # Keep existing summary on failure — don't destroy what we have
        if not existing_summary:
            # No prior summary: fall back to raw text of compacted messages
            _SUMMARY[session_id] = summary_input

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
    """Return only active (not ended) in-memory session metadata."""
    return {sid: meta for sid, meta in _META.items() if sid not in _ENDED}


def most_recent_active_session() -> str | None:
    """Return session_id of the most recently active session (by last activity), or None."""
    active = {sid: _LAST_ACTIVITY.get(sid, _META[sid].get("created_at", 0))
              for sid in _META if sid not in _ENDED}
    if not active:
        return None
    return max(active, key=active.get)


def ended_sessions() -> dict[str, dict[str, Any]]:
    """Return ended session metadata with ended_at timestamps."""
    return {
        sid: {**_META[sid], "ended_at": _ENDED[sid]}
        for sid in _ENDED
        if sid in _META
    }


def all_sessions() -> dict[str, dict[str, Any]]:
    """Return all in-memory session metadata (active + ended)."""
    return dict(_META)


def end_session(session_id: str) -> bool:
    """Mark a session as ended. Returns True if it was active."""
    if session_id not in _META or session_id in _ENDED:
        return False
    _ENDED[session_id] = time.time()
    bump_generation(session_id)  # invalidate any in-flight turns
    save_session(session_id)
    # Save caller history if keep_history is enabled and caller is known
    if config.get("keep_history", False):
        caller = _CALLER_INFO.get(session_id, {})
        number = caller.get("number", "")
        if number:
            history = _HISTORY.get(session_id, [])
            summary = _SUMMARY.get(session_id, "")
            save_caller_history(number, history, summary)
    # Clean up ALL instruction/knowledge state so nothing bleeds into future sessions
    import instruction_store
    instruction_store.clear_all_for_session(session_id)
    # Drain any pending TTS inject queue
    _INJECT_QUEUE.pop(session_id, None)
    # Release agent takeover if any
    import agent_interface
    for ws in list(agent_interface._AGENTS):
        if session_id in agent_interface._AGENTS[ws].get("takeover_sessions", set()):
            agent_interface._TAKEOVER.pop(session_id, None)
            agent_interface._AGENTS[ws]["takeover_sessions"].discard(session_id)
    return True


def is_ended(session_id: str) -> bool:
    return session_id in _ENDED


def touch(session_id: str) -> None:
    """Update last-activity timestamp for a session."""
    _LAST_ACTIVITY[session_id] = time.time()


def sweep_stale() -> list[str]:
    """Auto-end sessions with no activity for SESSION_TTL seconds. Returns ended IDs."""
    now = time.time()
    ttl = config.get("session_ttl", _SESSION_TTL)
    stale = []
    for sid in list(_META):
        if sid in _ENDED:
            continue
        last = _LAST_ACTIVITY.get(sid, _META[sid].get("created_at", now))
        if now - last > ttl:
            stale.append(sid)
            end_session(sid)
    return stale


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


# ---------------------------------------------------------------------------
# Caller history persistence
# ---------------------------------------------------------------------------

def _normalize_number(number: str) -> str:
    """Normalize a phone number for consistent file naming (strip whitespace, dashes, parens)."""
    return re.sub(r"[\s\-\(\)]", "", number)


def save_caller_history(number: str, history: list, summary: str) -> None:
    """Persist conversation history for a caller number."""
    normalized = _normalize_number(number)
    if not normalized:
        return
    path = _CALLER_HISTORY_DIR / f"{normalized}.json"
    existing = _load_caller_file(path)
    total_calls = (existing.get("total_calls", 0) + 1) if existing else 1
    data = {
        "number": normalized,
        "history": history,
        "summary": summary,
        "last_call_at": time.time(),
        "total_calls": total_calls,
    }
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_caller_history(number: str) -> dict | None:
    """Load persisted caller history. Returns dict with history/summary/total_calls/last_call_at or None."""
    normalized = _normalize_number(number)
    if not normalized:
        return None
    path = _CALLER_HISTORY_DIR / f"{normalized}.json"
    return _load_caller_file(path)


def _load_caller_file(path: Path) -> dict | None:
    """Read a caller history JSON file."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_caller_history(number: str) -> bool:
    """Delete persisted caller history. Returns True if file existed."""
    normalized = _normalize_number(number)
    if not normalized:
        return False
    path = _CALLER_HISTORY_DIR / f"{normalized}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_caller_histories() -> list[dict]:
    """List all saved caller histories with metadata."""
    histories = []
    for path in sorted(_CALLER_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            histories.append({
                "number": data.get("number", path.stem),
                "total_calls": data.get("total_calls", 0),
                "last_call_at": data.get("last_call_at"),
                "turn_count": len(data.get("history", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return histories
