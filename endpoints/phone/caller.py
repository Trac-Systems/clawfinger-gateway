"""Caller filtering, passphrase auth, and caller history logic."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

import session_store
from event_bus import bus


def normalize_number(number: str) -> str:
    """Normalize a phone number — strip whitespace, dashes, parentheses."""
    return re.sub(r"[\s\-\(\)]", "", number)


def check_caller_access(normalized: str, phone_cfg: dict) -> dict | None:
    """Check caller against blocklist/allowlist/unknown policy.

    Returns None if caller is allowed, or a dict with ``reason`` key if rejected.
    """
    blocklist = phone_cfg.get("caller_blocklist", [])
    if normalized and normalized in blocklist:
        return {"reason": "blocklisted"}

    allowlist = phone_cfg.get("caller_allowlist", [])
    if allowlist and normalized and normalized not in allowlist:
        return {"reason": "not_allowlisted"}

    if not normalized and not phone_cfg.get("unknown_callers_allowed", True):
        return {"reason": "unknown_caller"}

    return None


def make_passphrase_hook(
    phone_cfg: dict,
) -> Callable[[str, str], Awaitable[dict | None]] | None:
    """Create a pre-LLM hook for passphrase authentication.

    Returns None if no passphrase is configured.  Otherwise returns an async
    callable ``hook(sid, transcript) -> dict | None``.
    """
    passphrase = phone_cfg.get("auth_passphrase", "")
    if not passphrase:
        return None

    async def _hook(sid: str, transcript: str) -> dict[str, Any] | None:
        if session_store.is_authenticated(sid):
            return None  # already authenticated, continue to LLM

        # Fuzzy match: case-insensitive, strip punctuation, substring
        clean_phrase = re.sub(r"[^\w\s]", "", passphrase.lower()).strip()
        clean_input = re.sub(r"[^\w\s]", "", transcript.lower()).strip()

        if clean_phrase in clean_input:
            session_store.mark_authenticated(sid)
            await bus.publish("turn.authenticated", {"session_id": sid}, session_id=sid)
            return {"reply": "Authentication successful. How can I help you?"}

        attempts = session_store.record_auth_attempt(sid)
        max_attempts = phone_cfg.get("auth_max_attempts", 3)
        await bus.publish("turn.auth_failed", {"session_id": sid, "attempt": attempts}, session_id=sid)

        if max_attempts > 0 and attempts >= max_attempts:
            reject_msg = phone_cfg.get(
                "auth_reject_message",
                "I'm sorry, I can't help you right now. Goodbye.",
            )
            return {"reply": reject_msg, "hangup": True}

        return {"reply": "That's not correct. Please try again."}

    return _hook


def restore_history(
    sid: str,
    caller_number_clean: str,
    phone_cfg: dict,
) -> None:
    """Restore caller history into the session on reset (if keep_history enabled)."""
    normalized = normalize_number(caller_number_clean)
    if not normalized:
        return

    if phone_cfg.get("keep_history", False):
        prev = session_store.load_caller_history(normalized)
        if prev:
            for msg in prev.get("history", []):
                session_store.get_history(sid)  # ensure list exists
                session_store._HISTORY.setdefault(sid, []).append(msg)
            prev_summary = prev.get("summary", "")
            if prev_summary:
                session_store._SUMMARY[sid] = prev_summary
    else:
        session_store.delete_caller_history(normalized)
