"""In-memory instruction storage with base / session / turn layers."""

from __future__ import annotations

import hashlib

import config

_BASE: str = ""
_SESSION: dict[str, str] = {}
_TURN: dict[str, str] = {}

_COMPACT_THRESHOLD = 1500  # characters — rough proxy for ~400 tokens
_COMPACT_CACHE: dict[str, str] = {}  # {md5_hash: compacted_text}


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
        raw = base + "\n\n" + turn_extra
    else:
        raw = base

    if len(raw) > _COMPACT_THRESHOLD:
        return _compact(raw)
    return raw


def _compact(text: str) -> str:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cached = _COMPACT_CACHE.get(text_hash)
    if cached:
        return cached

    import llm_backend

    messages = [
        {
            "role": "system",
            "content": (
                "Compress the following instructions into a concise version. "
                "Preserve all actionable directives, constraints, and persona details. "
                "Remove redundancy. Output only the compressed instructions, nothing else."
            ),
        },
        {"role": "user", "content": text},
    ]
    compacted, _, _ = llm_backend.generate(messages)
    _COMPACT_CACHE[text_hash] = compacted

    # Keep cache small
    if len(_COMPACT_CACHE) > 50:
        oldest = next(iter(_COMPACT_CACHE))
        del _COMPACT_CACHE[oldest]

    return compacted


def snapshot() -> dict:
    return {
        "base": get_base(),
        "sessions": dict(_SESSION),
    }
