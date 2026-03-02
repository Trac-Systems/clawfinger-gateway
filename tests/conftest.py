"""Shared fixtures for gateway tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Add gateway root to sys.path so we can import modules
_GATEWAY_ROOT = Path(__file__).resolve().parent.parent
if str(_GATEWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_GATEWAY_ROOT))


@pytest.fixture()
def tmp_config(tmp_path: Path):
    """Create a temporary nested config.json and patch config module to use it."""
    cfg_path = tmp_path / "config.json"
    default_cfg = {
        "host": "127.0.0.1",
        "port": 8996,
        "bearer_token": "test-token",
        "asr": {
            "backend": "http://127.0.0.1:8765",
            "model": "test-asr-model",
            "language": "en",
        },
        "tts": {
            "model": "test-tts-model",
            "voice": "am_adam",
            "speed": 1.2,
            "lang": "en",
            "piper": {
                "base": "http://127.0.0.1:5123",
                "voice": "thorsten-high",
                "speaker": 0,
                "length_scale": 1.0,
                "noise_scale": 0.667,
                "noise_w": 0.8,
                "sentence_silence": 0.2,
            },
        },
        "llm": {
            "model": "test-llm-model",
            "base_url": "",
            "api_key": "",
            "max_tokens": 400,
            "temperature": 0.2,
            "top_p": 1.0,
            "top_p_enabled": True,
            "top_k": 0,
            "top_k_enabled": True,
            "repeat_penalty": 1.0,
            "stop": [],
            "context_tokens": 0,
            "system_prompt": "Test system prompt.",
            "max_history_turns": 8,
        },
        "phone": {
            "enabled": True,
            "adb_path": "/usr/bin/adb",
            "auto_answer": True,
            "auto_answer_delay_ms": 500,
            "caller_allowlist": [],
            "caller_blocklist": [],
            "unknown_callers_allowed": True,
            "greeting_incoming": "Hello from test.",
            "greeting_outgoing": "Test outgoing.",
            "greeting_owner": "tester",
            "max_duration_sec": 300,
            "max_duration_message": "Max duration reached.",
            "auth_passphrase": "",
            "auth_reject_message": "Rejected.",
            "auth_max_attempts": 3,
            "keep_history": False,
        },
        "robot": {
            "enabled": False,
            "model": "unitree_g1",
        },
        "agent": {
            "takeover_timeout": 60,
        },
    }
    cfg_path.write_text(json.dumps(default_cfg, indent=2))

    import config
    old_path = config._CFG_PATH
    old_loaded = config._LOADED
    config._CFG_PATH = cfg_path
    config._LOADED = {}
    yield cfg_path, default_cfg
    config._CFG_PATH = old_path
    config._LOADED = old_loaded


@pytest.fixture()
def fresh_session_store(tmp_path: Path):
    """Reset session_store state for isolation between tests."""
    import session_store
    old = {
        "_HISTORY": dict(session_store._HISTORY),
        "_META": dict(session_store._META),
        "_CALLER_INFO": dict(session_store._CALLER_INFO),
        "_SUMMARY": dict(session_store._SUMMARY),
        "_AUTH_STATE": dict(session_store._AUTH_STATE),
        "_ENDED": dict(session_store._ENDED),
        "_LAST_ACTIVITY": dict(session_store._LAST_ACTIVITY),
        "_SESSION_LOCKS": dict(session_store._SESSION_LOCKS),
        "_INJECT_QUEUE": dict(session_store._INJECT_QUEUE),
        "_GENERATION": dict(session_store._GENERATION),
    }
    # Redirect disk dirs to temp
    old_sessions_dir = session_store._SESSIONS_DIR
    old_history_dir = session_store._CALLER_HISTORY_DIR
    session_store._SESSIONS_DIR = tmp_path / "sessions"
    session_store._SESSIONS_DIR.mkdir()
    session_store._CALLER_HISTORY_DIR = tmp_path / "caller_history"
    session_store._CALLER_HISTORY_DIR.mkdir()

    session_store._HISTORY.clear()
    session_store._META.clear()
    session_store._CALLER_INFO.clear()
    session_store._SUMMARY.clear()
    session_store._AUTH_STATE.clear()
    session_store._ENDED.clear()
    session_store._LAST_ACTIVITY.clear()
    session_store._SESSION_LOCKS.clear()
    session_store._INJECT_QUEUE.clear()
    session_store._GENERATION.clear()
    yield session_store
    # Restore
    session_store._SESSIONS_DIR = old_sessions_dir
    session_store._CALLER_HISTORY_DIR = old_history_dir
    for attr, val in old.items():
        getattr(session_store, attr).update(val)
