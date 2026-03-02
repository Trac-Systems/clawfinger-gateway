"""Unit tests for session_store module."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import session_store


class TestSessionLifecycle:
    def test_get_or_create_new(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create()
        assert isinstance(sid, str)
        assert len(sid) == 32  # uuid hex
        assert sid in ss._META
        assert sid in ss._HISTORY

    def test_get_or_create_existing(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid1 = ss.get_or_create("test-session-1")
        sid2 = ss.get_or_create("test-session-1")
        assert sid1 == sid2

    def test_reset_clears_state(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("s1")
        ss.append(sid, "user", "hello")
        ss.set_caller_info(sid, "+49123", "incoming")
        ss.reset(sid)
        assert ss.get_history(sid) == []
        assert ss.get_caller_info(sid) == {"number": "", "direction": ""}

    def test_end_session(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("s1")
        assert ss.end_session(sid) is True
        assert ss.is_ended(sid) is True
        # Second end returns False
        assert ss.end_session(sid) is False


class TestHistory:
    def test_append_and_get(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("s1")
        ss.append(sid, "user", "hello")
        ss.append(sid, "assistant", "hi there")
        history = ss.get_history(sid)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "hello"}
        assert history[1] == {"role": "assistant", "content": "hi there"}


class TestGeneration:
    def test_generation_starts_at_zero(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        assert ss.get_generation("nonexistent") == 0

    def test_bump_increments(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        g1 = ss.bump_generation("s1")
        g2 = ss.bump_generation("s1")
        assert g1 == 1
        assert g2 == 2


class TestInjectQueue:
    def test_queue_and_drain(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        ss.queue_inject("s1", "hello", "base64data")
        item = ss.drain_inject("s1")
        assert item == {"text": "hello", "audio_base64": "base64data"}
        assert ss.drain_inject("s1") is None

    def test_fifo_order(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        ss.queue_inject("s1", "first", "a1")
        ss.queue_inject("s1", "second", "a2")
        assert ss.drain_inject("s1")["text"] == "first"
        assert ss.drain_inject("s1")["text"] == "second"


class TestCallerInfo:
    def test_set_and_get(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        ss.set_caller_info("s1", "+49123", "incoming")
        info = ss.get_caller_info("s1")
        assert info["number"] == "+49123"
        assert info["direction"] == "incoming"

    def test_unknown_session(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        info = ss.get_caller_info("nonexistent")
        assert info == {"number": "", "direction": ""}


class TestAuthState:
    def test_not_authenticated_by_default(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        assert ss.is_authenticated("s1") is False

    def test_mark_authenticated(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        ss.mark_authenticated("s1")
        assert ss.is_authenticated("s1") is True

    def test_record_auth_attempt(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        assert ss.record_auth_attempt("s1") == 1
        assert ss.record_auth_attempt("s1") == 2


class TestCallerHistory:
    def test_normalize_number(self, tmp_config):
        assert session_store._normalize_number("+49 (123) 456-789") == "+49123456789"

    def test_save_and_load(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        ss.save_caller_history("+49123", history, "Test summary")
        loaded = ss.load_caller_history("+49123")
        assert loaded is not None
        assert loaded["history"] == history
        assert loaded["summary"] == "Test summary"
        assert loaded["total_calls"] == 1

    def test_delete_history(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        ss.save_caller_history("+49999", [{"role": "user", "content": "x"}], "")
        assert ss.delete_caller_history("+49999") is True
        assert ss.load_caller_history("+49999") is None


class TestStaleSweep:
    def test_sweep_stale_sessions(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("stale-test")
        # Set activity to far in the past
        ss._LAST_ACTIVITY[sid] = time.time() - 9999
        stale = ss.sweep_stale()
        assert sid in stale
        assert ss.is_ended(sid)
