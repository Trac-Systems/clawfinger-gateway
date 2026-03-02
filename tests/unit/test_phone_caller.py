"""Unit tests for endpoints.phone.caller module."""

from __future__ import annotations

import asyncio

import pytest

from endpoints.phone.caller import (
    check_caller_access,
    make_passphrase_hook,
    normalize_number,
    restore_history,
)


class TestNormalizeNumber:
    def test_strips_spaces(self):
        assert normalize_number("+49 123 456") == "+49123456"

    def test_strips_dashes(self):
        assert normalize_number("+49-123-456") == "+49123456"

    def test_strips_parens(self):
        assert normalize_number("+49(0)123") == "+490123"

    def test_combined(self):
        assert normalize_number("+49 (123) 456-789") == "+49123456789"

    def test_empty(self):
        assert normalize_number("") == ""


class TestCheckCallerAccess:
    def test_blocklist_hit(self):
        cfg = {"caller_blocklist": ["+49123"]}
        result = check_caller_access("+49123", cfg)
        assert result == {"reason": "blocklisted"}

    def test_blocklist_miss(self):
        cfg = {"caller_blocklist": ["+49123"]}
        assert check_caller_access("+49999", cfg) is None

    def test_allowlist_not_in(self):
        cfg = {"caller_allowlist": ["+49111"], "caller_blocklist": []}
        result = check_caller_access("+49999", cfg)
        assert result == {"reason": "not_allowlisted"}

    def test_allowlist_match(self):
        cfg = {"caller_allowlist": ["+49111"], "caller_blocklist": []}
        assert check_caller_access("+49111", cfg) is None

    def test_empty_allowlist_allows_all(self):
        cfg = {"caller_allowlist": [], "caller_blocklist": []}
        assert check_caller_access("+49999", cfg) is None

    def test_unknown_caller_blocked(self):
        cfg = {"caller_blocklist": [], "caller_allowlist": [], "unknown_callers_allowed": False}
        result = check_caller_access("", cfg)
        assert result == {"reason": "unknown_caller"}

    def test_unknown_caller_allowed_by_default(self):
        cfg = {"caller_blocklist": [], "caller_allowlist": []}
        assert check_caller_access("", cfg) is None


class TestMakePassphraseHook:
    def test_no_passphrase_returns_none(self):
        cfg = {"auth_passphrase": ""}
        assert make_passphrase_hook(cfg) is None

    @pytest.mark.asyncio
    async def test_correct_passphrase_authenticates(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("auth-test")
        cfg = {"auth_passphrase": "blue harvest", "auth_max_attempts": 3}
        hook = make_passphrase_hook(cfg)
        assert hook is not None
        result = await hook(sid, "I said blue harvest please")
        assert result is not None
        assert result["reply"] == "Authentication successful. How can I help you?"
        assert ss.is_authenticated(sid)

    @pytest.mark.asyncio
    async def test_wrong_passphrase_increments(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("auth-test2")
        cfg = {"auth_passphrase": "blue harvest", "auth_max_attempts": 3}
        hook = make_passphrase_hook(cfg)
        result = await hook(sid, "red dawn")
        assert result is not None
        assert result["reply"] == "That's not correct. Please try again."
        assert not ss.is_authenticated(sid)

    @pytest.mark.asyncio
    async def test_max_attempts_hangup(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("auth-test3")
        cfg = {
            "auth_passphrase": "blue harvest",
            "auth_max_attempts": 2,
            "auth_reject_message": "Go away.",
        }
        hook = make_passphrase_hook(cfg)
        await hook(sid, "wrong1")
        result = await hook(sid, "wrong2")
        assert result is not None
        assert result["reply"] == "Go away."
        assert result.get("hangup") is True

    @pytest.mark.asyncio
    async def test_already_authenticated_passes_through(self, fresh_session_store, tmp_config):
        ss = fresh_session_store
        sid = ss.get_or_create("auth-test4")
        ss.mark_authenticated(sid)
        cfg = {"auth_passphrase": "blue harvest", "auth_max_attempts": 3}
        hook = make_passphrase_hook(cfg)
        result = await hook(sid, "anything at all")
        assert result is None  # None = continue to LLM
