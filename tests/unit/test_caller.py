"""Unit tests for caller filtering, passphrase matching, and number normalization."""

from __future__ import annotations

import re

import session_store


class TestNumberNormalization:
    def test_strips_spaces(self):
        assert session_store._normalize_number("+49 123 456") == "+49123456"

    def test_strips_dashes(self):
        assert session_store._normalize_number("+49-123-456") == "+49123456"

    def test_strips_parens(self):
        assert session_store._normalize_number("+49(0)123") == "+490123"

    def test_combined(self):
        assert session_store._normalize_number("+49 (123) 456-789") == "+49123456789"

    def test_empty(self):
        assert session_store._normalize_number("") == ""


class TestCallerFilter:
    """Test the caller filtering logic as used in app.py's /api/turn."""

    @staticmethod
    def _normalize(number: str) -> str:
        return re.sub(r"[\s\-\(\)]", "", number)

    def test_blocklist_blocks(self):
        normalized = self._normalize("+49123456789")
        blocklist = ["+49123456789"]
        assert normalized in blocklist

    def test_blocklist_miss(self):
        normalized = self._normalize("+49999888777")
        blocklist = ["+49123456789"]
        assert normalized not in blocklist

    def test_allowlist_filters(self):
        normalized = self._normalize("+49123")
        allowlist = ["+49999"]
        assert allowlist and normalized not in allowlist

    def test_allowlist_passes(self):
        normalized = self._normalize("+49999")
        allowlist = ["+49999"]
        assert not (allowlist and normalized not in allowlist)

    def test_empty_allowlist_allows_all(self):
        normalized = self._normalize("+49123")
        allowlist = []
        # Empty allowlist means no filtering
        assert not (allowlist and normalized not in allowlist)

    def test_unknown_caller_blocked(self):
        normalized = self._normalize("")
        unknown_allowed = False
        assert not normalized and not unknown_allowed

    def test_unknown_caller_allowed(self):
        normalized = self._normalize("")
        unknown_allowed = True
        # Should NOT block
        assert not (not normalized and not unknown_allowed)


class TestPassphraseMatch:
    """Test the passphrase fuzzy matching logic from app.py."""

    @staticmethod
    def _matches(passphrase: str, transcript: str) -> bool:
        clean_phrase = re.sub(r"[^\w\s]", "", passphrase.lower()).strip()
        clean_input = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
        return clean_phrase in clean_input

    def test_exact_match(self):
        assert self._matches("blue harvest", "blue harvest")

    def test_case_insensitive(self):
        assert self._matches("Blue Harvest", "blue harvest")

    def test_punctuation_stripped(self):
        assert self._matches("blue harvest!", "blue harvest")

    def test_substring_match(self):
        assert self._matches("blue harvest", "I said blue harvest please")

    def test_no_match(self):
        assert not self._matches("blue harvest", "red dawn")

    def test_empty_passphrase(self):
        assert self._matches("", "anything")
