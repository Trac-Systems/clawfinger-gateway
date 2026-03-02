"""Unit tests for voice_pipeline pure functions (no server needed)."""

from __future__ import annotations

import voice_pipeline


class TestSafeText:
    def test_strips_nulls(self):
        assert voice_pipeline.safe_text("hello\x00world") == "hello world"

    def test_normalizes_smart_quotes(self):
        result = voice_pipeline.safe_text("\u201cHello\u201d")
        assert '"' in result
        assert "\u201c" not in result

    def test_normalizes_whitespace(self):
        assert voice_pipeline.safe_text("  hello   world  ") == "hello world"

    def test_em_dash_to_hyphen(self):
        assert voice_pipeline.safe_text("hello\u2014world") == "hello-world"

    def test_ellipsis(self):
        assert voice_pipeline.safe_text("wait\u2026") == "wait..."

    def test_empty_string(self):
        assert voice_pipeline.safe_text("") == ""


class TestTrimForTts:
    def test_removes_think_tags(self):
        result = voice_pipeline.trim_for_tts("Hello <think>internal thought</think> world")
        assert "think" not in result
        assert "internal" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_markdown_links(self):
        result = voice_pipeline.trim_for_tts("Check [this link](https://example.com) out")
        assert "this link" in result
        assert "https" not in result
        assert "[" not in result

    def test_removes_markdown_formatting(self):
        result = voice_pipeline.trim_for_tts("**bold** and `code` and _italic_")
        assert "*" not in result
        assert "`" not in result
        assert "bold" in result

    def test_removes_special_chars(self):
        result = voice_pipeline.trim_for_tts("Hello! How are you? 😊")
        assert "Hello" in result
        assert "How are you" in result
        # Emoji should be stripped
        assert "😊" not in result

    def test_preserves_basic_punctuation(self):
        result = voice_pipeline.trim_for_tts("Hello, world! How are you?")
        assert result == "Hello, world! How are you?"
