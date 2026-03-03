"""Unit tests for LLM backend raw flag and multimodal content support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import llm_backend


class TestFlattenContent:
    def test_string_passthrough(self):
        assert llm_backend._flatten_content("hello") == "hello"

    def test_text_array(self):
        content = [
            {"type": "text", "text": "First part."},
            {"type": "text", "text": "Second part."},
        ]
        result = llm_backend._flatten_content(content)
        assert "First part." in result
        assert "Second part." in result

    def test_image_replaced_with_placeholder(self):
        content = [
            {"type": "text", "text": "Here is the image:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "What do you see?"},
        ]
        result = llm_backend._flatten_content(content)
        assert "[image]" in result
        assert "Here is the image:" in result
        assert "What do you see?" in result

    def test_empty_list(self):
        assert llm_backend._flatten_content([]) == ""

    def test_non_string_fallback(self):
        assert llm_backend._flatten_content(42) == "42"


class TestRawFlag:
    @patch("llm_backend._ensure_local_llm")
    @patch("llm_backend.mlx_generate")
    def test_raw_true_preserves_json(self, mock_gen, mock_ensure, tmp_config):
        """raw=True should use safe_text() only, not trim_for_tts()."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_ensure.return_value = (mock_model, mock_tokenizer)

        json_response = '{"say": "hello", "do": {"action": "look"}, "continue": true}'
        mock_gen.return_value = json_response

        text, _, _ = llm_backend.generate(
            [{"role": "user", "content": "test"}],
            raw=True,
        )
        # JSON characters should be preserved
        assert "{" in text
        assert "}" in text
        assert '"say"' in text
        assert '"action"' in text

    @patch("llm_backend._ensure_local_llm")
    @patch("llm_backend.mlx_generate")
    def test_raw_false_trims_for_tts(self, mock_gen, mock_ensure, tmp_config):
        """raw=False (default) should apply trim_for_tts()."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_ensure.return_value = (mock_model, mock_tokenizer)

        mock_gen.return_value = "Hello! How are you?"

        text, _, _ = llm_backend.generate(
            [{"role": "user", "content": "test"}],
        )
        # Should still be cleaned text
        assert isinstance(text, str)
        assert len(text) > 0

    @patch("llm_backend._ensure_local_llm")
    @patch("llm_backend.mlx_generate")
    def test_empty_response_gets_fallback(self, mock_gen, mock_ensure, tmp_config):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_ensure.return_value = (mock_model, mock_tokenizer)

        mock_gen.return_value = ""

        text, _, _ = llm_backend.generate(
            [{"role": "user", "content": "test"}],
            raw=True,
        )
        assert text == "Got it. Please continue."


class TestMultimodalMessages:
    @patch("llm_backend._ensure_local_llm")
    @patch("llm_backend.mlx_generate")
    def test_multimodal_messages_flattened_for_local(self, mock_gen, mock_ensure, tmp_config):
        """Multimodal content arrays should be flattened for local MLX."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_ensure.return_value = (mock_model, mock_tokenizer)
        mock_gen.return_value = "I see keys."

        messages = [
            {"role": "system", "content": "You are a robot."},
            {"role": "user", "content": [
                {"type": "text", "text": "detect_object result: 1 candidate"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Confidence: 0.82"},
            ]},
        ]

        text, _, _ = llm_backend.generate(messages, raw=True)
        # Should work without error
        assert isinstance(text, str)

        # Verify tokenizer received flattened content
        call_args = mock_tokenizer.apply_chat_template.call_args
        flat_messages = call_args[0][0]
        assert isinstance(flat_messages[1]["content"], str)
        assert "[image]" in flat_messages[1]["content"]
