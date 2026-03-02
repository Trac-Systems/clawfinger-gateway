"""Integration test fixtures — FastAPI TestClient with mocked backends."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_GATEWAY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_GATEWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_GATEWAY_ROOT))


@pytest.fixture()
def mock_backends(tmp_config, fresh_session_store):
    """Patch ASR, LLM, and TTS backends to return fixed results."""
    fake_wav = b"RIFF" + b"\x00" * 40  # minimal fake WAV header

    with (
        patch("voice_pipeline.transcribe", return_value=("Hello from test", 42.0)),
        patch("voice_pipeline.synthesize", return_value=(fake_wav, 15.0)),
        patch("voice_pipeline.check_mlx_audio", return_value={"ok": True}),
        patch("llm_backend.generate", return_value=("Test reply from LLM", 100.0, "test-model")),
        patch("llm_backend.check_health", return_value={"ok": True}),
        patch("llm_backend.preload"),
        patch("llm_backend.get_context_window", return_value=4096),
    ):
        yield {
            "transcript": "Hello from test",
            "reply": "Test reply from LLM",
            "fake_wav": fake_wav,
            "fake_wav_b64": base64.b64encode(fake_wav).decode("ascii"),
        }


class _AuthClient:
    """Thin wrapper around TestClient that injects Bearer token."""

    def __init__(self, client, token):
        self._client = client
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(self, url, **kw):
        kw.setdefault("headers", {}).update(self._headers)
        return self._client.get(url, **kw)

    def post(self, url, **kw):
        kw.setdefault("headers", {}).update(self._headers)
        return self._client.post(url, **kw)

    def delete(self, url, **kw):
        kw.setdefault("headers", {}).update(self._headers)
        return self._client.delete(url, **kw)

    def websocket_connect(self, url, **kw):
        return self._client.websocket_connect(url, **kw)


@pytest.fixture()
def client(mock_backends):
    """FastAPI TestClient with mocked backends + auth header."""
    from starlette.testclient import TestClient
    import app as app_module
    import config

    token = config.get("bearer_token", "")
    with TestClient(app_module.app) as c:
        yield _AuthClient(c, token), mock_backends
