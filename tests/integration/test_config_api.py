"""Integration tests for config API endpoints."""

from __future__ import annotations

import pytest


class TestTtsConfig:
    def test_get_tts_config(self, client):
        c, _ = client
        resp = c.get("/api/config/tts")
        assert resp.status_code == 200
        body = resp.json()
        assert "voice" in body or "piper_voice" in body
        assert "lang" in body

    def test_update_tts_voice(self, client):
        c, _ = client
        resp = c.post("/api/config/tts", json={"voice": "af_bella"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["voice"] == "af_bella"


class TestLlmConfig:
    def test_get_llm_config(self, client):
        c, _ = client
        resp = c.get("/api/config/llm")
        assert resp.status_code == 200
        body = resp.json()
        assert "model" in body
        assert "max_tokens" in body
        assert "temperature" in body

    def test_update_llm_temperature(self, client):
        c, _ = client
        resp = c.post("/api/config/llm", json={"temperature": 0.8})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["temperature"] == 0.8


class TestCallConfig:
    def test_get_call_config(self, client):
        c, _ = client
        resp = c.get("/api/config/call")
        assert resp.status_code == 200
        body = resp.json()
        assert "auto_answer" in body
        assert "greeting_owner" in body

    def test_update_call_config(self, client):
        c, _ = client
        resp = c.post("/api/config/call", json={"greeting_owner": "Alice"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["greeting_owner"] == "Alice"

    def test_legacy_flat_keys_accepted(self, client):
        c, _ = client
        resp = c.post("/api/config/call", json={"call_auto_answer": False})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["auto_answer"] is False


class TestHealth:
    def test_health_check(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True


class TestStatus:
    def test_system_status(self, client):
        c, _ = client
        resp = c.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "uptime_s" in body
        assert "config" in body
        # Sensitive keys should be filtered
        cfg = body["config"]
        assert "bearer_token" not in cfg
