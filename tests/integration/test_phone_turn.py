"""Integration tests for POST /api/turn."""

from __future__ import annotations

import io

import pytest


class TestPhoneTurn:
    def test_basic_turn(self, client):
        c, mocks = client
        audio = io.BytesIO(mocks["fake_wav"])
        resp = c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "int-test-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["session_id"] == "int-test-1"
        assert body["transcript"] == "Hello from test"
        assert body["reply"] == "Test reply from LLM"
        assert "audio_base64" in body
        assert "metrics" in body

    def test_forced_reply_skips_asr_llm(self, client):
        c, mocks = client
        audio = io.BytesIO(mocks["fake_wav"])
        resp = c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "int-test-2", "forced_reply": "Forced response"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["reply"] == "Forced response"
        assert body["transcript"] == ""

    def test_ended_session_rejected(self, client):
        c, mocks = client
        import session_store
        sid = session_store.get_or_create("ended-test")
        session_store.end_session(sid)

        audio = io.BytesIO(mocks["fake_wav"])
        resp = c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "ended-test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body.get("ended") is True

    def test_blocklisted_caller_rejected(self, client):
        c, mocks = client
        import config
        config.set("phone.caller_blocklist", ["+49123456"])
        config.save()

        audio = io.BytesIO(mocks["fake_wav"])
        resp = c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "block-test", "caller_number": "+49123456"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is False
        assert body.get("rejected") is True
        assert body.get("reason") == "blocklisted"

    def test_reset_session(self, client):
        c, mocks = client
        # First turn
        audio = io.BytesIO(mocks["fake_wav"])
        c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "reset-test"},
        )
        # Reset
        audio = io.BytesIO(mocks["fake_wav"])
        resp = c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "reset-test", "reset_session": "true"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
