"""Integration tests for session lifecycle."""

from __future__ import annotations

import io

import pytest


class TestSessionLifecycle:
    def test_create_session(self, client):
        c, _ = client
        resp = c.post("/api/session/new")
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert len(body["session_id"]) > 0

    def test_list_sessions(self, client):
        c, mocks = client
        # Create a turn to persist a session
        audio = io.BytesIO(mocks["fake_wav"])
        c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "list-test"},
        )
        resp = c.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        assert isinstance(sessions, list)

    def test_get_session_detail(self, client):
        c, mocks = client
        audio = io.BytesIO(mocks["fake_wav"])
        c.post(
            "/api/turn",
            files={"audio": ("test.wav", audio, "audio/wav")},
            data={"session_id": "detail-test"},
        )
        resp = c.get("/api/sessions/detail-test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "detail-test"
        assert len(body["turns"]) >= 1

    def test_end_session(self, client):
        c, _ = client
        import session_store
        session_store.get_or_create("end-test")
        resp = c.post("/api/session/end", json={"session_id": "end-test"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert session_store.is_ended("end-test")

    def test_reset_session(self, client):
        c, _ = client
        import session_store
        session_store.get_or_create("reset-test")
        session_store.append("reset-test", "user", "hi")
        resp = c.post("/api/session/reset", data={"session_id": "reset-test"})
        assert resp.status_code == 200
        # History should be cleared after reset
        assert session_store.get_history("reset-test") == []

    def test_nonexistent_session_returns_404(self, client):
        c, _ = client
        resp = c.get("/api/sessions/does-not-exist")
        assert resp.status_code == 404
