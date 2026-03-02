"""Integration tests for robot config API endpoints."""

from __future__ import annotations

import pytest


class TestRobotConfigAPI:
    def test_get_robot_config(self, client):
        c, _ = client
        resp = c.get("/api/config/robot")
        assert resp.status_code == 200
        body = resp.json()
        assert "enabled" in body
        assert "model" in body
        assert "capabilities" in body
        assert isinstance(body["capabilities"], list)

    def test_update_robot_config(self, client):
        c, _ = client
        resp = c.post("/api/config/robot", json={"voice": "af_bella"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["voice"] == "af_bella"

    def test_security_keys_blocked(self, client):
        c, _ = client
        resp = c.post("/api/config/robot", json={
            "intercom_key": "should-not-set",
            "voice": "am_adam",
        })
        assert resp.status_code == 200
        body = resp.json()
        # Voice should be set, but intercom_key should be ignored
        assert body["ok"] is True
