"""Integration tests for robot perception REST endpoints."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
FAKE_JPEG_B64 = base64.b64encode(FAKE_JPEG).decode()


class TestPerceptionSources:
    def test_list_sources(self, client):
        c, _ = client
        resp = c.get("/api/robot/perception")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "unitree_g1"
        assert isinstance(body["cameras"], list)
        assert isinstance(body["microphones"], list)
        assert len(body["cameras"]) == 2
        assert body["cameras"][0]["id"] == "head_rgb"
        assert body["cameras"][1]["id"] == "head_depth"
        assert len(body["microphones"]) == 1
        assert body["microphones"][0]["id"] == "mic_array"


class TestSnapshot:
    def test_snapshot_not_connected(self, client):
        c, _ = client
        resp = c.post("/api/robot/camera/snapshot", json={"source": "head_rgb"})
        assert resp.status_code == 503

    @patch("app._intercom_bridge")
    def test_snapshot_success(self, mock_bridge, client):
        c, _ = client
        # Mark robot as connected
        from endpoints import robot as robot_mod
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            mock_bridge.is_connected.return_value = True
            mock_bridge.send_and_wait = AsyncMock(return_value={
                "ok": True,
                "image_base64": FAKE_JPEG_B64,
                "width": 640,
                "height": 480,
            })
            resp = c.post("/api/robot/camera/snapshot", json={"source": "head_rgb"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert body["image_base64"] == FAKE_JPEG_B64
            assert body["source"] == "head_rgb"
        finally:
            robot_mod.set_connected("unitree_g1", False)

    @patch("app._intercom_bridge")
    def test_snapshot_invalid_source(self, mock_bridge, client):
        c, _ = client
        from endpoints import robot as robot_mod
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            resp = c.post("/api/robot/camera/snapshot", json={"source": "nonexistent"})
            assert resp.status_code == 400
            body = resp.json()
            assert "nonexistent" in body["detail"]
            assert "head_rgb" in body["detail"]
        finally:
            robot_mod.set_connected("unitree_g1", False)

    @patch("app._intercom_bridge")
    def test_snapshot_defaults_to_first_camera(self, mock_bridge, client):
        c, _ = client
        from endpoints import robot as robot_mod
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            mock_bridge.send_and_wait = AsyncMock(return_value={
                "ok": True, "image_base64": FAKE_JPEG_B64,
                "width": 640, "height": 480,
            })
            resp = c.post("/api/robot/camera/snapshot", json={})
            assert resp.status_code == 200
            body = resp.json()
            assert body["source"] == "head_rgb"
        finally:
            robot_mod.set_connected("unitree_g1", False)


class TestDescribe:
    @patch("app._intercom_bridge")
    def test_describe_success(self, mock_bridge, client):
        c, _ = client
        from endpoints import robot as robot_mod
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            mock_bridge.send_and_wait = AsyncMock(return_value={
                "ok": True,
                "description": "A room with a table and chairs.",
                "image_base64": FAKE_JPEG_B64,
            })
            resp = c.post("/api/robot/camera/describe", json={
                "source": "head_rgb",
                "prompt": "What do you see?",
            })
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert "table" in body["description"]
        finally:
            robot_mod.set_connected("unitree_g1", False)


class TestStreamControl:
    @patch("app._intercom_bridge")
    def test_stream_start_stop(self, mock_bridge, client):
        c, _ = client
        from endpoints import robot as robot_mod
        from endpoints.robot import perception as perc
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            mock_bridge.send = AsyncMock()
            # Start
            resp = c.post("/api/robot/camera/stream/start", json={"source": "head_rgb"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert perc.is_streaming("head_rgb")

            # Already streaming
            resp = c.post("/api/robot/camera/stream/start", json={"source": "head_rgb"})
            assert resp.json()["detail"] == "already streaming"

            # Stop
            resp = c.post("/api/robot/camera/stream/stop", json={"source": "head_rgb"})
            assert resp.status_code == 200
            assert not perc.is_streaming("head_rgb")
        finally:
            perc.stop_all()
            robot_mod.set_connected("unitree_g1", False)

    @patch("app._intercom_bridge")
    def test_stream_get_requires_active(self, mock_bridge, client):
        c, _ = client
        from endpoints import robot as robot_mod
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            resp = c.get("/api/robot/camera/stream?source=head_rgb")
            assert resp.status_code == 409
        finally:
            robot_mod.set_connected("unitree_g1", False)


class TestAudioMonitor:
    @patch("app._intercom_bridge")
    def test_audio_monitor_start_stop(self, mock_bridge, client):
        c, _ = client
        from endpoints import robot as robot_mod
        from endpoints.robot import perception as perc
        robot_mod.load_models()
        robot_mod.set_connected("unitree_g1", True)
        try:
            mock_bridge.send = AsyncMock()
            # Start
            resp = c.post("/api/robot/audio/monitor/start", json={"source": "mic_array"})
            assert resp.status_code == 200
            assert resp.json()["ok"] is True
            assert perc.is_monitoring("mic_array")

            # Already monitoring
            resp = c.post("/api/robot/audio/monitor/start", json={"source": "mic_array"})
            assert resp.json()["detail"] == "already monitoring"

            # Stop
            resp = c.post("/api/robot/audio/monitor/stop", json={"source": "mic_array"})
            assert resp.status_code == 200
            assert not perc.is_monitoring("mic_array")
        finally:
            perc.stop_all()
            robot_mod.set_connected("unitree_g1", False)
