"""Integration tests for robot agent takeover via WebSocket."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


def _recv_until(ws, target_type: str, max_messages: int = 10) -> dict:
    """Receive WS messages until we find the target type (skip event bus noise)."""
    for _ in range(max_messages):
        msg = json.loads(ws.receive_text())
        if msg.get("type") == target_type:
            return msg
    raise AssertionError(f"Never received message type '{target_type}' within {max_messages} messages")


class TestRobotTakeoverWS:
    def test_robot_takeover_ack(self, client):
        """Agent sends robot_takeover, gateway replies with ack."""
        c, _ = client
        from endpoints.robot import controller as robot_ctrl

        with c.websocket_connect("/api/agent/ws") as ws:
            ws.send_json({"type": "robot_takeover"})
            msg = _recv_until(ws, "robot_takeover.ack")
            assert msg["ok"] is True
            assert robot_ctrl.get_takeover_agent() is not None

        # Cleanup
        robot_ctrl.set_takeover_agent(None)

    def test_robot_release_ack(self, client):
        """Agent sends robot_release, gateway replies with ack."""
        c, _ = client
        from endpoints.robot import controller as robot_ctrl

        with c.websocket_connect("/api/agent/ws") as ws:
            # First takeover
            ws.send_json({"type": "robot_takeover"})
            msg = _recv_until(ws, "robot_takeover.ack")
            assert msg["ok"] is True

            # Then release
            ws.send_json({"type": "robot_release"})
            msg = _recv_until(ws, "robot_release.ack")
            assert msg["ok"] is True
            assert robot_ctrl.get_takeover_agent() is None

    def test_robot_release_on_disconnect(self, client):
        """Robot takeover is released when agent WebSocket disconnects."""
        c, _ = client
        from endpoints.robot import controller as robot_ctrl

        with c.websocket_connect("/api/agent/ws") as ws:
            ws.send_json({"type": "robot_takeover"})
            msg = _recv_until(ws, "robot_takeover.ack")
            assert msg["ok"] is True
            assert robot_ctrl.get_takeover_agent() is not None
        # After disconnect
        assert robot_ctrl.get_takeover_agent() is None


class TestRobotSkillListWS:
    def test_robot_skill_list(self, client):
        """Agent requests robot skill list via WS."""
        c, _ = client

        with c.websocket_connect("/api/agent/ws") as ws:
            ws.send_json({"type": "robot_skill_list"})
            msg = _recv_until(ws, "robot.skill.list")
            assert "skills" in msg
            assert isinstance(msg["skills"], list)


class TestRobotProjectStatusWS:
    def test_project_status_idle(self, client):
        """Agent requests project status when idle."""
        c, _ = client

        with c.websocket_connect("/api/agent/ws") as ws:
            ws.send_json({"type": "robot_project_status"})
            msg = _recv_until(ws, "robot.project.status")
            assert msg["status"] == "idle"

    def test_project_cancel_ws(self, client):
        """Agent cancels project via WS."""
        c, _ = client

        with c.websocket_connect("/api/agent/ws") as ws:
            ws.send_json({"type": "robot_project_cancel"})
            msg = _recv_until(ws, "robot.project.cancel.ack")
            assert msg["ok"] is True
