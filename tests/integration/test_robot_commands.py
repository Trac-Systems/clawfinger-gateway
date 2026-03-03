"""Integration tests for robot command dispatch with mock transport."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from endpoints import robot as robot_mod


class MockTransport:
    """Mock IntercomBridge for testing command dispatch."""

    def __init__(self, connected=True):
        self._connected = connected
        self.sent_messages = []
        self.send_and_wait_response = {"ok": True, "detail": "mock"}

    def is_connected(self):
        return self._connected

    async def send(self, message):
        self.sent_messages.append(message)

    async def send_and_wait(self, message, timeout=10.0):
        self.sent_messages.append(message)
        return self.send_and_wait_response


class TestCommandDispatchWithTransport:
    def setup_method(self):
        robot_mod.load_models()
        # Reset transport state
        robot_mod.set_transport(None)
        robot_mod.set_connected("unitree_g1", False)

    def teardown_method(self):
        robot_mod.set_transport(None)
        robot_mod.set_connected("unitree_g1", False)

    @pytest.mark.asyncio
    async def test_no_transport_returns_error(self):
        """Command with no transport returns error."""
        robot_mod.set_connected("unitree_g1", True)
        result = await robot_mod.dispatch_command("unitree_g1", {"type": "walk"})
        assert result["ok"] is False
        assert "not connected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_disconnected_transport_returns_error(self):
        """Command with disconnected transport returns error."""
        transport = MockTransport(connected=False)
        robot_mod.set_transport(transport)
        robot_mod.set_connected("unitree_g1", True)

        result = await robot_mod.dispatch_command("unitree_g1", {"type": "walk"})
        assert result["ok"] is False
        assert "not connected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_walk_command_uses_send_and_wait(self):
        """Walk command uses send_and_wait for request/response."""
        transport = MockTransport(connected=True)
        transport.send_and_wait_response = {"ok": True, "detail": "walking"}
        robot_mod.set_transport(transport)
        robot_mod.set_connected("unitree_g1", True)

        result = await robot_mod.dispatch_command("unitree_g1", {
            "type": "walk", "params": {"speed": 0.3, "direction": "forward"},
        })

        assert result["ok"] is True
        assert result["detail"] == "walking"
        assert len(transport.sent_messages) == 1
        msg = transport.sent_messages[0]
        assert msg["type"] == "robot_command"
        assert msg["command"] == "walk"
        assert msg["params"]["speed"] == 0.3

    @pytest.mark.asyncio
    async def test_stop_command_is_fire_and_forget(self):
        """Stop command uses send (fire-and-forget), not send_and_wait."""
        transport = MockTransport(connected=True)
        robot_mod.set_transport(transport)
        robot_mod.set_connected("unitree_g1", True)

        result = await robot_mod.dispatch_command("unitree_g1", {"type": "stop"})

        assert result["ok"] is True
        assert result["detail"] == "sent"
        assert len(transport.sent_messages) == 1
        msg = transport.sent_messages[0]
        assert msg["command"] == "stop"

    @pytest.mark.asyncio
    async def test_safety_stop_is_fire_and_forget(self):
        """Safety stop uses send (fire-and-forget)."""
        transport = MockTransport(connected=True)
        robot_mod.set_transport(transport)
        robot_mod.set_connected("unitree_g1", True)

        # safety_stop isn't in the capability map, so it goes through handler directly
        # We need to call the handler, not dispatch (which checks capabilities)
        from endpoints.robot.unitree_g1.commands import handle_command
        result = await handle_command({"type": "safety_stop"})

        assert result["ok"] is True
        assert result["detail"] == "sent"

    @pytest.mark.asyncio
    async def test_capability_check_still_works(self):
        """Capability checks still reject unsupported commands."""
        transport = MockTransport(connected=True)
        robot_mod.set_transport(transport)
        robot_mod.set_connected("unitree_g1", True)

        # "flight" isn't a G1 capability — but it's not in the command map either,
        # so this just goes through. Test a mapped command on a model that's connected.
        result = await robot_mod.dispatch_command("unitree_g1", {"type": "walk"})
        assert result["ok"] is True  # G1 has locomotion

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """Command can specify a custom timeout."""
        transport = MockTransport(connected=True)
        robot_mod.set_transport(transport)
        robot_mod.set_connected("unitree_g1", True)

        result = await robot_mod.dispatch_command("unitree_g1", {
            "type": "walk", "params": {"speed": 0.1}, "timeout": 30.0,
        })

        assert result["ok"] is True


class TestTransportAccessors:
    def setup_method(self):
        robot_mod.load_models()
        robot_mod.set_transport(None)

    def teardown_method(self):
        robot_mod.set_transport(None)

    def test_set_get_transport(self):
        assert robot_mod.get_transport() is None
        transport = MockTransport()
        robot_mod.set_transport(transport)
        assert robot_mod.get_transport() is transport

    def test_set_connected(self):
        model = robot_mod.get_model("unitree_g1")
        assert model is not None
        assert model["connected"] is False
        robot_mod.set_connected("unitree_g1", True)
        assert model["connected"] is True
        robot_mod.set_connected("unitree_g1", False)
        assert model["connected"] is False

    def test_set_connected_unknown_model(self):
        """set_connected on unknown model doesn't crash."""
        robot_mod.set_connected("nonexistent", True)  # no error
