"""Unit tests for robot capability registry and model loading."""

from __future__ import annotations

import pytest

from endpoints import robot as robot_mod


class TestCapabilityRegistry:
    def setup_method(self):
        """Load models before each test."""
        robot_mod.load_models()

    def test_g1_registered(self):
        assert "unitree_g1" in robot_mod.list_models()

    def test_g1_capabilities(self):
        caps = robot_mod.get_capabilities("unitree_g1")
        assert "locomotion" in caps
        assert "posture" in caps
        assert "manipulation" in caps
        assert "gesture" in caps
        assert "vision" in caps
        assert "audio" in caps
        assert "dexterous_hands" in caps

    def test_has_capability(self):
        assert robot_mod.has_capability("unitree_g1", "locomotion")
        assert not robot_mod.has_capability("unitree_g1", "flight")

    def test_unknown_model(self):
        assert robot_mod.get_model("nonexistent") is None
        assert robot_mod.get_capabilities("nonexistent") == set()

    def test_g1_defaults(self):
        model = robot_mod.get_model("unitree_g1")
        assert model is not None
        defaults = model["defaults"]
        assert defaults["jetson_ip"] == "192.168.123.164"
        assert defaults["locomotion_ip"] == "192.168.123.161"
        assert defaults["dds_domain"] == 0
        assert defaults["max_speed"] == 0.5
        assert defaults["enable_hands"] is True
        assert defaults["enable_low_level"] is False


class TestCommandDispatch:
    def setup_method(self):
        robot_mod.load_models()

    @pytest.mark.asyncio
    async def test_not_connected_returns_error(self):
        result = await robot_mod.dispatch_command("unitree_g1", {"type": "walk"})
        assert result["ok"] is False
        assert "not connected" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_unknown_model_returns_error(self):
        result = await robot_mod.dispatch_command("nonexistent", {"type": "walk"})
        assert result["ok"] is False
        assert "Unknown model" in result["error"]


