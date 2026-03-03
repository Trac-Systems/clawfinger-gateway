"""Unit tests for robot safety validator."""

from __future__ import annotations

import pytest

from endpoints.robot.controller import _validate_command


class TestSpeedClamping:
    def test_walk_within_limit(self, tmp_config):
        cmd = {"type": "walk", "params": {"speed": 0.2, "direction": "forward"}}
        result = _validate_command(cmd)
        assert result is None  # safe
        assert cmd["params"]["speed"] == 0.2  # unchanged

    def test_walk_exceeds_limit_clamped(self, tmp_config):
        cmd = {"type": "walk", "params": {"speed": 1.0, "direction": "forward"}}
        result = _validate_command(cmd)
        assert result is None  # safe (clamped, not rejected)
        assert cmd["params"]["speed"] == 0.3  # clamped to default max


class TestForceClamping:
    def test_grasp_within_limit(self, tmp_config):
        cmd = {"type": "grasp", "params": {"force": 0.5, "hand": "left"}}
        result = _validate_command(cmd)
        assert result is None
        assert cmd["params"]["force"] == 0.5

    def test_grasp_exceeds_limit_clamped(self, tmp_config):
        cmd = {"type": "grasp", "params": {"force": 1.0, "hand": "right"}}
        result = _validate_command(cmd)
        assert result is None
        assert cmd["params"]["force"] == 0.8


class TestReachEnvelope:
    def test_reach_within_limit(self, tmp_config):
        cmd = {"type": "reach", "params": {"x": 0.3, "y": 0.2, "z": 0.1}}
        result = _validate_command(cmd)
        assert result is None

    def test_reach_exceeds_limit_rejected(self, tmp_config):
        cmd = {"type": "reach", "params": {"x": 0.5, "y": 0.5, "z": 0.5}}
        result = _validate_command(cmd)
        assert result is not None
        assert result["ok"] is False
        assert "exceeds limit" in result["error"]

    def test_reach_exactly_at_limit(self, tmp_config):
        # sqrt(0.3^2 + 0.3^2 + 0.3^2) ≈ 0.52 < 0.6
        cmd = {"type": "reach", "params": {"x": 0.3, "y": 0.3, "z": 0.3}}
        result = _validate_command(cmd)
        assert result is None


class TestBlockedCommands:
    def test_blocked_command(self, tmp_config):
        import config
        config.set("robot.safety.blocked_commands", ["self_destruct"])
        cmd = {"type": "self_destruct", "params": {}}
        result = _validate_command(cmd)
        assert result is not None
        assert result["ok"] is False
        assert "blocked" in result["error"]

    def test_non_blocked_command(self, tmp_config):
        import config
        config.set("robot.safety.blocked_commands", ["self_destruct"])
        cmd = {"type": "walk", "params": {"speed": 0.1}}
        result = _validate_command(cmd)
        assert result is None


class TestOtherCommands:
    def test_look_always_safe(self, tmp_config):
        cmd = {"type": "look", "params": {}}
        result = _validate_command(cmd)
        assert result is None

    def test_nod_always_safe(self, tmp_config):
        cmd = {"type": "nod", "params": {}}
        result = _validate_command(cmd)
        assert result is None
