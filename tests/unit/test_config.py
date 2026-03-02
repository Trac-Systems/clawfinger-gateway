"""Unit tests for config module: nested sections, dotted paths, migration, env overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import config


class TestDottedPaths:
    def test_get_nested_value(self, tmp_config):
        assert config.get("llm.model") == "test-llm-model"
        assert config.get("tts.voice") == "am_adam"
        assert config.get("tts.piper.voice") == "thorsten-high"
        assert config.get("phone.auto_answer") is True
        assert config.get("agent.takeover_timeout") == 60

    def test_get_root_level(self, tmp_config):
        assert config.get("host") == "127.0.0.1"
        assert config.get("port") == 8996
        assert config.get("bearer_token") == "test-token"

    def test_get_missing_returns_default(self, tmp_config):
        assert config.get("nonexistent", "fallback") == "fallback"
        assert config.get("llm.nonexistent", 42) == 42

    def test_backward_compat_flat_keys(self, tmp_config):
        """Old flat keys should still resolve via _FLAT_TO_NESTED map."""
        assert config.get("llm_model") == "test-llm-model"
        assert config.get("tts_voice") == "am_adam"
        assert config.get("call_auto_answer") is True
        assert config.get("auth_passphrase") == ""
        assert config.get("piper_voice") == "thorsten-high"

    def test_set_dotted_path(self, tmp_config):
        config.set("llm.temperature", 0.5)
        assert config.get("llm.temperature") == 0.5

    def test_set_creates_nested(self, tmp_config):
        config.set("new.section.key", "value")
        assert config.get("new.section.key") == "value"

    def test_section_returns_dict(self, tmp_config):
        llm = config.section("llm")
        assert isinstance(llm, dict)
        assert llm["model"] == "test-llm-model"
        assert llm["temperature"] == 0.2

    def test_section_missing_returns_empty(self, tmp_config):
        assert config.section("nonexistent") == {}


class TestFlatMigration:
    def test_flat_config_auto_migrates(self, tmp_path):
        """A flat config.json should auto-migrate to nested on load."""
        flat_cfg = {
            "host": "127.0.0.1",
            "port": 8996,
            "bearer_token": "test",
            "mlx_audio_base": "http://localhost:8765",
            "stt_model": "test-stt",
            "stt_language": "en",
            "tts_model": "test-tts",
            "tts_voice": "af_heart",
            "tts_speed": 1.0,
            "llm_model": "test-llm",
            "llm_base_url": "",
            "llm_api_key": "",
            "llm_max_tokens": 200,
            "llm_temperature": 0.3,
            "call_auto_answer": False,
            "caller_blocklist": ["+49111"],
            "auth_passphrase": "secret",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(flat_cfg))

        old_path = config._CFG_PATH
        old_loaded = config._LOADED
        config._CFG_PATH = cfg_path
        config._LOADED = {}
        try:
            cfg = config.load()
            # Should be nested now
            assert config.get("asr.backend") == "http://localhost:8765"
            assert config.get("asr.model") == "test-stt"
            assert config.get("tts.voice") == "af_heart"
            assert config.get("llm.model") == "test-llm"
            assert config.get("llm.max_tokens") == 200
            assert config.get("phone.auto_answer") is False
            assert config.get("phone.caller_blocklist") == ["+49111"]
            assert config.get("phone.auth_passphrase") == "secret"
            # Robot defaults should be applied
            assert config.get("robot.enabled") is False
            assert config.get("robot.model") == "unitree_g1"
        finally:
            config._CFG_PATH = old_path
            config._LOADED = old_loaded

    def test_old_split_llm_fields_migrate(self, tmp_path):
        """Old llm_backend/llm_local_model/llm_remote_model should migrate."""
        flat_cfg = {
            "host": "127.0.0.1",
            "port": 8996,
            "llm_backend": "remote",
            "llm_remote_model": "gpt-4o",
            "llm_remote_base_url": "https://api.openai.com/v1",
            "llm_remote_api_key": "sk-test",
            "stt_model": "test",
            "tts_voice": "test",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(flat_cfg))

        old_path = config._CFG_PATH
        old_loaded = config._LOADED
        config._CFG_PATH = cfg_path
        config._LOADED = {}
        try:
            config.load()
            assert config.get("llm.model") == "gpt-4o"
            assert config.get("llm.base_url") == "https://api.openai.com/v1"
            assert config.get("llm.api_key") == "sk-test"
        finally:
            config._CFG_PATH = old_path
            config._LOADED = old_loaded


class TestSaveReload:
    def test_save_writes_nested(self, tmp_config):
        cfg_path, _ = tmp_config
        config.set("llm.temperature", 0.9)
        config.save()

        raw = json.loads(cfg_path.read_text())
        assert raw["llm"]["temperature"] == 0.9

    def test_reload_picks_up_changes(self, tmp_config):
        cfg_path, _ = tmp_config
        raw = json.loads(cfg_path.read_text())
        raw["llm"]["model"] = "changed-model"
        cfg_path.write_text(json.dumps(raw))
        config.reload()
        assert config.get("llm.model") == "changed-model"


class TestDefaults:
    def test_robot_defaults_applied(self, tmp_config):
        assert config.get("robot.enabled") is False
        assert config.get("robot.model") == "unitree_g1"
        assert config.get("robot.transport") == "intercom"
        assert config.get("robot.unitree_g1.jetson_ip") == "192.168.123.164"
        assert config.get("robot.unitree_g1.max_speed") == 0.5

    def test_phone_defaults_applied(self, tmp_config):
        assert config.get("phone.enabled") is True
        assert isinstance(config.get("phone.adb_path"), str)

    def test_agent_defaults_applied(self, tmp_config):
        assert config.get("agent.takeover_timeout") == 60
