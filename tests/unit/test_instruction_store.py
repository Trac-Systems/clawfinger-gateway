"""Unit tests for instruction_store module."""

from __future__ import annotations

import instruction_store


class TestLayeredInstructions:
    def test_base_from_config(self, tmp_config):
        base = instruction_store.get_base()
        assert base == "Test system prompt."

    def test_session_overrides_base(self, tmp_config):
        instruction_store.set_session("s1", "Session-level instructions")
        prompt = instruction_store.build_system_prompt("s1")
        assert prompt.startswith("Session-level instructions")
        assert "[Current time:" in prompt
        instruction_store.clear_session("s1")

    def test_turn_appended(self, tmp_config):
        instruction_store.set_turn("s1", "Turn extra")
        prompt = instruction_store.build_system_prompt("s1")
        assert "Turn extra" in prompt
        assert "[Current time:" in prompt

    def test_turn_consumed_after_build(self, tmp_config):
        instruction_store.set_turn("s1", "Once only")
        instruction_store.build_system_prompt("s1")
        # Second build should not include turn
        prompt = instruction_store.build_system_prompt("s1")
        assert "Once only" not in prompt

    def test_session_plus_turn(self, tmp_config):
        instruction_store.set_session("s1", "Session base")
        instruction_store.set_turn("s1", "Turn addon")
        prompt = instruction_store.build_system_prompt("s1")
        assert "Session base" in prompt
        assert "Turn addon" in prompt
        instruction_store.clear_session("s1")


class TestAgentKnowledge:
    def test_set_and_get(self, tmp_config):
        instruction_store.set_agent_knowledge("s1", "Caller is John")
        assert instruction_store.get_agent_knowledge("s1") == "Caller is John"
        instruction_store.clear_agent_knowledge("s1")

    def test_clear(self, tmp_config):
        instruction_store.set_agent_knowledge("s1", "data")
        instruction_store.clear_agent_knowledge("s1")
        assert instruction_store.get_agent_knowledge("s1") == ""


class TestClearAll:
    def test_clear_all_for_session(self, tmp_config):
        instruction_store.set_session("s1", "session")
        instruction_store.set_turn("s1", "turn")
        instruction_store.set_agent_knowledge("s1", "knowledge")
        instruction_store.clear_all_for_session("s1")
        assert instruction_store.get_session("s1") == ""
        assert instruction_store.get_turn("s1") == ""
        assert instruction_store.get_agent_knowledge("s1") == ""


class TestSnapshot:
    def test_snapshot(self, tmp_config):
        instruction_store.set_session("s1", "session prompt")
        snap = instruction_store.snapshot()
        assert snap["base"] == "Test system prompt."
        assert "s1" in snap["sessions"]
        instruction_store.clear_session("s1")
