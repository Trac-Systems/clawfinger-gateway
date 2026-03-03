"""Unit tests for robot controller — wake word, voice turns, project lifecycle, agent takeover."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from endpoints.robot import controller as robot_ctrl


@pytest.fixture(autouse=True)
def _reset_state(tmp_config):
    """Reset controller state between tests."""
    robot_ctrl._PROJECT = None
    robot_ctrl._TASK = None
    robot_ctrl._PAUSE_EVENT.set()
    robot_ctrl._LAST_INTERACTION = 0
    robot_ctrl._MESSAGES.clear()
    robot_ctrl._TAKEOVER_AGENT = None
    robot_ctrl._TURN_FUTURES.clear()
    yield
    # Cleanup
    if robot_ctrl._TASK and not robot_ctrl._TASK.done():
        robot_ctrl._TASK.cancel()
    robot_ctrl._PROJECT = None
    robot_ctrl._TASK = None
    robot_ctrl._TAKEOVER_AGENT = None


class TestCheckWakeWord:
    def test_default_wake_word(self):
        accepted, cleaned = robot_ctrl.check_wake_word("Robert find my keys")
        assert accepted is True
        assert cleaned == "find my keys"

    def test_hey_prefix(self):
        accepted, cleaned = robot_ctrl.check_wake_word("Hey Robert, what time is it?")
        assert accepted is True
        assert cleaned == "what time is it?"

    def test_no_wake_word_rejected(self):
        accepted, _ = robot_ctrl.check_wake_word("find my keys")
        assert accepted is False

    def test_stop_always_accepted(self):
        accepted, cleaned = robot_ctrl.check_wake_word("Robert stop")
        assert accepted is True
        assert cleaned == "stop"

    def test_stop_in_middle(self):
        accepted, cleaned = robot_ctrl.check_wake_word("please Robert stop now")
        assert accepted is True
        assert cleaned == "stop"

    def test_conversational_mode_no_wake_word_needed(self):
        robot_ctrl._LAST_INTERACTION = time.time()  # recent interaction
        accepted, cleaned = robot_ctrl.check_wake_word("what about on the shelf?")
        assert accepted is True
        assert cleaned == "what about on the shelf?"

    def test_conversational_mode_expired(self):
        robot_ctrl._LAST_INTERACTION = time.time() - 60  # expired
        accepted, _ = robot_ctrl.check_wake_word("what about on the shelf?")
        assert accepted is False

    def test_project_active_no_wake_word_needed(self):
        robot_ctrl._PROJECT = {"status": "active"}
        accepted, _ = robot_ctrl.check_wake_word("check behind the couch")
        assert accepted is True

    def test_case_insensitive(self):
        accepted, cleaned = robot_ctrl.check_wake_word("ROBERT do something")
        assert accepted is True
        assert "do something" in cleaned.lower()

    def test_hi_prefix(self):
        accepted, _ = robot_ctrl.check_wake_word("Hi Robert, hello!")
        assert accepted is True

    def test_ok_prefix(self):
        accepted, _ = robot_ctrl.check_wake_word("OK Robert, listen")
        assert accepted is True

    def test_listen_prefix(self):
        accepted, _ = robot_ctrl.check_wake_word("Listen Robert, go there")
        assert accepted is True


class TestParseRobotResponse:
    def test_clean_json(self):
        text = '{"say": "hello", "gesture": "nod", "do": null, "continue": false}'
        result = robot_ctrl._parse_robot_response(text)
        assert result["say"] == "hello"
        assert result["gesture"] == "nod"
        assert result["continue"] is False

    def test_json_in_fences(self):
        text = 'Here is my response:\n```json\n{"say": "hi", "continue": true}\n```'
        result = robot_ctrl._parse_robot_response(text)
        assert result["say"] == "hi"
        assert result["continue"] is True

    def test_json_in_prose(self):
        text = 'I think the best action is {"say": "looking around", "do": {"action": "look"}, "continue": true} and that should work.'
        result = robot_ctrl._parse_robot_response(text)
        assert result["say"] == "looking around"
        assert result["do"]["action"] == "look"

    def test_fallback_to_say(self):
        text = "I don't understand what you mean."
        result = robot_ctrl._parse_robot_response(text)
        assert result["say"] == text
        assert result["continue"] is False


class TestVoiceTurn:
    @pytest.mark.asyncio
    async def test_stop_cancels_project(self):
        robot_ctrl._PROJECT = {
            "project_id": "test",
            "started_at": time.time(),
            "status": "active",
            "description": "test project",
            "messages": [],
            "step": 0,
            "max_steps": 20,
            "timeout_sec": 120,
            "loaded_knowledge": [],
            "history": [],
        }
        with patch.object(robot_ctrl, "cancel_project", new_callable=AsyncMock,
                          return_value={"ok": True, "message": "Cancelled."}):
            result = await robot_ctrl.handle_voice_turn("stop")
        assert "Cancelled" in result["say"] or "Stopping" in result["say"]

    @pytest.mark.asyncio
    async def test_idle_new_interaction(self):
        with patch("llm_backend.generate", return_value=(
            '{"say": "Hello there!", "gesture": "wave", "do": null, "continue": false}',
            100.0, "test-model"
        )):
            result = await robot_ctrl.handle_voice_turn("hello")
        assert result["say"] == "Hello there!"
        assert result["gesture"] == "wave"

    @pytest.mark.asyncio
    async def test_idle_starts_project(self):
        with patch("llm_backend.generate", return_value=(
            '{"say": "Looking for keys.", "do": {"action": "look"}, "continue": true}',
            100.0, "test-model"
        )), patch("endpoints.robot.dispatch_command", new_callable=AsyncMock,
                  return_value={"ok": True}):
            result = await robot_ctrl.handle_voice_turn("find my keys")
        assert result["say"] == "Looking for keys."
        assert robot_ctrl._PROJECT is not None
        assert robot_ctrl._PROJECT["status"] == "active"
        # Cleanup
        if robot_ctrl._TASK and not robot_ctrl._TASK.done():
            robot_ctrl._TASK.cancel()
            try:
                await robot_ctrl._TASK
            except asyncio.CancelledError:
                pass


class TestAgentTakeover:
    @pytest.mark.asyncio
    async def test_takeover_forwards_to_agent(self):
        robot_ctrl._TAKEOVER_AGENT = "agent-1"
        # Set up a future that resolves immediately
        request_id_holder = []

        async def mock_wait(rid, **kw):
            request_id_holder.append(rid)
            return {"reply": "I'll handle that.", "commands": []}

        with patch.object(robot_ctrl, "_wait_for_agent_reply", side_effect=mock_wait), \
             patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl.handle_voice_turn("do something")

        assert result["say"] == "I'll handle that."

    @pytest.mark.asyncio
    async def test_takeover_timeout_fallback(self):
        robot_ctrl._TAKEOVER_AGENT = "agent-1"

        async def mock_wait(rid, **kw):
            return None  # timeout

        with patch.object(robot_ctrl, "_wait_for_agent_reply", side_effect=mock_wait), \
             patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl.handle_voice_turn("do something")

        assert "didn't respond" in result["say"]

    def test_set_and_get_takeover(self):
        assert robot_ctrl.get_takeover_agent() is None
        robot_ctrl.set_takeover_agent("agent-42")
        assert robot_ctrl.get_takeover_agent() == "agent-42"
        robot_ctrl.set_takeover_agent(None)
        assert robot_ctrl.get_takeover_agent() is None

    def test_resolve_agent_reply(self):
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        robot_ctrl._TURN_FUTURES["req-1"] = fut
        robot_ctrl.resolve_agent_reply("req-1", "done", [{"type": "nod"}])
        assert fut.done()
        result = fut.result()
        assert result["reply"] == "done"
        assert result["commands"] == [{"type": "nod"}]
        loop.close()


class TestProjectLifecycle:
    @pytest.mark.asyncio
    async def test_cancel_no_project(self):
        result = await robot_ctrl.cancel_project()
        assert result["ok"] is True
        assert "No project" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_active_project(self):
        robot_ctrl._PROJECT = {
            "project_id": "p1",
            "started_at": time.time(),
            "status": "active",
            "description": "test",
            "messages": [],
            "step": 3,
            "max_steps": 20,
            "timeout_sec": 120,
            "loaded_knowledge": [],
            "history": [],
        }
        with patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl.cancel_project()
        assert result["ok"] is True
        assert robot_ctrl._PROJECT is None

    def test_current_project_idle(self):
        assert robot_ctrl.current_project() is None

    def test_current_project_active(self):
        robot_ctrl._PROJECT = {
            "project_id": "p1",
            "started_at": 123.0,
            "status": "active",
            "description": "test",
            "messages": [{"role": "system", "content": "..."}],
            "step": 2,
            "max_steps": 20,
            "timeout_sec": 120,
            "loaded_knowledge": ["household_objects/common"],
            "history": [{"step": 1}, {"step": 2}],
        }
        snap = robot_ctrl.current_project()
        assert snap is not None
        assert snap["project_id"] == "p1"
        assert snap["step"] == 2
        assert snap["history_count"] == 2
        assert "messages" not in snap  # messages excluded from snapshot

    def test_is_active(self):
        assert robot_ctrl.is_active() is False
        robot_ctrl._PROJECT = {"status": "active"}
        assert robot_ctrl.is_active() is True
        robot_ctrl._PROJECT = {"status": "completed"}
        assert robot_ctrl.is_active() is False
