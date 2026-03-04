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
        # Project lingers for 5s after cancel before clearing
        assert robot_ctrl._PROJECT is not None
        assert robot_ctrl._PROJECT["status"] == "cancelled"

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


class TestNormalizePlanSteps:
    def test_flat_strings(self):
        steps = robot_ctrl._normalize_plan_steps(["Find glass", "Pick it up", "Bring to user"])
        assert len(steps) == 3
        assert steps[0]["index"] == 0
        assert steps[0]["description"] == "Find glass"
        assert steps[0]["status"] == "pending"
        assert steps[0]["depends_on"] == []
        assert steps[0]["attempts"] == 0
        assert steps[2]["index"] == 2

    def test_structured_dicts(self):
        raw = [
            {"description": "Find glass", "success_criteria": "Glass detected"},
            {"description": "Pick up", "success_criteria": "Grasped", "depends_on": [0]},
        ]
        steps = robot_ctrl._normalize_plan_steps(raw)
        assert len(steps) == 2
        assert steps[0]["success_criteria"] == "Glass detected"
        assert steps[1]["depends_on"] == [0]

    def test_mixed(self):
        raw = ["Simple step", {"description": "Complex step", "depends_on": [0]}]
        steps = robot_ctrl._normalize_plan_steps(raw)
        assert len(steps) == 2
        assert steps[0]["success_criteria"] == ""  # string step has no criteria
        assert steps[1]["depends_on"] == [0]

    def test_empty(self):
        assert robot_ctrl._normalize_plan_steps([]) == []


class TestDependencies:
    def test_deps_met_no_deps(self):
        plan = [{"index": 0, "status": "pending", "depends_on": []}]
        assert robot_ctrl._deps_met(plan[0], plan) is True

    def test_deps_met_completed(self):
        plan = [
            {"index": 0, "status": "completed", "depends_on": []},
            {"index": 1, "status": "pending", "depends_on": [0]},
        ]
        assert robot_ctrl._deps_met(plan[1], plan) is True

    def test_deps_not_met(self):
        plan = [
            {"index": 0, "status": "pending", "depends_on": []},
            {"index": 1, "status": "pending", "depends_on": [0]},
        ]
        assert robot_ctrl._deps_met(plan[1], plan) is False

    def test_next_pending_step(self):
        plan = [
            {"index": 0, "status": "completed", "depends_on": []},
            {"index": 1, "status": "pending", "depends_on": [0]},
            {"index": 2, "status": "pending", "depends_on": [1]},
        ]
        nxt = robot_ctrl._next_pending_step(plan)
        assert nxt["index"] == 1  # step 2 blocked by step 1

    def test_next_pending_skips_blocked(self):
        plan = [
            {"index": 0, "status": "active", "depends_on": []},
            {"index": 1, "status": "pending", "depends_on": [0]},  # blocked
            {"index": 2, "status": "pending", "depends_on": []},   # not blocked
        ]
        nxt = robot_ctrl._next_pending_step(plan)
        assert nxt["index"] == 2

    def test_no_pending(self):
        plan = [
            {"index": 0, "status": "completed", "depends_on": []},
            {"index": 1, "status": "completed", "depends_on": [0]},
        ]
        assert robot_ctrl._next_pending_step(plan) is None


class TestBuildPlanStatus:
    def test_no_project(self):
        assert robot_ctrl._build_plan_status() == ""

    def test_no_plan(self):
        robot_ctrl._PROJECT = {"plan": []}
        assert robot_ctrl._build_plan_status() == ""

    def test_with_plan(self):
        robot_ctrl._PROJECT = {"plan": [
            {"index": 0, "description": "Find glass", "status": "completed",
             "depends_on": [], "attempts": 0, "result": None, "verification": {"verdict": "yes"}, "max_attempts": 3},
            {"index": 1, "description": "Pick up glass", "status": "active",
             "depends_on": [0], "attempts": 0, "result": None, "verification": None, "max_attempts": 3},
            {"index": 2, "description": "Bring to user", "status": "pending",
             "depends_on": [1], "attempts": 0, "result": None, "verification": None, "max_attempts": 3},
        ]}
        status = robot_ctrl._build_plan_status()
        assert "✓ Find glass" in status
        assert "→ Pick up glass" in status
        assert "○ Bring to user" in status
        assert "1/3 steps completed" in status
        assert "Current step: 2" in status


class TestStepHandlers:
    def _make_project_with_plan(self):
        """Create a project with a 3-step plan."""
        plan = robot_ctrl._normalize_plan_steps([
            {"description": "Find glass", "success_criteria": "Glass located"},
            {"description": "Pick up glass", "depends_on": [0]},
            {"description": "Bring to user", "depends_on": [1]},
        ])
        plan[0]["status"] = "active"
        plan[0]["started_at"] = time.time()
        robot_ctrl._PROJECT = {
            "project_id": "test-plan",
            "started_at": time.time(),
            "status": "active",
            "description": "Get me a glass of water",
            "messages": [],
            "step": 0,
            "max_steps": 20,
            "timeout_sec": 300,
            "loaded_knowledge": [],
            "history": [],
            "plan": plan,
            "plan_step": 0,
        }

    @pytest.mark.asyncio
    async def test_step_complete_advances(self):
        self._make_project_with_plan()
        with patch.object(robot_ctrl, "_verify_step", new_callable=AsyncMock,
                          return_value={"verdict": "yes", "vlm_reply": "Yes"}), \
             patch("event_bus.bus.publish", new_callable=AsyncMock), \
             patch.object(robot_ctrl, "_SPEAK_CB", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_complete({"step": 0})
        assert result["ok"] is True
        assert robot_ctrl._PROJECT["plan"][0]["status"] == "completed"
        assert robot_ctrl._PROJECT["plan"][1]["status"] == "active"
        assert result["next_step"]["index"] == 1

    @pytest.mark.asyncio
    async def test_step_complete_vlm_denies(self):
        self._make_project_with_plan()
        with patch.object(robot_ctrl, "_verify_step", new_callable=AsyncMock,
                          return_value={"verdict": "no", "vlm_reply": "Not done"}), \
             patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_complete({"step": 0})
        assert result["ok"] is False
        assert result["verified"] is False
        assert robot_ctrl._PROJECT["plan"][0]["status"] == "active"  # stays active

    @pytest.mark.asyncio
    async def test_step_complete_all_done(self):
        self._make_project_with_plan()
        plan = robot_ctrl._PROJECT["plan"]
        plan[0]["status"] = "completed"
        plan[1]["status"] = "completed"
        plan[2]["status"] = "active"
        with patch.object(robot_ctrl, "_verify_step", new_callable=AsyncMock,
                          return_value={"verdict": "yes"}), \
             patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_complete({"step": 2})
        assert result["ok"] is True
        assert result["all_complete"] is True
        assert result["progress"] == "3/3"

    @pytest.mark.asyncio
    async def test_step_failed_can_retry(self):
        self._make_project_with_plan()
        with patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_failed({"step": 0, "reason": "Not visible"})
        assert result["status"] == "can_retry"
        assert robot_ctrl._PROJECT["plan"][0]["attempts"] == 1

    @pytest.mark.asyncio
    async def test_step_failed_max_attempts(self):
        self._make_project_with_plan()
        robot_ctrl._PROJECT["plan"][0]["attempts"] = 2  # already tried twice
        with patch("event_bus.bus.publish", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_failed({"step": 0, "reason": "Still not visible"})
        assert result["status"] == "failed"
        assert robot_ctrl._PROJECT["plan"][0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_step_retry(self):
        self._make_project_with_plan()
        robot_ctrl._PROJECT["plan"][0]["status"] = "failed"
        robot_ctrl._PROJECT["plan"][0]["attempts"] = 2
        with patch.object(robot_ctrl, "_SPEAK_CB", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_retry({"step": 0})
        assert result["ok"] is True
        assert robot_ctrl._PROJECT["plan"][0]["status"] == "active"

    @pytest.mark.asyncio
    async def test_step_skip_cascades(self):
        self._make_project_with_plan()
        with patch("event_bus.bus.publish", new_callable=AsyncMock), \
             patch.object(robot_ctrl, "_SPEAK_CB", None):
            result = await robot_ctrl._handle_step_skip({"step": 0, "reason": "Impossible"})
        assert result["ok"] is True
        assert robot_ctrl._PROJECT["plan"][0]["status"] == "skipped"
        # Step 1 depends on 0, step 2 depends on 1 → transitive cascade
        assert robot_ctrl._PROJECT["plan"][1]["status"] == "skipped"
        assert robot_ctrl._PROJECT["plan"][2]["status"] == "skipped"
        assert 1 in result["cascade_skipped"]
        assert 2 in result["cascade_skipped"]

    @pytest.mark.asyncio
    async def test_step_complete_no_project(self):
        result = await robot_ctrl._handle_step_complete({"step": 0})
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_step_invalid_index(self):
        self._make_project_with_plan()
        result = await robot_ctrl._handle_step_complete({"step": 99})
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_step_complete_respects_deps(self):
        """Completing step 0 should not activate step 2 (depends on 1)."""
        self._make_project_with_plan()
        with patch.object(robot_ctrl, "_verify_step", new_callable=AsyncMock,
                          return_value={"verdict": "skipped"}), \
             patch("event_bus.bus.publish", new_callable=AsyncMock), \
             patch.object(robot_ctrl, "_SPEAK_CB", new_callable=AsyncMock):
            result = await robot_ctrl._handle_step_complete({"step": 0})
        assert result["ok"] is True
        assert robot_ctrl._PROJECT["plan"][1]["status"] == "active"  # step 1 activated
        assert robot_ctrl._PROJECT["plan"][2]["status"] == "pending"  # step 2 still blocked


class TestCurrentProjectStructured:
    def test_plan_progress_included(self):
        plan = robot_ctrl._normalize_plan_steps([
            {"description": "A", "success_criteria": "done"},
            {"description": "B", "depends_on": [0]},
        ])
        plan[0]["status"] = "completed"
        robot_ctrl._PROJECT = {
            "project_id": "p1",
            "started_at": 100.0,
            "status": "active",
            "description": "test",
            "messages": [],
            "step": 1,
            "max_steps": 20,
            "timeout_sec": 300,
            "loaded_knowledge": [],
            "history": [],
            "plan": plan,
            "plan_step": 1,
        }
        snap = robot_ctrl.current_project()
        assert snap["plan_progress"]["total"] == 2
        assert snap["plan_progress"]["completed"] == 1
        assert snap["plan_progress"]["pending"] == 1
        assert snap["plan"][0]["status"] == "completed"
        assert snap["plan"][0]["success_criteria"] == "done"
        assert snap["plan"][1]["depends_on"] == [0]
