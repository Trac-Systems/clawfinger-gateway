"""Robot controller — wake word gating, voice turns, autonomous project loop, agent takeover."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any

import config
import llm_backend
import time_utils
from endpoints.robot import skill_loader
from endpoints.robot import dispatch_command as _dispatch_command
from event_bus import bus

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_PROJECT: dict[str, Any] | None = None
_TASK: asyncio.Task | None = None
_PAUSE_EVENT: asyncio.Event = asyncio.Event()
_LAST_INTERACTION: float = 0
_MESSAGES: list[dict[str, Any]] = []

# Agent takeover
_TAKEOVER_AGENT: str | None = None
_TURN_FUTURES: dict[str, asyncio.Future] = {}  # request_id -> Future for agent replies

# Speech callback — set by app.py to synthesize TTS + send over transport
_SPEAK_CB: Any | None = None  # async callable(text: str) -> None

# Ensure autonomous loop can run initially
_PAUSE_EVENT.set()


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------

def _is_conversational() -> bool:
    """True if the robot is in conversational mode (wake word not required)."""
    if _PROJECT and _PROJECT.get("status") == "active":
        return True
    timeout = config.get("robot.wake_word_timeout", 30)
    if _LAST_INTERACTION and (time.time() - _LAST_INTERACTION) < timeout:
        return True
    return False


def check_wake_word(transcript: str) -> tuple[bool, str]:
    """Check if transcript is directed at the robot.
    Returns (accepted, cleaned_transcript).
    """
    wake = config.get("robot.wake_word", "Robert").lower()
    lower = transcript.lower().strip()

    # Always-on: "{name} stop" cancels immediately regardless of mode
    if wake in lower and "stop" in lower:
        return True, "stop"

    # Conversational mode: wake word not required when robot is engaged
    if _is_conversational():
        return True, transcript

    # Idle mode: activation phrase required
    phrase_templates = config.get("robot.wake_phrases",
        ["hey {name}", "hi {name}", "ok {name}", "listen {name}", "{name}"])
    patterns = [t.format(name=wake) for t in phrase_templates]

    for p in patterns:
        if lower.startswith(p):
            cleaned = transcript[len(p):].lstrip(" ,!.?:")
            return True, cleaned or transcript

    return False, transcript


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

def _parse_robot_response(text: str) -> dict[str, Any]:
    """Parse LLM output into structured robot response.
    Handles clean JSON, JSON in markdown fences, JSON in prose, fallback to say.
    """
    text = text.strip()

    # Try clean JSON
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try JSON in markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try JSON embedded in prose
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict) and any(k in parsed for k in ("say", "do", "gesture", "continue")):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: treat entire text as speech
    return {"say": text, "gesture": None, "do": None, "continue": False}


# ---------------------------------------------------------------------------
# Safety validator
# ---------------------------------------------------------------------------

def _validate_command(command: dict) -> dict | None:
    """Validate command against safety limits.
    Returns None if safe, or {"ok": False, "error": "..."} if blocked.
    """
    cmd_type = command.get("type", "")
    params = command.get("params", {})

    # Check blocked commands
    blocked = config.get("robot.safety.blocked_commands", [])
    if cmd_type in blocked:
        return {"ok": False, "error": f"Command '{cmd_type}' is blocked by safety config"}

    # Speed limit
    if cmd_type == "walk":
        max_speed = config.get("robot.safety.max_speed", 0.3)
        if params.get("speed", 0) > max_speed:
            params["speed"] = max_speed

    # Force limit for grasping
    if cmd_type == "grasp":
        max_force = config.get("robot.safety.max_grasp_force", 0.8)
        if params.get("force", 0) > max_force:
            params["force"] = max_force

    # Reach envelope
    if cmd_type == "reach":
        max_reach = config.get("robot.safety.max_reach_m", 0.6)
        x = params.get("x", 0)
        y = params.get("y", 0)
        z = params.get("z", 0)
        distance = (x**2 + y**2 + z**2) ** 0.5
        if distance > max_reach:
            return {"ok": False, "error": f"Reach distance {distance:.2f}m exceeds limit {max_reach}m"}

    return None  # safe


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """Build robot system prompt with available skills and state."""
    wake_word = config.get("robot.wake_word", "Robert")
    slow_skills = skill_loader.list_slow_skills()
    all_skills = skill_loader.list_skills()

    skill_lines = []
    for s in slow_skills:
        topics = skill_loader.get_skill(s["name"])
        topic_names = topics.get("topics", []) if topics else []
        skill_lines.append(f"- {s['name']}: {s['description']}")

    fast_lines = []
    for s in all_skills:
        if s["execution_mode"] == "fast":
            status = s.get("status", "coming_soon")
            fast_lines.append(f"- {s['name']}: {s['description']} (fast-path, {status})")

    project_desc = "idle"
    plan_section = ""
    if _PROJECT and _PROJECT.get("status") == "active":
        project_desc = _PROJECT.get("description", "active project")
        plan_section = _build_plan_status()

    return f"""You are {wake_word}, a Unitree G1 humanoid robot.
You execute physical tasks. Respond with short, direct statements.
No emojis, no markdown formatting, no tables. Plain spoken language only.
The user speaks to you. You respond via speaker — keep it brief.

## Output Format
Always respond with a JSON object:
{{"say": "text to speak", "gesture": "nod|shake_head|wave|point|thumbs_up|null", "do": {{"action": "...", "params": {{...}}}} | null, "continue": true|false}}

## Available Primitives
- look: Camera scene description
- detect_object {{class}}: Find object, returns 3D position + image crop
- snapshot: Raw camera frame
- walk {{direction, speed, duration_s}}: Walk (forward/backward/left/right) for duration_s seconds (default 3)
- turn {{angle_deg, duration_s}}: Rotate in place for duration_s seconds (default 2)
- stand / sit / stop: Posture
- reach {{hand, x, y, z}}: Move hand to 3D position
- grasp {{hand, force}}: Close gripper
- release {{hand}}: Open gripper
- nod / shake_head / wave / point / thumbs_up: Gestures

## Available Knowledge
{chr(10).join(skill_lines) if skill_lines else "No knowledge packages loaded."}
Use {{"action": "load_skill", "params": {{"package": "<name>", "topic": "<topic>"}}}} to load details.

{('## Fast-Path Skills (coming soon)' + chr(10) + chr(10).join(fast_lines)) if fast_lines else ""}

## Multi-Step Projects
For complex tasks (3+ steps), create a structured plan:
{{"say": "Let me plan this...", "do": {{"action": "plan_project", "params": {{"steps": [
  {{"description": "Find the glass", "success_criteria": "Glass detected in 3D space"}},
  {{"description": "Pick up the glass", "success_criteria": "Glass grasped securely", "depends_on": [0]}},
  {{"description": "Bring to user", "depends_on": [1]}}
]}}}}, "continue": true}}

After completing actions for a step, mark it done:
{{"say": "Glass found.", "do": {{"action": "step_complete", "params": {{"step": 0}}}}, "continue": true}}

If a step fails:
{{"say": "Can't find the glass.", "do": {{"action": "step_failed", "params": {{"step": 0, "reason": "Not visible"}}}}, "continue": true}}

## Rules
1. For tasks you have knowledge about, load the relevant knowledge first.
2. Always look before acting — use detect_object for precise positions.
3. When uncertain about an object, use confirm_visual or ask the user (ask_user action).
4. Keep speech concise — you're talking in real-time.
5. If the user says "stop", cancel the current task immediately.
6. Output done/abort when a project is finished or impossible.
7. For complex tasks with 3+ steps, always create a plan first.
8. Use step_complete / step_failed to track progress after each plan step.
9. Do not skip steps that have unmet dependencies.
10. Never use emojis, markdown formatting, or tables in speech output.
11. Report what you see and do as plain facts.

## Current State
{time_utils.time_context_line()}
{_project_elapsed_line()}Current project: {project_desc}
{plan_section}"""


# ---------------------------------------------------------------------------
# Plan step helpers
# ---------------------------------------------------------------------------

def _project_elapsed_line() -> str:
    """Return project elapsed line if a project is active, else empty."""
    if _PROJECT and _PROJECT.get("status") == "active" and _PROJECT.get("started_at"):
        elapsed = time.time() - _PROJECT["started_at"]
        return f"Project elapsed: {time_utils.duration(elapsed)}\n"
    return ""


_STEP_STATUS_ICONS = {
    "completed": "✓",
    "active": "→",
    "failed": "✗",
    "skipped": "–",
    "pending": "○",
}


def _normalize_plan_steps(raw_steps: list) -> list[dict]:
    """Convert plan steps to structured format.

    Accepts both flat strings and structured dicts.
    """
    steps = []
    for i, s in enumerate(raw_steps):
        if isinstance(s, str):
            step = {
                "index": i,
                "description": s,
                "status": "pending",
                "success_criteria": "",
                "depends_on": [],
                "attempts": 0,
                "max_attempts": 3,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "verification": None,
            }
        elif isinstance(s, dict):
            step = {
                "index": i,
                "description": s.get("description", f"Step {i + 1}"),
                "status": "pending",
                "success_criteria": s.get("success_criteria", ""),
                "depends_on": s.get("depends_on", []),
                "attempts": 0,
                "max_attempts": s.get("max_attempts", 3),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "verification": None,
            }
        else:
            continue
        steps.append(step)
    return steps


def _build_plan_status() -> str:
    """Build plan progress text for system prompt injection."""
    if not _PROJECT or not _PROJECT.get("plan"):
        return ""

    lines = ["## Plan Progress"]
    plan = _PROJECT["plan"]
    total = len(plan)
    for s in plan:
        idx = s["index"] + 1
        icon = _STEP_STATUS_ICONS.get(s["status"], "?")
        extra = ""
        if s["status"] == "failed":
            extra = f" (failed: {s.get('result', {}).get('reason', 'unknown')}, attempt {s['attempts']}/{s['max_attempts']})"
        elif s["status"] == "completed" and s.get("verification"):
            v = s["verification"]
            extra = f" (verified: {v.get('verdict', '?')})"
        # Add step duration
        dur = ""
        if s["status"] == "completed" and s.get("started_at") and s.get("completed_at"):
            dur = f" [{time_utils.duration(s['completed_at'] - s['started_at'])}]"
        elif s["status"] == "active" and s.get("started_at"):
            dur = f" [running {time_utils.duration(time.time() - s['started_at'])}]"
        lines.append(f"Step {idx}/{total}: {icon} {s['description']}{dur}{extra}")

    active = [s for s in plan if s["status"] == "active"]
    if active:
        lines.append(f"\nCurrent step: {active[0]['index'] + 1} — {active[0]['description']}")

    completed = sum(1 for s in plan if s["status"] == "completed")
    lines.append(f"Progress: {completed}/{total} steps completed")
    return "\n".join(lines)


def _deps_met(step: dict, plan: list[dict]) -> bool:
    """Check if all dependencies for a step are completed."""
    for dep_idx in step.get("depends_on", []):
        if dep_idx < len(plan) and plan[dep_idx]["status"] != "completed":
            return False
    return True


def _next_pending_step(plan: list[dict]) -> dict | None:
    """Find the next pending step whose dependencies are met."""
    for s in plan:
        if s["status"] == "pending" and _deps_met(s, plan):
            return s
    return None


async def _verify_step(step: dict) -> dict:
    """Run visual verification for a completed step.

    Takes a snapshot and asks VLM if the success criteria are met.
    Returns {"verdict": "yes"|"no"|"unsure"|"skipped", ...}
    """
    criteria = step.get("success_criteria", "")
    if not criteria:
        return {"verdict": "skipped", "reason": "No success criteria defined"}

    # Take snapshot via robot command
    snapshot_result = await _dispatch_command(
        config.get("robot.model", "unitree_g1"),
        {"type": "snapshot", "params": {}},
    )

    image_b64 = ""
    if isinstance(snapshot_result, dict):
        image_b64 = snapshot_result.get("image_b64", snapshot_result.get("frame_b64", ""))

    if not image_b64:
        return {"verdict": "skipped", "reason": "No snapshot available"}

    # Ask VLM
    messages = [
        {"role": "system", "content": "You verify whether a robot completed a task step. "
         "Answer ONLY: YES, NO, or UNSURE."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": f"Task step: \"{step['description']}\"\n"
             f"Success criteria: \"{criteria}\"\n"
             f"Based on what you see, was this step completed successfully?"},
        ]},
    ]

    try:
        reply, _, _ = await asyncio.to_thread(
            llm_backend.generate, messages, True, config.robot_llm())
        reply_lower = reply.strip().lower()
        if "yes" in reply_lower:
            verdict = "yes"
        elif "no" in reply_lower:
            verdict = "no"
        else:
            verdict = "unsure"
        return {"verdict": verdict, "vlm_reply": reply.strip()}
    except Exception as exc:
        return {"verdict": "skipped", "reason": f"VLM error: {exc}"}


# ---------------------------------------------------------------------------
# Agent takeover
# ---------------------------------------------------------------------------

def set_speak_callback(cb) -> None:
    """Register an async callback for robot speech (TTS + transport send)."""
    global _SPEAK_CB
    _SPEAK_CB = cb


def set_takeover_agent(agent_id: str | None) -> None:
    """Set or clear the takeover agent."""
    global _TAKEOVER_AGENT
    _TAKEOVER_AGENT = agent_id


def get_takeover_agent() -> str | None:
    """Return current takeover agent ID, or None."""
    return _TAKEOVER_AGENT


def resolve_agent_reply(request_id: str, reply: str, commands: list[dict] | None = None) -> None:
    """Resolve a pending agent reply future."""
    fut = _TURN_FUTURES.pop(request_id, None)
    if fut and not fut.done():
        fut.set_result({"reply": reply, "commands": commands or []})


async def _wait_for_agent_reply(request_id: str, timeout: float = 60.0) -> dict | None:
    """Wait for an agent to reply to a turn request."""
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    _TURN_FUTURES[request_id] = fut
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        _TURN_FUTURES.pop(request_id, None)
        return None


# ---------------------------------------------------------------------------
# Voice turn handler
# ---------------------------------------------------------------------------

async def handle_voice_turn(transcript: str) -> dict:
    """Process a user voice command in robot context.

    Returns: {"say": str|None, "gesture": str|None}
    """
    global _PROJECT, _TASK, _LAST_INTERACTION

    _LAST_INTERACTION = time.time()

    # Handle "stop" command
    if transcript.strip().lower() == "stop":
        result = await cancel_project()
        return {"say": result.get("message", "Stopping."), "gesture": None}

    # Agent takeover — forward to agent
    if _TAKEOVER_AGENT:
        request_id = str(uuid.uuid4())
        await bus.publish("robot_turn.request", {
            "transcript": transcript,
            "request_id": request_id,
        }, endpoint="robot")
        agent_reply = await _wait_for_agent_reply(
            request_id,
            timeout=config.get("agent.takeover_timeout", 60),
        )
        if agent_reply:
            # Execute any robot commands from agent
            for cmd in agent_reply.get("commands", []):
                safety_err = _validate_command(cmd)
                if not safety_err:
                    await _dispatch_command(config.get("robot.model", "unitree_g1"), cmd)
            return {"say": agent_reply.get("reply"), "gesture": None}
        # Agent timed out — fall through to local LLM for this turn
        return {"say": "The agent didn't respond. Let me handle this.", "gesture": None}

    # Project running — pause autonomous loop, inject user input
    if _PROJECT and _PROJECT.get("status") == "active":
        _PAUSE_EVENT.clear()  # pause autonomous loop

        _PROJECT["messages"].append({"role": "user", "content": transcript})
        await bus.publish("robot.project.voice_interrupt", {
            "project_id": _PROJECT["project_id"],
            "transcript": transcript,
        }, endpoint="robot")

        text, _, _ = llm_backend.generate(_PROJECT["messages"], raw=True, llm_override=config.robot_llm())
        parsed = _parse_robot_response(text)
        _PROJECT["messages"].append({"role": "assistant", "content": text})

        # Handle stop/abort
        do = parsed.get("do")
        if do and do.get("action") in ("abort", "done"):
            await _finish_project(do["action"], do.get("params", {}))
        elif parsed.get("continue", True):
            _PAUSE_EVENT.set()  # resume autonomous loop
        else:
            _PAUSE_EVENT.set()

        return {"say": parsed.get("say"), "gesture": parsed.get("gesture")}

    # Idle — new interaction
    messages = [{"role": "system", "content": _build_system_prompt()}]
    messages.append({"role": "user", "content": transcript})

    text, _, _ = llm_backend.generate(messages, raw=True, llm_override=config.robot_llm())
    parsed = _parse_robot_response(text)
    messages.append({"role": "assistant", "content": text})
    print(f"[robot] LLM initial: {text[:500]}", flush=True)
    print(f"[robot] Parsed: continue={parsed.get('continue')}, do={parsed.get('do')}", flush=True)

    # If continue=true, start a background project
    if parsed.get("continue", False):
        do = parsed.get("do")

        _PROJECT = {
            "project_id": str(uuid.uuid4()),
            "started_at": time.time(),
            "status": "active",
            "description": transcript,
            "messages": messages,
            "step": 0,
            "max_steps": 20,
            "timeout_sec": 300,
            "loaded_knowledge": [],
            "history": [],
            "plan": [],
            "plan_step": 0,
        }

        # Publish immediately so UI shows response before action executes
        await bus.publish("robot.project.started", {
            "project_id": _PROJECT["project_id"],
            "description": transcript,
            "say": parsed.get("say"),
        }, endpoint="robot")

        # Execute initial action if present
        if do:
            result = await _execute_action(do)
            if result is not None:
                _PROJECT["messages"].append({"role": "user", "content": f"Result: {json.dumps(result)}"})
                _PROJECT["step"] += 1
                _PROJECT["history"].append({
                    "step": _PROJECT["step"],
                    "action": do,
                    "result": result,
                })

        # Start autonomous loop
        _TASK = asyncio.create_task(_run_autonomous())

    return {"say": parsed.get("say"), "gesture": parsed.get("gesture")}


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

async def _execute_action(do: dict) -> dict | None:
    """Execute a robot action, return result or None."""
    action = do.get("action", "")
    params = do.get("params", {})

    if action == "load_skill":
        pkg = params.get("package", "")
        topic = params.get("topic", "")
        content = skill_loader.get_topic(pkg, topic)
        if content and _PROJECT:
            _PROJECT["messages"].append({"role": "system", "content": content})
            _PROJECT["loaded_knowledge"].append(f"{pkg}/{topic}")
            await bus.publish("robot.skill.loaded", {"package": pkg, "topic": topic}, endpoint="robot")
        return {"ok": bool(content), "loaded": f"{pkg}/{topic}"}

    if action == "plan_project":
        raw_steps = params.get("steps", [])
        if _PROJECT and raw_steps:
            structured = _normalize_plan_steps(raw_steps)
            _PROJECT["plan"] = structured
            _PROJECT["plan_step"] = 0
            _PROJECT["max_steps"] = max(_PROJECT["max_steps"], len(structured) * 4)
            # Activate the first step whose deps are met
            first = _next_pending_step(structured)
            if first:
                first["status"] = "active"
                first["started_at"] = time.time()
            step_descs = [s["description"] for s in structured]
            await bus.publish("robot.project.planned", {
                "project_id": _PROJECT["project_id"],
                "steps": step_descs,
                "step_count": len(structured),
            }, endpoint="robot")
            # Announce first step via voice
            if first and _SPEAK_CB:
                msg = f"Step 1 of {len(structured)}: {first['description']}"
                try:
                    await _SPEAK_CB(msg)
                except Exception:
                    pass
        return {"ok": True, "step_count": len(raw_steps),
                "steps": [s["description"] for s in _PROJECT.get("plan", [])]}

    if action == "step_complete":
        return await _handle_step_complete(params)

    if action == "step_failed":
        return await _handle_step_failed(params)

    if action == "step_retry":
        return await _handle_step_retry(params)

    if action == "step_skip":
        return await _handle_step_skip(params)

    if action in ("done", "abort"):
        return None  # handled by caller

    if action in ("ask_user",):
        return None  # handled by caller — pauses for voice

    if action == "confirm_visual":
        # Visual confirmation pipeline — send cropped image to VLM
        return await _confirm_visual(params)

    if action == "delegate":
        await bus.publish("robot.project.delegate", params, endpoint="robot")
        return {"ok": True, "delegated": True}

    # Fast-path skill check
    skill = skill_loader.get_skill(action)
    if skill and skill.get("execution_mode") == "fast":
        return {"ok": False, "error": "This is a fast-path skill (coming soon). Use available primitives instead."}

    # detect_object: run detection + VLM confirmation pipeline
    if action == "detect_object":
        return await _detect_and_confirm(params)

    # Regular robot command — validate safety, then dispatch
    command = {"type": action, "params": params}
    safety_err = _validate_command(command)
    if safety_err:
        return safety_err

    result = await _dispatch_command(config.get("robot.model", "unitree_g1"), command)

    # Timed movement: walk/turn are fire-and-forget velocity commands.
    # Wait for the specified duration then stop, so the LLM gets a result
    # only after the movement is complete.
    if action in ("walk", "turn") and result.get("ok"):
        duration = float(params.get("duration_s", 3 if action == "walk" else 2))
        duration = min(duration, 30)  # safety cap
        await asyncio.sleep(duration)
        await _dispatch_command(
            config.get("robot.model", "unitree_g1"),
            {"type": "stop", "params": {}},
        )
        result["duration_s"] = duration
        result["detail"] = result.get("detail", "") + f" (ran {duration:.1f}s, stopped)"

    return result


async def _handle_step_complete(params: dict) -> dict:
    """Mark a plan step as completed, run visual verification, advance to next."""
    if not _PROJECT or not _PROJECT.get("plan"):
        return {"ok": False, "error": "No active plan"}

    step_idx = params.get("step", -1)
    plan = _PROJECT["plan"]
    if step_idx < 0 or step_idx >= len(plan):
        return {"ok": False, "error": f"Invalid step index {step_idx}"}

    step = plan[step_idx]
    if step["status"] not in ("active", "pending"):
        return {"ok": False, "error": f"Step {step_idx} is {step['status']}, cannot complete"}

    # Run visual verification if success criteria exist
    verification = await _verify_step(step)
    step["verification"] = verification

    if verification["verdict"] == "no":
        # VLM says step NOT done — inform LLM
        step["attempts"] += 1
        return {
            "ok": False,
            "verified": False,
            "verdict": "no",
            "message": f"Visual check failed: {verification.get('vlm_reply', 'Step does not appear completed')}",
            "attempts": step["attempts"],
            "max_attempts": step["max_attempts"],
        }

    # Mark completed
    step["status"] = "completed"
    step["completed_at"] = time.time()
    _PROJECT["plan_step"] = step_idx + 1

    completed = sum(1 for s in plan if s["status"] == "completed")
    total = len(plan)

    await bus.publish("robot.project.step_complete", {
        "project_id": _PROJECT["project_id"],
        "step_index": step_idx,
        "description": step["description"],
        "verification": verification["verdict"],
        "progress": f"{completed}/{total}",
    }, endpoint="robot")

    # Activate next pending step
    nxt = _next_pending_step(plan)
    if nxt:
        nxt["status"] = "active"
        nxt["started_at"] = time.time()
        # Voice announcement
        if _SPEAK_CB:
            msg = f"Step {nxt['index'] + 1} of {total}: {nxt['description']}"
            try:
                await _SPEAK_CB(msg)
            except Exception:
                pass
        return {
            "ok": True,
            "verified": verification["verdict"] != "no",
            "progress": f"{completed}/{total}",
            "next_step": {"index": nxt["index"], "description": nxt["description"]},
        }

    # All steps done
    if completed == total:
        return {
            "ok": True,
            "verified": verification["verdict"] != "no",
            "progress": f"{completed}/{total}",
            "all_complete": True,
            "message": "All plan steps completed. Use done action to finish.",
        }

    return {
        "ok": True,
        "verified": verification["verdict"] != "no",
        "progress": f"{completed}/{total}",
        "message": "Step completed. Some steps have unmet dependencies.",
    }


async def _handle_step_failed(params: dict) -> dict:
    """Mark a plan step as failed."""
    if not _PROJECT or not _PROJECT.get("plan"):
        return {"ok": False, "error": "No active plan"}

    step_idx = params.get("step", -1)
    plan = _PROJECT["plan"]
    if step_idx < 0 or step_idx >= len(plan):
        return {"ok": False, "error": f"Invalid step index {step_idx}"}

    step = plan[step_idx]
    reason = params.get("reason", "Unknown failure")
    step["attempts"] += 1
    step["result"] = {"reason": reason}

    if step["attempts"] >= step["max_attempts"]:
        step["status"] = "failed"
        await bus.publish("robot.project.step_failed", {
            "project_id": _PROJECT["project_id"],
            "step_index": step_idx,
            "description": step["description"],
            "reason": reason,
            "attempts": step["attempts"],
        }, endpoint="robot")
        return {
            "ok": True,
            "status": "failed",
            "message": f"Step {step_idx} failed after {step['attempts']} attempts: {reason}. "
                       f"Use step_skip to skip or abort the project.",
            "attempts": step["attempts"],
            "max_attempts": step["max_attempts"],
        }

    # Can retry
    return {
        "ok": True,
        "status": "can_retry",
        "message": f"Step {step_idx} attempt {step['attempts']}/{step['max_attempts']} failed: {reason}. "
                   f"Try a different approach or use step_skip.",
        "attempts": step["attempts"],
        "max_attempts": step["max_attempts"],
    }


async def _handle_step_retry(params: dict) -> dict:
    """Retry a failed step."""
    if not _PROJECT or not _PROJECT.get("plan"):
        return {"ok": False, "error": "No active plan"}

    step_idx = params.get("step", -1)
    plan = _PROJECT["plan"]
    if step_idx < 0 or step_idx >= len(plan):
        return {"ok": False, "error": f"Invalid step index {step_idx}"}

    step = plan[step_idx]
    if step["status"] not in ("failed",) and step["attempts"] == 0:
        return {"ok": False, "error": f"Step {step_idx} has not failed, no retry needed"}

    step["status"] = "active"
    step["started_at"] = time.time()
    step["verification"] = None

    if _SPEAK_CB:
        msg = f"Retrying step {step_idx + 1}: {step['description']}"
        try:
            await _SPEAK_CB(msg)
        except Exception:
            pass

    return {
        "ok": True,
        "message": f"Retrying step {step_idx}: {step['description']}",
        "attempt": step["attempts"] + 1,
        "max_attempts": step["max_attempts"],
    }


async def _handle_step_skip(params: dict) -> dict:
    """Skip a step (mark as skipped)."""
    if not _PROJECT or not _PROJECT.get("plan"):
        return {"ok": False, "error": "No active plan"}

    step_idx = params.get("step", -1)
    plan = _PROJECT["plan"]
    if step_idx < 0 or step_idx >= len(plan):
        return {"ok": False, "error": f"Invalid step index {step_idx}"}

    step = plan[step_idx]
    reason = params.get("reason", "Skipped by user/LLM")
    step["status"] = "skipped"
    step["result"] = {"reason": reason}
    step["completed_at"] = time.time()

    # Cascade skip to all transitively dependent steps
    skipped_deps = []
    skipped_set = {step_idx}
    changed = True
    while changed:
        changed = False
        for s in plan:
            if s["status"] == "pending" and s["index"] not in skipped_set:
                if any(d in skipped_set for d in s.get("depends_on", [])):
                    s["status"] = "skipped"
                    s["result"] = {"reason": f"Dependency step {step_idx} was skipped"}
                    s["completed_at"] = time.time()
                    skipped_deps.append(s["index"])
                    skipped_set.add(s["index"])
                    changed = True

    await bus.publish("robot.project.step_skipped", {
        "project_id": _PROJECT["project_id"],
        "step_index": step_idx,
        "description": step["description"],
        "reason": reason,
        "cascade_skipped": skipped_deps,
    }, endpoint="robot")

    # Activate next pending step
    nxt = _next_pending_step(plan)
    if nxt:
        nxt["status"] = "active"
        nxt["started_at"] = time.time()
        total = len(plan)
        if _SPEAK_CB:
            msg = f"Skipping ahead. Step {nxt['index'] + 1} of {total}: {nxt['description']}"
            try:
                await _SPEAK_CB(msg)
            except Exception:
                pass

    completed = sum(1 for s in plan if s["status"] in ("completed", "skipped"))
    return {
        "ok": True,
        "skipped": step_idx,
        "cascade_skipped": skipped_deps,
        "progress": f"{completed}/{len(plan)}",
        "next_step": {"index": nxt["index"], "description": nxt["description"]} if nxt else None,
    }


async def _detect_and_confirm(params: dict) -> dict:
    """Detect object via YOLO-World on robot, then confirm via VLM.

    Pipeline:
    1. Send detect_object command to robot/sim (runs YOLO-World)
    2. Get back detections with cropped images
    3. For each detection, send cropped image to VLM for confirmation
    4. Return confirmed/denied results
    """
    target = params.get("class", params.get("prompt", ""))
    if not target:
        return {"ok": False, "error": "detect_object requires 'class' param"}

    # Step 1: Run detection on robot/sim
    command = {"type": "detect_object", "params": params}
    safety_err = _validate_command(command)
    if safety_err:
        return safety_err

    result = await _dispatch_command(config.get("robot.model", "unitree_g1"), command)
    if not result.get("ok"):
        return result

    detections = result.get("detections", [])
    if not detections:
        return {"ok": True, "found": False, "detections": [], "message": f"No '{target}' detected"}

    # Step 2: VLM confirmation for each detection with a cropped image
    confirmed = []
    for det in detections:
        cropped_b64 = det.get("cropped_b64", "")
        if not cropped_b64:
            # No cropped image — trust the detection
            confirmed.append({**det, "vlm_verdict": "no_image"})
            continue

        # Build multimodal message for VLM
        confirmation_messages = [
            {"role": "system", "content": "You are a visual confirmation assistant. "
             "The user will show you a cropped image of a detected object. "
             "Answer ONLY with: YES, NO, or UNSURE. Nothing else."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_b64}"}},
                {"type": "text", "text": f"Is this a '{target}'? Answer YES, NO, or UNSURE."},
            ]},
        ]

        try:
            import asyncio
            reply, _, _ = await asyncio.to_thread(
                llm_backend.generate, confirmation_messages, True,
                config.robot_llm())
            reply_lower = reply.strip().lower()

            if "yes" in reply_lower:
                verdict = "yes"
            elif "no" in reply_lower:
                verdict = "no"
            else:
                verdict = "unsure"

            det_result = {**det, "vlm_verdict": verdict, "vlm_reply": reply.strip()}
            del det_result["cropped_b64"]  # Don't keep large base64 in result
            confirmed.append(det_result)
        except Exception as exc:
            det_result = {**det, "vlm_verdict": "error", "vlm_error": str(exc)}
            del det_result["cropped_b64"]
            confirmed.append(det_result)

    yes_count = sum(1 for d in confirmed if d.get("vlm_verdict") == "yes")
    unsure_count = sum(1 for d in confirmed if d.get("vlm_verdict") == "unsure")

    return {
        "ok": True,
        "found": yes_count > 0,
        "confirmed": [d for d in confirmed if d["vlm_verdict"] == "yes"],
        "unsure": [d for d in confirmed if d["vlm_verdict"] == "unsure"],
        "denied": [d for d in confirmed if d["vlm_verdict"] == "no"],
        "message": f"Found {yes_count} confirmed, {unsure_count} unsure '{target}'",
    }


async def _confirm_visual(params: dict) -> dict:
    """Confirm a cropped image via VLM — called when LLM outputs confirm_visual action."""
    image_b64 = params.get("image_base64", "")
    question = params.get("question", "What is this object?")
    if not image_b64:
        return {"ok": False, "error": "confirm_visual requires 'image_base64' param"}

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": question},
        ]},
    ]

    try:
        import asyncio
        reply, _, _ = await asyncio.to_thread(
            llm_backend.generate, messages, True, config.robot_llm())
        return {"ok": True, "answer": reply.strip()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Autonomous loop
# ---------------------------------------------------------------------------

async def _run_autonomous() -> None:
    """Background autonomous project execution loop."""
    global _PROJECT

    if not _PROJECT:
        return

    project_id = _PROJECT["project_id"]
    timeout = _PROJECT.get("timeout_sec", 120)
    max_steps = _PROJECT.get("max_steps", 20)
    start_time = time.time()

    try:
        while _PROJECT and _PROJECT["status"] == "active" and _PROJECT["step"] < max_steps:
            # Wait if paused (during voice turns)
            await _PAUSE_EVENT.wait()

            if not _PROJECT or _PROJECT["status"] != "active":
                break

            # Check timeout
            if time.time() - start_time > timeout:
                await _finish_project("abort", {"reason": "Project timed out"})
                return

            # Refresh system prompt (updates time context / step durations)
            fresh_prompt = _build_system_prompt()
            for msg in _PROJECT["messages"]:
                if msg["role"] == "system":
                    msg["content"] = fresh_prompt
                    break

            # Call LLM
            text, _, _ = llm_backend.generate(_PROJECT["messages"], raw=True, llm_override=config.robot_llm())
            parsed = _parse_robot_response(text)
            _PROJECT["messages"].append({"role": "assistant", "content": text})
            print(f"[robot] Auto step {_PROJECT['step']+1}: {text[:300]}", flush=True)
            print(f"[robot] Parsed: continue={parsed.get('continue')}, do={parsed.get('do')}", flush=True)

            # Trim context to prevent overflow (keep system + last ~20 messages)
            if len(_PROJECT["messages"]) > 24:
                system_msgs = [m for m in _PROJECT["messages"] if m["role"] == "system"]
                other_msgs = [m for m in _PROJECT["messages"] if m["role"] != "system"]
                _PROJECT["messages"] = system_msgs + other_msgs[-20:]

            _PROJECT["step"] += 1

            await bus.publish("robot.project.step", {
                "project_id": project_id,
                "step": _PROJECT["step"],
                "action": parsed.get("do"),
                "say": parsed.get("say"),
            }, endpoint="robot")

            # Handle say — synthesize TTS + send to robot speaker
            if parsed.get("say"):
                if _SPEAK_CB:
                    try:
                        await _SPEAK_CB(parsed["say"])
                    except Exception:
                        pass  # TTS errors logged by callback
                await bus.publish("robot.project.speak", {
                    "text": parsed["say"],
                    "project_id": project_id,
                }, endpoint="robot")

            # Handle gesture
            if parsed.get("gesture"):
                gesture_cmd = {"type": parsed["gesture"], "params": {}}
                safety_err = _validate_command(gesture_cmd)
                if not safety_err:
                    await _dispatch_command(config.get("robot.model", "unitree_g1"), gesture_cmd)

            # Handle action
            do = parsed.get("do")
            if do:
                action = do.get("action", "")

                # Terminal actions
                if action in ("done", "abort"):
                    await _finish_project(action, do.get("params", {}))
                    return

                # Ask user — pause and wait for voice
                if action in ("ask_user", "confirm_visual"):
                    question = do.get("params", {}).get("question") or do.get("params", {}).get("description", "")
                    if question:
                        await bus.publish("robot.project.ask_user", {
                            "project_id": project_id,
                            "question": question,
                        }, endpoint="robot")
                    _PAUSE_EVENT.clear()  # pause for user reply
                    continue

                # Execute action
                result = await _execute_action(do)
                if result is not None:
                    _PROJECT["messages"].append({"role": "user", "content": f"Result: {json.dumps(result)}"})
                    _PROJECT["history"].append({
                        "step": _PROJECT["step"],
                        "action": do,
                        "result": result,
                    })

            # If continue is false and no action pending, project is done
            if not parsed.get("continue", True):
                if not do or do.get("action") in (None, "done"):
                    await _finish_project("done", {"success": True, "reason": "Task completed"})
                    return
                _PAUSE_EVENT.clear()
                continue

            # Small yield to prevent tight loop
            await asyncio.sleep(0.1)

        # Max steps reached
        if _PROJECT and _PROJECT["status"] == "active":
            await _finish_project("abort", {"reason": f"Max steps ({max_steps}) reached"})

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        if _PROJECT:
            await _finish_project("abort", {"reason": f"Error: {exc}"})


async def _finish_project(action: str, params: dict) -> None:
    """Finalize a project (done or abort)."""
    global _PROJECT, _TASK

    if not _PROJECT:
        return

    # Safety: stop all robot movement when project ends
    try:
        await _dispatch_command(config.get("robot.model", "unitree_g1"), {"type": "stop", "params": {}})
    except Exception:
        pass

    project_id = _PROJECT["project_id"]
    success = action == "done" and params.get("success", False)
    _PROJECT["status"] = "completed" if success else "failed"

    event_type = "robot.project.completed" if success else "robot.project.failed"
    await bus.publish(event_type, {
        "project_id": project_id,
        "success": success,
        "reason": params.get("reason", ""),
        "steps": _PROJECT["step"],
    }, endpoint="robot")

    _PROJECT = None
    _TASK = None
    _PAUSE_EVENT.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def cancel_project() -> dict:
    """Cancel the running project."""
    global _PROJECT, _TASK

    if not _PROJECT:
        return {"ok": True, "message": "No project running."}

    # Safety: stop all robot movement
    try:
        await _dispatch_command(config.get("robot.model", "unitree_g1"), {"type": "stop", "params": {}})
    except Exception:
        pass

    project_id = _PROJECT["project_id"]
    _PROJECT["status"] = "cancelled"

    if _TASK and not _TASK.done():
        _TASK.cancel()

    await bus.publish("robot.project.cancelled", {
        "project_id": project_id,
    }, endpoint="robot")

    msg = f"Cancelled project: {_PROJECT.get('description', '')}"
    _PROJECT = None
    _TASK = None
    _PAUSE_EVENT.set()

    return {"ok": True, "message": msg}


def current_project() -> dict | None:
    """Snapshot of current project state (without messages)."""
    if not _PROJECT:
        return None

    plan = _PROJECT.get("plan", [])
    # Return structured plan info for API consumers
    plan_summary = []
    for s in plan:
        plan_summary.append({
            "index": s["index"],
            "description": s["description"],
            "status": s["status"],
            "success_criteria": s.get("success_criteria", ""),
            "depends_on": s.get("depends_on", []),
            "attempts": s.get("attempts", 0),
            "verification": s.get("verification"),
        })

    completed = sum(1 for s in plan if s["status"] == "completed")
    failed = sum(1 for s in plan if s["status"] == "failed")
    skipped = sum(1 for s in plan if s["status"] == "skipped")

    return {
        "project_id": _PROJECT["project_id"],
        "started_at": _PROJECT["started_at"],
        "status": _PROJECT["status"],
        "description": _PROJECT["description"],
        "step": _PROJECT["step"],
        "max_steps": _PROJECT["max_steps"],
        "loaded_knowledge": _PROJECT["loaded_knowledge"],
        "history_count": len(_PROJECT["history"]),
        "plan": plan_summary,
        "plan_step": _PROJECT.get("plan_step", 0),
        "plan_progress": {
            "total": len(plan),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "pending": len(plan) - completed - failed - skipped,
        } if plan else None,
    }


def is_active() -> bool:
    """True if a project is running."""
    return _PROJECT is not None and _PROJECT.get("status") == "active"
