"""Generic turn processor — ASR -> [hook] -> LLM -> TTS pipeline."""

from __future__ import annotations

import asyncio
import base64
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import agent_interface
import config
import instruction_store
import llm_backend
import session_store
import voice_pipeline
from event_bus import bus

_ROOT = Path(__file__).resolve().parent.parent
_TMP_DIR = _ROOT / "tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TurnResult:
    """Result of a turn through the pipeline."""

    ok: bool
    session_id: str
    transcript: str = ""
    reply: str = ""
    audio_base64: str = ""
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


async def process_turn(
    sid: str,
    audio_data: bytes | None = None,
    audio_filename: str = "turn.wav",
    transcript_hint: str = "",
    skip_asr: bool = False,
    forced_reply: str = "",
    pre_llm_hook: Callable[[str, str], Awaitable[dict | None]] | None = None,
) -> TurnResult:
    """Run a single conversational turn through the pipeline.

    Args:
        sid: Session ID (must already exist in session_store).
        audio_data: Raw audio bytes (WAV/PCM). Required unless skip_asr+hint.
        audio_filename: Original filename for suffix detection.
        transcript_hint: Fallback/override transcript text.
        skip_asr: If True and hint provided, skip ASR entirely.
        forced_reply: If set, skip ASR+LLM, go straight to TTS with this text.
        pre_llm_hook: async fn(sid, transcript) -> dict|None.  If returns dict
            with "reply" key, LLM is skipped.  Extra keys forwarded to
            TurnResult.extra.

    Returns:
        TurnResult with ok=True on success.
    """
    turn_gen = session_store.get_generation(sid)

    # --- Pending TTS inject?  Return it immediately, skip ASR/LLM ---
    pending = session_store.drain_inject(sid)
    if pending:
        await bus.publish("turn.started", {"session_id": sid}, session_id=sid)
        await bus.publish("turn.reply", {"reply": pending["text"]}, session_id=sid)
        await bus.publish("turn.complete", {
            "metrics": {"asr_ms": 0, "llm_ms": 0, "tts_ms": 0, "total_ms": 0},
            "transcript": "", "reply": pending["text"], "model": "inject",
        }, session_id=sid)
        return TurnResult(
            ok=True,
            session_id=sid,
            transcript="",
            reply=pending["text"],
            audio_base64=pending["audio_base64"],
            metrics={"asr_ms": 0, "llm_ms": 0, "tts_ms": 0, "total_ms": 0, "model": "inject"},
        )

    start = time.perf_counter()

    await bus.publish("turn.started", {"session_id": sid}, session_id=sid)

    transcript = ""
    reply = ""
    asr_ms = 0.0
    llm_ms = 0.0
    tts_ms = 0.0
    llm_model = ""
    extra: dict[str, Any] = {}
    hook_replied = False

    # --- forced_reply: skip ASR + LLM ---
    forced = voice_pipeline.safe_text(forced_reply)
    if forced:
        transcript = ""
        reply = forced
    else:
        # --- ASR ---
        hint = voice_pipeline.safe_text(transcript_hint)

        if skip_asr and hint:
            transcript = hint
        elif audio_data:
            suffix = Path(audio_filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(dir=_TMP_DIR, suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(audio_data)
            try:
                transcript, asr_ms = await asyncio.to_thread(voice_pipeline.transcribe, tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            # Fallback to hint
            if not transcript and hint:
                transcript = hint
        else:
            transcript = hint or ""

        await bus.publish("turn.transcript", {"transcript": transcript}, session_id=sid)

        # --- Pre-LLM hook (endpoint-specific: passphrase auth, etc.) ---
        if pre_llm_hook and transcript:
            hook_result = await pre_llm_hook(sid, transcript)
            if hook_result is not None:
                reply = hook_result.pop("reply", "")
                extra.update(hook_result)
                hook_replied = True

        # --- LLM (if hook didn't provide a reply) ---
        if not reply:
            if not transcript:
                reply = "I could not hear that clearly. Please try again."
            else:
                # Check agent takeover
                agent_ws = agent_interface.get_takeover_agent(sid)
                if agent_ws is not None:
                    takeover_timeout = config.get("agent.takeover_timeout", 60)
                    reply_text = await agent_interface.send_turn_request(
                        agent_ws, sid, transcript, timeout=takeover_timeout,
                    )
                    if reply_text is not None:
                        reply = voice_pipeline.safe_text(reply_text)
                        llm_model = "agent"
                    else:
                        agent_ws = None

                if agent_ws is None and not reply:
                    async with session_store.get_lock(sid):
                        system_prompt = instruction_store.build_system_prompt(sid)
                        knowledge = instruction_store.get_agent_knowledge(sid)
                        if knowledge:
                            system_prompt += f"\n\nIMPORTANT — use the following facts when answering:\n{knowledge}"

                        messages = [{"role": "system", "content": system_prompt}]
                        summary = session_store.get_summary(sid)
                        if summary:
                            messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
                        history = session_store.get_history(sid)
                        messages.extend(history)
                        messages.append({"role": "user", "content": transcript})

                        reply, llm_ms, llm_model = await asyncio.to_thread(llm_backend.generate, messages)

                # Commit to history + compact (only for LLM/agent-generated replies)
                session_store.append(sid, "user", transcript)
                session_store.append(sid, "assistant", reply)
                session_store.compact(sid)

    await bus.publish("turn.reply", {"reply": reply}, session_id=sid)

    # --- Stale check ---
    if session_store.get_generation(sid) != turn_gen:
        await bus.publish("turn.stale", {"session_id": sid, "reason": "generation_changed"}, session_id=sid)
        return TurnResult(ok=False, session_id=sid, extra={"stale": True, "detail": "session reset during turn"})

    # --- TTS ---
    audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, reply)

    # Stale check after TTS
    if session_store.get_generation(sid) != turn_gen:
        await bus.publish("turn.stale", {"session_id": sid, "reason": "generation_changed"}, session_id=sid)
        return TurnResult(ok=False, session_id=sid, extra={"stale": True, "detail": "session reset during turn"})

    total_ms = (time.perf_counter() - start) * 1000

    metrics = {
        "asr_ms": round(asr_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "tts_ms": round(tts_ms, 1),
        "total_ms": round(total_ms, 1),
        "llm_model": llm_model,
    }

    # Record turn for session persistence
    session_store.record_turn(sid, {
        "transcript": transcript,
        "reply": reply,
        "metrics": metrics,
        "forced_reply": bool(forced),
    })
    session_store.save_session(sid)

    await bus.publish("turn.complete", {
        "metrics": metrics,
        "transcript": transcript,
        "reply": reply,
        "session_id": sid,
    }, session_id=sid)

    return TurnResult(
        ok=True,
        session_id=sid,
        transcript=transcript,
        reply=reply,
        audio_base64=base64.b64encode(audio_bytes).decode("ascii"),
        metrics=metrics,
        extra=extra,
    )
