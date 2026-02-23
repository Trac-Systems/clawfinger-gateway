"""Local Voice Gateway — FastAPI application."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import tempfile
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import agent_interface
import config
import instruction_store
import llm_backend
import session_store
import voice_pipeline
from event_bus import bus

app = FastAPI(title="Local Voice Gateway", version="0.1.0")

_ROOT = Path(__file__).resolve().parent
_TMP_DIR = _ROOT / "tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_STATIC_DIR = _ROOT / "static"
_START_TIME = time.time()
_CALL_COUNT = 0
_ERROR_COUNT = 0


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_bearer(request: Request) -> None:
    token = config.get("bearer_token", "")
    if not token:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Phone API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> JSONResponse:
    _check_bearer(request)
    mlx_status = voice_pipeline.check_mlx_audio()
    llm_status = llm_backend.check_health()
    return JSONResponse({
        "ok": True,
        "mlx_audio": mlx_status,
        "llm": llm_status,
        "active_sessions": len(session_store.active_sessions()),
        "uptime_s": round(time.time() - _START_TIME),
    })


@app.post("/api/asr")
async def api_asr(
    request: Request,
    audio: UploadFile = File(...),
) -> JSONResponse:
    _check_bearer(request)
    suffix = Path(audio.filename or "turn.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(dir=_TMP_DIR, suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await audio.read())
    try:
        transcript, asr_ms = await asyncio.to_thread(voice_pipeline.transcribe, tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"ASR failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)
    return JSONResponse({"transcript": transcript, "asr_ms": round(asr_ms, 1)})


@app.post("/api/session/new")
async def session_new(request: Request) -> JSONResponse:
    _check_bearer(request)
    sid = session_store.get_or_create()
    return JSONResponse({"session_id": sid})


@app.post("/api/session/reset")
async def session_reset(request: Request, session_id: str = Form("")) -> JSONResponse:
    _check_bearer(request)
    sid = voice_pipeline.safe_text(session_id)
    if sid:
        sid = session_store.reset(sid)
    return JSONResponse({"ok": True, "session_id": sid})


@app.post("/api/turn")
async def api_turn(
    request: Request,
    audio: UploadFile = File(...),
    session_id: str = Form(""),
    reset_session: str = Form("false"),
    transcript_hint: str = Form(""),
    skip_asr: str = Form(""),
    forced_reply: str = Form(""),
    caller_number: str = Form(""),
    call_direction: str = Form(""),
) -> JSONResponse:
    global _CALL_COUNT, _ERROR_COUNT
    _check_bearer(request)

    sid = voice_pipeline.safe_text(session_id) or uuid.uuid4().hex
    sid = session_store.get_or_create(sid)

    if reset_session.lower() == "true":
        session_store.reset(sid)
        sid = session_store.get_or_create(sid)

    cfg = config.load()

    # --- Caller info ---
    caller_number_clean = voice_pipeline.safe_text(caller_number)
    call_direction_clean = voice_pipeline.safe_text(call_direction)
    if caller_number_clean:
        session_store.set_caller_info(sid, caller_number_clean, call_direction_clean)

    # --- Caller filtering ---
    normalized = re.sub(r"[\s\-\(\)]", "", caller_number_clean)
    blocklist = cfg.get("caller_blocklist", [])
    if normalized and normalized in blocklist:
        await bus.publish("turn.caller_rejected", {"number": normalized, "reason": "blocklisted"}, session_id=sid)
        reject_text = cfg.get("auth_reject_message", "I'm sorry, I can't help you right now. Goodbye.")
        audio_bytes, _ = await asyncio.to_thread(voice_pipeline.synthesize, reject_text)
        return JSONResponse({
            "ok": False, "rejected": True, "reason": "blocklisted",
            "session_id": sid, "reply": reject_text,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        })

    allowlist = cfg.get("caller_allowlist", [])
    if allowlist and normalized and normalized not in allowlist:
        await bus.publish("turn.caller_rejected", {"number": normalized, "reason": "not_allowlisted"}, session_id=sid)
        reject_text = cfg.get("auth_reject_message", "I'm sorry, I can't help you right now. Goodbye.")
        audio_bytes, _ = await asyncio.to_thread(voice_pipeline.synthesize, reject_text)
        return JSONResponse({
            "ok": False, "rejected": True, "reason": "not_allowlisted",
            "session_id": sid, "reply": reject_text,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        })

    if not normalized and not cfg.get("unknown_callers_allowed", True):
        await bus.publish("turn.caller_rejected", {"number": "", "reason": "unknown_caller"}, session_id=sid)
        reject_text = cfg.get("auth_reject_message", "I'm sorry, I can't help you right now. Goodbye.")
        audio_bytes, _ = await asyncio.to_thread(voice_pipeline.synthesize, reject_text)
        return JSONResponse({
            "ok": False, "rejected": True, "reason": "unknown_caller",
            "session_id": sid, "reply": reject_text,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        })

    start = time.perf_counter()
    _CALL_COUNT += 1

    await bus.publish("turn.started", {"session_id": sid}, session_id=sid)

    transcript = ""
    reply = ""
    asr_ms = 0.0
    llm_ms = 0.0
    tts_ms = 0.0
    llm_model = ""

    try:
        # --- forced_reply: skip ASR + LLM, go straight to TTS ---
        forced = voice_pipeline.safe_text(forced_reply)
        if forced:
            transcript = ""
            reply = forced
            asr_ms = 0.0
            llm_ms = 0.0
        else:
            # --- ASR ---
            skip = skip_asr.strip().lower() == "true"
            hint = voice_pipeline.safe_text(transcript_hint)

            if skip and hint:
                transcript = hint
                asr_ms = 0.0
            else:
                suffix = Path(audio.filename or "turn.wav").suffix or ".wav"
                with tempfile.NamedTemporaryFile(dir=_TMP_DIR, suffix=suffix, delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(await audio.read())
                try:
                    transcript, asr_ms = await asyncio.to_thread(voice_pipeline.transcribe, tmp_path)
                except Exception as exc:
                    _ERROR_COUNT += 1
                    raise HTTPException(status_code=400, detail=f"ASR failed: {exc}") from exc
                finally:
                    tmp_path.unlink(missing_ok=True)

                # Fallback to hint
                if not transcript and hint:
                    transcript = hint

            await bus.publish("turn.transcript", {"transcript": transcript}, session_id=sid)

            # --- Passphrase auth gate ---
            passphrase = cfg.get("auth_passphrase", "")
            if passphrase and transcript and not session_store.is_authenticated(sid):
                # Fuzzy match: case-insensitive, strip punctuation, substring
                clean_phrase = re.sub(r"[^\w\s]", "", passphrase.lower()).strip()
                clean_input = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
                if clean_phrase in clean_input:
                    session_store.mark_authenticated(sid)
                    reply = "Authentication successful. How can I help you?"
                    llm_ms = 0.0
                    await bus.publish("turn.authenticated", {"session_id": sid}, session_id=sid)
                else:
                    attempts = session_store.record_auth_attempt(sid)
                    max_attempts = cfg.get("auth_max_attempts", 3)
                    await bus.publish("turn.auth_failed", {"session_id": sid, "attempt": attempts}, session_id=sid)
                    if max_attempts > 0 and attempts >= max_attempts:
                        reply = cfg.get("auth_reject_message", "I'm sorry, I can't help you right now. Goodbye.")
                        llm_ms = 0.0
                        # Skip LLM, go to TTS, include hangup
                        await bus.publish("turn.reply", {"reply": reply}, session_id=sid)
                        audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, reply)
                        total_ms = (time.perf_counter() - start) * 1000
                        metrics = {"asr_ms": round(asr_ms, 1), "llm_ms": 0.0, "tts_ms": round(tts_ms, 1),
                                   "total_ms": round(total_ms, 1), "llm_model": ""}
                        session_store.record_turn(sid, {"transcript": transcript, "reply": reply,
                                                        "metrics": metrics, "forced_reply": False})
                        session_store.save_session(sid)
                        await bus.publish("turn.complete", {"metrics": metrics, "transcript": transcript,
                                                            "reply": reply, "session_id": sid}, session_id=sid)
                        return JSONResponse({
                            "ok": True, "session_id": sid, "transcript": transcript, "reply": reply,
                            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                            "metrics": metrics, "hangup": True,
                        })
                    else:
                        reply = "That's not correct. Please try again."
                        llm_ms = 0.0

                # Auth handled — skip LLM, go to TTS
                await bus.publish("turn.reply", {"reply": reply}, session_id=sid)
                audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, reply)
                total_ms = (time.perf_counter() - start) * 1000
                metrics = {"asr_ms": round(asr_ms, 1), "llm_ms": 0.0, "tts_ms": round(tts_ms, 1),
                           "total_ms": round(total_ms, 1), "llm_model": ""}
                session_store.record_turn(sid, {"transcript": transcript, "reply": reply,
                                                "metrics": metrics, "forced_reply": False})
                session_store.save_session(sid)
                await bus.publish("turn.complete", {"metrics": metrics, "transcript": transcript,
                                                    "reply": reply, "session_id": sid}, session_id=sid)
                return JSONResponse({
                    "ok": True, "session_id": sid, "transcript": transcript, "reply": reply,
                    "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                    "metrics": metrics,
                })

            # --- LLM ---
            if not transcript:
                transcript = ""
                reply = "I could not hear that clearly. Please try again."
                llm_ms = 0.0
            else:
                # Check if agent has taken over LLM for this session
                agent_ws = agent_interface.get_takeover_agent(sid)
                if agent_ws is not None:
                    # Route to agent via single-reader pattern (request_id correlation)
                    reply_text = await agent_interface.send_turn_request(
                        agent_ws, sid, transcript,
                    )
                    if reply_text is not None:
                        reply = voice_pipeline.safe_text(reply_text)
                        llm_ms = 0.0
                        llm_model = "agent"
                    else:
                        # Agent failed / timed out — fall back to local LLM
                        agent_ws = None

                if agent_ws is None and not reply:
                    async with session_store.get_lock(sid):
                        # 1. Master instructions (never compacted)
                        system_prompt = instruction_store.build_system_prompt(sid)
                        messages = [{"role": "system", "content": system_prompt}]

                        # 2. Compacted summary of older conversation (if any)
                        summary = session_store.get_summary(sid)
                        if summary:
                            messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})

                        # 3. Recent history (verbatim)
                        history = session_store.get_history(sid)
                        messages.extend(history)

                        # 4. Agent injected knowledge (if any)
                        knowledge = instruction_store.get_agent_knowledge(sid)
                        if knowledge:
                            messages.append({"role": "system", "content": f"Context from agent:\n{knowledge}"})

                        # 5. Current user turn
                        messages.append({"role": "user", "content": transcript})

                        reply, llm_ms, llm_model = await asyncio.to_thread(llm_backend.generate, messages)

                # Commit to history
                session_store.append(sid, "user", transcript)
                session_store.append(sid, "assistant", reply)

        await bus.publish("turn.reply", {"reply": reply}, session_id=sid)

        # --- TTS ---
        audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, reply)
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

        return JSONResponse({
            "ok": True,
            "session_id": sid,
            "transcript": transcript,
            "reply": reply,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            "metrics": metrics,
        })

    except HTTPException:
        raise
    except Exception as exc:
        _ERROR_COUNT += 1
        await bus.publish("turn.error", {"error": str(exc)}, session_id=sid)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# UI support endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions() -> JSONResponse:
    return JSONResponse(session_store.list_sessions())


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    detail = session_store.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(detail)


@app.get("/api/status")
async def system_status() -> JSONResponse:
    cfg = config.load()
    return JSONResponse({
        "uptime_s": round(time.time() - _START_TIME),
        "total_calls": _CALL_COUNT,
        "error_count": _ERROR_COUNT,
        "active_sessions": len(session_store.active_sessions()),
        "ui_subscribers": bus.subscriber_count,
        "agents": agent_interface.list_agents(),
        "mlx_audio": voice_pipeline.check_mlx_audio(),
        "llm": llm_backend.check_health(),
        "config": {k: v for k, v in cfg.items() if "token" not in k and "key" not in k and "bearer" not in k},
    })


@app.post("/api/config")
async def update_config(request: Request) -> JSONResponse:
    """Hot-reload config from disk."""
    cfg = config.reload()
    await bus.publish("status.update", {"event": "config_reloaded"})
    safe = {k: v for k, v in cfg.items() if "token" not in k and "key" not in k and "bearer" not in k}
    return JSONResponse({"ok": True, "config": safe})


@app.post("/api/call/inject")
async def call_inject(request: Request) -> JSONResponse:
    """Inject a TTS message into the event stream (for UI or agent)."""
    body = await request.json()
    text = voice_pipeline.safe_text(str(body.get("text", "")))
    session_id = str(body.get("session_id", ""))
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, text)
    await bus.publish("agent.inject", {
        "text": text,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "tts_ms": round(tts_ms, 1),
    }, session_id=session_id)
    return JSONResponse({"ok": True, "tts_ms": round(tts_ms, 1)})


# ---------------------------------------------------------------------------
# Instruction endpoints
# ---------------------------------------------------------------------------

@app.get("/api/instructions")
async def get_instructions() -> JSONResponse:
    return JSONResponse(instruction_store.snapshot())


@app.post("/api/instructions")
async def set_base_instruction(request: Request) -> JSONResponse:
    body = await request.json()
    text = str(body.get("text", ""))
    instruction_store.set_base(text)
    await bus.publish("instructions.updated", {"scope": "global"})
    return JSONResponse({"ok": True, "base": instruction_store.get_base()})


@app.post("/api/instructions/{sid}")
async def set_session_instruction(sid: str, request: Request) -> JSONResponse:
    body = await request.json()
    text = str(body.get("text", ""))
    instruction_store.set_session(sid, text)
    await bus.publish("instructions.updated", {"scope": "session", "session_id": sid}, session_id=sid)
    return JSONResponse({"ok": True, "session_id": sid})


@app.post("/api/instructions/{sid}/turn")
async def set_turn_instruction(sid: str, request: Request) -> JSONResponse:
    body = await request.json()
    text = str(body.get("text", ""))
    instruction_store.set_turn(sid, text)
    return JSONResponse({"ok": True, "session_id": sid, "scope": "turn"})


@app.delete("/api/instructions/{sid}")
async def clear_session_instruction(sid: str) -> JSONResponse:
    instruction_store.clear_session(sid)
    await bus.publish("instructions.updated", {"scope": "session", "session_id": sid}, session_id=sid)
    return JSONResponse({"ok": True, "session_id": sid})


# ---------------------------------------------------------------------------
# Dial endpoint
# ---------------------------------------------------------------------------

async def _do_dial(number: str) -> dict:
    """Send dial command to phone via ADB broadcast."""
    if not number:
        return {"ok": False, "detail": "number required"}
    # Check ADB connection
    try:
        proc = await asyncio.create_subprocess_exec(
            "adb", "devices",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode().strip().split("\n")
        devices = [l for l in lines[1:] if l.strip() and "device" in l]
        if not devices:
            return {"ok": False, "detail": "No ADB device connected"}
    except Exception as exc:
        return {"ok": False, "detail": f"ADB check failed: {exc}"}
    # Send broadcast
    try:
        proc = await asyncio.create_subprocess_exec(
            "adb", "shell", "am", "broadcast",
            "-a", "com.tracsystems.phonebridge.CALL_COMMAND",
            "-n", "com.tracsystems.phonebridge/.CallCommandReceiver",
            "--es", "type", "dial",
            "--es", "number", number,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        output = stdout.decode().strip()
        if proc.returncode != 0:
            return {"ok": False, "detail": f"ADB broadcast failed: {stderr.decode().strip()}"}
        return {"ok": True, "detail": output}
    except Exception as exc:
        return {"ok": False, "detail": f"Dial failed: {exc}"}


@app.post("/api/call/dial")
async def call_dial(request: Request) -> JSONResponse:
    body = await request.json()
    number = str(body.get("number", "")).strip()
    result = await _do_dial(number)
    if result["ok"]:
        await bus.publish("call.dial", {"number": number})
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@app.websocket("/api/agent/ws")
async def agent_ws(ws: WebSocket) -> None:
    await ws.accept()
    await agent_interface.agent_connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # --- request_id correlation: catch takeover replies first ---
            request_id = msg.get("request_id")
            if request_id and "reply" in msg:
                reply_text = str(msg["reply"])
                agent_interface.resolve_turn_reply(str(request_id), reply_text)
                continue

            msg_type = str(msg.get("type", ""))

            if msg_type == "takeover":
                sid = str(msg.get("session_id", ""))
                ok = await agent_interface.takeover(ws, sid)
                await ws.send_json({"type": "takeover.ack", "ok": ok, "session_id": sid})

            elif msg_type == "release":
                sid = str(msg.get("session_id", ""))
                ok = await agent_interface.release(ws, sid)
                await ws.send_json({"type": "release.ack", "ok": ok, "session_id": sid})

            elif msg_type == "inject":
                text = voice_pipeline.safe_text(str(msg.get("text", "")))
                sid = str(msg.get("session_id", ""))
                if text:
                    audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, text)
                    await bus.publish("agent.inject", {
                        "text": text,
                        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                        "tts_ms": round(tts_ms, 1),
                    }, session_id=sid)

            elif msg_type == "set_instructions":
                text = str(msg.get("instructions", ""))
                sid = str(msg.get("session_id", ""))
                scope = str(msg.get("scope", "turn"))
                if scope == "global":
                    instruction_store.set_base(text)
                elif scope == "session" and sid:
                    instruction_store.set_session(sid, text)
                elif scope == "turn" and sid:
                    instruction_store.set_turn(sid, text)
                await ws.send_json({"type": "set_instructions.ack", "ok": True, "scope": scope})
                await bus.publish("instructions.updated", {"scope": scope, "session_id": sid})

            elif msg_type == "set_call_config":
                cfg = config.load()
                for key, value in msg.get("config", {}).items():
                    if key in AGENT_ALLOWED_CALL_KEYS:
                        cfg[key] = value
                config.save()
                await ws.send_json({"type": "set_call_config.ack", "ok": True})
                await bus.publish("config.call_updated", _call_config_response())

            elif msg_type == "dial":
                number = str(msg.get("number", ""))
                result = await _do_dial(number)
                await ws.send_json({"type": "dial.ack", **result})
                if result["ok"]:
                    await bus.publish("call.dial", {"number": number})

            elif msg_type == "get_call_state":
                sid = str(msg.get("session_id", ""))
                history = session_store.get_history(sid)
                meta = session_store.active_sessions().get(sid)
                await ws.send_json({
                    "type": "call_state",
                    "session_id": sid,
                    "history": history,
                    "turn_count": len(meta.get("turns", [])) if meta else 0,
                    "instructions": {
                        "base": instruction_store.get_base(),
                        "session": instruction_store.get_session(sid),
                        "pending_turn": instruction_store.get_turn(sid),
                    },
                    "agent_takeover": agent_interface.get_takeover_agent(sid) is not None,
                })

            elif msg_type == "inject_context":
                sid = str(msg.get("session_id", ""))
                context = voice_pipeline.safe_text(str(msg.get("context", "")))
                if sid and context:
                    async with session_store.get_lock(sid):
                        instruction_store.set_agent_knowledge(sid, context)
                    await ws.send_json({"type": "inject_context.ack", "ok": True})
                    await bus.publish("agent.context_injected", {"session_id": sid}, session_id=sid)
                else:
                    await ws.send_json({"type": "inject_context.ack", "ok": False})

            elif msg_type == "clear_context":
                sid = str(msg.get("session_id", ""))
                async with session_store.get_lock(sid):
                    instruction_store.clear_agent_knowledge(sid)
                await ws.send_json({"type": "clear_context.ack", "ok": True})
                await bus.publish("agent.context_cleared", {"session_id": sid}, session_id=sid)

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        await agent_interface.agent_disconnect(ws)


@app.post("/api/agent/inject")
async def agent_inject_rest(request: Request) -> JSONResponse:
    body = await request.json()
    text = voice_pipeline.safe_text(str(body.get("text", "")))
    session_id = str(body.get("session_id", ""))
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, text)
    await bus.publish("agent.inject", {
        "text": text,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "tts_ms": round(tts_ms, 1),
    }, session_id=session_id)
    return JSONResponse({"ok": True, "tts_ms": round(tts_ms, 1)})


@app.get("/api/agent/sessions")
async def agent_sessions() -> JSONResponse:
    return JSONResponse(list(session_store.active_sessions().keys()))


@app.post("/api/agent/takeover")
async def agent_takeover_rest(request: Request) -> JSONResponse:
    """REST takeover — only works with connected agent WebSocket. Stubbed for now."""
    return JSONResponse({"ok": False, "detail": "Use WebSocket /api/agent/ws for takeover"})


@app.post("/api/agent/release")
async def agent_release_rest(request: Request) -> JSONResponse:
    """REST release — only works with connected agent WebSocket. Stubbed for now."""
    return JSONResponse({"ok": False, "detail": "Use WebSocket /api/agent/ws for release"})


@app.get("/api/agent/context/{session_id}")
async def get_agent_context(session_id: str) -> JSONResponse:
    knowledge = instruction_store.get_agent_knowledge(session_id)
    return JSONResponse({
        "session_id": session_id,
        "knowledge": knowledge,
        "has_knowledge": bool(knowledge),
    })


@app.post("/api/agent/context/{session_id}")
async def set_agent_context(session_id: str, request: Request) -> JSONResponse:
    body = await request.json()
    context = voice_pipeline.safe_text(str(body.get("context", "")))
    if not context:
        raise HTTPException(status_code=400, detail="context is required")
    async with session_store.get_lock(session_id):
        instruction_store.set_agent_knowledge(session_id, context)
    await bus.publish("agent.context_injected", {"session_id": session_id}, session_id=session_id)
    return JSONResponse({"ok": True, "session_id": session_id})


@app.delete("/api/agent/context/{session_id}")
async def clear_agent_context(session_id: str) -> JSONResponse:
    async with session_store.get_lock(session_id):
        instruction_store.clear_agent_knowledge(session_id)
    await bus.publish("agent.context_cleared", {"session_id": session_id}, session_id=session_id)
    return JSONResponse({"ok": True, "session_id": session_id})


# ---------------------------------------------------------------------------
# TTS config endpoints
# ---------------------------------------------------------------------------

_KOKORO_VOICES = {
    "American Female": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    ],
    "American Male": [
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "British Female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "British Male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}

_TTS_ALIAS = {"voice": "tts_voice", "speed": "tts_speed"}


def _tts_config_response() -> dict:
    cfg = config.load()
    model = cfg.get("tts_model", "")
    is_kokoro = "kokoro" in model.lower()
    return {
        "voice": cfg.get("tts_voice", "af_heart"),
        "speed": cfg.get("tts_speed", 1.2),
        "model": model,
        "voices": _KOKORO_VOICES if is_kokoro else {},
    }


@app.get("/api/config/tts")
async def get_tts_config() -> JSONResponse:
    return JSONResponse(_tts_config_response())


@app.post("/api/config/tts")
async def update_tts_config(request: Request) -> JSONResponse:
    body = await request.json()
    cfg = config.load()
    for body_key, value in body.items():
        cfg_key = _TTS_ALIAS.get(body_key, body_key)
        if cfg_key in {"tts_voice", "tts_speed"}:
            cfg[cfg_key] = value
    config.save()
    resp = _tts_config_response()
    await bus.publish("config.tts_updated", resp)
    return JSONResponse({"ok": True, **resp})


@app.post("/api/tts/preview")
async def tts_preview(request: Request) -> JSONResponse:
    body = await request.json()
    text = voice_pipeline.safe_text(str(body.get("text", ""))) or "Hello, this is a voice preview."
    preview_voice = str(body.get("voice", "")) or config.get("tts_voice", "af_heart")
    preview_speed = float(body.get("speed", 0)) or config.get("tts_speed", 1.2)

    cfg = config.load()
    mlx_base = cfg["mlx_audio_base"].rstrip("/")
    payload = {
        "model": cfg["tts_model"],
        "input": voice_pipeline.trim_for_tts(text),
        "voice": preview_voice,
        "speed": preview_speed,
        "response_format": "wav",
    }

    import httpx as _httpx
    response = await asyncio.to_thread(
        lambda: _httpx.post(f"{mlx_base}/v1/audio/speech", json=payload, timeout=180)
    )
    response.raise_for_status()

    audio_b64 = base64.b64encode(response.content).decode("ascii")
    return JSONResponse({
        "ok": True,
        "audio_base64": audio_b64,
        "voice": preview_voice,
        "speed": preview_speed,
    })


# ---------------------------------------------------------------------------
# LLM config endpoints
# ---------------------------------------------------------------------------

_LLM_PARAM_KEYS = {
    "llm_max_tokens", "llm_temperature", "llm_top_p", "llm_top_k",
    "llm_repeat_penalty", "llm_stop",
    "llm_top_p_enabled", "llm_top_k_enabled", "llm_context_tokens",
    "max_history_turns",
}
_LLM_IDENTITY_KEYS = {"llm_model", "llm_base_url", "llm_api_key"}

# Short aliases accepted by POST body → config key
_LLM_ALIAS = {
    "model": "llm_model",
    "base_url": "llm_base_url",
    "api_key": "llm_api_key",
    "max_tokens": "llm_max_tokens",
    "temperature": "llm_temperature",
    "top_p": "llm_top_p",
    "top_k": "llm_top_k",
    "top_p_enabled": "llm_top_p_enabled",
    "top_k_enabled": "llm_top_k_enabled",
    "repeat_penalty": "llm_repeat_penalty",
    "stop": "llm_stop",
    "context_tokens": "llm_context_tokens",
    "max_history_turns": "max_history_turns",
}


def _llm_config_response() -> dict:
    cfg = config.load()
    return {
        "model": cfg.get("llm_model", ""),
        "base_url": cfg.get("llm_base_url", ""),
        "has_api_key": bool(cfg.get("llm_api_key", "")),
        "max_tokens": cfg.get("llm_max_tokens", 400),
        "context_tokens": cfg.get("llm_context_tokens", 0),
        "max_history_turns": cfg.get("max_history_turns", 8),
        "temperature": cfg.get("llm_temperature", 0.2),
        "top_p": cfg.get("llm_top_p", 1.0),
        "top_p_enabled": cfg.get("llm_top_p_enabled", True),
        "top_k": cfg.get("llm_top_k", 0),
        "top_k_enabled": cfg.get("llm_top_k_enabled", True),
        "repeat_penalty": cfg.get("llm_repeat_penalty", 1.0),
        "stop": cfg.get("llm_stop", []),
        "is_local": not cfg.get("llm_base_url"),
    }


@app.get("/api/config/llm")
async def get_llm_config() -> JSONResponse:
    return JSONResponse(_llm_config_response())


@app.post("/api/config/llm")
async def update_llm_config(request: Request) -> JSONResponse:
    body = await request.json()
    cfg = config.load()

    for body_key, value in body.items():
        cfg_key = _LLM_ALIAS.get(body_key, body_key)
        if cfg_key in _LLM_PARAM_KEYS | _LLM_IDENTITY_KEYS:
            cfg[cfg_key] = value

    config.save()
    await bus.publish("config.llm_updated", _llm_config_response())
    return JSONResponse({"ok": True, **_llm_config_response()})


# ---------------------------------------------------------------------------
# Call config endpoints
# ---------------------------------------------------------------------------

_CALL_CONFIG_KEYS = {
    "call_auto_answer", "call_auto_answer_delay_ms",
    "caller_allowlist", "caller_blocklist", "unknown_callers_allowed",
    "greeting_incoming", "greeting_outgoing", "greeting_owner",
    "max_duration_sec", "max_duration_message",
    "auth_passphrase", "auth_reject_message", "auth_max_attempts",
}

# Keys agents are NOT allowed to set via WebSocket
_CALL_SECURITY_KEYS = {
    "auth_passphrase", "auth_reject_message", "auth_max_attempts",
    "caller_allowlist", "caller_blocklist", "unknown_callers_allowed",
}

AGENT_ALLOWED_CALL_KEYS = (_CALL_CONFIG_KEYS - _CALL_SECURITY_KEYS) | {"tts_voice", "tts_speed"}


def _call_config_response() -> dict:
    cfg = config.load()
    owner = cfg.get("greeting_owner", "the owner")
    greeting_in = cfg.get("greeting_incoming", "")
    greeting_out = cfg.get("greeting_outgoing", "")
    return {
        "auto_answer": cfg.get("call_auto_answer", True),
        "auto_answer_delay_ms": cfg.get("call_auto_answer_delay_ms", 500),
        "caller_allowlist": cfg.get("caller_allowlist", []),
        "caller_blocklist": cfg.get("caller_blocklist", []),
        "unknown_callers_allowed": cfg.get("unknown_callers_allowed", True),
        "greeting_incoming": greeting_in.replace("{owner}", owner),
        "greeting_outgoing": greeting_out.replace("{owner}", owner),
        "greeting_incoming_template": greeting_in,
        "greeting_outgoing_template": greeting_out,
        "greeting_owner": owner,
        "max_duration_sec": cfg.get("max_duration_sec", 300),
        "max_duration_message": cfg.get("max_duration_message", ""),
        "auth_required": bool(cfg.get("auth_passphrase", "")),
        "auth_reject_message": cfg.get("auth_reject_message", ""),
        "auth_max_attempts": cfg.get("auth_max_attempts", 3),
    }


@app.get("/api/config/call")
async def get_call_config() -> JSONResponse:
    return JSONResponse(_call_config_response())


@app.post("/api/config/call")
async def update_call_config(request: Request) -> JSONResponse:
    body = await request.json()
    cfg = config.load()
    for key, value in body.items():
        if key in _CALL_CONFIG_KEYS:
            cfg[key] = value
    config.save()
    await bus.publish("config.call_updated", _call_config_response())
    return JSONResponse({"ok": True, **_call_config_response()})


# ---------------------------------------------------------------------------
# Agent call state endpoint
# ---------------------------------------------------------------------------

@app.get("/api/agent/call/{sid}")
async def agent_call_state(sid: str) -> JSONResponse:
    history = session_store.get_history(sid)
    meta = session_store.active_sessions().get(sid)
    instructions = {
        "base": instruction_store.get_base(),
        "session": instruction_store.get_session(sid),
        "pending_turn": instruction_store.get_turn(sid),
    }
    has_takeover = agent_interface.get_takeover_agent(sid) is not None
    return JSONResponse({
        "session_id": sid,
        "history": history,
        "turn_count": len(meta.get("turns", [])) if meta else 0,
        "instructions": instructions,
        "agent_takeover": has_takeover,
        "created_at": meta.get("created_at") if meta else None,
    })


# ---------------------------------------------------------------------------
# UI WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    await bus.subscribe(ws)
    try:
        while True:
            # Keep connection alive, handle pings
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await bus.unsubscribe(ws)


# ---------------------------------------------------------------------------
# Static files + UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<html><body><h2>Control Center UI not found</h2></body></html>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    cfg = config.load()
    print(f"[gateway] Starting on {cfg['host']}:{cfg['port']}")
    print(f"[gateway] mlx_audio: {cfg['mlx_audio_base']}")
    backend_type = "local/MLX" if not cfg.get("llm_base_url") else f"remote/{cfg['llm_base_url']}"
    print(f"[gateway] LLM: {cfg['llm_model']} ({backend_type})")
    llm_backend.preload()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    cfg = config.load()
    uvicorn.run(
        "app:app",
        host=cfg["host"],
        port=cfg["port"],
        log_level="info",
    )
