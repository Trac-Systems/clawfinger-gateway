"""Local Voice Gateway — FastAPI application."""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import httpx

import agent_interface
import time_utils
import config
import instruction_store
import llm_backend
import session_store
import voice_pipeline
from event_bus import bus
from endpoints.phone import adb as phone_adb
from endpoints.phone import routes as phone_routes_mod
from endpoints import robot as robot_mod
from endpoints.robot import controller as robot_ctrl
from endpoints.robot import perception as robot_perception
from endpoints.robot import memory as robot_memory
from endpoints.robot import skill_loader as robot_skills
from endpoints.phone.routes import (
    router as phone_router,
    call_config_response as _call_config_response,
    AGENT_ALLOWED_CALL_KEYS,
    _CALL_BODY_REMAP,
)

app = FastAPI(title="Local Voice Gateway", version="0.1.0")
app.include_router(phone_router)

_ROOT = Path(__file__).resolve().parent
_TMP_DIR = _ROOT / "tmp"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_STATIC_DIR = _ROOT / "static"
_START_TIME = time.time()

# Robot transport state (set during startup if robot.enabled)
_intercom_process = None
_intercom_bridge = None     # IntercomBridge or WebSocketRobotBridge
_ws_robot_bridge = None     # Only set when transport=websocket


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
# Generic API endpoints
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


@app.post("/api/session/end")
async def session_end(request: Request) -> JSONResponse:
    """Mark a session as ended (call hung up)."""
    _check_bearer(request)
    body = await request.json()
    sid = voice_pipeline.safe_text(str(body.get("session_id", "")))
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    ok = session_store.end_session(sid)
    if ok:
        await bus.publish("session.ended", {"session_id": sid}, session_id=sid)
    return JSONResponse({"ok": ok, "session_id": sid})


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
    # Filter out sensitive keys from config for display
    safe_cfg = {}
    for k, v in cfg.items():
        if "token" in k or "key" in k or "bearer" in k:
            continue
        if isinstance(v, dict):
            safe_cfg[k] = {sk: sv for sk, sv in v.items() if "token" not in sk and "key" not in sk}
        else:
            safe_cfg[k] = v
    return JSONResponse({
        "uptime_s": round(time.time() - _START_TIME),
        "total_calls": phone_routes_mod._CALL_COUNT,
        "error_count": phone_routes_mod._ERROR_COUNT,
        "active_sessions": len(session_store.active_sessions()),
        "ended_sessions": len(session_store.ended_sessions()),
        "ui_subscribers": bus.subscriber_count,
        "agents": agent_interface.list_agents(),
        "mlx_audio": voice_pipeline.check_mlx_audio(),
        "llm": llm_backend.check_health(),
        "config": safe_cfg,
    })


@app.post("/api/config")
async def update_config(request: Request) -> JSONResponse:
    """Hot-reload config from disk."""
    cfg = config.reload()
    await bus.publish("status.update", {"event": "config_reloaded"})
    safe_cfg = {}
    for k, v in cfg.items():
        if "token" in k or "key" in k or "bearer" in k:
            continue
        if isinstance(v, dict):
            safe_cfg[k] = {sk: sv for sk, sv in v.items() if "token" not in sk and "key" not in sk}
        else:
            safe_cfg[k] = v
    return JSONResponse({"ok": True, "config": safe_cfg})


# ---------------------------------------------------------------------------
# Instruction endpoints
# ---------------------------------------------------------------------------

@app.get("/api/instructions")
async def get_instructions() -> JSONResponse:
    return JSONResponse(instruction_store.snapshot())


@app.post("/api/instructions")
async def set_base_instruction(request: Request) -> JSONResponse:
    """Update the default system prompt in config.  All new sessions (and existing
    sessions without a session-scoped override) will use this prompt."""
    body = await request.json()
    text = str(body.get("text", ""))
    config.set("llm.system_prompt", text)
    config.save()
    await bus.publish("instructions.updated", {"scope": "global"})
    return JSONResponse({"ok": True, "scope": "global"})


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
                commands = msg.get("commands")
                # Try robot takeover reply first, then phone
                if robot_ctrl.get_takeover_agent():
                    robot_ctrl.resolve_agent_reply(
                        str(request_id), reply_text,
                        commands=commands if isinstance(commands, list) else None,
                    )
                else:
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
                if not sid:
                    active = list(session_store.active_sessions().keys())
                    if active:
                        sid = active[0]
                if text and sid:
                    audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, text)
                    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                    session_store.queue_inject(sid, text, audio_b64)
                    await bus.publish("agent.inject", {
                        "text": text,
                        "audio_base64": audio_b64,
                        "tts_ms": round(tts_ms, 1),
                    }, session_id=sid)

            elif msg_type == "set_instructions":
                text = str(msg.get("instructions", ""))
                sid = str(msg.get("session_id", ""))
                scope = str(msg.get("scope", "turn"))
                if scope == "global":
                    # Global mutable instructions disabled — cross-session bleed risk
                    await ws.send_json({"type": "set_instructions.ack", "ok": False,
                                        "error": "Global scope disabled. Use session or turn scope."})
                elif scope == "session" and sid:
                    instruction_store.set_session(sid, text)
                    await ws.send_json({"type": "set_instructions.ack", "ok": True, "scope": scope})
                    await bus.publish("instructions.updated", {"scope": scope, "session_id": sid})
                elif scope == "turn" and sid:
                    instruction_store.set_turn(sid, text)
                    await ws.send_json({"type": "set_instructions.ack", "ok": True, "scope": scope})
                    await bus.publish("instructions.updated", {"scope": scope, "session_id": sid})
                else:
                    await ws.send_json({"type": "set_instructions.ack", "ok": False,
                                        "error": "Missing session_id for session/turn scope."})

            elif msg_type == "set_call_config":
                for key, value in msg.get("config", {}).items():
                    cfg_key = _CALL_BODY_REMAP.get(key, key)
                    if cfg_key in AGENT_ALLOWED_CALL_KEYS:
                        # Handle TTS keys separately
                        if cfg_key.startswith("tts."):
                            config.set(cfg_key, value)
                        else:
                            config.set(f"phone.{cfg_key}", value)
                config.save()
                await ws.send_json({"type": "set_call_config.ack", "ok": True})
                await bus.publish("config.call_updated", _call_config_response())

            elif msg_type == "dial":
                number = str(msg.get("number", ""))
                result = await phone_adb.do_dial(number)
                await ws.send_json({"type": "dial.ack", **result})
                if result["ok"]:
                    await bus.publish("call.dial", {"number": number})

            elif msg_type == "hangup":
                sid = str(msg.get("session_id", ""))
                result = await phone_adb.do_hangup()
                if result["ok"]:
                    target = sid or phone_adb.single_active_session()
                    if target:
                        session_store.end_session(target)
                        await bus.publish("session.ended", {"session_id": target}, session_id=target)
                        result["session_id"] = target
                    await bus.publish("call.hangup", result)
                await ws.send_json({"type": "hangup.ack", **result})

            elif msg_type == "get_call_state":
                sid = str(msg.get("session_id", ""))
                history = session_store.get_history(sid)
                meta = session_store.active_sessions().get(sid)
                ended = session_store.is_ended(sid)
                all_meta = session_store.all_sessions().get(sid)
                resp_meta = meta or all_meta
                await ws.send_json({
                    "type": "call_state",
                    "session_id": sid,
                    "status": "ended" if ended else ("active" if resp_meta else "unknown"),
                    "ended_at": session_store._ENDED.get(sid) if ended else None,
                    "history": history,
                    "turn_count": len(resp_meta.get("turns", [])) if resp_meta else 0,
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

            elif msg_type == "end_session":
                sid = str(msg.get("session_id", ""))
                if sid:
                    ok = session_store.end_session(sid)
                    await ws.send_json({"type": "end_session.ack", "ok": ok, "session_id": sid})
                    if ok:
                        await bus.publish("session.ended", {"session_id": sid}, session_id=sid)
                else:
                    await ws.send_json({"type": "end_session.ack", "ok": False})

            elif msg_type == "robot_command":
                result = await robot_mod.dispatch_command(
                    config.get("robot.model", "unitree_g1"),
                    msg.get("command", {}),
                )
                await ws.send_json({"type": "robot.command.ack", **result})

            elif msg_type == "robot_status":
                cfg = _robot_config_response()
                await ws.send_json({"type": "robot.status", **cfg})

            elif msg_type == "robot_skill_list":
                skills = robot_skills.list_skills()
                await ws.send_json({"type": "robot.skill.list", "skills": skills})

            elif msg_type == "robot_project_status":
                project = robot_ctrl.current_project()
                await ws.send_json({"type": "robot.project.status", **(project or {"status": "idle"})})

            elif msg_type == "robot_project_cancel":
                result = await robot_ctrl.cancel_project()
                await ws.send_json({"type": "robot.project.cancel.ack", **result})

            elif msg_type == "robot_takeover":
                agent_id = agent_interface.get_agent_id(ws)
                robot_ctrl.set_takeover_agent(agent_id)
                await ws.send_json({"type": "robot_takeover.ack", "ok": True})

            elif msg_type == "robot_release":
                robot_ctrl.set_takeover_agent(None)
                await ws.send_json({"type": "robot_release.ack", "ok": True})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        # Release robot takeover if this agent held it
        agent_id = agent_interface.get_agent_id(ws)
        if agent_id and robot_ctrl.get_takeover_agent() == agent_id:
            robot_ctrl.set_takeover_agent(None)
        await agent_interface.agent_disconnect(ws)


@app.post("/api/agent/inject")
async def agent_inject_rest(request: Request) -> JSONResponse:
    body = await request.json()
    text = voice_pipeline.safe_text(str(body.get("text", "")))
    session_id = str(body.get("session_id", ""))
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if not session_id:
        active = list(session_store.active_sessions().keys())
        if active:
            session_id = active[0]
    if not session_id:
        raise HTTPException(status_code=400, detail="no active session")
    audio_bytes, tts_ms = await asyncio.to_thread(voice_pipeline.synthesize, text)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    session_store.queue_inject(session_id, text, audio_b64)
    await bus.publish("agent.inject", {
        "text": text,
        "audio_base64": audio_b64,
        "tts_ms": round(tts_ms, 1),
    }, session_id=session_id)
    return JSONResponse({"ok": True, "tts_ms": round(tts_ms, 1), "session_id": session_id})


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


def _resolve_session(session_id: str) -> str:
    """Resolve '_active' to most recently active session, or return as-is."""
    if session_id == "_active":
        sid = session_store.most_recent_active_session()
        if not sid:
            raise HTTPException(status_code=404, detail="no active session")
        return sid
    return session_id


@app.get("/api/agent/context/{session_id}")
async def get_agent_context(session_id: str) -> JSONResponse:
    session_id = _resolve_session(session_id)
    knowledge = instruction_store.get_agent_knowledge(session_id)
    return JSONResponse({
        "session_id": session_id,
        "knowledge": knowledge,
        "has_knowledge": bool(knowledge),
    })


@app.post("/api/agent/context/{session_id}")
async def set_agent_context(session_id: str, request: Request) -> JSONResponse:
    session_id = _resolve_session(session_id)
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
    session_id = _resolve_session(session_id)
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

_PIPER_VOICES = {
    "Male": ["thorsten-high", "thorsten-medium", "thorsten-low", "karlsson-low", "pavoque-low"],
    "Female": ["eva_k-x_low", "kerstin-low", "ramona-low"],
    "Emotional": ["thorsten_emotional-medium"],
}

_PIPER_EMOTIONS = {
    "amused": 0, "angry": 1, "disgusted": 2, "drunk": 3,
    "neutral": 4, "sleepy": 5, "surprised": 6, "whisper": 7,
}

_TTS_ALIAS = {"voice": "tts.voice", "speed": "tts.speed", "lang": "tts.lang"}

_TTS_WRITABLE_KEYS = {
    "tts.voice", "tts.speed", "tts.lang",
    "tts.piper.voice", "tts.piper.speaker", "tts.piper.length_scale",
    "tts.piper.noise_scale", "tts.piper.noise_w", "tts.piper.sentence_silence",
}

# Aliases for TTS POST body keys → dotted config paths
_TTS_BODY_TO_PATH = {
    "voice": "tts.voice", "speed": "tts.speed", "lang": "tts.lang",
    "tts_voice": "tts.voice", "tts_speed": "tts.speed", "tts_lang": "tts.lang",
    "piper_voice": "tts.piper.voice", "piper_speaker": "tts.piper.speaker",
    "piper_length_scale": "tts.piper.length_scale",
    "piper_noise_scale": "tts.piper.noise_scale",
    "piper_noise_w": "tts.piper.noise_w",
    "piper_sentence_silence": "tts.piper.sentence_silence",
}


def _tts_config_response() -> dict:
    tts = config.section("tts")
    piper = tts.get("piper", {})
    lang = tts.get("lang", "en")
    resp = {"lang": lang, "model": tts.get("model", "")}
    if lang == "de":
        resp["piper_voice"] = piper.get("voice", "thorsten-high")
        resp["piper_speaker"] = piper.get("speaker", 0)
        resp["piper_length_scale"] = piper.get("length_scale", 1.0)
        resp["piper_noise_scale"] = piper.get("noise_scale", 0.667)
        resp["piper_noise_w"] = piper.get("noise_w", 0.8)
        resp["piper_sentence_silence"] = piper.get("sentence_silence", 0.2)
        resp["voices"] = _PIPER_VOICES
        resp["emotions"] = _PIPER_EMOTIONS
    else:
        resp["voice"] = tts.get("voice", "am_adam")
        resp["speed"] = tts.get("speed", 1.2)
        is_kokoro = "kokoro" in resp["model"].lower()
        resp["voices"] = _KOKORO_VOICES if is_kokoro else {}
    return resp


@app.get("/api/config/tts")
async def get_tts_config() -> JSONResponse:
    return JSONResponse(_tts_config_response())


@app.post("/api/config/tts")
async def update_tts_config(request: Request) -> JSONResponse:
    body = await request.json()
    # If switching to German, verify Piper is reachable first
    new_lang = body.get("lang") or body.get("tts_lang")
    if new_lang == "de" and config.get("tts.lang", "en") != "de":
        piper_base = config.get("tts.piper.base", "http://127.0.0.1:5123")
        try:
            probe = httpx.post(piper_base, json={"text": "test"}, timeout=5)
            if probe.status_code != 200:
                raise Exception(f"HTTP {probe.status_code}")
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "error": f"Piper TTS is not running on {piper_base} — cannot switch to German. Start the gateway with a Piper model to enable German TTS."},
                status_code=400,
            )
    for body_key, value in body.items():
        cfg_path = _TTS_BODY_TO_PATH.get(body_key)
        if cfg_path and cfg_path in _TTS_WRITABLE_KEYS:
            config.set(cfg_path, value)
    config.save()
    resp = _tts_config_response()
    await bus.publish("config.tts_updated", resp)
    return JSONResponse({"ok": True, **resp})


@app.post("/api/tts/preview")
async def tts_preview(request: Request) -> JSONResponse:
    body = await request.json()
    tts = config.section("tts")
    asr = config.section("asr")
    lang = tts.get("lang", "en")

    if lang == "de":
        text = voice_pipeline.safe_text(str(body.get("text", ""))) or "Hallo, das ist eine Sprachvorschau."
        trimmed = voice_pipeline.trim_for_tts(text)
        wav_bytes = await asyncio.to_thread(voice_pipeline._synthesize_piper, trimmed)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        return JSONResponse({
            "ok": True,
            "audio_base64": audio_b64,
            "lang": "de",
            "piper_voice": tts.get("piper", {}).get("voice", "thorsten-high"),
        })
    else:
        text = voice_pipeline.safe_text(str(body.get("text", ""))) or "Hello, this is a voice preview."
        preview_voice = str(body.get("voice", "")) or tts.get("voice", "am_adam")
        preview_speed = float(body.get("speed", 0)) or tts.get("speed", 1.2)

        mlx_base = (tts.get("mlx_audio_base") or asr["backend"]).rstrip("/")
        payload = {
            "model": tts["model"],
            "input": voice_pipeline.trim_for_tts(text),
            "voice": preview_voice,
            "speed": preview_speed,
            "response_format": "wav",
        }

        response = await asyncio.to_thread(
            lambda: httpx.post(f"{mlx_base}/v1/audio/speech", json=payload, timeout=180)
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

_LLM_WRITABLE_KEYS = {
    "llm.max_tokens", "llm.temperature", "llm.top_p", "llm.top_k",
    "llm.repeat_penalty", "llm.stop",
    "llm.top_p_enabled", "llm.top_k_enabled", "llm.context_tokens",
    "llm.max_history_turns",
    "llm.model", "llm.base_url", "llm.api_key",
}

# Short aliases accepted by POST body → dotted config path
_LLM_BODY_TO_PATH = {
    "model": "llm.model",
    "base_url": "llm.base_url",
    "api_key": "llm.api_key",
    "max_tokens": "llm.max_tokens",
    "temperature": "llm.temperature",
    "top_p": "llm.top_p",
    "top_k": "llm.top_k",
    "top_p_enabled": "llm.top_p_enabled",
    "top_k_enabled": "llm.top_k_enabled",
    "repeat_penalty": "llm.repeat_penalty",
    "stop": "llm.stop",
    "context_tokens": "llm.context_tokens",
    "max_history_turns": "llm.max_history_turns",
}


def _llm_config_response() -> dict:
    llm = config.section("llm")
    return {
        "model": llm.get("model", ""),
        "base_url": llm.get("base_url", ""),
        "has_api_key": bool(llm.get("api_key", "")),
        "max_tokens": llm.get("max_tokens", 400),
        "context_tokens": llm.get("context_tokens", 0),
        "context_tokens_effective": llm.get("context_tokens", 0) or llm_backend.get_context_window(),
        "max_history_turns": llm.get("max_history_turns", 8),
        "temperature": llm.get("temperature", 0.2),
        "top_p": llm.get("top_p", 1.0),
        "top_p_enabled": llm.get("top_p_enabled", True),
        "top_k": llm.get("top_k", 0),
        "top_k_enabled": llm.get("top_k_enabled", True),
        "repeat_penalty": llm.get("repeat_penalty", 1.0),
        "stop": llm.get("stop", []),
        "is_local": not llm.get("base_url"),
    }


@app.get("/api/config/llm")
async def get_llm_config() -> JSONResponse:
    return JSONResponse(_llm_config_response())


@app.post("/api/config/llm")
async def update_llm_config(request: Request) -> JSONResponse:
    body = await request.json()

    for body_key, value in body.items():
        cfg_path = _LLM_BODY_TO_PATH.get(body_key)
        if cfg_path and cfg_path in _LLM_WRITABLE_KEYS:
            config.set(cfg_path, value)

    config.save()
    await bus.publish("config.llm_updated", _llm_config_response())
    return JSONResponse({"ok": True, **_llm_config_response()})


# ---------------------------------------------------------------------------
# Agent call state endpoint
# ---------------------------------------------------------------------------

@app.get("/api/agent/call/{sid}")
async def agent_call_state(sid: str) -> JSONResponse:
    history = session_store.get_history(sid)
    meta = session_store.active_sessions().get(sid)
    ended = session_store.is_ended(sid)
    all_meta = session_store.all_sessions().get(sid)
    instructions = {
        "base": instruction_store.get_base(),
        "session": instruction_store.get_session(sid),
        "pending_turn": instruction_store.get_turn(sid),
    }
    has_takeover = agent_interface.get_takeover_agent(sid) is not None
    resp_meta = meta or all_meta
    return JSONResponse({
        "session_id": sid,
        "status": "ended" if ended else ("active" if resp_meta else "unknown"),
        "ended_at": session_store._ENDED.get(sid) if ended else None,
        "history": history,
        "turn_count": len(resp_meta.get("turns", [])) if resp_meta else 0,
        "instructions": instructions,
        "agent_takeover": has_takeover,
        "created_at": resp_meta.get("created_at") if resp_meta else None,
    })


# ---------------------------------------------------------------------------
# Robot config endpoints
# ---------------------------------------------------------------------------

# Keys agents are NOT allowed to set
_ROBOT_SECURITY_KEYS = {
    "intercom_key", "enable_low_level",
}

# Top-level robot config keys settable via REST API
_ROBOT_SETTABLE_KEYS = {
    "enabled", "model", "transport", "heartbeat_interval",
    "disconnect_timeout", "disconnect_debounce",
    "safety_stop_on_disconnect", "offline_mode",
    "voice", "voice_lang", "voice_speed",
    "wake_word", "wake_phrases", "wake_word_timeout", "wake_word_model",
}

# Nested robot config sections settable via REST API (sent as objects)
_ROBOT_SETTABLE_SECTIONS = {
    "safety", "camera", "audio_monitor", "memory", "llm",
}


def _robot_llm_response(llm: dict) -> dict:
    """Build robot LLM config response (mirrors _llm_config_response pattern)."""
    return {
        "model": llm.get("model", ""),
        "base_url": llm.get("base_url", ""),
        "has_api_key": bool(llm.get("api_key", "")),
        "max_tokens": llm.get("max_tokens", 80),
        "context_tokens": llm.get("context_tokens", 0),
        "max_history_turns": llm.get("max_history_turns", 8),
        "temperature": llm.get("temperature", 0.25),
        "top_p": llm.get("top_p", 1.0),
        "top_p_enabled": llm.get("top_p_enabled", True),
        "top_k": llm.get("top_k", 0),
        "top_k_enabled": llm.get("top_k_enabled", True),
        "repeat_penalty": llm.get("repeat_penalty", 1.0),
        "stop": llm.get("stop", []),
        "system_prompt": llm.get("system_prompt", ""),
        "is_local": not llm.get("base_url"),
    }


def _robot_config_response() -> dict:
    """Build robot config response dict — returns full robot config."""
    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")
    model_info = robot_mod.get_model(model_name)
    resp = {
        "enabled": robot_cfg.get("enabled", False),
        "model": model_name,
        "connected": model_info["connected"] if model_info else False,
        "capabilities": sorted(model_info["capabilities"]) if model_info else [],
        "transport": robot_cfg.get("transport", "intercom"),
        "heartbeat_interval": robot_cfg.get("heartbeat_interval", 5),
        "disconnect_timeout": robot_cfg.get("disconnect_timeout", 15),
        "disconnect_debounce": robot_cfg.get("disconnect_debounce", 3),
        "safety_stop_on_disconnect": robot_cfg.get("safety_stop_on_disconnect", True),
        "offline_mode": robot_cfg.get("offline_mode", "complete_task"),
        "voice": robot_cfg.get("voice", "am_adam"),
        "voice_lang": robot_cfg.get("voice_lang", "en"),
        "voice_speed": robot_cfg.get("voice_speed", 1.0),
        "wake_word": robot_cfg.get("wake_word", "Robert"),
        "wake_phrases": robot_cfg.get("wake_phrases", []),
        "wake_word_timeout": robot_cfg.get("wake_word_timeout", 30),
        "safety": robot_cfg.get("safety", {}),
        "camera": robot_cfg.get("camera", {}),
        "audio_monitor": robot_cfg.get("audio_monitor", {}),
        "memory": robot_cfg.get("memory", {}),
        "llm": _robot_llm_response(robot_cfg.get("llm", {})),
        "model_config": robot_cfg.get(model_name, {}),
        "kokoro_voices": _KOKORO_VOICES,
        "piper_voices": _PIPER_VOICES,
        "piper_emotions": _PIPER_EMOTIONS,
    }
    # Add transport state when bridge is available
    if _intercom_bridge is not None:
        resp["transport_connected"] = _intercom_bridge.is_connected()
        resp["intercom_channel"] = robot_cfg.get("intercom_channel", "clawfinger-robot-g1")
    if _intercom_process is not None:
        resp["intercom_running"] = _intercom_process.is_running
    return resp


@app.get("/api/config/robot")
async def get_robot_config() -> JSONResponse:
    return JSONResponse(_robot_config_response())


@app.post("/api/config/robot")
async def update_robot_config(request: Request) -> JSONResponse:
    body = await request.json()
    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")

    # Validate Piper availability before allowing German voice
    new_voice_lang = body.get("voice_lang")
    if new_voice_lang == "de" and robot_cfg.get("voice_lang", "en") != "de":
        piper_base = config.get("tts.piper.base", "http://127.0.0.1:5123")
        try:
            probe = httpx.post(piper_base, json={"text": "test"}, timeout=5)
            if probe.status_code != 200:
                raise Exception(f"HTTP {probe.status_code}")
        except Exception:
            return JSONResponse(
                {"ok": False, "error": f"Piper TTS is not running on {piper_base} — cannot switch to German."},
                status_code=400,
            )

    for key, value in body.items():
        if key in _ROBOT_SECURITY_KEYS:
            continue  # skip security keys
        if key.startswith(f"{model_name}."):
            # Model-specific key (e.g. unitree_g1.jetson_ip)
            config.set(f"robot.{model_name}.{key[len(model_name)+1:]}", value)
        elif key in _ROBOT_SETTABLE_SECTIONS and isinstance(value, dict):
            # Nested section — merge keys
            for sk, sv in value.items():
                if sk.startswith("_"):
                    continue  # skip comments
                config.set(f"robot.{key}.{sk}", sv)
        elif key in _ROBOT_SETTABLE_KEYS:
            config.set(f"robot.{key}", value)
    config.save()
    await bus.publish("config.robot_updated", _robot_config_response())
    return JSONResponse({"ok": True, **_robot_config_response()})


# ---------------------------------------------------------------------------
# Robot skills + project endpoints
# ---------------------------------------------------------------------------

@app.get("/api/robot/skills")
async def list_robot_skills(request: Request) -> JSONResponse:
    _check_bearer(request)
    return JSONResponse(robot_skills.list_skills())


@app.get("/api/robot/skills/{name}/{topic}")
async def get_robot_skill_topic(name: str, topic: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    skill = robot_skills.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill not found: {name}")
    if skill.get("execution_mode") == "fast":
        raise HTTPException(status_code=400, detail="Fast-path skills have no topic content")
    content = robot_skills.get_topic(name, topic)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Topic not found: {name}/{topic}")
    return JSONResponse({"skill": name, "topic": topic, "content": content})


@app.get("/api/robot/project")
async def get_robot_project(request: Request) -> JSONResponse:
    _check_bearer(request)
    project = robot_ctrl.current_project()
    if project:
        return JSONResponse(project)
    return JSONResponse({"status": "idle"})


@app.post("/api/robot/project/cancel")
async def cancel_robot_project(request: Request) -> JSONResponse:
    _check_bearer(request)
    result = await robot_ctrl.cancel_project()
    return JSONResponse(result)


@app.post("/api/robot/project/start")
async def start_robot_project(request: Request) -> JSONResponse:
    """Start a robot project from a text prompt (same as voice turn)."""
    _check_bearer(request)
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "text required"}, status_code=400)
    result = await robot_ctrl.handle_voice_turn(text)
    project = robot_ctrl.current_project()
    return JSONResponse({"ok": True, "say": result.get("say"), "project": project})


# ---------------------------------------------------------------------------
# Robot perception endpoints
# ---------------------------------------------------------------------------

def _check_perception_source(model_name: str, source: str, kind: str = "camera") -> None:
    """Validate source exists in model's perception. Raises HTTPException if not."""
    perception = robot_mod.get_perception(model_name)
    key = "cameras" if kind == "camera" else "microphones"
    sources = perception.get(key, [])
    if not sources:
        raise HTTPException(400, f"Robot model '{model_name}' has no {kind}s")
    valid_ids = [s["id"] for s in sources]
    if source not in valid_ids:
        raise HTTPException(400, f"Unknown {kind} source '{source}'. Available: {valid_ids}")


def _default_source(model_name: str, kind: str = "camera") -> str:
    """Return the first available source ID for a kind, or raise 400."""
    perception = robot_mod.get_perception(model_name)
    key = "cameras" if kind == "camera" else "microphones"
    sources = perception.get(key, [])
    if not sources:
        raise HTTPException(400, f"Robot model '{model_name}' has no {kind}s")
    return sources[0]["id"]


def _require_robot_connected() -> str:
    """Check robot is connected, return model name. Raises HTTPException if not."""
    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")
    model = robot_mod.get_model(model_name)
    if not model or not model.get("connected"):
        raise HTTPException(503, "Robot not connected")
    return model_name


@app.get("/api/robot/perception")
async def get_perception_sources(request: Request) -> JSONResponse:
    """List available perception sources (cameras + mics) from model defaults."""
    _check_bearer(request)
    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")
    perception = robot_mod.get_perception(model_name)
    return JSONResponse({
        "model": model_name,
        "cameras": perception.get("cameras", []),
        "microphones": perception.get("microphones", []),
        "active_streams": robot_perception.active_streams(),
        "active_monitors": robot_perception.active_monitors(),
    })


@app.post("/api/robot/camera/snapshot")
async def robot_camera_snapshot(request: Request) -> JSONResponse:
    """Capture a single frame from a camera source."""
    _check_bearer(request)
    model_name = _require_robot_connected()
    body = await request.json() if await request.body() else {}
    source = body.get("source") or _default_source(model_name, "camera")
    _check_perception_source(model_name, source, "camera")

    robot_cfg = config.section("robot")
    cam_cfg = robot_cfg.get("camera", {})
    result = await _intercom_bridge.send_and_wait({
        "type": "camera_snapshot",
        "source": source,
        "width": body.get("width", cam_cfg.get("width", 640)),
        "height": body.get("height", cam_cfg.get("height", 480)),
        "quality": body.get("quality", cam_cfg.get("quality", 50)),
    }, timeout=body.get("timeout", 10.0))

    if not result or not result.get("ok"):
        raise HTTPException(502, result.get("error", "Snapshot failed") if result else "No response from robot")

    # Store snapshot for later retrieval
    image_b64 = result.get("image_base64", "")
    if image_b64:
        robot_perception.set_snapshot(source, base64.b64decode(image_b64))

    await bus.publish("robot.snapshot", {
        "source": source,
        "image_base64": image_b64,
    }, endpoint="robot")

    return JSONResponse({
        "ok": True,
        "source": source,
        "image_base64": image_b64,
        "width": result.get("width"),
        "height": result.get("height"),
    })


@app.post("/api/robot/camera/describe")
async def robot_camera_describe(request: Request) -> JSONResponse:
    """Capture a frame and run VLM scene description on the Jetson."""
    _check_bearer(request)
    model_name = _require_robot_connected()
    body = await request.json() if await request.body() else {}
    source = body.get("source") or _default_source(model_name, "camera")
    _check_perception_source(model_name, source, "camera")

    result = await _intercom_bridge.send_and_wait({
        "type": "camera_describe",
        "source": source,
        "prompt": body.get("prompt", "Describe what you see."),
    }, timeout=body.get("timeout", 30.0))

    if not result or not result.get("ok"):
        raise HTTPException(502, result.get("error", "Describe failed") if result else "No response from robot")

    await bus.publish("robot.scene_description", {
        "source": source,
        "description": result.get("description", ""),
    }, endpoint="robot")

    return JSONResponse({
        "ok": True,
        "source": source,
        "description": result.get("description", ""),
        "image_base64": result.get("image_base64", ""),
    })


@app.get("/api/robot/camera/stream")
async def robot_camera_stream(
    request: Request,
    source: str = Query(""),
    token: str = Query(""),
) -> StreamingResponse:
    """MJPEG video stream from a camera source.

    Uses query-param auth (?token=) since <img> tags can't send headers.
    """
    # Auth: accept query param OR header
    bearer = config.get("bearer_token", "")
    if bearer:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {bearer}" and token != bearer:
            raise HTTPException(401, "Unauthorized")

    model_name = _require_robot_connected()
    if not source:
        source = _default_source(model_name, "camera")
    _check_perception_source(model_name, source, "camera")

    if not robot_perception.is_streaming(source):
        raise HTTPException(409, f"Stream not active for source '{source}'. POST /api/robot/camera/stream/start first.")

    async def mjpeg_generator():
        async for frame in robot_perception.frame_generator(source):
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                b"\r\n" + frame + b"\r\n"
            )

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/robot/camera/stream/start")
async def robot_camera_stream_start(request: Request) -> JSONResponse:
    """Start continuous video stream from a camera source."""
    _check_bearer(request)
    model_name = _require_robot_connected()
    body = await request.json() if await request.body() else {}
    source = body.get("source") or _default_source(model_name, "camera")
    _check_perception_source(model_name, source, "camera")

    if robot_perception.is_streaming(source):
        return JSONResponse({"ok": True, "source": source, "detail": "already streaming"})

    robot_cfg = config.section("robot")
    cam_cfg = robot_cfg.get("camera", {})
    robot_perception.set_streaming(source, True)
    await _intercom_bridge.send({
        "type": "camera_stream_start",
        "source": source,
        "fps": body.get("fps", cam_cfg.get("stream_fps", 5)),
        "width": body.get("width", cam_cfg.get("width", 640)),
        "height": body.get("height", cam_cfg.get("height", 480)),
        "quality": body.get("quality", cam_cfg.get("quality", 50)),
    })

    await bus.publish("robot.camera.stream_started", {"source": source}, endpoint="robot")
    return JSONResponse({"ok": True, "source": source})


@app.post("/api/robot/camera/stream/stop")
async def robot_camera_stream_stop(request: Request) -> JSONResponse:
    """Stop video stream from a camera source."""
    _check_bearer(request)
    model_name = _require_robot_connected()
    body = await request.json() if await request.body() else {}
    source = body.get("source") or _default_source(model_name, "camera")

    robot_perception.set_streaming(source, False)
    await _intercom_bridge.send({
        "type": "camera_stream_stop",
        "source": source,
    })

    await bus.publish("robot.camera.stream_stopped", {"source": source}, endpoint="robot")
    return JSONResponse({"ok": True, "source": source})


@app.post("/api/robot/audio/monitor/start")
async def robot_audio_monitor_start(request: Request) -> JSONResponse:
    """Start mic audio monitoring stream."""
    _check_bearer(request)
    model_name = _require_robot_connected()
    body = await request.json() if await request.body() else {}
    source = body.get("source") or _default_source(model_name, "microphone")
    _check_perception_source(model_name, source, "microphone")

    if robot_perception.is_monitoring(source):
        return JSONResponse({"ok": True, "source": source, "detail": "already monitoring"})

    robot_cfg = config.section("robot")
    audio_cfg = robot_cfg.get("audio_monitor", {})
    robot_perception.set_monitoring(source, True)
    await _intercom_bridge.send({
        "type": "audio_monitor_start",
        "source": source,
        "sample_rate": body.get("sample_rate", audio_cfg.get("sample_rate", 16000)),
        "channels": body.get("channels", audio_cfg.get("channels", 1)),
        "chunk_ms": body.get("chunk_ms", audio_cfg.get("chunk_ms", 100)),
    })

    await bus.publish("robot.audio.monitor_started", {"source": source}, endpoint="robot")
    return JSONResponse({"ok": True, "source": source})


@app.post("/api/robot/audio/monitor/stop")
async def robot_audio_monitor_stop(request: Request) -> JSONResponse:
    """Stop mic audio monitoring stream."""
    _check_bearer(request)
    model_name = _require_robot_connected()
    body = await request.json() if await request.body() else {}
    source = body.get("source") or _default_source(model_name, "microphone")

    robot_perception.set_monitoring(source, False)
    await _intercom_bridge.send({
        "type": "audio_monitor_stop",
        "source": source,
    })

    await bus.publish("robot.audio.monitor_stopped", {"source": source}, endpoint="robot")
    return JSONResponse({"ok": True, "source": source})


# ---------------------------------------------------------------------------
# Robot spatial memory endpoints
# ---------------------------------------------------------------------------

@app.get("/api/robot/memory/persons")
async def memory_list_persons(request: Request) -> JSONResponse:
    _check_bearer(request)
    return JSONResponse(robot_memory.list_persons())


@app.post("/api/robot/memory/persons")
async def memory_add_person(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    description = body.get("description", "")
    ref_images = []
    for b64 in body.get("reference_images", []):
        try:
            ref_images.append(base64.b64decode(b64))
        except Exception:
            raise HTTPException(400, "Invalid base64 in reference_images")
    if ref_images and robot_memory.clip_available():
        result = await asyncio.to_thread(
            robot_memory.add_person, name, description, ref_images)
    else:
        result = robot_memory.add_person(name, description,
                                         ref_images if ref_images else None)
    await bus.publish("robot.memory.person_added", result, endpoint="robot")
    return JSONResponse(result)


@app.get("/api/robot/memory/persons/{person_id}")
async def memory_get_person(person_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    person = robot_memory.get_person(person_id)
    if not person:
        raise HTTPException(404, f"Person not found: {person_id}")
    return JSONResponse(person)


@app.delete("/api/robot/memory/persons/{person_id}")
async def memory_delete_person(person_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    if not robot_memory.delete_person(person_id):
        raise HTTPException(404, f"Person not found: {person_id}")
    await bus.publish("robot.memory.person_deleted",
                      {"id": person_id}, endpoint="robot")
    return JSONResponse({"ok": True, "id": person_id})


@app.post("/api/robot/memory/persons/{person_id}/reference")
async def memory_add_person_ref(person_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    b64 = body.get("image", "")
    if not b64:
        raise HTTPException(400, "image (base64) is required")
    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")
    result = await asyncio.to_thread(
        robot_memory.add_person_reference, person_id, image_bytes)
    if not result:
        raise HTTPException(404, f"Person not found or CLIP unavailable: {person_id}")
    return JSONResponse(result)


@app.get("/api/robot/memory/objects")
async def memory_list_objects(request: Request) -> JSONResponse:
    _check_bearer(request)
    return JSONResponse(robot_memory.list_objects())


@app.post("/api/robot/memory/objects")
async def memory_add_object(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    description = body.get("description", "")
    ref_images = []
    for b64 in body.get("reference_images", []):
        try:
            ref_images.append(base64.b64decode(b64))
        except Exception:
            raise HTTPException(400, "Invalid base64 in reference_images")
    if ref_images and robot_memory.clip_available():
        result = await asyncio.to_thread(
            robot_memory.add_object, name, description, ref_images)
    else:
        result = robot_memory.add_object(name, description,
                                         ref_images if ref_images else None)
    await bus.publish("robot.memory.object_added", result, endpoint="robot")
    return JSONResponse(result)


@app.get("/api/robot/memory/objects/{object_id}")
async def memory_get_object(object_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    obj = robot_memory.get_object(object_id)
    if not obj:
        raise HTTPException(404, f"Object not found: {object_id}")
    return JSONResponse(obj)


@app.delete("/api/robot/memory/objects/{object_id}")
async def memory_delete_object(object_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    if not robot_memory.delete_object(object_id):
        raise HTTPException(404, f"Object not found: {object_id}")
    await bus.publish("robot.memory.object_deleted",
                      {"id": object_id}, endpoint="robot")
    return JSONResponse({"ok": True, "id": object_id})


@app.post("/api/robot/memory/objects/{object_id}/reference")
async def memory_add_object_ref(object_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    b64 = body.get("image", "")
    if not b64:
        raise HTTPException(400, "image (base64) is required")
    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")
    result = await asyncio.to_thread(
        robot_memory.add_object_reference, object_id, image_bytes)
    if not result:
        raise HTTPException(404, f"Object not found or CLIP unavailable: {object_id}")
    return JSONResponse(result)


@app.get("/api/robot/memory/rooms")
async def memory_list_rooms(request: Request) -> JSONResponse:
    _check_bearer(request)
    return JSONResponse(robot_memory.list_rooms())


@app.post("/api/robot/memory/rooms")
async def memory_add_room(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    description = body.get("description", "")
    ref_image = None
    if body.get("reference_image"):
        try:
            ref_image = base64.b64decode(body["reference_image"])
        except Exception:
            raise HTTPException(400, "Invalid base64 reference_image")
    if ref_image and robot_memory.clip_available():
        result = await asyncio.to_thread(
            robot_memory.add_room, name, description, ref_image)
    else:
        result = robot_memory.add_room(name, description, ref_image)
    await bus.publish("robot.memory.room_added", result, endpoint="robot")
    return JSONResponse(result)


@app.delete("/api/robot/memory/rooms/{room_id}")
async def memory_delete_room(room_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    if not robot_memory.delete_room(room_id):
        raise HTTPException(404, f"Room not found: {room_id}")
    await bus.publish("robot.memory.room_deleted",
                      {"id": room_id}, endpoint="robot")
    return JSONResponse({"ok": True, "id": room_id})


@app.get("/api/robot/memory/routines")
async def memory_list_routines(request: Request) -> JSONResponse:
    _check_bearer(request)
    return JSONResponse(robot_memory.list_routines())


@app.post("/api/robot/memory/routines")
async def memory_add_routine(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(400, "name is required")
    schedule = body.get("schedule", "")
    description = body.get("description", "")
    result = robot_memory.add_routine(name, schedule, description)
    await bus.publish("robot.memory.routine_added", result, endpoint="robot")
    return JSONResponse(result)


@app.delete("/api/robot/memory/routines/{routine_id}")
async def memory_delete_routine(routine_id: str, request: Request) -> JSONResponse:
    _check_bearer(request)
    if not robot_memory.delete_routine(routine_id):
        raise HTTPException(404, f"Routine not found: {routine_id}")
    await bus.publish("robot.memory.routine_deleted",
                      {"id": routine_id}, endpoint="robot")
    return JSONResponse({"ok": True, "id": routine_id})


@app.post("/api/robot/memory/observations")
async def memory_add_observation(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    metadata = body.get("metadata", {})

    # If embedding is provided, store directly (pre-computed from observer)
    embedding = body.get("embedding")
    if embedding and isinstance(embedding, list):
        obs_id = robot_memory.add_observation(embedding, metadata)
        return JSONResponse({"ok": True, "id": obs_id})

    # If image is provided, embed on Mac Mini (convenience/testing only)
    image_b64 = body.get("image")
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(400, "Invalid base64 image")
        obs_id = await asyncio.to_thread(
            robot_memory.ingest_frame, image_bytes, metadata)
        return JSONResponse({"ok": True, "id": obs_id})

    raise HTTPException(400, "Either embedding or image is required")


@app.post("/api/robot/memory/query")
async def memory_query(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    query_type = body.get("type", "text")
    n_results = body.get("n_results", 10)
    filters = body.get("filters", {})

    # Natural time filter support
    time_filter = body.get("time_filter", "")
    if time_filter:
        parsed = time_utils.parse_natural_time(time_filter)
        if parsed:
            filters["time_start"], filters["time_end"] = parsed

    if query_type == "text":
        text = body.get("text", "")
        if not text:
            raise HTTPException(400, "text is required for text query")
        results = await asyncio.to_thread(
            robot_memory.query_text, text, n_results, filters)
        return JSONResponse({"results": results, "count": len(results)})

    elif query_type == "image":
        image_b64 = body.get("image", "")
        if not image_b64:
            raise HTTPException(400, "image is required for image query")
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(400, "Invalid base64 image")
        results = await asyncio.to_thread(
            robot_memory.query_image, image_bytes, n_results, filters)
        return JSONResponse({"results": results, "count": len(results)})

    elif query_type == "nearby":
        x = body.get("x", 0.0)
        y = body.get("y", 0.0)
        z = body.get("z", 0.0)
        radius = body.get("radius", 2.0)
        results = robot_memory.query_nearby(x, y, z, radius, n_results, filters)
        return JSONResponse({"results": results, "count": len(results)})

    elif query_type == "person_sightings":
        person_id = body.get("person_id", "")
        if not person_id:
            raise HTTPException(400, "person_id is required")
        results = robot_memory.person_sightings(
            person_id,
            time_start=filters.get("time_start"),
            time_end=filters.get("time_end"),
            room=filters.get("room"),
        )
        return JSONResponse({"results": results, "count": len(results)})

    elif query_type == "object_sightings":
        object_id = body.get("object_id", "")
        if not object_id:
            raise HTTPException(400, "object_id is required")
        results = robot_memory.object_sightings(
            object_id,
            time_start=filters.get("time_start"),
            time_end=filters.get("time_end"),
            room=filters.get("room"),
        )
        return JSONResponse({"results": results, "count": len(results)})

    elif query_type == "room_activity":
        room = body.get("room", "")
        if not room:
            raise HTTPException(400, "room is required")
        results = robot_memory.room_activity(
            room,
            time_start=filters.get("time_start"),
            time_end=filters.get("time_end"),
        )
        return JSONResponse({"results": results, "count": len(results)})

    else:
        raise HTTPException(400, f"Unknown query type: {query_type}")


@app.get("/api/robot/memory/stats")
async def memory_stats(request: Request) -> JSONResponse:
    _check_bearer(request)
    return JSONResponse(robot_memory.stats())


@app.post("/api/robot/memory/last_seen")
async def memory_last_seen(request: Request) -> JSONResponse:
    _check_bearer(request)
    body = await request.json()
    result = robot_memory.last_seen(
        entity_name=body.get("entity_name"),
        entity_type=body.get("entity_type"),
        room=body.get("room"),
    )
    if result is None:
        return JSONResponse({"found": False})
    return JSONResponse({"found": True, "result": result})


# ---------------------------------------------------------------------------
# UI WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await ws.accept()
    await bus.subscribe(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type", "")
            if msg_type == "ping":
                await ws.send_json({"type": "pong"})
            elif msg_type == "robot_command":
                cmd = msg.get("command", {})
                result = await robot_mod.dispatch_command(
                    config.get("robot.model", "unitree_g1"),
                    cmd,
                )
                await ws.send_json({"type": "robot.command.ack", **result})
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
    html = index_path.read_text(encoding="utf-8")
    # Inject bearer token so control-center JS can auth robot API calls
    token = config.get("bearer_token", "")
    html = html.replace("<script>", f"<script>window.__BEARER__={json.dumps(token)};", 1)
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

async def _periodic_sweep() -> None:
    """Background task: sweep stale sessions every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        try:
            stale = session_store.sweep_stale()
            for stale_sid in stale:
                await bus.publish("session.ended", {"session_id": stale_sid, "reason": "stale"}, session_id=stale_sid)
        except Exception:
            pass  # Don't crash the background loop


async def _start_ws_transport() -> None:
    """Start WebSocket robot transport (for sim adapter / direct connections)."""
    global _intercom_bridge, _ws_robot_bridge

    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")

    from transport.ws_bridge import WebSocketRobotBridge
    bridge = WebSocketRobotBridge(
        heartbeat_interval=robot_cfg.get("heartbeat_interval", 5),
        disconnect_timeout=robot_cfg.get("disconnect_timeout", 15),
        disconnect_debounce=robot_cfg.get("disconnect_debounce", 3),
        bearer_token=config.get("bearer_token", ""),
    )

    async def _on_robot_connected():
        robot_mod.set_connected(model_name, True)
        await bus.publish("robot.connected", {"model": model_name}, endpoint="robot")
        print(f"[gateway] Robot connected via WebSocket: {model_name}")

    async def _on_robot_disconnected(reason: str):
        robot_mod.set_connected(model_name, False)
        robot_perception.stop_all()
        await bus.publish("robot.camera.stream_stopped", {}, endpoint="robot")
        await bus.publish("robot.audio.monitor_stopped", {}, endpoint="robot")
        await bus.publish("robot.disconnected", {"model": model_name, "reason": reason}, endpoint="robot")
        print(f"[gateway] Robot disconnected via WebSocket: {model_name} ({reason})")

    async def _on_robot_message(msg: dict):
        msg_type = msg.get("type", "")
        if msg_type == "camera_frame":
            robot_perception.push_frame(
                msg.get("source", "head_rgb"),
                msg.get("image_base64", ""),
                msg.get("seq", 0),
                msg.get("ts", time.time()),
            )
            return
        elif msg_type == "audio_chunk":
            source = msg.get("source", "mic_array")
            robot_perception.push_audio(
                source,
                msg.get("audio_base64", ""),
                msg.get("sample_rate", 16000),
                msg.get("channels", 1),
                msg.get("seq", 0),
                msg.get("ts", time.time()),
            )
            await bus.publish("robot.audio_chunk", {
                "source": source,
                "audio_base64": msg.get("audio_base64", ""),
                "sample_rate": msg.get("sample_rate", 16000),
            }, endpoint="robot")
            return
        elif msg_type == "observation":
            embedding = msg.get("embedding", [])
            metadata = msg.get("metadata", {})
            if robot_memory._initialized and embedding:
                robot_memory.add_observation(embedding, metadata)
            return
        elif msg_type == "robot_event":
            await bus.publish(f"robot.event.{msg.get('event', 'unknown')}", msg, endpoint="robot")
        elif msg_type == "robot_voice_input":
            transcript = msg.get("transcript", "")
            if not transcript:
                return
            wake_verified = msg.get("wake_word_detected", False)
            if not wake_verified:
                accepted, transcript = robot_ctrl.check_wake_word(transcript)
                if not accepted:
                    return
            result = await robot_ctrl.handle_voice_turn(transcript)
            if result.get("say"):
                try:
                    wav, tts_ms = await asyncio.to_thread(
                        voice_pipeline.synthesize_robot, result["say"])
                    import base64 as _b64
                    await bridge.send({
                        "type": "robot_speak",
                        "audio_base64": _b64.b64encode(wav).decode(),
                        "sample_rate": 24000,
                        "text": result["say"],
                    })
                except Exception as exc:
                    log.warning("Robot TTS failed, sending text only: %s", exc)
                    await bridge.send({
                        "type": "tts_speak",
                        "text": result["say"],
                        "voice": robot_cfg.get("voice", "am_adam"),
                    })

    bridge.on_connected(_on_robot_connected)
    bridge.on_disconnected(_on_robot_disconnected)
    bridge.on_message(_on_robot_message)

    # Use the same _intercom_bridge variable so existing robot endpoints work
    _intercom_bridge = bridge
    _ws_robot_bridge = bridge
    robot_mod.set_transport(bridge)

    # Register speak callback — synthesize TTS on gateway, send audio to robot
    async def _robot_speak(text: str) -> None:
        wav, _ = await asyncio.to_thread(voice_pipeline.synthesize_robot, text)
        import base64 as _b64
        await bridge.send({
            "type": "robot_speak",
            "audio_base64": _b64.b64encode(wav).decode(),
            "sample_rate": 24000,
            "text": text,
        })
    robot_ctrl.set_speak_callback(_robot_speak)

    await bridge.start()

    print("[gateway] WebSocket robot transport ready (waiting for /ws/robot connection)")


@app.websocket("/ws/robot")
async def ws_robot(ws: WebSocket) -> None:
    """WebSocket endpoint for direct robot/sim-adapter connections."""
    if _ws_robot_bridge is None:
        await ws.close(code=4003, reason="websocket transport not enabled")
        return
    await _ws_robot_bridge.handle_ws(ws)


async def _start_intercom_transport() -> None:
    """Start Intercom process and bridge when robot.enabled + transport=intercom."""
    global _intercom_process, _intercom_bridge

    robot_cfg = config.section("robot")
    channel = robot_cfg.get("intercom_channel", "clawfinger-robot-g1")
    sc_bridge_port = robot_cfg.get("sc_bridge_port", 49222)
    intercom_key = robot_cfg.get("intercom_key", "")
    pear_path = robot_cfg.get("pear_path", "")

    # Read SC-Bridge token
    token_path = _TMP_DIR / ".sc_bridge_token"
    sc_bridge_token = ""
    if token_path.exists():
        sc_bridge_token = token_path.read_text().strip()
    if not sc_bridge_token:
        print("[gateway] WARNING: No SC-Bridge token found at tmp/.sc_bridge_token")
        print("[gateway]   Run bin/intercom-pair.sh first to generate one")

    # Start Intercom Pear process
    from transport.intercom_manager import IntercomProcess
    _intercom_process = IntercomProcess(
        channel=channel,
        sc_bridge_port=sc_bridge_port,
        sc_bridge_token=sc_bridge_token,
        inviter_keys=intercom_key,
        pear_path=pear_path,
        intercom_limits=robot_cfg.get("intercom_limits", {}),
    )
    await _intercom_process.start()

    # Brief wait for SC-Bridge to come up
    await asyncio.sleep(2)

    # Start bridge
    from transport.intercom_bridge import IntercomBridge
    model_name = robot_cfg.get("model", "unitree_g1")
    _intercom_bridge = IntercomBridge(
        channel=channel,
        sc_bridge_url=f"ws://127.0.0.1:{sc_bridge_port}",
        sc_bridge_token=sc_bridge_token,
        heartbeat_interval=robot_cfg.get("heartbeat_interval", 5),
        disconnect_timeout=robot_cfg.get("disconnect_timeout", 15),
        disconnect_debounce=robot_cfg.get("disconnect_debounce", 3),
    )

    async def _on_robot_connected():
        robot_mod.set_connected(model_name, True)
        await bus.publish("robot.connected", {"model": model_name}, endpoint="robot")
        print(f"[gateway] Robot connected: {model_name}")

    async def _on_robot_disconnected(reason: str):
        robot_mod.set_connected(model_name, False)
        # Stop all perception streams/monitors on disconnect
        robot_perception.stop_all()
        await bus.publish("robot.camera.stream_stopped", {}, endpoint="robot")
        await bus.publish("robot.audio.monitor_stopped", {}, endpoint="robot")
        await bus.publish("robot.disconnected", {"model": model_name, "reason": reason}, endpoint="robot")
        print(f"[gateway] Robot disconnected: {model_name} ({reason})")
        # Safety stop on disconnect
        if robot_cfg.get("safety_stop_on_disconnect", True):
            try:
                await _intercom_bridge.send({"type": "robot_command", "command": "safety_stop"})
                await bus.publish("robot.safety_stop", {"model": model_name, "reason": reason}, endpoint="robot")
            except Exception:
                pass

    async def _on_robot_message(msg: dict):
        msg_type = msg.get("type", "")
        if msg_type == "camera_frame":
            robot_perception.push_frame(
                msg.get("source", "head_rgb"),
                msg.get("image_base64", ""),
                msg.get("seq", 0),
                msg.get("ts", time.time()),
            )
            return
        elif msg_type == "audio_chunk":
            source = msg.get("source", "mic_array")
            robot_perception.push_audio(
                source,
                msg.get("audio_base64", ""),
                msg.get("sample_rate", 16000),
                msg.get("channels", 1),
                msg.get("seq", 0),
                msg.get("ts", time.time()),
            )
            # Publish via event bus for control center Web Audio playback
            await bus.publish("robot.audio_chunk", {
                "source": source,
                "audio_base64": msg.get("audio_base64", ""),
                "sample_rate": msg.get("sample_rate", 16000),
            }, endpoint="robot")
            return
        elif msg_type == "observation":
            # Pre-computed embedding from Jetson/sim adapter — store directly
            embedding = msg.get("embedding", [])
            metadata = msg.get("metadata", {})
            if robot_memory._initialized and embedding:
                robot_memory.add_observation(embedding, metadata)
            return
        elif msg_type == "robot_event":
            await bus.publish(f"robot.event.{msg.get('event', 'unknown')}", msg, endpoint="robot")
        elif msg_type == "robot_voice_input":
            transcript = msg.get("transcript", "")
            wake_verified = msg.get("wake_word_detected", False)
            if not transcript:
                return
            # If Jetson didn't verify wake word, gateway checks
            if not wake_verified:
                accepted, transcript = robot_ctrl.check_wake_word(transcript)
                if not accepted:
                    return
            result = await robot_ctrl.handle_voice_turn(transcript)
            # Synthesize speech on gateway, send audio bytes via Intercom
            if result.get("say"):
                try:
                    wav, tts_ms = await asyncio.to_thread(
                        voice_pipeline.synthesize_robot, result["say"])
                    import base64 as _b64
                    await _intercom_bridge.send({
                        "type": "robot_speak",
                        "audio_base64": _b64.b64encode(wav).decode(),
                        "sample_rate": 24000,
                        "text": result["say"],
                    })
                except Exception as exc:
                    log.warning("Robot TTS failed, sending text only: %s", exc)
                    await _intercom_bridge.send({
                        "type": "robot_speak",
                        "text": result["say"],
                    })
            # Send gesture command
            if result.get("gesture"):
                await robot_mod.dispatch_command(
                    robot_cfg.get("model", "unitree_g1"),
                    {"type": result["gesture"], "params": {}},
                )
            # Forward robot_turn.request to takeover agent if active
            if robot_ctrl.get_takeover_agent():
                for agent_ws_conn in agent_interface.list_agent_connections():
                    agent_id = agent_interface.get_agent_id(agent_ws_conn)
                    if agent_id == robot_ctrl.get_takeover_agent():
                        try:
                            await agent_ws_conn.send_json({
                                "type": "robot_turn.request",
                                "transcript": transcript,
                                "request_id": str(__import__("uuid").uuid4()),
                            })
                        except Exception:
                            pass

    _intercom_bridge.on_connected(_on_robot_connected)
    _intercom_bridge.on_disconnected(_on_robot_disconnected)
    _intercom_bridge.on_message(_on_robot_message)

    robot_mod.set_transport(_intercom_bridge)

    # Register speak callback — synthesize TTS on gateway, send audio to robot
    async def _robot_speak(text: str) -> None:
        wav, _ = await asyncio.to_thread(voice_pipeline.synthesize_robot, text)
        import base64 as _b64
        await _intercom_bridge.send({
            "type": "robot_speak",
            "audio_base64": _b64.b64encode(wav).decode(),
            "sample_rate": 24000,
            "text": text,
        })
    robot_ctrl.set_speak_callback(_robot_speak)

    await _intercom_bridge.start()

    await bus.publish("transport.started", {"channel": channel}, endpoint="robot")
    print(f"[gateway] Intercom transport started (channel: {channel})")


@app.on_event("startup")
async def startup() -> None:
    cfg = config.load()
    asr = config.section("asr")
    llm = config.section("llm")
    print(f"[gateway] Starting on {cfg['host']}:{cfg['port']}")
    print(f"[gateway] mlx_audio: {asr['backend']}")
    backend_type = "local/MLX" if not llm.get("base_url") else f"remote/{llm['base_url']}"
    print(f"[gateway] LLM: {llm['model']} ({backend_type})")
    llm_backend.preload()
    asyncio.create_task(_periodic_sweep())
    # Load robot models (registers capabilities) and skills
    robot_mod.load_models()
    robot_skills.load_skills()
    skills_loaded = robot_skills.list_skills()
    if skills_loaded:
        print(f"[gateway] Robot skills loaded: {len(skills_loaded)} ({', '.join(s['name'] for s in skills_loaded)})")
    robot_cfg = config.section("robot")
    # Initialize spatial memory
    memory_cfg = robot_cfg.get("memory", {})
    if memory_cfg.get("enabled", True):
        try:
            db_path = memory_cfg.get("db_path", "data/spatial_memory")
            if not Path(db_path).is_absolute():
                db_path = str(_ROOT / db_path)
            robot_memory.init(
                db_path,
                memory_cfg.get("embedding_model", "ViT-B-32"),
                memory_cfg.get("embedding_device", "mps"),
            )
            print(f"[gateway] Spatial memory: initialized")
        except Exception as exc:
            print(f"[gateway] WARNING: Spatial memory init failed: {exc}")
            # Fall back to memory-only mode (no CLIP)
            try:
                robot_memory.init_memory_only(db_path)
                print(f"[gateway] Spatial memory: initialized (no CLIP)")
            except Exception:
                print(f"[gateway] WARNING: Spatial memory completely failed")
    if robot_cfg.get("enabled"):
        transport = robot_cfg.get("transport", "intercom")
        print(f"[gateway] Robot: {robot_cfg.get('model', 'unitree_g1')} (enabled, transport={transport})")
        if transport == "intercom":
            try:
                await _start_intercom_transport()
            except Exception as exc:
                print(f"[gateway] WARNING: Intercom transport failed to start: {exc}")
        elif transport == "websocket":
            try:
                await _start_ws_transport()
            except Exception as exc:
                print(f"[gateway] WARNING: WebSocket transport failed to start: {exc}")
    else:
        print("[gateway] Robot: disabled")


@app.on_event("shutdown")
async def shutdown() -> None:
    global _intercom_bridge, _intercom_process, _ws_robot_bridge
    if _intercom_bridge:
        await _intercom_bridge.stop()
        _intercom_bridge = None
    if _intercom_process:
        await _intercom_process.stop()
        _intercom_process = None
    _ws_robot_bridge = None
    robot_memory.shutdown()


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
