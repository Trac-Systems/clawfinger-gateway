"""Local Voice Gateway — FastAPI application."""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import httpx

import agent_interface
import config
import instruction_store
import llm_backend
import session_store
import voice_pipeline
from event_bus import bus
from endpoints.phone import adb as phone_adb
from endpoints.phone import routes as phone_routes_mod
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

        mlx_base = asr["backend"].rstrip("/")
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
    "intercom_key", "safety_stop_on_disconnect", "enable_low_level",
}


def _robot_config_response() -> dict:
    """Build robot config response dict."""
    from endpoints import robot as robot_mod
    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")
    model_info = robot_mod.get_model(model_name)
    return {
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
        "model_config": robot_cfg.get(model_name, {}),
    }


@app.get("/api/config/robot")
async def get_robot_config() -> JSONResponse:
    return JSONResponse(_robot_config_response())


@app.post("/api/config/robot")
async def update_robot_config(request: Request) -> JSONResponse:
    body = await request.json()
    robot_cfg = config.section("robot")
    model_name = robot_cfg.get("model", "unitree_g1")

    for key, value in body.items():
        if key in _ROBOT_SECURITY_KEYS:
            continue  # skip security keys
        if key.startswith(f"{model_name}."):
            # Model-specific key
            config.set(f"robot.{model_name}.{key[len(model_name)+1:]}", value)
        elif key in ("enabled", "model", "transport", "heartbeat_interval",
                     "disconnect_timeout", "disconnect_debounce",
                     "safety_stop_on_disconnect", "offline_mode",
                     "voice", "voice_lang"):
            config.set(f"robot.{key}", value)
    config.save()
    await bus.publish("config.robot_updated", _robot_config_response())
    return JSONResponse({"ok": True, **_robot_config_response()})


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
    # Load robot models (registers capabilities)
    from endpoints import robot as robot_mod
    robot_mod.load_models()
    robot_cfg = config.section("robot")
    if robot_cfg.get("enabled"):
        print(f"[gateway] Robot: {robot_cfg.get('model', 'unitree_g1')} (enabled)")
    else:
        print("[gateway] Robot: disabled")


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
