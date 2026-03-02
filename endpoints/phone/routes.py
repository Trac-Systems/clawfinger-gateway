"""Phone endpoint routes — FastAPI APIRouter."""

from __future__ import annotations

import asyncio
import base64
import re
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import config
import session_store
import voice_pipeline
from core.turn_processor import process_turn
from endpoints.phone import adb, caller
from event_bus import bus

router = APIRouter()

_CALL_COUNT = 0
_ERROR_COUNT = 0


def _check_bearer(request: Request) -> None:
    token = config.get("bearer_token", "")
    if not token:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Turn endpoint
# ---------------------------------------------------------------------------

@router.post("/api/turn")
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

    # Reject turns for ended sessions
    if sid and session_store.is_ended(sid):
        return JSONResponse({"ok": False, "session_id": sid, "ended": True, "detail": "session ended"})

    sid = session_store.get_or_create(sid)

    if reset_session.lower() == "true":
        session_store.reset(sid)
        sid = session_store.get_or_create(sid)

    phone_cfg = config.section("phone")

    # --- Caller info ---
    caller_number_clean = voice_pipeline.safe_text(caller_number)
    call_direction_clean = voice_pipeline.safe_text(call_direction)
    if caller_number_clean:
        session_store.set_caller_info(sid, caller_number_clean, call_direction_clean)

    # --- Caller history persistence ---
    if reset_session.lower() == "true" and caller_number_clean:
        caller.restore_history(sid, caller_number_clean, phone_cfg)

    # --- Caller filtering ---
    normalized = caller.normalize_number(caller_number_clean)
    rejection = caller.check_caller_access(normalized, phone_cfg)
    if rejection is not None:
        reason = rejection["reason"]
        await bus.publish("turn.caller_rejected", {"number": normalized, "reason": reason}, session_id=sid)
        reject_text = phone_cfg.get("auth_reject_message", "I'm sorry, I can't help you right now. Goodbye.")
        audio_bytes, _ = await asyncio.to_thread(voice_pipeline.synthesize, reject_text)
        return JSONResponse({
            "ok": False, "rejected": True, "reason": reason,
            "session_id": sid, "reply": reject_text,
            "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        })

    # Touch session activity + sweep stale sessions
    session_store.touch(sid)
    stale = session_store.sweep_stale()
    for stale_sid in stale:
        await bus.publish("session.ended", {"session_id": stale_sid, "reason": "stale"}, session_id=stale_sid)

    _CALL_COUNT += 1

    try:
        result = await process_turn(
            sid,
            audio_data=await audio.read(),
            audio_filename=audio.filename or "turn.wav",
            transcript_hint=voice_pipeline.safe_text(transcript_hint),
            skip_asr=skip_asr.strip().lower() == "true",
            forced_reply=voice_pipeline.safe_text(forced_reply),
            pre_llm_hook=caller.make_passphrase_hook(phone_cfg),
        )
    except HTTPException:
        raise
    except Exception as exc:
        _ERROR_COUNT += 1
        await bus.publish("turn.error", {"error": str(exc)}, session_id=sid)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not result.ok:
        return JSONResponse({"ok": False, "session_id": sid, **result.extra})

    response = {
        "ok": True,
        "session_id": sid,
        "transcript": result.transcript,
        "reply": result.reply,
        "audio_base64": result.audio_base64,
        "metrics": result.metrics,
    }
    response.update(result.extra)
    return JSONResponse(response)


# ---------------------------------------------------------------------------
# Dial / Hangup
# ---------------------------------------------------------------------------

@router.post("/api/call/dial")
async def call_dial(request: Request) -> JSONResponse:
    body = await request.json()
    number = str(body.get("number", "")).strip()
    result = await adb.do_dial(number)
    if result["ok"]:
        await bus.publish("call.dial", {"number": number})
    return JSONResponse(result)


@router.post("/api/call/hangup")
async def call_hangup(request: Request) -> JSONResponse:
    body = await request.json() if await request.body() else {}
    session_id = str(body.get("session_id", "")).strip()
    result = await adb.do_hangup()
    if result["ok"]:
        sid = session_id or adb.single_active_session()
        if sid:
            session_store.end_session(sid)
            await bus.publish("session.ended", {"session_id": sid}, session_id=sid)
            result["session_id"] = sid
        await bus.publish("call.hangup", result)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Inject
# ---------------------------------------------------------------------------

@router.post("/api/call/inject")
async def call_inject(request: Request) -> JSONResponse:
    """Inject a TTS message into the active call."""
    body = await request.json()
    text = voice_pipeline.safe_text(str(body.get("text", "")))
    session_id = str(body.get("session_id", ""))
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if not session_id:
        session_id = session_store.most_recent_active_session() or ""
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


# ---------------------------------------------------------------------------
# Call config
# ---------------------------------------------------------------------------

_CALL_CONFIG_KEYS = {
    "auto_answer", "auto_answer_delay_ms",
    "caller_allowlist", "caller_blocklist", "unknown_callers_allowed",
    "greeting_incoming", "greeting_outgoing", "greeting_owner",
    "max_duration_sec", "max_duration_message",
    "auth_passphrase", "auth_reject_message", "auth_max_attempts",
    "keep_history",
}

_CALL_SECURITY_KEYS = {
    "auth_passphrase", "auth_reject_message", "auth_max_attempts",
    "caller_allowlist", "caller_blocklist", "unknown_callers_allowed",
}

AGENT_ALLOWED_CALL_KEYS = (_CALL_CONFIG_KEYS - _CALL_SECURITY_KEYS) | {"tts.voice", "tts.speed"}

# Legacy flat body keys -> phone section keys
_CALL_BODY_REMAP = {
    "call_auto_answer": "auto_answer",
    "call_auto_answer_delay_ms": "auto_answer_delay_ms",
}


def call_config_response() -> dict:
    """Build call config response dict (used by REST + agent WS)."""
    phone = config.section("phone")
    owner = phone.get("greeting_owner", "the owner")
    greeting_in = phone.get("greeting_incoming", "")
    greeting_out = phone.get("greeting_outgoing", "")
    return {
        "auto_answer": phone.get("auto_answer", True),
        "auto_answer_delay_ms": phone.get("auto_answer_delay_ms", 500),
        "caller_allowlist": phone.get("caller_allowlist", []),
        "caller_blocklist": phone.get("caller_blocklist", []),
        "unknown_callers_allowed": phone.get("unknown_callers_allowed", True),
        "greeting_incoming": greeting_in.replace("{owner}", owner),
        "greeting_outgoing": greeting_out.replace("{owner}", owner),
        "greeting_incoming_template": greeting_in,
        "greeting_outgoing_template": greeting_out,
        "greeting_owner": owner,
        "max_duration_sec": phone.get("max_duration_sec", 300),
        "max_duration_message": phone.get("max_duration_message", ""),
        "auth_required": bool(phone.get("auth_passphrase", "")),
        "auth_reject_message": phone.get("auth_reject_message", ""),
        "auth_max_attempts": phone.get("auth_max_attempts", 3),
        "keep_history": phone.get("keep_history", False),
    }


@router.get("/api/config/call")
async def get_call_config() -> JSONResponse:
    return JSONResponse(call_config_response())


@router.post("/api/config/call")
async def update_call_config(request: Request) -> JSONResponse:
    body = await request.json()
    for key, value in body.items():
        cfg_key = _CALL_BODY_REMAP.get(key, key)
        if cfg_key in _CALL_CONFIG_KEYS:
            config.set(f"phone.{cfg_key}", value)
    config.save()
    await bus.publish("config.call_updated", call_config_response())
    return JSONResponse({"ok": True, **call_config_response()})


# ---------------------------------------------------------------------------
# Caller history
# ---------------------------------------------------------------------------

@router.get("/api/caller-history")
async def list_caller_history() -> JSONResponse:
    return JSONResponse(session_store.list_caller_histories())


@router.delete("/api/caller-history/{number}")
async def delete_caller_history(number: str) -> JSONResponse:
    ok = session_store.delete_caller_history(number)
    return JSONResponse({"ok": ok, "number": number})
