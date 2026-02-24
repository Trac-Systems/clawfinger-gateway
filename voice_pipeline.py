"""Voice pipeline — ASR and TTS via mlx_audio HTTP server."""

from __future__ import annotations

import mimetypes
import re
import time
from pathlib import Path

import httpx

import config

_SPOKEN_ALLOWED_RE = re.compile(r"[^\w\s\.,!?;:'\"()\-\n]", re.UNICODE)


def _safe_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2026", "...")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trim_for_tts(text: str) -> str:
    cleaned = _safe_text(re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL))
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = cleaned.replace("*", " ").replace("`", " ").replace("_", " ").replace("#", " ")
    cleaned = _SPOKEN_ALLOWED_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


safe_text = _safe_text
trim_for_tts = _trim_for_tts


def transcribe(file_path: Path) -> tuple[str, float]:
    """Run ASR on audio file via mlx_audio. Returns (transcript, asr_ms)."""
    cfg = config.load()
    base = cfg["mlx_audio_base"].rstrip("/")
    start = time.perf_counter()

    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, content_type)}
        data = {"model": cfg["stt_model"], "language": cfg["stt_language"]}
        response = httpx.post(f"{base}/v1/audio/transcriptions", files=files, data=data, timeout=180)

    response.raise_for_status()
    payload = response.json()

    transcript = ""
    if isinstance(payload, dict):
        transcript = payload.get("text") or payload.get("transcript") or ""
        if not transcript and payload.get("segments"):
            transcript = " ".join(
                str(seg.get("text", "")).strip()
                for seg in payload.get("segments", [])
                if isinstance(seg, dict)
            ).strip()
    elif isinstance(payload, str):
        transcript = payload

    return _safe_text(transcript or ""), (time.perf_counter() - start) * 1000


def _synthesize_piper(text: str) -> bytes:
    """Synthesize text via Piper HTTP server. Returns raw WAV bytes."""
    cfg = config.load()
    piper_base = cfg.get("piper_base", "http://127.0.0.1:5123")
    payload = {
        "text": text,
        "length_scale": cfg.get("piper_length_scale", 1.0),
        "noise_scale": cfg.get("piper_noise_scale", 0.667),
        "noise_w": cfg.get("piper_noise_w", 0.8),
        "sentence_silence": cfg.get("piper_sentence_silence", 0.2),
    }
    speaker = cfg.get("piper_speaker", 0)
    if speaker:
        payload["speaker_id"] = speaker
    response = httpx.post(piper_base, json=payload, timeout=30)
    response.raise_for_status()
    return response.content


def synthesize(text: str) -> tuple[bytes, float]:
    """Run TTS on text. Routes to Piper (German) or Kokoro (English) based on tts_lang."""
    cfg = config.load()
    start = time.perf_counter()

    if cfg.get("tts_lang", "en") == "de":
        wav = _synthesize_piper(_trim_for_tts(text))
    else:
        base = cfg["mlx_audio_base"].rstrip("/")
        payload = {
            "model": cfg["tts_model"],
            "input": _trim_for_tts(text),
            "voice": cfg["tts_voice"],
            "speed": cfg["tts_speed"],
            "response_format": "wav",
        }
        resp = httpx.post(f"{base}/v1/audio/speech", json=payload, timeout=180)
        resp.raise_for_status()
        wav = resp.content

    return wav, (time.perf_counter() - start) * 1000


def check_mlx_audio() -> dict:
    """Check mlx_audio + Piper server health. Returns model list or error."""
    cfg = config.load()
    result = {}
    # mlx_audio
    base = cfg["mlx_audio_base"].rstrip("/")
    try:
        response = httpx.get(f"{base}/v1/models", timeout=8)
        response.raise_for_status()
        models = (response.json() or {}).get("data", [])
        result["mlx_audio"] = {"ok": True, "models": [m.get("id") for m in models]}
    except Exception as exc:
        result["mlx_audio"] = {"ok": False, "error": str(exc)}
    # Piper
    piper_base = cfg.get("piper_base", "http://127.0.0.1:5123")
    try:
        resp = httpx.post(piper_base, json={"text": "test"}, timeout=8)
        result["piper"] = {"ok": resp.status_code == 200}
    except Exception as exc:
        result["piper"] = {"ok": False, "error": str(exc)}
    result["ok"] = result["mlx_audio"].get("ok", False)
    return result
