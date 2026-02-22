"""Voice pipeline — ASR and TTS via mlx_audio HTTP server."""

from __future__ import annotations

import mimetypes
import re
import time
from pathlib import Path

import httpx

import config

_SPOKEN_ALLOWED_RE = re.compile(r"[^A-Za-z0-9\s\.,!?;:'\"()\-\n]")


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


def synthesize(text: str) -> tuple[bytes, float]:
    """Run TTS on text via mlx_audio. Returns (wav_bytes, tts_ms)."""
    cfg = config.load()
    base = cfg["mlx_audio_base"].rstrip("/")
    start = time.perf_counter()

    payload = {
        "model": cfg["tts_model"],
        "input": _trim_for_tts(text),
        "voice": cfg["tts_voice"],
        "speed": cfg["tts_speed"],
        "response_format": "wav",
    }
    response = httpx.post(f"{base}/v1/audio/speech", json=payload, timeout=180)
    response.raise_for_status()

    return response.content, (time.perf_counter() - start) * 1000


def check_mlx_audio() -> dict:
    """Check mlx_audio server health. Returns model list or error."""
    cfg = config.load()
    base = cfg["mlx_audio_base"].rstrip("/")
    try:
        response = httpx.get(f"{base}/v1/models", timeout=8)
        response.raise_for_status()
        models = (response.json() or {}).get("data", [])
        return {"ok": True, "models": [m.get("id") for m in models]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
