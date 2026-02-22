"""LLM abstraction — local MLX or remote OpenAI-compatible endpoint."""

from __future__ import annotations

import time
from typing import Any

import httpx

import config
from voice_pipeline import safe_text, trim_for_tts

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
except Exception:
    mlx_generate = None
    mlx_load = None

_LOCAL_MODEL: Any | None = None
_LOCAL_TOKENIZER: Any | None = None


def _ensure_local_llm() -> tuple[Any, Any]:
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
        return _LOCAL_MODEL, _LOCAL_TOKENIZER
    if mlx_load is None:
        raise RuntimeError("mlx-lm is not available in this environment")
    cfg = config.load()
    _LOCAL_MODEL, _LOCAL_TOKENIZER = mlx_load(cfg["llm_local_model"])
    return _LOCAL_MODEL, _LOCAL_TOKENIZER


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def preload() -> None:
    """Preload local MLX model at startup."""
    cfg = config.load()
    if cfg["llm_backend"] != "mlx_local":
        return
    try:
        _ensure_local_llm()
        print(f"[gateway] LLM preloaded: {cfg['llm_local_model']}")
    except Exception as exc:
        print(f"[gateway] LLM preload failed: {exc}")


def generate(messages: list[dict[str, str]]) -> tuple[str, float, str]:
    """Generate LLM reply. Returns (reply_text, llm_ms, model_name)."""
    cfg = config.load()
    if cfg["llm_backend"] == "mlx_local":
        return _generate_local(messages, cfg)
    return _generate_remote(messages, cfg)


def _generate_local(messages: list[dict[str, str]], cfg: dict[str, Any]) -> tuple[str, float, str]:
    start = time.perf_counter()
    model, tokenizer = _ensure_local_llm()
    prompt = _apply_chat_template(tokenizer, messages)

    if mlx_generate is None:
        raise RuntimeError("mlx-lm generate is not available")

    try:
        text = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=cfg["llm_max_tokens"],
            temp=cfg["llm_temperature"],
            verbose=False,
        )
    except TypeError:
        text = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=cfg["llm_max_tokens"],
            verbose=False,
        )

    text = trim_for_tts(str(text or ""))
    if not text:
        text = "Got it. Please continue."

    return text, (time.perf_counter() - start) * 1000, f"local/{cfg['llm_local_model']}"


def _generate_remote(messages: list[dict[str, str]], cfg: dict[str, Any]) -> tuple[str, float, str]:
    start = time.perf_counter()
    base_url = cfg["llm_remote_base_url"].rstrip("/")
    if not base_url:
        raise RuntimeError("llm_remote_base_url not configured")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = cfg.get("llm_remote_api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": cfg["llm_remote_model"],
        "messages": messages,
        "max_tokens": cfg["llm_max_tokens"],
        "temperature": cfg["llm_temperature"],
        "stream": False,
    }

    response = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    body = response.json()

    text = _extract_openai_text(body)
    if not text:
        text = "Got it. Please continue."
    text = trim_for_tts(text)

    return text, (time.perf_counter() - start) * 1000, cfg["llm_remote_model"]


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return safe_text(content)
    if isinstance(content, list):
        texts = [str(item.get("text") or item.get("content") or "") for item in content if isinstance(item, dict)]
        return safe_text(" ".join(t for t in texts if t))
    return safe_text(str(choices[0].get("text", "")))


def check_health() -> dict:
    """Check LLM backend health."""
    cfg = config.load()
    if cfg["llm_backend"] == "mlx_local":
        return {
            "backend": "mlx_local",
            "model": cfg["llm_local_model"],
            "loaded": _LOCAL_MODEL is not None,
            "mlx_lm_available": mlx_load is not None,
        }
    base_url = cfg.get("llm_remote_base_url", "")
    return {
        "backend": "openai_remote",
        "base_url": base_url,
        "model": cfg.get("llm_remote_model", ""),
        "configured": bool(base_url),
    }
