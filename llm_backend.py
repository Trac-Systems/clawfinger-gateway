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
_LOCAL_MODEL_NAME: str = ""
_LOCAL_CONTEXT_WINDOW: int = 0  # auto-detected from model.args


def _is_local(cfg: dict[str, Any]) -> bool:
    return not cfg.get("llm_base_url")


def _ensure_local_llm(cfg: dict[str, Any]) -> tuple[Any, Any]:
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_MODEL_NAME, _LOCAL_CONTEXT_WINDOW
    model_name = cfg.get("llm_model", "")
    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None and _LOCAL_MODEL_NAME == model_name:
        return _LOCAL_MODEL, _LOCAL_TOKENIZER
    if mlx_load is None:
        raise RuntimeError("mlx-lm is not available in this environment")
    _LOCAL_MODEL, _LOCAL_TOKENIZER = mlx_load(model_name)
    _LOCAL_MODEL_NAME = model_name
    # Auto-detect context window from model args
    _LOCAL_CONTEXT_WINDOW = getattr(getattr(_LOCAL_MODEL, "args", None), "max_position_embeddings", 0)
    return _LOCAL_MODEL, _LOCAL_TOKENIZER


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = [f"{m['role']}: {m['content']}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def get_context_window() -> int:
    """Return effective context window size in tokens.

    For local MLX models: auto-detected from model.args.max_position_embeddings.
    For remote models: returns 0 (unknown — user must set llm_context_tokens manually).
    """
    return _LOCAL_CONTEXT_WINDOW


def preload() -> None:
    """Preload local MLX model at startup."""
    cfg = config.load()
    if not _is_local(cfg):
        return
    try:
        _ensure_local_llm(cfg)
        ctx = f", context_window={_LOCAL_CONTEXT_WINDOW}" if _LOCAL_CONTEXT_WINDOW else ""
        print(f"[gateway] LLM preloaded: {cfg['llm_model']}{ctx}")
    except Exception as exc:
        print(f"[gateway] LLM preload failed: {exc}")


def generate(messages: list[dict[str, str]]) -> tuple[str, float, str]:
    """Generate LLM reply. Returns (reply_text, llm_ms, model_name)."""
    cfg = config.load()
    if _is_local(cfg):
        return _generate_local(messages, cfg)
    return _generate_remote(messages, cfg)


def _generate_local(messages: list[dict[str, str]], cfg: dict[str, Any]) -> tuple[str, float, str]:
    start = time.perf_counter()
    model, tokenizer = _ensure_local_llm(cfg)
    prompt = _apply_chat_template(tokenizer, messages)

    if mlx_generate is None:
        raise RuntimeError("mlx-lm generate is not available")

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": cfg.get("llm_max_tokens", 400),
        "temp": cfg.get("llm_temperature", 0.2),
        "verbose": False,
    }
    if cfg.get("llm_top_p_enabled", True) and cfg.get("llm_top_p", 1.0) < 1.0:
        kwargs["top_p"] = cfg["llm_top_p"]
    if cfg.get("llm_top_k_enabled", True) and cfg.get("llm_top_k", 0) > 0:
        kwargs["top_k"] = cfg["llm_top_k"]
    if cfg.get("llm_repeat_penalty", 1.0) != 1.0:
        kwargs["repetition_penalty"] = cfg["llm_repeat_penalty"]

    try:
        text = mlx_generate(model, tokenizer, **kwargs)
    except TypeError:
        # Fallback if mlx_lm version doesn't support extra kwargs
        text = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=cfg.get("llm_max_tokens", 400),
            verbose=False,
        )

    text = trim_for_tts(str(text or ""))
    if not text:
        text = "Got it. Please continue."

    return text, (time.perf_counter() - start) * 1000, f"local/{cfg['llm_model']}"


def _generate_remote(messages: list[dict[str, str]], cfg: dict[str, Any]) -> tuple[str, float, str]:
    start = time.perf_counter()
    base_url = cfg["llm_base_url"].rstrip("/")
    if not base_url:
        raise RuntimeError("llm_base_url not configured")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = cfg.get("llm_api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": cfg["llm_model"],
        "messages": messages,
        "max_tokens": cfg.get("llm_max_tokens", 400),
        "temperature": cfg.get("llm_temperature", 0.2),
        "stream": False,
    }
    if cfg.get("llm_top_p_enabled", True) and cfg.get("llm_top_p", 1.0) < 1.0:
        payload["top_p"] = cfg["llm_top_p"]
    if cfg.get("llm_repeat_penalty", 1.0) != 1.0:
        payload["frequency_penalty"] = cfg["llm_repeat_penalty"] - 1.0
    stop = cfg.get("llm_stop", [])
    if stop:
        payload["stop"] = stop

    response = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    body = response.json()

    text = _extract_openai_text(body)
    if not text:
        text = "Got it. Please continue."
    text = trim_for_tts(text)

    return text, (time.perf_counter() - start) * 1000, cfg["llm_model"]


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
    if _is_local(cfg):
        return {
            "backend": "mlx_local",
            "model": cfg["llm_model"],
            "loaded": _LOCAL_MODEL is not None,
            "mlx_lm_available": mlx_load is not None,
        }
    return {
        "backend": "openai_remote",
        "base_url": cfg.get("llm_base_url", ""),
        "model": cfg.get("llm_model", ""),
        "configured": bool(cfg.get("llm_base_url")),
    }
