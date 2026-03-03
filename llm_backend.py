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


def _is_local(llm: dict[str, Any]) -> bool:
    return not llm.get("base_url")


def _ensure_local_llm(llm: dict[str, Any]) -> tuple[Any, Any]:
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_MODEL_NAME, _LOCAL_CONTEXT_WINDOW
    model_name = llm.get("model", "")
    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None and _LOCAL_MODEL_NAME == model_name:
        return _LOCAL_MODEL, _LOCAL_TOKENIZER
    if mlx_load is None:
        raise RuntimeError("mlx-lm is not available in this environment")
    _LOCAL_MODEL, _LOCAL_TOKENIZER = mlx_load(model_name)
    _LOCAL_MODEL_NAME = model_name
    # Auto-detect context window from model args
    _LOCAL_CONTEXT_WINDOW = getattr(getattr(_LOCAL_MODEL, "args", None), "max_position_embeddings", 0)
    return _LOCAL_MODEL, _LOCAL_TOKENIZER


def _flatten_content(content: Any) -> str:
    """Flatten multimodal content arrays to text for text-only backends."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(p for p in parts if p)
    return str(content)


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    # Flatten multimodal content for local MLX (text-only)
    flat_messages = []
    for m in messages:
        flat = {**m, "content": _flatten_content(m.get("content", ""))}
        flat_messages.append(flat)
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        # Qwen3 models: disable thinking mode for low-latency voice use
        try:
            return tokenizer.apply_chat_template(flat_messages, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(flat_messages, **kwargs)
    lines = [f"{m['role']}: {m['content']}" for m in flat_messages]
    lines.append("assistant:")
    return "\n".join(lines)


def get_context_window() -> int:
    """Return effective context window size in tokens.

    For local MLX models: auto-detected from model.args.max_position_embeddings.
    For remote models: returns 0 (unknown — user must set llm.context_tokens manually).
    """
    return _LOCAL_CONTEXT_WINDOW


def preload() -> None:
    """Preload local MLX model at startup."""
    llm = config.section("llm")
    if not _is_local(llm):
        return
    try:
        _ensure_local_llm(llm)
        ctx = f", context_window={_LOCAL_CONTEXT_WINDOW}" if _LOCAL_CONTEXT_WINDOW else ""
        print(f"[gateway] LLM preloaded: {llm['model']}{ctx}")
    except Exception as exc:
        print(f"[gateway] LLM preload failed: {exc}")


def generate(messages: list[dict[str, Any]], raw: bool = False) -> tuple[str, float, str]:
    """Generate LLM reply. Returns (reply_text, llm_ms, model_name).

    When raw=True: apply only safe_text() (sanitize control chars), skip
    trim_for_tts() which destroys JSON.  Used by robot controller.
    When raw=False (default): apply trim_for_tts() for phone/TTS pipeline.
    """
    llm = config.section("llm")
    if _is_local(llm):
        return _generate_local(messages, llm, raw=raw)
    return _generate_remote(messages, llm, raw=raw)


def _generate_local(messages: list[dict[str, Any]], llm: dict[str, Any], raw: bool = False) -> tuple[str, float, str]:
    start = time.perf_counter()
    model, tokenizer = _ensure_local_llm(llm)
    prompt = _apply_chat_template(tokenizer, messages)

    if mlx_generate is None:
        raise RuntimeError("mlx-lm generate is not available")

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": llm.get("max_tokens", 400),
        "temp": llm.get("temperature", 0.2),
        "verbose": False,
    }
    if llm.get("top_p_enabled", True) and llm.get("top_p", 1.0) < 1.0:
        kwargs["top_p"] = llm["top_p"]
    if llm.get("top_k_enabled", True) and llm.get("top_k", 0) > 0:
        kwargs["top_k"] = llm["top_k"]
    if llm.get("repeat_penalty", 1.0) != 1.0:
        kwargs["repetition_penalty"] = llm["repeat_penalty"]

    try:
        text = mlx_generate(model, tokenizer, **kwargs)
    except TypeError:
        # Fallback if mlx_lm version doesn't support extra kwargs
        text = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=llm.get("max_tokens", 400),
            verbose=False,
        )

    text = str(text or "")
    if raw:
        text = safe_text(text)
    else:
        text = trim_for_tts(text)
    if not text:
        text = "Got it. Please continue."

    return text, (time.perf_counter() - start) * 1000, f"local/{llm['model']}"


def _generate_remote(messages: list[dict[str, Any]], llm: dict[str, Any], raw: bool = False) -> tuple[str, float, str]:
    start = time.perf_counter()
    base_url = llm["base_url"].rstrip("/")
    if not base_url:
        raise RuntimeError("llm.base_url not configured")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = llm.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Remote OpenAI-compatible APIs support content arrays natively
    payload: dict[str, Any] = {
        "model": llm["model"],
        "messages": messages,
        "max_tokens": llm.get("max_tokens", 400),
        "temperature": llm.get("temperature", 0.2),
        "stream": False,
    }
    if llm.get("top_p_enabled", True) and llm.get("top_p", 1.0) < 1.0:
        payload["top_p"] = llm["top_p"]
    if llm.get("repeat_penalty", 1.0) != 1.0:
        payload["frequency_penalty"] = llm["repeat_penalty"] - 1.0
    stop = llm.get("stop", [])
    if stop:
        payload["stop"] = stop

    response = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    body = response.json()

    text = _extract_openai_text(body)
    if not text:
        text = "Got it. Please continue."
    if raw:
        text = safe_text(text)
    else:
        text = trim_for_tts(text)

    return text, (time.perf_counter() - start) * 1000, llm["model"]


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
    llm = config.section("llm")
    if _is_local(llm):
        return {
            "backend": "mlx_local",
            "model": llm["model"],
            "loaded": _LOCAL_MODEL is not None,
            "mlx_lm_available": mlx_load is not None,
        }
    return {
        "backend": "openai_remote",
        "base_url": llm.get("base_url", ""),
        "model": llm.get("model", ""),
        "configured": bool(llm.get("base_url")),
    }
