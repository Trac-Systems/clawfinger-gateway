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

try:
    from mlx_vlm import load as vlm_load
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
    from mlx_vlm.utils import load_config as vlm_load_config
except Exception:
    vlm_load = None
    vlm_generate = None
    vlm_apply_chat_template = None
    vlm_load_config = None

_LOCAL_MODEL: Any | None = None
_LOCAL_TOKENIZER: Any | None = None
_LOCAL_MODEL_NAME: str = ""
_LOCAL_CONTEXT_WINDOW: int = 0  # auto-detected from model.args

# VLM (Vision Language Model) state — separate from text-only LM
_VLM_MODEL: Any | None = None
_VLM_PROCESSOR: Any | None = None
_VLM_CONFIG: Any | None = None
_VLM_MODEL_NAME: str = ""


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


def _ensure_local_vlm(llm: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Load VLM model + processor + config. Returns (model, processor, config)."""
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_CONFIG, _VLM_MODEL_NAME
    model_name = llm.get("model", "")
    if _VLM_MODEL is not None and _VLM_PROCESSOR is not None and _VLM_MODEL_NAME == model_name:
        return _VLM_MODEL, _VLM_PROCESSOR, _VLM_CONFIG
    if vlm_load is None:
        raise RuntimeError("mlx-vlm is not available — install with: pip install mlx-vlm")
    _VLM_MODEL, _VLM_PROCESSOR = vlm_load(model_name)
    _VLM_CONFIG = vlm_load_config(model_name) if vlm_load_config else {}
    _VLM_MODEL_NAME = model_name
    return _VLM_MODEL, _VLM_PROCESSOR, _VLM_CONFIG


def _has_images(messages: list[dict[str, Any]]) -> bool:
    """Check if any message contains image content."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False


def _extract_images(messages: list[dict[str, Any]]) -> list[str]:
    """Extract image URLs/paths from message content arrays."""
    images = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url_data = item.get("image_url", {})
                    url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
                    if url:
                        images.append(url)
    return images


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
        if llm.get("multimodal") and vlm_load is not None:
            _ensure_local_vlm(llm)
            print(f"[gateway] VLM preloaded: {llm['model']} (multimodal)")
        else:
            _ensure_local_llm(llm)
            ctx = f", context_window={_LOCAL_CONTEXT_WINDOW}" if _LOCAL_CONTEXT_WINDOW else ""
            print(f"[gateway] LLM preloaded: {llm['model']}{ctx}")
    except Exception as exc:
        print(f"[gateway] LLM preload failed: {exc}")


def generate(messages: list[dict[str, Any]], raw: bool = False,
             llm_override: dict[str, Any] | None = None) -> tuple[str, float, str]:
    """Generate LLM reply. Returns (reply_text, llm_ms, model_name).

    When raw=True: apply only safe_text() (sanitize control chars), skip
    trim_for_tts() which destroys JSON.  Used by robot controller.
    When raw=False (default): apply trim_for_tts() for phone/TTS pipeline.
    llm_override: optional config dict to use instead of global llm section.
    """
    llm = llm_override or config.section("llm")
    if _is_local(llm):
        # Multimodal model: always use VLM backend (works for text-only too)
        if llm.get("multimodal") and vlm_load is not None:
            return _generate_local_vlm(messages, llm, raw=raw)
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
        result = mlx_generate(model, tokenizer, **kwargs)
    except TypeError:
        # Fallback if mlx_lm version doesn't support extra kwargs
        result = mlx_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=llm.get("max_tokens", 400),
            verbose=False,
        )

    # mlx_lm may return GenerationResult or string
    text = getattr(result, "text", None) or str(result or "")
    if raw:
        text = safe_text(text)
    else:
        text = trim_for_tts(text)
    if not text:
        text = "Got it. Please continue."

    return text, (time.perf_counter() - start) * 1000, f"local/{llm['model']}"


def _generate_local_vlm(messages: list[dict[str, Any]], llm: dict[str, Any], raw: bool = False) -> tuple[str, float, str]:
    """Generate via local VLM (multimodal — text + images)."""
    start = time.perf_counter()
    model, processor, vlm_cfg = _ensure_local_vlm(llm)

    # Extract images from messages
    images = _extract_images(messages)

    # Build prompt via tokenizer directly (supports enable_thinking=False).
    # mlx_vlm's apply_chat_template doesn't properly handle enable_thinking.
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = _flatten_content(m.get("content", ""))
            break

    # Use tokenizer for text prompt with thinking disabled
    tok_messages = [{"role": "user", "content": last_user}]
    # Prepend system prompt if present
    for m in messages:
        if m.get("role") == "system":
            tok_messages.insert(0, {"role": "system", "content": _flatten_content(m.get("content", ""))})
            break

    try:
        formatted_prompt = processor.tokenizer.apply_chat_template(
            tok_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        # Fallback if tokenizer doesn't support enable_thinking
        formatted_prompt = vlm_apply_chat_template(
            processor, vlm_cfg, last_user, num_images=len(images))

    kwargs: dict[str, Any] = {
        "max_tokens": llm.get("max_tokens", 400),
        "verbose": False,
    }
    if llm.get("temperature", 0.2) > 0:
        kwargs["temp"] = llm["temperature"]

    result = vlm_generate(model, processor, formatted_prompt,
                          images if images else None, **kwargs)

    # vlm_generate may return a GenerationResult object or string
    text = getattr(result, "text", None) or str(result or "")
    if raw:
        text = safe_text(text)
    else:
        text = trim_for_tts(text)
    if not text:
        text = "Got it. Please continue."

    return text, (time.perf_counter() - start) * 1000, f"local-vlm/{llm['model']}"


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
