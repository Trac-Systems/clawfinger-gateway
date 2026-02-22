"""Configuration loader — config.json + env var overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CFG_PATH = Path(__file__).resolve().parent / "config.json"
_LOADED: dict[str, Any] = {}

# Maps old split field names → new unified names
_MIGRATION_MAP = {
    "llm_local_model": "llm_model",
    "llm_remote_model": "llm_model",
    "llm_remote_base_url": "llm_base_url",
    "llm_remote_api_key": "llm_api_key",
}


def _cast(value: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, list):
        return json.loads(value)  # Env var must be JSON array string
    return value


def _migrate(cfg: dict[str, Any]) -> None:
    """Migrate old split LLM fields to unified names."""
    old_backend = cfg.pop("llm_backend", None)

    # Pop all old fields first
    local_model = cfg.pop("llm_local_model", "")
    remote_model = cfg.pop("llm_remote_model", "")
    remote_base = cfg.pop("llm_remote_base_url", "")
    remote_key = cfg.pop("llm_remote_api_key", "")

    if old_backend is not None and not cfg.get("llm_model"):
        # Choose model based on which backend was configured
        if old_backend == "mlx_local":
            cfg.setdefault("llm_model", local_model)
            cfg.setdefault("llm_base_url", "")
        else:
            cfg.setdefault("llm_model", remote_model or local_model)
            if remote_base:
                cfg.setdefault("llm_base_url", remote_base)
            if remote_key:
                cfg.setdefault("llm_api_key", remote_key)

    # Ensure new fields have defaults
    cfg.setdefault("llm_model", "")
    cfg.setdefault("llm_base_url", "")
    cfg.setdefault("llm_api_key", "")
    cfg.setdefault("llm_top_p", 1.0)
    cfg.setdefault("llm_top_k", 0)
    cfg.setdefault("llm_repeat_penalty", 1.0)
    cfg.setdefault("llm_stop", [])


def load() -> dict[str, Any]:
    global _LOADED
    if _LOADED:
        return _LOADED

    with _CFG_PATH.open() as f:
        cfg = json.load(f)

    # Migrate old field names
    _migrate(cfg)

    # Env vars override: GATEWAY_<UPPER_KEY> e.g. GATEWAY_PORT=9000
    for key, default in list(cfg.items()):
        env = os.environ.get(f"GATEWAY_{key.upper()}")
        if env is not None:
            cfg[key] = _cast(env, default)

    _LOADED = cfg
    return _LOADED


def reload() -> dict[str, Any]:
    global _LOADED
    _LOADED = {}
    return load()


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)
