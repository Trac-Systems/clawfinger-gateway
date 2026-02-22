"""Configuration loader — config.json + env var overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CFG_PATH = Path(__file__).resolve().parent / "config.json"
_LOADED: dict[str, Any] = {}


def _cast(value: str, reference: Any) -> Any:
    if isinstance(reference, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(reference, int):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def load() -> dict[str, Any]:
    global _LOADED
    if _LOADED:
        return _LOADED

    with _CFG_PATH.open() as f:
        cfg = json.load(f)

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
