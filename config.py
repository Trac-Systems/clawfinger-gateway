"""Configuration loader — nested config.json + env var overrides + auto-migration."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

_CFG_PATH = Path(__file__).resolve().parent / "config.json"
_LOADED: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Flat → nested migration map (old key → dotted nested path)
# ---------------------------------------------------------------------------

_FLAT_TO_NESTED: dict[str, str] = {
    "mlx_audio_base": "asr.backend",
    "stt_model": "asr.model",
    "stt_language": "asr.language",
    "tts_model": "tts.model",
    "tts_voice": "tts.voice",
    "tts_speed": "tts.speed",
    "tts_lang": "tts.lang",
    "piper_base": "tts.piper.base",
    "piper_voice": "tts.piper.voice",
    "piper_speaker": "tts.piper.speaker",
    "piper_length_scale": "tts.piper.length_scale",
    "piper_noise_scale": "tts.piper.noise_scale",
    "piper_noise_w": "tts.piper.noise_w",
    "piper_sentence_silence": "tts.piper.sentence_silence",
    "mlx_audio_tts_base": "tts.mlx_audio_base",
    "tts_ref_audio": "tts.ref_audio",
    "tts_ref_text": "tts.ref_text",
    "llm_model": "llm.model",
    "llm_base_url": "llm.base_url",
    "llm_api_key": "llm.api_key",
    "llm_max_tokens": "llm.max_tokens",
    "llm_temperature": "llm.temperature",
    "llm_top_p": "llm.top_p",
    "llm_top_p_enabled": "llm.top_p_enabled",
    "llm_top_k": "llm.top_k",
    "llm_top_k_enabled": "llm.top_k_enabled",
    "llm_repeat_penalty": "llm.repeat_penalty",
    "llm_stop": "llm.stop",
    "llm_context_tokens": "llm.context_tokens",
    "llm_system_prompt": "llm.system_prompt",
    "max_history_turns": "llm.max_history_turns",
    "call_auto_answer": "phone.auto_answer",
    "call_auto_answer_delay_ms": "phone.auto_answer_delay_ms",
    "caller_allowlist": "phone.caller_allowlist",
    "caller_blocklist": "phone.caller_blocklist",
    "unknown_callers_allowed": "phone.unknown_callers_allowed",
    "greeting_incoming": "phone.greeting_incoming",
    "greeting_outgoing": "phone.greeting_outgoing",
    "greeting_owner": "phone.greeting_owner",
    "max_duration_sec": "phone.max_duration_sec",
    "max_duration_message": "phone.max_duration_message",
    "auth_passphrase": "phone.auth_passphrase",
    "auth_reject_message": "phone.auth_reject_message",
    "auth_max_attempts": "phone.auth_max_attempts",
    "keep_history": "phone.keep_history",
    "adb_path": "phone.adb_path",
    "agent_takeover_timeout": "agent.takeover_timeout",
}


_ADB_SEARCH_PATHS = [
    "/opt/homebrew/bin/adb",
    "/opt/homebrew/Caskroom/android-platform-tools/36.0.2/platform-tools/adb",
    "/usr/local/bin/adb",
    "/usr/bin/adb",
]


def _find_adb() -> str:
    """Auto-discover ADB binary: PATH first, then known locations."""
    found = shutil.which("adb")
    if found:
        return found
    for path in _ADB_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return "adb"  # fallback to bare name


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


# ---------------------------------------------------------------------------
# Dotted path helpers
# ---------------------------------------------------------------------------

def _get_nested(d: dict, dotted: str, default: Any = None) -> Any:
    """Navigate nested dict via dotted path: 'llm.model' → d['llm']['model']."""
    keys = dotted.split(".")
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, _SENTINEL)
        if d is _SENTINEL:
            return default
    return d

_SENTINEL = object()


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    """Set value in nested dict via dotted path, creating intermediate dicts."""
    keys = dotted.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Migration: flat → nested
# ---------------------------------------------------------------------------

def _is_flat(cfg: dict[str, Any]) -> bool:
    """Detect if config is flat format (has known flat keys like llm_model)."""
    return any(k in cfg for k in ("llm_model", "stt_model", "tts_voice", "call_auto_answer", "mlx_audio_base"))


def _migrate_flat_to_nested(cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert flat config dict to nested structure."""
    # Handle old split LLM fields first
    old_backend = cfg.pop("llm_backend", None)
    local_model = cfg.pop("llm_local_model", "")
    remote_model = cfg.pop("llm_remote_model", "")
    remote_base = cfg.pop("llm_remote_base_url", "")
    remote_key = cfg.pop("llm_remote_api_key", "")

    if old_backend is not None and not cfg.get("llm_model"):
        if old_backend == "mlx_local":
            cfg.setdefault("llm_model", local_model)
        else:
            cfg.setdefault("llm_model", remote_model or local_model)
            if remote_base:
                cfg.setdefault("llm_base_url", remote_base)
            if remote_key:
                cfg.setdefault("llm_api_key", remote_key)

    # Build new nested dict — start with root-level keys
    nested: dict[str, Any] = {}
    for key in ("host", "port", "bearer_token", "session_ttl"):
        if key in cfg:
            nested[key] = cfg[key]

    # Map flat keys to nested paths
    for flat_key, dotted_path in _FLAT_TO_NESTED.items():
        if flat_key in cfg:
            _set_nested(nested, dotted_path, cfg[flat_key])

    return nested


def _ensure_defaults(cfg: dict[str, Any]) -> None:
    """Apply defaults for all sections, creating nested structure as needed."""
    cfg.setdefault("host", "127.0.0.1")
    cfg.setdefault("port", 8996)
    cfg.setdefault("bearer_token", "")
    cfg.setdefault("timezone", "Europe/Berlin")

    # ASR defaults
    asr = cfg.setdefault("asr", {})
    asr.setdefault("backend", "http://127.0.0.1:8765")
    asr.setdefault("model", "mlx-community/parakeet-tdt-0.6b-v3")
    asr.setdefault("language", "en")

    # TTS defaults
    tts = cfg.setdefault("tts", {})
    tts.setdefault("model", "mlx-community/Kokoro-82M-bf16")
    tts.setdefault("voice", "am_adam")
    tts.setdefault("speed", 1.2)
    tts.setdefault("lang", "en")
    tts.setdefault("mlx_audio_base", "")
    tts.setdefault("ref_audio", "")
    tts.setdefault("ref_text", "")
    piper = tts.setdefault("piper", {})
    piper.setdefault("base", "http://127.0.0.1:5123")
    piper.setdefault("voice", "thorsten-high")
    piper.setdefault("speaker", 0)
    piper.setdefault("length_scale", 1.0)
    piper.setdefault("noise_scale", 0.667)
    piper.setdefault("noise_w", 0.8)
    piper.setdefault("sentence_silence", 0.2)

    # LLM defaults
    llm = cfg.setdefault("llm", {})
    llm.setdefault("model", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
    llm.setdefault("base_url", "")
    llm.setdefault("api_key", "")
    llm.setdefault("max_tokens", 80)
    llm.setdefault("temperature", 0.25)
    llm.setdefault("top_p", 1.0)
    llm.setdefault("top_p_enabled", True)
    llm.setdefault("top_k", 0)
    llm.setdefault("top_k_enabled", True)
    llm.setdefault("repeat_penalty", 1.0)
    llm.setdefault("stop", [])
    llm.setdefault("context_tokens", 0)
    llm.setdefault("multimodal", False)
    llm.setdefault("system_prompt", "You are a concise, friendly real-time voice assistant. Respond in plain English with 2-3 short sentences and no markdown.")
    llm.setdefault("max_history_turns", 8)

    # Phone defaults
    phone = cfg.setdefault("phone", {})
    phone.setdefault("enabled", True)
    phone.setdefault("auto_answer", True)
    phone.setdefault("auto_answer_delay_ms", 500)
    phone.setdefault("caller_allowlist", [])
    phone.setdefault("caller_blocklist", [])
    phone.setdefault("unknown_callers_allowed", True)
    phone.setdefault("greeting_incoming", "Hello, I am {owner}'s assistant. Please wait for the beep before speaking.")
    phone.setdefault("greeting_outgoing", "Hello, this is {owner}'s assistant calling. Please wait for the beep before speaking.")
    phone.setdefault("greeting_owner", "the owner")
    phone.setdefault("max_duration_sec", 300)
    phone.setdefault("max_duration_message", "I'm sorry, but we have reached the maximum call duration. Goodbye!")
    phone.setdefault("auth_passphrase", "")
    phone.setdefault("auth_reject_message", "I'm sorry, I can't help you right now. Goodbye.")
    phone.setdefault("auth_max_attempts", 3)
    phone.setdefault("keep_history", False)
    # ADB path — auto-discover if not set
    if not phone.get("adb_path"):
        phone["adb_path"] = _find_adb()

    # Robot defaults
    robot = cfg.setdefault("robot", {})
    robot.setdefault("enabled", False)
    robot.setdefault("model", "unitree_g1")
    robot.setdefault("transport", "intercom")
    robot.setdefault("intercom_key", "")
    robot.setdefault("heartbeat_interval", 5)
    robot.setdefault("disconnect_timeout", 15)
    robot.setdefault("disconnect_debounce", 3)
    robot.setdefault("safety_stop_on_disconnect", True)
    robot.setdefault("offline_mode", "complete_task")
    robot.setdefault("voice", "am_adam")
    robot.setdefault("voice_lang", "en")
    robot.setdefault("voice_speed", 1.0)
    robot.setdefault("wake_word", "Robert")
    robot.setdefault("wake_phrases", ["hey {name}", "hi {name}", "ok {name}", "listen {name}", "{name}"])
    robot.setdefault("wake_word_timeout", 30)
    robot.setdefault("wake_word_model", "")
    safety = robot.setdefault("safety", {})
    safety.setdefault("max_speed", 0.3)
    safety.setdefault("max_grasp_force", 0.8)
    safety.setdefault("max_reach_m", 0.6)
    safety.setdefault("blocked_commands", [])
    robot.setdefault("sc_bridge_port", 49222)
    robot.setdefault("intercom_channel", "clawfinger-robot-g1")
    robot.setdefault("pear_path", "")
    camera = robot.setdefault("camera", {})
    camera.setdefault("width", 640)
    camera.setdefault("height", 480)
    camera.setdefault("quality", 50)
    camera.setdefault("stream_fps", 5)
    audio_monitor = robot.setdefault("audio_monitor", {})
    audio_monitor.setdefault("sample_rate", 16000)
    audio_monitor.setdefault("channels", 1)
    audio_monitor.setdefault("chunk_ms", 100)
    intercom_limits = robot.setdefault("intercom_limits", {})
    intercom_limits.setdefault("max_message_kb", 256)
    intercom_limits.setdefault("pow", 0)
    intercom_limits.setdefault("rate_limit", 0)
    g1 = robot.setdefault("unitree_g1", {})
    g1.setdefault("jetson_ip", "192.168.123.164")
    g1.setdefault("locomotion_ip", "192.168.123.161")
    g1.setdefault("dds_domain", 0)
    g1.setdefault("max_speed", 0.5)
    g1.setdefault("enable_hands", True)
    g1.setdefault("enable_low_level", False)
    g1.setdefault("wifi_networks", [])
    # Robot LLM — standalone config, same defaults as global llm.
    robot_llm = robot.setdefault("llm", {})
    robot_llm.setdefault("model", llm.get("model", ""))
    robot_llm.setdefault("base_url", "")
    robot_llm.setdefault("api_key", "")
    robot_llm.setdefault("max_tokens", llm.get("max_tokens", 80))
    robot_llm.setdefault("temperature", llm.get("temperature", 0.25))
    robot_llm.setdefault("top_p", 1.0)
    robot_llm.setdefault("top_p_enabled", True)
    robot_llm.setdefault("top_k", 0)
    robot_llm.setdefault("top_k_enabled", True)
    robot_llm.setdefault("repeat_penalty", 1.0)
    robot_llm.setdefault("stop", [])
    robot_llm.setdefault("context_tokens", 0)
    robot_llm.setdefault("multimodal", True)
    robot_llm.setdefault("system_prompt", "You are a physical robot. Respond with short, direct statements. No emojis, no markdown, no tables. Report observations and actions as plain facts.")
    robot_llm.setdefault("max_history_turns", 8)
    memory = robot.setdefault("memory", {})
    memory.setdefault("enabled", True)
    memory.setdefault("db_path", "data/spatial_memory")
    memory.setdefault("embedding_model", "ViT-B-32")
    memory.setdefault("embedding_device", "mps")
    memory.setdefault("max_observations", 100000)

    # Agent defaults
    agent = cfg.setdefault("agent", {})
    agent.setdefault("takeover_timeout", 60)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load() -> dict[str, Any]:
    global _LOADED
    if _LOADED:
        return _LOADED

    with _CFG_PATH.open() as f:
        cfg = json.load(f)

    # Auto-migrate flat format to nested
    if _is_flat(cfg):
        cfg = _migrate_flat_to_nested(cfg)

    # Ensure all defaults exist
    _ensure_defaults(cfg)

    # Env vars override: GATEWAY_<SECTION>_<KEY> e.g. GATEWAY_LLM_MODEL
    # Maps underscore segments to nested dict paths
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("GATEWAY_"):
            continue
        path = env_key[len("GATEWAY_"):].lower()
        # Try dotted path first (GATEWAY_LLM_MODEL → llm.model)
        dotted = path.replace("_", ".", 1)  # first underscore → dot
        current = _get_nested(cfg, dotted)
        if current is not None:
            _set_nested(cfg, dotted, _cast(env_val, current))
        else:
            # Try deeper nesting (GATEWAY_TTS_PIPER_VOICE → tts.piper.voice)
            parts = path.split("_")
            for i in range(1, len(parts)):
                dotted2 = ".".join(["_".join(parts[:i])] + parts[i:]) if i == 1 else ".".join(parts[:i+1])
                # Try each possible split
            # Simple fallback: map section_key pattern
            for depth in range(2, min(len(parts), 4)):
                candidate = ".".join(parts[:depth]) if depth <= len(parts) else None
                if candidate:
                    remaining = "_".join(parts[depth:])
                    if remaining:
                        full = f"{candidate}.{remaining}"
                    else:
                        full = candidate
                    val = _get_nested(cfg, full)
                    if val is not None:
                        _set_nested(cfg, full, _cast(env_val, val))
                        break

    _LOADED = cfg
    return _LOADED


def reload() -> dict[str, Any]:
    global _LOADED
    _LOADED = {}
    return load()


def save() -> None:
    """Persist current in-memory config to config.json (nested format)."""
    if not _LOADED:
        return
    with _CFG_PATH.open("w") as f:
        json.dump(_LOADED, f, indent=2)
        f.write("\n")


def get(key: str, default: Any = None) -> Any:
    """Get config value. Supports dotted paths ('llm.model') and flat compat ('llm_model')."""
    cfg = load()
    # Try dotted path first
    if "." in key:
        return _get_nested(cfg, key, default)
    # Try direct key (root-level: host, port, bearer_token, session_ttl)
    if key in cfg:
        val = cfg[key]
        # Don't return sub-dicts for ambiguous keys
        if not isinstance(val, dict):
            return val
    # Try backward-compat flat key → nested path
    if key in _FLAT_TO_NESTED:
        return _get_nested(cfg, _FLAT_TO_NESTED[key], default)
    return default


def set(key: str, value: Any) -> None:
    """Set config value. Supports dotted paths."""
    cfg = load()
    if "." in key:
        _set_nested(cfg, key, value)
    elif key in _FLAT_TO_NESTED:
        _set_nested(cfg, _FLAT_TO_NESTED[key], value)
    else:
        cfg[key] = value


def section(name: str) -> dict[str, Any]:
    """Get a full config section as a dict (e.g. 'llm', 'phone', 'robot')."""
    cfg = load()
    return cfg.get(name, {})


def robot_llm() -> dict[str, Any]:
    """Get robot LLM config (standalone — not merged with global llm)."""
    cfg = load()
    return dict(cfg.get("robot", {}).get("llm", {}))
