"""Robot endpoint — capability registry, model dispatch, command interface."""

from __future__ import annotations

from typing import Any

# Capability registry: populated by each model's __init__.py at import time
_MODELS: dict[str, dict[str, Any]] = {}

# Transport layer — set by app.py startup when Intercom is active
_TRANSPORT: Any = None  # IntercomBridge instance


def set_transport(transport: Any) -> None:
    """Set the active transport (IntercomBridge) for robot communication."""
    global _TRANSPORT
    _TRANSPORT = transport


def get_transport() -> Any:
    """Get the active transport, or None if not connected."""
    return _TRANSPORT


def set_connected(model_name: str, connected: bool) -> None:
    """Update connection status for a registered model."""
    model = _MODELS.get(model_name)
    if model:
        model["connected"] = connected


def register_model(
    name: str,
    capabilities: list[str],
    command_handler: Any = None,
    defaults: dict | None = None,
) -> None:
    """Register a robot model with its capabilities and command handler."""
    _MODELS[name] = {
        "capabilities": set(capabilities),
        "command_handler": command_handler,
        "defaults": defaults or {},
        "connected": False,
    }


def get_model(name: str) -> dict[str, Any] | None:
    """Get a registered model by name."""
    return _MODELS.get(name)


def list_models() -> list[str]:
    """List registered model names."""
    return list(_MODELS.keys())


def get_capabilities(model_name: str) -> set[str]:
    """Get capabilities for a registered model."""
    model = _MODELS.get(model_name)
    return model["capabilities"] if model else set()


def has_capability(model_name: str, capability: str) -> bool:
    """Check if a model supports a given capability."""
    return capability in get_capabilities(model_name)


def get_perception(model_name: str) -> dict:
    """Get perception sources (cameras, mics) for a model. Returns empty dict if none."""
    model = _MODELS.get(model_name)
    if not model:
        return {}
    return model.get("defaults", {}).get("perception", {})


async def dispatch_command(model_name: str, command: dict) -> dict:
    """Dispatch a command to the registered model's handler.

    Returns {"ok": False, ...} if the model is not connected or
    doesn't support the required capability.
    """
    model = _MODELS.get(model_name)
    if not model:
        return {"ok": False, "error": f"Unknown model: {model_name}"}
    if not model["connected"]:
        return {"ok": False, "error": f"Robot not connected ({model_name})"}

    handler = model.get("command_handler")
    if handler is None:
        return {"ok": False, "error": f"No command handler for {model_name}"}

    # Check capability requirement (command type maps to capability)
    cmd_type = command.get("type", "")
    required_cap = _COMMAND_CAPABILITIES.get(cmd_type)
    if required_cap and not has_capability(model_name, required_cap):
        return {"ok": False, "error": f"Capability '{required_cap}' not supported by {model_name}"}

    return await handler(command)


# Command type -> required capability mapping
_COMMAND_CAPABILITIES = {
    "walk": "locomotion",
    "turn": "locomotion",
    "stand": "posture",
    "sit": "posture",
    "crouch": "posture",
    "stop": "posture",
    "pick_up": "manipulation",
    "place": "manipulation",
    "hand_over": "manipulation",
    "reach": "manipulation",
    "pour": "manipulation",
    "grasp": "dexterous_hands",
    "release": "dexterous_hands",
    "wave": "gesture",
    "point": "gesture",
    "nod": "gesture",
    "thumbs_up": "gesture",
    "shake_head": "gesture",
    "look": "vision",
    "snapshot": "vision",
    "detect_object": "vision",
    "camera_snapshot": "vision",
    "camera_stream_start": "vision",
    "camera_stream_stop": "vision",
    "camera_describe": "vision",
    "speak": "audio",
    "listen": "audio",
    "audio_monitor_start": "audio",
    "audio_monitor_stop": "audio",
}


def load_models() -> None:
    """Import all model subpackages to trigger registration."""
    # Each model's __init__.py calls register_model() on import
    try:
        from endpoints.robot import unitree_g1  # noqa: F401
    except ImportError:
        pass
