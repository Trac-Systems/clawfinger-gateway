"""Unitree G1 command handler — dispatches commands via Intercom transport."""

from __future__ import annotations

from typing import Any

from endpoints import robot as robot_mod


async def handle_command(command: dict[str, Any]) -> dict[str, Any]:
    """Handle a G1 command by forwarding it through the Intercom transport."""
    transport = robot_mod.get_transport()
    if transport is None or not transport.is_connected():
        return {"ok": False, "error": "G1 not connected"}

    cmd_type = command.get("type", "unknown")

    # Fire-and-forget commands (stop, safety_stop) — don't wait for response
    if cmd_type in ("stop", "safety_stop"):
        await transport.send({
            "type": "robot_command",
            "command": cmd_type,
        })
        return {"ok": True, "detail": "sent"}

    # Request/response commands — wait for correlated response.
    # Forward all command keys as params (vx, vy, vyaw, etc. may be
    # at the top level or nested under "params").
    params = command.get("params", {})
    # Merge top-level keys into params (top-level keys like vx, vy take priority)
    for k, v in command.items():
        if k not in ("type", "params", "timeout"):
            params.setdefault(k, v)
    return await transport.send_and_wait({
        "type": "robot_command",
        "command": cmd_type,
        "params": params,
    }, timeout=command.get("timeout", 10.0))
