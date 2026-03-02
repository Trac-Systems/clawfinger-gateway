"""Unitree G1 command handler — placeholder stubs until Intercom transport is built."""

from __future__ import annotations

from typing import Any


async def handle_command(command: dict[str, Any]) -> dict[str, Any]:
    """Handle a G1 command. All commands return 'not connected' until
    the Intercom transport layer is implemented."""
    cmd_type = command.get("type", "unknown")
    return {
        "ok": False,
        "error": f"G1 not connected — Intercom transport not yet implemented",
        "command": cmd_type,
    }
