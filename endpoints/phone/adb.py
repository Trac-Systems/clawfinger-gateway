"""ADB dial/hangup commands for the phone endpoint."""

from __future__ import annotations

import asyncio

import config
import session_store


async def do_dial(number: str) -> dict:
    """Send dial command to phone via ADB broadcast."""
    if not number:
        return {"ok": False, "detail": "number required"}
    adb = config.get("phone.adb_path", "adb")
    # Check ADB connection
    try:
        proc = await asyncio.create_subprocess_exec(
            adb, "devices",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode().strip().split("\n")
        devices = [l for l in lines[1:] if l.strip() and "device" in l]
        if not devices:
            return {"ok": False, "detail": "No ADB device connected"}
    except Exception as exc:
        return {"ok": False, "detail": f"ADB check failed: {exc}"}
    # Send broadcast
    try:
        proc = await asyncio.create_subprocess_exec(
            adb, "shell", "am", "broadcast",
            "-a", "com.tracsystems.phonebridge.CALL_COMMAND",
            "-n", "com.tracsystems.phonebridge/.CallCommandReceiver",
            "--es", "type", "dial",
            "--es", "number", number,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        output = stdout.decode().strip()
        if proc.returncode != 0:
            return {"ok": False, "detail": f"ADB broadcast failed: {stderr.decode().strip()}"}
        return {"ok": True, "detail": output}
    except Exception as exc:
        return {"ok": False, "detail": f"Dial failed: {exc}"}


async def do_hangup() -> dict:
    """Send hangup command to phone via ADB broadcast."""
    adb = config.get("phone.adb_path", "adb")
    try:
        proc = await asyncio.create_subprocess_exec(
            adb, "devices",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        lines = stdout.decode().strip().split("\n")
        devices = [l for l in lines[1:] if l.strip() and "device" in l]
        if not devices:
            return {"ok": False, "detail": "No ADB device connected"}
    except Exception as exc:
        return {"ok": False, "detail": f"ADB check failed: {exc}"}
    try:
        proc = await asyncio.create_subprocess_exec(
            adb, "shell", "am", "broadcast",
            "-a", "com.tracsystems.phonebridge.CALL_COMMAND",
            "-n", "com.tracsystems.phonebridge/.CallCommandReceiver",
            "--es", "type", "hangup",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            return {"ok": False, "detail": f"ADB broadcast failed: {stderr.decode().strip()}"}
        return {"ok": True, "detail": stdout.decode().strip()}
    except Exception as exc:
        return {"ok": False, "detail": f"Hangup failed: {exc}"}


def single_active_session() -> str:
    """Return the one active session ID if exactly one exists, else empty string."""
    active = session_store.active_sessions()
    return next(iter(active)) if len(active) == 1 else ""
