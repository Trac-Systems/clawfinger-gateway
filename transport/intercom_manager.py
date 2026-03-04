"""Manage the Intercom Pear process as a child of the gateway."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
from pathlib import Path
from typing import Any

logger = logging.getLogger("gateway.transport.intercom_manager")

_ROOT = Path(__file__).resolve().parent.parent
_TMP_DIR = _ROOT / "tmp"
_PID_FILE = _TMP_DIR / "intercom.pid"

def _find_pear(configured: str = "") -> str:
    """Auto-discover pear binary: config value, then PATH, then ~/local/node/bin/."""
    if configured:
        return configured
    found = shutil.which("pear")
    if found:
        return found
    # Check common user-local Node install
    home_pear = os.path.expanduser("~/local/node/bin/pear")
    if os.path.isfile(home_pear) and os.access(home_pear, os.X_OK):
        return home_pear
    return "pear"  # fallback to bare name


class IntercomProcess:
    """Spawns and monitors the Intercom Pear process."""

    def __init__(
        self,
        channel: str,
        sc_bridge_port: int,
        sc_bridge_token: str,
        inviter_keys: str,
        pear_path: str = "",
        intercom_dir: str | None = None,
        peer_store_name: str = "gateway",
        intercom_limits: dict | None = None,
    ) -> None:
        self.channel = channel
        self.sc_bridge_port = sc_bridge_port
        self.sc_bridge_token = sc_bridge_token
        self.inviter_keys = inviter_keys
        self.pear_path = _find_pear(pear_path)
        self.intercom_dir = intercom_dir or str(_ROOT / "intercom")
        self.peer_store_name = peer_store_name
        self.intercom_limits = intercom_limits or {}
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task | None = None
        self._stopping = False
        self._restart_count = 0

    def _build_args(self) -> list[str]:
        args = [
            self.pear_path, "run", self.intercom_dir,
            "--peer-store-name", self.peer_store_name,
            "--sc-bridge", "1",
            "--sc-bridge-host", "127.0.0.1",
            "--sc-bridge-port", str(self.sc_bridge_port),
            "--sc-bridge-token", self.sc_bridge_token,
            "--sidechannels", self.channel,
            "--sidechannel-invite-required", "1",
            "--sidechannel-invite-channels", self.channel,
            "--sidechannel-inviter-keys", self.inviter_keys,
            "--sidechannel-welcome-required", "1",
            "--sidechannel-quiet", "1",
        ]
        # Perception/streaming tuning — raise limits for robot channel
        if self.intercom_limits:
            lim = self.intercom_limits
            if lim.get("max_message_kb"):
                args.extend(["--sidechannel-max-message-kb", str(lim["max_message_kb"])])
            if "pow" in lim:
                args.extend(["--sidechannel-pow", str(lim["pow"])])
            if lim.get("rate_limit"):
                args.extend(["--sidechannel-rate-limit", str(lim["rate_limit"])])
        return args

    async def start(self) -> None:
        """Start the Intercom Pear process."""
        self._stopping = False
        args = self._build_args()
        logger.info("Starting Intercom: %s", " ".join(args[:4]) + " ...")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error("Pear binary not found at %s", self.pear_path)
            return
        except Exception as exc:
            logger.error("Failed to start Intercom: %s", exc)
            return

        # Write PID file
        _TMP_DIR.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(self._process.pid))
        logger.info("Intercom started (PID %d)", self._process.pid)

        # Start monitor
        self._monitor_task = asyncio.create_task(self._monitor())

    async def stop(self) -> None:
        """Stop the Intercom Pear process gracefully."""
        self._stopping = True
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self._process and self._process.returncode is None:
            logger.info("Stopping Intercom (PID %d)...", self._process.pid)
            try:
                self._process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning("Intercom didn't stop in 10s, sending SIGKILL")
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass
        self._process = None
        _PID_FILE.unlink(missing_ok=True)
        logger.info("Intercom stopped")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def _monitor(self) -> None:
        """Wait for process exit, auto-restart with exponential cooldown."""
        while not self._stopping:
            if self._process is None:
                return
            try:
                retcode = await self._process.wait()
            except asyncio.CancelledError:
                return
            if self._stopping:
                return

            self._restart_count += 1
            cooldown = min(2 ** self._restart_count, 60)
            logger.warning(
                "Intercom exited (code %s), restarting in %ds (attempt %d)",
                retcode, cooldown, self._restart_count,
            )
            await asyncio.sleep(cooldown)
            if self._stopping:
                return

            args = self._build_args()
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _PID_FILE.write_text(str(self._process.pid))
                logger.info("Intercom restarted (PID %d)", self._process.pid)
            except Exception as exc:
                logger.error("Failed to restart Intercom: %s", exc)
                return
