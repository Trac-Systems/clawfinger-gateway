"""Direct WebSocket transport for robot/sim-adapter connections.

Alternative to Intercom P2P — used for simulation (Isaac Sim adapter) and
local development where NAT traversal isn't needed.  The gateway listens for
an incoming WS connection on ``/ws/robot``; the sim adapter (or Jetson in
dev mode) connects and speaks the same message protocol as IntercomBridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine

from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger("gateway.transport.ws_bridge")


class WebSocketRobotBridge:
    """Drop-in replacement for IntercomBridge that accepts a single direct
    WebSocket connection from a robot or sim adapter."""

    def __init__(
        self,
        heartbeat_interval: float = 5.0,
        disconnect_timeout: float = 15.0,
        disconnect_debounce: float = 3.0,
        bearer_token: str = "",
    ) -> None:
        self.heartbeat_interval = heartbeat_interval
        self.disconnect_timeout = disconnect_timeout
        self.disconnect_debounce = disconnect_debounce
        self.bearer_token = bearer_token

        self._ws: WebSocket | None = None
        self._peer_connected = False
        self._last_peer_message: float = 0.0
        self._heartbeat_task: asyncio.Task | None = None
        self._disconnect_signal_at: float | None = None
        self._stopping = False

        # Request/response correlation
        self._req_counter = 0
        self._pending: dict[int, asyncio.Future] = {}

        # Callbacks (same interface as IntercomBridge)
        self._on_message: Callable[[dict], Coroutine] | None = None
        self._on_connected: Callable[[], Coroutine] | None = None
        self._on_disconnected: Callable[[str], Coroutine] | None = None

    def on_message(self, handler: Callable[[dict], Coroutine]) -> None:
        self._on_message = handler

    def on_connected(self, handler: Callable[[], Coroutine]) -> None:
        self._on_connected = handler

    def on_disconnected(self, handler: Callable[[str], Coroutine]) -> None:
        self._on_disconnected = handler

    def is_connected(self) -> bool:
        return self._ws is not None and self._peer_connected

    async def start(self) -> None:
        """No-op — the bridge starts when a client connects via handle_ws."""
        self._stopping = False
        logger.info("WebSocket robot bridge ready (waiting for connection)")

    async def stop(self) -> None:
        self._stopping = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._peer_connected = False
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result({"ok": False, "error": "bridge stopped"})
        self._pending.clear()

    async def send(self, message: dict) -> None:
        if not self._ws or not self._peer_connected:
            logger.warning("Cannot send: no robot connected via WS")
            return
        try:
            await self._ws.send_json(message)
        except Exception as exc:
            logger.warning("WS send error: %s", exc)

    async def send_and_wait(self, message: dict, timeout: float = 10.0) -> dict:
        self._req_counter += 1
        req_id = self._req_counter
        message["_req_id"] = req_id

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        self._pending[req_id] = fut

        try:
            await self.send(message)
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout"}
        finally:
            self._pending.pop(req_id, None)

    # ------------------------------------------------------------------
    # WebSocket handler — called by the Starlette route
    # ------------------------------------------------------------------

    async def handle_ws(self, ws: WebSocket) -> None:
        """Accept and manage a robot/sim WebSocket connection."""
        # Auth check
        if self.bearer_token:
            auth = ws.headers.get("authorization", "")
            qs_token = ws.query_params.get("token", "")
            if not (auth == f"Bearer {self.bearer_token}" or qs_token == self.bearer_token):
                await ws.close(code=4001, reason="unauthorized")
                return

        await ws.accept()

        # Replace existing connection (only one robot at a time)
        if self._ws is not None:
            logger.warning("New robot WS connection replacing existing one")
            try:
                await self._ws.close(code=4002, reason="replaced")
            except Exception:
                pass

        self._ws = ws
        self._peer_connected = True
        self._last_peer_message = time.time()
        self._disconnect_signal_at = None
        logger.info("Robot/sim connected via WebSocket")

        if self._on_connected:
            try:
                await self._on_connected()
            except Exception as exc:
                logger.error("on_connected callback error: %s", exc)

        # Start heartbeat if not running
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Receive loop
        try:
            while not self._stopping:
                data = await ws.receive_json()
                self._last_peer_message = time.time()
                await self._handle_message(data)
        except WebSocketDisconnect:
            logger.info("Robot/sim WebSocket disconnected")
        except Exception as exc:
            logger.warning("Robot/sim WS error: %s", exc)
        finally:
            if self._ws is ws:
                self._ws = None
                was_connected = self._peer_connected
                self._peer_connected = False
                if was_connected and self._on_disconnected:
                    try:
                        await self._on_disconnected("ws_disconnect")
                    except Exception as exc:
                        logger.error("on_disconnected callback error: %s", exc)

    async def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type", "")

        # Heartbeat ack
        if msg_type == "heartbeat_ack":
            return

        # Response correlation
        resp_id = msg.get("_resp_id")
        if resp_id is not None and resp_id in self._pending:
            fut = self._pending.pop(resp_id)
            if not fut.done():
                fut.set_result(msg)
            return

        # General message callback
        if self._on_message:
            try:
                await self._on_message(msg)
            except Exception as exc:
                logger.error("on_message callback error: %s", exc)

    async def _heartbeat_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                return

            if self._stopping or not self._ws:
                return

            # Send heartbeat
            try:
                await self.send({"type": "heartbeat", "ts": time.time()})
            except Exception:
                pass

            # Check disconnect
            if self._last_peer_message > 0:
                elapsed = time.time() - self._last_peer_message
                if elapsed > self.disconnect_timeout:
                    if self._disconnect_signal_at is None:
                        self._disconnect_signal_at = time.time()
                    elif time.time() - self._disconnect_signal_at > self.disconnect_debounce:
                        if self._peer_connected:
                            self._peer_connected = False
                            logger.warning("Robot/sim heartbeat timeout")
                            if self._on_disconnected:
                                try:
                                    await self._on_disconnected("heartbeat_timeout")
                                except Exception as exc:
                                    logger.error("on_disconnected error: %s", exc)
                else:
                    self._disconnect_signal_at = None
