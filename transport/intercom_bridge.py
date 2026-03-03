"""WebSocket client connecting to the Intercom SC-Bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine

logger = logging.getLogger("gateway.transport.intercom_bridge")


class IntercomBridge:
    """Connects to the Intercom SC-Bridge WebSocket, manages heartbeat
    and disconnect detection, and provides send/receive for the robot channel."""

    def __init__(
        self,
        channel: str,
        sc_bridge_url: str = "ws://127.0.0.1:49222",
        sc_bridge_token: str = "",
        heartbeat_interval: float = 5.0,
        disconnect_timeout: float = 15.0,
        disconnect_debounce: float = 3.0,
    ) -> None:
        self.channel = channel
        self.sc_bridge_url = sc_bridge_url
        self.sc_bridge_token = sc_bridge_token
        self.heartbeat_interval = heartbeat_interval
        self.disconnect_timeout = disconnect_timeout
        self.disconnect_debounce = disconnect_debounce

        self._ws: Any = None  # websockets connection
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._stopping = False
        self._authenticated = False
        self._joined = False

        # Peer state
        self._peer_connected = False
        self._last_peer_message: float = 0.0
        self._disconnect_signal_at: float | None = None

        # Request/response correlation
        self._req_counter = 0
        self._pending: dict[int, asyncio.Future] = {}

        # Callbacks
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
        """True when SC-Bridge WS is open AND robot peer has sent heartbeat_ack."""
        return (
            self._ws is not None
            and self._authenticated
            and self._joined
            and self._peer_connected
        )

    async def start(self) -> None:
        """Connect to SC-Bridge, authenticate, join channel, start loops."""
        self._stopping = False
        await self._connect()

    async def stop(self) -> None:
        """Leave channel, close WS, cancel tasks."""
        self._stopping = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None
        if self._ws:
            try:
                if self._joined:
                    await self._ws.send(json.dumps({
                        "type": "leave", "channel": self.channel,
                    }))
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._authenticated = False
        self._joined = False
        self._peer_connected = False
        # Fail all pending requests
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result({"ok": False, "error": "bridge stopped"})
        self._pending.clear()

    async def send(self, message: dict) -> None:
        """Send a message on the channel (fire-and-forget)."""
        if not self._ws or not self._joined:
            logger.warning("Cannot send: bridge not connected/joined")
            return
        envelope = {
            "type": "send",
            "channel": self.channel,
            "message": message,
        }
        await self._ws.send(json.dumps(envelope))

    async def send_and_wait(self, message: dict, timeout: float = 10.0) -> dict:
        """Send with _req_id and wait for correlated _resp_id response."""
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
    # Internal
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Establish WS connection to SC-Bridge."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed — cannot connect to SC-Bridge")
            return

        try:
            self._ws = await websockets.connect(self.sc_bridge_url)
        except Exception as exc:
            logger.error("SC-Bridge connection failed (%s): %s", self.sc_bridge_url, exc)
            self._schedule_reconnect()
            return

        logger.info("SC-Bridge WS connected: %s", self.sc_bridge_url)

        # Authenticate
        await self._ws.send(json.dumps({
            "type": "auth", "token": self.sc_bridge_token,
        }))

        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            msg = json.loads(raw)
            if msg.get("type") != "auth_ok":
                logger.error("SC-Bridge auth failed: %s", msg)
                await self._ws.close()
                self._ws = None
                self._schedule_reconnect()
                return
        except (asyncio.TimeoutError, Exception) as exc:
            logger.error("SC-Bridge auth timeout/error: %s", exc)
            if self._ws:
                await self._ws.close()
            self._ws = None
            self._schedule_reconnect()
            return

        self._authenticated = True

        # Join channel
        await self._ws.send(json.dumps({
            "type": "join", "channel": self.channel,
        }))
        self._joined = True
        logger.info("Joined channel: %s", self.channel)

        # Start loops
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _recv_loop(self) -> None:
        """Receive messages from SC-Bridge."""
        while not self._stopping and self._ws:
            try:
                raw = await self._ws.recv()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._stopping:
                    logger.warning("SC-Bridge recv error: %s", exc)
                    self._on_ws_disconnect()
                return

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            await self._handle_message(msg)

    async def _handle_message(self, msg: dict) -> None:
        """Process an incoming SC-Bridge message."""
        msg_type = msg.get("type", "")

        # Channel messages from the robot peer
        if msg_type == "message":
            inner = msg.get("message", {})
            if not isinstance(inner, dict):
                return
            self._last_peer_message = time.time()

            inner_type = inner.get("type", "")

            # Heartbeat ack
            if inner_type == "heartbeat_ack":
                was_disconnected = not self._peer_connected
                self._peer_connected = True
                self._disconnect_signal_at = None
                if was_disconnected and self._on_connected:
                    try:
                        await self._on_connected()
                    except Exception as exc:
                        logger.error("on_connected callback error: %s", exc)
                return

            # Response correlation
            resp_id = inner.get("_resp_id")
            if resp_id is not None and resp_id in self._pending:
                fut = self._pending.pop(resp_id)
                if not fut.done():
                    fut.set_result(inner)
                return

            # General message callback
            if self._on_message:
                try:
                    await self._on_message(inner)
                except Exception as exc:
                    logger.error("on_message callback error: %s", exc)

        # Peer join/leave events
        elif msg_type == "peer_join":
            logger.info("Peer joined channel %s", self.channel)
        elif msg_type == "peer_leave":
            logger.info("Peer left channel %s", self.channel)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and check for disconnect."""
        while not self._stopping:
            try:
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                return

            if self._stopping or not self._ws or not self._joined:
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
                        logger.warning("Peer heartbeat timeout (%.1fs)", elapsed)
                    elif time.time() - self._disconnect_signal_at > self.disconnect_debounce:
                        if self._peer_connected:
                            self._peer_connected = False
                            logger.warning("Peer disconnected (debounce confirmed)")
                            if self._on_disconnected:
                                try:
                                    await self._on_disconnected("heartbeat_timeout")
                                except Exception as exc:
                                    logger.error("on_disconnected callback error: %s", exc)
                else:
                    self._disconnect_signal_at = None

    def _on_ws_disconnect(self) -> None:
        """Handle SC-Bridge WS disconnection."""
        self._ws = None
        self._authenticated = False
        self._joined = False
        was_connected = self._peer_connected
        self._peer_connected = False
        if was_connected and self._on_disconnected:
            asyncio.create_task(self._fire_disconnected("ws_disconnect"))
        self._schedule_reconnect()

    async def _fire_disconnected(self, reason: str) -> None:
        if self._on_disconnected:
            try:
                await self._on_disconnected(reason)
            except Exception as exc:
                logger.error("on_disconnected callback error: %s", exc)

    def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self._stopping:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        """Reconnect to SC-Bridge with exponential backoff."""
        delay = 1.0
        max_delay = 60.0
        while not self._stopping:
            logger.info("Reconnecting to SC-Bridge in %.0fs...", delay)
            await asyncio.sleep(delay)
            if self._stopping:
                return
            await self._connect()
            if self._ws and self._authenticated:
                logger.info("Reconnected to SC-Bridge")
                return
            delay = min(delay * 2, max_delay)
