"""Unit tests for IntercomBridge — heartbeat, disconnect detection, request correlation."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from transport.intercom_bridge import IntercomBridge


@pytest.fixture
def bridge():
    """Create a bridge instance with short timeouts for testing."""
    b = IntercomBridge(
        channel="test-channel",
        sc_bridge_url="ws://127.0.0.1:49222",
        sc_bridge_token="test-token",
        heartbeat_interval=0.5,
        disconnect_timeout=1.0,
        disconnect_debounce=0.5,
    )
    return b


class TestBridgeState:
    def test_initial_state(self, bridge):
        assert not bridge.is_connected()
        assert bridge.channel == "test-channel"
        assert bridge.sc_bridge_token == "test-token"

    def test_is_connected_requires_all(self, bridge):
        """is_connected() requires WS + auth + join + peer."""
        assert not bridge.is_connected()
        bridge._ws = MagicMock()
        assert not bridge.is_connected()
        bridge._authenticated = True
        assert not bridge.is_connected()
        bridge._joined = True
        assert not bridge.is_connected()
        bridge._peer_connected = True
        assert bridge.is_connected()


class TestMessageHandling:
    @pytest.mark.asyncio
    async def test_heartbeat_ack_sets_connected(self, bridge):
        """heartbeat_ack from peer triggers on_connected callback."""
        connected_called = asyncio.Event()
        async def on_connected():
            connected_called.set()
        bridge.on_connected(on_connected)
        bridge._ws = MagicMock()
        bridge._authenticated = True
        bridge._joined = True

        await bridge._handle_message({
            "type": "message",
            "message": {"type": "heartbeat_ack", "ts": time.time(), "battery": 85},
        })

        assert bridge._peer_connected
        assert connected_called.is_set()

    @pytest.mark.asyncio
    async def test_heartbeat_ack_only_fires_once(self, bridge):
        """on_connected callback only fires on transition from disconnected."""
        call_count = 0
        async def on_connected():
            nonlocal call_count
            call_count += 1
        bridge.on_connected(on_connected)
        bridge._ws = MagicMock()
        bridge._authenticated = True
        bridge._joined = True

        msg = {"type": "message", "message": {"type": "heartbeat_ack", "ts": time.time()}}
        await bridge._handle_message(msg)
        await bridge._handle_message(msg)
        await bridge._handle_message(msg)

        assert call_count == 1  # Only first transition fires

    @pytest.mark.asyncio
    async def test_response_correlation(self, bridge):
        """_resp_id in message resolves the matching pending future."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        bridge._pending[42] = fut

        await bridge._handle_message({
            "type": "message",
            "message": {"type": "robot_response", "_resp_id": 42, "ok": True, "detail": "walking"},
        })

        assert fut.done()
        result = fut.result()
        assert result["ok"] is True
        assert result["detail"] == "walking"
        assert 42 not in bridge._pending

    @pytest.mark.asyncio
    async def test_general_message_callback(self, bridge):
        """Non-heartbeat, non-response messages trigger on_message callback."""
        received = []
        async def on_msg(msg):
            received.append(msg)
        bridge.on_message(on_msg)

        await bridge._handle_message({
            "type": "message",
            "message": {"type": "robot_event", "event": "task_completed", "task_id": "abc"},
        })

        assert len(received) == 1
        assert received[0]["type"] == "robot_event"
        assert received[0]["task_id"] == "abc"

    @pytest.mark.asyncio
    async def test_updates_last_peer_message(self, bridge):
        """Any channel message updates _last_peer_message timestamp."""
        assert bridge._last_peer_message == 0.0

        await bridge._handle_message({
            "type": "message",
            "message": {"type": "heartbeat_ack", "ts": time.time()},
        })

        assert bridge._last_peer_message > 0


class TestSendAndWait:
    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, bridge):
        """send_and_wait returns timeout error when no response arrives."""
        bridge._ws = AsyncMock()
        bridge._joined = True
        bridge._ws.send = AsyncMock()

        result = await bridge.send_and_wait(
            {"type": "robot_command", "command": "walk"},
            timeout=0.1,
        )

        assert result["ok"] is False
        assert result["error"] == "timeout"
        # Pending should be cleaned up
        assert len(bridge._pending) == 0

    @pytest.mark.asyncio
    async def test_successful_correlation(self, bridge):
        """send_and_wait resolves when matching _resp_id arrives."""
        bridge._ws = AsyncMock()
        bridge._joined = True
        bridge._ws.send = AsyncMock()

        async def simulate_response():
            await asyncio.sleep(0.05)
            # The _req_id assigned will be 1 (first call)
            await bridge._handle_message({
                "type": "message",
                "message": {"_resp_id": 1, "ok": True, "detail": "done"},
            })

        task = asyncio.create_task(simulate_response())
        result = await bridge.send_and_wait(
            {"type": "robot_command", "command": "stand"},
            timeout=2.0,
        )
        await task

        assert result["ok"] is True
        assert result["detail"] == "done"


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_pending(self, bridge):
        """stop() resolves all pending futures with error."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        bridge._pending[99] = fut

        await bridge.stop()

        assert fut.done()
        assert fut.result()["ok"] is False
        assert bridge._pending == {}

    @pytest.mark.asyncio
    async def test_stop_resets_state(self, bridge):
        """stop() clears all connection state."""
        bridge._ws = MagicMock()
        bridge._ws.send = AsyncMock()
        bridge._ws.close = AsyncMock()
        bridge._authenticated = True
        bridge._joined = True
        bridge._peer_connected = True

        await bridge.stop()

        assert bridge._ws is None
        assert not bridge._authenticated
        assert not bridge._joined
        assert not bridge._peer_connected
