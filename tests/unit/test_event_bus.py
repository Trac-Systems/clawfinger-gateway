"""Unit tests for event_bus module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from event_bus import EventBus


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_subscribe_and_publish(bus, mock_ws):
    await bus.subscribe(mock_ws)
    await bus.publish("test.event", {"key": "value"}, session_id="s1")

    assert mock_ws.send_text.call_count == 1
    payload = json.loads(mock_ws.send_text.call_args[0][0])
    assert payload["type"] == "test.event"
    assert payload["session_id"] == "s1"
    assert payload["data"]["key"] == "value"
    assert "timestamp" in payload
    assert payload["endpoint"] == ""


@pytest.mark.asyncio
async def test_endpoint_field_present(bus, mock_ws):
    await bus.subscribe(mock_ws)
    await bus.publish("robot.connected", {"model": "g1"}, endpoint="robot")

    payload = json.loads(mock_ws.send_text.call_args[0][0])
    assert payload["endpoint"] == "robot"
    assert payload["type"] == "robot.connected"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery(bus, mock_ws):
    await bus.subscribe(mock_ws)
    await bus.unsubscribe(mock_ws)
    await bus.publish("test.event", {})

    assert mock_ws.send_text.call_count == 0


@pytest.mark.asyncio
async def test_dead_subscriber_removed(bus):
    ws = AsyncMock()
    ws.send_text = AsyncMock(side_effect=Exception("dead"))
    await bus.subscribe(ws)
    await bus.publish("test.event", {})

    # After failure, subscriber should be removed
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_multiple_subscribers(bus):
    ws1 = AsyncMock()
    ws1.send_text = AsyncMock()
    ws2 = AsyncMock()
    ws2.send_text = AsyncMock()

    await bus.subscribe(ws1)
    await bus.subscribe(ws2)
    await bus.publish("test.event", {})

    assert ws1.send_text.call_count == 1
    assert ws2.send_text.call_count == 1


@pytest.mark.asyncio
async def test_subscriber_count(bus, mock_ws):
    assert bus.subscriber_count == 0
    await bus.subscribe(mock_ws)
    assert bus.subscriber_count == 1
    await bus.unsubscribe(mock_ws)
    assert bus.subscriber_count == 0
