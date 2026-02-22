"""Async pub/sub event bus for real-time updates to UI and agent WebSockets."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._subscribers.add(ws)

    async def unsubscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._subscribers.discard(ws)

    async def publish(self, event_type: str, data: dict[str, Any] | None = None, session_id: str = "") -> None:
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "session_id": session_id,
            "data": data or {},
        }
        payload = json.dumps(event, default=str)
        dead: list[WebSocket] = []
        async with self._lock:
            subscribers = list(self._subscribers)
        for ws in subscribers:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subscribers.discard(ws)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


bus = EventBus()
