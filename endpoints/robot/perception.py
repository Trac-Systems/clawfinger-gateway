"""Per-source perception buffers — camera frames and audio chunks.

Each camera/mic source gets its own ring buffer and asyncio.Event for
signaling new data.  The MJPEG generator yields frames from the buffer
when signaled.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque

logger = logging.getLogger("gateway.robot.perception")

# ---------------------------------------------------------------------------
# Per-source state
# ---------------------------------------------------------------------------

_frame_buffers: dict[str, deque[bytes]] = {}      # source_id -> JPEG frames
_frame_events: dict[str, asyncio.Event] = {}
_audio_buffers: dict[str, deque[bytes]] = {}       # source_id -> PCM chunks
_audio_events: dict[str, asyncio.Event] = {}
_video_streams: set[str] = set()                   # actively streaming source IDs
_audio_monitors: set[str] = set()                  # actively monitoring source IDs
_snapshots: dict[str, tuple[bytes, float]] = {}    # source_id -> (jpeg, timestamp)

_FRAME_BUFFER_SIZE = 10   # ~300 KB at 30 KB/frame
_AUDIO_BUFFER_SIZE = 50   # ~160 KB at 3.2 KB/chunk


# ---------------------------------------------------------------------------
# Frame (camera) helpers
# ---------------------------------------------------------------------------

def push_frame(source: str, image_base64: str, seq: int = 0, ts: float = 0.0) -> None:
    """Decode base64 JPEG, append to ring buffer, signal waiters."""
    try:
        jpeg = base64.b64decode(image_base64)
    except Exception:
        logger.warning("push_frame: invalid base64 for source %s", source)
        return

    buf = _frame_buffers.setdefault(source, deque(maxlen=_FRAME_BUFFER_SIZE))
    buf.append(jpeg)

    ev = _frame_events.setdefault(source, asyncio.Event())
    ev.set()


def set_snapshot(source: str, jpeg_bytes: bytes) -> None:
    """Store the latest snapshot for a source."""
    _snapshots[source] = (jpeg_bytes, time.time())


def get_snapshot(source: str) -> tuple[bytes, float] | None:
    """Return (jpeg_bytes, timestamp) or None."""
    return _snapshots.get(source)


async def frame_generator(source: str):
    """Async generator yielding latest JPEG frames for MJPEG streaming.

    Yields bytes (raw JPEG).  Waits on the source event between frames.
    Stops when the source is no longer streaming.
    """
    ev = _frame_events.setdefault(source, asyncio.Event())
    buf = _frame_buffers.setdefault(source, deque(maxlen=_FRAME_BUFFER_SIZE))

    while is_streaming(source):
        ev.clear()
        # Yield all buffered frames, then wait for new data
        while buf:
            yield buf.popleft()
        try:
            await asyncio.wait_for(ev.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            # No data for 2 s — keep looping to check if still streaming
            continue


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def push_audio(
    source: str,
    audio_base64: str,
    sample_rate: int = 16000,
    channels: int = 1,
    seq: int = 0,
    ts: float = 0.0,
) -> None:
    """Decode base64 PCM S16LE, append to ring buffer, signal waiters."""
    try:
        pcm = base64.b64decode(audio_base64)
    except Exception:
        logger.warning("push_audio: invalid base64 for source %s", source)
        return

    buf = _audio_buffers.setdefault(source, deque(maxlen=_AUDIO_BUFFER_SIZE))
    buf.append(pcm)

    ev = _audio_events.setdefault(source, asyncio.Event())
    ev.set()


# ---------------------------------------------------------------------------
# Stream / monitor state
# ---------------------------------------------------------------------------

def is_streaming(source: str) -> bool:
    return source in _video_streams


def set_streaming(source: str, active: bool) -> None:
    if active:
        _video_streams.add(source)
    else:
        _video_streams.discard(source)


def is_monitoring(source: str) -> bool:
    return source in _audio_monitors


def set_monitoring(source: str, active: bool) -> None:
    if active:
        _audio_monitors.add(source)
    else:
        _audio_monitors.discard(source)


def active_streams() -> list[str]:
    """Return list of currently streaming camera source IDs."""
    return list(_video_streams)


def active_monitors() -> list[str]:
    """Return list of currently monitoring mic source IDs."""
    return list(_audio_monitors)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def stop_all() -> None:
    """Stop all streams and monitors, clear buffers.  Called on disconnect."""
    _video_streams.clear()
    _audio_monitors.clear()
    _frame_buffers.clear()
    _audio_buffers.clear()
    _snapshots.clear()
    # Signal all events so generators exit cleanly
    for ev in _frame_events.values():
        ev.set()
    for ev in _audio_events.values():
        ev.set()
    _frame_events.clear()
    _audio_events.clear()
    logger.info("Perception: all streams/monitors stopped, buffers cleared")
