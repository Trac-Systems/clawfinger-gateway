"""Central time utilities for system-wide time awareness.

All time-related helpers used across phone, robot, OpenClaw, and UI.
Uses stdlib ``zoneinfo`` (Python 3.9+) — no external dependencies.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _tz() -> ZoneInfo:
    return ZoneInfo(config.get("timezone", "Europe/Berlin"))


def now() -> datetime:
    """Current datetime in the configured timezone."""
    return datetime.now(_tz())


def now_unix() -> float:
    """Current Unix timestamp."""
    return time.time()


def relative(ts: float) -> str:
    """Unix timestamp → human-readable relative string.

    Examples: "just now", "5 min ago", "yesterday", "3 days ago".
    """
    delta = time.time() - ts
    if delta < 0:
        return "just now"
    if delta < 10:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        mins = int(delta / 60)
        return f"{mins} min ago"
    if delta < 86400:
        hours = int(delta / 3600)
        return f"{hours}h ago"
    days = int(delta / 86400)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    months = int(days / 30)
    if months == 1:
        return "1 month ago"
    return f"{months} months ago"


def duration(seconds: float) -> str:
    """Seconds → compact duration string.

    Examples: "45s", "3m 12s", "1h 5m".
    """
    s = int(seconds)
    if s < 0:
        s = 0
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, remainder = divmod(s, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


def time_of_day() -> str:
    """Current period: morning, afternoon, evening, or night."""
    hour = now().hour
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def time_context_line() -> str:
    """Full time context line for LLM injection.

    Example: ``Current time: Tuesday, 2026-03-04 14:32 CET (afternoon)``
    """
    dt = now()
    tz_abbr = dt.strftime("%Z")
    day_name = dt.strftime("%A")
    date_str = dt.strftime("%Y-%m-%d %H:%M")
    return f"Current time: {day_name}, {date_str} {tz_abbr} ({time_of_day()})"


# ---------------------------------------------------------------------------
# Natural time parsing
# ---------------------------------------------------------------------------

_NATURAL_PATTERNS: list[tuple[re.Pattern, callable]] = []


def _init_patterns() -> None:
    """Compile natural time expression patterns (lazy init)."""
    if _NATURAL_PATTERNS:
        return

    def _last_n(match: re.Match) -> tuple[float, float]:
        n = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        multipliers = {"minute": 60, "min": 60, "hour": 3600, "day": 86400, "week": 604800}
        secs = n * multipliers.get(unit, 3600)
        return (time.time() - secs, time.time())

    def _last_unit(match: re.Match) -> tuple[float, float]:
        unit = match.group(1).lower().rstrip("s")
        multipliers = {"minute": 60, "min": 60, "hour": 3600, "day": 86400, "week": 604800}
        secs = multipliers.get(unit, 3600)
        return (time.time() - secs, time.time())

    def _today(_match: re.Match) -> tuple[float, float]:
        dt = now().replace(hour=0, minute=0, second=0, microsecond=0)
        return (dt.timestamp(), time.time())

    def _yesterday(_match: re.Match) -> tuple[float, float]:
        dt = now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = (dt - timedelta(days=1)).timestamp()
        return (start, dt.timestamp())

    _NATURAL_PATTERNS.extend([
        (re.compile(r"last\s+(\d+)\s+(minutes?|mins?|hours?|days?|weeks?)", re.I), _last_n),
        (re.compile(r"last\s+(minute|min|hour|day|week)", re.I), _last_unit),
        (re.compile(r"today", re.I), _today),
        (re.compile(r"yesterday", re.I), _yesterday),
    ])


def parse_natural_time(expr: str) -> tuple[float, float] | None:
    """Parse natural time expression → ``(start_ts, end_ts)`` or ``None``.

    Supported: "last hour", "last 3 days", "today", "yesterday",
    "last 30 minutes", "last week", etc.
    """
    _init_patterns()
    expr = expr.strip()
    for pattern, handler in _NATURAL_PATTERNS:
        m = pattern.search(expr)
        if m:
            return handler(m)
    return None


# ---------------------------------------------------------------------------
# Result annotation
# ---------------------------------------------------------------------------


def annotate_results(results: list[dict]) -> list[dict]:
    """Add ``time_ago`` field to every dict that has a ``timestamp``.

    Modifies dicts in-place and returns the same list.
    """
    for item in results:
        ts = item.get("timestamp")
        if ts is not None:
            try:
                item["time_ago"] = relative(float(ts))
            except (ValueError, TypeError):
                pass
    return results
