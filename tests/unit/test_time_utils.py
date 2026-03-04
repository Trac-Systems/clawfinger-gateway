"""Unit tests for time_utils module."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import config
import time_utils


@pytest.fixture(autouse=True)
def _load_config(tmp_config):
    """Ensure config is loaded with timezone default."""
    config.load()


class TestRelative:
    def test_just_now(self):
        assert time_utils.relative(time.time()) == "just now"

    def test_seconds_ago(self):
        assert time_utils.relative(time.time() - 30) == "30s ago"

    def test_minutes_ago(self):
        assert time_utils.relative(time.time() - 300) == "5 min ago"

    def test_hours_ago(self):
        assert time_utils.relative(time.time() - 7200) == "2h ago"

    def test_yesterday(self):
        assert time_utils.relative(time.time() - 86400) == "yesterday"

    def test_days_ago(self):
        assert time_utils.relative(time.time() - 259200) == "3 days ago"

    def test_months_ago(self):
        assert time_utils.relative(time.time() - 86400 * 60) == "2 months ago"

    def test_future_timestamp(self):
        assert time_utils.relative(time.time() + 100) == "just now"


class TestDuration:
    def test_seconds(self):
        assert time_utils.duration(45) == "45s"

    def test_minutes(self):
        assert time_utils.duration(192) == "3m 12s"

    def test_minutes_exact(self):
        assert time_utils.duration(120) == "2m"

    def test_hours(self):
        assert time_utils.duration(3900) == "1h 5m"

    def test_hours_exact(self):
        assert time_utils.duration(3600) == "1h"

    def test_zero(self):
        assert time_utils.duration(0) == "0s"

    def test_negative(self):
        assert time_utils.duration(-5) == "0s"


class TestTimeOfDay:
    def test_returns_string(self):
        result = time_utils.time_of_day()
        assert result in ("morning", "afternoon", "evening", "night")


class TestTimeContextLine:
    def test_format(self):
        line = time_utils.time_context_line()
        assert line.startswith("Current time: ")
        assert "(" in line  # contains time-of-day in parens


class TestParseNaturalTime:
    def test_last_hour(self):
        result = time_utils.parse_natural_time("last hour")
        assert result is not None
        start, end = result
        assert end - start == pytest.approx(3600, abs=2)

    def test_last_3_days(self):
        result = time_utils.parse_natural_time("last 3 days")
        assert result is not None
        start, end = result
        assert end - start == pytest.approx(86400 * 3, abs=2)

    def test_today(self):
        result = time_utils.parse_natural_time("today")
        assert result is not None
        start, end = result
        assert start < end
        assert end == pytest.approx(time.time(), abs=2)

    def test_yesterday(self):
        result = time_utils.parse_natural_time("yesterday")
        assert result is not None
        start, end = result
        assert end - start == pytest.approx(86400, abs=2)

    def test_last_30_minutes(self):
        result = time_utils.parse_natural_time("last 30 minutes")
        assert result is not None
        start, end = result
        assert end - start == pytest.approx(1800, abs=2)

    def test_unknown_returns_none(self):
        assert time_utils.parse_natural_time("next Tuesday") is None

    def test_empty_returns_none(self):
        assert time_utils.parse_natural_time("") is None


class TestAnnotateResults:
    def test_adds_time_ago(self):
        results = [
            {"entity_name": "Alice", "timestamp": time.time() - 300},
            {"entity_name": "Cup", "timestamp": time.time() - 7200},
        ]
        annotated = time_utils.annotate_results(results)
        assert annotated[0]["time_ago"] == "5 min ago"
        assert annotated[1]["time_ago"] == "2h ago"

    def test_skips_missing_timestamp(self):
        results = [{"entity_name": "Test"}]
        annotated = time_utils.annotate_results(results)
        assert "time_ago" not in annotated[0]

    def test_modifies_in_place(self):
        results = [{"timestamp": time.time()}]
        annotated = time_utils.annotate_results(results)
        assert annotated is results

    def test_handles_invalid_timestamp(self):
        results = [{"timestamp": "not_a_number"}]
        annotated = time_utils.annotate_results(results)
        assert "time_ago" not in annotated[0]
