"""
Tests for tracker.multi_day — multi-day runner logic.
"""

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tracker.multi_day import run_multi_day, run_single_day


class TestMultiDayRunner:
    @pytest.mark.asyncio
    async def test_run_single_day_no_clients(self):
        """If no clients are available, returns empty list."""
        with patch("tracker.multi_day.build_selected_clients", return_value={}):
            records = await run_single_day(date.today(), [])
            assert records == []

    @pytest.mark.asyncio
    async def test_run_multi_day_summary_structure(self):
        """Multi-day run returns a properly structured summary dict."""
        mock_records = MagicMock()
        mock_records.__len__ = MagicMock(return_value=5)

        with (
            patch("tracker.multi_day.build_selected_clients", return_value={}),
            patch("tracker.multi_day.save_records", return_value=None),
            patch("tracker.multi_day.append_to_timeseries", return_value=None),
        ):
            summary = await run_multi_day(date.today(), 1, [], model_keys=["none"])
            assert "days" in summary
            assert "total_records" in summary
            assert "total_errors" in summary
            assert "total_refusals" in summary

    @pytest.mark.asyncio
    async def test_multi_day_multiple_days(self):
        """Running 3 days produces 3 day summaries."""
        with (
            patch("tracker.multi_day.build_selected_clients", return_value={}),
            patch("tracker.multi_day.save_records", return_value=None),
            patch("tracker.multi_day.append_to_timeseries", return_value=None),
        ):
            start = date(2026, 6, 1)
            summary = await run_multi_day(start, 3, [], delay_seconds=0)
            assert len(summary["days"]) == 3
            assert summary["days"][0]["date"] == "2026-06-01"
            assert summary["days"][1]["date"] == "2026-06-02"
            assert summary["days"][2]["date"] == "2026-06-03"


class TestDaySummary:
    def test_day_summary_creation(self):
        from app.schemas import DaySummary
        ds = DaySummary(date="2026-05-11", records=15, errors=1, refusals=2, successes=12)
        assert ds.date == "2026-05-11"
        assert ds.records == 15
        assert ds.successes == 12