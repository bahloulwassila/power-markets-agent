"""Tests for the daily-statistics module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.statistics import (
    DailyStats,
    compute_daily_stats,
    OFF_PEAK_HOURS,
    PEAK_HOURS,
)


def _make_hourly_series(prices: list[float], start_date: str = "2024-06-12") -> pd.Series:
    """Build an hourly-indexed price series starting at 00:00."""
    idx = pd.date_range(start=start_date, periods=len(prices), freq="h")
    return pd.Series(prices, index=idx)


class TestComputeDailyStats:

    def test_basic_24h_day(self):
        s = _make_hourly_series([50.0] * 24)
        stats = compute_daily_stats(s, market="AESO")

        assert isinstance(stats, DailyStats)
        assert stats.market == "AESO"
        assert stats.date == "2024-06-12"
        assert stats.n_hours == 24
        assert stats.mean == 50.0
        assert stats.min_price == 50.0
        assert stats.max_price == 50.0
        assert stats.std == 0.0
        assert stats.coefficient_of_variation == 0.0

    def test_identifies_min_and_max_hours(self):
        prices = [30.0] * 24
        prices[3] = 10.0
        prices[18] = 200.0
        s = _make_hourly_series(prices)

        stats = compute_daily_stats(s, market="AESO")
        assert stats.min_hour == 3
        assert stats.min_price == 10.0
        assert stats.max_hour == 18
        assert stats.max_price == 200.0

    def test_peak_hours_higher_than_offpeak(self):
        prices = []
        for h in range(24):
            if h in PEAK_HOURS:
                prices.append(120.0)
            elif h in OFF_PEAK_HOURS:
                prices.append(20.0)
            else:
                prices.append(50.0)
        s = _make_hourly_series(prices)

        stats = compute_daily_stats(s, market="AESO")
        assert stats.peak_avg == 120.0
        assert stats.off_peak_avg == 20.0
        assert stats.peak_to_offpeak_ratio == 6.0

    def test_coefficient_of_variation(self):
        s = _make_hourly_series([10.0, 20.0, 30.0, 40.0] * 6)
        stats = compute_daily_stats(s, market="AESO")
        expected_mean = 25.0
        expected_std = float(np.std([10.0, 20.0, 30.0, 40.0] * 6, ddof=1))
        expected_cv = expected_std / expected_mean
        assert abs(stats.mean - expected_mean) < 1e-6
        assert abs(stats.coefficient_of_variation - expected_cv) < 1e-3

    def test_drops_nan(self):
        prices = [50.0] * 24
        prices[5] = float("nan")
        prices[6] = float("nan")
        s = _make_hourly_series(prices)

        stats = compute_daily_stats(s, market="AESO")
        assert stats.n_hours == 22

    def test_partial_day_still_works(self):
        s = _make_hourly_series([50.0] * 6)
        stats = compute_daily_stats(s, market="AESO")
        assert stats.n_hours == 6

    def test_empty_series_raises(self):
        s = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match="empty"):
            compute_daily_stats(s, market="AESO")

    def test_wrong_index_type_raises(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            compute_daily_stats(s, market="AESO")

    def test_not_a_series_raises(self):
        with pytest.raises(TypeError, match="Series"):
            compute_daily_stats([1.0, 2.0, 3.0], market="AESO")

    def test_to_dict_serializable(self):
        s = _make_hourly_series([50.0] * 24)
        stats = compute_daily_stats(s, market="AESO")
        d = stats.to_dict()
        assert d["market"] == "AESO"
        assert d["mean"] == 50.0
        import json
        json.dumps(d)
