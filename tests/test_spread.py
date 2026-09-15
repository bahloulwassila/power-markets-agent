"""Tests for the inter-market spread module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.spread import SpreadStats, compute_spread


def _make_series(prices: list[float], start: str = "2024-06-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(prices), freq="h")
    return pd.Series(prices, index=idx)


class TestComputeSpread:

    def test_constant_offset(self):
        # AESO always $20 above IESO
        aeso = _make_series([50.0] * 24)
        ieso = _make_series([30.0] * 24)

        stats = compute_spread(aeso, ieso)
        assert isinstance(stats, SpreadStats)
        assert stats.n_matched_hours == 24
        assert stats.mean_spread == 20.0
        assert stats.median_spread == 20.0
        assert stats.std_spread == 0.0
        assert stats.min_spread == 20.0
        assert stats.max_spread == 20.0

    def test_perfect_correlation(self):
        # Two series that move together perfectly (both are 10, 20, 30, ...)
        vals = list(range(10, 34))
        aeso = _make_series([float(v) for v in vals])
        ieso = _make_series([float(v * 2) for v in vals])
        stats = compute_spread(aeso, ieso)
        # Correlation should be very close to 1.0
        assert abs(stats.correlation - 1.0) < 1e-6

    def test_widest_gap_identified(self):
        aeso = _make_series([40.0] * 24)
        ieso = _make_series([30.0] * 24)
        # Big Alberta spike at hour 18 → widest gap there
        aeso.iloc[18] = 300.0

        stats = compute_spread(aeso, ieso)
        assert stats.widest_gap_hour == 18
        assert stats.widest_gap_value == 270.0  # 300 - 30

    def test_wide_gaps_count_respects_threshold(self):
        aeso = _make_series([40.0] * 24)
        ieso = _make_series([30.0] * 24)
        aeso.iloc[10] = 100.0   # spread = 70
        aeso.iloc[18] = 150.0   # spread = 120

        # Threshold 50 → both flagged
        s1 = compute_spread(aeso, ieso, wide_gap_threshold=50.0)
        assert s1.n_wide_gaps == 2

        # Threshold 100 → only the 150.00 one
        s2 = compute_spread(aeso, ieso, wide_gap_threshold=100.0)
        assert s2.n_wide_gaps == 1

    def test_only_overlapping_hours_used(self):
        # AESO has 24h, IESO only 12h — inner join should give 12
        aeso = _make_series([50.0] * 24)
        ieso_short = _make_series([30.0] * 12)
        stats = compute_spread(aeso, ieso_short)
        assert stats.n_matched_hours == 12

    def test_no_overlap_raises(self):
        aeso = _make_series([50.0] * 24, start="2024-06-01")
        ieso = _make_series([30.0] * 24, start="2024-07-01")
        with pytest.raises(ValueError, match="no overlapping hours"):
            compute_spread(aeso, ieso)

    def test_wrong_type_raises(self):
        ieso = _make_series([30.0] * 24)
        with pytest.raises(TypeError, match="Series"):
            compute_spread([1.0, 2.0], ieso)
