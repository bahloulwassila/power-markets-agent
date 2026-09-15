"""Tests for the ReportData aggregator."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analysis.report_data import ReportData, build_report_data


def _make_hourly(prices: list[float], start: str = "2024-06-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(prices), freq="h")
    return pd.Series(prices, index=idx)


def _synthetic_baseline(days: int = 30, start: str = "2024-05-01") -> pd.Series:
    rng = np.random.default_rng(seed=7)
    n = days * 24
    idx = pd.date_range(start=start, periods=n, freq="h")
    vals = [40.0 + rng.normal(0, 3.0) for _ in range(n)]
    return pd.Series(vals, index=idx)


class TestBuildReportData:

    def test_full_report_all_inputs(self):
        aeso_day = _make_hourly([45.0] * 24)
        aeso_baseline = _synthetic_baseline()
        ieso_day = _make_hourly([35.0] * 24)
        ieso_baseline = _synthetic_baseline(start="2024-05-02")
        news = [{"title": "Something", "url": "http://x", "relevance_score": 0.5}]

        rd = build_report_data(
            target_date=date(2024, 6, 1),
            aeso_target_day=aeso_day,
            aeso_baseline=aeso_baseline,
            ieso_target_day=ieso_day,
            ieso_baseline=ieso_baseline,
            news_items=news,
        )

        assert isinstance(rd, ReportData)
        assert rd.report_date == "2024-06-01"
        assert rd.aeso_stats is not None
        assert rd.ieso_stats is not None
        assert rd.spread_stats is not None
        assert rd.aeso_stats["mean"] == 45.0
        assert rd.ieso_stats["mean"] == 35.0
        assert len(rd.news_items) == 1

    def test_ieso_missing_graceful_degradation(self):
        aeso_day = _make_hourly([45.0] * 24)
        aeso_baseline = _synthetic_baseline()

        rd = build_report_data(
            target_date=date(2024, 6, 1),
            aeso_target_day=aeso_day,
            aeso_baseline=aeso_baseline,
            ieso_target_day=None,
            ieso_baseline=None,
        )
        assert rd.aeso_stats is not None
        assert rd.ieso_stats is None
        assert rd.spread_stats is None
        assert any("IESO" in w for w in rd.warnings)

    def test_no_baseline_still_produces_stats(self):
        # If baseline is missing, we still get daily stats, just no anomalies
        aeso_day = _make_hourly([45.0] * 24)

        rd = build_report_data(
            target_date=date(2024, 6, 1),
            aeso_target_day=aeso_day,
            aeso_baseline=None,
        )
        assert rd.aeso_stats is not None
        assert rd.aeso_anomalies == []
        assert any("baseline missing" in w.lower() for w in rd.warnings)

    def test_all_missing_only_warnings(self):
        rd = build_report_data(target_date=date(2024, 6, 1))
        assert rd.aeso_stats is None
        assert rd.ieso_stats is None
        assert rd.spread_stats is None
        assert len(rd.warnings) >= 2

    def test_to_dict_is_json_serializable(self):
        aeso_day = _make_hourly([45.0] * 24)
        rd = build_report_data(
            target_date=date(2024, 6, 1),
            aeso_target_day=aeso_day,
        )
        import json
        json.dumps(rd.to_dict())  # should not raise
