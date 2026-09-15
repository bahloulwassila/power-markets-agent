"""Tests for the anomaly detection module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.anomaly import (
    Anomaly,
    detect_anomalies,
    anomaly_summary,
)


def _make_baseline(days: int = 30, start: str = "2024-05-01") -> pd.Series:
    """Build a synthetic 30-day baseline with typical daily seasonality.

    Prices are ~30 off-peak, ~80 peak, with small random noise (fixed seed
    so the tests are deterministic).
    """
    rng = np.random.default_rng(seed=42)
    n_hours = days * 24
    idx = pd.date_range(start=start, periods=n_hours, freq="h")
    values = []
    for ts in idx:
        h = ts.hour
        base = 80.0 if 17 <= h < 23 else 30.0
        values.append(base + rng.normal(0, 2.0))
    return pd.Series(values, index=idx)


def _make_target_day(prices: list[float], date: str = "2024-06-01") -> pd.Series:
    """Build a 24-hour target-day series."""
    idx = pd.date_range(start=date, periods=len(prices), freq="h")
    return pd.Series(prices, index=idx)


class TestDetectAnomalies:

    def test_no_anomalies_on_normal_day(self):
        baseline = _make_baseline()
        # A perfectly average day → no anomalies
        target = _make_target_day(
            [30.0] * 17 + [80.0] * 6 + [30.0]  # matches baseline shape
        )
        anomalies = detect_anomalies(target, baseline)
        assert anomalies == []

    def test_flags_massive_spike(self):
        baseline = _make_baseline()
        target = _make_target_day([30.0] * 17 + [80.0] * 6 + [30.0])
        # Inject a huge spike at hour 3 (normal is ~30)
        target.iloc[3] = 500.0

        anomalies = detect_anomalies(target, baseline)
        assert len(anomalies) >= 1
        assert anomalies[0].hour_of_day == 3
        assert anomalies[0].direction == "spike"
        assert anomalies[0].z_score > 2.0
        assert anomalies[0].observed_price == 500.0

    def test_flags_negative_dip(self):
        baseline = _make_baseline()
        target = _make_target_day([30.0] * 17 + [80.0] * 6 + [30.0])
        # Inject a big dip at hour 18 (normal peak is ~80)
        target.iloc[18] = -10.0

        anomalies = detect_anomalies(target, baseline)
        assert any(a.direction == "dip" for a in anomalies)
        dip = next(a for a in anomalies if a.direction == "dip")
        assert dip.hour_of_day == 18
        assert dip.z_score < -2.0

    def test_respects_custom_threshold(self):
        baseline = _make_baseline()
        target = _make_target_day([30.0] * 17 + [80.0] * 6 + [30.0])
        target.iloc[10] = 45.0  # moderate spike vs ~30

        # Loose threshold — should flag
        anomalies_loose = detect_anomalies(target, baseline, z_threshold=1.5)
        # Strict threshold — should not flag
        anomalies_strict = detect_anomalies(target, baseline, z_threshold=10.0)

        assert len(anomalies_loose) > len(anomalies_strict)

    def test_sorted_by_abs_zscore_descending(self):
        baseline = _make_baseline()
        target = _make_target_day([30.0] * 17 + [80.0] * 6 + [30.0])
        target.iloc[3] = 200.0    # big spike
        target.iloc[10] = 60.0    # smaller spike
        target.iloc[15] = 100.0   # medium spike

        anomalies = detect_anomalies(target, baseline)
        # Most extreme z first
        z_scores = [abs(a.z_score) for a in anomalies]
        assert z_scores == sorted(z_scores, reverse=True)

    def test_skips_nan_observations(self):
        baseline = _make_baseline()
        target = _make_target_day([30.0] * 17 + [80.0] * 6 + [30.0])
        target.iloc[3] = float("nan")
        # A NaN observation should just be skipped, not raise
        anomalies = detect_anomalies(target, baseline)
        assert all(not np.isnan(a.observed_price) for a in anomalies)

    def test_wrong_type_raises(self):
        baseline = _make_baseline()
        with pytest.raises(TypeError, match="Series"):
            detect_anomalies([1.0, 2.0], baseline)

    def test_wrong_index_type_raises(self):
        baseline = _make_baseline()
        bad = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            detect_anomalies(bad, baseline)


class TestAnomalySummary:

    def test_empty_summary(self):
        summary = anomaly_summary([])
        assert summary["n_anomalies"] == 0
        assert summary["n_spikes"] == 0
        assert summary["n_dips"] == 0
        assert summary["max_abs_z"] == 0.0
        assert summary["worst_spike_hour"] is None
        assert summary["worst_dip_hour"] is None

    def test_summary_counts(self):
        anomalies = [
            Anomaly("t1", 3, 500.0, 30.0, 5.0, 94.0, "spike"),
            Anomaly("t2", 10, 60.0, 30.0, 2.0, 15.0, "spike"),
            Anomaly("t3", 18, -10.0, 80.0, 5.0, -18.0, "dip"),
        ]
        summary = anomaly_summary(anomalies)
        assert summary["n_anomalies"] == 3
        assert summary["n_spikes"] == 2
        assert summary["n_dips"] == 1
        # Worst spike = hour 3 (z=94); worst dip = hour 18 (z=-18)
        assert summary["worst_spike_hour"] == 3
        assert summary["worst_dip_hour"] == 18
        assert summary["max_abs_z"] == 94.0
