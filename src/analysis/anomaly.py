"""Anomaly detection for hourly electricity prices.

For each observation, compare it to a baseline distribution built from the
same hour-of-day over a rolling look-back window (default 30 days). A
point is flagged as an anomaly if |z-score| > threshold (default 2.0).

Why hour-of-day baseline?
    Electricity prices have strong daily seasonality — 3am prices are
    almost always lower than 6pm prices. Comparing raw prices against a
    single mean would flag every peak-hour price as a "spike". Grouping
    by hour-of-day removes this seasonality.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ANALYSIS


@dataclass(frozen=True)
class Anomaly:
    """A single hourly observation flagged as anomalous."""
    timestamp: str          # ISO datetime of the flagged hour
    hour_of_day: int
    observed_price: float
    baseline_mean: float    # mean of same hour-of-day over lookback window
    baseline_std: float     # stdev of same hour-of-day over lookback window
    z_score: float          # (observed - baseline_mean) / baseline_std
    direction: str          # 'spike' if z > 0, 'dip' if z < 0

    def to_dict(self) -> dict:
        return asdict(self)


def detect_anomalies(
    target_day: pd.Series,
    baseline: pd.Series,
    *,
    z_threshold: float | None = None,
) -> list[Anomaly]:
    """Flag anomalous hours in target_day against a rolling baseline.

    Parameters
    ----------
    target_day : pd.Series
        Hourly prices for the day being examined. DatetimeIndex.
    baseline : pd.Series
        Hourly prices for the look-back window (e.g. the 30 days before
        target_day). DatetimeIndex.
    z_threshold : float, optional
        Absolute z-score above which a point is flagged. Defaults to the
        value in src.config (2.0).

    Returns
    -------
    list[Anomaly]
        Anomalies detected, ordered by absolute z-score descending
        (biggest spikes/dips first).
    """
    _validate_series(target_day, "target_day")
    _validate_series(baseline, "baseline")

    threshold = z_threshold if z_threshold is not None else ANALYSIS.zscore_spike_threshold

    # Group baseline by hour-of-day → mean & std lookup tables
    baseline_by_hour = baseline.groupby(baseline.index.hour)
    hour_mean = baseline_by_hour.mean()
    hour_std = baseline_by_hour.std(ddof=1)

    anomalies: list[Anomaly] = []
    for ts, price in target_day.dropna().items():
        h = ts.hour
        mu = hour_mean.get(h)
        sigma = hour_std.get(h)
        if mu is None or sigma is None or np.isnan(sigma) or sigma == 0:
            continue

        z = (price - mu) / sigma
        if abs(z) > threshold:
            anomalies.append(Anomaly(
                timestamp=_iso(ts),
                hour_of_day=int(h),
                observed_price=round(float(price), 3),
                baseline_mean=round(float(mu), 3),
                baseline_std=round(float(sigma), 3),
                z_score=round(float(z), 3),
                direction="spike" if z > 0 else "dip",
            ))

    # Sort by absolute z-score, most extreme first
    anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)
    return anomalies


def anomaly_summary(anomalies: list[Anomaly]) -> dict:
    """One-line rollup for the daily brief."""
    if not anomalies:
        return {
            "n_anomalies": 0,
            "n_spikes": 0,
            "n_dips": 0,
            "max_abs_z": 0.0,
            "worst_spike_hour": None,
            "worst_dip_hour": None,
        }
    spikes = [a for a in anomalies if a.direction == "spike"]
    dips = [a for a in anomalies if a.direction == "dip"]
    worst_spike = max(spikes, key=lambda a: a.z_score, default=None)
    worst_dip = min(dips, key=lambda a: a.z_score, default=None)
    return {
        "n_anomalies": len(anomalies),
        "n_spikes": len(spikes),
        "n_dips": len(dips),
        "max_abs_z": round(max(abs(a.z_score) for a in anomalies), 3),
        "worst_spike_hour": worst_spike.hour_of_day if worst_spike else None,
        "worst_dip_hour": worst_dip.hour_of_day if worst_dip else None,
    }


def _validate_series(s: pd.Series, name: str) -> None:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be a DatetimeIndex")


def _iso(ts) -> str:
    """Convert a pandas Timestamp / datetime to ISO string."""
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)
