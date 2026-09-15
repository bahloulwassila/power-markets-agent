"""Daily statistics for hourly electricity price series.

Computes summary metrics that feed the Claude research brief: level (mean),
dispersion (std, CV), extremes (min/max and their hours), and simple
period-of-day averages (off-peak, mid-day, peak).

These are all deterministic — the LLM never re-derives them, it just
narrates them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# Hour-of-day windows (0-indexed, matching pandas datetime.hour)
OFF_PEAK_HOURS = list(range(0, 7)) + list(range(23, 24))   # 23:00 - 07:00
MID_DAY_HOURS = list(range(7, 17))                          # 07:00 - 17:00
PEAK_HOURS = list(range(17, 23))                            # 17:00 - 23:00


@dataclass(frozen=True)
class DailyStats:
    """Summary statistics for one day of hourly prices."""
    market: str                # 'AESO' or 'IESO'
    date: str                  # ISO YYYY-MM-DD
    n_hours: int
    mean: float
    median: float
    std: float
    min_price: float
    min_hour: int
    max_price: float
    max_hour: int
    coefficient_of_variation: float
    off_peak_avg: float
    mid_day_avg: float
    peak_avg: float
    peak_to_offpeak_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_daily_stats(
    prices: pd.Series,
    market: str,
) -> DailyStats:
    """Compute one day of statistics from an hourly price series."""
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices index must be a DatetimeIndex")

    clean = prices.dropna()
    if clean.empty:
        raise ValueError("prices is empty after dropping NaN")

    date_str = clean.index[0].date().isoformat()
    n = int(len(clean))
    mean = float(clean.mean())
    median = float(clean.median())
    std = float(clean.std(ddof=1)) if n > 1 else 0.0
    idx_min = clean.idxmin()
    idx_max = clean.idxmax()

    cv = std / mean if mean != 0 else float("nan")

    off_peak = _hourly_mean(clean, OFF_PEAK_HOURS)
    mid_day = _hourly_mean(clean, MID_DAY_HOURS)
    peak = _hourly_mean(clean, PEAK_HOURS)
    ratio = peak / off_peak if off_peak and off_peak != 0 else float("nan")

    return DailyStats(
        market=market,
        date=date_str,
        n_hours=n,
        mean=round(mean, 3),
        median=round(median, 3),
        std=round(std, 3),
        min_price=round(float(clean.min()), 3),
        min_hour=int(idx_min.hour),
        max_price=round(float(clean.max()), 3),
        max_hour=int(idx_max.hour),
        coefficient_of_variation=round(cv, 4) if not np.isnan(cv) else float("nan"),
        off_peak_avg=round(off_peak, 3) if not np.isnan(off_peak) else float("nan"),
        mid_day_avg=round(mid_day, 3) if not np.isnan(mid_day) else float("nan"),
        peak_avg=round(peak, 3) if not np.isnan(peak) else float("nan"),
        peak_to_offpeak_ratio=round(ratio, 3) if not np.isnan(ratio) else float("nan"),
    )


def _hourly_mean(series: pd.Series, hours: list[int]) -> float:
    """Mean of the values at the given hours-of-day."""
    mask = series.index.hour.isin(hours)
    subset = series[mask]
    return float(subset.mean()) if len(subset) else float("nan")
