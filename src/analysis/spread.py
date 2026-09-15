"""Inter-market spread analysis: AESO Pool Price vs IESO Ontario Price.

Given two hourly price series (Alberta and Ontario), compute their spread
statistics: mean gap, correlation, and hours where the gap is unusually
wide. Useful signal for a trader watching cross-market dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.config import ANALYSIS


@dataclass(frozen=True)
class SpreadStats:
    """AESO minus IESO hourly spread, summarized over a day."""
    date: str
    n_matched_hours: int          # hours present in both series
    mean_spread: float             # avg(AESO - IESO), $/MWh
    median_spread: float
    std_spread: float
    min_spread: float
    max_spread: float
    correlation: float             # Pearson correlation of the two series
    n_wide_gaps: int               # hours where |spread| exceeds threshold
    widest_gap_hour: int | None
    widest_gap_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_spread(
    aeso: pd.Series,
    ieso: pd.Series,
    *,
    wide_gap_threshold: float | None = None,
) -> SpreadStats:
    """Compute spread statistics between AESO and IESO price series.

    Parameters
    ----------
    aeso, ieso : pd.Series
        Hourly price series with DatetimeIndex. They are inner-joined on
        their timestamps — hours present in only one series are dropped.
    wide_gap_threshold : float, optional
        Absolute spread ($/MWh) above which an hour is counted as a "wide
        gap". Defaults to the config value (50.0 $/MWh).

    Returns
    -------
    SpreadStats
        Summary statistics on the spread series.
    """
    _validate_series(aeso, "aeso")
    _validate_series(ieso, "ieso")

    threshold = (
        wide_gap_threshold
        if wide_gap_threshold is not None
        else ANALYSIS.inter_market_spread_threshold
    )

    # Align on shared timestamps
    df = pd.concat([aeso.rename("aeso"), ieso.rename("ieso")], axis=1).dropna()
    if df.empty:
        raise ValueError("no overlapping hours between AESO and IESO series")

    df["spread"] = df["aeso"] - df["ieso"]

    date_str = df.index[0].date().isoformat()
    n = int(len(df))

    corr_raw = df["aeso"].corr(df["ieso"])
    correlation = float(corr_raw) if not np.isnan(corr_raw) else 0.0

    abs_spread = df["spread"].abs()
    wide_mask = abs_spread > threshold
    n_wide = int(wide_mask.sum())

    idx_widest = abs_spread.idxmax()
    widest_hour = int(idx_widest.hour)
    widest_value = float(df.loc[idx_widest, "spread"])

    return SpreadStats(
        date=date_str,
        n_matched_hours=n,
        mean_spread=round(float(df["spread"].mean()), 3),
        median_spread=round(float(df["spread"].median()), 3),
        std_spread=round(float(df["spread"].std(ddof=1)), 3) if n > 1 else 0.0,
        min_spread=round(float(df["spread"].min()), 3),
        max_spread=round(float(df["spread"].max()), 3),
        correlation=round(correlation, 4),
        n_wide_gaps=n_wide,
        widest_gap_hour=widest_hour,
        widest_gap_value=round(widest_value, 3),
    )


def _validate_series(s: pd.Series, name: str) -> None:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be a DatetimeIndex")
