"""Aggregate all analytical outputs into a single object for the Claude brief.

This module is the boundary between the deterministic analytics layer
(Phase 2) and the LLM synthesis layer (Phase 3). Everything Claude reasons
about is contained in ReportData — the LLM never recomputes numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any

import pandas as pd

from src.analysis.statistics import DailyStats, compute_daily_stats
from src.analysis.anomaly import Anomaly, detect_anomalies, anomaly_summary
from src.analysis.spread import SpreadStats, compute_spread


@dataclass(frozen=True)
class ReportData:
    """Everything the LLM sees for one day."""
    report_date: str                    # ISO YYYY-MM-DD
    aeso_stats: dict | None             # DailyStats.to_dict() or None
    ieso_stats: dict | None
    aeso_anomalies: list[dict] = field(default_factory=list)
    ieso_anomalies: list[dict] = field(default_factory=list)
    aeso_anomaly_summary: dict = field(default_factory=dict)
    ieso_anomaly_summary: dict = field(default_factory=dict)
    spread_stats: dict | None = None    # SpreadStats.to_dict()
    news_items: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # e.g. "IESO data missing"

    def to_dict(self) -> dict:
        return asdict(self)


def build_report_data(
    target_date: date,
    *,
    aeso_target_day: pd.Series | None = None,
    aeso_baseline: pd.Series | None = None,
    ieso_target_day: pd.Series | None = None,
    ieso_baseline: pd.Series | None = None,
    news_items: list[dict] | None = None,
) -> ReportData:
    """Assemble the day's ReportData from raw hourly series.

    Every input is optional so the report gracefully degrades: if IESO
    data is missing, we still produce the AESO section and add a warning.
    """
    warnings: list[str] = []

    # -------- AESO block --------
    aeso_stats_dict: dict | None = None
    aeso_anom: list[Anomaly] = []
    aeso_anom_sum: dict = {}
    if aeso_target_day is not None and not aeso_target_day.dropna().empty:
        aeso_stats_dict = compute_daily_stats(aeso_target_day, market="AESO").to_dict()
        if aeso_baseline is not None and not aeso_baseline.dropna().empty:
            aeso_anom = detect_anomalies(aeso_target_day, aeso_baseline)
        else:
            warnings.append("AESO baseline missing — no anomaly detection")
        aeso_anom_sum = anomaly_summary(aeso_anom)
    else:
        warnings.append("AESO target-day data missing")

    # -------- IESO block --------
    ieso_stats_dict: dict | None = None
    ieso_anom: list[Anomaly] = []
    ieso_anom_sum: dict = {}
    if ieso_target_day is not None and not ieso_target_day.dropna().empty:
        ieso_stats_dict = compute_daily_stats(ieso_target_day, market="IESO").to_dict()
        if ieso_baseline is not None and not ieso_baseline.dropna().empty:
            ieso_anom = detect_anomalies(ieso_target_day, ieso_baseline)
        else:
            warnings.append("IESO baseline missing — no anomaly detection")
        ieso_anom_sum = anomaly_summary(ieso_anom)
    else:
        warnings.append("IESO target-day data missing")

    # -------- Spread block --------
    spread_dict: dict | None = None
    if (aeso_target_day is not None
            and ieso_target_day is not None
            and not aeso_target_day.dropna().empty
            and not ieso_target_day.dropna().empty):
        try:
            spread_dict = compute_spread(aeso_target_day, ieso_target_day).to_dict()
        except ValueError as e:
            warnings.append(f"Spread not computed: {e}")

    return ReportData(
        report_date=target_date.isoformat(),
        aeso_stats=aeso_stats_dict,
        ieso_stats=ieso_stats_dict,
        aeso_anomalies=[a.to_dict() for a in aeso_anom],
        ieso_anomalies=[a.to_dict() for a in ieso_anom],
        aeso_anomaly_summary=aeso_anom_sum,
        ieso_anomaly_summary=ieso_anom_sum,
        spread_stats=spread_dict,
        news_items=news_items or [],
        warnings=warnings,
    )
