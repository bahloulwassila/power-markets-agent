"""IESO (Independent Electricity System Operator) client for Ontario prices.

IMPORTANT — Market Renewal Program (May 2025):
    The Hourly Ontario Energy Price (HOEP) was retired on 30 April 2025 and
    replaced by the Ontario Price (OEMP), computed as:

        OEMP = DA-OZP + LFDA

    where DA-OZP is the Day-Ahead Ontario Zonal Price and LFDA is the
    Load Forecast Deviation Adjustment. See:

        https://www.ieso.ca/power-data/Price-Overview/Ontario-Market-Prices

IESO publishes daily / hourly CSV reports on a public reports server. No
API key is required — we just download the file for the requested date.
"""
from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytz
import requests

from src.config import ENDPOINTS, CACHE_DIR, IESO_TZ

logger = logging.getLogger(__name__)


HOEP_RETIREMENT_DATE = date(2025, 5, 1)


# ---------- Domain object ----------
@dataclass(frozen=True)
class OntarioPricePoint:
    """A single hourly Ontario wholesale price observation."""
    delivery_date: date
    hour: int
    begin_datetime_local: datetime
    price: float
    price_type: str

    def to_dict(self) -> dict:
        return {
            "delivery_date": self.delivery_date.isoformat(),
            "hour": self.hour,
            "begin_datetime_local": self.begin_datetime_local.isoformat(),
            "price": self.price,
            "price_type": self.price_type,
        }


# ---------- Client ----------
class IESOClient:
    """Client for IESO's public reports server (no API key required)."""

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.5,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._tz = pytz.timezone(IESO_TZ)
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "text/csv, text/xml, */*",
            "User-Agent": "PowerMarketsAgent/0.1",
        })

    def get_ontario_prices(
        self,
        target_date: date,
        *,
        use_cache: bool = True,
    ) -> list[OntarioPricePoint]:
        """Fetch hourly Ontario Price data for a single delivery date."""
        cache_path = self._cache_path(target_date)
        if use_cache and cache_path.exists() and self._cache_is_fresh(cache_path, target_date):
            logger.info("IESO cache hit: %s", cache_path.name)
            return self._load_cache(cache_path)

        if target_date >= HOEP_RETIREMENT_DATE:
            points = self._fetch_oemp(target_date)
        else:
            points = self._fetch_hoep(target_date)

        if use_cache:
            self._save_cache(cache_path, points)
        return points

    def get_range(
        self,
        start_date: date,
        end_date: date,
        *,
        use_cache: bool = True,
    ) -> list[OntarioPricePoint]:
        """Fetch across a date range (fetches one day at a time)."""
        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")
        out: list[OntarioPricePoint] = []
        d = start_date
        while d <= end_date:
            try:
                out.extend(self.get_ontario_prices(d, use_cache=use_cache))
            except Exception as e:
                logger.warning("Skipping %s: %s", d, e)
            d += timedelta(days=1)
        return out

    def to_dataframe(self, points: Iterable[OntarioPricePoint]) -> pd.DataFrame:
        if not points:
            return pd.DataFrame(columns=["price", "price_type", "hour"])
        df = pd.DataFrame([p.to_dict() for p in points])
        df["begin_datetime_local"] = pd.to_datetime(df["begin_datetime_local"])
        df = df.set_index("begin_datetime_local").sort_index()
        return df[["price", "price_type", "hour"]]

    def _fetch_oemp(self, target_date: date) -> list[OntarioPricePoint]:
        """OEMP = DA-OZP + LFDA (post-May-2025 market)."""
        yyyymmdd = target_date.strftime("%Y%m%d")
        url = (
            f"{ENDPOINTS.ieso_reports_base}"
            f"{ENDPOINTS.ieso_realtime_market_price}"
            f"/PUB_RealtimeMktPrice_{yyyymmdd}.csv"
        )
        csv_text = self._get_text(url)
        if not csv_text:
            return []
        return self._parse_oemp_csv(csv_text, target_date)

    def _parse_oemp_csv(self, text: str, target_date: date) -> list[OntarioPricePoint]:
        """Parse Real-Time Market Price CSV, aggregating 5-min intervals to hourly."""
        reader = csv.reader(io.StringIO(text))
        rows = [r for r in reader if r]

        header_idx = None
        for i, r in enumerate(rows):
            if r and r[0].strip().upper().startswith("DELIVERY"):
                header_idx = i
                break
        if header_idx is None:
            logger.warning("IESO CSV: header not found for %s", target_date)
            return []

        header = [h.strip().upper() for h in rows[header_idx]]
        try:
            i_hour = header.index("DELIVERY_HOUR")
            i_price = next(
                i for i, h in enumerate(header)
                if h in ("PRICE", "HOEP", "DA_OZP", "OZP", "OEMP")
            )
        except (ValueError, StopIteration):
            logger.warning("IESO CSV: expected columns not found: %s", header)
            return []

        buckets: dict[int, list[float]] = {}
        for r in rows[header_idx + 1:]:
            if len(r) <= max(i_hour, i_price):
                continue
            try:
                h = int(r[i_hour])
                p = float(r[i_price])
                buckets.setdefault(h, []).append(p)
            except (ValueError, TypeError):
                continue

        points: list[OntarioPricePoint] = []
        for h, prices in sorted(buckets.items()):
            hourly = sum(prices) / len(prices)
            local_dt = self._tz.localize(
                datetime.combine(target_date, datetime.min.time())
                + timedelta(hours=h - 1),
                is_dst=None,
            )
            points.append(OntarioPricePoint(
                delivery_date=target_date,
                hour=h,
                begin_datetime_local=local_dt,
                price=hourly,
                price_type="OEMP",
            ))
        return points

    def _fetch_hoep(self, target_date: date) -> list[OntarioPricePoint]:
        """Legacy HOEP report (pre-May-2025)."""
        yyyymmdd = target_date.strftime("%Y%m%d")
        url = (
            f"{ENDPOINTS.ieso_reports_base}/DispUnconsHOEP/"
            f"PUB_DispUnconsHOEP_{yyyymmdd}.csv"
        )
        csv_text = self._get_text(url)
        if not csv_text:
            return []
        return self._parse_hoep_daily(csv_text, target_date)

    def _parse_hoep_daily(self, text: str, target_date: date) -> list[OntarioPricePoint]:
        reader = csv.reader(io.StringIO(text))
        rows = [r for r in reader if r]
        header_idx = next(
            (i for i, r in enumerate(rows) if r and "HOUR" in r[0].upper()),
            None,
        )
        if header_idx is None:
            return []
        points: list[OntarioPricePoint] = []
        for r in rows[header_idx + 1:]:
            try:
                h = int(r[0])
                p = float(r[1])
                local_dt = self._tz.localize(
                    datetime.combine(target_date, datetime.min.time())
                    + timedelta(hours=h - 1),
                    is_dst=None,
                )
                points.append(OntarioPricePoint(
                    delivery_date=target_date, hour=h,
                    begin_datetime_local=local_dt, price=p, price_type="HOEP",
                ))
            except (ValueError, IndexError):
                continue
        return points

    def _get_text(self, url: str) -> str | None:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.get(url, timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.text
                if r.status_code == 404:
                    logger.info("IESO 404 (not yet published?): %s", url)
                    return None
                if r.status_code in (429, 502, 503, 504):
                    wait = self.backoff_base_s ** attempt
                    time.sleep(wait)
                    continue
                r.raise_for_status()
            except requests.RequestException as e:
                last_exc = e
                time.sleep(self.backoff_base_s ** attempt)
        logger.warning("IESO fetch failed for %s: %s", url, last_exc)
        return None

    def _cache_path(self, target_date: date) -> Path:
        return CACHE_DIR / f"ieso_{target_date}.json"

    def _cache_is_fresh(self, path: Path, target_date: date) -> bool:
        if target_date < date.today():
            return True
        return (time.time() - path.stat().st_mtime) < 3600

    def _save_cache(self, path: Path, points: list[OntarioPricePoint]) -> None:
        import json
        with path.open("w") as f:
            json.dump([p.to_dict() for p in points], f, indent=2)

    def _load_cache(self, path: Path) -> list[OntarioPricePoint]:
        import json
        with path.open() as f:
            records = json.load(f)
        points = []
        for r in records:
            points.append(OntarioPricePoint(
                delivery_date=date.fromisoformat(r["delivery_date"]),
                hour=r["hour"],
                begin_datetime_local=datetime.fromisoformat(r["begin_datetime_local"]),
                price=r["price"],
                price_type=r["price_type"],
            ))
        return points


def fetch_last_n_days(n: int = 7) -> pd.DataFrame:
    """Fetch the last N days of Ontario prices as a DataFrame."""
    client = IESOClient()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=n - 1)
    points = client.get_range(start, end)
    return client.to_dataframe(points)
