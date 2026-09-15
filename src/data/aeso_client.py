"""AESO (Alberta Electric System Operator) API client.

Fetches hourly Pool Price data from AESO's public API. A free API key is
required and can be obtained at:

    https://apim-aeso-connect.developer.azure-api.net/

The Pool Price is the settlement price for one MWh of electric energy
exchanged through the Alberta Interconnected Electric System (AIES). It
is computed hourly as the time-weighted average of the 1-minute System
Marginal Prices for that hour.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytz
import requests

from src.config import AESO_API_KEY, ENDPOINTS, CACHE_DIR, AESO_TZ

logger = logging.getLogger(__name__)


# ---------- Domain object ----------
@dataclass(frozen=True)
class PoolPricePoint:
    """A single hourly observation of the AESO Pool Price."""
    begin_datetime_utc: datetime
    begin_datetime_local: datetime
    pool_price: float
    forecast_pool_price: float | None
    rolling_30day_avg: float | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["begin_datetime_utc"] = self.begin_datetime_utc.isoformat()
        d["begin_datetime_local"] = self.begin_datetime_local.isoformat()
        return d


# ---------- Client ----------
class AESOClient:
    """Thin, typed HTTP client for the AESO Pool Price API.

    The client is deliberately minimal: it handles auth, retries, timezone
    conversion, and returns typed dataclasses instead of raw dicts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.5,
    ) -> None:
        self.api_key = api_key or AESO_API_KEY
        if not self.api_key:
            raise ValueError(
                "AESO_API_KEY is missing. Register for a free key at "
                "https://apim-aeso-connect.developer.azure-api.net/ and "
                "add it to your .env file."
            )
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._tz = pytz.timezone(AESO_TZ)
        self._session = requests.Session()
        self._session.headers.update({
            "API-KEY": self.api_key,
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "PowerMarketsAgent/0.1",
        })

    def get_pool_prices(
        self,
        start_date: date,
        end_date: date | None = None,
        *,
        use_cache: bool = True,
    ) -> list[PoolPricePoint]:
        """Fetch hourly Pool Price observations for a date range."""
        end_date = end_date or start_date
        if end_date < start_date:
            raise ValueError("end_date must be >= start_date")

        cache_path = self._cache_path(start_date, end_date)
        if use_cache and cache_path.exists() and self._cache_is_fresh(cache_path, end_date):
            logger.info("AESO cache hit: %s", cache_path.name)
            return self._load_cache(cache_path)

        raw = self._fetch_with_retry(start_date, end_date)
        points = self._parse(raw)

        if use_cache:
            self._save_cache(cache_path, points)

        return points

    def to_dataframe(self, points: Iterable[PoolPricePoint]) -> pd.DataFrame:
        """Convert points to a tidy DataFrame indexed on local time."""
        if not points:
            return pd.DataFrame(columns=[
                "pool_price", "forecast_pool_price", "rolling_30day_avg"
            ])
        df = pd.DataFrame([p.to_dict() for p in points])
        df["begin_datetime_local"] = pd.to_datetime(df["begin_datetime_local"])
        df = df.set_index("begin_datetime_local").sort_index()
        return df[["pool_price", "forecast_pool_price", "rolling_30day_avg"]]

    def _fetch_with_retry(self, start_date: date, end_date: date) -> dict:
        url = ENDPOINTS.aeso_base + ENDPOINTS.aeso_pool_price
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        }

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._session.get(url, params=params, timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429, 502, 503, 504):
                    wait = self.backoff_base_s ** attempt
                    logger.warning(
                        "AESO transient %s, retry %d/%d in %.1fs",
                        r.status_code, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                r.raise_for_status()
            except requests.RequestException as e:
                last_exc = e
                wait = self.backoff_base_s ** attempt
                logger.warning(
                    "AESO request failed (%s), retry %d/%d in %.1fs",
                    e, attempt, self.max_retries, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"AESO API failed after {self.max_retries} attempts"
        ) from last_exc

    def _parse(self, raw: dict) -> list[PoolPricePoint]:
        """Convert AESO's JSON to PoolPricePoint objects."""
        records = (
            raw.get("return", {}).get("Pool Price Report")
            or raw.get("return", {}).get("Pool Price")
            or raw.get("data", [])
            or []
        )

        points: list[PoolPricePoint] = []
        for r in records:
            try:
                utc_str = r.get("begin_datetime_utc") or r.get("beginDatetimeUtc")
                local_str = (
                    r.get("begin_datetime_mpt")
                    or r.get("beginDatetimeMpt")
                    or r.get("begin_datetime_local")
                )
                utc_dt = self._parse_datetime(utc_str, tz=pytz.UTC)
                local_dt = self._parse_datetime(local_str, tz=self._tz)

                points.append(PoolPricePoint(
                    begin_datetime_utc=utc_dt,
                    begin_datetime_local=local_dt,
                    pool_price=float(r.get("pool_price", 0.0)),
                    forecast_pool_price=_maybe_float(r.get("forecast_pool_price")),
                    rolling_30day_avg=_maybe_float(r.get("rolling_30day_avg")),
                ))
            except (ValueError, TypeError) as e:
                logger.warning("Skipping malformed AESO record: %s (%s)", r, e)

        return sorted(points, key=lambda p: p.begin_datetime_utc)

    @staticmethod
    def _parse_datetime(s: str | None, tz) -> datetime:
        if not s:
            raise ValueError("empty datetime string")
        dt = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")
        if tz is pytz.UTC:
            return pytz.UTC.localize(dt)
        return tz.localize(dt, is_dst=None) if hasattr(tz, "localize") else dt.replace(tzinfo=tz)

    def _cache_path(self, start_date: date, end_date: date) -> Path:
        return CACHE_DIR / f"aeso_{start_date}_{end_date}.json"

    def _cache_is_fresh(self, path: Path, end_date: date) -> bool:
        if end_date < date.today():
            return True
        age_s = time.time() - path.stat().st_mtime
        return age_s < 3600

    def _save_cache(self, path: Path, points: list[PoolPricePoint]) -> None:
        import json
        with path.open("w") as f:
            json.dump([p.to_dict() for p in points], f, indent=2)

    def _load_cache(self, path: Path) -> list[PoolPricePoint]:
        import json
        with path.open() as f:
            records = json.load(f)
        points = []
        for r in records:
            points.append(PoolPricePoint(
                begin_datetime_utc=datetime.fromisoformat(r["begin_datetime_utc"]),
                begin_datetime_local=datetime.fromisoformat(r["begin_datetime_local"]),
                pool_price=r["pool_price"],
                forecast_pool_price=r["forecast_pool_price"],
                rolling_30day_avg=r["rolling_30day_avg"],
            ))
        return points


def _maybe_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def fetch_last_n_days(n: int = 7, api_key: str | None = None) -> pd.DataFrame:
    """Fetch the last N days of Pool Prices as a DataFrame. Handy for notebooks."""
    client = AESOClient(api_key=api_key)
    end = date.today()
    start = end - timedelta(days=n - 1)
    points = client.get_pool_prices(start, end)
    return client.to_dataframe(points)
