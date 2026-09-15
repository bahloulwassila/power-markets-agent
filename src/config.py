"""Central configuration for the Power Markets Research Agent.

All environment variables, endpoints and tunable parameters live here.
Downstream modules import from this file — they never read env vars
directly. That keeps configuration testable and easy to override.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# -------- Paths --------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
BRIEFS_DIR = PROJECT_ROOT / "briefs"

# Load the .env at project root (safe if the file doesn't exist)
load_dotenv(PROJECT_ROOT / ".env")

# Create expected directories if they don't exist yet
for _dir in (DATA_DIR, CACHE_DIR, BRIEFS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# -------- API keys --------
AESO_API_KEY: str = os.getenv("AESO_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")


# -------- Timezones --------
AESO_TZ = "America/Edmonton"   # Alberta Mountain Time (MPT/MDT)
IESO_TZ = "America/Toronto"    # Ontario Eastern Time (EST/EDT)


# -------- API endpoints --------
@dataclass(frozen=True)
class Endpoints:
    """External API endpoints. Grouped here so a single URL change
    doesn't require hunting through the codebase."""

    # AESO - Azure APIM gateway (all AESO APIs migrated to APIM as of 2025)
    aeso_base: str = "https://apimgw.aeso.ca/public/poolprice-api/v1.1"
    aeso_pool_price: str = "/price/poolPrice"

    # IESO public reports server. No auth required - CSV downloads.
    # Note: HOEP was retired 30 April 2025, replaced by Ontario Price (OEMP).
    ieso_reports_base: str = "https://reports-public.ieso.ca/public"
    ieso_realtime_market_price: str = "/RealtimeMktPrice"


ENDPOINTS = Endpoints()


# -------- Analysis parameters --------
@dataclass(frozen=True)
class AnalysisConfig:
    """Tunable thresholds for the anomaly detection layer (Phase 2)."""

    # An hourly price is flagged as a spike if |z-score| > this value,
    # where z-score is computed against the baseline_lookback_days window.
    zscore_spike_threshold: float = 2.0

    # How many days of historical prices to use as the baseline for
    # z-score / mean / stdev calculations.
    baseline_lookback_days: int = 30

    # The AESO-vs-IESO spread is flagged as unusual above this value ($/MWh).
    inter_market_spread_threshold: float = 50.0


ANALYSIS = AnalysisConfig()


# -------- LLM configuration --------
# Anthropic model used for brief generation (Phase 3)
CLAUDE_MODEL = "claude-sonnet-4-5"
CLAUDE_MAX_TOKENS = 4096


# -------- News sources (Phase 1) --------
# RSS feeds are used first (no API key needed). Extend this list to broaden
# coverage - order does not matter, deduplication is done downstream by URL.
NEWS_RSS_FEEDS: list[tuple[str, str]] = [
    ("Canadian Energy Regulator", "https://www.cer-rec.gc.ca/en/news/rss.xml"),
    ("Reuters - Energy", "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best"),
    ("OilPrice.com", "https://oilprice.com/rss/main"),
]
