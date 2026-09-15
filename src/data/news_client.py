"""News aggregator for North American power markets.

Uses RSS feeds by default (no API key needed). Falls back to NewsAPI if a key
is configured. Includes a lightweight relevance filter (keyword-based) to
avoid dumping every energy story into the brief.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta

import feedparser
import requests

from src.config import NEWS_RSS_FEEDS, NEWSAPI_KEY

logger = logging.getLogger(__name__)


# Keywords used to score relevance of a news item to power markets.
_RELEVANCE_KEYWORDS = {
    # Markets / operators
    "aeso", "ieso", "alberta electric", "ontario electricity", "power pool",
    "wholesale electricity", "pool price", "hoep", "oemp", "spot price",
    # Grid / generation
    "grid", "outage", "curtailment", "capacity market", "reserve margin",
    "wind generation", "solar generation", "nuclear", "natural gas",
    "coal phase", "hydro", "battery storage",
    # Policy / demand
    "carbon price", "carbon tax", "cbam", "emissions", "clean electricity",
    "regulation", "tariff", "demand response",
    # Regions
    "alberta", "ontario", "quebec", "canada energy",
}


@dataclass(frozen=True)
class NewsItem:
    title: str
    summary: str
    url: str
    source: str
    published_utc: datetime
    relevance_score: float  # 0..1, higher = more relevant to power markets

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published_utc"] = self.published_utc.isoformat()
        return d


class NewsClient:
    """Aggregates news items from RSS feeds and (optionally) NewsAPI.

    Two-stage pipeline:
        1. Fetch — pull raw entries from all sources
        2. Filter/score — keep items with relevance >= min_relevance
           and published within the look-back window
    """

    def __init__(
        self,
        *,
        use_newsapi: bool = True,
        min_relevance: float = 0.15,
        lookback_hours: int = 24,
    ) -> None:
        self.use_newsapi = use_newsapi and bool(NEWSAPI_KEY)
        self.min_relevance = min_relevance
        self.lookback_hours = lookback_hours

    def fetch_relevant_news(self, *, max_items: int = 20) -> list[NewsItem]:
        """Return top-N most relevant recent news items."""
        raw: list[NewsItem] = []
        raw.extend(self._fetch_rss_all())
        if self.use_newsapi:
            try:
                raw.extend(self._fetch_newsapi())
            except Exception as e:
                logger.warning("NewsAPI fetch failed, RSS only: %s", e)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        recent = [n for n in raw if n.published_utc >= cutoff]
        relevant = [n for n in recent if n.relevance_score >= self.min_relevance]

        seen: set[str] = set()
        deduped: list[NewsItem] = []
        for n in sorted(relevant, key=lambda x: x.relevance_score, reverse=True):
            if n.url in seen:
                continue
            seen.add(n.url)
            deduped.append(n)

        return deduped[:max_items]

    def _fetch_rss_all(self) -> list[NewsItem]:
        items: list[NewsItem] = []
        for source_name, url in NEWS_RSS_FEEDS:
            try:
                items.extend(self._fetch_rss_one(source_name, url))
            except Exception as e:
                logger.warning("RSS %s failed: %s", source_name, e)
        return items

    def _fetch_rss_one(self, source: str, url: str) -> list[NewsItem]:
        feed = feedparser.parse(url)
        out: list[NewsItem] = []
        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            link = getattr(entry, "link", "") or ""
            published = _parse_feed_datetime(entry)
            if not (title and link):
                continue
            score = _relevance_score(f"{title} {summary}")
            out.append(NewsItem(
                title=title.strip(),
                summary=_strip_html(summary).strip()[:500],
                url=link,
                source=source,
                published_utc=published,
                relevance_score=score,
            ))
        return out

    def _fetch_newsapi(self) -> list[NewsItem]:
        if not NEWSAPI_KEY:
            return []
        params = {
            "q": '("electricity market" OR AESO OR IESO OR "pool price" OR '
                 '"Ontario Price" OR "power grid")',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 30,
            "from": (datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours))
                    .strftime("%Y-%m-%dT%H:%M:%S"),
        }
        headers = {"X-Api-Key": NEWSAPI_KEY}
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params=params, headers=headers, timeout=15,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])
        out: list[NewsItem] = []
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            score = _relevance_score(f"{title} {desc}")
            try:
                pub = datetime.fromisoformat(
                    a["publishedAt"].replace("Z", "+00:00"),
                )
            except Exception:
                pub = datetime.now(timezone.utc)
            out.append(NewsItem(
                title=title.strip(),
                summary=(desc or "").strip()[:500],
                url=a.get("url", ""),
                source=(a.get("source") or {}).get("name", "NewsAPI"),
                published_utc=pub,
                relevance_score=score,
            ))
        return out


def _relevance_score(text: str) -> float:
    """Crude keyword-hit relevance score in [0, 1].

    Deliberately simple: for v1, a hand-curated keyword list is more
    predictable than an embedding model. Swap for semantic similarity in v2.
    """
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in t)
    return min(1.0, hits / 5.0)


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def _parse_feed_datetime(entry) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        val = getattr(entry, key, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)
