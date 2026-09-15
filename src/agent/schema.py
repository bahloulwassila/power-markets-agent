"""JSON schema for the Claude research brief.

We use a JSON schema (rather than free-form text) for three reasons:
1. The brief is machine-parseable — the Streamlit dashboard and the
   markdown writer both consume the same structure.
2. Claude is forced to produce every section — no more "the model forgot
   the news section this time".
3. Malformed output can be detected and retried automatically.
"""
from __future__ import annotations


BRIEF_SCHEMA: dict = {
    "name": "power_markets_brief",
    "description": (
        "Daily research brief on Alberta (AESO) and Ontario (IESO) "
        "wholesale electricity markets."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "tldr",
            "aeso_section",
            "ieso_section",
            "spread_section",
            "news_highlights",
            "watchlist",
        ],
        "properties": {
            "tldr": {
                "type": "string",
                "description": (
                    "One to two sentence executive summary. Must mention "
                    "each market's directional move and any material anomaly."
                ),
                "minLength": 40,
                "maxLength": 400,
            },
            "aeso_section": {
                "type": "object",
                "required": ["headline", "bullets"],
                "properties": {
                    "headline": {
                        "type": "string",
                        "description": "One-line summary of Alberta's day.",
                    },
                    "bullets": {
                        "type": "array",
                        "description": (
                            "2-4 bullets. Each cites at least one number "
                            "from the provided stats. No made-up figures."
                        ),
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4,
                    },
                },
            },
            "ieso_section": {
                "type": "object",
                "required": ["headline", "bullets"],
                "properties": {
                    "headline": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4,
                    },
                },
            },
            "spread_section": {
                "type": "object",
                "required": ["headline", "bullets"],
                "properties": {
                    "headline": {"type": "string"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                },
            },
            "news_highlights": {
                "type": "array",
                "description": (
                    "Up to 3 news items with a one-line note on why each "
                    "matters for tomorrow's session."
                ),
                "items": {
                    "type": "object",
                    "required": ["title", "url", "source", "why_it_matters"],
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "source": {"type": "string"},
                        "why_it_matters": {
                            "type": "string",
                            "description": (
                                "One sentence tying the news to tomorrow's "
                                "market. Empty string if no clear link."
                            ),
                        },
                    },
                },
                "maxItems": 3,
            },
            "watchlist": {
                "type": "array",
                "description": (
                    "2-4 items to watch for tomorrow's session — events, "
                    "data releases, or price levels."
                ),
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
            },
        },
    },
}


REQUIRED_TOP_LEVEL_KEYS = {
    "tldr",
    "aeso_section",
    "ieso_section",
    "spread_section",
    "news_highlights",
    "watchlist",
}


def validate_brief_payload(payload: dict) -> list[str]:
    """Light client-side validation returning a list of error messages.

    We rely on Claude's tool-use / structured-output enforcement for the
    heavy lifting; this is a safety net for the shape we care about.
    Returns an empty list if the payload is valid.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload is not a dict"]

    missing = REQUIRED_TOP_LEVEL_KEYS - set(payload.keys())
    if missing:
        errors.append(f"missing required keys: {sorted(missing)}")

    tldr = payload.get("tldr", "")
    if not isinstance(tldr, str) or len(tldr) < 20:
        errors.append("tldr must be a non-trivial string")

    for section_key in ("aeso_section", "ieso_section", "spread_section"):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            errors.append(f"{section_key} must be a dict")
            continue
        if "headline" not in section:
            errors.append(f"{section_key}.headline missing")
        bullets = section.get("bullets")
        if not isinstance(bullets, list) or not bullets:
            errors.append(f"{section_key}.bullets must be a non-empty list")

    watchlist = payload.get("watchlist")
    if not isinstance(watchlist, list) or not watchlist:
        errors.append("watchlist must be a non-empty list")

    news = payload.get("news_highlights")
    if not isinstance(news, list):
        errors.append("news_highlights must be a list (possibly empty)")

    return errors
