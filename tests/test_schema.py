"""Tests for the brief JSON schema validation."""
from __future__ import annotations

import pytest

from src.agent.schema import (
    BRIEF_SCHEMA,
    REQUIRED_TOP_LEVEL_KEYS,
    validate_brief_payload,
)


def _valid_payload() -> dict:
    """A minimal payload that should pass validation."""
    return {
        "tldr": "Alberta traded around $52; Ontario firmed to $38. No major anomalies.",
        "aeso_section": {
            "headline": "AESO in line with baseline.",
            "bullets": [
                "Mean price $52.14/MWh, close to 30-day avg.",
                "Peak/off-peak ratio 2.81.",
            ],
        },
        "ieso_section": {
            "headline": "OEMP firmed 8% vs baseline.",
            "bullets": [
                "Mean price $38.07/MWh.",
                "No anomalies flagged (max |z|=1.6).",
            ],
        },
        "spread_section": {
            "headline": "Spread widened at hour 18.",
            "bullets": ["Mean spread +$14/MWh, correlation 0.62."],
        },
        "news_highlights": [
            {
                "title": "Genesee 2 outage",
                "url": "http://example.com/1",
                "source": "AESO",
                "why_it_matters": "May pressure AB tomorrow AM.",
            },
        ],
        "watchlist": [
            "Genesee 2 return-to-service",
            "Ontario capacity auction Friday",
        ],
    }


class TestSchemaStructure:

    def test_schema_has_a_name(self):
        assert BRIEF_SCHEMA["name"] == "power_markets_brief"

    def test_schema_has_all_required_top_level_keys(self):
        properties = BRIEF_SCHEMA["input_schema"]["properties"]
        for key in REQUIRED_TOP_LEVEL_KEYS:
            assert key in properties, f"missing property: {key}"

    def test_required_list_matches_top_level_keys(self):
        assert set(BRIEF_SCHEMA["input_schema"]["required"]) == REQUIRED_TOP_LEVEL_KEYS


class TestValidateBriefPayload:

    def test_valid_payload_passes(self):
        errors = validate_brief_payload(_valid_payload())
        assert errors == []

    def test_non_dict_fails(self):
        assert validate_brief_payload("not a dict") != []
        assert validate_brief_payload(None) != []
        assert validate_brief_payload([]) != []

    def test_missing_keys_reported(self):
        payload = _valid_payload()
        del payload["tldr"]
        del payload["watchlist"]
        errors = validate_brief_payload(payload)
        assert any("tldr" in e for e in errors)
        assert any("watchlist" in e for e in errors)

    def test_short_tldr_fails(self):
        payload = _valid_payload()
        payload["tldr"] = "too short"
        errors = validate_brief_payload(payload)
        assert any("tldr" in e for e in errors)

    def test_section_without_bullets_fails(self):
        payload = _valid_payload()
        payload["aeso_section"]["bullets"] = []
        errors = validate_brief_payload(payload)
        assert any("aeso_section" in e for e in errors)

    def test_empty_watchlist_fails(self):
        payload = _valid_payload()
        payload["watchlist"] = []
        errors = validate_brief_payload(payload)
        assert any("watchlist" in e for e in errors)

    def test_empty_news_ok(self):
        # news_highlights can be an empty list — days with no relevant news
        payload = _valid_payload()
        payload["news_highlights"] = []
        assert validate_brief_payload(payload) == []
