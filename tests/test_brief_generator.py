"""Tests for the Claude brief generator, using a fake Anthropic client."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.brief_generator import (
    Brief,
    BriefGenerator,
    BriefGenerationError,
)
from src.analysis.report_data import build_report_data


def _make_report():
    """Build a minimal ReportData for tests."""
    idx = pd.date_range(start="2024-06-01", periods=24, freq="h")
    aeso = pd.Series([50.0] * 24, index=idx)
    return build_report_data(
        target_date=date(2024, 6, 1),
        aeso_target_day=aeso,
    )


def _valid_payload() -> dict:
    return {
        "tldr": "Alberta traded flat around $50 with no anomalies observed.",
        "aeso_section": {
            "headline": "AESO stable.",
            "bullets": ["Mean $50/MWh.", "No spikes."],
        },
        "ieso_section": {
            "headline": "IESO data unavailable.",
            "bullets": ["No target day.", "No anomalies computed."],
        },
        "spread_section": {
            "headline": "Spread not computed.",
            "bullets": ["Missing IESO data."],
        },
        "news_highlights": [],
        "watchlist": ["Monitor for evening ramp", "Watch morning wind forecast"],
    }


def _fake_tool_use_response(payload: dict):
    """Build a fake Anthropic Message response with one tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "power_markets_brief"
    block.input = payload

    response = MagicMock()
    response.content = [block]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 200
    return response


class TestBriefGeneratorInit:

    @patch("src.agent.brief_generator.ANTHROPIC_API_KEY", "")
    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is missing"):
            BriefGenerator(api_key="")

    def test_accepts_injected_client(self):
        gen = BriefGenerator(_client=MagicMock())
        assert gen._client is not None


class TestBriefGeneratorHappyPath:

    def test_generates_brief_on_valid_response(self):
        fake_client = MagicMock()
        fake_client.messages.create.return_value = _fake_tool_use_response(
            _valid_payload()
        )
        gen = BriefGenerator(_client=fake_client)

        brief = gen.generate(_make_report())

        assert isinstance(brief, Brief)
        assert brief.report_date == "2024-06-01"
        assert brief.payload["tldr"].startswith("Alberta")
        assert brief.input_tokens == 100
        assert brief.output_tokens == 200
        # Only one API call needed
        assert fake_client.messages.create.call_count == 1


class TestBriefGeneratorRetry:

    def test_retries_on_invalid_payload_then_succeeds(self):
        # First response: missing 'watchlist' → validation fails
        bad_payload = _valid_payload()
        del bad_payload["watchlist"]
        good_response = _fake_tool_use_response(_valid_payload())
        bad_response = _fake_tool_use_response(bad_payload)

        fake_client = MagicMock()
        fake_client.messages.create.side_effect = [bad_response, good_response]

        gen = BriefGenerator(_client=fake_client, max_retries=1)
        brief = gen.generate(_make_report())

        assert isinstance(brief, Brief)
        assert fake_client.messages.create.call_count == 2

    def test_raises_after_all_retries_exhausted(self):
        bad_payload = _valid_payload()
        del bad_payload["watchlist"]
        bad_response = _fake_tool_use_response(bad_payload)

        fake_client = MagicMock()
        fake_client.messages.create.return_value = bad_response

        gen = BriefGenerator(_client=fake_client, max_retries=1)
        with pytest.raises(BriefGenerationError, match="failed to produce"):
            gen.generate(_make_report())

        # 1 initial attempt + 1 retry = 2 calls
        assert fake_client.messages.create.call_count == 2

    def test_raises_if_claude_ignores_tool(self):
        # Response with no tool_use block at all
        text_block = MagicMock()
        text_block.type = "text"
        response = MagicMock()
        response.content = [text_block]
        response.usage.input_tokens = 50
        response.usage.output_tokens = 20

        fake_client = MagicMock()
        fake_client.messages.create.return_value = response

        gen = BriefGenerator(_client=fake_client, max_retries=0)
        with pytest.raises(BriefGenerationError):
            gen.generate(_make_report())


class TestBriefSerialization:

    def test_to_dict_is_json_ready(self):
        fake_client = MagicMock()
        fake_client.messages.create.return_value = _fake_tool_use_response(
            _valid_payload()
        )
        gen = BriefGenerator(_client=fake_client)
        brief = gen.generate(_make_report())

        import json
        json.dumps(brief.to_dict())  # should not raise
