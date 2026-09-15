"""Unit tests for the News aggregator."""
from __future__ import annotations

import pytest

from src.data.news_client import _relevance_score, _strip_html


class TestRelevanceScoring:
    """The scorer is deliberately simple — verify the sanity properties."""

    @pytest.mark.parametrize("text,should_score_high", [
        ("AESO pool price spikes to $500/MWh amid Alberta outage", True),
        ("IESO announces new capacity market rules for Ontario", True),
        ("Wind generation curtailment hits record in Alberta", True),
        ("Local hockey team wins championship in overtime", False),
        ("Best pizza recipes for summer", False),
        ("", False),
    ])
    def test_relevance_direction(self, text, should_score_high):
        score = _relevance_score(text)
        if should_score_high:
            assert score >= 0.15, f"Expected high, got {score} for: {text!r}"
        else:
            assert score < 0.15, f"Expected low, got {score} for: {text!r}"

    def test_score_bounded(self):
        # Even a text stuffed with every keyword should not exceed 1.0
        text = " ".join([
            "aeso", "ieso", "pool price", "hoep", "oemp",
            "alberta", "ontario", "wind generation", "grid outage",
        ])
        assert 0.0 <= _relevance_score(text) <= 1.0


class TestHTMLStrip:

    def test_strips_tags(self):
        assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_leaves_plain_text_alone(self):
        assert _strip_html("Plain text.") == "Plain text."

    def test_handles_empty(self):
        assert _strip_html("") == ""
