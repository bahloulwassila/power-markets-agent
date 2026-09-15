"""Tests for the markdown renderer."""
from __future__ import annotations

from src.agent.brief_generator import Brief
from src.delivery.markdown_writer import brief_to_markdown


def _sample_brief() -> Brief:
    return Brief(
        report_date="2024-06-01",
        prompt_version="v1.0",
        model="claude-sonnet-4-5",
        input_tokens=100,
        output_tokens=200,
        payload={
            "tldr": "Alberta flat around $50; Ontario firmed 8%.",
            "aeso_section": {
                "headline": "AESO stable.",
                "bullets": ["Mean $50/MWh.", "No spikes."],
            },
            "ieso_section": {
                "headline": "OEMP up 8%.",
                "bullets": ["Mean $38/MWh.", "No anomalies."],
            },
            "spread_section": {
                "headline": "Spread widened at hour 18.",
                "bullets": ["Mean spread +$14/MWh."],
            },
            "news_highlights": [
                {
                    "title": "Genesee 2 outage",
                    "url": "http://example.com/1",
                    "source": "AESO",
                    "why_it_matters": "May pressure AB tomorrow AM.",
                },
            ],
            "watchlist": ["Genesee return", "Capacity auction"],
        },
    )


class TestBriefToMarkdown:

    def test_produces_a_string(self):
        md = brief_to_markdown(_sample_brief())
        assert isinstance(md, str)
        assert len(md) > 100

    def test_contains_header(self):
        md = brief_to_markdown(_sample_brief())
        assert "# Daily Power Markets Brief" in md
        assert "2024-06-01" in md

    def test_contains_all_sections(self):
        md = brief_to_markdown(_sample_brief())
        assert "TL;DR" in md
        assert "Alberta (AESO)" in md
        assert "Ontario (IESO)" in md
        assert "Inter-market spread" in md
        assert "Market context" in md
        assert "Watchlist" in md

    def test_bullets_rendered(self):
        md = brief_to_markdown(_sample_brief())
        assert "- Mean $50/MWh." in md
        assert "- Genesee return" in md

    def test_news_link_rendered(self):
        md = brief_to_markdown(_sample_brief())
        assert "[Genesee 2 outage](http://example.com/1)" in md

    def test_empty_news_still_works(self):
        brief = _sample_brief()
        brief = Brief(
            report_date=brief.report_date,
            prompt_version=brief.prompt_version,
            model=brief.model,
            input_tokens=brief.input_tokens,
            output_tokens=brief.output_tokens,
            payload={**brief.payload, "news_highlights": []},
        )
        md = brief_to_markdown(brief)
        # No "Market context" heading when there's no news
        assert "Market context" not in md

    def test_footer_contains_metadata(self):
        md = brief_to_markdown(_sample_brief())
        assert "claude-sonnet-4-5" in md
        assert "v1.0" in md
        assert "100" in md  # input tokens
