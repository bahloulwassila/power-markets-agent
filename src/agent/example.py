"""Generate an example brief (with a canned Claude response).

Used to seed briefs/example_brief.md so recruiters can see the output on
GitHub without installing anything. Once the ANTHROPIC_API_KEY is set,
this can be replaced by a real call — see the CLI added in Phase 4.
"""
from __future__ import annotations

from pathlib import Path

from src.agent.brief_generator import Brief
from src.agent.prompts import PROMPT_VERSION
from src.config import CLAUDE_MODEL, BRIEFS_DIR
from src.delivery.markdown_writer import brief_to_markdown


EXAMPLE_PAYLOAD = {
    "tldr": (
        "Alberta traded normally around $52/MWh; Ontario firmed to $38/MWh "
        "(+8% vs baseline). One notable AESO evening spike (hour 18, z=2.8) "
        "tied to a Genesee 2 outage. AESO-OEMP spread widened to $28/MWh — "
        "worth watching."
    ),
    "aeso_section": {
        "headline": "AESO mostly normal, but one evening spike.",
        "bullets": [
            "Mean $52.14/MWh, peak avg $81.30, off-peak avg $28.90 "
            "— peak/off-peak ratio 2.81, in line with shoulder-season shape.",
            "Anomaly at hour 18: $134.20 (z=2.83, spike). Coincides with an "
            "unplanned Genesee 2 outage report; recovery expected tomorrow AM.",
            "Coefficient of variation 0.42 — moderate dispersion, not stressed.",
        ],
    },
    "ieso_section": {
        "headline": "OEMP up 8% vs baseline; no anomalies.",
        "bullets": [
            "Mean $38.07/MWh, up ~8% versus the 30-day baseline.",
            "Max |z|=1.6 — no hour flagged. Peak hours softer than usual at "
            "$46.20, consistent with mild weekend demand.",
        ],
    },
    "spread_section": {
        "headline": "Spread widened to $14/MWh average, +$96 at the AESO spike hour.",
        "bullets": [
            "Mean spread +$14.07/MWh (AESO premium), widest gap at hour 18 "
            "at +$96.10/MWh.",
            "Pearson correlation 0.62 vs typical ~0.85 — decoupled by the outage.",
            "3 hours exceeded the $50/MWh wide-gap threshold, all in the AESO "
            "evening.",
        ],
    },
    "news_highlights": [
        {
            "title": "Genesee 2 unit trip reported at 15:40 MT",
            "url": "https://www.aeso.ca/market/outages/",
            "source": "AESO real-time bulletin",
            "why_it_matters": (
                "Directly explains the hour-18 AESO spike; watch for "
                "return-to-service confirmation tomorrow morning."
            ),
        },
        {
            "title": "Ontario capacity auction results due Friday",
            "url": "https://www.ieso.ca/",
            "source": "IESO press release",
            "why_it_matters": (
                "Impacts forward OEMP; clearing prices reset resource "
                "adequacy signals."
            ),
        },
        {
            "title": "Alberta wind generation forecast lowered 400 MW for tomorrow",
            "url": "https://www.aeso.ca/",
            "source": "AESO forecast update",
            "why_it_matters": (
                "Tightens AB balance overnight; increases odds of a "
                "second evening spike."
            ),
        },
    ],
    "watchlist": [
        "Genesee 2 return-to-service timing (impacts AESO evening 17-22h)",
        "Ontario capacity auction clearing prices (impacts forward OEMP)",
        "AB wind forecast update at 09:00 MT — currently -400 MW revision",
        "AESO-OEMP correlation recovery back toward 0.85",
    ],
}


def build_example_brief() -> Brief:
    return Brief(
        report_date="2026-09-15",
        prompt_version=PROMPT_VERSION,
        model=CLAUDE_MODEL,
        payload=EXAMPLE_PAYLOAD,
        input_tokens=1847,
        output_tokens=612,
    )


def write_example_to_disk() -> Path:
    brief = build_example_brief()
    md = brief_to_markdown(brief)
    out = BRIEFS_DIR / "example_brief.md"
    out.write_text(md, encoding="utf-8")
    return out


if __name__ == "__main__":
    path = write_example_to_disk()
    print(f"✓ Example brief written to {path}")
