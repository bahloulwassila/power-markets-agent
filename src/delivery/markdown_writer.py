"""Render a Brief as human-readable markdown.

Formatting only — no data logic. The Brief dataclass is the source of
truth; this module just presents it. That means we can change the
markdown format without touching the LLM pipeline.
"""
from __future__ import annotations

from src.agent.brief_generator import Brief


def brief_to_markdown(brief: Brief) -> str:
    """Convert a Brief into a markdown document."""
    p = brief.payload
    parts: list[str] = []

    parts.append(f"# Daily Power Markets Brief — {brief.report_date}")
    parts.append("")
    parts.append(f"**TL;DR:** {p['tldr']}")
    parts.append("")

    parts.extend(_render_section("Alberta (AESO)", p["aeso_section"]))
    parts.extend(_render_section("Ontario (IESO)", p["ieso_section"]))
    parts.extend(_render_section("Inter-market spread (AESO – OEMP)", p["spread_section"]))

    # News highlights
    news = p.get("news_highlights") or []
    if news:
        parts.append("## Market context (news)")
        parts.append("")
        for i, item in enumerate(news, start=1):
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            source = item.get("source", "").strip()
            why = item.get("why_it_matters", "").strip()
            link = f"[{title}]({url})" if url else title
            parts.append(f"{i}. **{link}** — *{source}*")
            if why:
                parts.append(f"   {why}")
        parts.append("")

    # Watchlist
    watch = p.get("watchlist") or []
    if watch:
        parts.append("## Watchlist for tomorrow")
        parts.append("")
        for item in watch:
            parts.append(f"- {item}")
        parts.append("")

    # Footer
    parts.append("---")
    parts.append(
        f"*Generated with {brief.model} · prompt {brief.prompt_version} · "
        f"{brief.input_tokens:,} in / {brief.output_tokens:,} out tokens.*"
    )
    parts.append("")

    return "\n".join(parts)


def _render_section(title: str, section: dict) -> list[str]:
    headline = (section.get("headline") or "").strip()
    bullets = section.get("bullets") or []
    out = [f"## {title}", ""]
    if headline:
        out.append(f"**{headline}**")
        out.append("")
    for b in bullets:
        out.append(f"- {b}")
    out.append("")
    return out
