# Power Markets Research Agent

> An autonomous AI agent that ingests Alberta (AESO) and Ontario (IESO)
> spot electricity prices, detects anomalies, aggregates market news, and
> generates a daily research brief via the Claude API.

Built as a hands-on prototype of an **internal research agent** for a
power-markets trading desk. Replaces ~45 minutes of daily manual market
scanning with a 30-second automated pipeline.

---

## Why this project

Power traders and analysts start every morning by piecing together the
same picture: what did spot prices do yesterday, were there any spikes,
what's in the news, and does anything need immediate attention. That's a
**repeatable workflow** with clear inputs and a clear deliverable — the
exact kind of task LLM-driven automation is now good at.

This project builds that agent end-to-end with a focus on:

- **Typed, testable data clients** — not just `requests.get` in a Jupyter cell
- **Deterministic analysis** — z-score anomaly detection, inter-market spreads,
  computed *before* any LLM call so numbers don't get hallucinated
- **Structured LLM outputs** — JSON-schema-validated briefs, machine-parseable
- **Reproducibility** — cached data, versioned prompts, deterministic outputs

---

## Architecture
ls src/

---

## Data sources

| Source | Data | Auth | Notes |
|---|---|---|---|
| [AESO Pool Price API](https://apim-aeso-connect.developer.azure-api.net/) | Hourly Alberta Pool Price ($/MWh) | Free API key | Azure APIM gateway |
| [IESO Public Reports](https://www.ieso.ca/power-data/data-directory) | Hourly Ontario Price / HOEP | None | CSV downloads |
| RSS (CER, Reuters, OilPrice) | Energy news | None | Falls back to NewsAPI if key present |

> **Note on the Ontario market:** The Hourly Ontario Energy Price (HOEP)
> was retired on 30 April 2025 under IESO's Market Renewal Program. The
> agent uses the new Ontario Price (OEMP = DA-OZP + LFDA) for dates from
> May 2025 onward, and HOEP for earlier dates.

---

## Quick start

```bash
# 1. Clone & install
git clone https://github.com/bahloulwassila/power-markets-agent.git
cd power-markets-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: add your AESO_API_KEY and ANTHROPIC_API_KEY

# 3. Run the daily brief (Phase 3+)
python -m src.main
```

Generated briefs go to `briefs/YYYY-MM-DD.md`.

---

## Project status

| Phase | Description | Status |
|---|---|---|
| 0 | Project scaffolding, config, CI-ready structure | ✅ Done |
| 1 | Data ingestion clients (AESO, IESO, news) | 🔜 |
| 2 | Quantitative analysis (anomalies, spreads) | 🔜 |
| 3 | Claude agent (structured briefs) | 🔜 |
| 4 | Delivery (CLI, Streamlit dashboard) | 🔜 |
| 5 | Documentation & example briefs | 🔜 |

---

## Tech

Python 3.11+ · pandas · requests · feedparser · Anthropic SDK · pytest · ruff

## License

MIT — see [LICENSE](LICENSE)
