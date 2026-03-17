# TonyStock — Stock Screener

A systematic stock screening system combining price data with institutional investor flow (籌碼面) to identify high-win-rate opportunities in the Taiwan market.

---

## System Architecture

```
Data Sources          Processing             Storage / Serving
────────────          ──────────             ─────────────────
yfinance          →   Polars ETL         →   DuckDB (.db file)
TWSE Open API     →   Scoring Model      →   FastAPI (REST)
                      APScheduler        →   Streamlit (UI)
                      (weekly refresh)
```

**Services:** `api` | `scheduler` | `ui`

---

## Core Concepts

**Institutional Score (1–5)**
三大法人 (外資/投信/自營商) net buy/sell → quantile-normalized weekly score
- 4–5: accumulation | 1–2: distribution

**Win Rate**
Win = price rises ≥ X% within N weeks after a signal week.
Historical win rate = wins / total signals per stock.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Price data | yfinance (TW + US weekly OHLCV) |
| Institutional data | TWSE API (T86) |
| Processing | Polars |
| Database | DuckDB |
| API | FastAPI |
| UI | Streamlit |
| Scheduling | APScheduler |
| Container | Docker + docker-compose |

---

## Folder Structure

```
StockScreener/
├── api/               # FastAPI REST endpoints
├── scheduler/         # Data fetchers + scoring + APScheduler jobs
├── ui/                # Streamlit dashboard
├── notebooks/         # EDA & experiments
├── data/              # DuckDB file (gitignored)
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

```bash
# Setup
python -m venv stock-env
source stock-env/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# Run individually
python scheduler/fetch_price.py
uvicorn api.main:app --reload
streamlit run ui/app.py

# Or run all via Docker
docker-compose up --build
```

---

## Roadmap

- [x] Phase 1 — Data Pipeline: weekly OHLCV + 三大法人 → DuckDB
- [x] Phase 2 — Scoring: quantile institutional score + historical win rate
- [ ] Phase 3 — API endpoint `POST /screen` + Streamlit dashboard + APScheduler
- [ ] Phase 4 — LightGBM direction prediction + MLflow + full Docker deployment
