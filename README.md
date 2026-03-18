# TonyStock — Stock Screener

A systematic stock screening system combining price data with institutional investor flow (籌碼面) and machine learning to identify high-probability opportunities in the Taiwan market.

---

## System Architecture

```
Data Sources          Processing               Storage / Serving
────────────          ──────────               ─────────────────
yfinance          →   Polars ETL           →   DuckDB (.db file)
TWSE Open API     →   Institutional Score  →   FastAPI (REST)
                      LightGBM Model       →   LINE Bot
                      APScheduler
                      (weekly refresh)
```

**Deployment:** Railway (single service — API + scheduler in one container)

---

## Core Concepts

**Institutional Score (1–5)**
三大法人 (外資/投信/自營商) net buy/sell → cross-sectional quantile-normalized weekly score
- 4–5: accumulation | 1–2: distribution

**Win Rate**
Win = price rises ≥ 20% within 1 week after a signal week.
Historical win rate = wins / total signals per stock.

**ML Model**
LightGBM binary classifier predicting next-week ≥20% return probability, calibrated via IsotonicRegression.
- Validation AUC: 0.7732
- Train: < 2024-01-01 | Val: 2024 | Test: ≥ 2025-01-01

---

## Features

| Feature | Description |
|---------|-------------|
| `foreign_net_log` | 外資 weekly net buy/sell (log-normalized) |
| `trust_net_log` | 投信 weekly net buy/sell (log-normalized) |
| `dealer_net_log` | 自營商 weekly net buy/sell (log-normalized) |
| `total_net_log` | 三大法人 combined (log-normalized) |
| `total_net_log_lag1/2` | 三大法人 prior 1–2 week flow |
| `return_1w/4w/12w` | Price momentum (1 / 4 / 12 weeks) |
| `vol_ratio` | Volume vs 4-week average |
| `close_vs_52w_high` | Close price as % of 52-week high |
| `bb_position` | Bollinger Band position (close vs MA20 ± 2σ) |
| `atr_ratio` | ATR(14) / close — volatility relative to price |
| `ma20_slope` | 4-week rate of change of MA20 |

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Price data | yfinance — TW listed stocks (`.TW`), 5yr weekly OHLCV |
| Institutional data | TWSE API T86 — 三大法人 daily → weekly aggregation |
| Processing | Polars |
| Database | DuckDB |
| API | FastAPI |
| Scheduling | APScheduler (weekly, Monday 08:00 Asia/Taipei) |
| ML | LightGBM + IsotonicRegression calibration |
| Notification | LINE Messaging API |
| Deployment | Railway |

---

## Folder Structure

```
StockScreener/
├── api/               # FastAPI REST endpoints + LINE webhook handler
├── scheduler/         # Data fetchers, scoring, ML training, APScheduler jobs
│   ├── fetch_price.py
│   ├── fetch_institutional.py
│   ├── score_institutional.py
│   ├── calc_win_rate.py
│   ├── train_lgbm.py
│   └── jobs.py
├── ui/                # Streamlit dashboard (local dev)
├── notebooks/         # EDA & experiments
├── data/              # DuckDB file (gitignored)
├── Dockerfile
├── railway.toml
└── requirements.txt
```

---

## Scheduler CLI

```bash
python -m scheduler.jobs           # start weekly cron
python -m scheduler.jobs --init    # full historical init (skips if DB populated)
python -m scheduler.jobs --reinit  # force re-fetch 5yr historical data
python -m scheduler.jobs --retrain # retrain model only, no data re-fetch
python -m scheduler.jobs --now     # run incremental pipeline once (14d)
```

---

## LINE Bot Commands

| Input | Output |
|-------|--------|
| `顯示股票` / `推薦股票` / `選股` | Top 10 stocks by win probability this week |
| `2330` (any 4–5 digit code) | Individual stock detail (price, score, win prob) |
| anything else | Help message |

---

## Roadmap

- [x] Phase 1 — Data Pipeline: weekly OHLCV + 三大法人 → DuckDB
- [x] Phase 2 — Scoring: quantile institutional score + historical win rate
- [x] Phase 3 — FastAPI + APScheduler weekly refresh
- [x] Phase 4 — LightGBM model + IsotonicRegression calibration + LINE Bot
- [x] Deployment on Railway
- [ ] 上櫃 (OTC/TPEx) stocks via TWT44U API
- [ ] MLflow experiment tracking
- [ ] Streamlit dashboard (public)
