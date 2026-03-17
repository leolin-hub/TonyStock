"""
api/main.py
FastAPI app — stock screening endpoints.

POST /screen
  body: { win_rate_threshold, symbols?, score_threshold, n_weeks, return_threshold }
  returns: stocks where historical win_rate >= threshold,
           each row includes latest win_prob from LightGBM

GET /stock/{symbol}
  returns: weekly scores + close price + win_prob per week
"""

from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"

_con: duckdb.DuckDBPyConnection | None = None


def get_con() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect(str(DB_PATH), read_only=True)
    return _con


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_con()
    yield
    if _con:
        _con.close()


app = FastAPI(title="StockScreener API", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    win_rate_threshold: float = Field(..., ge=0.0, le=1.0, description="e.g. 0.6 for 60%")
    symbols: list[str] | None = Field(None, description="Optional filter, e.g. ['2330', '2454']")
    score_threshold: int   = Field(4,    ge=1, le=5)
    n_weeks: int           = Field(4,    ge=1, le=52)
    return_threshold: float = Field(0.05, ge=0.0)


class ScreenResult(BaseModel):
    symbol:        str
    total_signals: int
    wins:          int
    win_rate:      float
    latest_score:  int   | None
    latest_week:   str   | None
    latest_win_prob: float | None   # LightGBM prediction for latest week


class StockDetail(BaseModel):
    symbol:        str
    week_start:    str
    score_foreign: int   | None
    score_trust:   int   | None
    score_dealer:  int   | None
    score_total:   int   | None
    close:         float | None
    win_prob:      float | None   # LightGBM prediction for this week


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/screen", response_model=list[ScreenResult])
def screen(req: ScreenRequest):
    con = get_con()

    symbol_filter = ""
    if req.symbols:
        quoted = ", ".join(f"'{s}'" for s in req.symbols)
        symbol_filter = f"AND w.symbol IN ({quoted})"

    rows = con.execute(f"""
        WITH latest_score AS (
            SELECT symbol, score_total, week_start
            FROM weekly_score
            WHERE (symbol, week_start) IN (
                SELECT symbol, MAX(week_start) FROM weekly_score GROUP BY symbol
            )
        ),
        latest_pred AS (
            SELECT symbol, win_prob, week_start
            FROM ml_predictions
            WHERE (symbol, week_start) IN (
                SELECT symbol, MAX(week_start) FROM ml_predictions GROUP BY symbol
            )
        )
        SELECT
            w.symbol,
            w.total_signals,
            w.wins,
            w.win_rate,
            s.score_total       AS latest_score,
            s.week_start        AS latest_week,
            m.win_prob          AS latest_win_prob
        FROM win_rate w
        LEFT JOIN latest_score s ON w.symbol = s.symbol
        LEFT JOIN latest_pred  m ON w.symbol = m.symbol
        WHERE w.win_rate        >= {req.win_rate_threshold}
          AND w.score_threshold  = {req.score_threshold}
          AND w.n_weeks          = {req.n_weeks}
          AND w.return_threshold = {req.return_threshold}
          {symbol_filter}
        ORDER BY w.win_rate DESC, w.total_signals DESC
    """).fetchall()

    return [
        ScreenResult(
            symbol=r[0],
            total_signals=r[1],
            wins=r[2],
            win_rate=r[3],
            latest_score=r[4],
            latest_week=str(r[5]) if r[5] else None,
            latest_win_prob=r[6],
        )
        for r in rows
    ]


@app.get("/stock/{symbol}", response_model=list[StockDetail])
def stock_detail(symbol: str):
    con = get_con()

    rows = con.execute(f"""
        SELECT
            s.symbol,
            s.week_start,
            s.score_foreign,
            s.score_trust,
            s.score_dealer,
            s.score_total,
            p.close,
            m.win_prob
        FROM weekly_score s
        LEFT JOIN weekly_price p
            ON p.symbol = s.symbol || '.TW'
            AND p.date  = s.week_start
        LEFT JOIN ml_predictions m
            ON m.symbol     = s.symbol
            AND m.week_start = s.week_start
        WHERE s.symbol = '{symbol}'
        ORDER BY s.week_start DESC
    """).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    return [
        StockDetail(
            symbol=r[0],
            week_start=str(r[1]),
            score_foreign=r[2],
            score_trust=r[3],
            score_dealer=r[4],
            score_total=r[5],
            close=r[6],
            win_prob=r[7],
        )
        for r in rows
    ]
