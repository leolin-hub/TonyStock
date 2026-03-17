"""
api/main.py
FastAPI app — stock screening endpoints.

GET /screen
  params: min_win_prob (float), top_n (int), symbols (comma-separated)
  returns: stocks ranked by latest LightGBM win_prob

GET /stock/{symbol}
  returns: weekly scores + close price + win_prob per week
"""

from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

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

class ScreenResult(BaseModel):
    symbol:       str
    win_prob:     float
    latest_score: int   | None
    latest_week:  str   | None


class StockDetail(BaseModel):
    symbol:        str
    week_start:    str
    score_foreign: int   | None
    score_trust:   int   | None
    score_dealer:  int   | None
    score_total:   int   | None
    close:         float | None
    win_prob:      float | None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/screen", response_model=list[ScreenResult])
def screen(
    min_win_prob: float = Query(0.5, ge=0.0, le=1.0, description="最低模型預測機率"),
    top_n:        int   = Query(50,  ge=1,   le=500,  description="顯示前幾名"),
    symbols:      str   = Query("",  description="指定股票代號，逗號分隔，留空=全部"),
):
    con = get_con()

    symbol_filter = ""
    if symbols.strip():
        quoted = ", ".join(f"'{s.strip()}'" for s in symbols.split(",") if s.strip())
        symbol_filter = f"AND m.symbol IN ({quoted})"

    rows = con.execute(f"""
        WITH latest_pred AS (
            SELECT symbol, win_prob, week_start
            FROM ml_predictions
            WHERE (symbol, week_start) IN (
                SELECT symbol, MAX(week_start) FROM ml_predictions GROUP BY symbol
            )
        ),
        latest_score AS (
            SELECT symbol, score_total
            FROM weekly_score
            WHERE (symbol, week_start) IN (
                SELECT symbol, MAX(week_start) FROM weekly_score GROUP BY symbol
            )
        )
        SELECT
            m.symbol,
            m.win_prob,
            s.score_total AS latest_score,
            CAST(m.week_start AS VARCHAR) AS latest_week
        FROM latest_pred m
        LEFT JOIN latest_score s ON m.symbol = s.symbol
        WHERE m.win_prob >= {min_win_prob}
          {symbol_filter}
        ORDER BY m.win_prob DESC
        LIMIT {top_n}
    """).fetchall()

    return [
        ScreenResult(symbol=r[0], win_prob=r[1], latest_score=r[2], latest_week=r[3])
        for r in rows
    ]


@app.get("/stock/{symbol}", response_model=list[StockDetail])
def stock_detail(symbol: str):
    con = get_con()

    rows = con.execute(f"""
        SELECT
            s.symbol,
            CAST(s.week_start AS VARCHAR),
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
            ON m.symbol      = s.symbol
            AND m.week_start = s.week_start
        WHERE s.symbol = '{symbol}'
        ORDER BY s.week_start DESC
    """).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    return [
        StockDetail(
            symbol=r[0], week_start=r[1],
            score_foreign=r[2], score_trust=r[3], score_dealer=r[4], score_total=r[5],
            close=r[6], win_prob=r[7],
        )
        for r in rows
    ]
