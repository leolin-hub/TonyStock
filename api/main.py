"""
api/main.py
FastAPI app — stock screening endpoints + LINE Bot webhook.

GET  /screen           — top stocks ranked by LightGBM win_prob
GET  /stock/{symbol}   — weekly detail for one stock
POST /webhook          — LINE Messaging API webhook
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel
from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    ApiClient, Configuration, MessagingApi,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"

LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_TOKEN  = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

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

line_handler   = WebhookHandler(LINE_SECRET)
line_config    = Configuration(access_token=LINE_TOKEN)


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


# ── DB helpers ────────────────────────────────────────────────────────────────

def query_top_stocks(min_win_prob: float = 0.03, top_n: int = 10) -> list[dict]:
    con = get_con()
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
        SELECT m.symbol, m.win_prob, s.score_total, CAST(m.week_start AS VARCHAR)
        FROM latest_pred m
        LEFT JOIN latest_score s ON m.symbol = s.symbol
        WHERE m.win_prob >= {min_win_prob}
        ORDER BY m.win_prob DESC
        LIMIT {top_n}
    """).fetchall()
    return [{"symbol": r[0], "win_prob": r[1], "latest_score": r[2], "latest_week": r[3]}
            for r in rows]


def query_stock(symbol: str) -> dict | None:
    con = get_con()
    row = con.execute(f"""
        SELECT m.symbol, CAST(m.week_start AS VARCHAR), m.win_prob, s.score_total, p.close
        FROM ml_predictions m
        LEFT JOIN weekly_score s ON m.symbol = s.symbol AND m.week_start = s.week_start
        LEFT JOIN weekly_price p ON p.symbol = m.symbol || '.TW' AND p.date = m.week_start
        WHERE m.symbol = '{symbol}'
          AND (m.symbol, m.week_start) IN (
              SELECT symbol, MAX(week_start) FROM ml_predictions GROUP BY symbol
          )
    """).fetchone()
    if not row:
        return None
    return {"symbol": row[0], "week": row[1], "win_prob": row[2],
            "score": row[3], "close": row[4]}


# ── LINE message formatters ───────────────────────────────────────────────────

def fmt_top_stocks(stocks: list[dict]) -> str:
    if not stocks:
        return "目前沒有符合條件的股票。"
    lines = ["📊 本週模型推薦（下週漲20%機率）\n"]
    for i, s in enumerate(stocks, 1):
        score_str = f"{'⭐'*s['latest_score']}" if s["latest_score"] else "—"
        lines.append(f"{i}. {s['symbol']}  {s['win_prob']*100:.2f}%  {score_str}")
    lines.append(f"\n資料週：{stocks[0]['latest_week']}")
    return "\n".join(lines)


def fmt_stock_detail(s: dict) -> str:
    score_str = f"{'⭐'*s['score']}" if s["score"] else "—"
    close_str = f"${s['close']:.1f}" if s["close"] else "—"
    return (
        f"📈 {s['symbol']} ({s['week']})\n"
        f"收盤價：{close_str}\n"
        f"籌碼分數：{score_str}\n"
        f"模型預測（漲20%機率）：{s['win_prob']*100:.2f}%"
    )


HELP_MSG = (
    "指令說明：\n"
    "・顯示股票 — 本週推薦清單\n"
    "・[股票代號] — 查詢個股（例：2330）"
)


# ── LINE webhook ──────────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    try:
        line_handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        return Response(content="Invalid signature", status_code=400)
    return Response(content="OK")


@line_handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    text = event.message.text.strip()

    if text in ("顯示股票", "推薦股票", "選股"):
        stocks = query_top_stocks(min_win_prob=0.03, top_n=10)
        reply  = fmt_top_stocks(stocks)
    elif text.isdigit() and len(text) in (4, 5):
        stock = query_stock(text)
        reply = fmt_stock_detail(stock) if stock else f"找不到股票代號 {text}"
    else:
        reply = HELP_MSG

    with ApiClient(line_config) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply)],
            )
        )


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/screen", response_model=list[ScreenResult])
def screen(
    min_win_prob: float = Query(0.03, ge=0.0, le=1.0),
    top_n:        int   = Query(50,   ge=1,   le=500),
    symbols:      str   = Query(""),
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
        SELECT m.symbol, m.win_prob, s.score_total, CAST(m.week_start AS VARCHAR)
        FROM latest_pred m
        LEFT JOIN latest_score s ON m.symbol = s.symbol
        WHERE m.win_prob >= {min_win_prob}
          {symbol_filter}
        ORDER BY m.win_prob DESC
        LIMIT {top_n}
    """).fetchall()

    return [ScreenResult(symbol=r[0], win_prob=r[1], latest_score=r[2], latest_week=r[3])
            for r in rows]


@app.get("/stock/{symbol}", response_model=list[StockDetail])
def stock_detail(symbol: str):
    con = get_con()
    rows = con.execute(f"""
        SELECT s.symbol, CAST(s.week_start AS VARCHAR),
               s.score_foreign, s.score_trust, s.score_dealer, s.score_total,
               p.close, m.win_prob
        FROM weekly_score s
        LEFT JOIN weekly_price p
            ON p.symbol = s.symbol || '.TW' AND p.date = s.week_start
        LEFT JOIN ml_predictions m
            ON m.symbol = s.symbol AND m.week_start = s.week_start
        WHERE s.symbol = '{symbol}'
        ORDER BY s.week_start DESC
    """).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    return [StockDetail(symbol=r[0], week_start=r[1], score_foreign=r[2],
                        score_trust=r[3], score_dealer=r[4], score_total=r[5],
                        close=r[6], win_prob=r[7])
            for r in rows]
