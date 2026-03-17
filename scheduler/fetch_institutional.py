"""
fetch_institutional.py
Fetches 三大法人 daily net buy/sell data from TWSE Open API,
aggregates to weekly, and stores into DuckDB.

Column index mapping (TWSE T86):
  0  = 證券代號 (symbol)
  4  = 外陸資買賣超股數 (foreign net)
  10 = 投信買賣超股數   (trust net)
  11 = 自營商買賣超股數 (dealer net)
  18 = 三大法人買賣超股數 (total net)
"""

import json
import time
import urllib3
import duckdb
import polars as pl
import requests
from datetime import date, timedelta
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"
TWSE_URL = "https://www.twse.com.tw/fund/T86"
DELAY = 1.0  # seconds between requests


# ── TWSE fetcher ────────────────────────────────────────────────────────────

def fetch_daily(trade_date: date) -> pl.DataFrame:
    date_str = trade_date.strftime("%Y%m%d")
    try:
        resp = requests.get(
            TWSE_URL,
            params={"response": "json", "date": date_str, "selectType": "ALL"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
            verify=False,
        )
        data = json.loads(resp.content.decode("utf-8"))
    except Exception as e:
        print(f"[WARN] {date_str} request failed: {e}")
        return pl.DataFrame()

    if data.get("stat") != "OK" or not data.get("data"):
        return pl.DataFrame()

    rows = data["data"]

    def parse_int(val: str) -> int | None:
        try:
            return int(val.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    records = []
    for row in rows:
        if len(row) < 19:  # skip malformed rows (ETFs, indices, etc.)
            continue
        symbol = row[0].strip()
        if len(symbol) != 4:  # keep only 4-digit TW stock codes
            continue
        records.append({
            "symbol":      symbol,
            "date":        trade_date,
            "foreign_net": parse_int(row[4]),
            "trust_net":   parse_int(row[10]),
            "dealer_net":  parse_int(row[11]),
            "total_net":   parse_int(row[18]),
        })

    if not records:
        return pl.DataFrame()

    return pl.DataFrame(records, schema={
        "symbol":      pl.Utf8,
        "date":        pl.Date,
        "foreign_net": pl.Int64,
        "trust_net":   pl.Int64,
        "dealer_net":  pl.Int64,
        "total_net":   pl.Int64,
    })


# ── DB helpers ───────────────────────────────────────────────────────────────

def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_institutional (
            symbol      VARCHAR NOT NULL,
            date        DATE    NOT NULL,
            foreign_net BIGINT,
            trust_net   BIGINT,
            dealer_net  BIGINT,
            total_net   BIGINT,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS weekly_institutional (
            symbol          VARCHAR NOT NULL,
            week_start      DATE    NOT NULL,
            foreign_net_sum BIGINT,
            trust_net_sum   BIGINT,
            dealer_net_sum  BIGINT,
            total_net_sum   BIGINT,
            PRIMARY KEY (symbol, week_start)
        )
    """)


def get_fetched_dates(con: duckdb.DuckDBPyConnection) -> set[date]:
    rows = con.execute("SELECT DISTINCT date FROM daily_institutional").fetchall()
    return {r[0] for r in rows}


def upsert_daily(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    con.execute("""
        INSERT OR REPLACE INTO daily_institutional
        SELECT symbol, date, foreign_net, trust_net, dealer_net, total_net FROM df
    """)


def aggregate_weekly(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        INSERT OR REPLACE INTO weekly_institutional
        SELECT
            symbol,
            date_trunc('week', date)::DATE AS week_start,
            SUM(foreign_net) AS foreign_net_sum,
            SUM(trust_net)   AS trust_net_sum,
            SUM(dealer_net)  AS dealer_net_sum,
            SUM(total_net)   AS total_net_sum
        FROM daily_institutional
        GROUP BY symbol, date_trunc('week', date)
    """)


# ── Main ─────────────────────────────────────────────────────────────────────

def trading_days(start: date, end: date) -> list[date]:
    """Weekdays only; TWSE returns empty for holidays — handled downstream."""
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def run(lookback_days: int = 90) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    end_date   = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    days       = trading_days(start_date, end_date)
    fetched    = get_fetched_dates(con)
    to_fetch   = [d for d in days if d not in fetched]

    print(f"[INFO] Fetching {len(to_fetch)} days (skipping {len(days) - len(to_fetch)} cached)")

    skipped = 0
    for i, d in enumerate(to_fetch):
        df = fetch_daily(d)
        if df.is_empty():
            skipped += 1
            print(f"[SKIP] {d} — no data (holiday?)")
        else:
            upsert_daily(con, df)
            print(f"[INFO] {d} — {len(df)} stocks upserted")

        if i < len(to_fetch) - 1:
            time.sleep(DELAY)

    print(f"\n[INFO] Aggregating to weekly ... (skipped {skipped} non-trading days)")
    aggregate_weekly(con)

    total = con.execute("SELECT COUNT(*) FROM weekly_institutional").fetchone()[0]
    print(f"[DONE] weekly_institutional total rows: {total}")
    con.close()


if __name__ == "__main__":
    run()
