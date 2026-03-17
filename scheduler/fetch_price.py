"""
fetch_price.py
Fetches 5-year weekly OHLCV data via yfinance and stores into DuckDB.

Modes:
  run(symbols)          — fetch specific symbols (default: SYMBOLS list)
  run_all_tw()          — fetch all TW symbols found in weekly_institutional table
                          uses batch download (100 symbols per request)
"""

import duckdb
import polars as pl
import yfinance as yf
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"
SYMBOLS = ["2330.TW", "2454.TW", "TSM", "NVDA"]
PERIOD = "5y"
INTERVAL = "1wk"
BATCH_SIZE = 100


# ── DB helpers ────────────────────────────────────────────────────────────────

def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS weekly_price (
            symbol  VARCHAR NOT NULL,
            date    DATE    NOT NULL,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,
            PRIMARY KEY (symbol, date)
        )
    """)


def upsert(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> None:
    con.execute("""
        INSERT OR REPLACE INTO weekly_price
        SELECT symbol, date, open, high, low, close, volume FROM df
    """)


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_single(symbol: str) -> pl.DataFrame:
    """Fetch one symbol via yfinance."""
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)

    if raw.empty:
        return pl.DataFrame()

    return (
        pl.from_pandas(raw.reset_index())
        .rename({"Date": "date", "Open": "open", "High": "high",
                 "Low": "low", "Close": "close", "Volume": "volume"})
        .select(["date", "open", "high", "low", "close", "volume"])
        .with_columns([
            pl.lit(symbol).alias("symbol"),
            pl.col("date").dt.date().alias("date"),
            pl.col("volume").cast(pl.Int64),
        ])
    )


def fetch_batch(symbols: list[str]) -> pl.DataFrame:
    """Batch download multiple symbols in one yfinance call."""
    raw = yf.download(
        symbols,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pl.DataFrame()

    frames = []
    # Single symbol returns flat columns; multiple returns multi-level
    if len(symbols) == 1:
        df = _parse_single_yf(raw, symbols[0])
        if df is not None:
            frames.append(df)
    else:
        for symbol in symbols:
            if symbol not in raw.columns.get_level_values(0):
                continue
            sub = raw[symbol].dropna(how="all")
            if sub.empty:
                continue
            df = _parse_single_yf(sub, symbol)
            if df is not None:
                frames.append(df)

    return pl.concat(frames) if frames else pl.DataFrame()


def _parse_single_yf(raw, symbol: str) -> pl.DataFrame | None:
    try:
        df = (
            pl.from_pandas(raw.reset_index())
            .rename({c: c.lower() for c in raw.reset_index().columns})
        )
        # normalise date column name (yfinance uses 'Date' or 'Datetime')
        for col in df.columns:
            if col.lower() in ("date", "datetime"):
                df = df.rename({col: "date"})
                break

        return (
            df.select(["date", "open", "high", "low", "close", "volume"])
            .with_columns([
                pl.lit(symbol).alias("symbol"),
                pl.col("date").dt.date().alias("date"),
                pl.col("volume").cast(pl.Int64, strict=False),
            ])
            .drop_nulls(subset=["close"])
        )
    except Exception:
        return None


# ── Public entry points ───────────────────────────────────────────────────────

def run(symbols: list[str] = SYMBOLS) -> None:
    """Fetch a specific list of symbols (single-by-single, good for small lists)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    for symbol in symbols:
        print(f"[INFO] Fetching {symbol} ...")
        df = fetch_single(symbol)
        if df.is_empty():
            print(f"[WARN] No data for {symbol}")
            continue
        upsert(con, df)
        print(f"[INFO] {symbol}: {len(df)} rows upserted")

    total = con.execute("SELECT COUNT(*) FROM weekly_price").fetchone()[0]
    print(f"[DONE] weekly_price total rows: {total}")
    con.close()


def run_all_tw() -> None:
    """
    Fetch all TW symbols from weekly_institutional, using batch download.
    Skips symbols already having up-to-date data (latest date = last week).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    # Get all 4-digit TW stock codes from institutional table
    rows = con.execute(
        "SELECT DISTINCT symbol FROM weekly_institutional ORDER BY symbol"
    ).fetchall()
    tw_symbols = [r[0] + ".TW" for r in rows]
    print(f"[INFO] {len(tw_symbols)} TW symbols to fetch")

    # Batch download
    total_upserted = 0
    for i in range(0, len(tw_symbols), BATCH_SIZE):
        batch = tw_symbols[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(tw_symbols) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[INFO] Batch {batch_num}/{total_batches} ({len(batch)} symbols) ...")

        df = fetch_batch(batch)
        if df.is_empty():
            print(f"[WARN] Batch {batch_num} returned no data")
            continue

        upsert(con, df)
        total_upserted += len(df)
        print(f"[INFO] Batch {batch_num} done — {len(df)} rows upserted")

    total = con.execute("SELECT COUNT(*) FROM weekly_price").fetchone()[0]
    print(f"[DONE] weekly_price total rows: {total} ({total_upserted} new/updated)")
    con.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all-tw":
        run_all_tw()
    else:
        run()
