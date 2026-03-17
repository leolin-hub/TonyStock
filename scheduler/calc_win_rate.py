"""
calc_win_rate.py
Computes historical win rate per stock based on institutional score signals.

Algorithm:
  1. Join weekly_price + weekly_score (strip '.TW' suffix to align symbols)
  2. LEAD window: get close price N weeks forward
  3. Signal week: score_total >= score_threshold
  4. Win: forward_return >= return_threshold
  5. Win rate = wins / total_signals (stocks with < min_signals are excluded)
  6. Store results into win_rate table

Default params:
  score_threshold  = 4    (score 4 or 5 = institutional accumulation)
  n_weeks          = 4    (look 4 trading weeks forward)
  return_threshold = 0.05 (5% gain counts as a win)
  min_signals      = 3    (need at least 3 signals to be statistically meaningful)
"""

import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS win_rate (
            symbol           VARCHAR NOT NULL,
            score_threshold  TINYINT NOT NULL,
            n_weeks          TINYINT NOT NULL,
            return_threshold DOUBLE  NOT NULL,
            total_signals    INTEGER,
            wins             INTEGER,
            win_rate         DOUBLE,
            PRIMARY KEY (symbol, score_threshold, n_weeks, return_threshold)
        )
    """)


def compute_win_rate(
    con: duckdb.DuckDBPyConnection,
    score_threshold: int = 4,
    n_weeks: int = 4,
    return_threshold: float = 0.05,
    min_signals: int = 3,
) -> int:
    """
    Computes win rates and upserts into win_rate table.
    Returns number of stocks written.
    """
    result = con.execute(f"""
        WITH price_normalized AS (
            -- Strip '.TW' suffix so symbols match weekly_score
            SELECT
                CASE
                    WHEN symbol LIKE '%.TW' THEN LEFT(symbol, LENGTH(symbol) - 3)
                    ELSE symbol
                END AS symbol,
                symbol AS raw_symbol,
                date   AS week_date,
                close
            FROM weekly_price
        ),
        price_with_forward AS (
            -- Attach forward close price N weeks later via LEAD
            SELECT
                symbol,
                raw_symbol,
                week_date,
                close,
                LEAD(close, {n_weeks}) OVER (
                    PARTITION BY symbol ORDER BY week_date
                ) AS close_forward
            FROM price_normalized
        ),
        joined AS (
            -- Join with institutional scores
            SELECT
                p.symbol,
                p.week_date,
                p.close,
                p.close_forward,
                (p.close_forward - p.close) / p.close AS forward_return,
                s.score_total
            FROM price_with_forward p
            INNER JOIN weekly_score s
                ON p.symbol = s.symbol
                AND p.week_date = s.week_start
            WHERE p.close_forward IS NOT NULL
              AND p.close > 0
        ),
        signals AS (
            -- Filter to signal weeks only
            SELECT
                symbol,
                week_date,
                forward_return,
                CASE WHEN forward_return >= {return_threshold} THEN 1 ELSE 0 END AS is_win
            FROM joined
            WHERE score_total >= {score_threshold}
        ),
        aggregated AS (
            SELECT
                symbol,
                COUNT(*)       AS total_signals,
                SUM(is_win)    AS wins,
                ROUND(1.0 * SUM(is_win) / COUNT(*), 4) AS win_rate
            FROM signals
            GROUP BY symbol
            HAVING COUNT(*) >= {min_signals}
        )
        SELECT
            symbol,
            {score_threshold}  AS score_threshold,
            {n_weeks}          AS n_weeks,
            {return_threshold} AS return_threshold,
            total_signals,
            wins,
            win_rate
        FROM aggregated
        ORDER BY win_rate DESC
    """).pl()

    if result.is_empty():
        return 0

    con.execute("""
        INSERT OR REPLACE INTO win_rate
        SELECT symbol, score_threshold, n_weeks, return_threshold,
               total_signals, wins, win_rate
        FROM result
    """)

    return len(result)


def run(
    score_threshold: int = 4,
    n_weeks: int = 4,
    return_threshold: float = 0.05,
    min_signals: int = 3,
) -> None:
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    # Check prerequisites
    price_count = con.execute("SELECT COUNT(*) FROM weekly_price").fetchone()[0]
    score_count = con.execute("SELECT COUNT(*) FROM weekly_score").fetchone()[0]

    if price_count == 0:
        print("[WARN] weekly_price is empty — run fetch_price.py first")
        con.close()
        return
    if score_count == 0:
        print("[WARN] weekly_score is empty — run score_institutional.py first")
        con.close()
        return

    print(f"[INFO] weekly_price: {price_count} rows | weekly_score: {score_count} rows")
    print(f"[INFO] Params — score>={score_threshold}, {n_weeks}w forward, "
          f">={return_threshold*100:.0f}% return, min {min_signals} signals")

    n_written = compute_win_rate(con, score_threshold, n_weeks, return_threshold, min_signals)

    if n_written == 0:
        total_score_weeks = con.execute(
            "SELECT COUNT(DISTINCT week_start) FROM weekly_score"
        ).fetchone()[0]
        print(f"[WARN] 0 stocks written. Only {total_score_weeks} week(s) of score data.")
        print(f"       Need >={n_weeks + min_signals} weeks of score data for min_signals={min_signals}.")
        print("       Run fetch_institutional.py with a larger lookback_days (e.g. 365) to fix this.")
    else:
        print(f"[DONE] win_rate written for {n_written} stocks")

        # Top 10 preview
        top = con.execute("""
            SELECT symbol, total_signals, wins, win_rate
            FROM win_rate
            ORDER BY win_rate DESC
            LIMIT 10
        """).fetchall()
        print("\n[TOP 10 by win rate]")
        print(f"  {'symbol':<10} {'signals':>8} {'wins':>6} {'win_rate':>10}")
        for sym, sigs, wins, wr in top:
            print(f"  {sym:<10} {sigs:>8} {wins:>6} {wr*100:>9.1f}%")

    con.close()


if __name__ == "__main__":
    run()
