"""
score_institutional.py
Computes 1-5 cross-sectional institutional score per stock per week
using quantile ranking, then stores into weekly_score table.

Scoring logic (per week, all stocks ranked together):
  bottom 20%  → score 1  (heavy selling)
  20–40%      → score 2
  40–60%      → score 3
  60–80%      → score 4
  top 20%     → score 5  (heavy buying)

Applied independently to: foreign / trust / dealer / total (三大法人合計)
Nulls are treated as 0 (no activity).
"""

import duckdb
import polars as pl
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "stocks.db"

SCORE_COLS = {
    "score_foreign": "foreign_net_sum",
    "score_trust":   "trust_net_sum",
    "score_dealer":  "dealer_net_sum",
    "score_total":   "total_net_sum",
}


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS weekly_score (
            symbol        VARCHAR NOT NULL,
            week_start    DATE    NOT NULL,
            score_foreign TINYINT,
            score_trust   TINYINT,
            score_dealer  TINYINT,
            score_total   TINYINT,
            PRIMARY KEY (symbol, week_start)
        )
    """)


def _score_expr(net_col: str, score_col: str) -> pl.Expr:
    """
    Cross-sectional 1-5 score for net_col within each week_start group.
    pct = rank / count  (0 < pct <= 1.0)
    """
    pct = (
        pl.col(net_col)
          .fill_null(0)
          .rank(method="min", descending=False)
          .over("week_start")
        / pl.col(net_col).fill_null(0).count().over("week_start")
    )
    return (
        pl.when(pct <= 0.2).then(pl.lit(1, dtype=pl.Int8))
          .when(pct <= 0.4).then(pl.lit(2, dtype=pl.Int8))
          .when(pct <= 0.6).then(pl.lit(3, dtype=pl.Int8))
          .when(pct <= 0.8).then(pl.lit(4, dtype=pl.Int8))
          .otherwise(pl.lit(5, dtype=pl.Int8))
          .alias(score_col)
    )


def compute_scores(df: pl.DataFrame) -> pl.DataFrame:
    score_exprs = [_score_expr(net, score) for score, net in SCORE_COLS.items()]
    return (
        df.with_columns(score_exprs)
          .select(["symbol", "week_start"] + list(SCORE_COLS.keys()))
    )


def run() -> None:
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    df = con.execute("SELECT * FROM weekly_institutional").pl()
    if df.is_empty():
        print("[WARN] weekly_institutional is empty — run fetch_institutional.py first")
        con.close()
        return

    print(f"[INFO] Loaded {len(df)} rows from weekly_institutional")

    score_df = compute_scores(df)

    con.execute("""
        INSERT OR REPLACE INTO weekly_score
        SELECT symbol, week_start, score_foreign, score_trust, score_dealer, score_total
        FROM score_df
    """)

    total = con.execute("SELECT COUNT(*) FROM weekly_score").fetchone()[0]
    print(f"[DONE] weekly_score total rows: {total}")

    # Sanity: score distribution should be ~20% each
    dist = con.execute("""
        SELECT score_total, COUNT(*) AS cnt,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM weekly_score
        GROUP BY score_total
        ORDER BY score_total
    """).fetchall()
    print("[INFO] score_total distribution:")
    for score, cnt, pct in dist:
        bar = "#" * int(pct / 2)
        print(f"  {score}  {bar:<25}  {cnt:>5} stocks  ({pct}%)")

    con.close()


if __name__ == "__main__":
    run()
