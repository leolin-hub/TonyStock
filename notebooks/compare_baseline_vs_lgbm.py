"""
compare_baseline_vs_lgbm.py
Compares rule-based win_rate (baseline) vs LightGBM on the test set (2025+).

Baseline signal: score_total >= 4, predicted probability = stock's historical win_rate
LightGBM signal: win_prob from ml_predictions

Run: python notebooks/compare_baseline_vs_lgbm.py
"""

import sys
sys.path.insert(0, ".")

import duckdb
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

DB_PATH  = Path("data/stocks.db")
TEST_START = "2025-01-01"
RETURN_THRESHOLD = 0.05
N_WEEKS_FORWARD  = 4


def build_test_set(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """
    Reconstruct ground truth (is_win) + baseline prediction for test period.
    Joins: weekly_price + weekly_score + win_rate (baseline) + ml_predictions
    """
    return con.execute(f"""
        WITH price_with_label AS (
            SELECT
                p.symbol,
                LEFT(p.symbol, LENGTH(p.symbol) - 3) AS symbol_short,
                p.date AS week_start,
                p.close,
                LEAD(p.close, {N_WEEKS_FORWARD}) OVER (
                    PARTITION BY p.symbol ORDER BY p.date
                ) AS close_forward
            FROM weekly_price p
            WHERE p.symbol LIKE '%.TW'
              AND p.close > 0
        ),
        labeled AS (
            SELECT
                symbol,
                symbol_short,
                week_start,
                close,
                CASE
                    WHEN (close_forward / close - 1) >= {RETURN_THRESHOLD} THEN 1
                    ELSE 0
                END AS is_win
            FROM price_with_label
            WHERE close_forward IS NOT NULL
              AND week_start >= '{TEST_START}'
        )
        SELECT
            l.symbol_short    AS symbol,
            l.week_start,
            l.is_win,
            s.score_total,
            -- baseline: use historical win_rate as predicted probability
            -- only available if stock had signals in training period
            w.win_rate        AS baseline_prob,
            m.win_prob        AS lgbm_prob
        FROM labeled l
        INNER JOIN weekly_score s
            ON l.symbol_short = s.symbol
            AND l.week_start  = s.week_start
        LEFT JOIN win_rate w
            ON l.symbol_short = w.symbol
            AND w.score_threshold  = 4
            AND w.n_weeks          = 4
            AND w.return_threshold = {RETURN_THRESHOLD}
        LEFT JOIN ml_predictions m
            ON l.symbol_short = m.symbol
            AND l.week_start  = m.week_start
        ORDER BY l.symbol_short, l.week_start
    """).pl()


def print_section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def compare(df: pl.DataFrame) -> None:

    print_section("Test Set Overview")
    print(f"  Period     : {TEST_START} ~ present")
    print(f"  Total rows : {len(df):,}")
    print(f"  Symbols    : {df['symbol'].n_unique():,}")
    print(f"  Actual win rate : {df['is_win'].mean():.1%}")

    # ── AUC comparison ────────────────────────────────────────────────────────
    print_section("AUC Comparison (higher = better)")

    # Baseline: only rows where score_total >= 4 AND baseline_prob available
    bl = df.filter(
        (pl.col("score_total") >= 4) & pl.col("baseline_prob").is_not_null()
    )
    # LightGBM: rows where lgbm_prob available
    lg = df.filter(pl.col("lgbm_prob").is_not_null())

    if len(bl) > 0:
        bl_auc = roc_auc_score(bl["is_win"], bl["baseline_prob"])
        print(f"  Baseline (score>=4, historical win_rate) "
              f"  n={len(bl):>6,}   AUC = {bl_auc:.4f}")
    else:
        print("  Baseline: no data")

    if len(lg) > 0:
        lg_auc = roc_auc_score(lg["is_win"], lg["lgbm_prob"])
        print(f"  LightGBM                                 "
              f"  n={len(lg):>6,}   AUC = {lg_auc:.4f}")

    # ── Precision at threshold ────────────────────────────────────────────────
    print_section("Actual Win Rate by Confidence Bucket")

    print("\n  [Baseline] score_total >= 4, grouped by historical win_rate")
    print(f"  {'win_rate bucket':<22} {'signals':>8} {'actual_wins':>12} {'actual_win%':>12}")
    bins   = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
    labels = ["<30%", "30–40%", "40–50%", "50–60%", ">60%"]
    for (lo, hi), label in zip(bins, labels):
        sub = df.filter(
            (pl.col("score_total") >= 4) &
            pl.col("baseline_prob").is_not_null() &
            (pl.col("baseline_prob") >= lo) &
            (pl.col("baseline_prob") <  hi)
        )
        if len(sub) == 0:
            continue
        actual_pct = sub["is_win"].mean()
        print(f"  {label:<22} {len(sub):>8,} {sub['is_win'].sum():>12,} {actual_pct:>11.1%}")

    print(f"\n  [LightGBM] grouped by win_prob")
    print(f"  {'win_prob bucket':<22} {'rows':>8} {'actual_wins':>12} {'actual_win%':>12}")
    prob_bins   = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0)]
    prob_labels = ["<20%", "20–30%", "30–40%", "40–50%", ">50%"]
    for (lo, hi), label in zip(prob_bins, prob_labels):
        sub = df.filter(
            pl.col("lgbm_prob").is_not_null() &
            (pl.col("lgbm_prob") >= lo) &
            (pl.col("lgbm_prob") <  hi)
        )
        if len(sub) == 0:
            continue
        actual_pct = sub["is_win"].mean()
        print(f"  {label:<22} {len(sub):>8,} {sub['is_win'].sum():>12,} {actual_pct:>11.1%}")

    # ── Top stocks comparison ─────────────────────────────────────────────────
    print_section("Latest Week — Top 10 by LightGBM win_prob")
    latest_week = df["week_start"].max()
    top = (
        df.filter(pl.col("week_start") == latest_week)
          .filter(pl.col("lgbm_prob").is_not_null())
          .sort("lgbm_prob", descending=True)
          .head(10)
          .select(["symbol", "score_total", "baseline_prob", "lgbm_prob"])
    )
    print(f"  Week: {latest_week}")
    print(f"  {'symbol':<8} {'score_total':>12} {'baseline_wr':>12} {'lgbm_prob':>10}")
    for row in top.iter_rows():
        sym, score, bl_wr, lg_prob = row
        bl_str = f"{bl_wr:.1%}" if bl_wr is not None else "  N/A"
        print(f"  {sym:<8} {str(score):>12} {bl_str:>12} {lg_prob:>9.1%}")


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Building test set ...")
    df = build_test_set(con)

    if df.is_empty():
        print("[WARN] No test data found.")
        return

    compare(df)
    con.close()


if __name__ == "__main__":
    main()
