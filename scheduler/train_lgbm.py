"""
scheduler/train_lgbm.py
Trains a LightGBM binary classifier to predict next-week price movement.

y: is_win = 1 if close[+1w] >= close * 1.20  (20% in 1 week)
X: institutional raw volumes (log-normalized) + price momentum + technical indicators

Technical indicators added (v3):
  RSI(14), MACD histogram (normalized), Bollinger Band position,
  ATR ratio (volatility), MA20 slope (trend direction)

Time-series split:
  train : week_start < 2024-01-01
  val   : 2024-01-01 <= week_start < 2025-01-01
  test  : week_start >= 2025-01-01  (held out)
"""

import joblib
import duckdb
import lightgbm as lgb
import polars as pl
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

DB_PATH         = Path(__file__).parent.parent / "data" / "stocks.db"
MODEL_PATH      = Path(__file__).parent.parent / "data" / "lgbm_model.txt"
CALIBRATOR_PATH = Path(__file__).parent.parent / "data" / "calibrator.joblib"

TRAIN_END        = "2024-01-01"
VAL_END          = "2025-01-01"
RETURN_THRESHOLD = 0.20
N_WEEKS_FORWARD  = 1


FEATURE_COLS = [
    # Raw institutional volumes (log-normalized)
    "foreign_net_log", "trust_net_log", "dealer_net_log", "total_net_log",
    "total_net_log_lag1", "total_net_log_lag2",
    # Price momentum
    "return_1w", "return_4w", "return_12w",
    # Volume
    "vol_ratio",
    # Price position
    "close_vs_52w_high",
    # Technical indicators
    "rsi", "macd_norm", "bb_position", "atr_ratio", "ma20_slope",
]


# ── Feature engineering ───────────────────────────────────────────────────────

def log_norm(col: str) -> pl.Expr:
    """sign(x) * log(1 + |x|) — preserves direction, compresses magnitude."""
    return (
        pl.col(col).sign() *
        (pl.col(col).abs() + 1).log(base=10)
    ).alias(f"{col}_log")


def build_features(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    raw = con.execute("""
        SELECT
            p.symbol,
            p.date          AS week_start,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume,
            s.score_foreign,
            s.score_trust,
            s.score_dealer,
            s.score_total,
            i.foreign_net_sum,
            i.trust_net_sum,
            i.dealer_net_sum,
            i.total_net_sum
        FROM weekly_price p
        INNER JOIN weekly_score s
            ON LEFT(p.symbol, LENGTH(p.symbol) - 3) = s.symbol
            AND p.date = s.week_start
        INNER JOIN weekly_institutional i
            ON LEFT(p.symbol, LENGTH(p.symbol) - 3) = i.symbol
            AND p.date = i.week_start
        WHERE p.symbol LIKE '%.TW'
          AND p.close > 0
        ORDER BY p.symbol, p.date
    """).pl()

    # Step 1: log-normalize institutional volumes
    df = raw.with_columns([
        log_norm("foreign_net_sum"),
        log_norm("trust_net_sum"),
        log_norm("dealer_net_sum"),
        log_norm("total_net_sum"),
    ])

    # Step 2: price momentum + volume + forward label + intermediate tech columns
    df = df.with_columns([
        # Price momentum
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("return_1w"),
        (pl.col("close") / pl.col("close").shift(4).over("symbol") - 1).alias("return_4w"),
        (pl.col("close") / pl.col("close").shift(12).over("symbol") - 1).alias("return_12w"),

        # Volume ratio vs 4w avg
        (pl.col("volume") /
         pl.col("volume").rolling_mean(window_size=4).over("symbol")
        ).alias("vol_ratio"),

        # Close vs 52w high
        (pl.col("close") /
         pl.col("close").rolling_max(window_size=52).over("symbol")
        ).alias("close_vs_52w_high"),

        # Institutional lags
        pl.col("total_net_sum_log").shift(1).over("symbol").alias("total_net_log_lag1"),
        pl.col("total_net_sum_log").shift(2).over("symbol").alias("total_net_log_lag2"),

        # Forward label
        pl.col("close").shift(-N_WEEKS_FORWARD).over("symbol").alias("close_forward"),

        # --- Technical indicator intermediates ---

        # RSI: price delta gain/loss
        (pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("_delta"),

        # True Range components for ATR
        (pl.col("high") - pl.col("low")).alias("_hl"),
        (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs().alias("_hpc"),
        (pl.col("low")  - pl.col("close").shift(1).over("symbol")).abs().alias("_lpc"),

        # MACD: rolling mean as EMA proxy
        pl.col("close").rolling_mean(window_size=12).over("symbol").alias("_ema12"),
        pl.col("close").rolling_mean(window_size=26).over("symbol").alias("_ema26"),

        # Bollinger Band
        pl.col("close").rolling_mean(window_size=20).over("symbol").alias("_ma20"),
        pl.col("close").rolling_std(window_size=20).over("symbol").alias("_std20"),
    ])

    # Step 3: RSI gain/loss split + True Range
    df = df.with_columns([
        pl.col("_delta").clip(lower_bound=0).alias("_gain"),
        pl.col("_delta").clip(upper_bound=0).abs().alias("_loss"),
        pl.max_horizontal("_hl", "_hpc", "_lpc").alias("_tr"),
    ])

    # Step 4: rolling averages for RSI + ATR
    df = df.with_columns([
        pl.col("_gain").rolling_mean(window_size=14).over("symbol").alias("_avg_gain"),
        pl.col("_loss").rolling_mean(window_size=14).over("symbol").alias("_avg_loss"),
        pl.col("_tr").rolling_mean(window_size=14).over("symbol").alias("_atr"),
    ])

    # Step 5: final technical indicators
    df = df.with_columns([
        # RSI (0–1 scale; 0.7+ = overbought, 0.3- = oversold)
        (pl.col("_avg_gain") /
         (pl.col("_avg_gain") + pl.col("_avg_loss") + 1e-8)
        ).alias("rsi"),

        # MACD histogram normalized by close (dimensionless)
        ((pl.col("_ema12") - pl.col("_ema26")) / pl.col("close")).alias("macd_norm"),

        # Bollinger Band position (-1 to 1; positive = above MA)
        ((pl.col("close") - pl.col("_ma20")) /
         (2 * pl.col("_std20") + 1e-8)
        ).alias("bb_position"),

        # ATR ratio: volatility relative to price
        (pl.col("_atr") / pl.col("close")).alias("atr_ratio"),

        # MA20 slope: 4-week change rate of MA20 (trend direction)
        (pl.col("_ma20") / pl.col("_ma20").shift(4).over("symbol") - 1).alias("ma20_slope"),
    ])

    # Rename institutional log columns
    df = df.rename({
        "foreign_net_sum_log": "foreign_net_log",
        "trust_net_sum_log":   "trust_net_log",
        "dealer_net_sum_log":  "dealer_net_log",
        "total_net_sum_log":   "total_net_log",
    })

    # Win label
    df = df.with_columns([
        ((pl.col("close_forward") / pl.col("close") - 1) >= RETURN_THRESHOLD)
        .cast(pl.Int8)
        .alias("is_win")
    ])

    return (
        df.select(["symbol", "week_start", "close", "is_win"] + FEATURE_COLS)
          .drop_nulls()
    )


# ── Train ─────────────────────────────────────────────────────────────────────

def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("week_start") < pl.lit(TRAIN_END).str.to_date())
    val   = df.filter(
        (pl.col("week_start") >= pl.lit(TRAIN_END).str.to_date()) &
        (pl.col("week_start") <  pl.lit(VAL_END).str.to_date())
    )
    test  = df.filter(pl.col("week_start") >= pl.lit(VAL_END).str.to_date())
    return train, val, test


def train(df: pl.DataFrame) -> lgb.Booster:
    train_df, val_df, _ = split(df)

    print(f"[INFO] Train: {len(train_df):,} | Val: {len(val_df):,}")
    print(f"[INFO] Train win rate: {train_df['is_win'].mean():.1%}  "
          f"(n_weeks={N_WEEKS_FORWARD}, threshold={RETURN_THRESHOLD:.0%})")

    X_tr = train_df[FEATURE_COLS].to_numpy()
    y_tr = train_df["is_win"].to_numpy()
    X_vl = val_df[FEATURE_COLS].to_numpy()
    y_vl = val_df["is_win"].to_numpy()

    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "learning_rate":    0.05,
        "num_leaves":       31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "scale_pos_weight": pos_weight,
        "verbose":         -1,
        "seed":             42,
    }

    model = lgb.train(
        params, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    val_pred = model.predict(X_vl)
    auc = roc_auc_score(y_vl, val_pred)
    print(f"[INFO] Validation AUC: {auc:.4f}")

    imp = sorted(
        zip(FEATURE_COLS, model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True
    )
    max_gain = imp[0][1] if imp[0][1] > 0 else 1
    print("\n[INFO] Feature importance (gain):")
    for feat, score in imp:
        bar = "#" * int(score / max_gain * 25)
        print(f"  {feat:<30} {bar}")

    # ── Calibration (Isotonic Regression on val set) ───────────────────────────
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_pred, y_vl)

    # Show before/after calibration on val set by decile
    import numpy as np
    cal_pred = calibrator.predict(val_pred)
    print("\n[INFO] Calibration check (val set, by raw prob decile):")
    print(f"  {'raw prob bucket':<18} {'n':>7} {'actual%':>9} {'before':>9} {'after':>9}")
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        mask = (val_pred >= lo) & (val_pred < hi)
        if mask.sum() == 0:
            continue
        actual  = y_vl[mask].mean()
        before  = val_pred[mask].mean()
        after   = cal_pred[mask].mean()
        print(f"  {lo:.0%}–{hi:.0%}              {mask.sum():>7,} {actual:>8.2%} {before:>8.2%} {after:>8.2%}")

    joblib.dump(calibrator, str(CALIBRATOR_PATH))
    print(f"[INFO] Calibrator saved → {CALIBRATOR_PATH}")

    return model, calibrator


# ── Store predictions ─────────────────────────────────────────────────────────

def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            symbol     VARCHAR NOT NULL,
            week_start DATE    NOT NULL,
            win_prob   DOUBLE,
            PRIMARY KEY (symbol, week_start)
        )
    """)


def store_predictions(
    con: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
    model: lgb.Booster,
    calibrator: IsotonicRegression,
) -> None:
    raw_probs = model.predict(df[FEATURE_COLS].to_numpy())
    cal_probs = calibrator.predict(raw_probs)
    pred_df = (
        df.select([
            pl.col("symbol").str.replace(r"\.TW$", ""),
            pl.col("week_start"),
            pl.Series("win_prob", cal_probs).round(4),
        ])
    )
    con.execute("""
        INSERT OR REPLACE INTO ml_predictions
        SELECT symbol, week_start, win_prob FROM pred_df
    """)
    total = con.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
    print(f"[INFO] ml_predictions: {total:,} rows stored (calibrated)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    print("[INFO] Building features ...")
    df = build_features(con)
    win_rate = df['is_win'].mean()
    win_rate_str = f"{win_rate:.1%}" if win_rate is not None else "N/A (empty dataset)"
    print(f"[INFO] Total: {len(df):,} rows | {df['symbol'].n_unique():,} symbols | "
          f"win rate: {win_rate_str}")

    if len(df) == 0:
        print("[WARN] Empty dataset after feature engineering — skipping training.")
        con.close()
        return

    print("\n[INFO] Training ...")
    model, calibrator = train(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[INFO] Model saved → {MODEL_PATH}")

    print("\n[INFO] Storing predictions ...")
    store_predictions(con, df, model, calibrator)
    con.close()
    print("[DONE]")


if __name__ == "__main__":
    run()
