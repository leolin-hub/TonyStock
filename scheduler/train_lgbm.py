"""
scheduler/train_lgbm.py
Trains a LightGBM binary classifier to predict next-week price movement.

y: is_win = 1 if close[+N_WEEKS_FORWARD] >= close * (1 + RETURN_THRESHOLD)
X: raw institutional net buy volumes (log-normalized) + price features

Key changes vs v1:
  - Use raw foreign/trust/dealer net volumes instead of 1-5 scores
    (preserves magnitude; log-normalized to handle scale differences)
  - Score 1-5 kept as separate features (relative ranking signal)
  - n_weeks=1 (next-week prediction, shorter horizon)
  - return_threshold=0.05 (5% in 1 week is aggressive but trainable)

Time-series split:
  train : week_start < 2024-01-01
  val   : 2024-01-01 <= week_start < 2025-01-01
  test  : week_start >= 2025-01-01  (held out)

Outputs:
  - model saved to data/lgbm_model.txt
  - predictions stored in DuckDB table ml_predictions
"""

import duckdb
import lightgbm as lgb
import polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score

DB_PATH    = Path(__file__).parent.parent / "data" / "stocks.db"
MODEL_PATH = Path(__file__).parent.parent / "data" / "lgbm_model.txt"

TRAIN_END        = "2024-01-01"
VAL_END          = "2025-01-01"
RETURN_THRESHOLD = 0.05
N_WEEKS_FORWARD  = 1   # predict next week


FEATURE_COLS = [
    # Raw institutional volumes (log-normalized)
    "foreign_net_log", "trust_net_log", "dealer_net_log", "total_net_log",
    # Lag of total net (momentum of institutional flow)
    "total_net_log_lag1", "total_net_log_lag2",
    # Price features
    "return_1w", "return_4w", "return_12w",
    "vol_ratio", "close_vs_52w_high",
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

    df = (
        raw
        # Log-normalize raw volumes
        .with_columns([
            log_norm("foreign_net_sum"),
            log_norm("trust_net_sum"),
            log_norm("dealer_net_sum"),
            log_norm("total_net_sum"),
        ])
        .with_columns([
            # Price returns
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

            # Raw volume lags
            pl.col("total_net_sum_log").shift(1).over("symbol").alias("total_net_log_lag1"),
            pl.col("total_net_sum_log").shift(2).over("symbol").alias("total_net_log_lag2"),

            # Forward label
            pl.col("close").shift(-N_WEEKS_FORWARD).over("symbol").alias("close_forward"),
        ])
        .rename({
            "foreign_net_sum_log": "foreign_net_log",
            "trust_net_sum_log":   "trust_net_log",
            "dealer_net_sum_log":  "dealer_net_log",
            "total_net_sum_log":   "total_net_log",
        })
    )

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

    # Handle class imbalance
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLS)
    dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

    params = {
        "objective":         "binary",
        "metric":            "auc",
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_data_in_leaf":  50,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "scale_pos_weight":  pos_weight,
        "verbose":          -1,
        "seed":              42,
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

    return model


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


def store_predictions(con: duckdb.DuckDBPyConnection, df: pl.DataFrame, model: lgb.Booster) -> None:
    pred_df = (
        df.with_columns([
            pl.Series("win_prob", model.predict(df[FEATURE_COLS].to_numpy())).round(4),
            pl.col("symbol").str.replace(r"\.TW$", "").alias("symbol"),
        ])
        .select(["symbol", "week_start", "win_prob"])
    )
    con.execute("""
        INSERT OR REPLACE INTO ml_predictions
        SELECT symbol, week_start, win_prob FROM pred_df
    """)
    total = con.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0]
    print(f"[INFO] ml_predictions: {total:,} rows stored")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    con = duckdb.connect(str(DB_PATH))
    init_db(con)

    print("[INFO] Building features ...")
    df = build_features(con)
    print(f"[INFO] Total: {len(df):,} rows | {df['symbol'].n_unique():,} symbols | "
          f"win rate: {df['is_win'].mean():.1%}")

    print("\n[INFO] Training ...")
    model = train(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"[INFO] Model saved → {MODEL_PATH}")

    print("\n[INFO] Storing predictions ...")
    store_predictions(con, df, model)
    con.close()
    print("[DONE]")


if __name__ == "__main__":
    run()
