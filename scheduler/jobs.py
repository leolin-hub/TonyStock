"""
scheduler/jobs.py
APScheduler weekly pipeline — runs every Monday 08:00 Asia/Taipei.

Pipeline order:
  1. fetch_price          — update weekly OHLCV
  2. fetch_institutional  — fetch last 14 days of 三大法人 (incremental)
  3. score_institutional  — recompute 1-5 scores (full recompute)
  4. calc_win_rate        — recompute win rates (full recompute)

CLI:
  python -m scheduler.jobs          # start weekly cron
  python -m scheduler.jobs --now    # run pipeline once (incremental, 14d)
  python -m scheduler.jobs --init   # run pipeline once (historical, 90d) then exit
"""

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from scheduler.fetch_price import run as fetch_price, run_all_tw as fetch_price_all_tw
from scheduler.fetch_institutional import run as fetch_institutional
from scheduler.score_institutional import run as score_institutional
from scheduler.calc_win_rate import run as calc_win_rate
from scheduler.train_lgbm import run as train_lgbm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run_pipeline(lookback_days: int = 14) -> None:
    log.info("=" * 50)
    log.info("Pipeline started")

    steps = [
        ("fetch_price",         lambda: fetch_price_all_tw()),
        ("fetch_institutional", lambda: fetch_institutional(lookback_days=lookback_days)),
        ("score_institutional", lambda: score_institutional()),
        ("calc_win_rate",       lambda: calc_win_rate()),
        ("train_lgbm",          lambda: train_lgbm()),
    ]

    for name, fn in steps:
        log.info(f"[{name}] starting ...")
        try:
            fn()
            log.info(f"[{name}] done")
        except Exception as e:
            log.error(f"[{name}] FAILED: {e}", exc_info=True)
            log.error("Pipeline aborted — fix the error and re-run manually.")
            return

    log.info("Pipeline completed")
    log.info("=" * 50)


def start() -> None:
    scheduler = BlockingScheduler(timezone="Asia/Taipei")

    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(day_of_week="mon", hour=8, minute=0),
        id="weekly_pipeline",
        name="Weekly data refresh",
        misfire_grace_time=3600,
    )

    log.info("Scheduler started. Waiting for Monday 08:00 Asia/Taipei ...")
    log.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "--init":
        import duckdb
        from pathlib import Path
        db_path = Path(__file__).parent.parent / "data" / "stocks.db"
        has_data = False
        if db_path.exists():
            try:
                con = duckdb.connect(str(db_path))
                count = con.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'weekly_institutional'"
                ).fetchone()[0]
                if count:
                    rows = con.execute("SELECT COUNT(*) FROM weekly_institutional").fetchone()[0]
                    pred_count = con.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_name = 'ml_predictions'"
                    ).fetchone()[0]
                    pred_rows = con.execute("SELECT COUNT(*) FROM ml_predictions").fetchone()[0] if pred_count else 0
                    has_data = rows > 0 and pred_rows > 0
                con.close()
            except Exception:
                pass
        if has_data:
            log.info("Init mode: DB already has data — skipping full init.")
        else:
            log.info("Init mode: fetching 365 days of historical data ...")
            run_pipeline(lookback_days=365)
    elif arg == "--now":
        log.info("Manual trigger: running pipeline (incremental 14d) ...")
        run_pipeline(lookback_days=14)
    else:
        start()
