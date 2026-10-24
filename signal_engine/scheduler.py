from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

from . import config
from .models import SignalType, Signal
from .database import (
    init_db,
    get_watchlist,
    save_prices,
    get_prices,
    save_signal,
    save_news,
    get_symbol_sentiment,
    set_status,
    backup_db,
)
from .data_fetcher import fetch_all
from .technical_analysis import compute_indicators
from .signal_generator import generate_signal
from .backtester import get_hit_rate, run_all_backtests
from .sentiment import run_sentiment_cycle
from .tracker import evaluate_pending_outcomes


def _completed_candles(df):
    """Drop the still-forming candle so signals only use finished data."""
    if df.empty:
        return df
    now = datetime.now(timezone.utc)
    return df[df.index + timedelta(minutes=config.CANDLE_MINUTES) <= now]


def run_price_and_signal_cycle(alert_callback: Optional[Callable[[Signal], None]] = None):
    """
    Fetch prices, compute indicators, generate signals and alert on new strong signals.
    A signal is only stored when a new completed candle exists, so closed markets
    don't produce a stream of identical signals overnight.
    """
    logger.info("Executing price & signal analysis cycle...")
    stats = {"symbols": 0, "new_signals": 0, "entries": 0, "alerts": 0, "errors": 0}
    try:
        watchlist = get_watchlist()
        if not watchlist:
            logger.warning("Watchlist is empty. No assets to process.")
            return

        prices = fetch_all(watchlist)
        if prices:
            save_prices(prices)
            logger.info(f"Saved {len(prices)} price updates to database.")

        for item in watchlist:
            stats["symbols"] += 1
            try:
                df = _completed_candles(get_prices(item.symbol, limit=config.LIVE_LOOKBACK_BARS))
                if len(df) < 30:
                    continue
                ind = compute_indicators(df, symbol=item.symbol)
                if not ind:
                    continue

                sentiment, news_count = get_symbol_sentiment(item.symbol, config.SENTIMENT_WINDOW_HOURS)
                sig = generate_signal(ind, market=item.market, sentiment=sentiment, news_count=news_count)
                sig.win_rate = get_hit_rate(item.symbol, sig.signal_type)

                if save_signal(sig) is None:
                    continue  # no new candle since the last signal (market closed or feed lagging)
                stats["new_signals"] += 1
                stats["entries"] += int(sig.is_entry)

                logger.info(
                    f"[{sig.market.value}] {sig.symbol}: {sig.signal_type.value}"
                    f"{' (new)' if sig.is_entry else ''} score {sig.confidence}, price {sig.current_price:.4f}"
                )

                # Alert only when a strong signal first appears, not on every 5-minute repeat
                if (
                    sig.is_entry
                    and sig.signal_type in (SignalType.STRONG_BUY, SignalType.STRONG_SELL)
                    and sig.confidence >= 75.0
                    and alert_callback
                ):
                    logger.info(f"Alert triggered for {sig.symbol} ({sig.signal_type.value})")
                    alert_callback(sig)
                    stats["alerts"] += 1
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error processing signals for {item.symbol}: {e}")
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Error in price & signal cycle: {e}")
    finally:
        set_status("price_cycle", stats)


def run_news_cycle():
    try:
        run_sentiment_cycle(save_news)
        set_status("news_cycle", {"ok": True})
    except Exception as e:
        logger.error(f"Error in news sentiment cycle: {e}")
        set_status("news_cycle", {"ok": False, "error": str(e)})


def run_tracking_cycle():
    try:
        graded = evaluate_pending_outcomes()
        set_status("tracking_cycle", {"graded": graded})
    except Exception as e:
        logger.error(f"Error in forward tracking cycle: {e}")
        set_status("tracking_cycle", {"error": str(e)})


def run_backtest_cycle():
    try:
        results, _, _ = run_all_backtests()
        set_status("backtest_cycle", {"symbols": len(results)})
        logger.info(f"Periodic backtest refreshed for {len(results)} symbols")
    except Exception as e:
        logger.error(f"Error in backtest cycle: {e}")
        set_status("backtest_cycle", {"error": str(e)})


def run_backup():
    path = backup_db(keep=config.BACKUP_KEEP)
    set_status("backup", {"path": str(path) if path else None})


def start_scheduler(alert_callback: Optional[Callable[[Signal], None]] = None):
    """Run the engine 24/7."""
    logger.info("Starting continuous 24/7 Signal Engine...")
    init_db()
    now = datetime.now(timezone.utc)

    sched = BlockingScheduler(
        timezone="UTC",
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 120},
    )
    sched.add_job(lambda: run_price_and_signal_cycle(alert_callback), "interval", minutes=5,
                  id="price_cycle", next_run_time=now)
    sched.add_job(run_news_cycle, "interval", minutes=15, id="news_cycle",
                  next_run_time=now + timedelta(seconds=30))
    sched.add_job(run_tracking_cycle, "interval", minutes=15, id="tracking_cycle",
                  next_run_time=now + timedelta(minutes=2))
    sched.add_job(run_backtest_cycle, "interval", hours=6, id="backtest_cycle",
                  next_run_time=now + timedelta(minutes=3))
    sched.add_job(run_backup, "cron", hour=4, minute=17, id="backup")

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Signal Engine stopped.")
