from datetime import datetime
from typing import Callable, Optional
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

from .models import SignalType, MarketType, Signal
from .database import (
    init_db,
    get_watchlist,
    save_prices,
    get_prices,
    save_signal,
    save_news,
)
from .data_fetcher import fetch_all
from .technical_analysis import compute_indicators
from .signal_generator import generate_signal
from .backtester import get_win_rate
from .sentiment import run_sentiment_cycle


def run_price_and_signal_cycle(alert_callback: Optional[Callable[[Signal], None]] = None):
    """
    Fetch market prices, compute indicators, generate confluence signals,
    and trigger alerts for high-conviction trades.
    """
    logger.info("Executing price & signal analysis cycle...")
    try:
        watchlist = get_watchlist()
        if not watchlist:
            logger.warning("Watchlist is empty. No assets to process.")
            return

        # 1. Fetch latest prices
        prices = fetch_all(watchlist)
        if prices:
            save_prices(prices)
            logger.info(f"Saved {len(prices)} price updates to database.")

        # 2. Analyze each symbol
        for item in watchlist:
            try:
                df = get_prices(item.symbol, hours=72)
                if df.empty or len(df) < 20:
                    continue

                ind = compute_indicators(df, symbol=item.symbol)
                if not ind:
                    continue

                # Estimate win rate from historical backtest
                win_rate = get_win_rate(item.symbol, SignalType.BUY, get_prices)
                sig = generate_signal(ind, market=item.market, win_rate=win_rate)
                save_signal(sig)

                logger.info(
                    f"[{sig.market.value}] {sig.symbol}: {sig.signal_type.value} "
                    f"(Conf: {sig.confidence}%, Price: ${sig.current_price:.2f})"
                )

                # Trigger alerts for strong actionable signals
                if sig.signal_type in (SignalType.STRONG_BUY, SignalType.STRONG_SELL):
                    if sig.confidence >= 75.0 and alert_callback:
                        logger.info(f"Alert triggered for {sig.symbol} ({sig.signal_type.value})")
                        alert_callback(sig)

            except Exception as e:
                logger.error(f"Error processing signals for {item.symbol}: {e}")

    except Exception as e:
        logger.error(f"Error in price & signal cycle: {e}")


def run_news_cycle():
    """Fetch news and score sentiment."""
    try:
        run_sentiment_cycle(save_news)
    except Exception as e:
        logger.error(f"Error in news sentiment cycle: {e}")


def start_scheduler(alert_callback: Optional[Callable[[Signal], None]] = None):
    """Start APScheduler to run signal engine 24/7."""
    logger.info("Starting continuous 24/7 Signal Engine...")
    init_db()

    # Run immediate initial cycle on startup
    run_price_and_signal_cycle(alert_callback)
    run_news_cycle()

    sched = BlockingScheduler()
    # Market prices & signals every 5 minutes
    sched.add_job(
        lambda: run_price_and_signal_cycle(alert_callback),
        "interval",
        minutes=5,
        id="price_cycle",
    )
    # News & sentiment every 15 minutes
    sched.add_job(
        run_news_cycle,
        "interval",
        minutes=15,
        id="news_cycle",
    )

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Signal Engine stopped.")
