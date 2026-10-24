"""
Download historical 5-minute candles into the local database so the backtester
has months of data to work with instead of a few hours.

    python -m signal_engine.backfill            # all watchlist symbols
    python -m signal_engine.backfill --days 90 --symbols BTCUSDT ETHUSDT

Crypto (Binance) can go back as far as you like. Stocks and forex (Yahoo) are
limited to about 60 days of 5-minute data. After that, the running engine keeps
extending the history on its own.
"""
import argparse

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from .database import init_db, get_watchlist, save_prices, get_price_span
from .data_fetcher import fetch_binance_history, fetch_yfinance_history
from .models import MarketType


def backfill(days: int = 90, symbols=None):
    init_db()
    watchlist = get_watchlist()
    if symbols:
        wanted = {s.upper() for s in symbols}
        watchlist = [w for w in watchlist if w.symbol.upper() in wanted]
    for item in watchlist:
        if item.market == MarketType.CRYPTO:
            prices = fetch_binance_history(item.symbol, days=days)
        else:
            prices = fetch_yfinance_history(item.symbol, days=days)
        save_prices(prices)
        first, last, count = get_price_span(item.symbol)
        logger.info(f"{item.symbol}: fetched {len(prices)} candles; database now has {count} ({first} -> {last})")


def main():
    parser = argparse.ArgumentParser(description="Backfill historical 5m candles")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", nargs="*", help="Only these symbols (default: whole watchlist)")
    args = parser.parse_args()
    backfill(days=args.days, symbols=args.symbols)


if __name__ == "__main__":
    main()
