import os
from datetime import datetime, timezone
from typing import List
import pandas as pd
import yfinance as yf
from binance.client import Client as BinanceClient
from loguru import logger

from .models import MarketType, Price, WatchlistItem

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")


def get_binance_client() -> BinanceClient:
    """Initialize Binance client (works without keys for public ticker data)."""
    return BinanceClient(api_key=BINANCE_API_KEY or None, api_secret=BINANCE_API_SECRET or None)


def fetch_yfinance(symbol: str, interval: str = "5m", period: str = "5d") -> List[Price]:
    """Fetch OHLCV candle data from Yahoo Finance for Stocks and Forex."""
    prices: List[Price] = []
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            logger.warning(f"No price data returned from yfinance for {symbol}")
            return prices

        for idx, row in df.iterrows():
            ts = idx.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            prices.append(
                Price(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
            )
    except Exception as e:
        logger.error(f"Error fetching {symbol} from yfinance: {e}")
    return prices


def fetch_binance(symbol: str, interval: str = "5m", limit: int = 200) -> List[Price]:
    """Fetch OHLCV candle data from Binance for Crypto."""
    prices: List[Price] = []
    try:
        client = get_binance_client()
        # map 5m interval
        interval_map = {
            "1m": BinanceClient.KLINE_INTERVAL_1MINUTE,
            "5m": BinanceClient.KLINE_INTERVAL_5MINUTE,
            "15m": BinanceClient.KLINE_INTERVAL_15MINUTE,
            "1h": BinanceClient.KLINE_INTERVAL_1HOUR,
            "1d": BinanceClient.KLINE_INTERVAL_1DAY,
        }
        b_interval = interval_map.get(interval, BinanceClient.KLINE_INTERVAL_5MINUTE)
        klines = client.get_klines(symbol=symbol, interval=b_interval, limit=limit)

        for k in klines:
            # k[0]: open time ms, k[1]: open, k[2]: high, k[3]: low, k[4]: close, k[5]: volume
            ts = datetime.fromtimestamp(k[0] / 1000.0, tz=timezone.utc)
            prices.append(
                Price(
                    symbol=symbol,
                    timestamp=ts,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
            )
    except Exception as e:
        logger.error(f"Error fetching {symbol} from binance: {e}")
    return prices


def fetch_all(watchlist: List[WatchlistItem]) -> List[Price]:
    """Fetch recent price data for all items in the watchlist."""
    all_prices: List[Price] = []
    for item in watchlist:
        if not item.enabled:
            continue
        try:
            if item.market == MarketType.CRYPTO:
                fetched = fetch_binance(item.symbol)
            else:
                fetched = fetch_yfinance(item.symbol)
            all_prices.extend(fetched)
            logger.info(f"Fetched {len(fetched)} price points for {item.symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch prices for {item.symbol}: {e}")
    return all_prices
