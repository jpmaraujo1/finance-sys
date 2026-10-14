import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import httpx
import yfinance as yf
from binance.client import Client as BinanceClient
from loguru import logger

from .models import MarketType, Price, WatchlistItem

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

CNBC_QUOTE_URL = "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
CNBC_CHART_URL = "https://ts-api.cnbc.com/harmony/app/charts/5D.json"
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}


def get_binance_client() -> BinanceClient:
    """Initialize Binance client (works without keys for public ticker data)."""
    return BinanceClient(api_key=BINANCE_API_KEY or None, api_secret=BINANCE_API_SECRET or None)


def is_crypto_symbol(symbol: str) -> bool:
    upper = (symbol or "").upper()
    return upper.endswith("USDT") or upper.endswith("USDC")


def cnbc_symbol(symbol: str) -> Optional[str]:
    """Map a watchlist symbol onto a CNBC stock/forex listing.

    Crypto has no stock listing here, so the caller skips CNBC.
    Unknown forex pairs are skipped the same way.
    """
    if not symbol or is_crypto_symbol(symbol):
        return None
    if symbol.endswith("=X"):
        pair = symbol[:-2].upper()
        if len(pair) == 6 and pair.endswith("USD"):
            return pair[:3] + "="
        if len(pair) == 6 and pair.startswith("USD"):
            return pair[3:] + "="
        return None
    return symbol.upper()


def _parse_number(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("+", "")
    if not text or text.upper() in {"N/A", "N/D", "UNCH", "NONE", "NULL"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number:
        return None
    return number


def parse_cnbc_quote(payload: dict) -> dict:
    """Turn a CNBC quote payload into a labeled snapshot.

    `code` other than 0 means CNBC has no listing for that symbol.
    """
    try:
        quote = payload["FormattedQuoteResult"]["FormattedQuote"][0]
    except (KeyError, IndexError, TypeError):
        return {"source": "CNBC", "status": "unavailable"}
    if quote.get("code") != 0:
        return {"source": "CNBC", "status": "no_listing"}
    price = _parse_number(quote.get("last"))
    if price is None:
        return {"source": "CNBC", "status": "no_listing"}
    return {
        "source": "CNBC",
        "status": "ok",
        "price": price,
        "as_of": _parse_cnbc_time(quote.get("last_time")),
        "currency": quote.get("currencyCode") or None,
    }


def _parse_cnbc_time(raw) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip()
    if "T" in text:
        if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
            text = text[:-2] + ":" + text[-2:]
        try:
            return datetime.fromisoformat(text).isoformat()
        except ValueError:
            return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return None


def prices_from_cnbc_chart(symbol: str, payload: dict) -> List[Price]:
    bars = ((payload or {}).get("barData") or {}).get("priceBars") or []
    prices: List[Price] = []
    for bar in bars:
        close = _parse_number(bar.get("close"))
        millis = bar.get("tradeTimeinMills")
        if close is None or not millis:
            continue
        ts = datetime.fromtimestamp(float(millis) / 1000.0, tz=timezone.utc)
        prices.append(
            Price(
                symbol=symbol,
                timestamp=ts,
                open=_parse_number(bar.get("open")) or close,
                high=_parse_number(bar.get("high")) or close,
                low=_parse_number(bar.get("low")) or close,
                close=close,
                volume=_parse_number(bar.get("volume")) or 0.0,
            )
        )
    return prices


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


def fetch_cnbc_prices(symbol: str, market: Optional[MarketType] = None) -> List[Price]:
    """Recent CNBC candles for a stock, ETF, forex pair, or B3 name.

    Returns an empty list when the symbol is crypto, unmapped, or has no
    CNBC listing. Callers keep the watchlist row either way.
    """
    del market  # market is inferred from the symbol; kept for call-site clarity
    mapped = cnbc_symbol(symbol)
    if not mapped:
        return []
    try:
        resp = httpx.get(
            CNBC_CHART_URL,
            params={"symbol": mapped},
            headers=HTTP_HEADERS,
            timeout=12.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        prices = prices_from_cnbc_chart(symbol, resp.json())
        if not prices:
            logger.info(f"CNBC has no candles for {symbol} ({mapped})")
        return prices
    except Exception as e:
        logger.warning(f"CNBC candles failed for {symbol}: {e}")
        return []


def fetch_binance(symbol: str, interval: str = "5m", limit: int = 200) -> List[Price]:
    """Fetch OHLCV candle data from Binance for Crypto."""
    prices: List[Price] = []
    try:
        client = get_binance_client()
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


def fetch_yahoo_quote(symbol: str) -> dict:
    """Latest Yahoo close. Failure stays on this source and does not raise."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5d", interval="1d")
        if df is None or df.empty or "Close" not in df.columns:
            return {"source": "Yahoo Finance", "status": "no_listing"}
        df = df.dropna(subset=["Close"])
        if df.empty:
            return {"source": "Yahoo Finance", "status": "no_listing"}
        last = df.iloc[-1]
        ts = df.index[-1].to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        price = float(last["Close"])
        if price != price:
            return {"source": "Yahoo Finance", "status": "no_listing"}
        return {
            "source": "Yahoo Finance",
            "status": "ok",
            "price": price,
            "as_of": ts.isoformat(),
        }
    except Exception as e:
        logger.warning(f"Yahoo quote failed for {symbol}: {e}")
        return {"source": "Yahoo Finance", "status": "unavailable"}


def fetch_cnbc_quote(symbol: str) -> dict:
    """Latest CNBC quote. No listing and outages are separate statuses."""
    mapped = cnbc_symbol(symbol)
    if not mapped:
        return {"source": "CNBC", "status": "no_listing"}
    try:
        resp = httpx.get(
            CNBC_QUOTE_URL,
            params={
                "symbols": mapped,
                "requestMethod": "itv",
                "noform": "1",
                "partnerId": "2",
                "fund": "1",
                "exthrs": "1",
                "output": "json",
                "events": "1",
            },
            headers=HTTP_HEADERS,
            timeout=8.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return parse_cnbc_quote(resp.json())
    except Exception as e:
        logger.warning(f"CNBC quote failed for {symbol}: {e}")
        return {"source": "CNBC", "status": "unavailable"}


def fetch_binance_quote(symbol: str) -> dict:
    try:
        client = get_binance_client()
        data = client.get_symbol_ticker(symbol=symbol)
        price = _parse_number(data.get("price"))
        if price is None:
            return {"source": "Binance", "status": "no_listing"}
        return {
            "source": "Binance",
            "status": "ok",
            "price": price,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        message = str(e).lower()
        if "invalid symbol" in message or "unknown symbol" in message:
            return {"source": "Binance", "status": "no_listing"}
        logger.warning(f"Binance quote failed for {symbol}: {e}")
        return {"source": "Binance", "status": "unavailable"}


def _quote_or_unavailable(fn, source: str, timeout: float) -> dict:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            logger.warning(f"{source} quote timed out: {e}")
            return {"source": source, "status": "unavailable"}


def fetch_quote_pair(symbol: str) -> List[dict]:
    """Live quotes for the symbol view. One source failing leaves the other.

    Crypto uses Binance. CNBC is the keyless stock alternate and is skipped
    when the symbol has no stock or forex listing.
    """
    if is_crypto_symbol(symbol):
        primary = _quote_or_unavailable(lambda: fetch_binance_quote(symbol), "Binance", 8)
        return [primary, {"source": "CNBC", "status": "no_listing"}]

    yahoo = {"source": "Yahoo Finance", "status": "unavailable"}
    cnbc = {"source": "CNBC", "status": "unavailable"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        yahoo_future = pool.submit(fetch_yahoo_quote, symbol)
        cnbc_future = pool.submit(fetch_cnbc_quote, symbol)
        try:
            yahoo = yahoo_future.result(timeout=12)
        except Exception as e:
            logger.warning(f"Yahoo quote timed out for {symbol}: {e}")
            yahoo = {"source": "Yahoo Finance", "status": "unavailable"}
        try:
            cnbc = cnbc_future.result(timeout=12)
        except Exception as e:
            logger.warning(f"CNBC quote timed out for {symbol}: {e}")
            cnbc = {"source": "CNBC", "status": "unavailable"}
    return [yahoo, cnbc]


def fetch_all(watchlist: List[WatchlistItem]) -> List[Price]:
    """Fetch recent price data for all items in the watchlist.

    Stocks, ETFs, forex, and B3 use Yahoo. If Yahoo returns nothing, CNBC
    candles fill the gap. Crypto stays on Binance. A missing CNBC listing
    skips that source only.
    """
    all_prices: List[Price] = []
    for item in watchlist:
        if not item.enabled:
            continue
        try:
            if item.market == MarketType.CRYPTO or is_crypto_symbol(item.symbol):
                fetched = fetch_binance(item.symbol)
            else:
                fetched = fetch_yfinance(item.symbol)
                if not fetched:
                    fetched = fetch_cnbc_prices(item.symbol, item.market)
                    if fetched:
                        logger.info(f"{item.symbol}: Yahoo empty, using {len(fetched)} CNBC candles")
            all_prices.extend(fetched)
            logger.info(f"Fetched {len(fetched)} price points for {item.symbol}")
        except Exception as e:
            logger.error(f"Failed to fetch prices for {item.symbol}: {e}")
    return all_prices
