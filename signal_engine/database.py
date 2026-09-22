import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import pandas as pd
from loguru import logger

from .models import Signal, SignalType, MarketType, Price, NewsItem, WatchlistItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "finance.db"

DEFAULT_WATCHLIST = [
    WatchlistItem("AAPL", MarketType.US_STOCKS, "Apple", True),
    WatchlistItem("MSFT", MarketType.US_STOCKS, "Microsoft", True),
    WatchlistItem("NVDA", MarketType.US_STOCKS, "NVIDIA", True),
    WatchlistItem("TSLA", MarketType.US_STOCKS, "Tesla", True),
    WatchlistItem("AMZN", MarketType.US_STOCKS, "Amazon", True),
    WatchlistItem("GOOGL", MarketType.US_STOCKS, "Alphabet", True),
    WatchlistItem("META", MarketType.US_STOCKS, "Meta", True),
    WatchlistItem("SPY", MarketType.US_STOCKS, "S&P 500 ETF", True),
    WatchlistItem("QQQ", MarketType.US_STOCKS, "NASDAQ ETF", True),
    WatchlistItem("BTCUSDT", MarketType.CRYPTO, "Bitcoin", True),
    WatchlistItem("ETHUSDT", MarketType.CRYPTO, "Ethereum", True),
    WatchlistItem("SOLUSDT", MarketType.CRYPTO, "Solana", True),
    WatchlistItem("BNBUSDT", MarketType.CRYPTO, "BNB", True),
    WatchlistItem("XRPUSDT", MarketType.CRYPTO, "XRP", True),
    WatchlistItem("ADAUSDT", MarketType.CRYPTO, "Cardano", True),
    WatchlistItem("DOGEUSDT", MarketType.CRYPTO, "Dogecoin", True),
    WatchlistItem("EURUSD=X", MarketType.FOREX, "EUR/USD", True),
    WatchlistItem("GBPUSD=X", MarketType.FOREX, "GBP/USD", True),
    WatchlistItem("USDJPY=X", MarketType.FOREX, "USD/JPY", True),
    WatchlistItem("PETR4.SA", MarketType.BR_STOCKS, "Petrobras", True),
    WatchlistItem("VALE3.SA", MarketType.BR_STOCKS, "Vale", True),
    WatchlistItem("ITUB4.SA", MarketType.BR_STOCKS, "Itaú Unibanco", True),
    WatchlistItem("BBDC4.SA", MarketType.BR_STOCKS, "Bradesco", True),
    WatchlistItem("ABEV3.SA", MarketType.BR_STOCKS, "Ambev", True),
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create SQLite tables and seed default assets."""
    logger.info(f"Initializing SQLite database at {DB_PATH}...")
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume REAL,
            PRIMARY KEY (symbol, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts ON prices(symbol, timestamp DESC);

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            reasons TEXT NOT NULL DEFAULT '[]',
            current_price REAL NOT NULL,
            rsi REAL,
            macd REAL,
            win_rate REAL
        );
        CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type, timestamp DESC);

        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            source TEXT,
            timestamp TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            symbols TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news(timestamp DESC);

        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            market TEXT NOT NULL,
            display_name TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1
        );
        """)

        cur.execute("SELECT COUNT(*) FROM watchlist;")
        if cur.fetchone()[0] == 0:
            logger.info("Seeding default watchlist items...")
            for item in DEFAULT_WATCHLIST:
                cur.execute(
                    "INSERT OR IGNORE INTO watchlist (symbol, market, display_name, enabled) VALUES (?, ?, ?, ?)",
                    (item.symbol, item.market.value, item.display_name, 1 if item.enabled else 0),
                )
        conn.commit()
    logger.info("Database initialized successfully.")


def save_prices(prices: List[Price]):
    if not prices:
        return
    records = [
        (p.symbol, p.timestamp.isoformat(), p.open, p.high, p.low, p.close, p.volume)
        for p in prices
    ]
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO prices (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timestamp) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume;
            """,
            records,
        )
        conn.commit()


def get_prices(symbol: str, hours: int = 48) -> pd.DataFrame:
    with get_connection() as conn:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM prices
            WHERE symbol = ?
            ORDER BY timestamp ASC;
        """
        df = pd.read_sql_query(query, conn, params=(symbol,))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
    return df


def save_signal(signal: Signal):
    ts_str = signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO signals (symbol, market, timestamp, signal_type, confidence, reasons, current_price, rsi, macd, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.symbol,
                signal.market.value if isinstance(signal.market, MarketType) else signal.market,
                ts_str,
                signal.signal_type.value if isinstance(signal.signal_type, SignalType) else signal.signal_type,
                signal.confidence,
                json.dumps(signal.reasons),
                signal.current_price,
                signal.rsi,
                signal.macd,
                signal.win_rate,
            ),
        )
        signal.id = cur.lastrowid
        conn.commit()
    return signal.id


def get_signals(symbol: Optional[str] = None, limit: int = 50, latest_only: bool = True) -> List[Signal]:
    params = []
    if symbol:
        query = "SELECT id, symbol, market, timestamp, signal_type, confidence, reasons, current_price, rsi, macd, win_rate FROM signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?;"
        params = [symbol, limit]
    elif latest_only:
        query = """
            SELECT id, symbol, market, timestamp, signal_type, confidence, reasons, current_price, rsi, macd, win_rate 
            FROM signals 
            WHERE id IN (SELECT MAX(id) FROM signals GROUP BY symbol)
            ORDER BY CASE signal_type 
                WHEN 'STRONG_BUY' THEN 1 
                WHEN 'BUY' THEN 2 
                WHEN 'STRONG_SELL' THEN 3 
                WHEN 'SELL' THEN 4 
                ELSE 5 END, confidence DESC, timestamp DESC 
            LIMIT ?;
        """
        params = [limit]
    else:
        query = "SELECT id, symbol, market, timestamp, signal_type, confidence, reasons, current_price, rsi, macd, win_rate FROM signals ORDER BY timestamp DESC LIMIT ?;"
        params = [limit]

    signals = []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, tuple(params))
        for row in cur.fetchall():
            reasons = row["reasons"]
            if isinstance(reasons, str):
                try:
                    reasons = json.loads(reasons)
                except Exception:
                    reasons = [reasons]
            signals.append(
                Signal(
                    id=row["id"],
                    symbol=row["symbol"],
                    market=MarketType(row["market"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]) if "T" in str(row["timestamp"]) else datetime.now(timezone.utc),
                    signal_type=SignalType(row["signal_type"]),
                    confidence=row["confidence"],
                    reasons=reasons,
                    current_price=row["current_price"],
                    rsi=row["rsi"],
                    macd=row["macd"],
                    win_rate=row["win_rate"],
                )
            )
    return signals


def save_news(item: NewsItem):
    ts_str = item.timestamp.isoformat() if hasattr(item.timestamp, "isoformat") else str(item.timestamp)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO news (headline, url, source, timestamp, sentiment_score, symbols)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                item.headline,
                item.url,
                item.source,
                ts_str,
                item.sentiment_score,
                json.dumps(item.symbols),
            ),
        )
        conn.commit()


def get_recent_news(symbol: Optional[str] = None, hours: int = 12, limit: int = 30) -> List[NewsItem]:
    query = "SELECT id, headline, url, source, timestamp, sentiment_score, symbols FROM news"
    params = []
    if symbol:
        query += " WHERE symbols LIKE ?"
        params.append(f"%{symbol}%")
    query += " ORDER BY timestamp DESC LIMIT ?;"
    params.append(limit)

    news_items = []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, tuple(params))
        for row in cur.fetchall():
            syms = row["symbols"]
            if isinstance(syms, str):
                try:
                    syms = json.loads(syms)
                except Exception:
                    syms = [syms]
            news_items.append(
                NewsItem(
                    id=row["id"],
                    headline=row["headline"],
                    url=row["url"],
                    source=row["source"],
                    timestamp=datetime.fromisoformat(row["timestamp"]) if "T" in str(row["timestamp"]) else datetime.now(timezone.utc),
                    sentiment_score=row["sentiment_score"],
                    symbols=syms,
                )
            )
    return news_items


def get_watchlist() -> List[WatchlistItem]:
    query = "SELECT symbol, market, display_name, enabled FROM watchlist WHERE enabled = 1 ORDER BY market, symbol;"
    items = []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query)
        for row in cur.fetchall():
            items.append(
                WatchlistItem(
                    symbol=row["symbol"],
                    market=MarketType(row["market"]),
                    display_name=row["display_name"],
                    enabled=bool(row["enabled"]),
                )
            )
    return items


def add_to_watchlist(item: WatchlistItem):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO watchlist (symbol, market, display_name, enabled)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                market=excluded.market,
                display_name=excluded.display_name,
                enabled=excluded.enabled;
            """,
            (item.symbol, item.market.value, item.display_name, 1 if item.enabled else 0),
        )
        conn.commit()


def remove_from_watchlist(symbol: str):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()

