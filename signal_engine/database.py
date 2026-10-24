import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from loguru import logger

from .models import Signal, SignalType, MarketType, Price, NewsItem, WatchlistItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "finance.db"
BACKUP_DIR = DATA_DIR / "backups"

SCHEMA_VERSION = 2

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

ACTIONABLE = ("STRONG_BUY", "BUY", "SELL", "STRONG_SELL")


# ---------------------------------------------------------------------------
# Time helpers. Everything is stored as UTC text in one fixed format so that
# string comparison in SQL equals chronological comparison.
# ---------------------------------------------------------------------------

def to_utc_iso(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00")


def parse_ts(value) -> datetime:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Connection & schema
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """Open a connection, commit on success, roll back on error, and always close it."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _columns(cur, table: str) -> List[str]:
    return [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]


def init_db():
    """Create tables, run migrations and seed the default watchlist."""
    logger.info(f"Initializing SQLite database at {DB_PATH}...")
    with get_connection() as conn:
        cur = conn.cursor()
        # WAL lets the dashboard read while the engine writes
        cur.execute("PRAGMA journal_mode=WAL;")
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

        -- Live forward tracking: what actually happened after each actionable signal
        CREATE TABLE IF NOT EXISTS signal_outcomes (
            signal_id INTEGER NOT NULL,
            horizon_min INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            is_entry INTEGER NOT NULL DEFAULT 0,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_time TEXT NOT NULL,
            exit_price REAL,              -- NULL when the exit candle was unusable (e.g. data gap)
            gross_return REAL,            -- raw price change, (exit - entry) / entry
            net_return REAL,              -- direction * gross - round-trip cost
            PRIMARY KEY (signal_id, horizon_min)
        );
        CREATE INDEX IF NOT EXISTS idx_outcomes_time ON signal_outcomes(entry_time);

        -- Latest periodic backtest per symbol and signal type
        CREATE TABLE IF NOT EXISTS backtest_results (
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (symbol, signal_type)
        );

        CREATE TABLE IF NOT EXISTS engine_status (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """)

        cols = _columns(cur, "signals")
        if "sentiment" not in cols:
            cur.execute("ALTER TABLE signals ADD COLUMN sentiment REAL;")
        if "is_entry" not in cols:
            cur.execute("ALTER TABLE signals ADD COLUMN is_entry INTEGER NOT NULL DEFAULT 0;")

        row = cur.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row[0]) if row else 1
        if version < 2:
            _migrate_timestamps_to_utc(cur)
            cur.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

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


def _migrate_timestamps_to_utc(cur):
    """
    Older versions stored exchange-local timestamps (e.g. -04:00 for NYSE, -03:00 for B3).
    Mixed offsets make SQL time filters and ordering wrong, so rewrite everything as UTC.
    """
    rows = cur.execute(
        "SELECT symbol, timestamp, open, high, low, close, volume FROM prices WHERE timestamp NOT LIKE '%+00:00'"
    ).fetchall()
    if rows:
        logger.info(f"Migrating {len(rows)} price timestamps to UTC...")
        cur.executemany(
            "INSERT OR REPLACE INTO prices (symbol, timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r[0], to_utc_iso(r[1]), r[2], r[3], r[4], r[5], r[6]) for r in rows],
        )
        cur.executemany(
            "DELETE FROM prices WHERE symbol = ? AND timestamp = ?",
            [(r[0], r[1]) for r in rows],
        )
    for table in ("signals", "news"):
        rows = cur.execute(f"SELECT id, timestamp FROM {table} WHERE timestamp NOT LIKE '%+00:00'").fetchall()
        if rows:
            cur.executemany(
                f"UPDATE {table} SET timestamp = ? WHERE id = ?",
                [(to_utc_iso(r[1]), r[0]) for r in rows],
            )


def backup_db(keep: int = 7) -> Optional[Path]:
    """Write a consistent snapshot of the database and keep the newest `keep` copies."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUP_DIR / f"finance-{utc_now().strftime('%Y%m%d-%H%M')}.db"
    try:
        dst = sqlite3.connect(str(target))
        try:
            with get_connection() as src:
                src.backup(dst)
        finally:
            dst.close()
        backups = sorted(BACKUP_DIR.glob("finance-*.db"))
        for old in backups[:-keep] if keep > 0 else []:
            old.unlink(missing_ok=True)
        logger.info(f"Database backup written to {target}")
        return target
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def save_prices(prices: List[Price]):
    if not prices:
        return
    records = [
        (p.symbol, to_utc_iso(p.timestamp), p.open, p.high, p.low, p.close, p.volume)
        for p in prices
    ]
    with get_connection() as conn:
        conn.executemany(
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


def _price_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in rows], columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    return df


def get_prices(
    symbol: str,
    hours: Optional[float] = None,
    limit: Optional[int] = None,
    since: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Price candles for one symbol, oldest first, indexed by UTC timestamp.
    `hours` / `since` bound the window; `limit` keeps only the newest N candles.
    """
    query = "SELECT timestamp, open, high, low, close, volume FROM prices WHERE symbol = ?"
    params: list = [symbol]
    start = since
    if hours is not None:
        h_start = utc_now() - timedelta(hours=hours)
        start = max(start, h_start) if start else h_start
    if start is not None:
        query += " AND timestamp >= ?"
        params.append(to_utc_iso(start))
    query += " ORDER BY timestamp DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return _price_frame(rows)


def get_price_span(symbol: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM prices WHERE symbol = ?", (symbol,)
        ).fetchone()
    if not row or row[2] == 0:
        return None, None, 0
    return parse_ts(row[0]), parse_ts(row[1]), int(row[2])


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

SIGNAL_COLUMNS = (
    "id, symbol, market, timestamp, signal_type, confidence, reasons, current_price, "
    "rsi, macd, win_rate, sentiment, is_entry"
)


def save_signal(signal: Signal) -> Optional[int]:
    """
    Store a signal. Returns None (and stores nothing) if a signal for the same
    symbol and candle already exists, e.g. when a market is closed and no new
    candle has arrived. Marks the signal as an entry when its type differs from
    the previous one for that symbol.
    """
    ts_str = to_utc_iso(signal.timestamp)
    sig_type = signal.signal_type.value if isinstance(signal.signal_type, SignalType) else signal.signal_type
    market = signal.market.value if isinstance(signal.market, MarketType) else signal.market
    with get_connection() as conn:
        cur = conn.cursor()
        prev = cur.execute(
            "SELECT timestamp, signal_type FROM signals WHERE symbol = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
            (signal.symbol,),
        ).fetchone()
        if prev and prev["timestamp"] >= ts_str:
            return None
        signal.is_entry = prev is None or prev["signal_type"] != sig_type
        cur.execute(
            """
            INSERT INTO signals (symbol, market, timestamp, signal_type, confidence, reasons,
                                 current_price, rsi, macd, win_rate, sentiment, is_entry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.symbol, market, ts_str, sig_type, signal.confidence,
                json.dumps(signal.reasons), signal.current_price, signal.rsi, signal.macd,
                signal.win_rate, signal.sentiment, 1 if signal.is_entry else 0,
            ),
        )
        signal.id = cur.lastrowid
        conn.commit()
    return signal.id


def _row_to_signal(row) -> Signal:
    reasons = row["reasons"]
    if isinstance(reasons, str):
        try:
            reasons = json.loads(reasons)
        except Exception:
            reasons = [reasons]
    return Signal(
        id=row["id"],
        symbol=row["symbol"],
        market=MarketType(row["market"]),
        timestamp=parse_ts(row["timestamp"]),
        signal_type=SignalType(row["signal_type"]),
        confidence=row["confidence"],
        reasons=reasons,
        current_price=row["current_price"],
        rsi=row["rsi"],
        macd=row["macd"],
        win_rate=row["win_rate"],
        sentiment=row["sentiment"],
        is_entry=bool(row["is_entry"]),
    )


def get_signals(symbol: Optional[str] = None, limit: int = 50, latest_only: bool = True) -> List[Signal]:
    if symbol:
        query = f"SELECT {SIGNAL_COLUMNS} FROM signals WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?;"
        params: tuple = (symbol, limit)
    elif latest_only:
        query = f"""
            SELECT {SIGNAL_COLUMNS}
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
        params = (limit,)
    else:
        query = f"SELECT {SIGNAL_COLUMNS} FROM signals ORDER BY timestamp DESC LIMIT ?;"
        params = (limit,)
    with get_connection() as conn:
        return [_row_to_signal(r) for r in conn.execute(query, params).fetchall()]


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def save_news(item: NewsItem):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO news (headline, url, source, timestamp, sentiment_score, symbols)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (item.headline, item.url, item.source, to_utc_iso(item.timestamp),
             item.sentiment_score, json.dumps(item.symbols)),
        )
        conn.commit()


def _row_to_news(row) -> NewsItem:
    syms = row["symbols"]
    if isinstance(syms, str):
        try:
            syms = json.loads(syms)
        except Exception:
            syms = [syms]
    return NewsItem(
        id=row["id"],
        headline=row["headline"],
        url=row["url"],
        source=row["source"],
        timestamp=parse_ts(row["timestamp"]),
        sentiment_score=row["sentiment_score"],
        symbols=syms,
    )


def get_recent_news(symbol: Optional[str] = None, hours: Optional[float] = 12, limit: int = 30) -> List[NewsItem]:
    query = "SELECT id, headline, url, source, timestamp, sentiment_score, symbols FROM news WHERE 1=1"
    params: list = []
    if hours is not None:
        query += " AND timestamp >= ?"
        params.append(to_utc_iso(utc_now() - timedelta(hours=hours)))
    if symbol:
        # symbols is a JSON list, match the quoted symbol so "ETH" can't match "ETHUSDT"
        query += " AND symbols LIKE ?"
        params.append(f'%"{symbol}"%')
    query += " ORDER BY timestamp DESC LIMIT ?;"
    params.append(limit)
    with get_connection() as conn:
        return [_row_to_news(r) for r in conn.execute(query, params).fetchall()]


def get_symbol_sentiment(symbol: str, hours: float, until: Optional[datetime] = None) -> Tuple[Optional[float], int]:
    """Average sentiment of headlines tagged with `symbol` in the window ending at `until` (default now)."""
    end = until or utc_now()
    start = end - timedelta(hours=hours)
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT AVG(sentiment_score), COUNT(*) FROM news
            WHERE symbols LIKE ? AND timestamp > ? AND timestamp <= ?
            """,
            (f'%"{symbol}"%', to_utc_iso(start), to_utc_iso(end)),
        ).fetchone()
    count = int(row[1] or 0)
    return (float(row[0]) if count else None), count


def get_news_frame(symbol: str) -> pd.DataFrame:
    """All stored headlines for a symbol as a (timestamp-indexed) sentiment series, for backtests."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT timestamp, sentiment_score FROM news WHERE symbols LIKE ? ORDER BY timestamp",
            (f'%"{symbol}"%',),
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=["timestamp", "sentiment_score"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def get_watchlist() -> List[WatchlistItem]:
    query = "SELECT symbol, market, display_name, enabled FROM watchlist WHERE enabled = 1 ORDER BY market, symbol;"
    with get_connection() as conn:
        return [
            WatchlistItem(
                symbol=row["symbol"],
                market=MarketType(row["market"]),
                display_name=row["display_name"],
                enabled=bool(row["enabled"]),
            )
            for row in conn.execute(query).fetchall()
        ]


def add_to_watchlist(item: WatchlistItem):
    with get_connection() as conn:
        conn.execute(
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
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()


# ---------------------------------------------------------------------------
# Live outcome tracking
# ---------------------------------------------------------------------------

def get_signals_awaiting_outcome(horizon_min: int, limit: int = 5000) -> List[dict]:
    """Actionable signals old enough to grade at this horizon that have no outcome yet."""
    cutoff = to_utc_iso(utc_now() - timedelta(minutes=horizon_min))
    placeholders = ",".join("?" for _ in ACTIONABLE)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT s.id, s.symbol, s.market, s.timestamp, s.signal_type, s.current_price, s.is_entry
            FROM signals s
            LEFT JOIN signal_outcomes o ON o.signal_id = s.id AND o.horizon_min = ?
            WHERE o.signal_id IS NULL
              AND s.signal_type IN ({placeholders})
              AND s.timestamp <= ?
            ORDER BY s.timestamp ASC
            LIMIT ?
            """,
            (horizon_min, *ACTIONABLE, cutoff, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_first_price_at_or_after(symbol: str, when: datetime) -> Optional[Tuple[datetime, float]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT timestamp, close FROM prices WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
            (symbol, to_utc_iso(when)),
        ).fetchone()
    return (parse_ts(row[0]), float(row[1])) if row else None


def save_outcomes(rows: Iterable[tuple]):
    rows = list(rows)
    if not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO signal_outcomes
            (signal_id, horizon_min, symbol, market, signal_type, is_entry, entry_time, entry_price,
             exit_time, exit_price, gross_return, net_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def get_outcomes_frame(days: Optional[float] = None) -> pd.DataFrame:
    query = "SELECT * FROM signal_outcomes"
    params: list = []
    if days is not None:
        query += " WHERE entry_time >= ?"
        params.append(to_utc_iso(utc_now() - timedelta(days=days)))
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    cols = ["signal_id", "horizon_min", "symbol", "market", "signal_type", "is_entry", "entry_time",
            "entry_price", "exit_time", "exit_price", "gross_return", "net_return"]
    return pd.DataFrame([dict(r) for r in rows], columns=cols)


def get_tracking_start() -> Optional[datetime]:
    with get_connection() as conn:
        row = conn.execute("SELECT MIN(timestamp) FROM signals WHERE is_entry = 1").fetchone()
    return parse_ts(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------
# Backtest results & engine status
# ---------------------------------------------------------------------------

def save_backtest_results(results: Dict[str, dict]):
    """results: {symbol: {signal_type: stats_dict}}"""
    now = to_utc_iso(utc_now())
    rows = [
        (symbol, sig_type, now, json.dumps(stats))
        for symbol, per_type in results.items()
        for sig_type, stats in per_type.items()
    ]
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO backtest_results (symbol, signal_type, computed_at, payload) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def get_backtest_results(symbol: Optional[str] = None) -> List[dict]:
    query = "SELECT symbol, signal_type, computed_at, payload FROM backtest_results"
    params: list = []
    if symbol:
        query += " WHERE symbol = ?"
        params.append(symbol)
    with get_connection() as conn:
        rows = conn.execute(query + " ORDER BY symbol, signal_type", params).fetchall()
    out = []
    for r in rows:
        item = json.loads(r["payload"])
        item.update({"symbol": r["symbol"], "signal_type": r["signal_type"], "computed_at": r["computed_at"]})
        out.append(item)
    return out


def set_status(key: str, value):
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO engine_status (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), to_utc_iso(utc_now())),
        )
        conn.commit()


def get_status() -> Dict[str, dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value, updated_at FROM engine_status").fetchall()
    return {r["key"]: {"value": json.loads(r["value"]), "updated_at": r["updated_at"]} for r in rows}
