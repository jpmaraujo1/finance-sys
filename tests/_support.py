"""
Test helpers: synthetic market data, a throwaway database, and minimal stand-ins
for optional third-party packages so the core logic can be tested even where they
aren't installed. When the real packages are installed, the real ones are used.
"""
import importlib
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub(name, **attrs):
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


class _Logger:
    def __getattr__(self, _):
        return lambda *a, **k: None


_stub("loguru", logger=_Logger())
_stub("feedparser", parse=lambda url: types.SimpleNamespace(entries=[]))


class _Vader:
    def polarity_scores(self, text):
        return {"compound": 0.0}


_stub("vaderSentiment")
_stub("vaderSentiment.vaderSentiment", SentimentIntensityAnalyzer=_Vader)
_stub("dotenv", load_dotenv=lambda *a, **k: None)
_stub("yfinance", Ticker=None)
_stub("binance")
_stub("binance.client", Client=type("Client", (), {}))
_stub("apscheduler")
_stub("apscheduler.schedulers")
_stub("apscheduler.schedulers.blocking", BlockingScheduler=object)


def synthetic_candles(n=3000, start_price=100.0, vol=0.002, drift=0.0, seed=0,
                      start="2026-01-05 00:00", freq="5min"):
    """Geometric random walk 5-minute OHLCV candles (UTC index)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(10, 0.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TempDB:
    """Point the database module at a fresh temporary file."""

    def __enter__(self):
        from signal_engine import database
        self._db = database
        self._old = database.DB_PATH
        self._dir = tempfile.TemporaryDirectory()
        database.DB_PATH = Path(self._dir.name) / "test.db"
        database.init_db()
        return database

    def __exit__(self, *exc):
        self._db.DB_PATH = self._old
        self._dir.cleanup()
