"""
Central settings for the signal engine.

Everything here can be overridden with environment variables (see .env.example),
so you can tune costs and horizons without touching code.
"""
import os

from .models import MarketType


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Candle size used everywhere (minutes)
CANDLE_MINUTES = 5

# Round-trip trading cost per market, as a fraction of price (fees + spread + slippage).
# These are deliberately conservative retail estimates. A strategy that only works
# with zero costs does not work.
ROUND_TRIP_COST = {
    MarketType.CRYPTO: _env_float("COST_CRYPTO", 0.0020),        # ~0.10% taker fee per side
    MarketType.US_STOCKS: _env_float("COST_US_STOCKS", 0.0005),  # commission-free, spread + slippage
    MarketType.BR_STOCKS: _env_float("COST_BR_STOCKS", 0.0010),  # B3 fees + spread
    MarketType.FOREX: _env_float("COST_FOREX", 0.0003),          # retail spread
}


def cost_for(market: MarketType) -> float:
    return ROUND_TRIP_COST.get(market, 0.0010)


# Forward horizons (minutes) used to grade live signals.
TRACK_HORIZONS_MIN = [
    int(h) for h in os.getenv("TRACK_HORIZONS_MIN", "60,240,1440").split(",") if h.strip()
]

# Backtest horizon, in candles (12 x 5m = 1 hour)
BACKTEST_HORIZON_BARS = _env_int("BACKTEST_HORIZON_BARS", 12)
# How much stored history the periodic backtest uses
BACKTEST_LOOKBACK_DAYS = _env_int("BACKTEST_LOOKBACK_DAYS", 90)
# Below this many independent entries, a hit rate is not shown (it would be noise)
MIN_SAMPLES_FOR_STATS = _env_int("MIN_SAMPLES_FOR_STATS", 20)

# News sentiment as a signal input.
# Weight 0 disables it. A symbol needs at least SENTIMENT_MIN_ARTICLES tagged
# headlines inside the window before sentiment is allowed to vote.
SENTIMENT_WEIGHT = _env_int("SENTIMENT_WEIGHT", 1)
SENTIMENT_WINDOW_HOURS = _env_int("SENTIMENT_WINDOW_HOURS", 6)
SENTIMENT_MIN_ARTICLES = _env_int("SENTIMENT_MIN_ARTICLES", 2)
SENTIMENT_THRESHOLD = _env_float("SENTIMENT_THRESHOLD", 0.25)

# Bollinger z-score at which price counts as "at the band"
BB_Z_THRESHOLD = _env_float("BB_Z_THRESHOLD", 2.0)

# How many recent candles the live engine loads to compute indicators
LIVE_LOOKBACK_BARS = _env_int("LIVE_LOOKBACK_BARS", 600)

# Database backups
BACKUP_KEEP = _env_int("BACKUP_KEEP", 7)
