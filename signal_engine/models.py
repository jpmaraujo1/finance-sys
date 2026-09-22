from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketType(str, Enum):
    US_STOCKS = "us_stocks"
    BR_STOCKS = "br_stocks"
    CRYPTO = "crypto"
    FOREX = "forex"


@dataclass
class Signal:
    symbol: str
    market: MarketType
    timestamp: datetime
    signal_type: SignalType
    confidence: float  # 0 to 100
    reasons: List[str]
    current_price: float
    rsi: Optional[float] = None
    macd: Optional[float] = None
    win_rate: Optional[float] = None  # backtested win rate percentage
    id: Optional[int] = None


@dataclass
class Price:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class NewsItem:
    headline: str
    url: str
    source: str
    timestamp: datetime
    sentiment_score: float  # -1.0 to +1.0
    symbols: List[str] = field(default_factory=list)
    id: Optional[int] = None


@dataclass
class WatchlistItem:
    symbol: str
    market: MarketType
    display_name: str
    enabled: bool = True
