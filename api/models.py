from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    use_rag: bool = True


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[str]] = []
    model: str


class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    status: str


class TransactionCategory(str, Enum):
    housing = "Housing"
    food = "Food & Dining"
    transport = "Transport"
    utilities = "Utilities"
    entertainment = "Entertainment"
    healthcare = "Healthcare"
    income = "Income"
    savings = "Savings & Investments"
    shopping = "Shopping"
    other = "Other"


class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    category: Optional[TransactionCategory] = None


class AnalyzeTransactionsRequest(BaseModel):
    transactions: List[Transaction]


class AnalyzeTransactionsResponse(BaseModel):
    categorized: List[Transaction]
    summary: str
    total_income: float
    total_expenses: float
    top_categories: List[dict]


# Trading & Market Models

class SignalTypeResponse(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketTypeResponse(str, Enum):
    US_STOCKS = "us_stocks"
    BR_STOCKS = "br_stocks"
    CRYPTO = "crypto"
    FOREX = "forex"


class SignalResponse(BaseModel):
    id: Optional[int] = None
    symbol: str
    display_name: Optional[str] = None
    market: MarketTypeResponse
    timestamp: datetime
    signal_type: SignalTypeResponse
    confidence: float
    reasons: List[str]
    current_price: float
    rsi: Optional[float] = None
    macd: Optional[float] = None
    win_rate: Optional[float] = None


class NewsItemResponse(BaseModel):
    id: Optional[int] = None
    headline: str
    url: str
    source: Optional[str] = None
    timestamp: datetime
    sentiment_score: float
    sentiment_label: str
    symbols: List[str] = []


class WatchlistItemResponse(BaseModel):
    symbol: str
    market: MarketTypeResponse
    display_name: str
    enabled: bool


class AddToWatchlistRequest(BaseModel):
    symbol: str
    market: MarketTypeResponse
    display_name: str


class MarketSummaryResponse(BaseModel):
    total_tracked: int
    strong_buys: int
    buys: int
    holds: int
    sells: int
    strong_sells: int
    top_signals: List[SignalResponse]
    overall_sentiment: float
    last_updated: datetime


class AIReportRequest(BaseModel):
    focus_symbols: Optional[List[str]] = None
    report_type: str = "daily"


class AIReportResponse(BaseModel):
    report: str
    generated_at: datetime
    model: str
    symbols_covered: List[str]
