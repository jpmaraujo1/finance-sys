from datetime import datetime, timezone
import email.utils
from typing import Callable, List, Optional
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger

from .models import NewsItem, WatchlistItem

NEWS_FEEDS = {
    "general": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.investing.com/rss/news.rss",
    ],
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ],
    "brazil": [
        "https://feeds.folha.uol.com.br/mercado/rss091.xml",
        "https://www.infomoney.com.br/feed/",
    ],
}

COMPANY_MAP = {
    "AAPL": ["Apple", "AAPL", "iPhone", "Mac"],
    "MSFT": ["Microsoft", "MSFT", "Windows", "Azure"],
    "NVDA": ["Nvidia", "NVDA", "GeForce", "AI chip"],
    "TSLA": ["Tesla", "TSLA", "Elon Musk"],
    "AMZN": ["Amazon", "AMZN", "AWS"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "META": ["Meta", "Facebook", "Instagram"],
    "SPY": ["S&P 500", "SPY", "Wall Street", "Federal Reserve", "Fed"],
    "QQQ": ["Nasdaq", "QQQ", "Tech stocks"],
    "BTCUSDT": ["Bitcoin", "BTC", "crypto"],
    "ETHUSDT": ["Ethereum", "ETH", "Ether"],
    "SOLUSDT": ["Solana", "SOL"],
    "BNBUSDT": ["Binance", "BNB"],
    "XRPUSDT": ["Ripple", "XRP"],
    "ADAUSDT": ["Cardano", "ADA"],
    "DOGEUSDT": ["Dogecoin", "DOGE"],
    "EURUSD=X": ["Euro", "EUR/USD", "ECB"],
    "GBPUSD=X": ["Pound", "GBP/USD", "Bank of England"],
    "USDJPY=X": ["Yen", "USD/JPY", "Bank of Japan"],
    "PETR4.SA": ["Petrobras", "PETR4", "petr?leo"],
    "VALE3.SA": ["Vale", "VALE3", "min?rio de ferro"],
    "ITUB4.SA": ["Ita?", "ITUB4"],
    "BBDC4.SA": ["Bradesco", "BBDC4"],
    "ABEV3.SA": ["Ambev", "ABEV3"],
}

analyzer = SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> float:
    """Calculate sentiment score between -1.0 and +1.0 using VADER."""
    if not text:
        return 0.0
    scores = analyzer.polarity_scores(text)
    return round(float(scores["compound"]), 3)


def tag_symbols(headline: str, summary: str = "") -> List[str]:
    """Find all watchlist symbols mentioned in text."""
    full_text = f"{headline} {summary}".lower()
    matched: List[str] = []
    for symbol, keywords in COMPANY_MAP.items():
        for kw in keywords:
            if kw.lower() in full_text:
                matched.append(symbol)
                break
    return matched


def fetch_news() -> List[NewsItem]:
    """Scrape and parse headlines from financial RSS feeds."""
    items: List[NewsItem] = []
    for category, feeds in NEWS_FEEDS.items():
        for url in feeds:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:15]:
                    headline = entry.get("title", "")
                    link = entry.get("link", "")
                    summary = entry.get("summary", "")
                    if not headline or not link:
                        continue

                    # Parse timestamp if available
                    dt = datetime.now(timezone.utc)
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                    score = score_sentiment(f"{headline}. {summary}")
                    syms = tag_symbols(headline, summary)

                    items.append(
                        NewsItem(
                            headline=headline,
                            url=link,
                            source=url.split("/")[2],
                            timestamp=dt,
                            sentiment_score=score,
                            symbols=syms,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {url}: {e}")

    logger.info(f"Fetched {len(items)} news headlines across feeds")
    return items


def run_sentiment_cycle(save_news_fn: Callable[[NewsItem], None]):
    """Execute news fetch and store in database."""
    logger.info("Running news sentiment cycle...")
    news_items = fetch_news()
    for item in news_items:
        try:
            save_news_fn(item)
        except Exception as e:
            logger.debug(f"Could not save news item: {e}")
