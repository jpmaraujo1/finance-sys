from datetime import datetime, timezone
import re
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

# Keywords per symbol. Matching uses word boundaries, and short all-caps tickers
# (e.g. "ETH", "SOL", "ADA") are matched case-sensitively so ordinary words like
# "method", "solution" or "Canada" don't get tagged.
COMPANY_MAP = {
    "AAPL": ["Apple", "AAPL", "iPhone"],
    "MSFT": ["Microsoft", "MSFT", "Azure"],
    "NVDA": ["Nvidia", "NVDA", "GeForce"],
    "TSLA": ["Tesla", "TSLA", "Elon Musk"],
    "AMZN": ["Amazon", "AMZN", "AWS"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "META": ["Meta Platforms", "Facebook", "Instagram", "META"],
    "SPY": ["S&P 500", "SPY", "Wall Street", "Federal Reserve", "Fed"],
    "QQQ": ["Nasdaq", "QQQ"],
    "BTCUSDT": ["Bitcoin", "BTC"],
    "ETHUSDT": ["Ethereum", "ETH", "Ether"],
    "SOLUSDT": ["Solana", "SOL"],
    "BNBUSDT": ["BNB", "Binance Coin"],
    "XRPUSDT": ["Ripple", "XRP"],
    "ADAUSDT": ["Cardano", "ADA"],
    "DOGEUSDT": ["Dogecoin", "DOGE"],
    "EURUSD=X": ["EUR/USD", "ECB", "European Central Bank"],
    "GBPUSD=X": ["GBP/USD", "Bank of England", "sterling"],
    "USDJPY=X": ["USD/JPY", "Bank of Japan", "BOJ", "yen"],
    "PETR4.SA": ["Petrobras", "PETR4", "PETR3"],
    "VALE3.SA": ["VALE3", "Vale S.A.", "mineradora Vale", "Vale (VALE3)"],
    "ITUB4.SA": ["Itaú", "Itau Unibanco", "ITUB4"],
    "BBDC4.SA": ["Bradesco", "BBDC4"],
    "ABEV3.SA": ["Ambev", "ABEV3"],
}


def _compile_patterns(company_map):
    patterns = {}
    for symbol, keywords in company_map.items():
        compiled = []
        for kw in keywords:
            case_sensitive = kw.isupper() and len(kw) <= 5
            flags = 0 if case_sensitive else re.IGNORECASE
            compiled.append(re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", flags))
        patterns[symbol] = compiled
    return patterns


_PATTERNS = _compile_patterns(COMPANY_MAP)

analyzer = SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> float:
    """Calculate sentiment score between -1.0 and +1.0 using VADER."""
    if not text:
        return 0.0
    scores = analyzer.polarity_scores(text)
    return round(float(scores["compound"]), 3)


def tag_symbols(headline: str, summary: str = "") -> List[str]:
    """Find all watchlist symbols mentioned in text."""
    full_text = f"{headline} {summary}"
    return [sym for sym, pats in _PATTERNS.items() if any(p.search(full_text) for p in pats)]


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
