from datetime import datetime, timezone
from typing import Callable, List, Optional
from urllib.parse import urlencode
import re

import feedparser
import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger

from .database import get_watchlist
from .models import MarketType, NewsItem, WatchlistItem

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

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

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
FEED_HEADERS = {"User-Agent": "finance-sys/1.0"}


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


def _plain(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _load_feed(url: str):
    resp = httpx.get(url, timeout=10.0, follow_redirects=True, headers=FEED_HEADERS)
    resp.raise_for_status()
    body = resp.content
    parsed = feedparser.parse(body)
    head = body[:400].lower()
    if not parsed.entries and b"<rss" not in head and b"<feed" not in head:
        raise ValueError(f"Feed did not return RSS: {url}")
    return parsed


def items_from_parsed(
    parsed,
    source: str,
    limit: int = 15,
    extra_symbol: Optional[str] = None,
) -> List[NewsItem]:
    items: List[NewsItem] = []
    for entry in list(getattr(parsed, "entries", []) or [])[:limit]:
        headline = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        if not headline or not link:
            continue

        dt = datetime.now(timezone.utc)
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        score = score_sentiment(f"{_plain(headline)}. {_plain(summary)}")
        syms = tag_symbols(headline, _plain(summary))
        if extra_symbol and extra_symbol not in syms:
            syms.insert(0, extra_symbol)

        items.append(
            NewsItem(
                headline=headline,
                url=link,
                source=source,
                timestamp=dt,
                sentiment_score=score,
                symbols=syms,
            )
        )
    return items


def google_news_search_url(item: WatchlistItem) -> str:
    """Public Google News RSS search for one watchlist name. Not a Google HTML search."""
    name = (item.display_name or item.symbol or "").strip() or item.symbol
    if item.market == MarketType.BR_STOCKS:
        params = {"q": name, "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
    else:
        params = {"q": name, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    return GOOGLE_NEWS_RSS + "?" + urlencode(params)


def fetch_news() -> List[NewsItem]:
    """Parse headlines from the existing financial RSS feeds."""
    items: List[NewsItem] = []
    for _category, feeds in NEWS_FEEDS.items():
        for url in feeds:
            try:
                parsed = _load_feed(url)
                host = url.split("/")[2]
                items.extend(items_from_parsed(parsed, source=host, limit=15))
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {url}: {e}")

    logger.info(f"Fetched {len(items)} news headlines across feeds")
    return items


def fetch_google_news(watchlist: Optional[List[WatchlistItem]] = None) -> List[NewsItem]:
    """Google News RSS search once per watchlist name, including B3 names.

    A failed name is skipped. The other names, and the wire feeds, still store.
    """
    items: List[NewsItem] = []
    rows = watchlist if watchlist is not None else get_watchlist()
    for item in rows:
        if not item.enabled:
            continue
        url = google_news_search_url(item)
        try:
            parsed = _load_feed(url)
            found = items_from_parsed(
                parsed,
                source="Google News",
                limit=5,
                extra_symbol=item.symbol,
            )
            items.extend(found)
            logger.info(f"Google News {item.display_name}: {len(found)} headlines")
        except Exception as e:
            logger.warning(f"Google News failed for {item.display_name or item.symbol}: {e}")
    logger.info(f"Fetched {len(items)} Google News headlines")
    return items


def run_sentiment_cycle(save_news_fn: Callable[[NewsItem], None]):
    """Execute news fetch and store in database."""
    logger.info("Running news sentiment cycle...")
    news_items = fetch_news()
    try:
        news_items.extend(fetch_google_news(get_watchlist()))
    except Exception as e:
        logger.warning(f"Google News cycle skipped: {e}")
    for item in news_items:
        try:
            save_news_fn(item)
        except Exception as e:
            logger.debug(f"Could not save news item: {e}")
