import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import feedparser
import pandas as pd

import signal_engine.database as database
from signal_engine.backtester import get_win_rate
from signal_engine.data_fetcher import (
    cnbc_symbol,
    fetch_all,
    parse_cnbc_quote,
    prices_from_cnbc_chart,
)
from signal_engine.models import MarketType, Price, SignalType, WatchlistItem
from signal_engine.sentiment import (
    fetch_google_news,
    fetch_news,
    google_news_search_url,
    items_from_parsed,
)


FIXTURE_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
<title>Petrobras cuts capex after a strong quarter</title>
<link>https://news.google.com/rss/articles/petrobras-example</link>
<pubDate>Tue, 13 Oct 2026 15:00:00 GMT</pubDate>
<description>Shares rose in Sao Paulo.</description>
</item>
</channel></rss>
"""


class PriceWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = database.DB_PATH
        database.DB_PATH = Path(self.tmp.name) / "t.db"
        database.init_db()

    def tearDown(self):
        database.DB_PATH = self.old
        self.tmp.cleanup()

    def _insert(self, symbol, when: datetime):
        with database.get_connection() as conn:
            conn.execute(
                "INSERT INTO prices (symbol, timestamp, open, high, low, close, volume) VALUES (?, ?, 1, 2, 0.5, 1.5, 10)",
                (symbol, when.isoformat()),
            )
            conn.commit()

    def test_hours_keeps_only_the_requested_window(self):
        now = datetime.now(timezone.utc)
        self._insert("AAPL", now - timedelta(hours=1))
        self._insert("AAPL", now - timedelta(hours=3))
        self._insert("AAPL", now - timedelta(hours=10))
        self._insert("AAPL", now - timedelta(hours=200))
        eastern = (now - timedelta(hours=2)).astimezone(ZoneInfo("America/New_York"))
        self._insert("AAPL", eastern)
        self._insert("MSFT", now - timedelta(hours=1))

        six = database.get_prices("AAPL", hours=6)
        self.assertEqual(len(six), 3)
        self.assertLess(database.get_prices("AAPL", hours=6).shape[0], database.get_prices("AAPL", hours=48).shape[0])
        week = database.get_prices("AAPL", hours=168)
        self.assertEqual(len(week), 4)
        self.assertEqual(len(database.get_prices("AAPL", hours=400)), 5)
        self.assertTrue(database.get_prices("MSFT", hours=1).empty or len(database.get_prices("MSFT", hours=1)) == 1)
        self.assertTrue(database.get_prices("AAPL", hours=0).empty)


class WinRateTests(unittest.TestCase):
    def test_failure_is_none_not_65(self):
        self.assertIsNone(get_win_rate("EMPTY", SignalType.BUY, lambda symbol, hours=48: pd.DataFrame()))

        def boom(symbol, hours=48):
            raise RuntimeError("down")

        self.assertIsNone(get_win_rate("BOOM", SignalType.BUY, boom))


class QuoteSourceTests(unittest.TestCase):
    def test_cnbc_mapping_skips_crypto_and_unknown_forex(self):
        self.assertEqual(cnbc_symbol("aapl"), "AAPL")
        self.assertEqual(cnbc_symbol("PETR4.SA"), "PETR4.SA")
        self.assertEqual(cnbc_symbol("EURUSD=X"), "EUR=")
        self.assertEqual(cnbc_symbol("USDJPY=X"), "JPY=")
        self.assertIsNone(cnbc_symbol("BTCUSDT"))
        self.assertIsNone(cnbc_symbol("EURGBP=X"))

    def test_cnbc_code_means_no_listing(self):
        missing = parse_cnbc_quote({"FormattedQuoteResult": {"FormattedQuote": [{"symbol": "NOPE", "code": 1}]}})
        self.assertEqual(missing["status"], "no_listing")
        self.assertNotIn("price", missing)
        quoted = parse_cnbc_quote({
            "FormattedQuoteResult": {
                "FormattedQuote": [{
                    "symbol": "AAPL",
                    "code": 0,
                    "last": "1,234.50",
                    "last_time": "2026-10-14T07:54:00.000-0400",
                    "currencyCode": "USD",
                }]
            }
        })
        self.assertEqual(quoted["status"], "ok")
        self.assertEqual(quoted["price"], 1234.5)
        self.assertEqual(quoted["source"], "CNBC")

    def test_chart_bars_keep_the_watchlist_symbol(self):
        prices = prices_from_cnbc_chart("EURUSD=X", {
            "barData": {"priceBars": [{
                "open": "1.10",
                "high": "1.20",
                "low": "1.00",
                "close": "1.15",
                "volume": "5",
                "tradeTimeinMills": 1791932400000,
            }]}
        })
        self.assertEqual(len(prices), 1)
        self.assertEqual(prices[0].symbol, "EURUSD=X")
        self.assertEqual(prices[0].close, 1.15)

    def test_yahoo_miss_uses_cnbc_and_crypto_does_not(self):
        stock = WatchlistItem("AAPL", MarketType.US_STOCKS, "Apple", True)
        crypto = WatchlistItem("BTCUSDT", MarketType.CRYPTO, "Bitcoin", True)
        candle = Price("AAPL", datetime.now(timezone.utc), 1, 2, 0.5, 1.5, 10)
        with mock.patch("signal_engine.data_fetcher.fetch_yfinance", return_value=[]), \
             mock.patch("signal_engine.data_fetcher.fetch_cnbc_prices", return_value=[candle]) as cnbc, \
             mock.patch("signal_engine.data_fetcher.fetch_binance", return_value=[candle]) as binance:
            filled = fetch_all([stock])
            self.assertEqual(len(filled), 1)
            cnbc.assert_called_once()
            crypto_rows = fetch_all([crypto])
            self.assertEqual(len(crypto_rows), 1)
            binance.assert_called_once()
            self.assertEqual(cnbc.call_count, 1)


class NewsSourceTests(unittest.TestCase):
    def test_google_news_url_is_rss_search_per_name(self):
        b3 = WatchlistItem("PETR4.SA", MarketType.BR_STOCKS, "Petrobras", True)
        url = google_news_search_url(b3)
        self.assertIn("https://news.google.com/rss/search?", url)
        self.assertNotIn("google.com/search?", url)
        self.assertIn("Petrobras", url)
        self.assertIn("pt-BR", url)
        us = WatchlistItem("AAPL", MarketType.US_STOCKS, "Apple", True)
        us_url = google_news_search_url(us)
        self.assertIn("Apple", us_url)
        self.assertIn("en-US", us_url)

    def test_parsed_google_item_is_labeled_and_scored(self):
        parsed = feedparser.parse(FIXTURE_RSS)
        items = items_from_parsed(parsed, source="Google News", limit=5, extra_symbol="PETR4.SA")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "Google News")
        self.assertIn("PETR4.SA", items[0].symbols)
        self.assertIsInstance(items[0].sentiment_score, float)

    def test_one_google_name_down_keeps_the_other(self):
        petr = WatchlistItem("PETR4.SA", MarketType.BR_STOCKS, "Petrobras", True)
        apple = WatchlistItem("AAPL", MarketType.US_STOCKS, "Apple", True)

        def loader(url):
            if "pt-BR" in url:
                raise RuntimeError("google down for B3")
            return feedparser.parse(FIXTURE_RSS)

        with mock.patch("signal_engine.sentiment._load_feed", side_effect=loader):
            items = fetch_google_news([petr, apple])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "Google News")
        self.assertIn("AAPL", items[0].symbols)

    def test_one_wire_down_keeps_the_other(self):
        def loader(url):
            if "investing.com" in url:
                raise RuntimeError("wire down")
            if "yahoo" in url:
                return feedparser.parse(FIXTURE_RSS)
            return feedparser.parse("<rss version='2.0'><channel></channel></rss>")

        with mock.patch("signal_engine.sentiment._load_feed", side_effect=loader):
            items = fetch_news()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "feeds.finance.yahoo.com")


if __name__ == "__main__":
    unittest.main()
