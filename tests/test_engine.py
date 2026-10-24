import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from tests._support import synthetic_candles, TempDB  # noqa: F401  (installs stand-ins first)

from signal_engine import config
from signal_engine.backtester import (
    backtest_frame, build_signal_frame, entry_records, simulate_long_only, rolling_sentiment, summarize,
)
from signal_engine.models import MarketType, SignalType, Signal, Price, NewsItem
from signal_engine.sentiment import tag_symbols
from signal_engine.signal_generator import generate_signal
from signal_engine.technical_analysis import compute_rsi, compute_indicator_frame, compute_indicators


class TestIndicators(unittest.TestCase):
    def test_rsi_extremes(self):
        up = pd.Series(np.arange(1, 60, dtype=float))
        flat = pd.Series(np.full(60, 5.0))
        self.assertAlmostEqual(compute_rsi(up).iloc[-1], 100.0)
        self.assertAlmostEqual(compute_rsi(flat).iloc[-1], 50.0)

    def test_rsi_in_range(self):
        df = synthetic_candles(500, seed=1)
        rsi = compute_rsi(df["close"]).dropna()
        self.assertTrue(((rsi >= 0) & (rsi <= 100)).all())

    def test_indicators_are_trailing(self):
        """Indicators at candle k must not change when later candles are added (no look-ahead)."""
        df = synthetic_candles(400, seed=2)
        full = compute_indicator_frame(df)
        for k in (120, 250, 399):
            partial = compute_indicator_frame(df.iloc[: k + 1])
            for col in ("rsi", "macd", "bb_z", "ema21", "atr"):
                self.assertAlmostEqual(partial[col].iloc[-1], full[col].iloc[k], places=9, msg=col)

    def test_bollinger_vote_is_scale_free(self):
        """
        The old rule (price within 1% of the lower band) fired on almost every candle of a
        low-volatility FX pair. The z-score rule should fire rarely on a random walk.
        """
        fx = synthetic_candles(3000, start_price=1.08, vol=0.0003, seed=3)
        frame = compute_indicator_frame(fx).dropna(subset=["bb_lower", "bb_upper"])
        old_rule = (frame["close"] <= frame["bb_lower"] * 1.01).mean()
        new_rule = (frame["bb_z"].abs() >= config.BB_Z_THRESHOLD).mean()
        self.assertGreater(old_rule, 0.95)
        self.assertLess(new_rule, 0.15)


def make_ind(**kw):
    ind = compute_indicators(synthetic_candles(100, seed=4), "TEST")
    for k, v in kw.items():
        setattr(ind, k, v)
    return ind


class TestSignalGenerator(unittest.TestCase):
    def test_sentiment_needs_enough_headlines(self):
        base = dict(rsi=40.0, macd=1.0, macd_signal=0.5, macd_hist=0.5, bb_z=0.0, ema9=2.0, ema21=1.0,
                    volume=1.0, volume_sma20=1.0)
        ind = make_ind(**base)
        with_one = generate_signal(ind, MarketType.CRYPTO, sentiment=0.8, news_count=1)
        with_many = generate_signal(ind, MarketType.CRYPTO, sentiment=0.8, news_count=5)
        self.assertFalse(any("news sentiment" in r for r in with_one.reasons))
        self.assertTrue(any("news sentiment" in r for r in with_many.reasons))

    def test_news_can_tip_the_decision(self):
        base = dict(rsi=50.0, macd=1.0, macd_signal=0.5, macd_hist=0.5, bb_z=0.0, ema9=2.0, ema21=1.0,
                    volume=1.0, volume_sma20=1.0)
        ind = make_ind(**base)
        self.assertEqual(generate_signal(ind, MarketType.CRYPTO).signal_type, SignalType.HOLD)
        good_news = generate_signal(ind, MarketType.CRYPTO, sentiment=0.7, news_count=4)
        self.assertEqual(good_news.signal_type, SignalType.BUY)
        self.assertEqual(good_news.sentiment, 0.7)

    def test_falling_knife_holds(self):
        ind = make_ind(rsi=22.0, macd=-1.0, macd_signal=-0.5, macd_hist=-0.5, bb_z=-2.5, ema9=1.0, ema21=2.0,
                       volume=1.0, volume_sma20=1.0)
        self.assertEqual(generate_signal(ind, MarketType.CRYPTO).signal_type, SignalType.HOLD)


class TestBacktester(unittest.TestCase):
    def test_counts_entries_not_repeats(self):
        idx = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105, 106, 107.0],
            "signal": ["HOLD", "BUY", "BUY", "BUY", "HOLD", "SELL", "SELL", "HOLD"],
            "score": 60.0,
        }, index=idx)
        rec = entry_records(frame, "X", MarketType.CRYPTO, horizon=1, cost=0.001)
        self.assertEqual(list(rec["signal_type"]), ["BUY", "SELL"])
        buy = rec.iloc[0]
        self.assertAlmostEqual(buy["gross"], 102 / 101 - 1)
        self.assertAlmostEqual(buy["net"], 102 / 101 - 1 - 0.001)
        sell = rec.iloc[1]
        self.assertAlmostEqual(sell["gross"], -(106 / 105 - 1))

    def test_long_only_simulation(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
        frame = pd.DataFrame({"close": [100, 100, 110, 121, 121.0],
                              "signal": ["BUY", "HOLD", "HOLD", "SELL", "HOLD"]}, index=idx)
        res = simulate_long_only(frame, cost=0.0)
        self.assertAlmostEqual(res["strategy_return_pct"], 21.0, places=6)
        self.assertEqual(res["trades"], 1)
        with_cost = simulate_long_only(frame, cost=0.01)
        self.assertLess(with_cost["strategy_return_pct"], 21.0)

    def test_random_walk_has_no_edge(self):
        """On pure noise, signals should show no reliable edge and lose money after costs."""
        records = []
        for seed in range(10):
            df = synthetic_candles(4000, seed=10 + seed)
            _, _, rec = backtest_frame(df, f"RW{seed}", MarketType.CRYPTO, horizon=12)
            records.append(rec)
        pooled = summarize(pd.concat(records, ignore_index=True))
        checked = [s for s in pooled.values() if s["n_entries"] >= 30]
        self.assertTrue(checked)
        for s in checked:
            self.assertLess(abs(s["t_stat"]), 2.5)
        all_net = pd.concat(records)["net"].mean()
        self.assertLess(all_net, 0.0)  # costs make noise trading lose

    def test_backtest_matches_live_signals(self):
        """The signal the backtest assigns to candle k equals what the live engine computes from data up to k."""
        df = synthetic_candles(300, seed=21)
        frame = build_signal_frame(df, "X", MarketType.CRYPTO)
        for k in (100, 200, 299):
            live = generate_signal(compute_indicators(df.iloc[: k + 1], "X"), MarketType.CRYPTO)
            self.assertEqual(frame["signal"].iloc[k], live.signal_type.value)

    def test_rolling_sentiment_window(self):
        idx = pd.date_range("2026-01-01 00:00", periods=4, freq="1h", tz="UTC")
        news = pd.DataFrame({"sentiment_score": [0.5, -0.1]},
                            index=pd.to_datetime(["2025-12-31 23:30", "2026-01-01 01:30"], utc=True))
        means, counts = rolling_sentiment(idx, news, hours=1)
        self.assertEqual(list(counts), [1, 0, 1, 0])
        self.assertAlmostEqual(means[0], 0.5)
        self.assertAlmostEqual(means[2], -0.1)


class TestTagging(unittest.TestCase):
    def test_no_false_positives(self):
        self.assertEqual(tag_symbols("Canada's method for Europe solutions"), [])
        self.assertEqual(tag_symbols("Vale a pena investir agora?"), [])

    def test_accents_and_tickers(self):
        self.assertIn("ITUB4.SA", tag_symbols("Itaú reporta lucro recorde"))
        self.assertIn("ETHUSDT", tag_symbols("ETH rallies as Ethereum upgrade lands"))
        self.assertIn("VALE3.SA", tag_symbols("Ações da VALE3 sobem"))


class TestDatabaseAndTracking(unittest.TestCase):
    def _prices(self, symbol, df):
        return [Price(symbol, ts.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume)
                for ts, r in df.iterrows()]

    def test_utc_storage_and_hour_filter(self):
        with TempDB() as db:
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            idx = pd.date_range(end=now, periods=48, freq="1h")
            df = synthetic_candles(48).set_axis(idx)
            # store with a New York offset, as yfinance returns
            prices = [Price("AAPL", ts.tz_convert("America/New_York").to_pydatetime(), r.open, r.high, r.low,
                            r.close, r.volume) for ts, r in df.iterrows()]
            db.save_prices(prices)
            got = db.get_prices("AAPL", hours=10)
            self.assertTrue(9 <= len(got) <= 11)
            self.assertEqual(str(got.index.tz), "UTC")
            self.assertEqual(len(db.get_prices("AAPL", limit=5)), 5)

    def test_migrates_old_local_timestamps(self):
        with TempDB() as db:
            with db.get_connection() as conn:
                conn.execute("INSERT INTO prices VALUES ('PETR4.SA', '2026-03-02T10:00:00-03:00', 1, 1, 1, 1, 1)")
                conn.execute("DELETE FROM meta")
                conn.commit()
            db.init_db()
            with db.get_connection() as conn:
                ts = conn.execute("SELECT timestamp FROM prices").fetchone()[0]
            self.assertEqual(ts, "2026-03-02T13:00:00+00:00")

    def test_signal_dedupe_and_entries(self):
        with TempDB() as db:
            t0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)

            def sig(t, st):
                return Signal("BTCUSDT", MarketType.CRYPTO, t, st, 60.0, [], 100.0)

            self.assertIsNotNone(db.save_signal(sig(t0, SignalType.BUY)))
            self.assertIsNone(db.save_signal(sig(t0, SignalType.BUY)))  # same candle again
            s2 = sig(t0 + timedelta(minutes=5), SignalType.BUY)
            db.save_signal(s2)
            self.assertFalse(s2.is_entry)
            s3 = sig(t0 + timedelta(minutes=10), SignalType.HOLD)
            db.save_signal(s3)
            self.assertTrue(s3.is_entry)

    def test_symbol_sentiment_exact_match(self):
        with TempDB() as db:
            now = datetime.now(timezone.utc)
            db.save_news(NewsItem("a", "u1", "s", now - timedelta(hours=1), 0.6, ["ETHUSDT"]))
            db.save_news(NewsItem("b", "u2", "s", now - timedelta(hours=1), -0.9, ["ETH"]))
            db.save_news(NewsItem("c", "u3", "s", now - timedelta(hours=30), 0.9, ["ETHUSDT"]))
            avg, n = db.get_symbol_sentiment("ETHUSDT", hours=6)
            self.assertEqual(n, 1)
            self.assertAlmostEqual(avg, 0.6)

    def test_forward_tracking(self):
        from signal_engine.tracker import evaluate_pending_outcomes, performance_summary

        with TempDB() as db:
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            idx = pd.date_range(end=now, periods=600, freq="5min")
            df = synthetic_candles(600, seed=5).set_axis(idx)
            db.save_prices(self._prices("BTCUSDT", df))

            t_buy = idx[100].to_pydatetime()
            price = float(df["close"].iloc[100])
            s = Signal("BTCUSDT", MarketType.CRYPTO, t_buy, SignalType.BUY, 60.0, [], price)
            db.save_signal(s)
            graded = evaluate_pending_outcomes([60, 240])
            self.assertEqual(graded, 2)

            out = db.get_outcomes_frame()
            row = out[out["horizon_min"] == 60].iloc[0]
            expected_exit = float(df["close"].iloc[112])
            self.assertAlmostEqual(row["exit_price"], expected_exit)
            self.assertAlmostEqual(row["net_return"], expected_exit / price - 1 - config.cost_for(MarketType.CRYPTO))
            self.assertEqual(evaluate_pending_outcomes([60, 240]), 0)  # not graded twice

            summary = performance_summary()
            self.assertEqual(len(summary["horizons"]), 2)
            self.assertEqual(summary["horizons"][0]["by_signal_type"][0]["n"], 1)

    def test_backtest_results_feed_hit_rate(self):
        from signal_engine.backtester import get_hit_rate

        with TempDB() as db:
            db.save_backtest_results({"BTCUSDT": {
                "BUY": {"hit_rate": 51.2, "n_entries": 80, "enough_data": True},
                "SELL": {"hit_rate": 70.0, "n_entries": 4, "enough_data": False},
            }})
            self.assertEqual(get_hit_rate("BTCUSDT", SignalType.BUY), 51.2)
            self.assertIsNone(get_hit_rate("BTCUSDT", SignalType.SELL))
            self.assertIsNone(get_hit_rate("BTCUSDT", SignalType.HOLD))

    def test_backup(self):
        with TempDB() as db:
            old = db.BACKUP_DIR
            db.BACKUP_DIR = db.DB_PATH.parent / "backups"
            try:
                for _ in range(3):
                    self.assertIsNotNone(db.backup_db(keep=2))
                self.assertLessEqual(len(list(db.BACKUP_DIR.glob("*.db"))), 2)
            finally:
                db.BACKUP_DIR = old


if __name__ == "__main__":
    unittest.main()
