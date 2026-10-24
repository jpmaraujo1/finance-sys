"""
ML experiment: can a model trained on indicators + news sentiment beat the rules?

Run it after the engine has collected some history (or after a backfill):

    pip install scikit-learn            # once (see requirements_research.txt)
    python -m research.ml_experiment
    python -m research.ml_experiment --market crypto --horizon 48 --threshold 0.56

What it does, and how it avoids fooling you:
  * Builds features from the same trailing indicators the engine uses, plus recent
    returns, time of day and rolling news sentiment. Features are scale-free (ratios,
    z-scores), so all symbols of a market can be pooled into one dataset.
  * Label: did price rise over the next `horizon` candles?
  * Walk-forward validation: the data is split into consecutive time blocks; each
    block is predicted by a model trained only on data from *before* it, with a gap
    ("purge") of `horizon` candles so training labels never overlap the test period.
  * Trades are evaluated on non-overlapping bars (one every `horizon` candles), net
    of costs, and compared with: always long, the rule-based BUY signals, and a
    logistic regression.
  * Nothing here touches the live engine. If a model only wins in some folds, or
    by a hair, treat it as noise.
"""
import argparse
import math
from typing import Optional

import numpy as np
import pandas as pd

from signal_engine import config
from signal_engine.backtester import build_signal_frame, rolling_sentiment
from signal_engine.models import MarketType

SIGNAL_CODES = {"STRONG_SELL": -2, "SELL": -1, "HOLD": 0, "BUY": 1, "STRONG_BUY": 2}


def make_features(df: pd.DataFrame, symbol: str, market: MarketType, news: Optional[pd.DataFrame],
                  horizon: int) -> pd.DataFrame:
    frame = build_signal_frame(df, symbol, market, news)
    close = frame["close"]
    logc = np.log(close)

    feat = pd.DataFrame(index=frame.index)
    feat["rsi"] = frame["rsi"] / 100.0
    feat["macd_hist_rel"] = frame["macd_hist"] / close
    feat["bb_z"] = frame["bb_z"]
    feat["ema9_21"] = frame["ema9"] / frame["ema21"] - 1.0
    feat["close_ema50"] = close / frame["ema50"] - 1.0
    feat["close_ema200"] = close / frame["ema200"] - 1.0
    feat["vol_ratio"] = np.log((frame["volume"] + 1.0) / (frame["volume_sma20"] + 1.0))
    feat["atr_rel"] = frame["atr"] / close
    for lag in (1, 3, 12, 48):
        feat[f"ret_{lag}"] = logc.diff(lag)
    hour = frame.index.hour + frame.index.minute / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["weekday"] = frame.index.dayofweek
    s_mean, s_count = rolling_sentiment(frame.index, news, config.SENTIMENT_WINDOW_HOURS)
    feat["news_sent"] = np.nan_to_num(s_mean, nan=0.0)
    feat["news_count"] = np.log1p(s_count)
    feat["rule_signal"] = frame["signal"].map(SIGNAL_CODES)

    feat["fwd_return"] = close.shift(-horizon) / close - 1.0
    feat["symbol"] = symbol
    feat["rule"] = frame["signal"]
    feat["bar"] = np.arange(len(feat))
    return feat.replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "rsi", "macd_hist_rel", "bb_z", "ema9_21", "close_ema50", "close_ema200", "vol_ratio", "atr_rel",
    "ret_1", "ret_3", "ret_12", "ret_48", "hour_sin", "hour_cos", "weekday",
    "news_sent", "news_count", "rule_signal",
]


def walk_forward(data: pd.DataFrame, horizon: int, folds: int = 5, min_train_frac: float = 0.3):
    """Yield (train_idx, test_idx) over time-ordered blocks with a purge gap."""
    times = data.index.unique().sort_values()
    start = int(len(times) * min_train_frac)
    edges = np.linspace(start, len(times), folds + 1).astype(int)
    purge = pd.Timedelta(minutes=horizon * config.CANDLE_MINUTES)
    for k in range(folds):
        t0, t1 = times[edges[k]], times[min(edges[k + 1], len(times) - 1)]
        train = data.index < (t0 - purge)
        test = (data.index >= t0) & (data.index < t1) if k < folds - 1 else (data.index >= t0)
        if train.sum() > 500 and test.sum() > 50:
            yield k, np.where(train)[0], np.where(test)[0]


def evaluate_trades(df: pd.DataFrame, take: np.ndarray, cost: float) -> dict:
    r = df.loc[take, "fwd_return"].to_numpy()
    if len(r) == 0:
        return {"trades": 0, "hit_rate": None, "avg_net_pct": None, "t_stat": None}
    net = r - cost
    sd = np.std(net, ddof=1) if len(net) > 1 else np.nan
    t = float(np.mean(net) / (sd / math.sqrt(len(net)))) if len(net) > 1 and sd > 0 else None
    return {
        "trades": int(len(net)),
        "hit_rate": round(float((net > 0).mean() * 100), 1),
        "avg_net_pct": round(float(net.mean() * 100), 4),
        "t_stat": round(t, 2) if t is not None else None,
    }


def run_experiment(data: pd.DataFrame, horizon: int, cost: float, threshold: float, folds: int = 5) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = data.dropna(subset=["fwd_return", "rule_signal"]).sort_index()
    data = data.dropna(subset=[c for c in FEATURES if c not in ("close_ema200",)])
    X = data[FEATURES].to_numpy(dtype=float)
    y = (data["fwd_return"] > 0).astype(int).to_numpy()

    pred_gb = np.full(len(data), np.nan)
    pred_lr = np.full(len(data), np.nan)
    fold_rows = []
    for k, tr, te in walk_forward(data, horizon, folds):
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                            l2_regularization=1.0, random_state=0)
        gb.fit(X[tr], y[tr])
        pred_gb[te] = gb.predict_proba(X[te])[:, 1]

        Xtr = np.nan_to_num(X[tr]); Xte = np.nan_to_num(X[te])
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))
        lr.fit(Xtr, y[tr])
        pred_lr[te] = lr.predict_proba(Xte)[:, 1]

        auc = roc_auc_score(y[te], pred_gb[te]) if len(np.unique(y[te])) > 1 else float("nan")
        fold_rows.append({"fold": k, "test_from": str(data.index[te[0]]), "test_rows": int(len(te)),
                          "gb_auc": round(float(auc), 3), "up_rate": round(float(y[te].mean()), 3)})

    tested = ~np.isnan(pred_gb)
    # Non-overlapping evaluation bars: one every `horizon` candles per symbol
    sample = tested & (data["bar"].to_numpy() % horizon == 0)
    idx = np.arange(len(data))

    results = {
        "rows": int(len(data)),
        "tested_rows": int(tested.sum()),
        "folds": fold_rows,
        "always_long": evaluate_trades(data.reset_index(), idx[sample], cost),
        "rules_buy": evaluate_trades(
            data.reset_index(), idx[sample & data["rule"].isin(["BUY", "STRONG_BUY"]).to_numpy()], cost),
        "gradient_boosting": evaluate_trades(data.reset_index(), idx[sample & (pred_gb > threshold)], cost),
        "logistic_regression": evaluate_trades(data.reset_index(), idx[sample & (pred_lr > threshold)], cost),
    }
    if tested.any() and len(np.unique(y[tested])) > 1:
        results["gb_auc_overall"] = round(float(roc_auc_score(y[tested], pred_gb[tested])), 3)
        results["lr_auc_overall"] = round(float(roc_auc_score(y[tested], pred_lr[tested])), 3)
    return results


def print_results(res: dict, market: str, horizon: int, threshold: float):
    hours = horizon * config.CANDLE_MINUTES / 60
    print(f"\nML experiment: {market}, predicting {hours:g}h ahead, trade when P(up) > {threshold}")
    print(f"{res['rows']} rows, {res['tested_rows']} predicted out-of-sample\n")
    print(f"{'fold':>4} {'test from':<27} {'rows':>7} {'AUC':>6} {'up rate':>8}")
    for f in res["folds"]:
        print(f"{f['fold']:>4} {f['test_from']:<27} {f['test_rows']:>7} {f['gb_auc']:>6} {f['up_rate']:>8}")
    if "gb_auc_overall" in res:
        print(f"\nOverall AUC: gradient boosting {res['gb_auc_overall']}, logistic {res['lr_auc_overall']} "
              f"(0.5 = coin flip; anything under ~0.52 is effectively no signal)")
    print(f"\n{'approach':<22} {'trades':>7} {'hit%':>6} {'avg net%':>9} {'t':>6}")
    for name in ("always_long", "rules_buy", "logistic_regression", "gradient_boosting"):
        r = res[name]
        print(f"{name:<22} {r['trades']:>7} {str(r['hit_rate']):>6} {str(r['avg_net_pct']):>9} {str(r['t_stat']):>6}")
    print("\nA model is only interesting if it beats always_long AND rules_buy, with t above ~2, consistently\n"
          "across folds. Re-run after more months of data before trusting any result.\n")


def load_market_data(market: MarketType, horizon: int, days: int) -> pd.DataFrame:
    from signal_engine.database import init_db, get_watchlist, get_prices, get_news_frame

    init_db()
    frames = []
    for item in get_watchlist():
        if item.market != market:
            continue
        df = get_prices(item.symbol, hours=days * 24)
        if len(df) < 1000:
            print(f"Skipping {item.symbol}: only {len(df)} candles")
            continue
        frames.append(make_features(df, item.symbol, market, get_news_frame(item.symbol), horizon))
    return pd.concat(frames) if frames else pd.DataFrame()


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Walk-forward ML experiment on stored data")
    parser.add_argument("--market", default="crypto", choices=[m.value for m in MarketType])
    parser.add_argument("--horizon", type=int, default=12, help="candles ahead to predict (12 = 1h)")
    parser.add_argument("--threshold", type=float, default=0.55, help="trade when P(up) exceeds this")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    market = MarketType(args.market)
    data = load_market_data(market, args.horizon, args.days)
    if data.empty:
        print("Not enough data yet. Run `python -m signal_engine.backfill` or let the engine collect history.")
        return
    res = run_experiment(data, args.horizon, config.cost_for(market), args.threshold, args.folds)
    print_results(res, market.value, args.horizon, args.threshold)


if __name__ == "__main__":
    main()
