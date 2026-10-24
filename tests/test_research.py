import unittest

import numpy as np
import pandas as pd

from tests._support import synthetic_candles

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def planted_mean_reversion(n, seed):
    """Returns that partly undo the previous hour: a real, learnable pattern."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.002, n)
    r = np.zeros(n)
    for t in range(12, n):
        r[t] = e[t] - 0.08 * r[t - 12:t].sum()
    df = synthetic_candles(n, seed=seed)
    close = 100 * np.exp(np.cumsum(r))
    df["close"] = close
    df["open"] = np.r_[100, close[:-1]]
    df["high"] = np.maximum(df.open, df.close) * 1.001
    df["low"] = np.minimum(df.open, df.close) * 0.999
    return df


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
class TestMLExperiment(unittest.TestCase):
    def _run(self, gen):
        from research.ml_experiment import make_features, run_experiment
        from signal_engine.models import MarketType

        data = pd.concat([make_features(gen(6000, sd), f"S{sd}", MarketType.CRYPTO, None, 12) for sd in range(3)])
        return run_experiment(data, 12, 0.002, 0.55, folds=4)

    def test_noise_has_no_signal(self):
        res = self._run(lambda n, sd: synthetic_candles(n, seed=sd))
        self.assertLess(res["gb_auc_overall"], 0.55)

    def test_finds_planted_pattern(self):
        res = self._run(planted_mean_reversion)
        self.assertGreater(res["gb_auc_overall"], 0.58)
        for fold in res["folds"]:
            self.assertGreater(fold["gb_auc"], 0.55)


if __name__ == "__main__":
    unittest.main()
