"""Adapter contract checks with a stub, not an actual LEAN integration test."""
from datetime import datetime, timedelta
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from vix_research.core import ASSETS, StrategyConfig, SignalEngine
from vix_research.data import synthetic_data


class AdapterTests(unittest.TestCase):
    def make_adapter(self):
        imports = ModuleType("AlgorithmImports")
        imports.QCAlgorithm = object
        imports.Resolution = SimpleNamespace(DAILY="Daily")
        imports.PortfolioTarget = lambda symbol, weight: (symbol, weight)
        imports.OrderStatus = SimpleNamespace(INVALID="Invalid", CANCELED="Canceled")
        with patch.dict(sys.modules, {"AlgorithmImports": imports}):
            namespace = runpy.run_path(str(Path(__file__).parents[1] / "qc_adapter.py"))
        algorithm = namespace["VixFearRegimeStrategy"]()
        data = synthetic_data(300)
        next_session = data.closes.index[-1] + pd.offsets.BDay(1)
        algorithm.time = next_session.to_pydatetime().replace(hour=9, minute=35)
        algorithm.utc_time = algorithm.time + timedelta(hours=5)
        algorithm.is_warming_up = False
        algorithm._config = StrategyConfig()
        algorithm._engine = SignalEngine(algorithm._config)
        algorithm._symbols = dict(zip(ASSETS, ASSETS))
        algorithm._vix = "VIX"
        algorithm._audit, algorithm._fills = [], []
        algorithm._retry = False
        algorithm._last_callback_date = algorithm._last_skip_reason = None
        algorithm.securities = {
            symbol: SimpleNamespace(price=100, has_data=True,
                get_last_data=lambda: SimpleNamespace(end_time=algorithm.time - timedelta(minutes=1)),
                exchange=SimpleNamespace(hours=SimpleNamespace(is_date_open=lambda when: when.weekday() < 5)))
            for symbol in ASSETS}
        frames = {symbol: pd.DataFrame({"close": data.closes[symbol].to_numpy()},
                                       index=data.closes.index + pd.Timedelta(hours=16))
                  for symbol in (*ASSETS, "VIX")}
        history = pd.concat(frames)
        algorithm.history = lambda *args, **kwargs: history.copy()
        algorithm.transactions = SimpleNamespace(get_open_orders=lambda: [])
        class Portfolio(dict):
            total_portfolio_value = 100_000
        algorithm.portfolio = Portfolio({symbol: SimpleNamespace(holdings_value=0) for symbol in ASSETS})
        algorithm.plot = lambda *args: None
        algorithm.logs = []
        algorithm.log = algorithm.logs.append
        algorithm.submissions = []
        algorithm.set_holdings = lambda targets, **kwargs: algorithm.submissions.append((targets, kwargs))
        return algorithm

    def test_completed_history_keeps_prior_session(self):
        algorithm = self.make_adapter()
        closes = algorithm._completed_closes()
        self.assertLess(closes.index[-1].date(), algorithm.time.date())
        self.assertEqual(len(closes), 300)

    def test_decision_submits_all_assets_once(self):
        algorithm = self.make_adapter()
        algorithm._daily_decision()
        algorithm._daily_decision()
        self.assertEqual(len(algorithm.submissions), 1)
        targets = algorithm.submissions[0][0]
        self.assertEqual({s for s, _ in targets}, set(ASSETS))
        self.assertLessEqual(sum(w for _, w in targets), .995 + 1e-10)
        self.assertEqual(len(algorithm._audit), 1)
        self.assertLess(algorithm._audit[0]["signal_date"], str(algorithm.time.date()))

    def test_stale_execution_price_skips_trade(self):
        algorithm = self.make_adapter()
        algorithm.securities["GLD"].get_last_data = lambda: SimpleNamespace(end_time=algorithm.time - timedelta(days=1))
        algorithm._daily_decision()
        self.assertEqual(algorithm.submissions, [])
        self.assertIn("stale execution price", algorithm.logs[-1])

    def test_open_order_prevents_duplicate_submission(self):
        algorithm = self.make_adapter()
        algorithm.transactions.get_open_orders = lambda: [SimpleNamespace(time=algorithm.utc_time, id=1)]
        algorithm._daily_decision()
        self.assertEqual(algorithm.submissions, [])
        self.assertTrue(algorithm._retry)

    def test_invalid_order_sets_retry(self):
        algorithm = self.make_adapter()
        event = SimpleNamespace(order_id=1, symbol="SPY", status="Invalid", fill_quantity=0,
                                 fill_price=0, message="Insufficient buying power")
        algorithm.on_order_event(event)
        self.assertTrue(algorithm._retry)
        self.assertEqual(len(algorithm._fills), 1)


if __name__ == "__main__":
    unittest.main()
