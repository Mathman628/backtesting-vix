import ast
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from vix_research.core import (ASSETS, StrategyConfig, Snapshot, SignalEngine,
                               build_targets, make_snapshot, estimated_vol, cap_portfolio_vol)
from vix_research.data import synthetic_data, save_data, load_data
from vix_research.backtest import (ExecutionConfig, prepare_snapshots, run_backtest,
                                   rebalance_values, compare)
from vix_research.analytics import performance, drawdown, block_bootstrap, walk_forward
from vix_research.cli import build_quantconnect


def snapshot(day=date(2020, 1, 3), percentile=.4, vix=20, **kwargs):
    values = dict(date=day, vix=vix, percentile=percentile, trend_ok=True,
                  momentum_ok=True, defensive_ok={"IEF": True, "GLD": True},
                  covariance=np.diag(np.array([.20, .10, .25, .005]) ** 2), equity_vol=.20)
    values.update(kwargs)
    return Snapshot(**values)


def small_config(**kwargs):
    return StrategyConfig(percentile_window=5, trend_window=4, momentum_window=5,
                          defensive_window=3, covariance_window=4, equity_vol_window=3, **kwargs)


class RegimeTests(unittest.TestCase):
    def test_percentile_can_reach_panic(self):
        engine = SignalEngine()
        engine.regime = 2
        decision = engine.decide(snapshot(percentile=1.0, vix=30), date(2020, 1, 6), {})
        self.assertEqual(decision.regime, 3)

    def test_same_signal_does_not_advance_twice(self):
        engine = SignalEngine()
        signal = snapshot(percentile=.99)
        first = engine.decide(signal, date(2020, 1, 6), {})
        second = engine.decide(signal, date(2020, 1, 6), {})
        self.assertEqual(first.regime, 1)
        self.assertIsNone(second)
        self.assertEqual(engine.regime, 1)

    def test_month_week_overlap_is_one_transition(self):
        engine = SignalEngine()
        signal = snapshot(day=date(2020, 5, 29), percentile=.99)
        decision = engine.decide(signal, date(2020, 6, 1), {})
        self.assertEqual(decision.regime, 1)

    def test_daily_hard_panic(self):
        engine = SignalEngine()
        engine.decide(snapshot(), date(2020, 1, 6), {})
        decision = engine.decide(snapshot(day=date(2020, 1, 6), vix=40),
                                 date(2020, 1, 7), {"SPY": .9})
        self.assertEqual(decision.regime, 3)
        self.assertEqual(decision.reason, "panic")
        self.assertEqual(decision.targets["SPY"], 0)

    def test_normal_regime_only_changes_weekly(self):
        engine = SignalEngine()
        engine.decide(snapshot(), date(2020, 1, 6), {})
        decision = engine.decide(snapshot(day=date(2020, 1, 6), percentile=.99),
                                 date(2020, 1, 7), {})
        self.assertEqual(decision.regime, 0)

    def test_panic_exit_requires_level_and_percentile(self):
        engine = SignalEngine()
        engine.regime = 3
        decision = engine.decide(snapshot(vix=32, percentile=.1), date(2020, 1, 6), {})
        self.assertEqual(decision.regime, 3)
        decision = engine.decide(snapshot(day=date(2020, 1, 10), vix=29, percentile=.1),
                                 date(2020, 1, 13), {})
        self.assertEqual(decision.regime, 2)

    def test_same_day_signal_rejected(self):
        with self.assertRaises(ValueError):
            SignalEngine().decide(snapshot(), date(2020, 1, 3), {})

    def test_actual_weight_drift_is_detected(self):
        cfg = StrategyConfig(equity_filter="none", vol_model="none")
        engine = SignalEngine(cfg)
        first = engine.decide(snapshot(), date(2020, 1, 6), {})
        actual = dict(first.targets)
        actual["SPY"] -= .15
        actual["BIL"] += .15
        result = engine.decide(snapshot(day=date(2020, 1, 10)), date(2020, 1, 13), actual)
        self.assertTrue(result.trade)
        self.assertEqual(result.reason, "weekly-drift")


class RiskTests(unittest.TestCase):
    def test_full_portfolio_vol_includes_gold_and_bonds(self):
        signal = snapshot(covariance=np.diag(np.array([.2, .30, .50, .001]) ** 2))
        targets = build_targets(signal, 2, StrategyConfig())
        self.assertLessEqual(estimated_vol(targets, signal.covariance), .10 + 1e-10)
        self.assertLess(targets["GLD"], .35)
        self.assertGreater(targets["BIL"], .25)

    def test_correlated_assets_affect_risk(self):
        covariance = np.diag(np.array([.20, .20, .20, .001]) ** 2)
        weights = {"SPY": .5, "IEF": .5}
        independent = estimated_vol(weights, covariance)
        covariance[0, 1] = covariance[1, 0] = .035
        self.assertGreater(estimated_vol(weights, covariance), independent)

    def test_defensive_guard_transfers_failed_asset(self):
        signal = snapshot(defensive_ok={"IEF": False, "GLD": True})
        targets = build_targets(signal, 2, StrategyConfig(vol_model="none", cash_buffer=0))
        self.assertEqual(targets["IEF"], 0)
        self.assertAlmostEqual(targets["BIL"], .65)

    def test_equity_filter_ablation(self):
        signal = snapshot(trend_ok=True, momentum_ok=False)
        self.assertEqual(build_targets(signal, 0, StrategyConfig())["SPY"], 0)
        self.assertGreater(build_targets(signal, 0, StrategyConfig(equity_filter="sma"))["SPY"], 0)

    def test_cash_fallback_if_bil_exceeds_budget(self):
        covariance = np.diag(np.array([.5, .5, .5, .2]) ** 2)
        weights = cap_portfolio_vol({"BIL": 1}, covariance, .1)
        self.assertAlmostEqual(weights["BIL"], .5)

    def test_weights_and_risk_limits_over_random_covariances(self):
        rng = np.random.default_rng(4)
        for _ in range(60):
            matrix = rng.normal(0, .12, (4, 4))
            covariance = matrix @ matrix.T
            signal = snapshot(covariance=covariance)
            for regime in range(4):
                weights = build_targets(signal, regime, StrategyConfig())
                self.assertTrue(all(w >= 0 for w in weights.values()))
                self.assertLessEqual(sum(weights.values()), .995 + 1e-9)
                self.assertLessEqual(estimated_vol(weights, covariance), .1 + 1e-9)

    def test_invalid_parameters_rejected(self):
        for params in ({"target_vol": -1}, {"entry_thresholds": (.65, .85, 1.05)},
                       {"covariance_window": 1}, {"panic_exit_vix": 40},
                       {"cash_buffer": np.nan}, {"equity_filter": "guess"}):
            with self.assertRaises(ValueError):
                StrategyConfig(**params)


class DataTests(unittest.TestCase):
    def test_missing_covariance_observation_not_ready(self):
        data = synthetic_data(300)
        data.closes.iloc[-3, data.closes.columns.get_loc("IEF")] = np.nan
        self.assertIsNone(make_snapshot(data.closes, StrategyConfig()))

    def test_short_history_not_ready(self):
        self.assertIsNone(make_snapshot(synthetic_data(200).closes, StrategyConfig()))

    def test_bil_inception_does_not_discard_spy_warmup(self):
        data = synthetic_data(300)
        data.closes.loc[data.closes.index[:230], "BIL"] = np.nan
        self.assertIsNotNone(make_snapshot(data.closes, StrategyConfig()))

    def test_missing_price_not_forward_filled(self):
        data = synthetic_data(300)
        data.closes.iloc[270, 0] = np.nan
        with self.assertRaises(ValueError):
            data.validate()

    def test_risk_free_rate_lag(self):
        data = synthetic_data(300)
        before = data.risk_free_returns()
        data.rf_annual.iloc[200] = .9
        after = data.risk_free_returns()
        self.assertEqual(before.iloc[200], after.iloc[200])
        self.assertNotEqual(before.iloc[201], after.iloc[201])

    def test_stale_risk_free_rate_rejected(self):
        data = synthetic_data(300)
        data.rf_annual = data.rf_annual.iloc[:100]
        with self.assertRaises(ValueError):
            data.risk_free_returns()

    def test_csv_roundtrip_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = synthetic_data(300)
            save_data(data, directory)
            loaded = load_data(directory)
            np.testing.assert_allclose(data.closes, loaded.closes, rtol=1e-12)
            self.assertIn("prices_sha256", loaded.metadata)
            self.assertTrue(loaded.metadata["synthetic"])

    def test_duplicate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            save_data(synthetic_data(300), directory)
            path = Path(directory) / "prices.csv"
            frame = pd.read_csv(path)
            pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_data(directory)


class BacktestTests(unittest.TestCase):
    def test_gap_loss_is_paid_before_panic_exit(self):
        data = synthetic_data(25)
        data.closes.loc[:, list(ASSETS)] = 100.0
        data.opens.loc[:, :] = 100.0
        data.rf_annual.loc[:] = 0.0
        data.closes["VIX"] = np.linspace(30, 10, len(data.closes))
        data.closes.loc[data.closes.index[8], "VIX"] = 40
        data.closes.loc[data.closes.index[9]:, "SPY"] = 50
        data.opens.loc[data.opens.index[9]:, "SPY"] = 50
        cfg = small_config(equity_filter="none", defensive_filter=False, vol_model="none", cash_buffer=0)
        result = run_backtest(data, cfg, ExecutionConfig(100000, 0, 0))
        crash = result.daily.loc[data.closes.index[9]]
        self.assertEqual(crash["reason"], "panic")
        self.assertAlmostEqual(crash["nav"], 50_000)
        self.assertAlmostEqual(crash["return"], -.5)

    def test_future_data_cannot_change_past_results(self):
        original = synthetic_data(350)
        changed = synthetic_data(350)
        changed.closes.iloc[320:] *= 1.5
        changed.opens.iloc[320:] *= 1.5
        a = run_backtest(original)
        b = run_backtest(changed)
        pd.testing.assert_frame_equal(a.daily.loc[:original.closes.index[319]],
                                      b.daily.loc[:original.closes.index[319]])
        pd.testing.assert_frame_equal(a.signals.loc[:original.closes.index[320]],
                                      b.signals.loc[:original.closes.index[320]])

    def test_costs_are_self_financing_and_both_sides(self):
        old = np.array([50_000., 0, 0, 50_000.])
        target = np.array([0., .5, .5, 0.])
        values, cash, deltas, cost = rebalance_values(old, 0, target, .001)
        self.assertAlmostEqual(values.sum() + cash + cost, 100_000, places=7)
        self.assertAlmostEqual(cost, abs(deltas).sum() * .001)
        self.assertGreater(cost, 190)
        self.assertGreaterEqual(cash, 0)

    def test_higher_costs_reduce_buy_and_hold_nav(self):
        data = synthetic_data(350)
        zero = run_backtest(data, execution=ExecutionConfig(100000, 0, 0), baseline="spy")
        costly = run_backtest(data, execution=ExecutionConfig(100000, 1, 25), baseline="spy")
        self.assertLess(costly.daily.nav.iloc[-1], zero.daily.nav.iloc[-1])

    def test_cash_and_holdings_weights_sum_to_one(self):
        result = run_backtest(synthetic_data(350))
        np.testing.assert_allclose(result.weights.sum(axis=1), 1, atol=1e-10)
        self.assertTrue((result.weights >= 0).all().all())

    def test_comparison_dates_match_for_different_windows(self):
        data = synthetic_data(380)
        variants = {"short": (StrategyConfig(), None),
                    "long": (StrategyConfig(momentum_window=300), None)}
        results = compare(data, variants)
        self.assertTrue(results["short"].daily.index.equals(results["long"].daily.index))


class AnalyticsTests(unittest.TestCase):
    def test_initial_loss_is_drawdown(self):
        series = pd.Series([-.1, 0, .05])
        self.assertAlmostEqual(drawdown(series).min(), -.1)

    def test_flat_excess_returns_have_undefined_sharpe(self):
        daily = pd.DataFrame({"return": [0.] * 10, "risk_free_return": [0.] * 10,
                              "turnover": [0.] * 10, "cost": [0.] * 10, "traded": [False] * 10})
        self.assertTrue(np.isnan(performance(daily)["sharpe"]))

    def test_missing_returns_are_not_silently_dropped(self):
        daily = pd.DataFrame({"return": [.01, np.nan], "risk_free_return": [0., 0.],
                              "turnover": [0., 0.], "cost": [0., 0.], "traded": [False, False]})
        with self.assertRaises(ValueError):
            performance(daily)

    def test_bootstrap_is_reproducible(self):
        daily = run_backtest(synthetic_data(320)).daily
        a = block_bootstrap(daily, samples=20)
        b = block_bootstrap(daily, samples=20)
        pd.testing.assert_frame_equal(a, b)

    def test_walk_forward_training_precedes_test(self):
        folds = walk_forward(synthetic_data(1100), train_years=1, test_years=1)
        self.assertGreater(len(folds), 0)
        self.assertTrue((folds.train_end < folds.test_start).all())
        self.assertTrue((folds.test_start.shift(-1).dropna().to_numpy()
                         > folds.test_end.iloc[:-1].to_numpy()).all())

    def test_generated_quantconnect_contains_same_core(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_quantconnect(Path(directory) / "main.py")
            text = path.read_text(encoding="utf-8")
            ast.parse(text)
            core = (Path(__file__).parents[1] / "core.py").read_text(encoding="utf-8")
            self.assertIn(core, text)
            self.assertNotIn("from vix_research.core import", text)


if __name__ == "__main__":
    unittest.main()
