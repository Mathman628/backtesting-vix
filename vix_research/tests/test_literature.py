from dataclasses import replace
from datetime import date
import unittest
import numpy as np
import pandas as pd

from vix_research.core import Snapshot, ASSETS, estimated_vol
from vix_research.data import synthetic_data
from vix_research.backtest import run_backtest, prepare_snapshots
from vix_research.literature_strategy import ResearchConfig, ResearchSnapshot, ResearchEngine, research_snapshots, variants
from vix_research.literature_analysis import attribution, paired_sharpes, deflated_sharpe


def snap(day=date(2020,1,3), scores=None, vix=20):
    return ResearchSnapshot(day,vix,.5,True,True,{"IEF":True,"GLD":True},
        np.diag(np.array([.2,.1,.2,.001])**2),.2,scores or {s:1. for s in ASSETS[:3]})


class LiteratureTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"signal":"bad"},{"allocation":"bad"},{"adjustment_fraction":0},
                       {"vix_overlay":"false"},{"routine_band":-1},{"reentry_cap":2}):
            with self.assertRaises(ValueError):
                ResearchConfig(**kwargs)

    def test_cash_hurdle_and_rate_lag(self):
        data=synthetic_data(300)
        dates=data.closes.index
        for s in ASSETS[:3]:
            data.closes[s]=100*np.exp(np.arange(300)*.00005)
            data.opens[s]=data.closes[s]
        data.rf_annual[:]=.10
        absolute=research_snapshots(data,ResearchConfig(signal="absolute12"))
        excess=research_snapshots(data,ResearchConfig(signal="excess12"))
        self.assertEqual(absolute[-1].scores["SPY"],1.)
        self.assertEqual(excess[-1].scores["SPY"],0.)
        # Rates on the execution day and later cannot alter today's signal.
        before=excess[280].scores.copy()
        data.rf_annual.loc[dates[280]:]=.001
        self.assertEqual(before,research_snapshots(data,ResearchConfig(signal="excess12"))[280].scores)

    def test_future_prices_do_not_change_prefix(self):
        data=synthetic_data(330)
        cfg=ResearchConfig(signal="ensemble")
        before=research_snapshots(data,cfg)
        data.closes.iloc[301:,:4]*=3
        after=research_snapshots(data,cfg)
        for i in range(253,302):
            self.assertEqual(before[i].scores,after[i].scores)
            np.testing.assert_array_equal(before[i].covariance,after[i].covariance)

    def test_ensemble_uses_all_three_signals(self):
        data=synthetic_data(300)
        data.rf_annual[:]=0
        for s in ASSETS[:3]:
            data.closes[s]=100.
            data.closes.iloc[46,data.closes.columns.get_loc(s)]=90
            data.closes.iloc[235,data.closes.columns.get_loc(s)]=110
            data.closes.iloc[277,data.closes.columns.get_loc(s)]=90
        score=research_snapshots(data,ResearchConfig(signal="ensemble"))[299].scores["SPY"]
        self.assertAlmostEqual(score,2/3)

    def test_vix_overlay_preserves_risky_mix(self):
        a=ResearchEngine(ResearchConfig(signal="none",vol_model="none"))
        b=ResearchEngine(ResearchConfig(signal="none",vol_model="none",vix_overlay=True))
        s=replace(snap(),percentile=.7)
        x=a.decide(s,date(2020,1,6),{}).targets
        y=b.decide(s,date(2020,1,6),{}).targets
        for symbol in ASSETS[:3]:
            self.assertAlmostEqual(y[symbol],x[symbol]*.75)
        self.assertAlmostEqual(sum(y.values()),.995)

    def test_smoothing_never_delays_zero_target_exit(self):
        cfg=ResearchConfig(vol_model="none",adjustment_fraction=.5,reentry_cap=.2)
        engine=ResearchEngine(cfg)
        engine.decide(snap(),date(2020,1,6),{})
        decision=engine.decide(snap(date(2020,1,6),{"SPY":0.,"IEF":1.,"GLD":1.}),date(2020,1,7),{"SPY":.5,"BIL":.495})
        self.assertEqual(decision.reason,"guard-exit")
        self.assertTrue(decision.trade)
        self.assertEqual(decision.targets["SPY"],0.)

    def test_partial_routine_and_reentry_budget(self):
        engine=ResearchEngine(ResearchConfig(vol_model="none",adjustment_fraction=.5,reentry_cap=.2))
        engine.decide(snap(),date(2020,1,6),{})
        actual={"BIL":.995}
        decision=engine.decide(snap(date(2020,1,31)),date(2020,2,3),actual)
        self.assertAlmostEqual(sum(decision.targets[s] for s in ASSETS[:3]),.2)
        self.assertAlmostEqual(sum(decision.targets.values()),.995)

    def test_config_family_and_risk_cap(self):
        models=variants()
        self.assertEqual(len(models),16)
        for cfg in models.values():
            decision=ResearchEngine(cfg).decide(snap(),date(2020,1,6),{})
            self.assertLessEqual(sum(decision.targets.values()),1.)
            if cfg.vol_model=="portfolio":
                self.assertLessEqual(estimated_vol(decision.targets,snap().covariance),cfg.target_vol+1e-10)

    def test_attribution_reconciles_overnight_and_costs(self):
        data=synthetic_data(420)
        # Include substantial overnight moves and trading costs.
        data.opens.iloc[300,0]*=.8
        r=run_backtest(data)
        summary,_,error=attribution(r)
        self.assertLess(error,1e-12)
        self.assertAlmostEqual(summary.linked_total_return_contribution.sum(),(1+r.daily['return']).prod()-1,places=11)
        self.assertLess(summary.loc['contribution_cost','linked_total_return_contribution'],0.)

    def test_identical_returns_have_zero_paired_difference(self):
        r=run_backtest(synthetic_data(420))
        draws=paired_sharpes({"a":r,"b":r},samples=100)
        np.testing.assert_array_equal(draws.a-draws.b,np.zeros(100))
        rows=deflated_sharpe({"a":r,"b":r},["a","b"])
        self.assertTrue(rows.dsr_probability_approx.between(0,1).all())


if __name__=="__main__":
    unittest.main()
