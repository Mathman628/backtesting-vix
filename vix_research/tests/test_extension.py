import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from vix_research.extension_engine import FiveAsset, GradualVix, simulate, FIVE, UNIVERSE
from vix_research.extension_models import filter_step, mixture, wasserstein_cost, optimise, forecasts
from vix_research.extension_study import inputs
from vix_research.backtest import prepare_snapshots, run_backtest
from vix_research.core import StrategyConfig, ASSETS
from vix_research.literature_strategy import ResearchConfig, ResearchEngine, research_snapshots


class ExtensionTests(unittest.TestCase):
    def test_mixture_label_invariance_and_between_state_variance(self):
        p=np.array([.2,.3,.5]);m=np.array([[0,1],[2,0],[1,2.]])
        c=np.array([np.eye(2)]*3)
        mu,cov=mixture(p,m,c);idx=[2,0,1]
        other=mixture(p[idx],m[idx],c[idx])
        np.testing.assert_allclose(mu,other[0]);np.testing.assert_allclose(cov,other[1])
        self.assertGreater(np.linalg.eigvalsh(cov).min(),1)
        d=wasserstein_cost(m,c[:,0,:]+1,m,c[:,0,:]+1)
        np.testing.assert_allclose(np.diag(d),0);np.testing.assert_allclose(d,d.T)

    def test_filter_and_optimiser(self):
        p=filter_step(np.array([.5,.5]),np.array([0.]),np.array([[0.],[2.]]),np.ones((2,1)))
        self.assertAlmostEqual(p[0],1/(1+np.exp(-2)))
        current=np.array([.25]*4);cov=np.eye(4)*.0001;mu=np.array([.0001,0,0,0])
        a=optimise(mu,cov,current,0);b=optimise(mu,cov,current,.001)
        self.assertAlmostEqual(b.sum(),1);self.assertLessEqual(b[:3].max(),.600001)
        self.assertLessEqual(abs(b-current).sum(),abs(a-current).sum()+1e-7)

    def test_survivors_slots_and_future(self):
        dates=pd.bdate_range('2000-01-01',periods=230)
        frame=pd.DataFrame(100.,index=dates,columns=list(UNIVERSE))
        frame.loc[dates[209],'SPY']=101
        snap=SimpleNamespace(date=dates[209].date());day=dates[210]
        a=FiveAsset(frame,'survivors').decide(snap,day,{})
        b=FiveAsset(frame,'slots').decide(snap,day,{})
        self.assertAlmostEqual(a.targets['SPY'],.995);self.assertAlmostEqual(b.targets['SPY'],.199)
        frame.loc[day:]=1e8
        self.assertEqual(a.targets,FiveAsset(frame,'survivors').decide(snap,day,{}).targets)
        frame.iloc[:210]=100
        self.assertEqual(sum(FiveAsset(frame,'survivors').decide(snap,day,{}).targets.values()),0)

    def test_causal_forecast_prefix(self):
        rng=np.random.default_rng(8);dates=pd.bdate_range('2000-01-01',periods=240)
        prices=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,.005,(240,4)),axis=0)),index=dates,columns=list(ASSETS))
        cut=dates[220];a,diag=forecasts(prices,start=str(dates[200].date()),window=120)
        altered=prices.copy();altered.loc[altered.index>cut]*=10
        b,_=forecasts(altered,start=str(dates[200].date()),window=120)
        for day in a:
            if day<=cut:
                for mode in ['hmm','tracked','unconditional']:
                    for x,y in zip(a[day][mode],b[day][mode]):np.testing.assert_allclose(x,y,atol=1e-12)
        self.assertTrue((diag.train_end<diag.execution_date).all())

    def test_real_accounting_and_hard_vix_parity(self):
        data,opens,closes=inputs()
        # Four years include the financial crisis and its recovery.
        stop='2010-12-31';data.opens=data.opens.loc[:stop];data.closes=data.closes.loc[:stop]
        opens=opens.loc[:stop];closes=closes.loc[:stop]
        cfg=ResearchConfig(signal='absolute12',vix_overlay=True)
        snaps=research_snapshots(data,cfg,prepare_snapshots(data,StrategyConfig()))
        ref=run_backtest(data,cfg,start='2007-08-29',end=stop,snapshots=snaps,engine=ResearchEngine(cfg))
        hard=GradualVix(False)
        result=simulate(opens,closes,data.risk_free_returns(),snaps,hard,'2007-08-29',stop,'hard')
        np.testing.assert_allclose(result.daily.nav,ref.daily.nav,rtol=1e-12)
        np.testing.assert_allclose(result.daily['return'],ref.daily['return'],atol=1e-12)
        smooth=GradualVix(True)
        simulate(opens,closes,data.risk_free_returns(),snaps,smooth,'2007-08-29',stop,'smooth')
        history=pd.DataFrame(smooth.history)
        self.assertTrue((history.loc[history.regime==3,'applied_multiplier']==0).all())
        self.assertTrue(history.applied_multiplier.between(0,1).all())


if __name__=='__main__':unittest.main()
