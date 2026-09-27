import unittest
import numpy as np
import pandas as pd
from vix_research.data import synthetic_data
from vix_research.backtest import run_backtest,prepare_snapshots
from vix_research.core import StrategyConfig,ASSETS
from vix_research.rolling_evaluation import ScheduledEngine,engine_for,select_policy


class RollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=synthetic_data(900)
        cls.base=prepare_snapshots(cls.data,StrategyConfig())

    def schedule(self,change=False):
        return {y:dict(selected='monthly_50_25_25' if change and y>=2019 else 'treasury_60_40',
                       train_start='2017-01-01',train_end=f'{y-1}-12-31') for y in (2018,2019,2020)}

    def test_repeated_choice_preserves_continuous_account(self):
        router=ScheduledEngine(self.schedule(),self.data.closes.index,{'base':self.base})
        r=run_backtest(self.data,start='2018-01-01',snapshots=self.base,engine=router)
        control=run_backtest(self.data,start='2018-01-01',snapshots=self.base,engine=engine_for('treasury_60_40'))
        np.testing.assert_allclose(r.daily.nav,control.daily.nav,rtol=1e-13)
        self.assertEqual(sum(x['policy_changed'] for x in router.history),1)

    def test_switch_keeps_old_holdings_gap_and_charges_net_cost(self):
        router=ScheduledEngine(self.schedule(True),self.data.closes.index,{'base':self.base})
        r=run_backtest(self.data,start='2018-01-01',snapshots=self.base,engine=router)
        day=r.daily.loc['2019':].index[0]
        i=r.daily.index.get_loc(day)
        prev=r.daily.index[i-1]
        w=r.weights.loc[prev,list(ASSETS)]
        gap=(w*(self.data.opens.loc[day,list(ASSETS)]/self.data.closes.loc[prev,list(ASSETS)]-1)).sum()
        self.assertAlmostEqual(r.daily.loc[day,'gap_return'],gap,places=12)
        self.assertEqual(r.daily.loc[day,'reason'],'policy-switch')
        trades=r.trades[r.trades.date==day]
        self.assertAlmostEqual(trades.dollars.abs().sum()*.0006,r.daily.loc[day,'cost'],places=8)
        self.assertGreater(r.weights.loc[day,'GLD'],.2)
        self.assertAlmostEqual(r.daily.loc[day,'nav']/r.daily.loc[prev,'nav']-1,r.daily.loc[day,'return'],places=12)

    def test_forbids_overlapping_training_and_evaluation(self):
        schedule=self.schedule();schedule[2018]['train_end']='2018-12-31'
        router=ScheduledEngine(schedule,self.data.closes.index,{'base':self.base})
        with self.assertRaises(ValueError):run_backtest(self.data,start='2018-01-01',snapshots=self.base,engine=router)

    def test_tie_break_and_nonfinite_scores(self):
        table=pd.DataFrame({'sharpe':[1.,1.,np.nan],'gross_turnover_per_year':[2.,1.,0.]},index=['a','b','c'])
        self.assertEqual(select_policy(table),'b')
