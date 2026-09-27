import unittest
from datetime import date
import numpy as np
from vix_research.benchmark_diagnostic import MonthlyAllocation
from vix_research.core import Snapshot, ASSETS


class MonthlyBenchmarkTests(unittest.TestCase):
    def test_month_boundary_without_signal_or_risk_overrides(self):
        engine = MonthlyAllocation({'SPY':.6,'IEF':.4})
        snap = Snapshot(date(2024,1,30),80,.99,False,False,{'IEF':False,'GLD':False},np.eye(4),1.)
        first = engine.decide(snap,date(2024,1,31),{})
        self.assertTrue(first.trade)
        self.assertAlmostEqual(first.targets['SPY'],.597)
        self.assertAlmostEqual(first.targets['IEF'],.398)
        feb = Snapshot(date(2024,1,31),80,.99,False,False,{},np.eye(4),1.)
        self.assertTrue(engine.decide(feb,date(2024,2,1),{}).trade)
        feb2 = Snapshot(date(2024,2,1),80,.99,False,False,{},np.eye(4),1.)
        hold = engine.decide(feb2,date(2024,2,2),{'SPY':.9})
        self.assertFalse(hold.trade)
        self.assertEqual(hold.targets,first.targets)

    def test_invalid_weights_and_same_session_snapshot(self):
        for weights in ({'SPY':1.1},{'SPY':1.1,'IEF':-.1},{'BAD':1}):
            with self.assertRaises(ValueError): MonthlyAllocation(weights)
        engine=MonthlyAllocation({'SPY':.6,'IEF':.4})
        snap=Snapshot(date(2024,1,1),20,.5,True,True,{},np.eye(4),.2)
        with self.assertRaises(ValueError):engine.decide(snap,date(2024,1,1),{})
