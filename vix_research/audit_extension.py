"""Read-only numerical audit of a completed research extension; writes audit.json."""
import json
import hashlib
import numpy as np
import pandas as pd
from .extension_study import ROOT, OUT, PERIODS, freeze
from .literature_study import verify, restore


def run():
    verify(ROOT/'outputs/literature',ROOT/'data/market');freeze()
    records=[]
    for period in PERIODS:
        results=restore(OUT/period);calendar=None
        for name,r in results.items():
            if calendar is None:calendar=r.daily.index
            assert r.daily.index.equals(calendar)
            assert (pd.to_datetime(r.daily.signal_date).to_numpy()<r.daily.index.to_numpy()).all()
            assert (pd.to_datetime(r.trades.signal_date)<pd.to_datetime(r.trades.date)).all()
            np.testing.assert_allclose(r.weights.sum(axis=1),1,atol=1e-12)
            assert r.weights.min().min()>=-1e-10
            err=(r.daily.filter(like='contribution_').sum(axis=1)-r.daily['return']).abs().max()
            assert err<1e-10
            np.testing.assert_allclose(r.daily.nav/100000,(1+r.daily['return']).cumprod(),rtol=1e-11)
            np.testing.assert_allclose(r.trades.cost.sum(),r.daily.cost.sum(),rtol=1e-10)
            records.append(dict(period=period,policy=name,sessions=len(r.daily),max_daily_accounting_error=float(err)))
    pairs={'ensemble':'ensemble','static_risk_control':'factor_t0_r1_v0','passive_50_25_25':'monthly_50_25_25',
           'previous_revised':'original_revised','treasury_60_40':'treasury_60_40'}
    parity={}
    for a,b in pairs.items():
        new=pd.read_csv(OUT/'original_horizon'/a/'daily.csv')
        old=pd.read_csv(ROOT/'outputs/benchmark_full_period'/b/'daily.csv')
        assert new.date.equals(old.date)
        error=float(abs(new.nav-old.nav).max());assert error<1e-6;parity[a]=error
    fits=pd.read_csv(OUT/'hmm_fits.csv',parse_dates=['execution_date','train_end'])
    assert (fits.train_end<fits.execution_date).all()
    p=pd.read_csv(OUT/'regime_probabilities.csv',parse_dates=['date','signal_date'])
    assert (p.signal_date<p.date).all()
    probabilities=p.filter(like='probability');assert probabilities.min().min()>=0
    np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-12)
    record=dict(status='passed',model_period_checks=records,original_nav_parity_max_dollar_error=parity,
                monthly_hmm_fits=len(fits),iteration_limit_hits=int(fits.hit_iteration_limit.sum()),
                report_source_sha256=hashlib.sha256((ROOT/'extension_report.py').read_bytes()).hexdigest(),
                tests='Five test methods passed before protocol freeze: future-data perturbation, accounting/strategy parity, filter/optimiser, mixture/label invariance, trend concentration/cash')
    (OUT/'audit.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='model_period_checks'},indent=2))


if __name__=='__main__':run()
