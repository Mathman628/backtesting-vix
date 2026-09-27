"""Run with: python -m vix_research.extension_study"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .data import load_data
from .core import StrategyConfig, SignalEngine
from .backtest import prepare_snapshots
from .benchmark_diagnostic import MonthlyAllocation
from .literature_strategy import ResearchConfig, ResearchEngine, research_snapshots
from .literature_study import persist, verify, utc
from .analytics import comparison_table
from .literature_analysis import paired_sharpes
from .extension_engine import simulate, FiveAsset, GradualVix
from .extension_models import forecasts, OptimisedPolicy

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/research_extension'
PERIODS={'original_horizon':('2007-08-29','2023-12-29'),
         'common_horizon':('2011-01-03','2023-12-29'),
         'later_horizon':('2024-01-02','2026-09-23')}
PAIRS=[('trend_survivors','passive_five'),('trend_slots','passive_five'),('smooth_vix','hard_vix'),
       ('hmm','unconditional'),('tracked_hmm','hmm'),('tracked_hmm','tracked_no_penalty')]


def inputs():
    data=load_data(ROOT/'data/market')
    extra=pd.read_csv(ROOT/'data/research_extension/prices.csv',parse_dates=['date'])
    opens=data.opens.join(extra.pivot(index='date',columns='symbol',values='adjusted_open'))
    closes=data.closes.join(extra.pivot(index='date',columns='symbol',values='adjusted_close'))
    return data,opens,closes


def freeze():
    paths=['extension_engine.py','extension_models.py','extension_study.py','EXTENSION_REGISTER.md','tests/test_extension.py',
           'data/market/prices.csv','data/market/risk_free.csv','data/research_extension/prices.csv']
    protocol={'created_utc':utc(),'status':'Post-study research on previously inspected history; no untouched test set',
              'periods':PERIODS,'paired_comparisons':PAIRS,'window':756,'states':3,'seed':1729,
              'source_and_data_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('created_utc');compare=dict(protocol);compare.pop('created_utc')
        if old!=json.loads(json.dumps(compare)):raise ValueError('Frozen extension changed; use a new study version')
    else:path.write_text(json.dumps(protocol,indent=2))


def run():
    verify(ROOT/'outputs/literature',ROOT/'data/market');freeze()
    print('Protocol and hypotheses frozen before evaluation.',flush=True)
    data,opens,closes=inputs();rates=data.risk_free_returns()
    base=prepare_snapshots(data,StrategyConfig())
    snapshots={s:research_snapshots(data,ResearchConfig(signal=s),base) for s in ['none','absolute12','ensemble']}
    cache,diagnostics=forecasts(closes,start='2011-01-03')
    diagnostics.to_csv(OUT/'hmm_fits.csv',index=False)
    pd.DataFrame([dict(date=day,signal_date=item['signal_date'],**{f'template_probability_{i}':p for i,p in enumerate(item['probabilities'])}) for day,item in cache.items()]).to_csv(OUT/'regime_probabilities.csv',index=False)
    all_hypotheses=[]
    for period,(start,end) in PERIODS.items():
        factories={
          'treasury_60_40':(lambda:MonthlyAllocation({'SPY':.6,'IEF':.4}),'none'),
          'passive_50_25_25':(lambda:MonthlyAllocation({'SPY':.5,'IEF':.25,'GLD':.25}),'none'),
          'static_risk_control':(lambda:ResearchEngine(ResearchConfig(signal='none')),'none'),
          'ensemble':(lambda:ResearchEngine(ResearchConfig(signal='ensemble')),'ensemble'),
          'previous_revised':(lambda:SignalEngine(StrategyConfig()),'base'),
          'passive_five':(lambda:FiveAsset(closes,'passive'),'none'),
          'trend_survivors':(lambda:FiveAsset(closes,'survivors'),'none'),
          'trend_slots':(lambda:FiveAsset(closes,'slots'),'none'),
          'hard_vix':(lambda:GradualVix(False),'absolute12'),
          'smooth_vix':(lambda:GradualVix(True),'absolute12')}
        if period!='original_horizon':
            factories.update({'unconditional':(lambda:OptimisedPolicy(cache,'unconditional'),'none'),
              'hmm':(lambda:OptimisedPolicy(cache,'hmm'),'none'),
              'tracked_hmm':(lambda:OptimisedPolicy(cache,'tracked'),'none'),
              'tracked_no_penalty':(lambda:OptimisedPolicy(cache,'tracked',0.),'none')})
        results={}
        for name,(factory,signal) in factories.items():
            print(period,name,flush=True);engine=factory()
            results[name]=simulate(opens,closes,rates,base if signal=='base' else snapshots[signal],engine,start,end,name)
            if name in ('hard_vix','smooth_vix'):
                dest=OUT/period;dest.mkdir(exist_ok=True)
                pd.DataFrame(engine.history).to_csv(dest/(name+'_multipliers.csv'),index=False)
        persist(results,OUT/period)
        table=comparison_table(results);draws=paired_sharpes(results)
        for a,b in PAIRS:
            if a not in results:continue
            low,high=np.quantile(draws[a]-draws[b],[.025,.975])
            all_hypotheses.append(dict(period=period,treatment=a,control=b,sharpe_difference=table.loc[a,'sharpe']-table.loc[b,'sharpe'],
               pointwise_95_low=low,pointwise_95_high=high,drawdown_improvement=table.loc[a,'max_drawdown']-table.loc[b,'max_drawdown'],
               turnover_change=table.loc[a,'gross_turnover_per_year']-table.loc[b,'gross_turnover_per_year']))
        # Wealth-linked exact contribution in dollars / initial capital, not sum of daily %.
        attribution=[]
        for name,result in results.items():
            prior=result.daily.nav.shift(1).fillna(100000)/100000
            linked=result.daily.filter(like='contribution_').mul(prior,axis=0).sum()
            if not np.isclose(linked.sum(),result.daily.nav.iloc[-1]/100000-1):raise ArithmeticError('Linked contribution mismatch')
            attribution.append(dict(policy=name,**linked.to_dict()))
        pd.DataFrame(attribution).to_csv(OUT/period/'attribution.csv',index=False)
    pd.DataFrame(all_hypotheses).to_csv(OUT/'hypotheses.csv',index=False)
    verify(ROOT/'outputs/literature',ROOT/'data/market')
    from .extension_report import build
    build()


if __name__=='__main__':run()
