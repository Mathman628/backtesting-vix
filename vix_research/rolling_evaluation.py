"""Causal annual policy selection and a single continuous execution account.

Run: python -m vix_research.rolling_evaluation
Frozen literature source files and old outputs are not modified.
"""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from .core import StrategyConfig, SignalEngine, cap_portfolio_vol, estimated_vol
from .data import load_data, MarketData
from .backtest import run_backtest, prepare_snapshots, ExecutionConfig, BacktestResult
from .analytics import comparison_table, performance
from .literature_strategy import variants, research_snapshots, ResearchEngine
from .literature_study import verify, digest, utc, persist
from .literature_analysis import attribution, paired_sharpes
from .benchmark_diagnostic import MonthlyAllocation
from .literature_report import LABELS as STUDY_LABELS
from .report import save_json

ROOT=Path(__file__).resolve().parent
ALLOCATIONS={'treasury_60_40':{'SPY':.6,'IEF':.4},
             'monthly_50_25_25':{'SPY':.5,'IEF':.25,'GLD':.25}}
LABELS={**STUDY_LABELS,'treasury_60_40':'Monthly 60/40 Treasuries',
        'monthly_50_25_25':'Monthly passive 50/25/25','walk_forward':'Annual walk-forward selection'}


def configurations():
    return {**variants(),'original_revised':StrategyConfig(),
            'original_sma':replace(StrategyConfig(),equity_filter='sma'),
            'original_static_vol':StrategyConfig(),
            **{n:StrategyConfig() for n in ALLOCATIONS}}


class MonthlyRisk(MonthlyAllocation):
    def __init__(self):
        super().__init__(ALLOCATIONS['monthly_50_25_25'])

    def decide(self,snapshot,execution_date,actual_weights):
        decision=super().decide(snapshot,execution_date,actual_weights)
        w=cap_portfolio_vol(ALLOCATIONS['monthly_50_25_25'],snapshot.covariance,.10)
        targets={s:w.get(s,0)*.995 for s in self.targets}
        return replace(decision,targets=targets,target_vol=estimated_vol(targets,snapshot.covariance))


def engine_for(name):
    if name in ALLOCATIONS:return MonthlyAllocation(ALLOCATIONS[name])
    if name=='original_static_vol':return MonthlyRisk()
    cfg=configurations()[name]
    return ResearchEngine(cfg) if name in variants() else SignalEngine(cfg)


def signal_key(name):
    return variants()[name].signal if name in variants() else 'base'


def select_policy(table):
    finite=table[np.isfinite(table.sharpe)]
    if finite.empty:raise ValueError('No finite training Sharpe')
    return (finite.rename_axis('model').reset_index()
            .sort_values(['sharpe','gross_turnover_per_year','model'],ascending=[False,True,True])
            .iloc[0]['model'])


class ScheduledEngine:
    """Routes causal signals; the original simulator retains holdings and cash.

    Warm the selected policy's regime state using prior training-window sessions.
    At an actual policy change, rebalance at the next open from existing holdings.
    If selection repeats, preserve engine state and ordinary rebalance rules.
    """
    def __init__(self,schedule,dates,cache):
        self.schedule=schedule
        self.dates=dates
        self.cache=cache
        self.positions={d:i for i,d in enumerate(dates)}
        self.active=None
        self.engine=None
        self.history=[]

    def decide(self,snapshot,execution_date,actual_weights):
        day=pd.Timestamp(execution_date)
        item=self.schedule[day.year]
        if pd.Timestamp(item['train_end'])>=day:raise ValueError('Training includes evaluation date')
        name=item['selected']
        changed=name!=self.active
        idx=self.positions[day]
        if changed:
            self.engine=engine_for(name)
            regime=self.engine.regimes if isinstance(self.engine,ResearchEngine) else self.engine if isinstance(self.engine,SignalEngine) else None
            if regime is not None:
                for j in range(self.dates.searchsorted(pd.Timestamp(item['train_start'])),idx):
                    s=self.cache[signal_key(name)][j]
                    if s is not None:regime.decide(s,self.dates[j],{})
            self.active=name
        s=self.cache[signal_key(name)][idx]
        decision=self.engine.decide(s,day,actual_weights)
        if changed:decision=replace(decision,trade=True,reason='policy-switch')
        self.history.append(dict(date=day,selected=name,policy_changed=changed,train_end=item['train_end']))
        return decision


def slice_result(result,start,end):
    daily=result.daily.loc[start:end].copy()
    return BacktestResult(result.name,daily,result.weights.loc[start:end],result.trades,
                          result.signals.loc[start:end],result.config,result.execution,result.metadata)


def run():
    out=ROOT/'outputs/rolling_evaluation'
    data_dir=ROOT/'data/market'
    study=ROOT/'outputs/literature'
    verify(study,data_dir)
    data=load_data(data_dir)
    end=data.closes.index[-1]
    cfgs=configurations()
    selection_hash=digest(study/'selection.json')
    protocol=dict(created_utc=utc(),training='Five preceding full calendar years; independent cash-initialised candidate accounts; marked endpoint, no forced liquidation',
                  selection='Highest training excess-return Sharpe; ties lower gross turnover then alphabetical name; 21 fixed policies; no validation tuning',
                  first_evaluation='2013-01-01',last_evaluation=str(end.date()),
                  primary_summary='2013-01-01 through 2023-12-31; excludes 2024 onward',
                  extended_summary='2013 through available 2026 data; final year and quarter partial',
                  holdings='One continuous account, including overnight gap on old holdings before year-boundary trades; repeated selection keeps engine state; changed selection pays net reallocation costs',
                  state='On policy change warm regime state over preceding training window; force target rebalance at next open',
                  quarterly='Quarterly slices of annual-selection returns; no quarterly model reselection',
                  history_status='Previously inspected history and research-derived candidate family; retrospective chronological evaluation, not untouched evidence',
                  expectations=['Annual selection may fail to beat passive controls after switching costs.',
                                'Risk controls may reduce losses but miss recoveries.',
                                'Report 2013–2023 separately so 2024–2026 cannot dominate the primary conclusion.'],
                  configurations={n:c.to_dict() for n,c in cfgs.items()},allocations=ALLOCATIONS,
                  execution=ExecutionConfig().__dict__,source_sha256=digest(Path(__file__)),
                  adapter_sha256=digest(ROOT/'benchmark_diagnostic.py'),
                  data_sha256={f:digest(data_dir/f) for f in ('prices.csv','risk_free.csv')},
                  frozen_selection_sha256=selection_hash)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        old=json.loads((out/'protocol.json').read_text())
        canonical=json.loads(json.dumps(protocol))
        assert all(old[k]==v for k,v in canonical.items() if k!='created_utc'),'Version changed experiment before rerunning'
    else:save_json(out/'protocol.json',protocol)
    base=prepare_snapshots(data,StrategyConfig())
    cache={'base':base,**{key:research_snapshots(data,replace(next(iter(variants().values())),signal=key),base)
                        for key in ('none','absolute12','excess12','ensemble')}}
    records=[]
    schedule={}
    for year in range(2013,end.year+1):
        train_start=f'{year-5}-01-01'
        train_end=f'{year-1}-12-31'
        # Remove every future observation from the training data object itself.
        cut=data.closes.index.searchsorted(pd.Timestamp(train_end),side='right')
        past=MarketData(data.opens.iloc[:cut],data.closes.iloc[:cut],
                        data.rf_annual.loc[:train_end],data.metadata)
        results={n:run_backtest(past,c,start=train_start,end=train_end,name=n,
                               snapshots=cache[signal_key(n)][:cut],engine=engine_for(n)) for n,c in cfgs.items()}
        table=comparison_table(results)
        selected=select_policy(table)
        table.rename_axis('model').reset_index().assign(evaluation_year=year).to_csv(out/f'training_{year}.csv',index=False)
        record=dict(evaluation_year=year,train_start=train_start,train_end=train_end,
                    train_actual_end=str(past.closes.index[-1].date()),
                    evaluation_start=str(data.closes.loc[f'{year}-01-01':].index[0].date()),
                    selected=selected,training_sharpe=float(table.loc[selected,'sharpe']))
        assert pd.Timestamp(record['train_actual_end'])<pd.Timestamp(record['evaluation_start'])
        records.append(record)
        schedule[year]=record
        print(f'{year}: selected {selected}, training Sharpe {record["training_sharpe"]:.3f}',flush=True)
    pd.DataFrame(records).to_csv(out/'schedule.csv',index=False)
    save_json(out/'schedule_frozen.json',dict(created_utc=utc(),schedule=records,statement='Deterministic per-year decisions use only preceding training data; no evaluation-return selection'))
    router=ScheduledEngine(schedule,data.closes.index,cache)
    live=run_backtest(data,start='2013-01-01',end=str(end.date()),name='walk_forward',snapshots=base,engine=router)
    live.config={'selection_schedule':records,'protocol':'protocol.json'}
    live.metadata['walk_forward']='Continuous account; one initialisation, no fold splicing'
    results={'walk_forward':live}
    for n in ('treasury_60_40','monthly_50_25_25','factor_t0_r1_v0','ensemble','original_revised'):
        results[n]=run_backtest(data,cfgs[n],start='2013-01-01',end=str(end.date()),name=n,
                               snapshots=cache[signal_key(n)],engine=engine_for(n))
    persist(results,out/'continuous')
    pd.DataFrame(router.history).to_csv(out/'policy_history.csv',index=False)
    checks=[]
    for n,r in results.items():
        a,_,err=attribution(r)
        assert err<1e-10
        assert abs(a.linked_total_return_contribution.sum()-performance(r.daily)['total_return'])<1e-10
        a.to_csv(out/'continuous'/n/'attribution.csv')
        checks.append(dict(model=n,error=err))
    pd.DataFrame(checks).to_csv(out/'accounting_checks.csv',index=False)
    annual=[];quarterly=[];summaries=[];paired=[]
    for n,r in results.items():
        for year,d in r.daily.groupby(r.daily.index.year):
            annual.append(dict(model=n,year=year,partial=year==end.year and end.month<12,**performance(d)))
        for q,d in r.daily.groupby(r.daily.index.to_period('Q')):
            quarterly.append(dict(model=n,quarter=str(q),partial=q==end.to_period('Q') and end<q.end_time.normalize(),**performance(d)))
    for label,stop in [('primary_2013_2023','2023-12-31'),('extended_2013_2026',str(end.date()))]:
        group={n:slice_result(r,'2013-01-01',stop) for n,r in results.items()}
        table=comparison_table(group)
        summaries.append(table.rename_axis('model').reset_index().assign(period=label))
        draws=paired_sharpes(group)
        for control in ('treasury_60_40','monthly_50_25_25'):
            lo,hi=np.quantile(draws.walk_forward-draws[control],[.025,.975])
            paired.append(dict(period=label,control=control,sharpe_difference=table.loc['walk_forward','sharpe']-table.loc[control,'sharpe'],ci_low=lo,ci_high=hi))
    pd.DataFrame(annual).to_csv(out/'annual.csv',index=False)
    pd.DataFrame(quarterly).to_csv(out/'quarterly.csv',index=False)
    pd.concat(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    verify(study,data_dir)
    assert digest(study/'selection.json')==selection_hash
    save_json(out/'complete.json',dict(completed_utc=utc(),protocol_sha256=digest(out/'protocol.json'),folds=len(schedule),candidate_runs=len(schedule)*len(cfgs),accounting_checks=len(checks),selection_unchanged=True))
    from .rolling_report import build
    build(out)
    print(pd.concat(summaries)[['period','model','cagr','sharpe','max_drawdown']].round(4).to_string(index=False))


if __name__=='__main__':run()
