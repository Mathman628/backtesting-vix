"""Independent extended-universe simulator; original frozen engine is unchanged."""
from dataclasses import replace
import numpy as np
import pandas as pd
from .backtest import BacktestResult, ExecutionConfig, rebalance_values
from .core import ASSETS, StrategyConfig, SignalEngine, Decision, cap_portfolio_vol, estimated_vol
from .literature_strategy import ResearchConfig

UNIVERSE=(*ASSETS,'EFA','VNQ','GSG')
FIVE=('SPY','EFA','IEF','VNQ','GSG')


def simulate(opens,closes,rates,snapshots,engine,start,end,name):
    dates=closes.loc[start:end].index
    if dates.empty:raise ValueError('Empty evaluation')
    if not np.isfinite(opens.loc[dates,list(UNIVERSE)]).all().all() or not np.isfinite(closes.loc[dates,list(UNIVERSE)]).all().all():
        raise ValueError('Incomplete execution data')
    values=np.zeros(len(UNIVERSE));cash=previous=100000.
    records=[];holdings=[];trades=[];signals=[]
    all_dates=closes.index
    for day in dates:
        i=all_dates.get_loc(day);snap=snapshots[i]
        if snap is None or pd.Timestamp(snap.date)!=all_dates[i-1]:raise ValueError('Noncausal snapshot')
        op=opens.loc[day,list(UNIVERSE)].to_numpy(float)
        cl=closes.loc[day,list(UNIVERSE)].to_numpy(float)
        prior=closes.iloc[i-1].loc[list(UNIVERSE)].to_numpy(float)
        gap=values*(op/prior-1);values+=gap
        interest=cash*rates.loc[day];cash+=interest
        pre=float(values.sum()+cash)
        actual=dict(zip(UNIVERSE,values/pre))
        d=engine.decide(snap,day,actual)
        target=np.array([d.targets.get(s,0.) for s in UNIVERSE])
        if not np.isfinite(target).all() or min(target)<-1e-10 or sum(target)>1+1e-9:raise ValueError('Invalid targets')
        cost=turnover=0.
        if d.trade:
            values,cash,delta,cost=rebalance_values(values,cash,target,.0006)
            turnover=abs(delta).sum()/pre
            for j,s in enumerate(UNIVERSE):
                if abs(delta[j])>1e-7:trades.append(dict(date=day,signal_date=snap.date,symbol=s,dollars=delta[j],adjusted_fill_price=op[j],adjusted_units=delta[j]/op[j],cost=abs(delta[j])*.0006,reason=d.reason))
        session=values*(cl/op-1);values+=session
        nav=float(values.sum()+cash)
        row=dict(date=day,nav=nav,return_=nav/previous-1,risk_free_return=rates.loc[day],cost=cost,turnover=turnover,traded=d.trade,reason=d.reason,signal_date=snap.date,regime=d.regime,gap_return=gap.sum()/previous)
        row['return']=row.pop('return_')
        row.update({f'contribution_{s}':(gap[j]+session[j])/previous for j,s in enumerate(UNIVERSE)})
        row.update(contribution_cash=interest/previous,contribution_cost=-cost/previous)
        if abs(sum(v for k,v in row.items() if k.startswith('contribution_'))-row['return'])>1e-10:raise ArithmeticError('Attribution mismatch')
        records.append(row);holdings.append(dict(date=day,**dict(zip(UNIVERSE,values/nav)),CASH=cash/nav))
        signals.append(dict(date=day,signal_date=snap.date,**{f'target_{s}':target[j] for j,s in enumerate(UNIVERSE)},reason=d.reason))
        previous=nav
    return BacktestResult(name,pd.DataFrame(records).set_index('date'),pd.DataFrame(holdings).set_index('date'),
                          pd.DataFrame(trades,columns=['date','signal_date','symbol','dollars','adjusted_fill_price','adjusted_units','cost','reason']),
                          pd.DataFrame(signals).set_index('date'),{'policy':name,'source':'extension_study.py protocol'},ExecutionConfig().__dict__,
                          {'actual_start':str(dates[0].date()),'actual_end':str(dates[-1].date()),'execution':'Prior-close signals, next adjusted open, old holdings earn gap, six bps per side, no terminal liquidation'})


class FiveAsset:
    def __init__(self,closes,mode='passive'):
        self.closes=closes;self.mode=mode;self.month=None
    def decide(self,snapshot,execution_date,actual_weights):
        day=pd.Timestamp(execution_date);i=self.closes.index.get_loc(day)
        month=(day.year,day.month);trade=month!=self.month;self.month=month
        h=self.closes.iloc[i-210:i].loc[:,list(FIVE)]
        if len(h)!=210 or h.isna().any().any():raise ValueError('Incomplete 210-session trend history')
        eligible=list(FIVE) if self.mode=='passive' else list(h.columns[h.iloc[-1]>h.mean()])
        w={s:0. for s in UNIVERSE}
        for s in eligible:w[s]=.995/(len(eligible) if self.mode=='survivors' else 5)
        return Decision(snapshot.date,day.date(),-1,w,trade,'monthly' if trade else 'hold',0,0)


class GradualVix:
    """Same regime/filter/risk/execution rules; only exposure multiplier is smoothed."""
    def __init__(self,smooth=False):
        self.smooth=smooth;self.multiplier=None;self.month=None;self.week=None
        self.cfg=ResearchConfig(signal='absolute12',vix_overlay=True)
        self.regimes=SignalEngine(self.cfg)
        self.history=[]
    def decide(self,snapshot,execution_date,actual_weights):
        day=pd.Timestamp(execution_date);initial=self.month is None
        monthly=(day.year,day.month)!=self.month;weekly=tuple(day.isocalendar()[:2])!=self.week
        self.month=(day.year,day.month);self.week=tuple(day.isocalendar()[:2])
        state=self.regimes.decide(snapshot,day,actual_weights)
        raw=(1.,.75,.5,0.)[state.regime]
        if self.multiplier is None or not self.smooth or state.regime==3:self.multiplier=raw
        else:self.multiplier=.8*self.multiplier+.2*raw
        w={s:b*snapshot.scores[s]*self.multiplier for s,b in zip(ASSETS[:3],[.5,.25,.25])}
        w['BIL']=1-sum(w.values());w=cap_portfolio_vol(w,snapshot.covariance,.1)
        targets={s:w.get(s,0)*.995 for s in ASSETS}
        cv=estimated_vol(actual_weights,snapshot.covariance);tv=estimated_vol(targets,snapshot.covariance)
        exit_needed=any(actual_weights.get(s,0)>.005 and targets[s]<=1e-12 for s in ASSETS[:3])
        breach=cv>.12 and tv<cv
        drift=max(abs(targets[s]-actual_weights.get(s,0)) for s in ASSETS)
        reason='initial' if initial else 'guard-exit' if exit_needed else 'risk-cut' if breach else 'monthly' if monthly else 'weekly-drift' if weekly and drift>.05 else 'hold'
        trade=reason!='hold' and (exit_needed or breach or drift>=.005)
        self.history.append(dict(date=day,raw_multiplier=raw,applied_multiplier=self.multiplier,regime=state.regime))
        return Decision(snapshot.date,day.date(),state.regime,targets,trade,reason,tv,cv)
