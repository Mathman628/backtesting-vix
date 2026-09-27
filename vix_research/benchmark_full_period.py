"""Continuous benchmark comparison on the exact original 2007–2023 calendar."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .benchmark_diagnostic import MonthlyAllocation, LABELS
from .backtest import run_backtest, prepare_snapshots, ExecutionConfig
from .core import StrategyConfig
from .data import load_data
from .analytics import comparison_table
from .literature_strategy import variants, research_snapshots, ResearchEngine
from .literature_analysis import attribution, paired_sharpes
from .literature_study import persist, verify, digest, utc
from .report import save_json


def run():
    root=Path(__file__).resolve().parent
    old=root/'outputs/development'
    study=root/'outputs/literature'
    out=root/'outputs/benchmark_full_period'
    data_dir=root/'data/market'
    verify(study,data_dir)
    original=pd.read_csv(old/'revised/daily.csv',parse_dates=['date']).set_index('date')
    start,end=str(original.index[0].date()),str(original.index[-1].date())
    cfg=StrategyConfig()
    execution=ExecutionConfig()
    baseline_manifest=json.loads((old/'revised/manifest.json').read_text())
    assert baseline_manifest['execution']==execution.__dict__
    selection_hash=digest(study/'selection.json')
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=utc(),start=start,end=end,sessions=len(original),
                  status='Retrospective continuous comparison on original development calendar; no reselection',
                  initial_cash=execution.initial_cash,execution=execution.__dict__,
                  portfolio_reset='Once at the start only; no 2019 reset or stitched subperiods',
                  expectations='Test whether the later-period asset-mix advantage persists over the original 2007–2023 sample. Risk control may be more valuable across the GFC and 2022. No presumed winner.',
                  source_sha256=digest(Path(__file__)),adapter_sha256=digest(root/'benchmark_diagnostic.py'),
                  input_sha256={f:digest(data_dir/f) for f in ('prices.csv','risk_free.csv')},
                  frozen_selection_sha256=selection_hash)
    path=out/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        assert all(previous[k]==v for k,v in protocol.items() if k!='created_utc'), 'Version changed diagnostic before rerun'
    else: save_json(path,protocol)
    print(f'Continuous run: {start} to {end}, {len(original)} sessions',flush=True)
    data=load_data(data_dir)
    snapshots=prepare_snapshots(data,cfg)
    results={}
    for name,weights in [('treasury_60_40',{'SPY':.6,'IEF':.4}),('monthly_50_25_25',{'SPY':.5,'IEF':.25,'GLD':.25})]:
        results[name]=run_backtest(data,cfg,execution,start,end,name,snapshots=snapshots,engine=MonthlyAllocation(weights))
        results[name].config.update(diagnostic_allocation=weights,diagnostic_rebalance='monthly',diagnostic_overrides='No trend, VIX or risk cap')
    for name in ('factor_t0_r0_v0','factor_t0_r1_v0','ensemble'):
        config=variants()[name]
        signals=research_snapshots(data,config,snapshots)
        results[name]=run_backtest(data,config,execution,start,end,name,snapshots=signals,engine=ResearchEngine(config))
    results['original_revised']=run_backtest(data,cfg,execution,start,end,'original_revised',snapshots=snapshots)
    results['original_static_vol']=run_backtest(data,cfg,execution,start,end,'original_static_vol',baseline='static_vol',snapshots=snapshots)
    labels={**LABELS,'original_static_vol':'Original monthly static + risk control'}
    parity=[]
    for new_name,old_name in [('original_revised','revised'),('monthly_50_25_25','static_50_25_25'),('original_static_vol','static_vol_target')]:
        earlier=pd.read_csv(old/old_name/'daily.csv',parse_dates=['date']).set_index('date')
        fresh=results[new_name].daily
        assert earlier.index.equals(fresh.index)
        np.testing.assert_allclose(earlier.nav,fresh.nav,rtol=1e-12,atol=1e-8)
        parity.append(new_name)
    persist(results,out)
    table=comparison_table(results)
    checks=[]
    for name,r in results.items():
        assert r.daily.index.equals(original.index)
        summary,corr,error=attribution(r)
        assert error<1e-10
        assert abs(summary.linked_total_return_contribution.sum()-table.loc[name,'total_return'])<1e-10
        summary.to_csv(out/name/'attribution.csv')
        checks.append(dict(model=name,daily_reconciliation_error=error))
    pd.DataFrame(checks).to_csv(out/'accounting_checks.csv',index=False)
    draws=paired_sharpes(results)
    comparisons=[]
    for name in results:
        for control in ('treasury_60_40','monthly_50_25_25'):
            if name==control:continue
            lo,hi=np.quantile(draws[name]-draws[control],[.025,.975])
            comparisons.append(dict(model=name,control=control,sharpe_difference=table.loc[name,'sharpe']-table.loc[control,'sharpe'],ci_low=lo,ci_high=hi))
    paired=pd.DataFrame(comparisons)
    paired.to_csv(out/'paired_comparisons.csv',index=False)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=('Continuous wealth, initial 100','Drawdown (%)'))
    colors=['#333333','#bd851c','#9b7aa3','#376a9c','#008b8b','#b64d40','#809638']
    for (name,r),color in zip(results.items(),colors):
        growth=(1+r.daily['return']).cumprod()
        fig.add_trace(go.Scatter(x=growth.index,y=growth*100,name=labels[name],legendgroup=name,line_color=color),row=1,col=1)
        fig.add_trace(go.Scatter(x=growth.index,y=(growth/growth.cummax().clip(lower=1)-1)*100,name=labels[name],legendgroup=name,line_color=color,showlegend=False),row=2,col=1)
    fig.update_layout(template='plotly_white',height=850,legend=dict(orientation='h',y=-.1),margin=dict(b=160))
    shown=table[['cagr','annual_vol','sharpe','max_drawdown','gross_turnover_per_year']].copy()
    shown.index=shown.index.map(labels)
    for col in ('cagr','annual_vol','max_drawdown'):shown[col]=shown[col].map(lambda v:f'{v:.2%}')
    for col in ('sharpe','gross_turnover_per_year'):shown[col]=shown[col].map(lambda v:f'{v:.3f}')
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Original-period 60/40 comparison</title>
    <style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;color:#23313d}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}p{{line-height:1.6}}</style></head><body>
    <h1>Same-period comparison: {start} to {end}</h1>
    <p>{len(original):,} sessions, one initial $100,000 balance, no portfolio reset in 2019. Exact original development calendar. Same adjusted prices, lagged cash interest, 0.5% target cash reserve and 1 bp commission + 5 bps slippage per side. No terminal liquidation. 60/40 uses SPY/IEF, not aggregate bonds or global assets.</p>
    <p>Retrospective diagnostic, not an untouched holdout or revised selection. Monthly passive portfolios share a schedule; active policies retain their own execution rules. Original revised, monthly passive and monthly risk-controlled NAV series reproduced within numerical tolerance.</p>
    <p><a href="ANALYSIS.md">Findings</a> | <a href="comparison.csv">Metrics CSV</a> | <a href="protocol.json">Protocol</a></p>
    {shown.to_html()}{fig.to_html(full_html=False,include_plotlyjs=True)}
    <h2>Paired Sharpe differences</h2><p>2,000 paired circular 20-session block resamples; 95% pointwise intervals, no multiple-comparison adjustment. These intervals do not test drawdown significance.</p>{paired.round(4).to_html(index=False)}</body></html>'''
    (out/'report.html').write_text(text,encoding='utf-8')
    verify(study,data_dir)
    assert digest(study/'selection.json')==selection_hash
    save_json(out/'complete.json',dict(completed_utc=utc(),protocol_sha256=digest(path),original_nav_parity=parity,accounting_checks=len(checks),frozen_selection_unchanged=True))
    print(table[['cagr','annual_vol','sharpe','max_drawdown']].round(5).to_string())
    print(paired.round(4).to_string(index=False))


if __name__=='__main__':run()
