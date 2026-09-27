"""Post-study 60/40 Treasury comparison; never changes the frozen selection.

Run: python -m vix_research.benchmark_diagnostic
"""
from pathlib import Path
import json
from html import escape
import numpy as np
import pandas as pd
from .core import ASSETS, StrategyConfig, Decision, estimated_vol
from .data import load_data
from .backtest import run_backtest, prepare_snapshots, ExecutionConfig
from .analytics import comparison_table
from .literature_analysis import paired_sharpes, attribution
from .literature_study import PERIODS, verify, restore, persist, digest, utc
from .report import save_json

LABELS = {
    'treasury_60_40': 'Monthly 60% SPY / 40% IEF',
    'monthly_50_25_25': 'Monthly 50% SPY / 25% IEF / 25% GLD',
    'factor_t0_r0_v0': 'Fixed 50/25/25 with drift bands',
    'factor_t0_r1_v0': 'Selected static + risk control',
    'ensemble': 'Multi-horizon ensemble',
    'original_revised': 'Previous revised VIX strategy',
}


class MonthlyAllocation:
    """Calendar rebalancing only: signals and volatility never gate exposure."""
    def __init__(self, weights, cash_buffer=.005):
        if set(weights)-set(ASSETS) or any(not np.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError('Invalid asset weights')
        if not np.isclose(sum(weights.values()), 1) or not 0 <= cash_buffer < 1:
            raise ValueError('Require fully allocated unlevered weights and valid reserve')
        self.targets = {s: weights.get(s, 0.)*(1-cash_buffer) for s in ASSETS}
        self.month = None

    def decide(self, snapshot, execution_date, actual_weights):
        day = pd.Timestamp(execution_date).date()
        if snapshot.date >= day:
            raise ValueError('Snapshot must precede execution')
        month = (day.year, day.month)
        trade = month != self.month
        self.month = month
        return Decision(snapshot.date, day, -1, self.targets.copy(), trade,
                        'baseline-rebalance' if trade else 'hold',
                        estimated_vol(self.targets, snapshot.covariance),
                        estimated_vol(actual_weights, snapshot.covariance))


def run():
    root = Path(__file__).resolve().parent
    study = root/'outputs/literature'
    data_folder = root/'data/market'
    out = root/'outputs/benchmark_diagnostic'
    verify(study, data_folder)
    selection_hash = digest(study/'selection.json')
    allocations = {'treasury_60_40': {'SPY': .6, 'IEF': .4},
                   'monthly_50_25_25': {'SPY': .5, 'IEF': .25, 'GLD': .25}}
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(created_utc=utc(), status='Post-study diagnostic; all periods previously viewed',
                    periods=PERIODS, allocations=allocations,
                    rebalance='First evaluation session then first session of each calendar month',
                    cash_reserve=.005, execution=ExecutionConfig().__dict__,
                    selected_policy_unchanged=selection_hash,
                    expectations=['Passive 50/25/25 may outperform 60/40 in the later period because of gold participation; not a causal gold-only test.',
                                  'VIX/trend protection may reduce some losses but lag simpler portfolios during recoveries.',
                                  'No assumption that any model beats 60/40 on Sharpe across all periods.'],
                    interpretation='SPY/IEF is US equity/intermediate Treasury 60/40, not aggregate bonds or a global institutional benchmark. No fund-manager comparison.',
                    input_sha256={f:digest(data_folder/f) for f in ('prices.csv','risk_free.csv')},
                    implementation_sha256=digest(Path(__file__)))
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        for key in protocol:
            if key != 'created_utc' and old[key] != json.loads(json.dumps(protocol[key])):
                raise ValueError(f'Diagnostic protocol changed: {key}; archive/version before rerunning')
    else:
        save_json(protocol_path, protocol)
    print('Diagnostic expectations saved before new benchmark simulations.', flush=True)
    data = load_data(data_folder)
    cfg = StrategyConfig()
    snapshots = prepare_snapshots(data, cfg)
    records, intervals, checks, period_results = [], [], [], {}
    for split,(start,end) in PERIODS.items():
        print(f'Evaluating {split}', flush=True)
        prior = restore(study/split)
        results = {n:prior[n] for n in LABELS if n in prior}
        for name, weights in allocations.items():
            r = run_backtest(data, cfg, start=start, end=end, name=name,
                             snapshots=snapshots, engine=MonthlyAllocation(weights))
            r.config.update(diagnostic_allocation=weights, diagnostic_rebalance='monthly',
                            diagnostic_overrides='SignalEngine filters, VIX and risk caps are not applied')
            results[name] = r
        results = {n:results[n] for n in LABELS}
        # Exact adapter parity with the independently existing static baseline.
        reference = run_backtest(data, cfg, start=start, end=end, baseline='static', snapshots=snapshots)
        np.testing.assert_allclose(results['monthly_50_25_25'].daily.nav, reference.daily.nav, rtol=1e-12)
        persist(results, out/split)
        table = comparison_table(results)
        records.append(table.rename_axis('model').reset_index().assign(period=split))
        draws = paired_sharpes(results)
        for name, r in results.items():
            summary, corr, error = attribution(r)
            summary.to_csv(out/split/name/'attribution.csv')
            assert error < 1e-10
            assert abs(summary.linked_total_return_contribution.sum()-table.loc[name,'total_return']) < 1e-10
            assert r.daily.index.equals(results['treasury_60_40'].daily.index)
            checks.append(dict(period=split, model=name, attribution_error=error))
            for control in ('treasury_60_40', 'monthly_50_25_25'):
                if name == control:
                    continue
                lo,hi = np.quantile(draws[name]-draws[control], [.025,.975])
                intervals.append(dict(period=split, model=name, control=control,
                                      sharpe_difference=table.loc[name,'sharpe']-table.loc[control,'sharpe'],
                                      ci_low=lo, ci_high=hi))
        period_results[split] = results
    combined = pd.concat(records, ignore_index=True)
    paired = pd.DataFrame(intervals)
    combined.to_csv(out/'comparison.csv', index=False)
    paired.to_csv(out/'paired_comparisons.csv', index=False)
    pd.DataFrame(checks).to_csv(out/'accounting_checks.csv', index=False)
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Historical test wealth, initial 100', 'Drawdown (%)'))
    colors = ['#333333','#ba7915','#a38ab6','#456d96','#008b8b','#bb4b43']
    for (name,r),color in zip(period_results['test'].items(), colors):
        wealth = (1+r.daily['return']).cumprod()
        fig.add_trace(go.Scatter(x=wealth.index,y=wealth*100,name=LABELS[name],line_color=color,legendgroup=name),row=1,col=1)
        fig.add_trace(go.Scatter(x=wealth.index,y=(wealth/wealth.cummax().clip(lower=1)-1)*100,
                                name=LABELS[name],line_color=color,legendgroup=name,showlegend=False),row=2,col=1)
    fig.update_layout(height=850,template='plotly_white',legend=dict(orientation='h',y=-.1),margin=dict(b=140))
    notes = ('Post-study diagnostic; historical periods were previously viewed. No reselection. '
             'Monthly targets reserve 0.5% cash, so initial invested weights are 59.7% SPY and 39.8% IEF. '
             'Each period starts independently with $100,000; no stitching. '
             'Adjusted opens/closes, lagged cash yield, 1 bp commission + 5 bps slippage per side. '
             'No terminal liquidation. Confidence intervals: 2,000 paired circular 20-session block resamples, '
             '95% pointwise, not multiple-testing-adjusted. Monthly 50/25/25 and 60/40 share an execution calendar; '
             'frozen research policies retain their original calendars. These are not GS/JPM/PIMCO fund returns.')
    html = ['<!doctype html><html><head><meta charset="utf-8"><title>60/40 benchmark diagnostic</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;color:#23313d}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}p{line-height:1.6}</style></head><body>',
            '<h1>Does the strategy beat 60/40?</h1><p>'+escape(notes)+'</p>',
            '<p><a href="ANALYSIS.md">Interpretation and expectations versus results</a> | <a href="comparison.csv">Metrics CSV</a> | <a href="paired_comparisons.csv">Paired comparisons</a> | <a href="protocol.json">Dated protocol</a></p>']
    md = ['# 60/40 Treasury diagnostic results', '', notes, '']
    for split in PERIODS:
        table = combined[combined.period==split].set_index('model')[['cagr','annual_vol','sharpe','max_drawdown','gross_turnover_per_year']]
        table.index = table.index.map(LABELS)
        shown = table.copy()
        for col in ('cagr','annual_vol','max_drawdown'):
            shown[col] = shown[col].map(lambda v:f'{v:.2%}')
        for col in ('sharpe','gross_turnover_per_year'):
            shown[col] = shown[col].map(lambda v:f'{v:.3f}')
        html += [f'<h2>{split.title()}</h2>',shown.to_html()]
        md += [f'## {split.title()}', '', shown.to_csv(), '']
    html += [fig.to_html(full_html=False,include_plotlyjs=True),'<h2>Paired Sharpe differences</h2>',paired.round(4).to_html(index=False),'</body></html>']
    (out/'report.html').write_text('\n'.join(html),encoding='utf-8')
    (out/'RESULTS.md').write_text('\n'.join(md),encoding='utf-8')
    verify(study, data_folder)
    assert digest(study/'selection.json') == selection_hash
    save_json(out/'complete.json',dict(completed_utc=utc(),protocol_sha256=digest(protocol_path),
                                     frozen_selection_unchanged=True,accounting_checks=len(checks),
                                     monthly_adapter_parity='Passed all three periods'))
    print(combined[['period','model','cagr','sharpe','max_drawdown']].round(4).to_string(index=False))
    print(f'Open {out / "report.html"}')


if __name__ == '__main__':
    run()
