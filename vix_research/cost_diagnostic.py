"""Reproduce saved core results, then rerun without explicit trading costs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import load_data
from .core import StrategyConfig
from .backtest import ExecutionConfig, prepare_snapshots, run_backtest
from .analytics import performance
from .benchmark_diagnostic import MonthlyAllocation
from .literature_strategy import variants, research_snapshots, ResearchEngine


def main():
    root = Path(__file__).resolve().parent
    out = root / 'outputs/zero_cost_diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(root / 'data/market')
    cfg = StrategyConfig()
    base = prepare_snapshots(data, cfg)
    policies = variants()
    allocations = {'treasury_60_40': {'SPY': .6, 'IEF': .4},
                   'monthly_50_25_25': {'SPY': .5, 'IEF': .25, 'GLD': .25}}
    names = [*allocations, 'factor_t0_r1_v0', 'ensemble', 'original_revised']
    signals = {n: research_snapshots(data, policies[n], base)
               for n in names if n in policies}
    periods = [('original', '2007-08-29', '2023-12-29'),
               ('later', '2024-01-02', '2026-09-23')]
    rows = []
    for period, start, end in periods:
        for name in names:
            for cost_name, execution in [('net', ExecutionConfig()),
                                         ('zero', ExecutionConfig(commission_bps=0, slippage_bps=0))]:
                config = policies.get(name, cfg)
                engine = (MonthlyAllocation(allocations[name]) if name in allocations else
                          ResearchEngine(config) if name in policies else None)
                r = run_backtest(data, config, execution, start, end, name,
                                 snapshots=signals.get(name, base), engine=engine)
                metrics = performance(r.daily)
                if cost_name == 'net':
                    folder = ('benchmark_full_period' if period == 'original' else
                              'benchmark_diagnostic/test' if name in allocations else 'literature/test')
                    saved = pd.read_csv(root / 'outputs' / folder / name / 'daily.csv',
                                        parse_dates=['date']).set_index('date')
                    assert r.daily.index.equals(saved.index)
                    np.testing.assert_allclose(r.daily.nav, saved.nav, rtol=1e-11, atol=1e-7)
                else:
                    assert metrics['total_cost_dollars'] == 0
                row = dict(period=period, start=str(r.daily.index[0].date()),
                           end=str(r.daily.index[-1].date()), model=name, costs=cost_name, **metrics)
                rows.append(row)
                r.daily.to_csv(out / f'{period}_{name}_{cost_name}_daily.csv')
                print(f'{period} {name} {cost_name}: CAGR={metrics["cagr"]:.6%}, Sharpe={metrics["sharpe"]:.6f}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    (out / 'methodology.json').write_text(json.dumps(dict(
        method='Full engine reruns, not cost add-backs; net NAV reproduced against saved results.',
        net_commission_bps=1, net_slippage_bps=5, zero_commission_bps=0, zero_slippage_bps=0,
        unchanged='Data, policies, dates, cash reserve, signal lag, execution timing and risk controls.',
        limitation='ETF expenses embedded in prices remain; taxes and financing were not newly removed. Previously inspected historical periods, not new holdout.'
    ), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
