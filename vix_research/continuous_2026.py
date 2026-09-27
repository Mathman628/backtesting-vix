"""Continuous unchanged core policies through the stored 2026 data endpoint."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from .core import StrategyConfig
from .data import load_data
from .backtest import ExecutionConfig, prepare_snapshots, run_backtest
from .analytics import performance, drawdown
from .benchmark_diagnostic import MonthlyAllocation, LABELS
from .literature_strategy import variants, research_snapshots, ResearchEngine


def main():
    root = Path(__file__).resolve().parent
    out = root / 'outputs/continuous_through_2026'
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    data = load_data(root / 'data/market')
    cfg = StrategyConfig()
    snapshots = prepare_snapshots(data, cfg)
    allocations = {'treasury_60_40': {'SPY': .6, 'IEF': .4},
                   'monthly_50_25_25': {'SPY': .5, 'IEF': .25, 'GLD': .25}}
    policies = variants()
    names = [*allocations, 'factor_t0_r1_v0', 'ensemble', 'original_revised']
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    rows = []
    for name, color in zip(names, ['#697783', '#b48729', '#126e82', '#39855c', '#bd5945']):
        config = policies.get(name, cfg)
        signals = research_snapshots(data, config, snapshots) if name in policies else snapshots
        engine = (MonthlyAllocation(allocations[name]) if name in allocations else
                  ResearchEngine(config) if name in policies else None)
        result = run_backtest(data, config, ExecutionConfig(), '2007-08-29', '2026-09-23',
                              name, snapshots=signals, engine=engine)
        d = result.daily
        saved = pd.read_csv(root / 'outputs/benchmark_full_period' / name / 'daily.csv',
                            parse_dates=['date']).set_index('date')
        assert d.loc[:'2023-12-29'].index.equals(saved.index)
        np.testing.assert_allclose(d.loc[saved.index].nav, saved.nav, rtol=1e-11, atol=1e-7)
        d.to_csv(out / f'{name}_daily.csv')
        for period, part in [('Full 2007-2026', d), ('Later 2024-2026', d.loc['2024-01-01':])]:
            rows.append(dict(model=name, period=period, start=str(part.index[0].date()),
                             end=str(part.index[-1].date()), **performance(part)))
        axes[0].plot(d.index, 100 * d.nav / 100000, label=LABELS[name], color=color, lw=1.3)
        axes[1].plot(d.index, drawdown(d['return']), color=color, lw=1)
        print(name, 'verified original-period NAV; extended through', d.index[-1].date(), flush=True)
    for ax in axes:
        ax.axvline(pd.Timestamp('2024-01-01'), color='#555555', ls='--', lw=1)
        ax.axvspan(pd.Timestamp('2024-01-01'), pd.Timestamp('2026-09-23'), alpha=.06, color='blue')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title('Unchanged policies, continuous accounts: 29 August 2007 - 23 September 2026')
    axes[0].set_ylabel('Wealth from $100')
    axes[0].legend(fontsize=8, loc='upper left', ncol=2)
    axes[1].set_ylabel('Drawdown')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel('Dashed line: start of later evaluation; no account reset or strategy reselection')
    fig.tight_layout()
    fig.savefig(out / 'continuous_2007_2026.png', dpi=180)
    fig.savefig(out / 'continuous_2007_2026.pdf')
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'comparison.csv', index=False)
    note = '''# Continuous history through September 2026

All five unchanged policies rerun from 29 August 2007 through 23 September 2026. Original-period NAV reproduces saved 2007-2023 results. Accounts, positions and policy state carry through the 2024 boundary; no stitched reset accounts. Costs remain 1 bp commission plus 5 bps slippage per side. No new selection or parameter fitting occurs in this diagnostic. Rolling signals and risk estimates continue updating from prior available observations as required by the fixed rules.

This is retrospective analysis of previously inspected historical data, not a claim of untouched holdout or prospective live returns. The data end on 23 September, not the end of 2026. Later-period metrics are calculated from the continuous accounts and can differ from earlier independently initialised later-period tests. Later drawdown resets its reference peak at the start of that measurement window; the full chart retains all historical peaks.

## Overleaf

Upload continuous_2007_2026.pdf and use it in an includegraphics command. Suggested caption: Continuous cost-adjusted performance of unchanged strategies, 29 August 2007--23 September 2026. The dashed line marks the start of the later evaluation period; portfolios are not reset. Later history was inspected during earlier research and is not an untouched holdout.
'''
    (out / 'README.md').write_text(note, encoding='utf-8')
    print(table[['model', 'period', 'cagr', 'sharpe', 'max_drawdown']].to_string(index=False))


if __name__ == '__main__':
    main()
