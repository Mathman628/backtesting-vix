# VIX regime and multi-asset backtesting

Research code accompanying algorithmic-trading project. It compares VIX regime allocation, ETF momentum and volatility control with passive portfolios, annual strategy selection, broader asset-class trend and an experimental hidden Markov model (HMM).

## Strategies and evidence

- Original revised VIX strategy: risk-on/caution/fear/panic allocations across SPY, IEF, GLD and BIL, with trend filters and volatility control.
- Static + risk control: 50/25/25 SPY/IEF/GLD base; risk is reduced into BIL when the estimated volatility target is exceeded. No momentum or VIX signal.
- Ensemble: equal votes from 21/63/252-session returns exceeding cash; unused allocation enters BIL.
- The literature study includes eight trend/risk/VIX combinations, eight further variants and three earlier policies: **19 candidates**. Annual walk-forward adds monthly passive 60/40 SPY/IEF and 50/25/25 SPY/IEF/GLD: **21 candidates**.
- HMM and five-asset extensions are separate experiments, not candidates in that annual selector.

`results/` contains saved summary metrics from the reported runs. The earlier continuous comparison covers 29 August 2007--29 December 2023; the extended continuous comparison ends 23 September 2026. Split later-period accounts and the continuation of an existing account are different experiments. Later history was previously inspected; it is not an untouched holdout. These are simulated results, not a live trading record.

## Install

Run from the repository root. Python 3.11+ is suggested; development used Python 3.14.

```sh
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r vix_research/requirements.txt
python -m vix_research --help
```

## Offline demonstration and tests

```sh
python -m vix_research demo --sessions 1600
python -m unittest discover -s vix_research/tests -p 'test_research.py'
python -m unittest discover -s vix_research/tests -p 'test_literature.py'
```

The demo uses **synthetic data**, not the historical results in the report. It writes an HTML report under `vix_research/outputs/demo/`.

## Historical study, in dependency order

```sh
python -m vix_research fetch --start 2006-01-01 --end 2026-09-24
python -m vix_research compare --end 2023-12-31 --output vix_research/outputs/development
python -m vix_research literature --stage all
python -m vix_research.benchmark_diagnostic
python -m vix_research.benchmark_full_period
python -m vix_research.rolling_evaluation
python -m vix_research.continuous_2026
python -m vix_research.cost_diagnostic
```

Download end dates are exclusive. Network access is required for Yahoo Finance/FRED data. The study records source and input hashes and refuses to silently overwrite changed frozen studies. Fresh downloads may contain revisions and may not exactly reproduce the saved historical results. Raw downloaded prices are not distributed here. Some diagnostics verify earlier saved runs, so execute their prerequisites first. Full studies and bootstrap comparisons can take time.

Optional HMM/broader-universe study (after the literature study):

```sh
python -m pip install -r vix_research/requirements-extension.txt
python -m vix_research.fetch_extension
python -m vix_research.extension_study
```

## Execution assumptions

USD returns, adjusted-price total-return approximation, fractional holdings, prior-close signals and next adjusted-open execution. Existing holdings experience overnight gaps. Baseline costs are 1 bp commission plus 5 bps slippage per purchase/sale; a 0.5% cash reserve is retained without leverage. The four-ETF covariance estimate uses 63 sessions and fixed 25% diagonal shrinkage. A forecast volatility cap is not a guaranteed realised-risk or loss limit.

## Code map

| Files | Role |
|---|---|
| `core.py`, `backtest.py`, `data.py` | Signals, portfolio risk, accounting and data validation |
| `literature_strategy.py`, `literature_study.py` | Candidate policies and chronological research |
| `analytics.py`, `literature_analysis.py` | Performance and statistical diagnostics |
| `rolling_evaluation.py` | Five-year trailing, next-year policy selection |
| `extension_models.py`, `extension_engine.py` | HMM and related extensions |
| `continuous_2026.py`, `cost_diagnostic.py` | Extended core chart and zero-cost comparison |
| `tests/` | Accounting, causality and model checks |
| `LITERATURE_REGISTER.md`, `EXTENSION_REGISTER.md` | Hypotheses, sources, adaptations and omissions |

The standalone `VIX Fear-Regime Multi-Asset Strategy.py` is a QuantConnect adapter; run local research using the module commands above. Local accounting does not exactly emulate QuantConnect execution. The archive retains the original strategy for comparison.

Broker execution modules, account settings, credentials, runtime logs and private portfolio records are deliberately excluded. This repository is a research submission, not a configured brokerage deployment.
