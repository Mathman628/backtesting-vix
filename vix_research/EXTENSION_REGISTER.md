# Trend, gradual VIX exposure and regime-model extension

This register is frozen before this extension's performance is inspected. These are exploratory comparisons on previously inspected market history, not a fresh holdout or evidence of live profitability. Original studies, selection and trading code remain unchanged.

## Sources and implementation ledger

1. [QuantConnect: Asset Class Trend Following](https://www.quantconnect.com/research/15339/asset-class-trend-following/), including its [public legacy backtest code](https://www.quantconnect.com/terminal/cache/embedded_backtest_20251cb416bd2fdbef23227ec59e696d.html). Implemented: SPY/EFA/IEF/VNQ/GSG, 210 daily closing-price SMA, monthly decision, equal allocation among eligible assets; all fail means cash. Same five-asset passive portfolio isolates timing from asset-universe changes. An additional fixed-slot variant gives each eligible asset 20% and leaves failed slots in cash. This latter rule is an adaptation, not the linked code. Our next-open fills, cash interest, reserve and explicit costs differ from the source; we do not reproduce its historic headline returns. Monthly decisions use information through the previous close.

2. [Boukardagha, Explainable Regime-Aware Investing, arXiv:2603.04441](https://arxiv.org/abs/2603.04441). Implemented concepts: Gaussian HMM, rolling estimation, causal filtered state probabilities, predicted next-state probabilities, conditional return/covariance estimates, Gaussian Wasserstein regime matching, gradual template updates, constrained allocation with an L1 turnover penalty. This is a deliberately smaller adaptation. Fixed three states and 756 observations; monthly refits; seed 1729; diagonal feature covariance; 100 EM iterations, tolerance .01. Features: daily log returns, 60-session realised volatility and 20-session average log returns for SPY/IEF/GLD. Fit scaling uses only the trailing window. BIL enters portfolio return estimates, not state features. Historical posterior smoothing is confined to the already observed estimation window; evaluation uses forward filtering only. No future regime labels enter decisions.

   Adaptations: SPY/IEF/GLD/BIL instead of the paper's broader index universe; no oil or dollar sleeves. Conditional simple-return means shrink 50% toward the window mean; state covariances blend 25% global Ledoit-Wolf covariance. Mixture covariance includes between-state mean variation. Three persistent templates are matched one-to-one by diagonal-Gaussian squared Wasserstein distance in fixed initial feature units; feature templates and return moments update with 20% new information. This is not the paper's variable-number-of-states many-to-one aggregation. Labels themselves cannot create alpha: a consistent permutation leaves mixture moments unchanged. Tracking plus smoothed moments is compared with raw HMM moments; it tests that combined stabilisation treatment.

   Not implemented: adaptive state-count selection, predictive K-selection validation, KNN comparator, exact author calibration, variable-K template aggregation, oil/dollar instruments, exact event case studies, or reproduction of the reported Sharpe. These need additional identification/calibration and would enlarge the research search. No numerical defaults are claimed to be the author's settings. Model iteration-limit hits are disclosed, not hidden as convergence. No hyperparameters are selected using these backtest results.

3. [QuantConnect: Risk Regime Sector Rotation With Hedge](https://www.quantconnect.com/strategies/485/Risk-Regime-Sector-Rotation-With-Hedge). Deferred: the linked page returned an application shell without inspectable strategy code. No sector implementation or claimed replication is inferred from its title.

4. Existing project literature variants: static 50/25/25 with risk control, 1/3/12-month ensemble, previous revised VIX strategy. Reused without modifying their frozen sources. Benchmarks include monthly SPY/IEF 60/40 and monthly SPY/IEF/GLD 50/25/25. Treasury 60/40 is not an aggregate-bond or institutional-fund benchmark.

## Controlled smoother VIX experiment

Both policies retain 50/25/25 base weights, the same 12-month absolute momentum filter, hysteretic VIX states, 10% portfolio volatility cap, 0.5% cash reserve, monthly/weekly drift schedule and daily safety checks. Hard multiplier is 1/.75/.5/0. Smooth multiplier is 80% previous plus 20% today's hard multiplier. Initial state and PANIC apply immediately. The treatment changes exposure transitions only; mechanically resulting trades can differ. This is exponential smoothing, not an estimated probability of fear. It differs from the earlier full revised VIX allocation, which remains a separate comparator.

## Optimiser and controls

Unconditional moments, raw HMM moments and tracked/smoothed HMM moments all feed the same optimiser. Four assets, long-only, sum of weights one before reserve, each risky weight capped at 60%, BIL up to 100%, no leverage. Objective: five trading days times (negative daily expected return + gamma/2 times daily variance), gamma=10, plus .0006 times L1 change from actual weights. Auxiliary variables express the absolute turnover penalty. Weekly decisions; daily risk-cap breach check above 12% estimated volatility; common 10% volatility cap and .5% reserve applied after optimisation. Post-cap weights need not be the unconstrained optimiser optimum. A no-penalty tracked variant still pays actual trading costs. A solver failure stops the run; no unreported fallback.

## Predeclared hypotheses and expectations

- Five-asset trend versus passive five assets: expect smaller persistent-crisis drawdowns, but whipsaw and recovery lag. Sharpe improvement is uncertain. Compare both survivor reweighting and fixed slots; concentration can make survivor drawdowns worse.
- Smooth versus hard VIX: expect lower turnover and less transition whipsaw, but delayed reductions outside panic can worsen drawdowns. Primary practical criterion: lower turnover without a Sharpe decline exceeding .02; also report paired Sharpe difference and drawdown.
- Raw HMM versus unconditional optimiser: expect state information might improve risk-adjusted returns, but noisy estimated means can negate it. Primary criterion: higher Sharpe; pointwise block-bootstrap interval must exclude zero for stronger within-sample evidence.
- Tracked HMM versus raw HMM: expect steadier estimates, lower turnover, potentially less responsiveness. Practical criterion: lower turnover without a Sharpe decline exceeding .02.
- Turnover penalty versus zero penalty: expect lower trading and actual costs; it may sacrifice gross return. Practical criterion: lower turnover without a Sharpe decline exceeding .02.

These are a small disclosed hypothesis family, not a competition to deploy the largest observed Sharpe. Bootstrap intervals are paired 20-session circular blocks, 2,000 samples, seed 20260926. Intervals are pointwise, not multiple-testing-adjusted, and do not cure selection bias or structural change.

## Periods, data and execution

- Original horizon: 29 August 2007–29 December 2023, preserving the earlier long-history evaluation. Trend, VIX and existing controls only.
- Common HMM comparison: 3 January 2011–29 December 2023. All policies start together. BIL begins in 2007; 756 complete estimation observations are required before the first HMM decision. The 2011 start satisfies this without fabricated early BIL prices.
- Later horizon: 2 January 2024–23 September 2026, already viewed in prior research; separately reported to avoid treating its equity/gold exposure as broad-regime skill.

Each horizon starts a separate $100,000 portfolio, with continuous holdings within that horizon. HMM forecasts roll forward using only preceding data, including earlier observations within the evaluation span once they become available. This is causal rolling estimation, not an untouched strategy-selection test. Annual tables slice each continuous account; they do not reset yearly.

Original adjusted OHLC and lagged Treasury cash-rate files are unchanged. EFA/VNQ/GSG adjusted data were downloaded separately on 26 September 2026; original data vintage is 25 September. New assets are aligned to the original calendar with no internal forward-filling. Signals use prior close, trade at next adjusted open, old holdings earn overnight returns, cash earns lagged published-rate accrual. Commission 1bp plus slippage 5bp per dollar bought or sold; dividends are reflected in adjusted data. No taxes, market impact model, borrowing, or terminal liquidation. Dollar turnover is self-financing after costs. Exact wealth-linked asset/cash/cost attribution is exported to distinguish timing from favourable exposures.

## Reproduction

From the project directory: `.\vix_research\.venv\Scripts\python.exe -m vix_research.extension_study`.
Install additional dependencies from `requirements-extension.txt` if needed. The run verifies the original literature hashes and freezes extension/data hashes before evaluating results. Existing protocol mismatch requires a new versioned study, not silent overwrite. Open `outputs/research_extension/report.html`; read `ANALYSIS.md` for findings and `hypotheses.csv` for uncertainty. Source/data choices and parameters were fixed before reading this run's performance.
