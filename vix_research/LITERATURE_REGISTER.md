# Literature register: implemented, adapted and not implemented

This register accompanies a retrospective extension of the existing VIX study.
It is a research record, not a claim to replicate the cited papers. The original
QuantConnect strategy remains the original revised default. New policies run in
the local simulator through `literature_strategy.py`; they have not been deployed
to a brokerage or verified in LEAN.

## Sources and implementation decisions

| Source | Evidence read | Implemented or adapted | Not implemented, and why |
|---|---|---|---|
| Moskowitz, Ooi & Pedersen (2012), [Time Series Momentum](https://doi.org/10.1016/j.jfineco.2011.11.003) | Supplied complete paper; especially sections 2.4, 3.2 and Figure 1 | Twelve-month own-return signals; lagged cash-relative momentum; volatility sizing; next-session execution | Short futures, 58-market universe, hedger/speculator positioning, paper's exact volatility estimator. Our long-only ETF/cash policy cannot replicate its two-sided crisis payoff. |
| Hurst, Ooi & Pedersen (2017), [A Century of Evidence on Trend-Following Investing](https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing) | Supplied paper; construction and Exhibits 6–10 | Fixed 21/63/252-session excess-momentum ensemble; portfolio risk cap; 80/20 static/trend allocation blend; crisis/recovery comparisons | 67 futures/FX markets, 1880 history, shorting, leverage, simulated hedge-fund fees. Our blend combines target allocations before risk capping, not the paper's separately managed long-short fund returns. |
| Babu et al. (2020), [You Can't Always Trend When You Want](https://www.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-You-Cant-Always-Trend-When-You-Want.pdf) | Supplied journal version; decomposition and Exhibit 8 | Exact asset/cash/cost attribution; annual market-move/capture scatter; contribution correlations; crisis and missed-recovery analysis | Exact fitted efficacy/diversification decomposition. Long-only exposures, limited assets and short sample do not satisfy the paper's setup. Diagnostics do not prove why past performance changed. |
| Thapar, Nielsen & Villalon (2019), [Chasing Your Own Tail (Risk), Revisited](https://www.aqr.com/insights/research/white-papers/chasing-your-own-tail-risk-revisited) | Supplied paper; managed futures, risk targeting and drawdown-control sections, Exhibits 8–9 | Explicit re-entry-cap variant; risk/return trade-off; slow-crisis and rapid-recovery diagnostics; inverse-volatility asset-weight variant | Protective options, low-beta stock selection, leverage, new portfolio drawdown stop. Options/stock selection need different data; another stop may duplicate existing controls. Inverse volatility is not exact equal-risk-contribution optimisation. |
| Kim, Tse & Wald (2016), [Time Series Momentum and Volatility Scaling](https://doi.org/10.1016/j.finmar.2016.05.003) | Publisher abstract/highlights and introductory discussion | Full 2x2x2 trend/risk-sizing/VIX experiment, matched assets and execution assumptions | Full futures replication; alpha-factor regressions. Results here concern ETF portfolio utility, not a factor anomaly replication. |
| Moreira & Muir (2017), [Volatility-Managed Portfolios](https://doi.org/10.1111/jofi.12513) | Publisher abstract and author manuscript reviewed in preceding work | Risk-scaling hypothesis and unscaled controls | Exact inverse-variance factor strategy and leverage. Our cap is unlevered covariance-based sizing. |
| Cederburg et al. (2020), [On the Performance of Volatility-Managed Portfolios](https://experts.arizona.edu/en/publications/on-the-performance-of-volatility-managed-portfolios/) | Author-university abstract | Chronological selection; unchanged benchmark; separately evaluated historical test; no retrospective tuning after test | Replication across 103 equity strategies and spanning regressions. No factor panel in our dataset. |
| Gârleanu & Pedersen (2013), [Dynamic Trading with Predictable Returns and Transaction Costs](https://pages.stern.nyu.edu/~lpederse/papers/DynamicTrading.pdf) | Author manuscript's trading framework | Separate wider-band and 50% partial-routine-adjustment policies; cost stress at 0/5/10/25 bps slippage | Exact optimal dynamic policy, fitted expected returns, signal decay and market-impact matrix. Our bands and partial steps are transparent heuristics, not the paper's optimum. Emergency exits bypass smoothing. |
| Bailey & López de Prado (2014), [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Author manuscript, selection bias and DSR sections | Complete current-trial inventory; monthly DSR sensitivity for current trial count and 50/100 trials; paired daily block-bootstrap differences | Certified global DSR or probability of backtest overfitting. Earlier informal trials are not fully recoverable; correlated trials and serial dependence make the displayed DSR an approximate sensitivity diagnostic. No arbitrary claim that a low DSR proves no skill. |
| Huang et al. (2020), [Time-Series Momentum: Is It There?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3165284) | Author-posted abstract and publisher conclusion | Counterevidence in report; simple matched controls and restrained inference | Their predictive-regression replication, futures panel and parametric bootstrap. Our paired bootstrap tests portfolio differences, a different hypothesis. |
| Bollerslev, Tauchen & Zhou (2009), [Expected Stock Returns and Variance Risk Premia](https://public.econ.duke.edu/~get/wpapers/btz.pdf) | Author-hosted paper, abstract and economic mechanism | Documented distinction between risk forecasts and return forecasts; VIX treated as a testable overlay | Variance-premium trading signal. Accurate high-frequency realized variance and a separately frozen forecasting design are missing. VIX minus daily volatility is not silently substituted for their measure. |
| Brooks (2017), [A Half Century of Macro Momentum](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/A-Half-Century-of-Macro-Momentum.pdf) | AQR paper, construction, signal table and data appendix | Macro extension documented as deferred | Growth/inflation/policy/FX signals. Point-in-time forecasts and release vintages are not in the current input set. No revised economic data are used as though known historically. |

The four supplied papers overlap in authors, data and AQR affiliation. They are
not four independent validations of this implementation. None establishes our
VIX thresholds or the ETF allocation map. `JPM` denotes the journal, not JPMorgan.

## Expected outcomes recorded before the new market-data runs

These expectations follow earlier research and already-seen results. They are
prospective to this implementation run, not a genuine historical preregistration.

| ID | Frozen comparison | Expectation / primary directional criterion |
|---|---|---|
| H1 | Trend on vs off; portfolio risk control on; VIX off | Trend may reduce persistent losses, but our earlier static result makes higher Sharpe uncertain. Test Sharpe improvement. |
| H2 | Portfolio risk cap on vs off; trend on; VIX off | Expect lower realised volatility and a smaller maximum loss; Sharpe improvement is not assumed. |
| H3 | VIX multiplier on vs off; trend/risk on | Expect less drawdown, possibly lower returns and more switching. Test Sharpe separately; no assumed directional return-forecast skill. |
| H4 | Excess-return vs absolute twelve-month momentum | Expect cash hurdle to help when rates are high; test higher Sharpe, recognising slower re-entry as a cost. |
| H5 | 1/3/12-month excess ensemble vs excess twelve-month | Expect more gradual exposure and faster adaptation; higher Sharpe is uncertain because fast signals can increase turnover. |
| H6 | 3-percentage-point routine band / 8-point weekly drift vs ensemble | Expect lower turnover and costs; require Sharpe loss no worse than 0.02 as an economic tolerance, not a statistical noninferiority test. |
| H7 | Half routine adjustment vs full ensemble adjustment | Same cost/Sharpe criterion as H6; lower turnover expected, potentially slower response. |
| H8 | Maximum 20-point risky re-entry per routine event vs ensemble | Expect fewer large false re-entries and smaller drawdown, but greater missed recovery. Compare drawdowns; examine recovery trade-off. |
| H9 | 80% static / 20% ensemble target blend vs static risk-controlled | Expect more upside participation than pure trend and some downside benefit; require higher Sharpe and smaller drawdown than static. |
| H10 | Inverse-volatility risky weights vs 50/25/25 ensemble | Expect more balanced stand-alone risk, but covariance can defeat diversification. Test higher Sharpe. |
| H11 | Original four-regime map with ensemble vs static-map ensemble | Tests the complete allocation-map alternative, not VIX alone. Higher Sharpe uncertain. |

All controlled models can hold SPY, IEF, GLD and BIL. The isolated VIX overlay
multiplies risky weights by 1, .75, .5 or 0 in the four existing fear states and
transfers the balance to BIL. These multipliers are a new explicit hypothesis,
not parameters claimed to come from a paper. Trend, risk and VIX switches form
eight controlled combinations. Additional variants change one declared feature.

Every decision uses the previous completed session. All controlled models use
monthly routine rebalancing, weekly drift checks and daily exits from now-zero
targets. Portfolio-risk emergency cuts exist only when that risk treatment is on.
This means a risk-treatment comparison includes its risk-triggered execution.
No paper strategy is described as exactly replicated.

## Chronological protocol

- Development: 29 August 2007 through 31 December 2018.
- Validation: 1 January 2019 through 31 December 2023.
- Historical test: 1 January 2024 through 23 September 2026.
- Each period independently starts in cash with earlier data for signal warm-up.
  Results use mark-to-market ending NAV without forced liquidation, consistently
  for every model. They are not concatenated into an investable track record.
- Freeze configuration, code hashes, input hashes, expected directions, selection
  rule and comparisons before development. Run development and validation first.
- Development Sharpe selects three candidates; validation Sharpe selects one of
  those three. If its validation Sharpe does not exceed the controlled static
  risk benchmark, select the benchmark. Tie-break by lower turnover then name.
- Save the selection before running any new test-period backtest. Run all frozen
  variants on the test for the predeclared hypothesis comparisons; do not select
  a replacement from test results. Test cost scenarios are robustness diagnostics.
- Bootstrap paired return/cash observations with identical circular 20-session
  blocks, 2,000 resamples and fixed seed; show pointwise 95% intervals. Additional
  5/60-session block checks apply to the selected model vs benchmark in the test.
  These intervals do not correct the family of comparisons or prior exploration.
- Monthly DSR uses skew/kurtosis and the dispersion of candidate Sharpes. Display
  current-family, 50-trial and 100-trial assumptions, never a definitive global
  multiple-testing correction. Historical test history is short for this purpose.
- An expected direction can be supported descriptively without a positive paired
  interval. Report those outcomes separately. A confidence interval crossing zero
  means uncertainty, not equivalence or proof that a feature has no value.

The test period was already examined in the previous project. This study can
demonstrate reproducible chronological evaluation, but cannot manufacture an
untouched holdout. A genuinely prospective test must begin after these rules are
frozen and needs future observations; it cannot be completed now.
