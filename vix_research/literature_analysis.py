"""Paired uncertainty, approximate selection diagnostics, and exact attribution."""
from statistics import NormalDist
import numpy as np
import pandas as pd
from .core import ASSETS, RISKY
from .analytics import performance

BENCHMARK = "factor_t0_r1_v0"
HYPOTHESES = [
    ("H1", "factor_t1_r1_v0", BENCHMARK, "sharpe"),
    ("H2", "factor_t1_r1_v0", "factor_t1_r0_v0", "risk"),
    ("H3", "factor_t1_r1_v1", "factor_t1_r1_v0", "drawdown"),
    ("H4", "excess12", "factor_t1_r1_v0", "sharpe"),
    ("H5", "ensemble", "excess12", "sharpe"),
    ("H6", "ensemble_band", "ensemble", "cost"),
    ("H7", "ensemble_partial", "ensemble", "cost"),
    ("H8", "ensemble_reentry", "ensemble", "drawdown"),
    ("H9", "blend_80_20", BENCHMARK, "joint"),
    ("H10", "inverse_vol_ensemble", "ensemble", "sharpe"),
    ("H11", "regime_ensemble", "ensemble", "sharpe"),
]


def paired_sharpes(results, samples=2000, block=20, seed=20260926):
    names = list(results)
    dates = results[names[0]].daily.index
    if any(not r.daily.index.equals(dates) for r in results.values()):
        raise ValueError("Paired comparisons require identical dates")
    values = np.column_stack([(r.daily["return"] - r.daily.risk_free_return).to_numpy() for r in results.values()])
    n = len(values)
    if not 1 <= block <= n or samples < 100:
        raise ValueError("Invalid bootstrap size")
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(names)))
    for j in range(samples):
        starts = rng.integers(n, size=int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        sample = values[idx]
        std = sample.std(axis=0, ddof=1)
        draws[j] = np.divide(sample.mean(axis=0) * np.sqrt(252), std,
                             out=np.full(len(names), np.nan), where=std > 1e-12)
    return pd.DataFrame(draws, columns=names)


def compare_hypotheses(table, draws, split):
    records = []
    for key, treatment, control, criterion in HYPOTHESES:
        a, b = table.loc[treatment], table.loc[control]
        delta = draws[treatment] - draws[control]
        low, high = np.nanquantile(delta, [.025, .975])
        ds = a.sharpe - b.sharpe
        dd = a.max_drawdown - b.max_drawdown  # positive is a smaller loss
        dt = a.gross_turnover_per_year - b.gross_turnover_per_year
        supported = {"sharpe": ds > 0,
                     "risk": a.annual_vol < b.annual_vol and dd > 0,
                     "drawdown": dd > 0,
                     "cost": dt < 0 and a.total_cost_dollars < b.total_cost_dollars and ds >= -.02,
                     "joint": ds > 0 and dd > 0}[criterion]
        records.append(dict(hypothesis=key, split=split, treatment=treatment, control=control,
                            criterion=criterion, descriptive_support=bool(supported),
                            sharpe_difference=ds, sharpe_ci_low=low, sharpe_ci_high=high,
                            sharpe_evidence="positive pointwise interval" if low > 0 else "negative pointwise interval" if high < 0 else "inconclusive",
                            drawdown_improvement=dd, volatility_change=a.annual_vol-b.annual_vol,
                            turnover_change=dt, cost_dollars_change=a.total_cost_dollars-b.total_cost_dollars))
    return pd.DataFrame(records)


def deflated_sharpe(results, candidate_names):
    """Monthly IID approximation. Trial-count sensitivity, NOT a certified p-value."""
    monthly = {}
    for name in candidate_names:
        d = results[name].daily
        monthly[name] = (1+d["return"]).resample("ME").prod() - (1+d.risk_free_return).resample("ME").prod()
    frame = pd.DataFrame(monthly)
    sharpes = frame.mean() / frame.std(ddof=1)
    dispersion = float(sharpes.std(ddof=1))
    normal = NormalDist()
    rows = []
    for name in candidate_names:
        x = frame[name].to_numpy()
        sr = sharpes[name]
        centered = x - x.mean()
        m2 = np.mean(centered**2)
        skew = np.mean(centered**3) / m2**1.5 if m2 > 0 else np.nan
        kurt = np.mean(centered**4) / m2**2 if m2 > 0 else np.nan
        denom = 1 - skew*sr + (kurt-1)*sr*sr/4
        for trials in sorted(set((len(candidate_names), 50, 100))):
            gamma = .5772156649015329
            threshold = dispersion * ((1-gamma)*normal.inv_cdf(1-1/trials) + gamma*normal.inv_cdf(1-1/(trials*np.e)))
            z = (sr-threshold)*np.sqrt(len(x)-1)/np.sqrt(denom) if denom > 0 else np.nan
            rows.append(dict(model=name, assumed_independent_trials=trials, months=len(x),
                             monthly_sharpe=sr, threshold_monthly_sharpe=threshold,
                             dsr_probability_approx=normal.cdf(z) if np.isfinite(z) else np.nan,
                             skewness=skew, kurtosis=kurt,
                             caveat="IID monthly approximation; unknown earlier trials; correlated candidates; partial boundary months"))
    return pd.DataFrame(rows)


def attribution(result):
    columns = [f"contribution_{s}" for s in (*ASSETS, "cash", "cost")]
    d = result.daily
    error = float(np.max(np.abs(d[columns].sum(axis=1)-d["return"])))
    if error > 1e-10:
        raise ArithmeticError(f"Attribution does not reconcile: {error}")
    # Multiplying each period's contribution by wealth entering that period
    # produces additive contributions to terminal wealth / initial capital.
    wealth_before = (1+d["return"]).cumprod().shift(1, fill_value=1.)
    linked = d[columns].mul(wealth_before, axis=0)
    summary = pd.DataFrame({"linked_total_return_contribution": linked.sum(),
                            "mean_daily_contribution": d[columns].mean()})
    summary.index.name = "component"
    return summary, d[columns].corr(), error


EPISODES = {
    "GFC_decline": ("2007-10-09", "2009-03-09"),
    "GFC_recovery": ("2009-03-10", "2009-12-31"),
    "COVID_decline": ("2020-02-19", "2020-03-23"),
    "COVID_recovery": ("2020-03-24", "2020-12-31"),
    "Inflation_2022": ("2022-01-01", "2022-12-31"),
    "Recovery_2023": ("2023-01-01", "2023-12-31"),
    "Test_2024": ("2024-01-01", "2024-12-31"),
    "Test_2025": ("2025-01-01", "2025-12-31"),
    "Test_2026_partial": ("2026-01-01", "2026-09-23"),
}


def episode_analysis(results, data):
    rows = []
    asset_returns = data.closes[list(ASSETS)].pct_change(fill_method=None)
    cash = data.risk_free_returns()
    for name, result in results.items():
        for episode, (start, end) in EPISODES.items():
            d = result.daily.loc[start:end]
            if d.empty:
                continue
            # Do not silently report truncated historical episodes for a split.
            available = data.closes.loc[start:end].index
            if not d.index.equals(available):
                continue
            b = results[BENCHMARK].daily.loc[d.index]
            total = float((1+d["return"]).prod()-1)
            bench = float((1+b["return"]).prod()-1)
            spy = float((1+asset_returns.loc[d.index,"SPY"]).prod()-1)
            # Ex-post counterfactual: replace previous-close BIL+cash weight by
            # SPY through the NEXT close. Includes negative avoided-loss values.
            # Not executable alpha, not the effect of changing the actual policy.
            w = result.weights.reindex(result.daily.index).shift(1)
            w.iloc[0] = [0, 0, 0, 0, 1]
            bill = w.loc[d.index,"BIL"]*(asset_returns.loc[d.index,"SPY"]-asset_returns.loc[d.index,"BIL"])
            usd = w.loc[d.index,"CASH"]*(asset_returns.loc[d.index,"SPY"]-cash.loc[d.index])
            rows.append(dict(model=name, episode=episode, start=str(d.index[0].date()), end=str(d.index[-1].date()),
                             return_total=total, benchmark_return=bench, return_gap=total-bench,
                             spy_return=spy, mean_spy_weight=result.weights.loc[d.index,"SPY"].mean(),
                             cost=d.cost.sum(), turnover=d.turnover.sum(),
                             ex_post_cash_drag_daily_sum=float((bill+usd).sum())))
    return pd.DataFrame(rows)


def market_capture(results, data):
    asset_returns = data.closes[list(RISKY)].pct_change(fill_method=None)
    rf = data.risk_free_returns()
    rows=[]
    for name, result in results.items():
        for year, daily in result.daily.groupby(result.daily.index.year):
            if len(daily) < 126:
                continue
            for s in RISKY:
                x = asset_returns.loc[daily.index,s]-rf.loc[daily.index]
                y = daily[f"contribution_{s}"]
                rows.append(dict(model=name, year=year, asset=s, sessions=len(daily),
                    absolute_market_sharpe=abs(x.mean()/x.std(ddof=1)*np.sqrt(252)),
                    contribution_sharpe=y.mean()/y.std(ddof=1)*np.sqrt(252) if y.std(ddof=1)>1e-12 else np.nan,
                    mean_weight=result.weights.loc[daily.index,s].mean()))
    return pd.DataFrame(rows)
