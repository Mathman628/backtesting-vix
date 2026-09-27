"""Performance attribution, uncertainty, and chronological model evaluation."""
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import (ExecutionConfig, compare, comparison_variants,
                       run_backtest, prepare_snapshots, feature_key)
from .core import StrategyConfig, REGIMES


def drawdown(returns):
    growth = (1 + returns).cumprod()
    peak = growth.cummax().clip(lower=1.0)  # include capital before first trade
    return growth / peak - 1


def performance(daily):
    required = ["return", "risk_free_return", "turnover", "cost", "traded"]
    if not set(required).issubset(daily.columns):
        raise ValueError(f"Daily data require {required}")
    numeric = daily[["return", "risk_free_return", "turnover", "cost"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (daily["return"] <= -1).any():
        raise ValueError("Performance inputs must be finite, with returns greater than -100%")
    if (daily[["turnover", "cost"]] < 0).any().any():
        raise ValueError("Turnover and costs must be nonnegative")
    returns = daily["return"].astype(float)
    excess = returns - daily["risk_free_return"]
    count = len(returns)
    if count == 0:
        return {}
    growth = float((1 + returns).prod())
    years = count / 252
    cagr = growth ** (1 / years) - 1
    vol = float(returns.std(ddof=1) * np.sqrt(252)) if count > 1 else np.nan
    excess_std = float(excess.std(ddof=1)) if count > 1 else np.nan
    sharpe = float(excess.mean() / excess_std * np.sqrt(252)) if excess_std > 1e-12 else np.nan
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    sortino = float(excess.mean() / downside * np.sqrt(252)) if downside > 1e-12 else np.nan
    dd = drawdown(returns)
    max_dd = float(dd.min())
    duration = longest = 0
    for value in dd:
        duration = duration + 1 if value < -1e-12 else 0
        longest = max(longest, duration)
    tail = returns[returns <= returns.quantile(.05)]
    return {"sessions": count, "total_return": growth - 1, "cagr": cagr,
            "annual_vol": vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_dd, "calmar": cagr / abs(max_dd) if max_dd < -1e-12 else np.nan,
            "longest_underwater_sessions": longest, "currently_underwater_sessions": duration,
            "worst_day": float(returns.min()), "expected_shortfall_95_daily": float(tail.mean()),
            "positive_day_fraction": float((returns > 0).mean()),
            "gross_turnover_per_year": float(daily["turnover"].sum() / years),
            "rebalance_count": int(daily["traded"].sum()), "total_cost_dollars": float(daily["cost"].sum())}


def comparison_table(results):
    calendars = [r.daily.index for r in results.values()]
    if any(not index.equals(calendars[0]) for index in calendars[1:]):
        raise ValueError("Compare models on the same trading sessions")
    return pd.DataFrame({name: performance(result.daily) for name, result in results.items()}).T


def monthly_returns(daily):
    series = (1 + daily["return"]).resample("ME").prod() - 1
    table = pd.DataFrame({"year": series.index.year, "month": series.index.month,
                          "return": series.to_numpy()})
    return table.pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))


def yearly_metrics(daily):
    return pd.DataFrame({year: performance(group) for year, group in daily.groupby(daily.index.year)}).T


def regime_attribution(daily):
    records = []
    for regime, group in daily.groupby("regime"):
        records.append({"regime": REGIMES[int(regime)] if regime >= 0 else "BENCHMARK",
                        "sessions": len(group), "fraction": len(group) / len(daily),
                        "mean_daily_return": group["return"].mean(),
                        "daily_vol": group["return"].std(),
                        "log_return_contribution": np.log1p(group["return"]).sum(),
                        "cost_dollars": group["cost"].sum()})
    # Log-return contributions are additive; they are not regime CAGR estimates.
    return pd.DataFrame(records)


def rolling_metrics(daily, window=126):
    returns = daily["return"]
    excess = returns - daily["risk_free_return"]
    denominator = excess.rolling(window).std().replace(0, np.nan)
    return pd.DataFrame({"volatility": returns.rolling(window).std() * np.sqrt(252),
                         "sharpe": excess.rolling(window).mean() / denominator * np.sqrt(252),
                         "drawdown": drawdown(returns)}, index=daily.index)


def block_bootstrap(daily, samples=500, block=20, seed=17):
    """Circular block bootstrap; conditional uncertainty, not a future forecast."""
    if samples < 10 or block < 1 or block > len(daily):
        raise ValueError("Require samples >=10 and 1 <= block <= history length")
    rng = np.random.default_rng(seed)
    count = len(daily)
    pairs = daily[["return", "risk_free_return"]].to_numpy()
    output = []
    for _ in range(samples):
        starts = rng.integers(0, count, size=int(np.ceil(count / block)))
        indices = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
        values = pairs[indices]
        excess = values[:, 0] - values[:, 1]
        std = np.std(excess, ddof=1)
        returns = pd.Series(values[:, 0])
        output.append({"cagr": (1 + returns).prod() ** (252 / count) - 1,
                       "sharpe": excess.mean() / std * np.sqrt(252) if std > 1e-12 else np.nan,
                       "max_drawdown": drawdown(returns).min()})
    return pd.DataFrame(output).quantile([.025, .5, .975]).rename_axis("quantile")


def stress_variants(config=None):
    config = config or StrategyConfig()
    variants = {"base": (config, None)}
    for target in (.08, .12):
        variants[f"target_vol_{target:.0%}"] = (replace(config, target_vol=target), None)
    for window in (42, 126):
        variants[f"covariance_{window}"] = (replace(config, covariance_window=window), None)
    for shift in (-.03, .03):
        variants[f"threshold_shift_{shift:+.2f}"] = (replace(
            config, entry_thresholds=tuple(x + shift for x in config.entry_thresholds),
            exit_thresholds=tuple(x + shift for x in config.exit_thresholds)), None)
    variants["weekly_risk_only"] = (replace(config, daily_risk_checks=False), None)
    return variants


def cost_stress(data, config=None, execution=None, start=None, end=None):
    config, execution = config or StrategyConfig(), execution or ExecutionConfig()
    snapshots = prepare_snapshots(data, config)
    records = []
    for bps in (0, 5, 10, 25):
        result = run_backtest(data, config, replace(execution, slippage_bps=bps), start, end,
                              f"slippage_{bps}bps", snapshots=snapshots)
        records.append({"slippage_bps_per_side": bps, "commission_bps_per_side": execution.commission_bps,
                        **performance(result.daily)})
    return pd.DataFrame(records)


def walk_forward(data, config=None, execution=None, start=None, end=None,
                 train_years=5, test_years=1):
    """Select on trailing training years, then evaluate untouched next fold.

    Each fold starts in cash and includes terminal liquidation costs. Fold
    results are independent experiments, NOT a stitched investable NAV curve.
    Historical periods already examined by a human are not truly untouched.
    """
    if train_years < 1 or test_years < 1:
        raise ValueError("Training and test years must be positive")
    config, execution = config or StrategyConfig(), execution or ExecutionConfig()
    available = comparison_variants(config)
    candidates = {k: available[k] for k in ("revised", "sma_only", "momentum_only", "spy_bil_no_vix")}
    cache = {}
    lower = pd.Timestamp(start) if start else data.closes.index[0]
    upper = pd.Timestamp(end) if end else data.closes.index[-1]
    for cfg, _ in candidates.values():
        key = feature_key(cfg)
        if key not in cache:
            cache[key] = prepare_snapshots(data, cfg)
        ready = next((i for i, s in enumerate(cache[key]) if s is not None), None)
        if ready is None:
            raise ValueError("Not enough history")
        lower = max(lower, data.closes.index[ready])
    test_start = lower + pd.DateOffset(years=train_years)
    records = []

    def evaluate(cfg, first, last, name):
        result = run_backtest(data, cfg, execution, first, last, name,
                              snapshots=cache[feature_key(cfg)])
        # Pay for closing positions at the end of this independent experiment.
        last_day = result.daily.index[-1]
        liquidation = (result.weights.loc[last_day, list(ASSETS)].sum()
                       * result.daily.loc[last_day, "nav"]
                       * (execution.commission_bps + execution.slippage_bps) / 10_000)
        nav_before = result.daily.loc[last_day, "nav"]
        result.daily.loc[last_day, "return"] = ((1 + result.daily.loc[last_day, "return"])
                                                 * (1 - liquidation / nav_before) - 1)
        result.daily.loc[last_day, "nav"] -= liquidation
        result.daily.loc[last_day, "cost"] += liquidation
        result.daily.loc[last_day, "turnover"] += result.weights.loc[last_day, list(ASSETS)].sum()
        result.daily.loc[last_day, "traded"] = True
        return performance(result.daily)

    from .core import ASSETS
    while test_start <= upper:
        train_start = test_start - pd.DateOffset(years=train_years)
        train_end = test_start - pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), upper)
        if len(data.closes.loc[test_start:test_end]) < 63:
            break
        training = {name: evaluate(cfg, train_start, train_end, name)
                    for name, (cfg, _) in candidates.items()}
        valid = {name: row["sharpe"] for name, row in training.items() if np.isfinite(row["sharpe"])}
        if not valid:
            raise ValueError("No finite training Sharpe ratios")
        winner = max(valid, key=valid.get)
        print(f"Walk-forward {test_start.date()}: training selects {winner}", flush=True)
        test_stats = evaluate(candidates[winner][0], test_start, test_end, winner)
        fixed_stats = evaluate(config, test_start, test_end, "fixed_revised")
        records.append({"train_start": train_start.date(), "train_end": train_end.date(),
                        "test_start": test_start.date(), "test_end": test_end.date(),
                        "selected": winner, "train_sharpe": valid[winner],
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        "fixed_revised_test_sharpe": fixed_stats["sharpe"],
                        "fixed_revised_test_cagr": fixed_stats["cagr"]})
        test_start += pd.DateOffset(years=test_years)
    if not records:
        raise ValueError("Not enough sessions for training plus a >=63-session test fold")
    return pd.DataFrame(records)
