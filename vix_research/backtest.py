"""Daily research simulator: prior close signals, next adjusted open fills.

This is not a LEAN execution emulator. Fractional adjusted-price units model
total returns. Gaps accrue to OLD positions before any rebalance. Costs are
charged to both purchases and sales; remaining cash earns the lagged rate.
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from .core import (ASSETS, StrategyConfig, SignalEngine, make_snapshot,
                   cap_portfolio_vol, estimated_vol)
from .data import MarketData


@dataclass(frozen=True)
class ExecutionConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 5.0

    def __post_init__(self):
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        for value in (self.commission_bps, self.slippage_bps):
            if not np.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Costs must be between 0 and 100 bps per side")


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    config: dict
    execution: dict
    metadata: dict


def feature_key(config):
    return tuple(getattr(config, field) for field in (
        "percentile_window", "trend_window", "momentum_window", "defensive_window",
        "covariance_window", "equity_vol_window", "covariance_shrinkage"))


def prepare_snapshots(data, config):
    """Index i contains only information available at the end of session i-1."""
    return [None if i < config.history_rows else
            make_snapshot(data.closes.iloc[max(0, i - config.history_rows):i], config)
            for i in range(len(data.closes))]


def rebalance_values(values, cash, targets, cost_rate):
    """Solve self-financing post-cost target allocations, with no borrowing."""
    pre_nav = float(values.sum() + cash)
    post_nav = pre_nav
    for _ in range(40):
        dollars = post_nav * targets
        costs = np.abs(dollars - values).sum() * cost_rate
        updated = pre_nav - costs
        if abs(updated - post_nav) < 1e-9:
            post_nav = updated
            break
        post_nav = updated
    dollars = post_nav * targets
    deltas = dollars - values
    costs = np.abs(deltas).sum() * cost_rate
    new_cash = pre_nav - costs - dollars.sum()
    if new_cash < -1e-7:
        raise ArithmeticError("Rebalance would borrow cash")
    return dollars, max(0.0, new_cash), deltas, float(costs)


def run_backtest(data: MarketData, config=None, execution=None, start=None, end=None,
                 name="revised", baseline=None, snapshots=None, engine=None):
    data.validate()
    config = config or StrategyConfig()
    execution = execution or ExecutionConfig()
    if baseline not in (None, "spy", "static", "static_vol"):
        raise ValueError("Unknown baseline")
    snapshots = snapshots if snapshots is not None else prepare_snapshots(data, config)
    if len(snapshots) != len(data.closes):
        raise ValueError("Snapshot calendar mismatch")
    rates = data.risk_free_returns()
    dates = data.closes.index
    lower = pd.Timestamp(start) if start else dates[0]
    upper = pd.Timestamp(end) if end else dates[-1]
    valid = [i for i, d in enumerate(dates) if lower <= d <= upper and snapshots[i] is not None]
    if not valid:
        raise ValueError("No ready trading sessions. Supply longer history (including pre-start warm-up).")
    first, last = valid[0], valid[-1]
    values = np.zeros(4, dtype=float)
    cash = previous_nav = execution.initial_cash
    engine = engine if engine is not None else SignalEngine(config)
    close_array = data.closes.loc[:, ASSETS].to_numpy(dtype=float)
    open_array = data.opens.loc[:, ASSETS].to_numpy(dtype=float)
    daily, weights_rows, signal_rows, trades = [], [], [], []
    baseline_month = None
    cost_rate = (execution.commission_bps + execution.slippage_bps) / 10_000
    for i in range(first, last + 1):
        day = dates[i]
        snapshot = snapshots[i]
        if snapshot is None:
            raise ValueError(f"Signals are not ready on {day.date()}; refusing to skip a held session")
        if snapshot.date != dates[i - 1].date():
            raise ValueError("Snapshot must use the immediately preceding observed SPY session")
        # Mark existing holdings through the overnight gap BEFORE trading.
        gap_pnl = values * (open_array[i] / close_array[i - 1] - 1)
        values += gap_pnl
        cash_income = cash * rates.iloc[i]
        cash += cash_income
        pre_nav = float(values.sum() + cash)
        actual = dict(zip(ASSETS, values / pre_nav))
        decision = engine.decide(snapshot, day, actual)
        targets, trade, reason = decision.targets, decision.trade, decision.reason
        regime = decision.regime
        if baseline:
            base = ({"SPY": 1.0} if baseline == "spy"
                    else {"SPY": .50, "IEF": .25, "GLD": .25})
            if baseline == "static_vol":
                base = cap_portfolio_vol(base, snapshot.covariance, config.target_vol)
            targets = {s: base.get(s, 0) * (1 - config.cash_buffer) for s in ASSETS}
            month = (day.year, day.month)
            trade = i == first or (baseline != "spy" and month != baseline_month)
            baseline_month = month
            reason = "baseline-rebalance" if trade else "hold"
            regime = -1
        target_array = np.array([targets.get(s, 0) for s in ASSETS])
        costs = turnover = 0.0
        if trade:
            values, cash, deltas, costs = rebalance_values(values, cash, target_array, cost_rate)
            turnover = float(np.abs(deltas).sum() / pre_nav)  # gross buys + sells
            for j, symbol in enumerate(ASSETS):
                if abs(deltas[j]) > 1e-7:
                    trades.append({"date": day, "signal_date": snapshot.date, "symbol": symbol,
                                   "dollars": deltas[j], "adjusted_fill_price": open_array[i, j],
                                   "adjusted_units": deltas[j] / open_array[i, j],
                                   "cost": abs(deltas[j]) * cost_rate, "reason": reason})
        # Then mark the NEW holdings through the trading session.
        session_pnl = values * (close_array[i] / open_array[i] - 1)
        values += session_pnl
        nav = float(values.sum() + cash)
        if nav <= 0 or not np.isfinite(nav):
            raise ArithmeticError("Invalid portfolio NAV")
        daily.append({"date": day, "nav": nav, "return": nav / previous_nav - 1,
                      "risk_free_return": rates.iloc[i], "cost": costs,
                      "turnover": turnover, "traded": trade, "regime": regime,
                      "vix": snapshot.vix, "vix_percentile": snapshot.percentile,
                      "estimated_vol": estimated_vol(targets, snapshot.covariance),
                      "current_estimated_vol": decision.current_vol,
                      "signal_date": snapshot.date, "reason": reason})
        # Exact arithmetic contributions: old holdings earn gaps; new holdings
        # earn the session. Linked contributions can subsequently explain wealth.
        daily[-1].update({f"contribution_{s}": float((gap_pnl[j] + session_pnl[j]) / previous_nav)
                          for j, s in enumerate(ASSETS)})
        daily[-1].update({"contribution_cash": float(cash_income / previous_nav),
                          "contribution_cost": float(-costs / previous_nav),
                          "gap_return": float(gap_pnl.sum() / previous_nav)})
        weights_rows.append({"date": day, **dict(zip(ASSETS, values / nav)), "CASH": cash / nav})
        signal_rows.append({"date": day, "signal_date": snapshot.date,
                            **{f"target_{s}": targets[s] for s in ASSETS},
                            "trend_ok": snapshot.trend_ok, "momentum_ok": snapshot.momentum_ok,
                            "ief_ok": snapshot.defensive_ok["IEF"], "gld_ok": snapshot.defensive_ok["GLD"],
                            "regime": regime, "trade": trade, "reason": reason})
        previous_nav = nav
    metadata = {**data.metadata, "evaluation_requested_start": str(lower.date()),
                "actual_start": str(dates[first].date()), "actual_end": str(dates[last].date()),
                "execution_model": "Prior-session signals; next adjusted open; fractional total-return units",
                "cash_accrual": "Prior close cash accrues lagged Treasury yield for the calendar-day interval",
                "baseline": baseline}
    return BacktestResult(name, pd.DataFrame(daily).set_index("date"),
                          pd.DataFrame(weights_rows).set_index("date"),
                          pd.DataFrame(trades, columns=["date", "signal_date", "symbol", "dollars",
                                                        "adjusted_fill_price", "adjusted_units", "cost", "reason"]),
                          pd.DataFrame(signal_rows).set_index("date"),
                          config.to_dict(), execution.__dict__, metadata)


def comparison_variants(config=None):
    config = config or StrategyConfig()
    return {
        "revised": (config, None),
        "corrected_original_rules": (replace(config, defensive_filter=False, vol_model="equity",
                                               daily_risk_checks=False), None),
        "sma_only": (replace(config, equity_filter="sma"), None),
        "momentum_only": (replace(config, equity_filter="momentum"), None),
        "no_equity_filter": (replace(config, equity_filter="none"), None),
        "no_defensive_filter": (replace(config, defensive_filter=False), None),
        "equity_vol_only": (replace(config, vol_model="equity"), None),
        "spy_bil_no_vix": (replace(config, use_vix=False), None),
        "spy_buy_hold": (config, "spy"),
        "static_50_25_25": (config, "static"),
        "static_vol_target": (config, "static_vol"),
    }


def compare(data, variants=None, execution=None, start=None, end=None):
    variants = variants or comparison_variants()
    cache = {}
    # All models start together, including sensitivity runs with longer windows.
    common = pd.Timestamp(start) if start else data.closes.index[0]
    for config, _ in variants.values():
        key = feature_key(config)
        if key not in cache:
            cache[key] = prepare_snapshots(data, config)
        ready = [i for i, item in enumerate(cache[key]) if item is not None]
        if not ready:
            raise ValueError("Insufficient history for a comparison variant")
        common = max(common, data.closes.index[ready[0]])
    results = {}
    for name, (config, baseline) in variants.items():
        print(f"Backtesting {name} ...", flush=True)
        results[name] = run_backtest(data, config, execution, common, end, name, baseline,
                                     snapshots=cache[feature_key(config)])
    return results
