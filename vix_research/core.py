"""Shared, causal signal engine. No broker or market-data network dependencies.

Weights describe a fully allocated, unlevered model portfolio. Execution adapters
apply the same cash buffer. A decision uses only data through snapshot.date.
"""
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

ASSETS = ("SPY", "IEF", "GLD", "BIL")
RISKY = ASSETS[:3]
REGIMES = ("RISK-ON", "CAUTION", "FEAR", "PANIC")
REGIME_WEIGHTS = (
    {"SPY": 1.0},
    {"SPY": 0.50, "IEF": 0.25, "GLD": 0.25},
    {"IEF": 0.40, "GLD": 0.35, "BIL": 0.25},
    {"BIL": 1.0},
)


@dataclass(frozen=True)
class StrategyConfig:
    percentile_window: int = 252
    trend_window: int = 200
    momentum_window: int = 252
    defensive_window: int = 126
    covariance_window: int = 63
    equity_vol_window: int = 21
    covariance_shrinkage: float = 0.25
    target_vol: float = 0.10
    entry_thresholds: tuple = (0.65, 0.85, 0.95)
    exit_thresholds: tuple = (0.55, 0.75, 0.90)
    panic_vix: float = 35.0
    panic_exit_vix: float = 30.0
    equity_filter: str = "both"  # both, sma, momentum, none
    defensive_filter: bool = True
    use_vix: bool = True
    vol_model: str = "portfolio"  # portfolio, equity, none
    drift_tolerance: float = 0.05
    minimum_trade_weight: float = 0.005
    cash_buffer: float = 0.005
    daily_risk_checks: bool = True
    risk_breach_tolerance: float = 0.20

    def __post_init__(self):
        for key in ("defensive_filter", "use_vix", "daily_risk_checks"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a JSON/Python boolean")
        for key in ("percentile_window", "trend_window", "momentum_window",
                    "defensive_window", "covariance_window", "equity_vol_window"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{key} must be an integer >= 2")
        for key in ("covariance_shrinkage", "cash_buffer", "drift_tolerance",
                    "minimum_trade_weight", "risk_breach_tolerance"):
            if not np.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be in [0, 1)")
        if not np.isfinite(self.target_vol) or not 0 < self.target_vol <= 1:
            raise ValueError("target_vol must be in (0, 1]")
        if not (np.isfinite(self.panic_vix) and np.isfinite(self.panic_exit_vix)
                and 0 < self.panic_exit_vix < self.panic_vix):
            raise ValueError("Require 0 < panic_exit_vix < panic_vix")
        if self.equity_filter not in ("both", "sma", "momentum", "none"):
            raise ValueError("Invalid equity_filter")
        if self.vol_model not in ("portfolio", "equity", "none"):
            raise ValueError("Invalid vol_model")
        if len(self.entry_thresholds) != 3 or len(self.exit_thresholds) != 3:
            raise ValueError("Three entry and exit thresholds are required")
        if not all(0 <= lo < hi <= 1 for lo, hi in
                   zip(self.exit_thresholds, self.entry_thresholds)):
            raise ValueError("Require 0 <= exit < entry <= 1 at each boundary")
        if any(a >= b for seq in (self.entry_thresholds, self.exit_thresholds)
               for a, b in zip(seq, seq[1:])):
            raise ValueError("Thresholds must be strictly increasing")

    @property
    def history_rows(self):
        return max(self.percentile_window, self.trend_window,
                   self.momentum_window + 1, self.defensive_window + 1,
                   self.covariance_window + 1, self.equity_vol_window + 1)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Snapshot:
    date: date
    vix: float
    percentile: float
    trend_ok: bool
    momentum_ok: bool
    defensive_ok: dict
    covariance: np.ndarray  # annualized, ASSETS order, including BIL
    equity_vol: float


def make_snapshot(closes: pd.DataFrame, config: StrategyConfig) -> Optional[Snapshot]:
    """Build from completed session closes, without padding missing prices.

    Earlier BIL history may be missing before inception. It only needs the
    covariance window; SPY and VIX must have their own full longer windows.
    """
    if len(closes) < config.history_rows:
        return None
    required = (*ASSETS, "VIX")
    if not set(required).issubset(closes.columns):
        raise ValueError(f"Required close columns: {required}")
    if not closes.index.is_monotonic_increasing or closes.index.has_duplicates:
        raise ValueError("Close dates must be sorted and unique")
    data = closes.loc[:, required].iloc[-config.history_rows:].to_numpy(dtype=float)
    if not np.all(np.isfinite(data[-1])) or np.any(data[-1] <= 0):
        return None

    def complete(array):
        return np.all(np.isfinite(array)) and np.all(array > 0)

    spy = data[:, 0]
    vix = data[-config.percentile_window:, 4]
    risk_prices = data[-config.covariance_window - 1:, :4]
    spy_vol_prices = spy[-config.equity_vol_window - 1:]
    defensive_prices = data[-config.defensive_window - 1:, 1:3]
    if not all(complete(a) for a in (
            spy[-config.trend_window:], spy[-config.momentum_window - 1:],
            vix, risk_prices, spy_vol_prices, defensive_prices)):
        return None
    returns = risk_prices[1:] / risk_prices[:-1] - 1
    covariance = np.cov(returns, rowvar=False, ddof=1) * 252
    diagonal = np.diag(np.diag(covariance))
    alpha = config.covariance_shrinkage
    covariance = (1 - alpha) * covariance + alpha * diagonal
    spy_returns = spy_vol_prices[1:] / spy_vol_prices[:-1] - 1
    return Snapshot(
        date=pd.Timestamp(closes.index[-1]).date(),
        vix=float(vix[-1]),
        percentile=float(np.mean(vix[:-1] <= vix[-1])),
        trend_ok=bool(spy[-1] > np.mean(spy[-config.trend_window:])),
        momentum_ok=bool(spy[-1] > spy[-config.momentum_window - 1]),
        defensive_ok={s: bool(defensive_prices[-1, i] > defensive_prices[0, i])
                      for i, s in enumerate(("IEF", "GLD"))},
        covariance=covariance,
        equity_vol=float(np.std(spy_returns, ddof=1) * np.sqrt(252)),
    )


def estimated_vol(weights, covariance):
    w = np.array([weights.get(s, 0.0) for s in ASSETS], dtype=float)
    variance = float(w @ covariance @ w)
    if not np.isfinite(variance) or variance < -1e-12:
        raise ValueError("Invalid estimated portfolio variance")
    return float(np.sqrt(max(0.0, variance)))


def cap_portfolio_vol(weights, covariance, target):
    """Move risky weight to BIL using full covariance, never add leverage.

    Variance along the BIL-to-portfolio line is convex. Bisection finds the
    largest feasible risky allocation when the starting portfolio exceeds
    target. If even BIL exceeds target, the remaining allocation is USD cash.
    """
    if estimated_vol(weights, covariance) <= target:
        return dict(weights)
    bil_vol = float(np.sqrt(max(0, covariance[3, 3])))
    if bil_vol > target:
        return {"BIL": target / bil_vol}
    risky = {s: weights.get(s, 0.0) for s in RISKY}
    budget = sum(weights.values())

    def scaled(scale):
        result = {s: w * scale for s, w in risky.items()}
        result["BIL"] = budget - sum(result.values())
        return result

    low, high = 0.0, 1.0
    for _ in range(50):
        mid = (low + high) / 2
        if estimated_vol(scaled(mid), covariance) <= target:
            low = mid
        else:
            high = mid
    return scaled(low)


def build_targets(snapshot: Snapshot, regime: int, config: StrategyConfig):
    """Pure function: constructing weights never mutates the regime."""
    weights = {s: REGIME_WEIGHTS[regime].get(s, 0.0) for s in ASSETS}
    mode = config.equity_filter
    equity_ok = ((mode not in ("sma", "both") or snapshot.trend_ok)
                 and (mode not in ("momentum", "both") or snapshot.momentum_ok))
    if not equity_ok:
        weights["BIL"] += weights["SPY"]
        weights["SPY"] = 0.0
    if config.defensive_filter:
        for ticker in ("IEF", "GLD"):
            if not snapshot.defensive_ok[ticker]:
                weights["BIL"] += weights[ticker]
                weights[ticker] = 0.0
    if config.vol_model == "portfolio":
        weights = cap_portfolio_vol(weights, snapshot.covariance, config.target_vol)
    elif config.vol_model == "equity" and snapshot.equity_vol > 0:
        scaled = weights["SPY"] * min(1.0, config.target_vol / snapshot.equity_vol)
        weights["BIL"] += weights["SPY"] - scaled
        weights["SPY"] = scaled
    targets = {s: float(weights.get(s, 0.0)) * (1 - config.cash_buffer) for s in ASSETS}
    if any(not np.isfinite(w) or w < -1e-12 for w in targets.values()):
        raise ValueError("Invalid target weights")
    if sum(targets.values()) > 1 + 1e-10:
        raise ValueError("Unlevered target budget exceeded")
    return targets


@dataclass(frozen=True)
class Decision:
    signal_date: date
    execution_date: date
    regime: int
    targets: dict
    trade: bool
    reason: str
    target_vol: float
    current_vol: float


class SignalEngine:
    def __init__(self, config=None):
        self.config = config or StrategyConfig()
        self.regime = 0
        self.last_signal_date = None
        self.last_week = None
        self.last_month = None

    def decide(self, snapshot, execution_date, actual_weights, force=False):
        execution_date = pd.Timestamp(execution_date).date()
        if snapshot.date >= execution_date:
            raise ValueError("Signals must precede the execution session")
        if self.last_signal_date is not None and snapshot.date <= self.last_signal_date:
            return None  # one state update and order decision per completed session
        config = self.config
        iso = execution_date.isocalendar()
        week, month = (iso.year, iso.week), (execution_date.year, execution_date.month)
        initial = self.last_signal_date is None
        monthly, weekly = month != self.last_month, week != self.last_week
        regular = monthly or weekly or initial
        old = self.regime
        hard_panic = config.use_vix and snapshot.vix >= config.panic_vix
        if not config.use_vix:
            self.regime = 0
        elif hard_panic:
            self.regime = 3
        elif regular:
            if self.regime < 3 and snapshot.percentile >= config.entry_thresholds[self.regime]:
                self.regime += 1
            elif (self.regime > 0
                  and snapshot.percentile < config.exit_thresholds[self.regime - 1]
                  and (self.regime != 3 or snapshot.vix < config.panic_exit_vix)):
                self.regime -= 1
        self.last_signal_date = snapshot.date
        self.last_week, self.last_month = week, month
        targets = build_targets(snapshot, self.regime, config)
        differences = [abs(targets.get(s, 0) - actual_weights.get(s, 0)) for s in ASSETS]
        drift = max(differences)
        current_vol = estimated_vol(actual_weights, snapshot.covariance)
        target_vol = estimated_vol(targets, snapshot.covariance)
        guard_exit = any(actual_weights.get(s, 0) > config.minimum_trade_weight
                         and targets.get(s, 0) <= 1e-12 for s in RISKY)
        risk_cut = (config.vol_model == "portfolio"
                    and current_vol > config.target_vol * (1 + config.risk_breach_tolerance)
                    and target_vol < current_vol)
        reason = "hold"
        if initial:
            reason = "initial"
        elif hard_panic and (old != 3 or guard_exit):
            reason = "panic"
        elif force:
            reason = "retry"
        elif old != self.regime:
            reason = "regime"
        elif config.daily_risk_checks and guard_exit:
            reason = "guard-exit"
        elif config.daily_risk_checks and risk_cut:
            reason = "risk-cut"
        elif monthly:
            reason = "monthly"
        elif weekly and drift > config.drift_tolerance:
            reason = "weekly-drift"
        # Exits override the no-trade band, including small residual positions.
        needs_exit = any(actual_weights.get(s, 0) > 1e-8 and targets[s] <= 1e-12
                         for s in ASSETS)
        trade = reason != "hold" and (drift >= config.minimum_trade_weight or needs_exit)
        return Decision(snapshot.date, execution_date, self.regime, targets, trade,
                        reason, target_vol, current_vol)
