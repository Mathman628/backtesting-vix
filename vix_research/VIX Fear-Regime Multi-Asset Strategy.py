# GENERATED from vix_research/core.py and qc_adapter.py.
# Rebuild with: python -m vix_research build-qc
# Gold is a commodity allocation; IEF and BIL are Treasury allocations.
# Local research and this adapter share signals, but differ in execution.

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


"""QuantConnect adapter. build-qc bundles this with core.py into one file.

This module requires LEAN. Local research imports core.py, not this adapter.
"""
from AlgorithmImports import *
from datetime import datetime, timedelta
import json
import pandas as pd



class VixFearRegimeStrategy(QCAlgorithm):
    def initialize(self):
        self.set_time_zone(TimeZones.NEW_YORK)
        start = datetime.fromisoformat(self.get_parameter("start_date") or "2007-06-01")
        self.set_start_date(start.year, start.month, start.day)
        end_parameter = self.get_parameter("end_date")
        if end_parameter:
            end = datetime.fromisoformat(end_parameter)
            self.set_end_date(end.year, end.month, end.day)
        self.set_cash(100_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        parameters = json.loads(self.get_parameter("strategy_config") or "{}")
        self._config = StrategyConfig(**parameters)
        self._engine = SignalEngine(self._config)
        self._slippage_bps = float(self.get_parameter("slippage_bps") or "5")
        if not 0 <= self._slippage_bps <= 100:
            raise ValueError("slippage_bps must be between 0 and 100")
        self.settings.daily_precise_end_time = True
        # The shared target builder already reserves cash; do not reserve twice.
        self.settings.free_portfolio_value_percentage = 0
        self._symbols = {}
        for ticker in ASSETS:
            security = self.add_equity(ticker, Resolution.MINUTE,
                                       data_normalization_mode=DataNormalizationMode.ADJUSTED)
            security.set_slippage_model(ConstantSlippageModel(self._slippage_bps / 10_000))
            self._symbols[ticker] = security.symbol
        self._vix = self.add_index("VIX", Resolution.DAILY).symbol
        self.set_benchmark(self._symbols["SPY"])
        self._audit, self._fills = [], []
        self._retry = False
        self._last_callback_date = None
        self._last_skip_reason = None
        # One callback handles daily safety, weekly regime evaluation and monthly
        # rebalancing. At 09:35, minute subscriptions supply executable prices.
        self.schedule.on(self.date_rules.every_day(self._symbols["SPY"]),
                         self.time_rules.after_market_open(self._symbols["SPY"], 5),
                         self._daily_decision)

    def _completed_closes(self):
        symbols = {**self._symbols, "VIX": self._vix}
        # Re-request adjusted history so corporate actions cannot mix adjustment
        # bases in an incrementally maintained price buffer. This costs runtime
        # but is deliberately simple and reproducible for four ETFs.
        history = self.history(list(symbols.values()), self._config.history_rows + 5,
                               Resolution.DAILY, fill_forward=False)
        if history.empty:
            return None
        series = {}
        today = pd.Timestamp(self.time.date())
        for ticker, symbol in symbols.items():
            try:
                frame = history.loc[symbol]
            except KeyError:
                return None
            stamps = pd.DatetimeIndex(frame.index)
            if stamps.tz is not None:
                stamps = stamps.tz_localize(None)
            # DailyPreciseEndTime makes the timestamp the actual session close.
            values = pd.Series(frame["close"].to_numpy(dtype=float), index=stamps.normalize())
            values = values[values.index < today]
            if values.index.has_duplicates:
                raise ValueError(f"Duplicate daily history for {ticker}")
            series[ticker] = values
        closes = pd.DataFrame(series).sort_index()
        spy_dates = series["SPY"].index
        closes = closes.reindex(spy_dates)
        if closes.empty:
            return None
        expected = self.time - timedelta(days=1)
        hours = self.securities[self._symbols["SPY"]].exchange.hours
        while not hours.is_date_open(expected):
            expected -= timedelta(days=1)
        if closes.index[-1].date() != expected.date():
            return None
        return closes

    def _skip(self, reason):
        # Avoid thousands of identical messages during the BIL inception window.
        if reason != self._last_skip_reason:
            self.log(f"{self.time} skipped: {reason}")
            self._last_skip_reason = reason

    def _daily_decision(self):
        if self.is_warming_up or self._last_callback_date == self.time.date():
            return
        self._last_callback_date = self.time.date()
        outstanding = self.transactions.get_open_orders()
        if outstanding:
            # Never stack a new rebalance on unfilled orders. Request cancellation
            # of orders left from earlier sessions, then retry on the next session.
            for order in outstanding:
                if order.time.date() < self.utc_time.date():
                    self.transactions.cancel_order(order.id, "Stale rebalance; recalculate next session")
            self._retry = True
            self._skip("outstanding orders; awaiting fills/cancellation")
            return
        for ticker, symbol in self._symbols.items():
            security = self.securities[symbol]
            last = security.get_last_data()
            if (not security.has_data or security.price <= 0 or last is None
                    or self.time - last.end_time > timedelta(minutes=2)):
                self._skip(f"missing/stale execution price for {ticker}")
                return
        closes = self._completed_closes()
        snapshot = make_snapshot(closes, self._config) if closes is not None else None
        if snapshot is None:
            self._skip("incomplete/stale daily history or indicator warm-up")
            return
        self._last_skip_reason = None
        nav = float(self.portfolio.total_portfolio_value)
        if nav <= 0:
            self._skip("nonpositive portfolio value")
            return
        actual = {ticker: float(self.portfolio[symbol].holdings_value) / nav
                  for ticker, symbol in self._symbols.items()}
        decision = self._engine.decide(snapshot, self.time.date(), actual, force=self._retry)
        if decision is None:
            return
        self._audit.append({"decision_time": str(self.time), "signal_date": str(snapshot.date),
                            "nav_before_orders": nav, "regime": REGIMES[decision.regime],
                            "vix": snapshot.vix, "percentile": snapshot.percentile,
                            "estimated_target_vol": decision.target_vol,
                            "estimated_actual_vol": decision.current_vol,
                            "trade": decision.trade, "reason": decision.reason,
                            **{f"actual_{s}": actual[s] for s in ASSETS},
                            **{f"target_{s}": decision.targets[s] for s in ASSETS}})
        self.plot("Regime", "index", decision.regime)
        self.plot("VIX", "level", snapshot.vix)
        self.plot("VIX percentile", "percentile", snapshot.percentile)
        self.plot("Risk", "estimated target vol", decision.target_vol)
        self.plot("Risk", "estimated actual vol", decision.current_vol)
        for ticker in ASSETS:
            # Include zeros so exited positions disappear from the chart.
            self.plot("Target weights", ticker, decision.targets[ticker])
            self.plot("Actual weights at decision", ticker, actual[ticker])
        if not decision.trade:
            return
        self._retry = False  # order callbacks can set this back to True
        targets = [PortfolioTarget(self._symbols[ticker], decision.targets[ticker]) for ticker in ASSETS]
        tag = f"{decision.reason}; signal={snapshot.date}; {REGIMES[decision.regime]}"
        self.set_holdings(targets, liquidate_existing_holdings=True, tag=tag)
        self.log(f"{self.time} {tag}: " + ", ".join(f"{s} {decision.targets[s]:.1%}" for s in ASSETS))

    def on_order_event(self, event):
        self._fills.append({"event_time": str(self.time), "order_id": event.order_id,
                            "symbol": str(event.symbol), "status": str(event.status),
                            "fill_quantity": float(event.fill_quantity),
                            "fill_price": float(event.fill_price), "message": event.message})
        if event.status in (OrderStatus.INVALID, OrderStatus.CANCELED):
            self._retry = True
            self.log(f"Order {event.order_id}: {event.status} {event.message}; retry with fresh signals")

    def on_end_of_algorithm(self):
        if not self._audit:
            self.log("No ready decisions. Inspect start date, required history, and data availability.")
            return
        prefix = f"{self.project_id}/{self.algorithm_id}/vix-regime"
        try:
            saved_audit = self.object_store.save(prefix + "-decisions.csv", pd.DataFrame(self._audit).to_csv(index=False))
            saved_fills = self.object_store.save(prefix + "-orders.csv", pd.DataFrame(self._fills).to_csv(index=False))
            saved_config = self.object_store.save(prefix + "-config.json", json.dumps(self._config.to_dict()))
            self.log(f"Audit export {prefix}: decisions={saved_audit}, orders={saved_fills}, config={saved_config}")
        except Exception as error:
            self.log(f"Audit export unavailable: {error}")
