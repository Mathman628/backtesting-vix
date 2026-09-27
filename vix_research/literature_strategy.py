"""Explicit long-only adaptations for the literature study, not paper replications.

Uses the existing accounting engine. Production/QuantConnect defaults are intact.
All controlled variants share asset access, lagged signals and an execution policy.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

from .core import (ASSETS, RISKY, REGIME_WEIGHTS, StrategyConfig, Snapshot,
                   SignalEngine, Decision, cap_portfolio_vol, estimated_vol)
from .backtest import prepare_snapshots


@dataclass(frozen=True)
class ResearchConfig(StrategyConfig):
    signal: str = "absolute12"
    allocation: str = "static"  # static, regime, inverse_vol
    vix_overlay: bool = False
    adjustment_fraction: float = 1.0
    reentry_cap: float = 1.0  # total risky increase per routine rebalance
    trend_fraction: float = 1.0  # .2 implements an allocation-level 80/20 blend
    routine_band: float = .005

    def __post_init__(self):
        super().__post_init__()
        if self.signal not in ("none", "absolute12", "excess12", "ensemble"):
            raise ValueError("Unknown research signal")
        if self.allocation not in ("static", "regime", "inverse_vol"):
            raise ValueError("Unknown allocation")
        if not isinstance(self.vix_overlay, bool):
            raise ValueError("vix_overlay must be boolean")
        for key in ("adjustment_fraction", "reentry_cap", "trend_fraction"):
            if not np.isfinite(getattr(self, key)) or not 0 < getattr(self, key) <= 1:
                raise ValueError(f"Invalid {key}")
        if not np.isfinite(self.routine_band) or not 0 <= self.routine_band < 1:
            raise ValueError("Invalid routine_band")
        if self.history_rows < 253:
            raise ValueError("Research signals require at least 253 history rows")


@dataclass(frozen=True)
class ResearchSnapshot(Snapshot):
    scores: dict


def research_snapshots(data, config, base=None):
    """Snapshot at execution index i uses prices/rates only through i-1."""
    base = prepare_snapshots(data, config) if base is None else base
    rates = data.risk_free_returns().to_numpy()
    cash_index = np.cumprod(1 + rates)
    prices = data.closes.loc[:, RISKY].to_numpy()
    horizons = (21, 63, 252) if config.signal == "ensemble" else (252,)
    output = []
    for i, snapshot in enumerate(base):
        if snapshot is None:
            output.append(None)
            continue
        scores = np.ones(3) if config.signal == "none" else np.zeros(3)
        if config.signal != "none":
            for h in horizons:
                start, stop = i - 1 - h, i - 1
                growth = prices[stop] / prices[start]
                hurdle = cash_index[stop] / cash_index[start] if config.signal in ("excess12", "ensemble") else 1.0
                if not np.isfinite(growth).all():
                    raise ValueError("Insufficient risky-asset history")
                scores += (growth > hurdle) / len(horizons)
        output.append(ResearchSnapshot(**snapshot.__dict__, scores=dict(zip(RISKY, scores))))
    return output


class ResearchEngine:
    def __init__(self, config):
        self.config = config
        self.regimes = SignalEngine(replace(config, use_vix=config.vix_overlay or config.allocation == "regime"))
        self.last_month = None
        self.last_week = None

    def decide(self, snapshot, execution_date, actual_weights):
        cfg = self.config
        day = pd.Timestamp(execution_date)
        initial = self.last_month is None
        monthly = (day.year, day.month) != self.last_month
        week = tuple(day.isocalendar()[:2])
        weekly = week != self.last_week
        self.last_month, self.last_week = (day.year, day.month), week
        state = self.regimes.decide(snapshot, day, actual_weights)
        if state is None:
            return None
        if cfg.allocation == "regime":
            base = dict(REGIME_WEIGHTS[state.regime])
        elif cfg.allocation == "inverse_vol":
            inv = 1 / np.maximum(np.sqrt(np.diag(snapshot.covariance)[:3]), .01)
            base = dict(zip(RISKY, inv / inv.sum()))
        else:
            base = dict(zip(RISKY, (.5, .25, .25)))
        multiplier = (1., .75, .5, 0.)[state.regime] if cfg.vix_overlay else 1.
        weights = {s: base.get(s, 0.) * ((1 - cfg.trend_fraction) + cfg.trend_fraction * snapshot.scores[s]) * multiplier
                   for s in RISKY}
        weights["BIL"] = 1 - sum(weights.values())
        if cfg.vol_model == "portfolio":
            weights = cap_portfolio_vol(weights, snapshot.covariance, cfg.target_vol)
        targets = {s: weights.get(s, 0.) * (1 - cfg.cash_buffer) for s in ASSETS}
        current_vol = estimated_vol(actual_weights, snapshot.covariance)
        desired_vol = estimated_vol(targets, snapshot.covariance)
        # Identical safety policy across the factorial variants. VIX changes only
        # desired exposures; it does not add a separate rebalance calendar.
        exit_needed = any(actual_weights.get(s, 0.) > cfg.minimum_trade_weight and targets[s] <= 1e-12 for s in RISKY)
        breach = (cfg.vol_model == "portfolio" and current_vol > cfg.target_vol * 1.2 and desired_vol < current_vol)
        emergency = exit_needed or breach
        drift = max(abs(targets[s] - actual_weights.get(s, 0.)) for s in ASSETS)
        reason = ("initial" if initial else "guard-exit" if exit_needed else "risk-cut" if breach
                  else "monthly" if monthly else "weekly-drift" if weekly and drift > cfg.drift_tolerance else "hold")
        trade = reason != "hold" and (emergency or drift >= cfg.routine_band)
        if trade and not emergency and not initial:
            targets = {s: actual_weights.get(s, 0.) + cfg.adjustment_fraction * (targets[s] - actual_weights.get(s, 0.)) for s in ASSETS}
            increases = {s: max(0., targets[s] - actual_weights.get(s, 0.)) for s in RISKY}
            total = sum(increases.values())
            if total > cfg.reentry_cap:
                freed = 0.
                for s, amount in increases.items():
                    cut = amount * (1 - cfg.reentry_cap / total)
                    targets[s] -= cut
                    freed += cut
                targets["BIL"] += freed
        if sum(targets.values()) > 1 + 1e-9 or min(targets.values()) < -1e-9:
            raise ArithmeticError("Invalid research weights")
        return Decision(snapshot.date, day.date(), state.regime if cfg.vix_overlay or cfg.allocation == "regime" else -1,
                        targets, trade, reason, estimated_vol(targets, snapshot.covariance), current_vol)


def variants():
    """Frozen, small hypothesis family; no grid search or fitted signal weights."""
    models = {}
    for trend in (False, True):
        for vol in (False, True):
            for vix in (False, True):
                name = f"factor_t{int(trend)}_r{int(vol)}_v{int(vix)}"
                models[name] = ResearchConfig(signal="absolute12" if trend else "none",
                    vol_model="portfolio" if vol else "none", vix_overlay=vix)
    base = models["factor_t1_r1_v0"]
    models.update({
        "excess12": replace(base, signal="excess12"),
        "ensemble": replace(base, signal="ensemble"),
        "ensemble_band": replace(base, signal="ensemble", routine_band=.03, drift_tolerance=.08),
        "ensemble_partial": replace(base, signal="ensemble", adjustment_fraction=.5),
        "ensemble_reentry": replace(base, signal="ensemble", reentry_cap=.20),
        "blend_80_20": replace(base, signal="ensemble", trend_fraction=.20),
        "inverse_vol_ensemble": replace(base, signal="ensemble", allocation="inverse_vol"),
        "regime_ensemble": replace(base, signal="ensemble", allocation="regime"),
    })
    return models
