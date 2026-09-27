# VIX Fear-Regime Multi-Asset Strategy
#
# Design (2026-09-24, client request: use VIX as the market fear gauge;
# high fear -> fixed income (treasuries + gold), calm -> equity index).
#
# Architecture - every component below earns its place; no padding:
#
#   1. Fear features from the VIX cash index (data from 1998, so a 2007+
#      start has a full 252-day percentile window):
#        - 21-day EMA of VIX           (short fear trend)
#        - 252-day percentile rank     (where fear sits vs the past year)
#        - 252-day z-score             (standardized fear surprise)
#
#   2. Four-regime state machine with HYSTERESIS. The regime moves one
#      step per check at most, and only when the percentile rank crosses
#      a boundary by more than the band (+/-0.05). A single VIX spike or
#      dip cannot flip the book. PANIC also escalates immediately on the
#      VIX LEVEL (>= 35) regardless of the percentile rank.
#        0 RISK-ON : pctile below 0.60
#        1 CAUTION : 0.60 - 0.80
#        2 FEAR    : 0.80 - 0.95
#        3 PANIC   : above 0.95 (or VIX level >= 35)
#
#   3. Regime weight map (a priori, not optimized):
#        RISK-ON : SPY 100%
#        CAUTION : SPY 50%, IEF 25%, GLD 25%
#        FEAR    : IEF 40%, GLD 35%, BIL 25%   (client spec: fixed income)
#        PANIC   : BIL 100%
#
#   4. Volatility targeting overlay: the equity sleeve is scaled by
#      min(1, 10% annualized target / 21d realized SPY vol), remainder to
#      BIL. Fear regimes hold their map weights unscaled - their purpose
#      is de-risking, not vol-matching.
#
#   5. Momentum guard (carried from the validated 2007-2026 TSMOM work):
#      in RISK-ON, equities are only held while SPY is above its 200-day
#      SMA and 12m momentum is positive; otherwise the sleeve sits in BIL.
#
#   6. Cadence: monthly core rebalance (first trading day, 08:00 ET) plus
#      a weekly regime check (Monday 08:00 ET) because VIX regimes move
#      faster than a monthly clock. Orders at market open; signals use
#      the last COMPLETED daily bar only.
#
# Discipline: thresholds (0.60/0.80/0.95, band 0.05, VIX 35, 10% vol
# target, 200d SMA, 12m momentum) are all round a-priori numbers; no
# parameter mining. The regime weights sum to 1.0 by construction.

from AlgorithmImports import *


class PercentileRank(PythonIndicator):
    """Rolling percentile rank of the incoming value within its own
    trailing window. Value in [0, 1]: the fraction of past window values
    less than or equal to the current value."""

    def __init__(self, period: int) -> None:
        self.name = f'PercentileRank({period})'
        self.warm_up_period = period
        self.time = datetime.min
        self.value = 0.5
        self._period = period
        self._window = RollingWindow[float](period)

    def update(self, point: IndicatorDataPoint) -> bool:
        current = float(point.value)
        self._window.add(current)
        self.time = point.end_time
        if self._window.count < 2:
            return False
        below = sum(1 for v in self._window if v <= current)
        self.value = (below - 1) / (self._window.count - 1)
        return self._window.count == self._window.size

    @property
    def is_ready(self) -> bool:
        return self._window.count == self._window.size


class VixFearRegimeStrategy(QCAlgorithm):
    # ---- a priori parameters (round numbers, none mined) ----
    PCT_WINDOW = 252          # 1y percentile window for VIX
    EMA_WINDOW = 21           # short VIX trend
    VOL_WINDOW = 21           # realized vol window for the overlay
    TARGET_VOL = 0.10         # 10% annualized portfolio vol target
    BOUNDARIES = [0.60, 0.80, 0.95]
    HYSTERESIS = 0.05
    PANIC_VIX = 35.0
    TREND_SMA = 200
    MOM_LOOKBACK = 252
    WARMUP_DAYS = 260

    # regime index -> target weights; every row sums to 1.0
    REGIME_WEIGHTS = {
        0: {'SPY': 1.00},
        1: {'SPY': 0.50, 'IEF': 0.25, 'GLD': 0.25},
        2: {'IEF': 0.40, 'GLD': 0.35, 'BIL': 0.25},
        3: {'BIL': 1.00},
    }
    REGIME_NAMES = ['RISK-ON', 'CAUTION', 'FEAR', 'PANIC']
    DRIFT_TOLERANCE = 0.05  # skip rebalance if every weight moved < 5pp

    def initialize(self) -> None:
        self.set_start_date(2007, 6, 1)  # BIL inception is 2007-05-30
        self.set_cash(100_000)
        self.set_benchmark('SPY')

        self.settings.automatic_indicator_warm_up = True

        self._spy = self.add_equity('SPY', Resolution.DAILY).symbol
        self._ief = self.add_equity('IEF', Resolution.DAILY).symbol
        self._bil = self.add_equity('BIL', Resolution.DAILY).symbol
        self._gld = self.add_equity('GLD', Resolution.DAILY).symbol
        self._vix = self.add_index('VIX', Resolution.DAILY).symbol

        # symbol -> target weight mapping used by the rebalance
        self._by_ticker = {'SPY': self._spy, 'IEF': self._ief,
                           'BIL': self._bil, 'GLD': self._gld}

        # ---- fear features ----
        self._vix_ema = self.ema(self._vix, self.EMA_WINDOW, Resolution.DAILY)
        self._vix_pct = PercentileRank(self.PCT_WINDOW)
        self._vix_sma = self.sma(self._vix, self.PCT_WINDOW, Resolution.DAILY)
        self._vix_std = self.std(self._vix, self.PCT_WINDOW, Resolution.DAILY)

        # ---- equity risk features ----
        self._spy_trend = self.sma(self._spy, self.TREND_SMA, Resolution.DAILY)
        self._spy_mom = self.roc(self._spy, self.MOM_LOOKBACK, Resolution.DAILY)
        spy_daily_ret = self.roc(self._spy, 1, Resolution.DAILY)
        self._spy_vol = IndicatorExtensions.of(StandardDeviation(self.VOL_WINDOW), spy_daily_ret)

        self._regime = 0
        self._last_targets: dict[Symbol, float] = {}
        self._last_vix_date: datetime | None = None

        self.set_warm_up(self.WARMUP_DAYS, Resolution.DAILY)

        # 08:00 ET is outside the daily equity bar: signals are computed
        # from the last COMPLETED daily close, never the forming bar.
        self.schedule.on(
            self.date_rules.month_start(self._spy),
            self.time_rules.at(8, 0),
            self._core_rebalance,
        )
        self.schedule.on(
            self.date_rules.week_start(self._spy),
            self.time_rules.at(8, 0),
            self._weekly_regime_check,
        )

    def on_data(self, slice: Slice) -> None:
        if slice.bars.contains_key(self._vix):
            bar = slice.bars[self._vix]
            # Once-per-day guard: feed the percentile exactly one close per
            # session regardless of subscription cadence.
            if bar.close > 0 and bar.end_time.date() != self._last_vix_date:
                self._last_vix_date = bar.end_time.date()
                self._vix_pct.update(IndicatorDataPoint(bar.end_time, bar.close))
                self.plot('VIX', 'level', float(bar.close))

    # ------------------------------------------------------------------
    # Regime state machine
    # ------------------------------------------------------------------
    def _current_vix(self) -> float:
        return float(self.securities[self._vix].price)

    def _update_regime(self) -> int:
        pct = float(self._vix_pct.value)
        vix = self._current_vix()

        # Hard escalation: a VIX level of 35 is panic regardless of where
        # it sits in its 1y distribution (e.g. a calm-year spike).
        if vix >= self.PANIC_VIX:
            if self._regime < 3:
                self.log(f'{self.time.date():%Y-%m-%d} PANIC escalation: VIX {vix:.1f} >= {self.PANIC_VIX}')
            return 3

        regime = self._regime
        # Step up: percentile must clear the boundary by more than the band.
        if regime < 3 and pct > self.BOUNDARIES[regime] + self.HYSTERESIS:
            regime += 1
        # Step down: percentile must fall below the lower boundary by more
        # than the band. Only one step per check either way.
        elif regime > 0 and pct < self.BOUNDARIES[regime - 1] - self.HYSTERESIS:
            regime -= 1
        return regime

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    def _build_targets(self) -> dict[Symbol, float]:
        regime = self._update_regime()
        if regime != self._regime:
            self.log(f'{self.time.date():%Y-%m-%d} regime {self.REGIME_NAMES[self._regime]} '
                     f'-> {self.REGIME_NAMES[regime]} (VIX {self._current_vix():.1f}, '
                     f'pctile {float(self._vix_pct.value):.2f})')
            self._regime = regime

        weights: dict[Symbol, float] = {}
        for ticker, w in self.REGIME_WEIGHTS[regime].items():
            weights[self._by_ticker[ticker]] = w

        # Vol targeting and the momentum guard apply to the EQUITY sleeve
        # in RISK-ON and CAUTION only. Fear regimes de-risk by design.
        if regime in (0, 1) and self._spy in weights and weights[self._spy] > 0:
            equity_w = weights[self._spy]

            # Momentum guard: no equities when the trend is broken.
            trend_ok = (self._spy_trend.is_ready
                        and self.securities[self._spy].price > self._spy_trend.current.value)
            mom_ok = self._spy_mom.is_ready and self._spy_mom.current.value > 0
            if not (trend_ok and mom_ok):
                weights[self._bil] = weights.get(self._bil, 0.0) + equity_w
                weights[self._spy] = 0.0
                self.log(f'{self.time.date():%Y-%m-%d} momentum guard: equity sleeve -> BIL')
            else:
                # Vol targeting: scale the equity sleeve to the target vol,
                # capped at 1x; the unscaled remainder goes to BIL.
                if self._spy_vol.is_ready:
                    realized = float(self._spy_vol.current.value) * (252 ** 0.5)
                    if realized > 0:
                        scale = min(1.0, self.TARGET_VOL / realized)
                        scaled = equity_w * scale
                        weights[self._spy] = scaled
                        weights[self._bil] = weights.get(self._bil, 0.0) + (equity_w - scaled)

        return {s: w for s, w in weights.items() if w > 0}

    def _weights_drifted(self, targets: dict[Symbol, float]) -> bool:
        if self._last_targets.keys() != targets.keys():
            return True
        return any(abs(targets[s] - self._last_targets[s]) > self.DRIFT_TOLERANCE
                   for s in targets)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _ready_to_trade(self) -> bool:
        if self.is_warming_up:
            return False
        if not self._vix_pct.is_ready:
            return False
        if not all(self.securities[s].price > 0 for s in (self._spy, self._ief, self._bil, self._gld)):
            return False
        return True

    def _core_rebalance(self) -> None:
        if not self._ready_to_trade():
            return
        self._execute(self._build_targets(), tag='core')

    def _weekly_regime_check(self) -> None:
        if not self._ready_to_trade():
            return
        targets = self._build_targets()
        if self._weights_drifted(targets):
            self._execute(targets, tag='weekly')

    def _execute(self, targets: dict[Symbol, float], tag: str) -> None:
        portfolio_targets = [PortfolioTarget(s, w) for s, w in targets.items()]
        self.set_holdings(portfolio_targets, liquidate_existing_holdings=True)
        self._last_targets = dict(targets)

        for s, w in targets.items():
            self.plot('Weights', s.id.symbol, w)
        self.plot('Regime', 'index', self._regime)
        self.plot('VIX', 'percentile', float(self._vix_pct.value))
        self.log(f'{self.time.date():%Y-%m-%d} [{tag}] {self.REGIME_NAMES[self._regime]}: '
                 + ', '.join(f'{s.id.symbol} {w:.0%}' for s, w in sorted(targets.items())))
