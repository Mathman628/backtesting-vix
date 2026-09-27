"""QuantConnect adapter. build-qc bundles this with core.py into one file.

This module requires LEAN. Local research imports core.py, not this adapter.
"""
from AlgorithmImports import *
from datetime import datetime, timedelta
import json
import pandas as pd

from vix_research.core import ASSETS, REGIMES, StrategyConfig, SignalEngine, make_snapshot


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
