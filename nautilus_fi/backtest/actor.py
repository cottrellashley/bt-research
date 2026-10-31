"""Environment actor: the only bridge between the engine's data stream and the strategy's restricted
context. It publishes each datum into the ``DecisionContext`` when (and only when) it becomes
available, refreshes the visible portfolio state before each decision, binds the shared clock, and
publishes daily portfolio snapshots."""

from __future__ import annotations

import pandas as pd
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import DataType

from nautilus_fi.backtest.context import FixedIncomeContext
from nautilus_fi.core.context import CashState, VisibleOrder, VisiblePosition
from nautilus_fi.data.types import (
    BondMark,
    CorporateAction,
    FactorUpdate,
    IndexFixing,
    LiquiditySnapshot,
    PortfolioSnapshot,
    RatingUpdate,
    StressIndex,
    wrap,
)
from nautilus_fi.execution.windows import NS_PER_DAY
from nautilus_fi.instruments.bond import units_to_face
from nautilus_fi.instruments.curves import ZeroCurve


def _num(x) -> float:
    if x is None:
        return 0.0
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)


class FixedIncomeEnvironmentActor(DataActor):
    SNAPSHOT_TIMER = "fi-snapshot"
    REFRESH_TIMER = "fi-refresh"

    def __init__(self, ctx: FixedIncomeContext) -> None:
        super().__init__(DataActorConfig(log_events=False, log_commands=False))
        self.ctx = ctx
        self._published: set = set()
        self._sovereign_names = {c.name for c in ctx.env.curves.values()} | {ctx.env.curve_name}

    def on_start(self) -> None:
        self.ctx.fill_model.clock = self.clock
        if self.ctx.overlay_fill_model is not None:
            self.ctx.overlay_fill_model.clock = self.clock
        for cls in (BondMark, LiquiditySnapshot, RatingUpdate, IndexFixing, FactorUpdate, CorporateAction, StressIndex):
            self.subscribe_data(DataType(cls.__name__))
        self.subscribe_data(DataType("YieldCurve"))
        env = self.ctx.env
        now = self.clock.timestamp_ns()
        snap = env.schedule.snapshot_ns(env.start_date)
        while snap < now:
            snap += NS_PER_DAY
        self.clock.set_timer_ns(self.SNAPSHOT_TIMER, NS_PER_DAY, start_time_ns=snap, callback=None, fire_immediately=True)
        refresh = env.schedule.decision_ns(env.start_date) - 60 * 1_000_000_000
        while refresh < now:
            refresh += NS_PER_DAY
        self.clock.set_timer_ns(self.REFRESH_TIMER, NS_PER_DAY, start_time_ns=refresh, callback=None, fire_immediately=True)
        self.ctx.decision._set_now(now)

    # ------------------------------------------------------------------ data -> context
    def on_data(self, data) -> None:
        payload = getattr(data, "data", data)
        ts = int(getattr(data, "ts_init", getattr(payload, "ts_init", self.clock.timestamp_ns())))
        dc = self.ctx.decision
        if ts > dc.now_ns:
            dc._set_now(ts)
        name = type(payload).__name__
        if name == "BondMark":
            dc._publish("mark", payload.instrument_id.symbol.value, payload, ts)
        elif name == "LiquiditySnapshot":
            dc._publish("liquidity", payload.instrument_id.symbol.value, payload, ts)
        elif name == "RatingUpdate":
            dc._publish("rating", payload.instrument_id.symbol.value, int(payload.score), ts)
        elif name == "IndexFixing":
            dc._publish("fixing", payload.index_name, float(payload.rate), ts)
        elif name == "FactorUpdate":
            dc._publish("factor", payload.instrument_id.symbol.value, float(payload.factor), ts)
        elif name == "StressIndex":
            dc._publish("stress", "STRESS", float(payload.value), ts)
        elif name == "CorporateAction":
            dc._publish("action", payload.instrument_id.symbol.value, payload, ts)
        elif name == "YieldCurve":
            curve = ZeroCurve(dc.today, list(payload.tenors), list(payload.interest_rates), payload.curve_name)
            dc._publish("curve", payload.curve_name, curve, ts)
            if payload.curve_name in self._sovereign_names:          # only the sovereign curve answers to GOVT
                dc._publish("curve", "GOVT", curve, ts)

    # ------------------------------------------------------------------ timers
    def on_time_event(self, event) -> None:
        dc = self.ctx.decision
        if event.ts_event > dc.now_ns:
            dc._set_now(event.ts_event)
        d = pd.Timestamp(event.ts_event, tz="UTC").date()
        if event.name == self.REFRESH_TIMER:
            self._publish_fields(d)
            self._refresh_portfolio(d)
        elif event.name == self.SNAPSHOT_TIMER:
            self._publish_snapshot(event, d)

    def _publish_fields(self, d) -> None:
        """Generic fields reach the context as whole cross-sections, by reference, once per day: the
        row known at this decision time (the field's availability lag in trading days applied)."""
        panel = self.ctx.env.field_panel
        if panel is None or not panel.fields:
            return
        i = panel.date_pos(d)
        if i < 0:
            return
        sched = self.ctx.env.schedule
        dc = self.ctx.decision
        for name in panel.fields:
            j = i - panel.lags.get(name, 0)
            if j < 0:
                continue
            avail = sched.publish_ns(panel.dates[j].date())
            if avail <= dc.now_ns:
                dc._publish_row(name, panel.ids, panel.values[name][j], avail)

    def _refresh_portfolio(self, d) -> None:
        ledger = self.ctx.ledger
        env = self.ctx.env
        positions: dict[str, VisiblePosition] = {}
        for pos in self.cache.positions_open(None, None, None, None, None):
            symbol = pos.instrument_id.symbol.value
            bond = env.bonds.get(symbol)
            if bond is None:
                continue
            units = _num(pos.signed_qty)
            factor = env.factor(symbol, d)
            face = units_to_face(units) * factor
            unsettled_units = ledger.queue.unsettled_quantity(symbol)
            positions[symbol] = VisiblePosition(symbol, face, bond.currency, face - units_to_face(unsettled_units) * factor, ledger.policy.book.cost(symbol), ledger.policy.book.carrying(symbol), ledger.policy.accrued.get((symbol, bond.currency), 0.0), factor)
        orders: dict[str, VisibleOrder] = {}
        for o in self.cache.orders_open(None, None, None, None):
            symbol = o.instrument_id.symbol.value
            if symbol not in env.bonds:
                continue
            factor = env.factor(symbol, d)
            orders[str(o.client_order_id)] = VisibleOrder(str(o.client_order_id), symbol, "BUY" if str(o.side).endswith("BUY") else "SELL", units_to_face(_num(o.quantity)) * factor, units_to_face(_num(o.leaves_qty)) * factor, _num(o.price) / factor if getattr(o, "price", None) is not None and factor else None, str(o.status), int(o.ts_init), int(o.expire_time_ns) if getattr(o, "expire_time_ns", None) else None)
        cash: dict[str, CashState] = {}
        currencies = {self.ctx.base_currency} | {b.currency for b in env.bonds.values()}
        for ccy in currencies:
            s = ledger.cash_view.state(ccy)
            cash[ccy] = CashState(ccy, s["settled"], s["unsettled_receivable"], s["unsettled_payable"], s["reserved"], s["collateral_posted"])
        nav = float(ledger.snapshots[-1]["nav"]) if ledger.snapshots else ledger.ledger.nav(ledger._fx(d)).amount
        self.ctx.decision._set_portfolio(positions, orders, cash, nav)
        if ledger.funding_policy.repo_for_leverage:
            self.ctx.decision._publish_slice("funding.repo_capacity", {self.ctx.base_currency: float(ledger.repo_capacity_cached)}, self.ctx.decision.now_ns)
        link = ledger.margin
        if link is not None:
            now = self.ctx.decision.now_ns
            book = link.s.book
            self.ctx.decision._publish_slice("overlay.units", link.units(), now)
            self.ctx.decision._publish_slice("overlay.price", {k: v for k in book.contracts if (v := book.price(k, d)) is not None}, now)
            self.ctx.decision._publish_slice("overlay.margin", {k: float(v) for k, v in link.last.items()}, now)
        self.ctx.decision._set_funding(ledger.funding_shortfall)

    def _publish_snapshot(self, event, d) -> None:
        if not self.ctx.ledger.snapshots:
            return
        snap = self.ctx.ledger.snapshots[-1]
        if pd.Timestamp(snap["date"]).date() != d or d in self._published:
            return
        self._published.add(d)
        payload = PortfolioSnapshot(ts_event=event.ts_event, ts_init=event.ts_event, nav=float(snap["nav"]), cash=float(snap["cash"]), market_value=float(snap["market_value"]), accrued=float(snap["accrued"]), unsettled_cash=float(snap["unsettled_cash"]), duration=float(snap["duration"]), spread_duration=float(snap["spread_duration"]), dts=float(snap["dts"]), n_positions=int(snap["n_positions"]), day_return=float(snap["day_return"]))
        self.publish_data(DataType("PortfolioSnapshot"), wrap(payload))


__all__ = ["FixedIncomeEnvironmentActor"]
