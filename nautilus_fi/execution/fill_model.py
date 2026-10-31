"""The Nautilus ``FillModel`` adapter around the order process simulator.

When an order is marketable against the venue's indicative (prior-close) quote, the matching engine
asks this model for a synthetic ``OrderBook``. The model runs the execution *process* for the order
on this day (eligibility, approval delay, accessible liquidity, dealer quotes, limit check, cost
decomposition) and returns the resulting price ladder. An empty book means no fill: market orders are
rejected, limit orders keep resting until the next window. Every evaluation is recorded as an order
event and a decision row so a reviewer can see why an order did or did not fill.

Funding is enforced here too: a buy whose settlement cost exceeds spendable (settled, unreserved)
cash under the funding policy rests instead of filling, so unsettled sale proceeds never fund a
purchase without an explicit bridge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from nautilus_trader.execution import FillModel
from nautilus_trader.model import BookOrder, BookType, OrderBook, OrderSide, Price, Quantity

from nautilus_fi.core.events import Event, EventLog, EventSource, EventType
from nautilus_fi.core.time import TimePoint
from nautilus_fi.execution.assumptions import NET_FULL, ExecutionAssumptions
from nautilus_fi.execution.costs import CostModel, ParametricCostModel
from nautilus_fi.execution.liquidity import LiquidityModel, VolumeGatedLiquidityModel
from nautilus_fi.execution.process import (
    ExecutionContext,
    OrderProcessSimulator,
    OrderRequest,
    OrderState,
    ProcessConfig,
    ProcessOutcome,
    ReferencePolicy,
)
from nautilus_fi.instruments.bond import face_to_units, units_to_face


def _num(x) -> float:
    if x is None:
        return 0.0
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)


@dataclass
class FillDecision:
    ts: pd.Timestamp
    client_order_id: str
    symbol: str
    side: str
    requested_face: float
    p_fill: float
    draw: float
    offered_face: float
    half_spread_bp: float
    mid: float
    reason: str
    protocol: str = "RFQ"
    state: str = ""
    venue: str = ""
    evidence: str = ""
    reference_source: str = ""
    cost_bp: float = 0.0
    n_quotes: int = 0


@dataclass
class NewIssueTerms:
    reoffer_price: float
    allocation_fraction: float = 0.3
    issue_date: date | None = None


def _protocol_from_tags(order) -> str:
    for t in order.tags or []:
        t = str(t)
        if t.startswith("protocol="):
            return t.split("=", 1)[1].upper()
        if t.upper() == "NEW_ISSUE":
            return "NEW_ISSUE"
    return "RFQ"


class FixedIncomeFillModel(FillModel):
    def __init__(self, env, assumptions: ExecutionAssumptions = NET_FULL, liquidity_model: LiquidityModel | None = None, cost_model: CostModel | None = None, process_config: ProcessConfig | None = None, new_issues: dict[str, NewIssueTerms] | None = None, event_log: EventLog | None = None, available_cash: Callable[[str], float] | None = None, borrow_available: Callable[[str, date], float] | None = None) -> None:
        super().__init__()
        self.env = env
        self.assumptions = assumptions
        cfg = process_config or ProcessConfig(reference_policy=ReferencePolicy(env.reference_policy) if hasattr(env, "reference_policy") else ReferencePolicy.PRIOR_CLOSE, allow_shorts=False)
        self.process = OrderProcessSimulator(liquidity_model or VolumeGatedLiquidityModel(), cost_model or ParametricCostModel(), assumptions, cfg)
        self.new_issues = new_issues or {}
        self.event_log = event_log
        self.available_cash = available_cash
        self.borrow_available = borrow_available
        self.clock = None
        self.cache = None
        self.decisions: list[FillDecision] = []
        self.outcomes: dict[str, ProcessOutcome] = {}
        self._seq = 0
        self.funding_rejects: list[dict] = []
        self._committed_base = 0.0          # dirty value of buys offered in the current window, base currency
        self.touched: set[str] = set()      # orders evaluated since the ledger's last pass (their fills are processed incrementally)

    # ---- Nautilus FillModel interface ---------------------------------------------------
    def is_limit_filled(self) -> bool:
        return True

    def is_slipped(self) -> bool:
        return False

    def fill_limit_inside_spread(self) -> bool:
        return False

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        book = OrderBook(instrument.id, BookType.L2_MBP)
        self.touched.add(str(order.client_order_id))
        if self.clock is None:
            return book
        now_ns = self.clock.timestamp_ns()
        schedule = self.env.schedule
        if order.ts_init >= now_ns or not schedule.is_window(now_ns):
            return book
        d = schedule.date_of(now_ns)
        symbol = instrument.id.symbol.value
        bond = self.env.bonds.get(symbol)
        if bond is None:
            return book
        side = "BUY" if order.side == OrderSide.BUY else "SELL"
        factor = self.env.factor(symbol, d)
        requested_face = units_to_face(_num(order.leaves_qty)) * factor
        protocol = _protocol_from_tags(order)
        ref = self.env.reference(symbol, d)
        stats = self.env.liquidity_stats(symbol, d, "window")
        an = self.env.analytics(symbol, self.env.prev_trading_date(d) or d)
        limit_clean = (_num(order.price) / factor) if getattr(order, "price", None) is not None and factor else None
        held = 0.0
        if self.cache is not None:
            for pos in self.cache.positions_open(None, instrument.id, None, None, None):
                held += _num(pos.signed_qty)
        is_short = side == "SELL" and held <= 0
        ni = self.new_issues.get(symbol)
        req = OrderRequest(str(order.client_order_id), symbol, side, requested_face, limit_clean, order.ts_init, schedule.date_of(order.ts_init), protocol, tuple(str(t) for t in (order.tags or [])), is_short=is_short)
        ctx = ExecutionContext(d, now_ns, ref[0] if ref else None, ref[3] if ref else "NONE", ref[1] if ref else None, ref[2] if ref else None, stats, self.env.stress(d), self.env.rating_score(symbol, d), an.mod_duration if an else None, factor,
                               borrow_available=self.borrow_available(symbol, d) if (self.borrow_available and is_short) else None,
                               new_issue_reoffer=ni.reoffer_price if (ni and (ni.issue_date or bond.issue_date) == d) else None, new_issue_allocation=ni.allocation_fraction if ni else 0.0)
        # funding gate: settled, unreserved cash must cover a buy's dirty cost (plus a tolerance)
        fx = (self.env.fx(bond.currency, d) or 1.0) if hasattr(self.env, "fx") else 1.0
        if side == "BUY" and self.available_cash is not None and ref is not None:
            cost = requested_face * (ref[0] * 1.01) / 100.0
            avail = self.available_cash(bond.currency) - self._committed_base / fx      # fills already offered this window count as spent
            if avail < cost:
                reason = f"settled cash {avail:,.0f} {bond.currency} below required {cost:,.0f}: order rests (settle-before-spend)"
                self.funding_rejects.append({"ts": pd.Timestamp(now_ns, tz="UTC"), "order_id": str(order.client_order_id), "symbol": symbol, "available": avail, "required": cost})
                self._record(now_ns, req, None, ref, reason)
                self._emit(now_ns, req, EventType.FUNDING, {"reason": reason, "available": avail, "required": cost})
                return book
        outcome = self.process.evaluate(bond, req, ctx, self.env.calendar)
        self.outcomes[str(order.client_order_id)] = outcome
        self._record(now_ns, req, outcome, ref, outcome.reason)
        for st in outcome.transitions:
            et = {OrderState.CREATED: EventType.ORDER_CREATED, OrderState.APPROVED: EventType.ORDER_APPROVED, OrderState.ROUTED: EventType.ORDER_ROUTED, OrderState.QUOTED: EventType.QUOTE_RECEIVED, OrderState.ACCEPTED: EventType.ORDER_ACCEPTED, OrderState.PARTIALLY_FILLED: EventType.ORDER_PARTIALLY_FILLED, OrderState.FILLED: EventType.ORDER_FILLED, OrderState.REJECTED: EventType.ORDER_REJECTED}.get(st)
            if et is not None and st not in (OrderState.CREATED,):
                self._emit(now_ns, req, et, {"reason": outcome.reason, "venue": outcome.venue.value, "evidence": outcome.evidence.value})
        book_side = OrderSide.SELL if side == "BUY" else OrderSide.BUY
        for px, face in outcome.fills:
            self._add_level(book, instrument, book_side, px * factor, face / factor, now_ns)
            if side == "BUY" and (limit_clean is None or px <= limit_clean + 1e-9):
                self._committed_base += face * px * 1.005 / 100.0 * fx
        return book

    # ---- bookkeeping -----------------------------------------------------------------------
    def on_fill(self, order_id: str, filled_face: float) -> None:
        self.process.budget.settle(order_id, filled_face)

    def end_of_window(self) -> None:
        self.process.budget.release_all_offers()
        self._committed_base = 0.0

    def _emit(self, ns: int, req: OrderRequest, et: EventType, payload: dict) -> None:
        if self.event_log is None:
            return
        try:
            self.event_log.append(Event(et, TimePoint(ns, ns), EventSource.EXECUTION, req.symbol, order_id=req.order_id, payload={**payload, "side": req.side, "face": req.face}, observed=False))
        except ValueError:
            pass

    def _add_level(self, book: OrderBook, instrument, side: OrderSide, price: float, face: float, ts: int) -> None:
        units = face_to_units(face)
        if units <= 0:
            return
        self._seq += 1
        px = Price(round(price, instrument.price_precision), instrument.price_precision)
        book.add(BookOrder(side, px, Quantity(units, 0), self._seq), 0, self._seq, ts)

    def _record(self, ts: int, req: OrderRequest, outcome: ProcessOutcome | None, ref, reason: str) -> None:
        self.decisions.append(FillDecision(
            pd.Timestamp(ts, tz="UTC"), req.order_id, req.symbol, req.side, req.face,
            outcome.p_fill if outcome else 0.0, outcome.draw if outcome else 0.0, outcome.offered_face if outcome else 0.0,
            outcome.cost.half_spread_bp if (outcome and outcome.cost) else 0.0, ref[0] if ref else 0.0, reason, req.protocol,
            outcome.state.value if outcome else "FUNDING_HOLD", outcome.venue.value if outcome else "", outcome.evidence.value if outcome else "",
            ref[3] if ref else "", outcome.cost.total_bp if (outcome and outcome.cost) else 0.0, len(outcome.quotes) if outcome else 0,
        ))

    def decisions_frame(self) -> pd.DataFrame:
        if not self.decisions:
            return pd.DataFrame(columns=list(FillDecision.__dataclass_fields__))
        return pd.DataFrame([d.__dict__ for d in self.decisions])


__all__ = ["FillDecision", "FixedIncomeFillModel", "NewIssueTerms"]
_ = field
