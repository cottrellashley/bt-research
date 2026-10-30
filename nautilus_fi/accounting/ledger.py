"""The ledger of record for a fixed income backtest.

Nautilus tracks trade-date positions and trade-date cash. This ledger is the balanced double-entry
record the blueprint requires: trades at clean value plus accrued, settlement on dated conventions,
daily accrual, coupon entitlement and payment by market rule, factor paydowns, calls, tenders,
maturities, defaults (non-accrual, claims), recoveries, cash interest, financing, FX translation and
the funding policy. Every posting cites the event that caused it, and NAV is assets less liabilities.
Nautilus cash is reconciled against the ledger's trade-date cash each snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from nautilus_trader.model import Currency, Money, OrderSide

from nautilus_fi.accounting.financing import CashRateModel, FinancingTerms, FlatCashRate
from nautilus_fi.accounting.postings import AccountingSettings, StandardAccountingPolicy
from nautilus_fi.core.events import Event, EventLog, EventSource, EventType
from nautilus_fi.core.ledger import Account, AccountClass, Ledger
from nautilus_fi.core.quantities import Money as LedgerMoney
from nautilus_fi.core.time import TimePoint
from nautilus_fi.funding.cash import (
    CashView,
    PendingSettlement,
    ReservationBook,
    SettlementQueue,
    TieredCashRate,
)
from nautilus_fi.funding.policy import FundingAssessor, FundingPolicy, Obligation
from nautilus_fi.instruments.analytics import accrued_interest
from nautilus_fi.instruments.bond import BondSpec, units_to_face
from nautilus_fi.instruments.cashflows import period_coupon_amount
from nautilus_fi.instruments.conventions import EntitlementMode
from nautilus_fi.instruments.enums import CouponType, PrincipalType


def _num(x) -> float:
    if x is None:
        return 0.0
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)


@dataclass
class FillRecord:
    ts: pd.Timestamp
    date: date
    symbol: str
    side: str
    face: float
    price: float
    clean_price: float
    factor: float
    accrued: float
    settle_date: date
    decision_mid: float | None
    fill_mid: float | None
    delay_days: int
    spread_cost: float
    delay_cost: float
    client_order_id: str
    protocol: str = "RFQ"
    orig_face: float = 0.0
    reference_source: str = ""
    cost_bp: float = 0.0
    evidence: str = ""              # execution evidence level the fill simulation rested on (process simulator)
    mark_evidence: str = ""         # valuation evidence class of the traded bond's mark on the fill date


@dataclass
class CashflowRecord:
    ts: pd.Timestamp
    date: date
    symbol: str
    type: str
    amount: float
    face: float = 0.0
    note: str = ""
    currency: str = "USD"


@dataclass
class HoldingSnapshot:
    symbol: str
    face: float
    original_face: float
    factor: float
    clean_price: float
    accrued: float
    market_value: float
    duration: float | None
    spread_duration: float | None
    z_spread_bp: float | None
    tier: str
    evidence: str = ""
    currency: str = "USD"


class FixedIncomeLedger:
    def __init__(self, env, base_currency: str = "USD", cash_rate: CashRateModel | TieredCashRate | None = None, financing: FinancingTerms | None = None, funding_policy: FundingPolicy | None = None, accounting: AccountingSettings | None = None, event_log: EventLog | None = None, fill_model=None, record_date_lag_days: int = 0) -> None:
        self.env = env
        self.base = base_currency
        self.currency = Currency.from_str(base_currency)
        self.cash_rate = cash_rate or FlatCashRate(0.0)
        self.financing = financing or FinancingTerms()
        self.funding_policy = funding_policy or FundingPolicy()
        self.assessor = FundingAssessor(self.funding_policy)
        self.events = event_log or EventLog()
        self.ledger = Ledger(base_currency)
        self.policy = StandardAccountingPolicy(self.ledger, accounting)
        self.queue = SettlementQueue()
        self.reservations = ReservationBook()
        self.cash_view = CashView(self.ledger, self.queue, self.reservations)
        self.fill_model = fill_model
        self.cache = None
        self.venue = None
        self.fills: list[FillRecord] = []
        self.cashflows: list[CashflowRecord] = []
        self.snapshots: list[dict] = []
        self.holdings_history: list[dict] = []
        self.reconciliation: list[dict] = []
        self.funding_events: list[dict] = []
        self.decision_marks: dict[str, tuple[float, date]] = {}
        self._seen_fill_counts: dict[str, int] = {}
        self._coupon_cursor: dict[str, int] = {}
        self._entitlement_cursor: dict[str, int] = {}
        self._entitled: dict[tuple[str, date], tuple[float, float, float]] = {}   # (symbol, payment date) -> (interest cash, principal per 100, entitled original face)
        self._tranche_paid: set[tuple[str, date]] = set()
        self._factor_applied: dict[str, float] = {}
        self._actions_applied: set[tuple[str, str, str]] = set()
        self._defaulted: dict[str, date] = {}
        self._last_interest_date: date | None = None
        self._last_accrual_date: date | None = None
        self._last_snapshot_date: date | None = None
        self._last_nav: float | None = None
        self._holdings_by_date: dict[date, dict[str, float]] = {}
        self._cum_income = 0.0
        self._cum_costs = 0.0
        self.starting_cash: float | None = None
        self.funding_default = False
        self.repo_balance = 0.0                       # funding repo drawn under the breach policy (base currency)
        self.funding_shortfall = 0.0                  # residual the policy could not source; strategies liquidate
        self.funding_actions: list[dict] = []
        self._shortfall_days = 0
        self._repo_id = "REPO-FUNDING"
        self._fx_converted: set[tuple[str, date]] = set()
        self._today: date | None = None

    # ------------------------------------------------------------------ binding
    def bind(self, cache, venue) -> None:
        self.cache = cache
        self.venue = venue

    def _fx(self, d: date):
        return lambda ccy: (self.env.fx(ccy, d) or 1.0)

    # ------------------------------------------------------------------ helpers
    def nautilus_cash(self, currency: str | None = None) -> float:
        if self.cache is None or self.venue is None:
            return 0.0
        acct = self.cache.account_for_venue(self.venue)
        if acct is None:
            return 0.0
        bal = acct.balance_total(Currency.from_str(currency or self.base))
        return _num(bal) if bal is not None else 0.0

    def account_cash(self) -> float:
        """Trade-date cash view in base currency (settled + receivables - payables), comparable with Nautilus."""
        rec, pay = self.queue.unsettled_cash(self.base)
        return self.ledger.cash(self.base) + rec - pay

    def _bond(self, instrument_id) -> BondSpec | None:
        return self.env.bonds.get(instrument_id.symbol.value)

    def _event(self, et: EventType, ns: int, symbol: str | None, payload: dict, source: EventSource = EventSource.LIFECYCLE, observed: bool = True, caused_by: tuple[str, ...] = (), order_id: str | None = None) -> Event:
        return self.events.append(Event(et, TimePoint(ns, ns), source, symbol, order_id=order_id, payload=payload, caused_by=caused_by, observed=observed))

    def available_cash(self, currency: str) -> float:
        """Spendable cash for the fill model's funding gate: settled minus reserved plus policy bridge,
        plus base-currency cash convertible at spot when the policy allows FX conversion."""
        avail = self.cash_view.available(currency, self.funding_policy.bridge_allowance() if currency == self.base else 0.0)
        if currency != self.base and self.funding_policy.allow_fx_conversion:
            fx = self.env.fx(currency, self._today or self.env.start_date) or 1.0
            avail += max(self.cash_view.available(self.base), 0.0) / fx
        return avail

    def unsettled_cash(self, d: date) -> float:
        rec, pay = self.queue.unsettled_cash(self.base)
        return rec - pay

    def _holder_accrued_per_100(self, bond: BondSpec, settle: date, inputs, rule) -> float:
        """Accrued per 100 current face carried by a holder whose trade settles on ``settle``: the
        contractual accrual of the period containing ``settle``; negative inside an ex-coupon period
        (gilt style), where the registered holder keeps the whole coupon and the buyer is compensated."""
        if bond.coupon_type == CouponType.ZERO:
            return 0.0
        base = accrued_interest(bond, settle, inputs)
        period = bond.current_period(settle)
        if period is not None and rule.negative_accrued_in_ex_period and rule.in_ex_period(settle, period.payment_date, bond.calendar):
            return base - period_coupon_amount(bond, period, inputs)
        return base

    @staticmethod
    def _entitlement_triggered(bond: BondSpec, rule, settle: date, payment_date: date) -> bool:
        """True once trades dated today no longer carry the coupon paid on ``payment_date`` - so the
        current (trade-date) holder is the one who will receive it."""
        if settle >= payment_date:
            return True
        if rule.mode == EntitlementMode.EX_DATE_HOLDER:
            return rule.in_ex_period(settle, payment_date, bond.calendar)
        if rule.mode == EntitlementMode.RECORD_DATE_HOLDER:
            return settle > rule.cutoff(payment_date, bond.calendar)
        return False

    def _held_and_today(self, d: date, positions) -> tuple[dict[str, float], dict[str, float]]:
        held: dict[str, float] = {}
        for pos in positions:
            bond = self._bond(pos.instrument_id)
            if bond is not None:
                held[bond.symbol] = _num(pos.signed_qty)
        today: dict[str, float] = {}
        for f in self.fills:
            if f.date == d and f.protocol != "EXPIRATION":
                today[f.symbol] = today.get(f.symbol, 0.0) + (f.orig_face if f.side == "BUY" else -f.orig_face) / 100.0
        return held, today

    def _book_entitlement(self, ts_now: int, d: date, bond: BondSpec, payment_date: date, interest_cash: float, principal_per_100: float, face_orig: float, source: str) -> None:
        key = (bond.symbol, payment_date)
        self._entitled[key] = (interest_cash, principal_per_100, face_orig)
        if interest_cash > 1e-9:
            ev = self._event(EventType.COUPON_ENTITLEMENT, ts_now, bond.symbol, {"amount": interest_cash, "payment_date": payment_date.isoformat(), "face": face_orig, "source": source})
            self.policy.coupon_entitlement(ev.id, ts_now, bond.symbol, bond.currency, interest_cash)

    # ------------------------------------------------------------------ main hook
    def on_timestamp(self, ts_now: int, positions) -> list[Money]:
        adjustments: list[Money] = []
        now = pd.Timestamp(ts_now, tz="UTC")
        d = now.date()
        self._today = d
        if self.starting_cash is None:
            self.starting_cash = self.nautilus_cash()
            self.policy.capital_flow(self._event(EventType.CAPITAL_FLOW, ts_now, None, {"amount": self.starting_cash, "currency": self.base}, EventSource.SYSTEM).id, ts_now, LedgerMoney(self.starting_cash, self.base), "initial capital")
        if d not in self._holdings_by_date:
            self._holdings_by_date[d] = {self._bond(p.instrument_id).symbol: _num(p.signed_qty) for p in positions if self._bond(p.instrument_id) is not None}
            for old in [k for k in self._holdings_by_date if (d - k).days > 45]:
                self._holdings_by_date.pop(old, None)
        self._process_settlements(ts_now, d)
        adjustments += self._process_new_fills(ts_now, d)
        adjustments += self._process_cash_interest(ts_now, d)
        self._process_entitlements(ts_now, d, positions)
        self._process_daily_accrual(ts_now, d, positions)
        adjustments += self._process_coupons_and_factors(ts_now, d, positions)
        adjustments += self._process_actions(ts_now, d, positions)
        adjustments += self._process_funding(ts_now, d, positions)
        if self.env.schedule.is_publish(ts_now) and self._last_snapshot_date != d:
            self._mark_positions(ts_now, d, positions)
            self._snapshot(now, d, positions, adjustments)
        if self.fill_model is not None and self.env.schedule.is_publish(ts_now):
            self.fill_model.end_of_window()
        return adjustments

    # ------------------------------------------------------------------ settlement
    def _process_settlements(self, ts_now: int, d: date) -> None:
        for item in self.queue.due(d):
            ev = self._event(EventType.SETTLEMENT, ts_now, item.instrument, {"amount": item.cash_amount, "currency": item.currency, "settle_date": item.settle_date.isoformat()}, caused_by=(item.trade_event_id,) if self.events.get(item.trade_event_id) else ())
            self.policy.settle(ev.id, ts_now, item.instrument, item.currency, abs(item.cash_amount), "PAY" if item.cash_amount < 0 else "RECEIVE")
            self.queue.settle(item, d)

    # ------------------------------------------------------------------ fills
    def _process_new_fills(self, ts_now: int, d: date) -> list[Money]:
        adjustments: list[Money] = []
        if self.cache is None:
            return adjustments
        for order in self.cache.orders(None, None, None, None):
            events = [e for e in order.events() if type(e).__name__ == "OrderFilled"]
            key = str(order.client_order_id)
            seen = self._seen_fill_counts.get(key, 0)
            if len(events) <= seen:
                continue
            bond = self._bond(order.instrument_id)
            for e in events[seen:]:
                if bond is None:
                    continue
                fill_ts = pd.Timestamp(e.ts_event, tz="UTC")
                fill_date = fill_ts.date()
                side = "BUY" if e.order_side == OrderSide.BUY else "SELL"
                units = _num(e.last_qty)
                px = _num(e.last_px)
                factor = self.env.factor(bond.symbol, fill_date)
                clean = px / factor if factor else px
                orig_face = units_to_face(units)
                face = orig_face * factor
                is_expiration = any(str(t).startswith("EXPIRATION") for t in (order.tags or []))
                settle = fill_date if is_expiration else self.env.settle_date(bond, fill_date)
                defaulted = bond.symbol in self._defaulted and self._defaulted[bond.symbol] <= fill_date
                inputs = self.env.market_inputs(bond.symbol, fill_date)
                rule = self.env.entitlement_rule(bond)
                accrued_per_100 = 0.0 if (is_expiration or defaulted) else self._holder_accrued_per_100(bond, settle, inputs, rule)
                accrued_cash = accrued_per_100 * face / 100.0
                sign = 1.0 if side == "BUY" else -1.0
                ccy = bond.currency
                trade_ev = self._event(EventType.TRADE, e.ts_event, bond.symbol, {"side": side, "face": face, "clean": clean, "accrued": accrued_cash, "settle": settle.isoformat(), "currency": ccy, "expiration": is_expiration}, EventSource.EXECUTION, observed=False, order_id=key)
                qty_signed = orig_face if side == "BUY" else -orig_face
                held_before = self.policy.book.quantity(bond.symbol)
                if defaulted and side == "SELL" and held_before > 1e-9:          # selling (or recovering) a claim, not a security
                    self.policy.recovery(trade_ev.id, e.ts_event, bond.symbol, ccy, face * clean / 100.0, min(orig_face / held_before, 1.0), settled=False)
                else:
                    self.policy.trade(trade_ev.id, e.ts_event, bond.symbol, ccy, qty_signed, face * clean / 100.0, accrued_cash, 0.0)
                    if defaulted and side == "BUY":
                        self.policy.reclassify_to_claim(trade_ev.id, e.ts_event, bond.symbol, ccy, face * clean / 100.0)
                if abs(self.policy.book.quantity(bond.symbol)) < 1e-9:            # fully closed: no accrued survives the position
                    self.policy.set_accrued(trade_ev.id, e.ts_event, bond.symbol, ccy, 0.0)
                cash_amount = -(face * clean / 100.0 + accrued_cash) if side == "BUY" else (face * clean / 100.0 + accrued_cash)
                self.queue.add(PendingSettlement(trade_ev.id, bond.symbol, ccy, cash_amount, qty_signed, fill_date, settle))
                if is_expiration or settle <= d:
                    self._process_settlements(ts_now, d)
                if abs(accrued_cash) > 1e-9:
                    adjustments.append(Money(-sign * accrued_cash, Currency.from_str(ccy)))
                    self.cashflows.append(CashflowRecord(fill_ts, fill_date, bond.symbol, "ACCRUED_PAID" if side == "BUY" else "ACCRUED_RECEIVED", -sign * accrued_cash, face, currency=ccy))
                if ccy != self.base:
                    adjustments += self._fx_fund(ts_now, d, ccy, cash_amount)
                decision = self.decision_marks.get(key)
                fill_mark = self.env.mark(bond.symbol, fill_date)
                ref = self.env.reference(bond.symbol, fill_date)
                fill_mid = fill_mark.clean_mid if fill_mark else None
                decision_mid, decision_date = (decision if decision else (None, None))
                delay_days = self.env.calendar.business_days_between(decision_date, fill_date) if decision_date else 0
                ref_mid = ref[0] if ref else fill_mid
                spread_cost = (clean - ref_mid) * sign * face / 100.0 if ref_mid is not None else 0.0
                delay_cost = (ref_mid - decision_mid) * sign * face / 100.0 if (ref_mid is not None and decision_mid is not None) else 0.0
                if not is_expiration:
                    self._cum_costs += spread_cost
                outcome = self.fill_model.outcomes.get(key) if self.fill_model is not None else None
                protocol = "EXPIRATION" if is_expiration else next((str(t).split("=", 1)[1] for t in (order.tags or []) if str(t).startswith("protocol=")), "RFQ")
                self.fills.append(FillRecord(fill_ts, fill_date, bond.symbol, side, face, px, clean, factor, sign * accrued_cash, settle, decision_mid, fill_mid, delay_days, spread_cost, delay_cost, key, protocol, orig_face, ref[3] if ref else "", outcome.cost.total_bp if (outcome and outcome.cost) else 0.0, outcome.evidence.value if outcome else "", self.env.evidence(bond.symbol, fill_date).value if fill_mark else ""))
                if self.fill_model is not None and not is_expiration:
                    self.fill_model.on_fill(key, face)
            self._seen_fill_counts[key] = len(events)
        return adjustments

    def _fx_fund(self, ts_now: int, d: date, ccy: str, cash_amount: float) -> list[Money]:
        """Foreign-currency trades: convert base cash at spot (recorded as an FX fixing event) so the
        Nautilus balances and the ledger both carry the right currencies."""
        rate = self.env.fx(ccy, d)
        if rate is None:
            return []
        if cash_amount >= 0:
            return []
        need = -cash_amount
        ev = self._event(EventType.FX_FIXING, ts_now, None, {"pair": ccy + self.base, "rate": rate, "amount": need}, EventSource.FUNDING)
        from nautilus_fi.core.ledger import JournalEntry, Posting

        # Each currency leg balances on its own through the FX translation (equity) account, so the
        # books stay balanced per currency and later spot moves on the foreign balance show up as
        # translation P&L rather than as an unexplained NAV change.
        self.ledger.post(JournalEntry(ev.id, ts_now, (
            Posting.dr(Account.CASH_SETTLED, LedgerMoney(need, ccy), "", "fx purchase"), Posting.cr(Account.FX_TRANSLATION, LedgerMoney(need, ccy), "", "fx purchase"),
            Posting.dr(Account.FX_TRANSLATION, LedgerMoney(need * rate, self.base), "", "fx purchase"), Posting.cr(Account.CASH_SETTLED, LedgerMoney(need * rate, self.base), "", "fx purchase"),
        ), "fx conversion"))
        return [Money(need, Currency.from_str(ccy)), Money(-need * rate, self.currency)]

    # ------------------------------------------------------------------ cash interest
    def _process_cash_interest(self, ts_now: int, d: date) -> list[Money]:
        if self._last_interest_date is None:
            self._last_interest_date = d
            return []
        if d <= self._last_interest_date:
            return []
        days = (d - self._last_interest_date).days
        settled = self.ledger.cash(self.base)
        if isinstance(self.cash_rate, TieredCashRate):
            interest = self.cash_rate.interest(self.base, settled, days)
        else:
            rate = self.cash_rate.rate(self._last_interest_date) + (self.financing.borrow_spread_bp / 1e4 if settled < 0 else 0.0)
            interest = settled * rate * days / float(self.financing.day_count_basis)
        self._last_interest_date = d
        if self.repo_balance > 1e-6:                                                    # funding repo accrues at cash rate + spread into the payable
            r_rate = (self.cash_rate.rate(self.base, self.repo_balance) if isinstance(self.cash_rate, TieredCashRate) else self.cash_rate.rate(self._last_interest_date)) + self.financing.borrow_spread_bp / 1e4
            r_int = self.repo_balance * r_rate * days / float(self.financing.day_count_basis)
            if r_int > 1e-9:
                ev = self._event(EventType.FUNDING, ts_now, None, {"kind": "REPO_INTEREST", "amount": r_int}, EventSource.FUNDING)
                self.policy.repo_interest(ev.id, ts_now, self._repo_id, self.base, r_int)
                self.repo_balance += r_int
                self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, "", "FINANCING", -r_int, 0.0, f"{days}d funding repo interest"))
        if abs(interest) < 1e-9:
            return []
        ev = self._event(EventType.FUNDING, ts_now, None, {"kind": "CASH_INTEREST" if interest >= 0 else "FINANCING", "amount": interest, "days": days, "balance": settled}, EventSource.FUNDING)
        self.policy.cash_interest(ev.id, ts_now, self.base, interest)
        self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, "", "CASH_INTEREST" if interest >= 0 else "FINANCING", interest, 0.0, f"{days}d on settled cash"))
        self._cum_income += interest
        return [Money(interest, self.currency)]

    def _process_entitlements(self, ts_now: int, d: date, positions) -> None:
        """Crystallise coupon entitlements: on the first trade date whose settlement no longer carries a
        coupon, the trade-date holder (start of day; today's buyers excluded) acquires a receivable for
        it and accrued interest of the completed period is released. Payment clears the receivable."""
        held, today = self._held_and_today(d, positions)
        for symbol, bond in self.env.bonds.items():
            if bond.issue_date > d or bond.coupon_type == CouponType.ZERO:
                continue
            rule = self.env.entitlement_rule(bond)
            settle = self.env.settle_date(bond, d)
            defaulted = symbol in self._defaulted and self._defaulted[symbol] <= d
            units_start = held.get(symbol, 0.0) - today.get(symbol, 0.0)
            face_orig = units_to_face(units_start)
            if self.env.is_structured(symbol):
                for (sym, pay_date), (interest_per_100, principal_per_100) in self.env.tranche_payments.items():
                    if sym != symbol or pay_date < self.env.start_date or (symbol, pay_date) in self._entitled or not self._entitlement_triggered(bond, rule, settle, pay_date):
                        continue
                    if defaulted or abs(face_orig) < 1e-9:
                        self._entitled[(symbol, pay_date)] = (0.0, principal_per_100, 0.0)
                        continue
                    self._book_entitlement(ts_now, d, bond, pay_date, interest_per_100 * face_orig / 100.0, principal_per_100, face_orig, "DEAL_WATERFALL")
                continue
            sched = bond.schedule
            i = self._entitlement_cursor.get(symbol, 0)
            inputs = None
            while i < len(sched) and self._entitlement_triggered(bond, rule, settle, sched[i].payment_date):
                period = sched[i]
                i += 1
                key = (symbol, period.payment_date)
                if period.payment_date < self.env.start_date or key in self._entitled:
                    continue
                if defaulted or abs(face_orig) < 1e-9:
                    self._entitled[key] = (0.0, 0.0, 0.0)
                    continue
                inputs = inputs or self.env.market_inputs(symbol, d)
                self._book_entitlement(ts_now, d, bond, period.payment_date, period_coupon_amount(bond, period, inputs) * face_orig / 100.0, 0.0, face_orig, "SCHEDULE")
            self._entitlement_cursor[symbol] = i

    # ------------------------------------------------------------------ accrual
    def _process_daily_accrual(self, ts_now: int, d: date, positions) -> None:
        if self._last_accrual_date == d:
            return
        self._last_accrual_date = d
        for pos in positions:
            bond = self._bond(pos.instrument_id)
            if bond is None:
                continue
            units = _num(pos.signed_qty)
            if units == 0:
                continue
            factor = self.env.factor(bond.symbol, d)
            face = units_to_face(units) * factor
            defaulted = bond.symbol in self._defaulted and self._defaulted[bond.symbol] <= d
            target = 0.0 if defaulted else self._holder_accrued_per_100(bond, self.env.settle_date(bond, d), self.env.market_inputs(bond.symbol, d), self.env.entitlement_rule(bond)) * face / 100.0
            self.policy.set_accrued(f"ACCRUAL-{bond.symbol}-{d.isoformat()}", ts_now, bond.symbol, bond.currency, target)

    # ------------------------------------------------------------------ coupons / paydowns
    def _entitled_face(self, bond: BondSpec, payment_date: date, d: date, positions_units: float, today_delta_units: float) -> float:
        rule = self.env.entitlement_rule(bond)
        if rule.mode == EntitlementMode.PAYMENT_DATE_HOLDER:
            return units_to_face(positions_units - today_delta_units)
        cutoff = rule.cutoff(payment_date, bond.calendar)
        snap = self._holdings_by_date.get(cutoff)
        if snap is not None and bond.symbol in snap:
            return units_to_face(snap[bond.symbol])
        return units_to_face(positions_units - today_delta_units)

    def _process_coupons_and_factors(self, ts_now: int, d: date, positions) -> list[Money]:
        """Payment date: pay what was entitled (receivable -> cash); deal-waterfall principal and factor
        paydowns reduce the position at the paydown price."""
        adjustments: list[Money] = []
        held, today = self._held_and_today(d, positions)
        for symbol, bond in self.env.bonds.items():
            if bond.issue_date > d:
                continue
            units_now = held.get(symbol, 0.0)
            defaulted = symbol in self._defaulted and self._defaulted[symbol] <= d
            ccy = Currency.from_str(bond.currency)
            if self.env.is_structured(symbol):
                for (sym, pay_date), (interest_per_100, principal_per_100) in self.env.tranche_payments.items():
                    if sym != symbol or pay_date > d or (symbol, pay_date) in self._tranche_paid or pay_date < self.env.start_date:
                        continue
                    self._tranche_paid.add((symbol, pay_date))
                    ent = self._entitled.get((symbol, pay_date))
                    if ent is None:                                           # entitlement not seen (window started late): holder at start of day
                        face_orig = 0.0 if defaulted else self._entitled_face(bond, pay_date, d, units_now, today.get(symbol, 0.0))
                        self._book_entitlement(ts_now, d, bond, pay_date, interest_per_100 * face_orig / 100.0, principal_per_100, face_orig, "DEAL_WATERFALL")
                        ent = self._entitled[(symbol, pay_date)]
                    interest_cash, _, face_orig = ent
                    if abs(face_orig) < 1e-9:
                        continue
                    if interest_cash > 1e-9:
                        ev = self._event(EventType.COUPON_PAYMENT, ts_now, symbol, {"amount": interest_cash, "per_100": interest_per_100, "source": "DEAL_WATERFALL", "currency": bond.currency})
                        self.policy.coupon_payment(ev.id, ts_now, symbol, bond.currency, interest_cash)
                        adjustments.append(Money(interest_cash, ccy))
                        self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, symbol, "COUPON", interest_cash, face_orig, "deal waterfall interest", bond.currency))
                        self._cum_income += interest_cash
                    if principal_per_100 > 1e-12 and abs(units_now) > 1e-9:
                        paid_face = min(units_to_face(units_now), face_orig)
                        cash = principal_per_100 * paid_face / 100.0
                        frac = min(principal_per_100 / max(self.env.factor(symbol, self.env.prev_trading_date(d) or d) * 100.0, 1e-9), 1.0)
                        ev = self._event(EventType.PRINCIPAL_REDUCTION, ts_now, symbol, {"amount": cash, "per_100": principal_per_100, "source": "DEAL_WATERFALL"})
                        self.policy.principal_reduction(ev.id, ts_now, symbol, bond.currency, frac, cash)
                        adjustments.append(Money(cash, ccy))
                        self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, symbol, "PRINCIPAL", cash, principal_per_100 * paid_face / 100.0, "deal waterfall principal", bond.currency))
                continue
            if bond.coupon_type != CouponType.ZERO:
                sched = bond.schedule
                i = self._coupon_cursor.get(symbol, 0)
                inputs = None
                while i < len(sched) and sched[i].payment_date <= d:
                    period = sched[i]
                    i += 1
                    if period.payment_date < self.env.start_date:
                        continue
                    key = (symbol, period.payment_date)
                    ent = self._entitled.get(key)
                    if ent is None:
                        face_orig = 0.0 if defaulted else self._entitled_face(bond, period.payment_date, d, units_now, today.get(symbol, 0.0))
                        inputs = inputs or self.env.market_inputs(symbol, d)
                        self._book_entitlement(ts_now, d, bond, period.payment_date, period_coupon_amount(bond, period, inputs) * face_orig / 100.0 if abs(face_orig) > 1e-9 else 0.0, 0.0, face_orig, "SCHEDULE")
                        ent = self._entitled[key]
                    cash, _, face_orig = ent
                    if abs(cash) > 1e-9:
                        ev = self._event(EventType.COUPON_PAYMENT, ts_now, symbol, {"amount": cash, "currency": bond.currency, "payment_date": period.payment_date.isoformat()})
                        self.policy.coupon_payment(ev.id, ts_now, symbol, bond.currency, cash)
                        adjustments.append(Money(cash, ccy))
                        self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, symbol, "COUPON", cash, face_orig * self.env.factor(symbol, period.accrual_start), f"period {period.accrual_start}..{period.accrual_end}", bond.currency))
                        self._cum_income += cash
                self._coupon_cursor[symbol] = i
            if bond.principal_type in (PrincipalType.AMORTIZING, PrincipalType.SINKING):
                f_now = self.env.factor(symbol, d)
                f_prev = self._factor_applied.get(symbol)
                if f_prev is not None and f_now < f_prev - 1e-12 and abs(units_now) > 1e-9 and not defaulted:
                    price = bond.amortization.paydown_price if bond.amortization else 100.0
                    orig_face = units_to_face(units_now)
                    cash = (f_prev - f_now) * orig_face * price / 100.0
                    ev = self._event(EventType.PRINCIPAL_REDUCTION, ts_now, symbol, {"factor_from": f_prev, "factor_to": f_now, "amount": cash})
                    self.policy.principal_reduction(ev.id, ts_now, symbol, bond.currency, (f_prev - f_now) / f_prev if f_prev else 0.0, cash)
                    adjustments.append(Money(cash, ccy))
                    self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, symbol, "PRINCIPAL", cash, (f_prev - f_now) * orig_face, f"factor {f_prev:.6f}->{f_now:.6f}", bond.currency))
                self._factor_applied[symbol] = f_now
        return adjustments

    # ------------------------------------------------------------------ corporate actions
    def _process_actions(self, ts_now: int, d: date, positions) -> list[Money]:
        adjustments: list[Money] = []
        held = {}
        for pos in positions:
            bond = self._bond(pos.instrument_id)
            if bond is not None:
                held[bond.symbol] = _num(pos.signed_qty)
        for act in self.env.actions_effective(d):
            key = (act.symbol, act.action, act.effective_date.isoformat())
            if key in self._actions_applied:
                continue
            self._actions_applied.add(key)
            bond = self.env.bonds.get(act.symbol)
            if bond is None:
                continue
            if act.action == "DEFAULT":
                self._defaulted[act.symbol] = act.effective_date
                ev = self._event(EventType.DEFAULT, ts_now, act.symbol, {"date": act.effective_date.isoformat(), "note": act.note, "price": act.price})
                if abs(held.get(act.symbol, 0.0)) > 0:
                    self.policy.default_nonaccrual(ev.id, ts_now, act.symbol, bond.currency)
                    wo = self.policy.write_off_coupon_receivable(ev.id, ts_now, act.symbol, bond.currency)
                    if wo is not None:
                        for key in [k for k in self._entitled if k[0] == act.symbol and k[1] >= d]:
                            self._entitled[key] = (0.0, self._entitled[key][1], self._entitled[key][2])
                        self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, act.symbol, "MISSED_COUPON", 0.0, 0.0, "entitled coupon written off", bond.currency))
                self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, act.symbol, "DEFAULT", 0.0, 0.0, act.note, bond.currency))
                continue
            units = held.get(act.symbol, 0.0)
            if units == 0:
                continue
            if act.action in ("CALL", "PUT", "TENDER", "MATURITY"):
                face = units_to_face(units) * self.env.factor(act.symbol, d)
                accrued_per_100 = accrued_interest(bond, act.effective_date, self.env.market_inputs(act.symbol, d))
                period = bond.current_period(act.effective_date)
                if period is not None and period.accrual_start == act.effective_date:
                    accrued_per_100 = 0.0
                cash = accrued_per_100 * face / 100.0
                self._event(EventType.REDEMPTION if act.action == "MATURITY" else EventType.EXERCISE, ts_now, act.symbol, {"action": act.action, "price": act.price, "accrued": cash})
                if abs(cash) > 1e-9:
                    ev = self._event(EventType.COUPON_PAYMENT, ts_now, act.symbol, {"amount": cash, "kind": "accrued at redemption", "currency": bond.currency})
                    self.policy.coupon_payment(ev.id, ts_now, act.symbol, bond.currency, cash)
                    adjustments.append(Money(cash, Currency.from_str(bond.currency)))
                    self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, act.symbol, "ACCRUED_AT_REDEMPTION", cash, face, act.action, bond.currency))
            elif act.action == "RECOVERY":
                self._event(EventType.RECOVERY_DISTRIBUTION, ts_now, act.symbol, {"price": act.price, "note": act.note})
        return adjustments

    # ------------------------------------------------------------------ funding
    def _repo_capacity(self, d: date, positions) -> float:
        """Base-currency cash obtainable by repoing eligible holdings (governments, agencies, IG credit,
        agency MBS) after haircut and the policy's eligible share, net of what is already drawn."""
        cap = 0.0
        for h in self.holdings(d, positions):
            bond = self.env.bonds[h.symbol]
            score = self.env.rating_score(h.symbol, d) or 99
            defaulted = h.symbol in self._defaulted and self._defaulted[h.symbol] <= d
            eligible = (not defaulted) and (bond.asset_type.is_government or bond.asset_type.value in ("AGENCY", "SUPRANATIONAL", "MBS") or (bond.asset_type.value == "CORPORATE" and score <= 10))
            if eligible and h.market_value > 0:
                cap += h.market_value * (1.0 - self.funding_policy.repo_haircut)
        return max(0.0, cap * self.funding_policy.repo_eligible_share - self.repo_balance)

    def _process_funding(self, ts_now: int, d: date, positions) -> list[Money]:
        """Daily funding assessment in the base currency and the policy's ordered breach actions:
        hold buys (the fill gate), source repo against eligible holdings, ask the strategy to liquidate
        by rule, and declare a funding default only when the shortfall persists."""
        adjustments: list[Money] = []
        obligations = [Obligation(p.settle_date, p.currency, -p.cash_amount, "SETTLEMENT", p.trade_event_id) for p in self.queue.pending(self.base) if p.cash_amount < 0]
        receipts = [Obligation(p.settle_date, p.currency, p.cash_amount, "SETTLEMENT", p.trade_event_id) for p in self.queue.pending(self.base) if p.cash_amount > 0]
        a = self.assessor.assess(d, self.base, self.ledger.cash(self.base), self.reservations.reserved_cash(self.base), obligations, receipts)
        free = self.cash_view.available(self.base) - self.funding_policy.min_cash_buffer
        if not a.breached:
            self.funding_shortfall = 0.0
            self._shortfall_days = 0
            if self.repo_balance > 1e-6 and free > 1e-6:                              # repay funding repo as cash frees up
                repay = min(self.repo_balance, free)
                ev = self._event(EventType.FUNDING, ts_now, None, {"kind": "REPO_REPAY", "amount": repay}, EventSource.FUNDING, observed=False)
                self.policy.repo_close(ev.id, ts_now, self._repo_id, self.base, repay)
                self.repo_balance -= repay
                adjustments.append(Money(-repay, self.currency))
                self.funding_actions.append({"date": d, "action": "REPO_REPAY", "amount": repay})
                self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, "", "REPO_REPAY", -repay, 0.0, "funding repo repaid"))
            return adjustments
        self.funding_events.append({"date": d, "shortfall": a.shortfall, "settled": a.settled_cash, "obligations": a.obligations_in_window})
        self._event(EventType.FUNDING_BREACH, ts_now, None, {"shortfall": a.shortfall, "currency": self.base}, EventSource.FUNDING, observed=False)
        remaining = a.shortfall
        for step in self.funding_policy.breach_actions:
            if remaining <= 1e-6:
                break
            name = step.value
            if name == "CANCEL_BUYS":
                self.funding_actions.append({"date": d, "action": "HOLD_BUYS", "amount": remaining})   # the fill gate rests buys while cash is short
            elif name == "SOURCE_REPO":
                cap = self._repo_capacity(d, positions)
                draw = min(cap, remaining)
                if draw > 1e-6:
                    ev = self._event(EventType.FUNDING, ts_now, None, {"kind": "REPO_OPEN", "amount": draw, "capacity": cap}, EventSource.FUNDING, observed=False)
                    self.policy.repo_open(ev.id, ts_now, self._repo_id, self.base, draw)
                    self.repo_balance += draw
                    adjustments.append(Money(draw, self.currency))
                    self.cashflows.append(CashflowRecord(pd.Timestamp(ts_now, tz="UTC"), d, "", "REPO_OPEN", draw, 0.0, "funding repo against eligible holdings"))
                    self.funding_actions.append({"date": d, "action": "SOURCE_REPO", "amount": draw})
                    remaining -= draw
            elif name == "LIQUIDATE_BY_RULE":
                self.funding_actions.append({"date": d, "action": "LIQUIDATE_BY_RULE", "amount": remaining})
                self.funding_shortfall = remaining                                      # picked up by the strategy's liquidation rule
                remaining_after_liquidation = remaining
                remaining = 0.0
                self._shortfall_days += 1
                if self._shortfall_days > self.funding_policy.failure_after_business_days and remaining_after_liquidation > 1e-6:
                    self.funding_default = True
                    self.funding_actions.append({"date": d, "action": "DECLARE_FAILURE", "amount": remaining_after_liquidation})
            elif name == "DECLARE_FAILURE":
                self.funding_default = True
                self.funding_actions.append({"date": d, "action": "DECLARE_FAILURE", "amount": remaining})
                remaining = 0.0
        return adjustments

    # ------------------------------------------------------------------ marks and snapshot
    def _mark_positions(self, ts_now: int, d: date, positions) -> None:
        for pos in positions:
            bond = self._bond(pos.instrument_id)
            if bond is None:
                continue
            units = _num(pos.signed_qty)
            if units == 0:
                continue
            mark = self.env.mark(bond.symbol, d)
            if mark is None:
                continue
            factor = self.env.factor(bond.symbol, d)
            value = units_to_face(units) * factor * mark.clean_mid / 100.0
            if bond.symbol in self._defaulted and self._defaulted[bond.symbol] <= d:
                self.policy.claim_mark(f"MARK-{bond.symbol}-{d.isoformat()}", ts_now, bond.symbol, bond.currency, value)
            else:
                self.policy.mark(f"MARK-{bond.symbol}-{d.isoformat()}", ts_now, bond.symbol, bond.currency, value)

    def holdings(self, d: date, positions) -> list[HoldingSnapshot]:
        out = []
        for pos in positions:
            bond = self._bond(pos.instrument_id)
            if bond is None:
                continue
            units = _num(pos.signed_qty)
            if units == 0:
                continue
            orig_face = units_to_face(units)
            factor = self.env.factor(bond.symbol, d)
            face = orig_face * factor
            mark = self.env.mark(bond.symbol, d)
            clean = mark.clean_mid if mark else 0.0
            defaulted = bond.symbol in self._defaulted and self._defaulted[bond.symbol] <= d
            accrued = 0.0 if defaulted else self._holder_accrued_per_100(bond, self.env.settle_date(bond, d), self.env.market_inputs(bond.symbol, d), self.env.entitlement_rule(bond))
            an = self.env.analytics(bond.symbol, d)
            fx = self.env.fx(bond.currency, d) or 1.0
            out.append(HoldingSnapshot(bond.symbol, face, orig_face, factor, clean, accrued, face * (clean + accrued) / 100.0 * fx, an.mod_duration if an else None, an.spread_duration if an else None, an.z_spread_bp if an else None, mark.tier.value if mark else "NONE", self.env.evidence(bond.symbol, d).value if mark else "STALE", bond.currency))
        return out

    def _snapshot(self, now: pd.Timestamp, d: date, positions, pending: list[Money]) -> None:
        fx = self._fx(d)
        nav = self.ledger.nav(fx).amount
        holds = self.holdings(d, positions)
        mv = sum(h.market_value for h in holds)
        accrued = sum(h.accrued * h.face / 100.0 * (self.env.fx(h.currency, d) or 1.0) for h in holds)
        currencies = {self.base} | {b.currency for b in self.env.bonds.values()}
        settled = sum(self.ledger.cash(c) * fx(c) for c in currencies)                 # all cash balances translated to base
        rec = pay = 0.0
        for c in currencies:
            r_c, p_c = self.queue.unsettled_cash(c)
            rec += r_c * fx(c)
            pay += p_c * fx(c)
        coupon_receivable = sum(v * fx(ccy) for (_, ccy), v in self.policy.coupon_receivable.items())
        foreign_cash = {c: self.ledger.cash(c) for c in currencies if c != self.base and abs(self.ledger.cash(c)) > 1e-9}
        w = [(h.market_value / nav) if nav else 0.0 for h in holds]
        dur = sum(wi * (h.duration or 0.0) for wi, h in zip(w, holds))
        sdur = sum(wi * (h.spread_duration or 0.0) for wi, h in zip(w, holds))
        dts = sum(wi * (h.spread_duration or 0.0) * (h.z_spread_bp or 0.0) for wi, h in zip(w, holds))
        day_ret = (nav / self._last_nav - 1.0) if self._last_nav else 0.0
        by_evidence = {}
        for h in holds:
            by_evidence[h.evidence] = by_evidence.get(h.evidence, 0.0) + h.market_value
        nautilus_cash = self.nautilus_cash() + sum(m.as_double() for m in pending if str(m.currency) == self.base)
        ledger_td_cash = self.account_cash()
        self.reconciliation.append({"date": pd.Timestamp(d), "nautilus_cash": nautilus_cash, "ledger_trade_date_cash": ledger_td_cash, "difference": nautilus_cash - ledger_td_cash})
        try:
            self.ledger.conservation_check(fx)
            balanced = True
        except Exception:  # noqa: BLE001
            balanced = False
        self.snapshots.append({
            "date": pd.Timestamp(d), "nav": nav, "cash": settled + rec - pay - self.repo_balance, "settled_cash": settled, "unsettled_receivable": rec, "unsettled_payable": pay,
            "market_value": mv, "accrued": accrued, "coupon_receivable": coupon_receivable, "unsettled_cash": rec - pay, "gross_exposure": sum(abs(h.market_value) for h in holds), "n_positions": len(holds),
            "duration": dur, "spread_duration": sdur, "dts": dts, "day_return": day_ret, "cum_income": self._cum_income, "cum_spread_cost": self._cum_costs,
            "model_marked_share": (sum(v for k, v in by_evidence.items() if k in ("MODEL", "STALE")) / mv) if mv else 0.0,
            **{f"evidence_{k.lower()}_share": (v / mv if mv else 0.0) for k, v in by_evidence.items()},
            "ledger_balanced": balanced, "funding_breach": bool(self.funding_events and self.funding_events[-1]["date"] == d),
            "repo_balance": self.repo_balance, "funding_shortfall": self.funding_shortfall, "foreign_cash_base": sum(v * fx(c) for c, v in foreign_cash.items()), "fx_translation": self.ledger.class_total(AccountClass.EQUITY, self.base, fx) - sum(self.ledger.balance(a, currency=self.base) for a in (Account.CAPITAL, Account.REALIZED_PNL, Account.UNREALIZED_PNL)) if foreign_cash else 0.0,
        })
        for h in holds:
            self.holdings_history.append({"date": pd.Timestamp(d), **h.__dict__})
        self._last_nav = nav
        self._last_snapshot_date = d

    # ------------------------------------------------------------------ frames
    def nav_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots).set_index("date") if self.snapshots else pd.DataFrame()

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.fills]) if self.fills else pd.DataFrame(columns=list(FillRecord.__dataclass_fields__))

    def cashflows_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.cashflows]) if self.cashflows else pd.DataFrame(columns=list(CashflowRecord.__dataclass_fields__))

    def holdings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.holdings_history) if self.holdings_history else pd.DataFrame()

    def reconciliation_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.reconciliation)

    def entries_frame(self) -> pd.DataFrame:
        return self.ledger.entries_frame()

    def events_frame(self) -> pd.DataFrame:
        return self.events.to_frame()

    def trial_balance(self) -> pd.DataFrame:
        return self.ledger.trial_balance()


__all__ = ["AccountClass", "CashflowRecord", "FillRecord", "FixedIncomeLedger", "HoldingSnapshot"]
_ = field
