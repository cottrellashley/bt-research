"""Contracts as state machines (blueprint section 11).

Announcements, effective dates and cash movements are distinct events. A call notice changes
trading and valuation before redemption; a tender needs an election and possible proration; a put
needs timely notice; a default starts another lifecycle (missed payment, grace, legal default,
acceleration, claim, recovery distributions, restructuring into new claims). Events on the same
timestamp are processed in a documented precedence order with a deterministic tie-breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import Any

from nautilus_fi.core.events import Event, EventType
from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.enums import CouponType


class ContractState(str, Enum):
    ANNOUNCED = "ANNOUNCED"
    OUTSTANDING = "OUTSTANDING"
    CALL_NOTICED = "CALL_NOTICED"
    TENDER_OPEN = "TENDER_OPEN"
    GRACE_PERIOD = "GRACE_PERIOD"
    DEFAULTED = "DEFAULTED"
    ACCELERATED = "ACCELERATED"
    IN_RESTRUCTURING = "IN_RESTRUCTURING"
    EXCHANGED = "EXCHANGED"
    MATURED = "MATURED"
    REDEEMED = "REDEEMED"
    DISCHARGED = "DISCHARGED"


# Same-timestamp precedence: lower runs first. Entitlement is resolved before a matured position is
# removed; fixings precede accrual; settlement precedes funding so settled cash can fund purchases;
# marks precede decisions; orders come last.
PRECEDENCE: dict[EventType, int] = {
    EventType.CORRECTION: 0, EventType.AMENDMENT: 1, EventType.FIXING: 2, EventType.RATE_RESET: 3, EventType.FX_FIXING: 4,
    EventType.INDEX_RATIO_UPDATE: 5, EventType.FACTOR_UPDATE: 6, EventType.ACCRUAL: 7, EventType.COUPON_ENTITLEMENT: 8,
    EventType.COUPON_DEFERRAL: 9, EventType.PIK_CAPITALISATION: 10, EventType.COUPON_PAYMENT: 11, EventType.PRINCIPAL_REDUCTION: 12,
    EventType.DRAWDOWN: 13, EventType.REPAYMENT: 14, EventType.PARTIAL_CALL: 15, EventType.TENDER_SETTLEMENT: 16,
    EventType.EXERCISE: 17, EventType.REDEMPTION: 18, EventType.MATURITY: 19, EventType.MISSED_PAYMENT: 20,
    EventType.GRACE_PERIOD_END: 21, EventType.DEFAULT: 22, EventType.ACCELERATION: 23, EventType.CLAIM_REGISTERED: 24,
    EventType.RECOVERY_DISTRIBUTION: 25, EventType.RESTRUCTURING: 26, EventType.EXCHANGE: 27, EventType.WRITE_DOWN: 28,
    EventType.CONVERSION: 29, EventType.CREDIT_EVENT: 30, EventType.DELIVERY: 31, EventType.TRIGGER_TEST: 32,
    EventType.SETTLEMENT: 40, EventType.SETTLEMENT_FAIL: 41, EventType.VARIATION_MARGIN: 42, EventType.MARGIN_CALL: 43,
    EventType.COLLATERAL_TRANSFER: 44, EventType.REPO_OPEN: 45, EventType.REPO_ROLL: 46, EventType.REPO_CLOSE: 47,
    EventType.BORROW_OPEN: 48, EventType.BORROW_RECALL: 49, EventType.BORROW_CLOSE: 50, EventType.MANUFACTURED_PAYMENT: 51,
    EventType.FUNDING: 52, EventType.FUNDING_BREACH: 53, EventType.FORCED_ACTION: 54,
    EventType.MARK: 60, EventType.EXERCISE_NOTICE: 61, EventType.TENDER_OFFER: 62, EventType.PUT_NOTICE: 63, EventType.TENDER_ELECTION: 64,
    EventType.ISSUANCE: 65, EventType.AUCTION_ALLOCATION: 66,
    EventType.DECISION: 70, EventType.ORDER_CREATED: 71, EventType.ORDER_APPROVED: 72, EventType.ORDER_ROUTED: 73,
    EventType.QUOTE_REQUESTED: 74, EventType.QUOTE_RECEIVED: 75, EventType.QUOTE_DECLINED: 76, EventType.ORDER_ACCEPTED: 77,
    EventType.ORDER_PARTIALLY_FILLED: 78, EventType.ORDER_FILLED: 79, EventType.TRADE: 80, EventType.ORDER_CANCELED: 81,
    EventType.ORDER_EXPIRED: 82, EventType.ORDER_REJECTED: 83, EventType.CONSTRAINT_RELAXED: 84, EventType.CAPITAL_FLOW: 90,
    EventType.FEE: 91, EventType.TAX: 92, EventType.RUN_STATUS: 99,
}


def order_events(events: list[Event]) -> list[Event]:
    """Deterministic ordering: economic time, then precedence, then insertion sequence / id."""
    return sorted(events, key=lambda e: (e.time.economic_ns, PRECEDENCE.get(e.type, 50), e.seq, e.id))


@dataclass
class Obligation:
    date: date
    kind: str                          # COUPON | PRINCIPAL | REDEMPTION | FEE | RECOVERY
    amount_per_100: float              # per 100 original face (or per unit for non-bond contracts)
    currency: str
    status: str = "SCHEDULED"          # SCHEDULED | ENTITLED | PAID | MISSED | CANCELLED | DEFERRED
    note: str = ""


@dataclass
class TenderOffer:
    announce: date
    expiry: date
    settlement: date
    price: float                       # clean per 100
    max_fraction: float | None = None  # proration cap on the amount accepted
    early_premium: float = 0.0
    consent: bool = False


@dataclass
class DefaultState:
    missed_payment_date: date | None = None
    grace_end: date | None = None
    legal_default_date: date | None = None
    accelerated: bool = False
    claim_per_100: float = 0.0         # principal + accrued at default (contractual claim)
    distributions: list[tuple[date, float, str]] = field(default_factory=list)   # (date, per 100 original, kind)
    discharged: bool = False


@dataclass
class InstrumentState:
    symbol: str
    state: ContractState = ContractState.OUTSTANDING
    factor: float = 1.0                # fraction of original face outstanding (after calls, paydowns, write-downs)
    index_ratio: float = 1.0
    current_coupon: float | None = None
    deferred_coupons: list[tuple[date, float]] = field(default_factory=list)  # (date, per 100) cumulative deferrals
    pik_added: float = 0.0             # extra original face created by PIK (per 100 original)
    drawn_fraction: float = 1.0        # loans: drawn / committed
    call_notice: tuple[date, float] | None = None
    tender: TenderOffer | None = None
    default: DefaultState | None = None
    amendments: list[dict] = field(default_factory=list)
    last_event_ns: int = 0
    obligations: list[Obligation] = field(default_factory=list)

    def clone(self) -> InstrumentState:
        return replace(self, deferred_coupons=list(self.deferred_coupons), amendments=list(self.amendments), obligations=list(self.obligations), default=(replace(self.default, distributions=list(self.default.distributions)) if self.default else None))


@dataclass
class PositionChange:
    kind: str                          # SCALE | CLOSE | NEW | CONVERT
    fraction: float = 1.0              # for SCALE: fraction of the position affected
    price: float | None = None         # per 100 current face (redemption price) or per unit
    new_instrument: str | None = None
    ratio: float | None = None         # units of new instrument per unit of old
    cash_per_100: float = 0.0
    note: str = ""


@dataclass
class LifecycleResult:
    state: InstrumentState
    obligations: list[Obligation] = field(default_factory=list)
    position_changes: list[PositionChange] = field(default_factory=list)
    cash_per_100: float = 0.0          # cash paid to holders per 100 original face by this event
    cash_kind: str = ""                # COUPON | PRINCIPAL | REDEMPTION | RECOVERY | FEE
    emitted: list[Event] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class TenderPolicy:
    """Actor policy for tender offers: tender when the offer beats the mark by a threshold."""

    def __init__(self, threshold_pts: float = 0.25, participation: float = 1.0) -> None:
        self.threshold_pts = threshold_pts
        self.participation = participation

    def elect(self, offer: TenderOffer, mark_clean: float | None) -> float:
        if mark_clean is None:
            return 0.0
        return self.participation if offer.price >= mark_clean + self.threshold_pts else 0.0


class PutPolicy:
    def __init__(self, threshold_pts: float = 0.25) -> None:
        self.threshold_pts = threshold_pts

    def exercise(self, put_price: float, mark_clean: float | None) -> bool:
        return mark_clean is not None and mark_clean < put_price - self.threshold_pts


class BondLifecycleHandler:
    """Lifecycle for bond-like contracts (fixed, floating, step, PIK, hybrid, callable, tender, default)."""

    def __init__(self, tender_policy: TenderPolicy | None = None, put_policy: PutPolicy | None = None) -> None:
        self.tender_policy = tender_policy or TenderPolicy()
        self.put_policy = put_policy or PutPolicy()

    def initial_state(self, spec: BondSpec, as_of: date) -> InstrumentState:
        st = InstrumentState(spec.symbol)
        if as_of < spec.issue_date:
            st.state = ContractState.ANNOUNCED
        if spec.maturity is not None and as_of >= spec.maturity:
            st.state = ContractState.MATURED
            st.factor = 0.0
        return st

    # ------------------------------------------------------------------ dispatch
    def apply(self, state: InstrumentState, event: Event, spec: BondSpec, context: dict[str, Any] | None = None) -> LifecycleResult:
        ctx = context or {}
        st = state.clone()
        st.last_event_ns = event.time.economic_ns
        p = event.payload
        t = event.type
        res = LifecycleResult(st)
        if st.state in (ContractState.MATURED, ContractState.REDEEMED, ContractState.EXCHANGED, ContractState.DISCHARGED) and t not in (EventType.CORRECTION, EventType.RECOVERY_DISTRIBUTION):
            res.notes.append(f"ignored {t.value}: contract is {st.state.value}")
            return res
        handler = getattr(self, f"_on_{t.value.lower()}", None)
        if handler is None:
            res.notes.append(f"no lifecycle effect for {t.value}")
            return res
        return handler(st, p, spec, ctx, res, event)

    # ------------------------------------------------------------------ issuance / fixings / accrual
    def _on_issuance(self, st, p, spec, ctx, res, ev):
        st.state = ContractState.OUTSTANDING
        return res

    def _on_fixing(self, st, p, spec, ctx, res, ev):
        return res  # fixings live in the market state; RATE_RESET applies them to the contract

    def _on_rate_reset(self, st, p, spec, ctx, res, ev):
        rate = float(p["rate"])
        ft = spec.floating
        if ft is not None:
            all_in = rate + ft.spread_bp / 1e4
            if ft.floor is not None:
                all_in = max(all_in, ft.floor)
            if ft.cap is not None:
                all_in = min(all_in, ft.cap)
            st.current_coupon = all_in
        else:
            st.current_coupon = rate
        return res

    def _on_amendment(self, st, p, spec, ctx, res, ev):
        st.amendments.append(dict(p))
        if "coupon_rate" in p:
            st.current_coupon = float(p["coupon_rate"])
        res.notes.append(f"amendment effective {p.get('effective_date')}: {sorted(k for k in p if k != 'effective_date')}")
        return res

    # ------------------------------------------------------------------ coupons
    def _on_coupon_entitlement(self, st, p, spec, ctx, res, ev):
        amt = float(p["amount_per_100"])
        if st.default is not None and st.state in (ContractState.DEFAULTED, ContractState.ACCELERATED):
            res.notes.append("coupon entitlement recorded as contractual claim (defaulted)")
            st.default.claim_per_100 += amt
            return res
        res.obligations.append(Obligation(p["payment_date"] if isinstance(p.get("payment_date"), date) else ev.time and date.fromisoformat(str(p["payment_date"])), "COUPON", amt, spec.currency, "ENTITLED"))
        return res

    def _on_coupon_payment(self, st, p, spec, ctx, res, ev):
        amt = float(p["amount_per_100"])
        if st.state in (ContractState.DEFAULTED, ContractState.ACCELERATED, ContractState.GRACE_PERIOD) and not p.get("cure"):
            res.notes.append("payment while in default treated as recovery distribution")
            return self._on_recovery_distribution(st, {"amount_per_100": amt, "kind": "INTEREST"}, spec, ctx, res, ev)
        if p.get("cure") and st.state == ContractState.GRACE_PERIOD:
            st.state = ContractState.OUTSTANDING
            st.default = None
            res.notes.append("missed payment cured within grace period")
        res.cash_per_100, res.cash_kind = amt * st.factor if p.get("per_100_current", True) else amt, "COUPON"
        return res

    def _on_coupon_deferral(self, st, p, spec, ctx, res, ev):
        amt = float(p["amount_per_100"])
        cumulative = bool(p.get("cumulative", True))
        if cumulative:
            st.deferred_coupons.append((p.get("date") or ev.time.economic_ns, amt))
            res.notes.append(f"coupon deferred (cumulative): {amt:.4f} per 100")
        else:
            res.notes.append(f"coupon cancelled (non-cumulative): {amt:.4f} per 100 lost")
        return res

    def _on_pik_capitalisation(self, st, p, spec, ctx, res, ev):
        amt = float(p["amount_per_100"])  # per 100 original face, added to principal
        st.pik_added += amt
        st.factor += amt / 100.0
        res.position_changes.append(PositionChange("SCALE", fraction=1.0 + amt / 100.0, note="PIK adds principal"))
        return res

    # ------------------------------------------------------------------ principal
    def _on_factor_update(self, st, p, spec, ctx, res, ev):
        new_factor = float(p["factor"])
        if new_factor < st.factor - 1e-12:
            paid = st.factor - new_factor
            price = float(p.get("paydown_price", 100.0))
            res.cash_per_100, res.cash_kind = paid * price, "PRINCIPAL"
            res.position_changes.append(PositionChange("SCALE", fraction=new_factor / st.factor if st.factor else 0.0, price=price, note="factor paydown"))
        st.factor = new_factor
        return res

    def _on_index_ratio_update(self, st, p, spec, ctx, res, ev):
        st.index_ratio = float(p["index_ratio"])
        return res

    def _on_principal_reduction(self, st, p, spec, ctx, res, ev):
        return self._on_factor_update(st, {"factor": st.factor * (1.0 - float(p["fraction"])), "paydown_price": p.get("price", 100.0)}, spec, ctx, res, ev)

    def _on_drawdown(self, st, p, spec, ctx, res, ev):
        st.drawn_fraction = min(1.0, st.drawn_fraction + float(p["fraction"]))
        res.position_changes.append(PositionChange("SCALE", fraction=1.0, cash_per_100=-float(p["fraction"]) * 100.0, note="drawdown funded by lender"))
        res.cash_per_100, res.cash_kind = -float(p["fraction"]) * 100.0, "PRINCIPAL"
        return res

    def _on_repayment(self, st, p, spec, ctx, res, ev):
        frac = float(p["fraction"])
        st.drawn_fraction = max(0.0, st.drawn_fraction - frac)
        res.cash_per_100, res.cash_kind = frac * 100.0, "PRINCIPAL"
        return res

    def _on_partial_call(self, st, p, spec, ctx, res, ev):
        """Pro-rata partial call: ``fraction`` of the *outstanding* face redeemed at ``price`` plus accrued."""
        frac = float(p["fraction"])
        price = float(p.get("price", 100.0))
        accrued = float(p.get("accrued_per_100", 0.0))
        redeemed = st.factor * frac
        res.cash_per_100, res.cash_kind = redeemed * (price + accrued), "REDEMPTION"
        res.position_changes.append(PositionChange("SCALE", fraction=1.0 - frac, price=price, note=f"partial call {frac:.1%} at {price}"))
        st.factor -= redeemed
        return res

    # ------------------------------------------------------------------ notices, calls, puts, tenders
    def _on_exercise_notice(self, st, p, spec, ctx, res, ev):
        st.call_notice = (p["effective_date"] if isinstance(p["effective_date"], date) else date.fromisoformat(str(p["effective_date"])), float(p.get("price", 100.0)))
        st.state = ContractState.CALL_NOTICED
        res.notes.append(f"call notice: redemption {st.call_notice[0]} at {st.call_notice[1]}")
        return res

    def _on_exercise(self, st, p, spec, ctx, res, ev):
        price = float(p.get("price", (st.call_notice or (None, 100.0))[1]))
        accrued = float(p.get("accrued_per_100", 0.0))
        res.cash_per_100, res.cash_kind = st.factor * (price + accrued), "REDEMPTION"
        res.position_changes.append(PositionChange("CLOSE", price=price, note=p.get("kind", "call")))
        st.state = ContractState.REDEEMED
        st.factor = 0.0
        return res

    def _on_redemption(self, st, p, spec, ctx, res, ev):
        return self._on_exercise(st, p, spec, ctx, res, ev)

    def _on_maturity(self, st, p, spec, ctx, res, ev):
        price = float(p.get("price", 100.0))
        if spec.coupon_type == CouponType.INFLATION_LINKED and spec.inflation is not None:
            floor = 1.0 if spec.inflation.deflation_floor else 0.0
            price = 100.0 * max(st.index_ratio, floor) / max(st.index_ratio, 1e-9) if st.index_ratio else price
            res.notes.append("linker redemption at max(indexed principal, floor)")
        res.cash_per_100, res.cash_kind = st.factor * price, "REDEMPTION"
        res.position_changes.append(PositionChange("CLOSE", price=price, note="maturity"))
        st.state = ContractState.MATURED
        st.factor = 0.0
        return res

    def _on_put_notice(self, st, p, spec, ctx, res, ev):
        put_price = float(p.get("price", 100.0))
        if self.put_policy.exercise(put_price, ctx.get("mark_clean")):
            res.notes.append("put exercised by policy")
            res.emitted.append(Event(EventType.EXERCISE, ev.time, ev.source, spec.symbol, payload={"price": put_price, "kind": "put"}, caused_by=(ev.id,), observed=False))
        else:
            res.notes.append("put not exercised by policy")
        return res

    def _on_tender_offer(self, st, p, spec, ctx, res, ev):
        offer = TenderOffer(p["announce"], p["expiry"], p["settlement"], float(p["price"]), p.get("max_fraction"), float(p.get("early_premium", 0.0)), bool(p.get("consent", False)))
        st.tender, st.state = offer, ContractState.TENDER_OPEN
        return res

    def _on_tender_election(self, st, p, spec, ctx, res, ev):
        if st.tender is None:
            res.notes.append("election without an open tender ignored")
            return res
        elected = float(p["fraction"]) if "fraction" in p else self.tender_policy.elect(st.tender, ctx.get("mark_clean"))
        st.amendments.append({"tender_elected_fraction": elected})
        res.notes.append(f"tender election: {elected:.1%}")
        return res

    def _on_tender_settlement(self, st, p, spec, ctx, res, ev):
        if st.tender is None:
            return res
        elected = next((a["tender_elected_fraction"] for a in reversed(st.amendments) if "tender_elected_fraction" in a), 0.0)
        accepted = elected
        if st.tender.max_fraction is not None and elected > st.tender.max_fraction:
            accepted = st.tender.max_fraction
            res.notes.append(f"tender prorated: elected {elected:.1%}, accepted {accepted:.1%}")
        accrued = float(p.get("accrued_per_100", 0.0))
        price = st.tender.price + st.tender.early_premium
        res.cash_per_100, res.cash_kind = st.factor * accepted * (price + accrued), "REDEMPTION"
        if accepted > 0:
            res.position_changes.append(PositionChange("SCALE", fraction=1.0 - accepted, price=price, note="tendered"))
        st.factor *= 1.0 - accepted
        st.tender = None
        st.state = ContractState.OUTSTANDING if st.factor > 1e-12 else ContractState.REDEEMED
        return res

    # ------------------------------------------------------------------ default lifecycle
    def _on_missed_payment(self, st, p, spec, ctx, res, ev):
        st.default = st.default or DefaultState()
        st.default.missed_payment_date = p["date"]
        st.default.grace_end = p.get("grace_end")
        st.state = ContractState.GRACE_PERIOD
        res.notes.append(f"missed payment; grace until {p.get('grace_end')}")
        return res

    def _on_grace_period_end(self, st, p, spec, ctx, res, ev):
        if st.state == ContractState.GRACE_PERIOD:
            res.emitted.append(Event(EventType.DEFAULT, ev.time, ev.source, spec.symbol, payload={"date": p.get("date"), "reason": "grace period expired"}, caused_by=(ev.id,)))
        return res

    def _on_default(self, st, p, spec, ctx, res, ev):
        st.default = st.default or DefaultState()
        st.default.legal_default_date = p.get("date")
        st.default.claim_per_100 = 100.0 * st.factor + float(p.get("accrued_per_100", 0.0)) * st.factor
        st.state = ContractState.DEFAULTED
        res.notes.append(f"legal default; contractual claim {st.default.claim_per_100:.4f} per 100 original")
        return res

    def _on_acceleration(self, st, p, spec, ctx, res, ev):
        if st.default is None:
            st.default = DefaultState()
        st.default.accelerated = True
        st.state = ContractState.ACCELERATED
        return res

    def _on_claim_registered(self, st, p, spec, ctx, res, ev):
        if st.default is not None and "claim_per_100" in p:
            st.default.claim_per_100 = float(p["claim_per_100"])
        return res

    def _on_recovery_distribution(self, st, p, spec, ctx, res, ev):
        amt = float(p["amount_per_100"])
        kind = p.get("kind", "CASH")
        st.default = st.default or DefaultState()
        st.default.distributions.append((p.get("date"), amt, kind))
        res.cash_per_100, res.cash_kind = amt, "RECOVERY"
        remaining_claim = max(st.default.claim_per_100 - sum(a for _, a, k in st.default.distributions if k == "CASH"), 0.0)
        if p.get("final") or remaining_claim <= 1e-9:
            res.position_changes.append(PositionChange("CLOSE", price=0.0, note="final recovery distribution"))
            st.state, st.default.discharged = ContractState.DISCHARGED, True
        else:
            frac = amt / st.default.claim_per_100 if st.default.claim_per_100 else 0.0
            res.position_changes.append(PositionChange("SCALE", fraction=max(0.0, 1.0 - frac), note="partial recovery"))
        return res

    def _on_restructuring(self, st, p, spec, ctx, res, ev):
        st.state = ContractState.IN_RESTRUCTURING
        res.notes.append(f"restructuring: {p.get('note', '')}")
        return res

    def _on_exchange(self, st, p, spec, ctx, res, ev):
        """Old claim becomes cash plus new securities: ``new`` = [{instrument, ratio, value_per_100}]."""
        for leg in p.get("new", []):
            res.position_changes.append(PositionChange("NEW", new_instrument=leg["instrument"], ratio=float(leg.get("ratio", 1.0)), price=leg.get("value_per_100"), note="exchange"))
        res.cash_per_100, res.cash_kind = float(p.get("cash_per_100", 0.0)), "RECOVERY"
        res.position_changes.append(PositionChange("CLOSE", price=0.0, note="exchanged"))
        st.state, st.factor = ContractState.EXCHANGED, 0.0
        return res

    def _on_write_down(self, st, p, spec, ctx, res, ev):
        frac = float(p["fraction"])
        st.factor *= 1.0 - frac
        res.position_changes.append(PositionChange("SCALE", fraction=1.0 - frac, price=0.0, note="principal write-down"))
        res.notes.append(f"write-down of {frac:.1%} (no cash)")
        return res

    def _on_conversion(self, st, p, spec, ctx, res, ev):
        res.position_changes.append(PositionChange("CONVERT", new_instrument=p["instrument"], ratio=float(p["ratio"]), price=p.get("value_per_100"), note="conversion"))
        st.state, st.factor = ContractState.EXCHANGED, 0.0
        return res

    def _on_correction(self, st, p, spec, ctx, res, ev):
        res.notes.append(f"correction of {ev.corrects}: {p.get('note', '')}")
        return res


__all__ = [
    "PRECEDENCE",
    "BondLifecycleHandler",
    "ContractState",
    "DefaultState",
    "InstrumentState",
    "LifecycleResult",
    "Obligation",
    "PositionChange",
    "PutPolicy",
    "TenderOffer",
    "TenderPolicy",
    "order_events",
]
