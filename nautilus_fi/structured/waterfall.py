"""Deal-level waterfall (blueprint section 7): collateral cash -> fees -> interest by seniority with
over-collateralisation and interest-coverage tests -> principal (reinvestment, then sequential or
pro-rata) -> equity residual. The waterfall runs once per deal and allocates to held tranches, so
conservation holds: cash out equals cash in (plus or minus reserve movements)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Tranche:
    name: str
    original_balance: float
    coupon_fixed: float | None = None          # fixed annual rate, or
    index: str | None = "SOFR"                 # floating index
    spread_bp: float = 0.0
    seniority: int = 1                         # 1 = most senior
    kind: str = "DEBT"                         # DEBT | EQUITY
    oc_trigger: float | None = None            # minimum OC ratio (collateral / (this and senior balances))
    ic_trigger: float | None = None            # minimum IC ratio
    deferrable: bool = False                   # interest shortfall PIKs (mezzanine)
    floor: float | None = None


@dataclass
class Deal:
    deal_id: str
    tranches: list[Tranche]
    senior_fee_bp: float = 15.0                # on collateral balance (trustee + senior mgmt)
    subordinate_fee_bp: float = 35.0           # junior management fee (after debt interest)
    reinvestment_end: date | None = None
    principal_mode_after_reinvestment: str = "SEQUENTIAL"   # SEQUENTIAL | PRO_RATA
    reserve_target: float = 0.0
    incentive_fee_share: float = 0.0           # share of residual to manager above hurdle (ignored if 0)
    payment_frequency: int = 4

    def debt_tranches(self) -> list[Tranche]:
        return sorted([t for t in self.tranches if t.kind == "DEBT"], key=lambda t: t.seniority)

    def equity(self) -> Tranche | None:
        eq = [t for t in self.tranches if t.kind == "EQUITY"]
        return eq[0] if eq else None


@dataclass
class DealState:
    balances: dict[str, float]
    deferred_interest: dict[str, float] = field(default_factory=dict)
    reserve: float = 0.0
    cumulative_losses: float = 0.0
    reinvested_total: float = 0.0
    periods: int = 0

    @classmethod
    def initial(cls, deal: Deal) -> DealState:
        return cls({t.name: t.original_balance for t in deal.tranches}, {t.name: 0.0 for t in deal.tranches}, deal.reserve_target)

    def factor(self, tranche: str, deal: Deal) -> float:
        orig = next(t.original_balance for t in deal.tranches if t.name == tranche)
        return self.balances[tranche] / orig if orig else 0.0


@dataclass
class WaterfallResult:
    period_end: date
    available_interest: float
    available_principal: float
    fees: float
    interest_paid: dict[str, float]
    principal_paid: dict[str, float]
    deferred: dict[str, float]
    reinvested: float
    reserve_delta: float
    equity_cash: float
    tests: dict[str, dict]
    diverted_interest: float
    conservation_error: float
    notes: list[str] = field(default_factory=list)


def _tranche_rate(t: Tranche, fixings: dict[str, float]) -> float:
    if t.coupon_fixed is not None:
        return t.coupon_fixed
    idx = fixings.get(t.index or "", 0.0)
    if t.floor is not None:
        idx = max(idx, t.floor)
    return idx + t.spread_bp / 1e4


def run_waterfall_period(deal: Deal, state: DealState, period_end: date, collateral_balance_end: float, interest_collected: float, principal_collected: float, losses: float, fixings: dict[str, float], in_reinvestment: bool | None = None) -> WaterfallResult:
    """Allocate one period's collateral cash through the deal waterfall and update tranche state."""
    notes: list[str] = []
    months = 12 / deal.payment_frequency
    in_reinv = in_reinvestment if in_reinvestment is not None else (deal.reinvestment_end is not None and period_end <= deal.reinvestment_end)
    debt = deal.debt_tranches()
    interest_paid = {t.name: 0.0 for t in deal.tranches}
    principal_paid = {t.name: 0.0 for t in deal.tranches}
    deferred = {t.name: 0.0 for t in deal.tranches}
    tests: dict[str, dict] = {}
    state.cumulative_losses += losses

    # 1. fees from interest
    avail_i = interest_collected + state.reserve
    reserve_used = state.reserve
    state.reserve = 0.0
    fees = min(avail_i, collateral_balance_end * deal.senior_fee_bp / 1e4 * months / 12.0)
    avail_i -= fees

    # 2. senior interest in order, with coverage tests after each class
    diverted = 0.0
    avail_p = principal_collected
    for i, t in enumerate(debt):
        bal = state.balances[t.name]
        due = bal * _tranche_rate(t, fixings) * months / 12.0 + state.deferred_interest.get(t.name, 0.0)
        pay = min(avail_i, due)
        interest_paid[t.name] = pay
        avail_i -= pay
        short = due - pay
        if short > 1e-9:
            if t.deferrable:
                state.deferred_interest[t.name] = short
                deferred[t.name] = short
                notes.append(f"{t.name}: interest shortfall {short:,.2f} deferred")
            else:
                state.deferred_interest[t.name] = short
                notes.append(f"{t.name}: interest shortfall {short:,.2f} (non-deferrable: event of default risk)")
        else:
            state.deferred_interest[t.name] = 0.0
        # coverage tests at this class level
        senior_and_this = sum(state.balances[x.name] for x in debt[: i + 1])
        oc = collateral_balance_end / senior_and_this if senior_and_this else float("inf")
        due_senior = sum(state.balances[x.name] * _tranche_rate(x, fixings) * months / 12.0 for x in debt[: i + 1])
        ic = (interest_collected - fees) / due_senior if due_senior else float("inf")
        oc_pass = t.oc_trigger is None or oc >= t.oc_trigger
        ic_pass = t.ic_trigger is None or ic >= t.ic_trigger
        tests[t.name] = {"oc_ratio": oc, "oc_pass": oc_pass, "ic_ratio": ic, "ic_pass": ic_pass}
        if not (oc_pass and ic_pass):
            # divert remaining interest to pay down the senior-most tranches until cured
            needed = 0.0
            if not oc_pass and t.oc_trigger:
                needed = max(0.0, senior_and_this - collateral_balance_end / t.oc_trigger)
            divert = min(avail_i, needed if needed > 0 else avail_i)
            if divert > 0:
                diverted += divert
                avail_i -= divert
                for s in debt:
                    if divert <= 1e-9:
                        break
                    pay_p = min(divert, state.balances[s.name])
                    principal_paid[s.name] += pay_p
                    state.balances[s.name] -= pay_p
                    divert -= pay_p
                notes.append(f"{t.name}: coverage test failed (OC {oc:.3f}, IC {ic:.3f}); interest diverted to senior principal")

    # 3. subordinate fee then equity gets residual interest
    sub_fee = min(avail_i, collateral_balance_end * deal.subordinate_fee_bp / 1e4 * months / 12.0)
    fees += sub_fee
    avail_i -= sub_fee
    reserve_delta = 0.0
    if deal.reserve_target > 0 and avail_i > 0:
        topup = min(avail_i, max(0.0, deal.reserve_target - state.reserve))
        state.reserve += topup
        avail_i -= topup
        reserve_delta = topup - reserve_used
    else:
        reserve_delta = -reserve_used
    equity_cash = avail_i

    # 4. principal: reinvest during the reinvestment period, else sequential / pro-rata
    reinvested = 0.0
    if in_reinv and avail_p > 0:
        reinvested = avail_p
        state.reinvested_total += avail_p
        avail_p = 0.0
    elif avail_p > 0:
        if deal.principal_mode_after_reinvestment == "PRO_RATA":
            total = sum(state.balances[t.name] for t in debt)
            for t in debt:
                share = avail_p * state.balances[t.name] / total if total else 0.0
                pay_p = min(share, state.balances[t.name])
                principal_paid[t.name] += pay_p
                state.balances[t.name] -= pay_p
            avail_p -= sum(principal_paid[t.name] for t in debt) - diverted
        else:
            for t in debt:
                if avail_p <= 1e-9:
                    break
                pay_p = min(avail_p, state.balances[t.name])
                principal_paid[t.name] += pay_p
                state.balances[t.name] -= pay_p
                avail_p -= pay_p
    eq = deal.equity()
    if eq is not None:
        equity_cash += max(avail_p, 0.0)      # residual principal after debt is retired
        avail_p = 0.0
    # losses reduce the most junior balances first (equity then mezz) without cash
    remaining_loss = losses
    for t in sorted(deal.tranches, key=lambda x: -x.seniority):
        if remaining_loss <= 1e-9:
            break
        hit = min(remaining_loss, state.balances[t.name])
        state.balances[t.name] -= hit
        remaining_loss -= hit
    state.periods += 1
    cash_in = interest_collected + principal_collected + reserve_used
    cash_out = fees + sum(interest_paid.values()) + sum(principal_paid.values()) + reinvested + equity_cash + state.reserve
    return WaterfallResult(period_end, interest_collected, principal_collected, fees, interest_paid, principal_paid, deferred, reinvested, reserve_delta, equity_cash, tests, diverted, cash_out - cash_in, notes)


__all__ = ["Deal", "DealState", "Tranche", "WaterfallResult", "run_waterfall_period"]
