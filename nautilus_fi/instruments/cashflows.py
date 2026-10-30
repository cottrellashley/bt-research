"""Contractual cashflow projection for every supported coupon / principal type.

Amounts are expressed **per 100 of original face**. Factor-driven instruments (MBS/ABS/CLO/loans,
sinkers, linkers) carry the factor (fraction of original face outstanding, or inflation index ratio)
at the start of each period so callers can convert to per-100-current-face conventions.

Point-in-time discipline: ``MarketInputs`` should only contain fixings, factors and index ratios that
were *known* on the analysis date; anything beyond is projected (flat fixings, CPR/CDR assumptions,
assumed inflation), never read from the future.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.daycount import year_fraction
from nautilus_fi.instruments.enums import CouponType, PrincipalType
from nautilus_fi.instruments.schedule import CouponPeriod, add_months


class CashflowType(str, Enum):
    COUPON = "COUPON"
    PRINCIPAL = "PRINCIPAL"        # scheduled / prepaid / sinking principal
    REDEMPTION = "REDEMPTION"      # maturity, call or put redemption
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class Cashflow:
    date: date
    amount: float                      # per 100 original face
    type: CashflowType
    accrual_start: date | None = None
    accrual_end: date | None = None
    rate: float | None = None          # annual coupon rate applied (decimal)
    factor: float = 1.0                # factor at period start


@dataclass
class MarketInputs:
    """Known market history available to the projection (point-in-time).

    ``fixings`` are term / in-advance fixings by index; ``daily_fixings`` are overnight rates for
    compounding in arrears; ``index_values`` are published compounded indices; ``cpi`` is the monthly
    price index keyed by (year, month) with ``cpi_base`` the reference CPI at the dated date.
    ``known_as_of`` restricts every lookup to what was knowable on that date.
    """

    fixings: dict[str, dict[date, float]] | Callable[[str, date], float | None] | None = None
    realized_factors: dict[date, float] | None = None
    index_ratios: dict[date, float] | None = None
    flat_forward_rate: float | None = None
    projected_inflation: float = 0.02
    default_date: date | None = None
    daily_fixings: dict[str, dict[date, float]] | None = None
    index_values: dict[str, dict[date, float]] | None = None
    cpi: dict[tuple[int, int], float] | None = None
    cpi_base: float | None = None
    known_as_of: date | None = None
    last_determination: object | None = field(default=None, repr=False)
    _factor_dates: list[date] = field(default_factory=list, repr=False)

    def fixing(self, index: str, d: date) -> float | None:
        if self.fixings is None:
            return None
        if callable(self.fixings):
            return self.fixings(index, d)
        series = self.fixings.get(index)
        if not series:
            return None
        best: date | None = None
        for k in series:
            if k <= d and (best is None or k > best):
                best = k
        return series[best] if best is not None else None

    def latest_fixing(self, index: str) -> float | None:
        if self.fixings is None or callable(self.fixings):
            return None
        series = self.fixings.get(index)
        if not series:
            return None
        return series[max(series)]

    def factor(self, d: date) -> tuple[date, float] | None:
        """Latest realized factor on or before ``d`` as ``(date, factor)``."""
        if not self.realized_factors:
            return None
        best: date | None = None
        for k in self.realized_factors:
            if k <= d and (best is None or k > best):
                best = k
        return (best, self.realized_factors[best]) if best is not None else None

    def index_ratio(self, d: date) -> tuple[date, float] | None:
        if not self.index_ratios:
            return None
        best: date | None = None
        for k in self.index_ratios:
            if k <= d and (best is None or k > best):
                best = k
        return (best, self.index_ratios[best]) if best is not None else None


def _scheduled_balance(n_months: int, rate_annual: float, term_months: int) -> float:
    """Remaining balance fraction of a level-pay loan after ``n_months``."""
    if term_months <= 0:
        return 1.0
    n = min(max(n_months, 0), term_months)
    r = rate_annual / 12.0
    if abs(r) < 1e-12:
        return 1.0 - n / term_months
    return ((1 + r) ** term_months - (1 + r) ** n) / ((1 + r) ** term_months - 1)


def _project_factor(bond: BondSpec, from_date: date, from_factor: float, to_date: date) -> tuple[float, float]:
    """Project the factor forward with scheduled amortization, CPR prepayments and CDR defaults.

    Returns ``(factor, loss_fraction)`` where ``loss_fraction`` (of original face) is defaulted
    principal written off without cash over the interval.
    """
    am = bond.amortization
    if am is None:
        return from_factor, 0.0
    months = (to_date.year - from_date.year) * 12 + (to_date.month - from_date.month)
    if to_date.day < from_date.day:
        months -= 1  # only completed months roll the factor
    if months <= 0:
        return from_factor, 0.0
    cpr = am.projected_cpr or 0.0
    cdr = am.projected_cdr or 0.0
    smm = 1.0 - (1.0 - cpr) ** (1.0 / 12.0) if cpr > 0 else 0.0
    mdr = 1.0 - (1.0 - cdr) ** (1.0 / 12.0) if cdr > 0 else 0.0
    factor = from_factor
    loss = 0.0
    if am.original_term_months:
        age0 = max(0, (from_date.year - bond.issue_date.year) * 12 + (from_date.month - bond.issue_date.month))
    for m in range(months):
        sched_ratio = 1.0
        if am.original_term_months:
            s0 = _scheduled_balance(age0 + m, bond.coupon_rate, am.original_term_months)
            s1 = _scheduled_balance(age0 + m + 1, bond.coupon_rate, am.original_term_months)
            sched_ratio = (s1 / s0) if s0 > 0 else 0.0
        defaulted = factor * mdr
        loss += defaulted * am.projected_severity
        factor = (factor - defaulted) * sched_ratio * (1.0 - smm)
    return max(factor, 0.0), loss


def _sinking_factor(bond: BondSpec, d: date) -> float:
    am = bond.amortization
    f = am.initial_factor if am else 1.0
    if am:
        for sink_date, fraction in am.sinking_schedule:
            if sink_date <= d:
                f -= fraction
    return max(f, 0.0)


def coupon_rate_for_period(bond: BondSpec, period: CouponPeriod, inputs: MarketInputs | None) -> float:
    """Annual coupon rate applying to a period (fixings resolved point-in-time)."""
    if bond.coupon_type in (CouponType.FIXED, CouponType.STEP, CouponType.INFLATION_LINKED):
        return bond.coupon_rate_on(period.accrual_start)
    if bond.coupon_type == CouponType.ZERO:
        return 0.0
    if bond.coupon_type == CouponType.FLOATING:
        ft = bond.floating
        assert ft is not None
        from nautilus_fi.instruments import floating as fl

        index, adj_bp, method_override = fl.effective_index(ft.index, period.accrual_start, ft.transition)
        method = (method_override.value if method_override else ft.method)
        known_as_of = inputs.known_as_of if inputs else None
        projection = inputs.flat_forward_rate if inputs else None
        determination = None
        if method == "COMPOUNDED_IN_ARREARS" and inputs is not None and inputs.daily_fixings and index in inputs.daily_fixings:
            lookback = ft.lookback_days or (ft.transition.lookback_days if ft.transition and method_override else 0)
            determination = fl.compounded_in_arrears(inputs.daily_fixings[index], period.accrual_start, period.accrual_end, bond.calendar, lookback, ft.lockout_days, ft.observation_shift, known_as_of=known_as_of, projection_rate=projection if projection is not None else (inputs.latest_fixing(index) if not callable(inputs.fixings) and inputs.fixings else None))
        elif method == "SIMPLE_AVERAGE_IN_ARREARS" and inputs is not None and inputs.daily_fixings and index in inputs.daily_fixings:
            determination = fl.simple_average_in_arrears(inputs.daily_fixings[index], period.accrual_start, period.accrual_end, bond.calendar, ft.lookback_days, known_as_of, projection)
        elif method == "INDEX" and inputs is not None and inputs.index_values and index in inputs.index_values:
            determination = fl.index_ratio_rate(inputs.index_values[index], period.accrual_start, period.accrual_end)
        if determination is not None:
            if inputs is not None:
                inputs.last_determination = determination
            base = determination.rate
        else:
            fixing_date = bond.calendar.add_business_days(period.accrual_end if ft.in_arrears else period.accrual_start, -ft.fixing_lag_days)
            fixing = inputs.fixing(index, fixing_date) if inputs else None
            if inputs is not None and known_as_of is not None and fixing_date > known_as_of:
                fixing = None
            if fixing is None and inputs is not None:
                fixing = inputs.flat_forward_rate if inputs.flat_forward_rate is not None else inputs.latest_fixing(index)
            if fixing is None:
                fixing = 0.0
            base = fixing
        if ft.floor_on_index and ft.floor is not None:
            base = max(base, ft.floor)
        rate = base + (ft.spread_bp + adj_bp) / 1e4
        if ft.floor is not None and not ft.floor_on_index:
            rate = max(rate, ft.floor)
        if ft.cap is not None:
            rate = min(rate, ft.cap)
        return rate
    raise ValueError(f"Unsupported coupon type {bond.coupon_type}")


def factor_at(bond: BondSpec, d: date, inputs: MarketInputs | None) -> float:
    """Factor (fraction of original face, or index ratio for linkers) applying at date ``d``."""
    if bond.coupon_type == CouponType.INFLATION_LINKED:
        if inputs is not None and inputs.cpi and inputs.cpi_base:
            from nautilus_fi.instruments import linkers as lk

            conv = lk.CONVENTIONS.get(bond.inflation.convention if bond.inflation else "US_TIPS", lk.US_TIPS)
            ratio, _status = lk.index_ratio(inputs.cpi, d, inputs.cpi_base, conv, inputs.known_as_of, inputs.projected_inflation)
            return ratio
        known = inputs.index_ratio(d) if inputs else None
        if known is None:
            return 1.0
        k_date, ratio = known
        years = max((d - k_date).days, 0) / 365.25
        infl = inputs.projected_inflation if inputs else 0.02
        return ratio * (1.0 + infl) ** years
    if bond.principal_type == PrincipalType.SINKING:
        return _sinking_factor(bond, d)
    if bond.principal_type != PrincipalType.AMORTIZING:
        return 1.0
    am = bond.amortization
    known = inputs.factor(d) if inputs else None
    if known is None:
        start_date, start_factor = bond.issue_date, (am.initial_factor if am else 1.0)
    else:
        start_date, start_factor = known
    if d <= start_date:
        return start_factor
    factor, _ = _project_factor(bond, start_date, start_factor, d)
    return factor


def project_cashflows(
    bond: BondSpec,
    as_of: date,
    inputs: MarketInputs | None = None,
    redemption_date: date | None = None,
    redemption_price: float | None = None,
) -> list[Cashflow]:
    """Cashflows (per 100 original face) paid strictly after ``as_of`` (the settlement date).

    ``redemption_date``/``redemption_price`` truncate the schedule for yield-to-call style analyses;
    accrued interest to the redemption date is paid along with the redemption amount.
    """
    inputs = inputs or MarketInputs()
    if inputs.default_date is not None and inputs.default_date <= as_of:
        return []
    flows: list[Cashflow] = []
    periods = bond.schedule
    end_date = redemption_date or bond.maturity
    final_price = redemption_price if redemption_price is not None else 100.0
    if bond.is_perpetual and redemption_date is None:
        end_date = None

    for p in periods:
        if p.accrual_end <= as_of:
            continue  # completed accrual periods belong to the previous holder
        if end_date is not None and p.accrual_start >= end_date:
            break
        accrual_end = p.accrual_end
        pay_date = p.payment_date
        truncated = end_date is not None and p.accrual_end > end_date
        if truncated:
            accrual_end = end_date
            pay_date = end_date
        if pay_date <= as_of:
            continue
        f_start = factor_at(bond, p.accrual_start, inputs)
        f_end = factor_at(bond, accrual_end, inputs)
        rate = coupon_rate_for_period(bond, p, inputs)
        if bond.coupon_type != CouponType.ZERO and rate != 0.0:
            yf = year_fraction(p.accrual_start, accrual_end, bond.day_count, p.ref_start, p.ref_end, bond.frequency)
            coupon_base = f_start if bond.coupon_type != CouponType.INFLATION_LINKED else f_end
            flows.append(Cashflow(pay_date, rate * yf * 100.0 * coupon_base, CashflowType.COUPON, p.accrual_start, accrual_end, rate, f_start))
        # principal between period start and end (amortization / sinking fund)
        if bond.principal_type in (PrincipalType.AMORTIZING, PrincipalType.SINKING) and not truncated:
            paid = max(f_start - f_end, 0.0)
            if paid > 1e-12:
                price = bond.amortization.paydown_price if bond.amortization else 100.0
                flows.append(Cashflow(pay_date, paid * price, CashflowType.PRINCIPAL, p.accrual_start, accrual_end, None, f_start))
        is_final = truncated or (end_date is not None and accrual_end == end_date) or (p is periods[-1] and not bond.is_perpetual)
        if is_final and end_date is not None:
            remaining = f_end
            if bond.coupon_type == CouponType.INFLATION_LINKED:
                floor = 1.0 if (bond.inflation and bond.inflation.deflation_floor) else 0.0
                remaining = max(f_end, floor) if redemption_date is None else f_end
            if remaining > 1e-12:
                flows.append(Cashflow(pay_date, remaining * final_price, CashflowType.REDEMPTION, None, None, None, f_end))
            break
    return flows


def period_coupon_amount(bond: BondSpec, period: CouponPeriod, inputs: MarketInputs | None) -> float:
    """Coupon paid at the end of ``period`` per 100 original face (used by the ledger)."""
    if bond.coupon_type == CouponType.ZERO:
        return 0.0
    rate = coupon_rate_for_period(bond, period, inputs)
    yf = year_fraction(period.accrual_start, period.accrual_end, bond.day_count, period.ref_start, period.ref_end, bond.frequency)
    base = factor_at(bond, period.accrual_end if bond.coupon_type == CouponType.INFLATION_LINKED else period.accrual_start, inputs)
    return rate * yf * 100.0 * base


__all__ = [
    "Cashflow",
    "CashflowType",
    "MarketInputs",
    "add_months",
    "coupon_rate_for_period",
    "factor_at",
    "period_coupon_amount",
    "project_cashflows",
]
