"""Coupon schedule generation (backward from maturity, optional stubs)."""

from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass
from datetime import date

from nautilus_fi.instruments.calendar import BondCalendar
from nautilus_fi.instruments.enums import BusinessDayConvention, Frequency


def add_months(d: date, months: int, end_of_month: bool = False) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last = _cal.monthrange(y, m)[1]
    if end_of_month and d.day == _cal.monthrange(d.year, d.month)[1]:
        return date(y, m, last)
    return date(y, m, min(d.day, last))


@dataclass(frozen=True)
class CouponPeriod:
    """One accrual period. ``ref_start``/``ref_end`` is the notional regular period (ICMA basis)."""

    accrual_start: date
    accrual_end: date
    payment_date: date
    ref_start: date
    ref_end: date
    is_stub: bool = False


def generate_schedule(
    accrual_start: date,
    maturity: date,
    frequency: Frequency,
    first_coupon_date: date | None = None,
    calendar: BondCalendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    end_of_month: bool = False,
) -> list[CouponPeriod]:
    """Generate coupon periods.

    Without ``first_coupon_date`` dates roll backward from maturity (the market standard), leaving a
    possible short/long first stub. With ``first_coupon_date`` the first period is an explicit stub
    and regular periods roll forward from it.
    """
    calendar = calendar or BondCalendar()
    if maturity <= accrual_start:
        raise ValueError("maturity must be after accrual start")
    if frequency == Frequency.ZERO:
        pay = calendar.adjust(maturity, convention)
        return [CouponPeriod(accrual_start, maturity, pay, accrual_start, maturity, False)]

    step = frequency.months
    unadjusted: list[date] = []
    if first_coupon_date is None:
        d = maturity
        while d > accrual_start:
            unadjusted.append(d)
            d = add_months(maturity, -step * len(unadjusted), end_of_month)
        unadjusted.append(accrual_start)
        unadjusted.reverse()
        # first period may be a stub: its reference period is [prev regular, first regular]
        periods: list[CouponPeriod] = []
        for i in range(1, len(unadjusted)):
            s, e = unadjusted[i - 1], unadjusted[i]
            is_stub = i == 1 and (add_months(e, -step, end_of_month) != s)
            ref_s = add_months(e, -step, end_of_month) if is_stub else s
            periods.append(CouponPeriod(s, e, calendar.adjust(e, convention), ref_s, e, is_stub))
        return periods

    if not (accrual_start < first_coupon_date <= maturity):
        raise ValueError("first_coupon_date must lie in (accrual_start, maturity]")
    unadjusted = [accrual_start, first_coupon_date]
    k = 1
    while True:
        nxt = add_months(first_coupon_date, step * k, end_of_month)
        if nxt >= maturity:
            break
        unadjusted.append(nxt)
        k += 1
    if unadjusted[-1] != maturity:
        unadjusted.append(maturity)
    periods = []
    for i in range(1, len(unadjusted)):
        s, e = unadjusted[i - 1], unadjusted[i]
        regular_start = add_months(e, -step, end_of_month)
        is_stub = regular_start != s
        periods.append(CouponPeriod(s, e, calendar.adjust(e, convention), regular_start if is_stub else s, e, is_stub))
    return periods
