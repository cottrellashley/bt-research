"""Bond analytics: accrued interest, yields, durations, spreads, key-rate durations.

Conventions
-----------
* Prices are **per 100 current face** (market convention); the factor is tracked separately.
* Yields use the street convention: compounding at the coupon frequency with time measured in coupon
  periods (fractional first period). Zero coupon bonds use the frequency passed (default semi-annual).
* Risk measures are computed numerically (1bp bumps), which keeps every instrument type on the same
  code path (fixed, floating, amortizing, callable to worst).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.cashflows import (
    Cashflow,
    CashflowType,
    MarketInputs,
    coupon_rate_for_period,
    factor_at,
    project_cashflows,
)
from nautilus_fi.instruments.curves import ZeroCurve, _bisect
from nautilus_fi.instruments.daycount import year_fraction
from nautilus_fi.instruments.enums import CouponType, DayCount, Frequency

BP = 1e-4


@dataclass
class BondAnalytics:
    settle: date
    clean_price: float
    accrued: float
    dirty_price: float
    factor: float = 1.0
    ytm: float | None = None
    ytw: float | None = None
    ytw_date: date | None = None
    mod_duration: float | None = None
    mac_duration: float | None = None
    convexity: float | None = None
    dv01: float | None = None                  # per 100 current face, per 1bp
    spread_duration: float | None = None
    z_spread_bp: float | None = None
    g_spread_bp: float | None = None
    discount_margin_bp: float | None = None
    wal: float | None = None
    years_to_maturity: float | None = None
    key_rate_durations: dict[float, float] = field(default_factory=dict)

    @property
    def dts(self) -> float | None:
        """Duration times spread (years x bp), the standard credit risk measure."""
        if self.spread_duration is None or self.z_spread_bp is None:
            return None
        return self.spread_duration * self.z_spread_bp


# ------------------------------------------------------------------------------------------ accrued
def accrued_interest(bond: BondSpec, settle: date, inputs: MarketInputs | None = None) -> float:
    """Accrued interest per 100 **current** face on ``settle``."""
    if bond.coupon_type == CouponType.ZERO:
        return 0.0
    period = bond.current_period(settle)
    if period is None:
        return 0.0
    rate = coupon_rate_for_period(bond, period, inputs)
    yf = year_fraction(period.accrual_start, settle, bond.day_count, period.ref_start, period.ref_end, bond.frequency)
    return rate * yf * 100.0


# ------------------------------------------------------------------------------------------ pricing
def _grouped(cashflows: Sequence[Cashflow]) -> list[tuple[date, float, float, date | None]]:
    """Group flows by date -> (date, total, principal_part, accrual_start_of_first_flow)."""
    by_date: dict[date, list[float]] = {}
    starts: dict[date, date | None] = {}
    for cf in cashflows:
        tot, prin = by_date.setdefault(cf.date, [0.0, 0.0])
        by_date[cf.date][0] = tot + cf.amount
        if cf.type != CashflowType.COUPON:
            by_date[cf.date][1] = prin + cf.amount
        if cf.date not in starts or (starts[cf.date] is None and cf.accrual_start is not None):
            starts[cf.date] = cf.accrual_start
    return [(d, v[0], v[1], starts[d]) for d, v in sorted(by_date.items())]


def _period_times(settle: date, grouped: list[tuple[date, float, float, date | None]], bond: BondSpec, freq: int) -> np.ndarray:
    """Time to each cashflow in coupon periods (street convention)."""
    if not grouped:
        return np.array([])
    first_date, _, _, first_start = grouped[0]
    if freq <= 0 or bond.coupon_type == CouponType.ZERO or first_start is None:
        years = np.array([year_fraction(settle, d, DayCount.ACT_ACT_ISDA) for d, *_ in grouped])
        f = max(freq, 1)
        return years * f
    period_days = max((first_date - first_start).days, 1)
    w = (first_date - settle).days / period_days
    # subsequent flows: count regular periods from the first flow date
    times = [w]
    for d, *_ in grouped[1:]:
        months = (d.year - first_date.year) * 12 + (d.month - first_date.month)
        k = months * freq / 12.0
        times.append(w + k)
    return np.asarray(times)


def pv_from_yield(amounts: np.ndarray, times: np.ndarray, y: float, freq: int) -> float:
    f = max(freq, 1)
    base = 1.0 + y / f
    if base <= 0:
        return float("inf")
    return float(np.sum(amounts / base ** times))


def yield_from_price(amounts: np.ndarray, times: np.ndarray, dirty: float, freq: int) -> float | None:
    if len(amounts) == 0 or dirty <= 0:
        return None
    f = lambda y: pv_from_yield(amounts, times, y, freq) - dirty
    try:
        return _bisect(f, -0.5, 3.0, tol=1e-10)
    except ValueError:
        return None


def _years(settle: date, dates: Sequence[date]) -> np.ndarray:
    return np.array([year_fraction(settle, d, DayCount.ACT_ACT_ISDA) for d in dates])


def pv_from_curve(amounts: np.ndarray, years: np.ndarray, curve: ZeroCurve, spread: float = 0.0) -> float:
    dfs = np.asarray(curve.df(years)) * np.exp(-spread * years)
    return float(np.sum(amounts * dfs))


def z_spread(amounts: np.ndarray, years: np.ndarray, dirty: float, curve: ZeroCurve) -> float | None:
    if len(amounts) == 0 or dirty <= 0:
        return None
    try:
        return _bisect(lambda s: pv_from_curve(amounts, years, curve, s) - dirty, -0.2, 2.0, tol=1e-10)
    except ValueError:
        return None


def weighted_average_life(grouped: list[tuple[date, float, float, date | None]], settle: date) -> float | None:
    prin = np.array([g[2] for g in grouped])
    if prin.sum() <= 0:
        return None
    yrs = _years(settle, [g[0] for g in grouped])
    return float(np.sum(prin * yrs) / prin.sum())


# ------------------------------------------------------------------------------------------ analyse
def analyse(
    bond: BondSpec,
    settle: date,
    clean_price: float,
    curve: ZeroCurve | None = None,
    inputs: MarketInputs | None = None,
    factor: float | None = None,
    krd_tenors: Sequence[float] | None = None,
) -> BondAnalytics:
    """Full analytics for a bond given a clean price per 100 current face."""
    inputs = inputs or MarketInputs()
    if factor is None:
        factor = factor_at(bond, settle, inputs)
    factor = factor if factor and factor > 0 else 1.0
    accrued = accrued_interest(bond, settle, inputs)
    dirty = clean_price + accrued
    freq = int(bond.frequency) if bond.frequency != Frequency.ZERO else 2
    result = BondAnalytics(settle=settle, clean_price=clean_price, accrued=accrued, dirty_price=dirty, factor=factor)
    result.years_to_maturity = bond.remaining_years(settle)

    flows = project_cashflows(bond, settle, inputs)
    if not flows:
        return result
    grouped = _grouped(flows)
    # convert per-100-original-face to per-100-current-face
    amounts = np.array([g[1] for g in grouped]) / factor
    times = _period_times(settle, grouped, bond, freq)
    years = _years(settle, [g[0] for g in grouped])

    result.wal = weighted_average_life([(g[0], g[1] / factor, g[2] / factor, g[3]) for g in grouped], settle)
    ytm = yield_from_price(amounts, times, dirty, freq)
    result.ytm = ytm
    if ytm is not None:
        p_up = pv_from_yield(amounts, times, ytm + BP, freq)
        p_dn = pv_from_yield(amounts, times, ytm - BP, freq)
        result.dv01 = (p_dn - p_up) / 2.0
        result.mod_duration = result.dv01 / dirty / BP
        result.mac_duration = result.mod_duration * (1.0 + ytm / freq)
        result.convexity = (p_up + p_dn - 2.0 * dirty) / (dirty * BP * BP)

    # yield to worst over the call schedule
    result.ytw, result.ytw_date = ytm, bond.maturity
    for call in bond.call_schedule:
        if call.date <= settle:
            continue
        cflows = project_cashflows(bond, settle, inputs, redemption_date=call.date, redemption_price=call.price)
        if not cflows:
            continue
        g = _grouped(cflows)
        a = np.array([x[1] for x in g]) / factor
        t = _period_times(settle, g, bond, freq)
        y = yield_from_price(a, t, dirty, freq)
        if y is not None and (result.ytw is None or y < result.ytw):
            result.ytw, result.ytw_date = y, call.date

    if curve is not None:
        zs = z_spread(amounts, years, dirty, curve)
        if zs is not None:
            result.z_spread_bp = zs / BP
            p_up = pv_from_curve(amounts, years, curve, zs + BP)
            p_dn = pv_from_curve(amounts, years, curve, zs - BP)
            result.spread_duration = (p_dn - p_up) / 2.0 / dirty / BP
            if krd_tenors:
                grid = curve.with_pillars(krd_tenors)  # common grid so the key-rate bumps partition
                for tenor in krd_tenors:
                    up = pv_from_curve(amounts, years, grid.bumped(tenor, 1.0), zs)
                    dn = pv_from_curve(amounts, years, grid.bumped(tenor, -1.0), zs)
                    result.key_rate_durations[float(tenor)] = (dn - up) / 2.0 / dirty / BP
        if ytm is not None:
            horizon = result.wal or result.years_to_maturity or 1.0
            result.g_spread_bp = (ytm - curve.par_yield(horizon, freq)) / BP

    if bond.coupon_type == CouponType.FLOATING and bond.floating is not None:
        result.discount_margin_bp = discount_margin(bond, settle, clean_price, inputs, factor)
    return result


def discount_margin(bond: BondSpec, settle: date, clean_price: float, inputs: MarketInputs | None, factor: float = 1.0) -> float | None:
    """Discount margin in the money-market convention of the floater's own day count.

    Projected coupons assume the index flat at its latest fixing; each period is discounted at
    ``(index + DM) * year_fraction`` compounded period by period, with the current stub period
    measured from settle to the next payment.
    """
    ft = bond.floating
    if ft is None:
        return None
    inputs = inputs or MarketInputs()
    index_rate = inputs.latest_fixing(ft.index)
    if index_rate is None:
        index_rate = inputs.flat_forward_rate if inputs.flat_forward_rate is not None else 0.0
    flat_inputs = MarketInputs(fixings=inputs.fixings, realized_factors=inputs.realized_factors, flat_forward_rate=index_rate)
    flows = project_cashflows(bond, settle, flat_inputs)
    if not flows:
        return None
    g = _grouped(flows)
    amounts = np.array([x[1] for x in g]) / factor
    dates = [x[0] for x in g]
    dirty = clean_price + accrued_interest(bond, settle, flat_inputs)
    # period year fractions in the bond's day count: stub from settle, then payment to payment
    yfs = []
    prev = settle
    for d in dates:
        yfs.append(max(year_fraction(prev, d, bond.day_count if bond.day_count != DayCount.ACT_ACT_ICMA else DayCount.ACT_365F), 1e-9))
        prev = d
    yfs = np.asarray(yfs)

    def pv(dm: float) -> float:
        dfs = np.cumprod(1.0 / (1.0 + (index_rate + dm) * yfs))
        return float(np.sum(amounts * dfs))

    try:
        dm = _bisect(lambda m: pv(m) - dirty, -0.2, 2.0, tol=1e-10)
    except ValueError:
        return None
    return dm / BP


def price_from_yield(bond: BondSpec, settle: date, ytm: float, inputs: MarketInputs | None = None, factor: float | None = None) -> float:
    """Clean price per 100 current face implied by a yield (street convention)."""
    inputs = inputs or MarketInputs()
    if factor is None:
        factor = factor_at(bond, settle, inputs) or 1.0
    flows = project_cashflows(bond, settle, inputs)
    if not flows:
        return 0.0
    g = _grouped(flows)
    amounts = np.array([x[1] for x in g]) / factor
    freq = int(bond.frequency) if bond.frequency != Frequency.ZERO else 2
    times = _period_times(settle, g, bond, freq)
    dirty = pv_from_yield(amounts, times, ytm, freq)
    return dirty - accrued_interest(bond, settle, inputs)


def price_from_spread(bond: BondSpec, settle: date, curve: ZeroCurve, spread_bp: float, inputs: MarketInputs | None = None, factor: float | None = None) -> float:
    """Clean price per 100 current face from a z-spread over ``curve`` (matrix pricing primitive)."""
    inputs = inputs or MarketInputs()
    if factor is None:
        factor = factor_at(bond, settle, inputs) or 1.0
    flows = project_cashflows(bond, settle, inputs)
    if not flows:
        return 0.0
    g = _grouped(flows)
    amounts = np.array([x[1] for x in g]) / factor
    years = _years(settle, [x[0] for x in g])
    dirty = pv_from_curve(amounts, years, curve, spread_bp * BP)
    return dirty - accrued_interest(bond, settle, inputs)


__all__ = [
    "BP",
    "BondAnalytics",
    "accrued_interest",
    "analyse",
    "discount_margin",
    "price_from_spread",
    "price_from_yield",
    "pv_from_yield",
    "yield_from_price",
    "z_spread",
]
