"""Floating rate coupon determination (blueprint section 6 / 11).

Known fixings are distinguished from projected rates. Overnight compounding follows the contract:
lookback, lockout and observation shift, weekend/holiday weighting, caps and floors; a published
SOFR average or index is only used when the contract says so. Benchmark transitions (LIBOR -> SOFR
fallbacks) switch the index on the effective date with the ISDA spread adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from nautilus_fi.instruments.calendar import BondCalendar


class RateMethod(str, Enum):
    IN_ADVANCE = "IN_ADVANCE"                          # term rate fixed before the period (LIBOR / Term SOFR style)
    COMPOUNDED_IN_ARREARS = "COMPOUNDED_IN_ARREARS"    # daily overnight compounding over the accrual period
    SIMPLE_AVERAGE_IN_ARREARS = "SIMPLE_AVERAGE_IN_ARREARS"
    INDEX = "INDEX"                                    # published compounded index ratio (e.g. NY Fed SOFR Index)


class MissingFixing(LookupError):
    pass


@dataclass(frozen=True)
class BenchmarkTransition:
    old_index: str
    new_index: str
    effective_date: date
    spread_adjustment_bp: float = 0.0           # ISDA fallback spread adjustment
    new_method: RateMethod = RateMethod.COMPOUNDED_IN_ARREARS
    lookback_days: int = 2


@dataclass(frozen=True)
class RateDetermination:
    rate: float                                  # annualised index rate before the margin (decimal)
    status: str                                  # KNOWN | PARTIAL | PROJECTED
    method: RateMethod
    index: str
    observations: int = 0
    missing: int = 0
    note: str = ""


def _business_days(cal: BondCalendar, start: date, end: date) -> list[date]:
    """Business days in [start, end)."""
    out = []
    d = start
    while d < end:
        if cal.is_business_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _latest_on_or_before(fixings: dict[date, float], d: date) -> tuple[date, float] | None:
    best = None
    for k in fixings:
        if k <= d and (best is None or k > best):
            best = k
    return (best, fixings[best]) if best is not None else None


def compounded_in_arrears(
    fixings: dict[date, float],
    accrual_start: date,
    accrual_end: date,
    calendar: BondCalendar,
    lookback_days: int = 0,
    lockout_days: int = 0,
    observation_shift: bool = False,
    day_basis: int = 360,
    known_as_of: date | None = None,
    projection_rate: float | None = None,
) -> RateDetermination:
    """Daily compounded overnight rate for [accrual_start, accrual_end).

    Each business day's rate applies for the calendar days until the next business day. With a
    lookback the rate observed ``lookback_days`` business days earlier is used; with observation
    shift the weights come from the observation period too. A lockout freezes the rate observed
    ``lockout_days`` business days before the period end. Fixings dated after ``known_as_of`` are not
    knowable: those days use ``projection_rate`` (or the last known rate) and the status becomes
    PARTIAL / PROJECTED.
    """
    days = _business_days(calendar, accrual_start, accrual_end)
    if not days:
        raise ValueError("empty accrual period")
    growth = 1.0
    missing = 0
    used = 0
    lockout_start_idx = len(days) - lockout_days if lockout_days > 0 else None
    locked_rate: float | None = None
    for i, d in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else accrual_end
        obs_date = calendar.add_business_days(d, -lookback_days) if lookback_days else d
        if observation_shift and lookback_days:
            obs_next = calendar.add_business_days(nxt, -lookback_days)
            weight = (obs_next - obs_date).days
        else:
            weight = (nxt - d).days
        if lockout_start_idx is not None and i >= lockout_start_idx and locked_rate is not None:
            rate = locked_rate
        else:
            fx = fixings.get(obs_date)
            knowable = known_as_of is None or obs_date <= known_as_of
            if fx is None or not knowable:
                missing += 1
                if projection_rate is not None:
                    fx = projection_rate
                else:
                    last = _latest_on_or_before({k: v for k, v in fixings.items() if known_as_of is None or k <= known_as_of}, obs_date)
                    if last is None:
                        raise MissingFixing(f"no fixing available for {obs_date}")
                    fx = last[1]
            else:
                used += 1
            rate = fx
            if lockout_start_idx is not None and i == lockout_start_idx - 1:
                locked_rate = rate
        growth *= 1.0 + rate * weight / day_basis
    total_days = (accrual_end - accrual_start).days
    annualised = (growth - 1.0) * day_basis / total_days
    status = "KNOWN" if missing == 0 else ("PARTIAL" if used > 0 else "PROJECTED")
    return RateDetermination(annualised, status, RateMethod.COMPOUNDED_IN_ARREARS, "", used, missing)


def simple_average_in_arrears(fixings: dict[date, float], accrual_start: date, accrual_end: date, calendar: BondCalendar, lookback_days: int = 0, known_as_of: date | None = None, projection_rate: float | None = None) -> RateDetermination:
    days = _business_days(calendar, accrual_start, accrual_end)
    total, missing, used, wsum = 0.0, 0, 0, 0
    for i, d in enumerate(days):
        nxt = days[i + 1] if i + 1 < len(days) else accrual_end
        w = (nxt - d).days
        obs = calendar.add_business_days(d, -lookback_days) if lookback_days else d
        fx = fixings.get(obs)
        if fx is None or (known_as_of is not None and obs > known_as_of):
            missing += 1
            last = _latest_on_or_before(fixings, obs)
            fx = projection_rate if projection_rate is not None else (last[1] if last else None)
            if fx is None:
                raise MissingFixing(f"no fixing for {obs}")
        else:
            used += 1
        total += fx * w
        wsum += w
    status = "KNOWN" if missing == 0 else ("PARTIAL" if used > 0 else "PROJECTED")
    return RateDetermination(total / wsum, status, RateMethod.SIMPLE_AVERAGE_IN_ARREARS, "", used, missing)


def index_ratio_rate(index_values: dict[date, float], start: date, end: date, day_basis: int = 360) -> RateDetermination:
    """Rate implied by a published compounded index (NY Fed SOFR Index method): (I_end / I_start - 1) * basis / days."""
    if start not in index_values or end not in index_values:
        raise MissingFixing(f"index value missing for {start} or {end}")
    days = (end - start).days
    return RateDetermination((index_values[end] / index_values[start] - 1.0) * day_basis / days, "KNOWN", RateMethod.INDEX, "", 2, 0)


def term_rate_in_advance(fixings: dict[date, float], accrual_start: date, calendar: BondCalendar, fixing_lag_days: int = 2, known_as_of: date | None = None, projection_rate: float | None = None) -> RateDetermination:
    fixing_date = calendar.add_business_days(accrual_start, -fixing_lag_days)
    knowable = known_as_of is None or fixing_date <= known_as_of
    fx = fixings.get(fixing_date)
    if fx is None or not knowable:
        last = _latest_on_or_before({k: v for k, v in fixings.items() if known_as_of is None or k <= known_as_of}, fixing_date)
        if projection_rate is not None:
            return RateDetermination(projection_rate, "PROJECTED", RateMethod.IN_ADVANCE, "", 0, 1, "projected flat")
        if last is None:
            raise MissingFixing(f"no fixing for {fixing_date}")
        return RateDetermination(last[1], "PROJECTED", RateMethod.IN_ADVANCE, "", 0, 1, f"carried from {last[0]}")
    return RateDetermination(fx, "KNOWN", RateMethod.IN_ADVANCE, "", 1, 0)


def apply_margin(determination: RateDetermination, spread_bp: float, floor: float | None = None, cap: float | None = None) -> float:
    rate = determination.rate + spread_bp / 1e4
    if floor is not None:
        rate = max(rate, floor)
    if cap is not None:
        rate = min(rate, cap)
    return rate


def effective_index(index: str, period_start: date, transition: BenchmarkTransition | None) -> tuple[str, float, RateMethod | None]:
    """Index, spread adjustment (bp) and method override applying to a period after a benchmark transition."""
    if transition is None or period_start < transition.effective_date or index != transition.old_index:
        return index, 0.0, None
    return transition.new_index, transition.spread_adjustment_bp, transition.new_method


__all__ = [
    "BenchmarkTransition",
    "MissingFixing",
    "RateDetermination",
    "RateMethod",
    "apply_margin",
    "compounded_in_arrears",
    "effective_index",
    "index_ratio_rate",
    "simple_average_in_arrears",
    "term_rate_in_advance",
]
