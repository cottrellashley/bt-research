"""TBA and dollar rolls (blueprint section 7): eligibility, class settlement calendar (SIFMA dates
supplied as data, with a flagged approximation fallback), 48-hour notification, drop and implied
financing, and delivery converting a forward position into specified pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from nautilus_fi.instruments.calendar import BondCalendar

# SIFMA class: A = 30y conventional FN/FH (UMBS); B = 15y; C = 30y GNMA; D = balloons / ARMs / other
PROGRAM_CLASS = {("FN", 30): "A", ("FH", 30): "A", ("UMBS", 30): "A", ("FN", 15): "B", ("FH", 15): "B", ("UMBS", 15): "B", ("GN", 30): "C", ("GN", 15): "C"}


@dataclass
class TBASettlementCalendar:
    """Class settlement dates by (year, month, class). Supply SIFMA's published dates; the fallback
    approximates (flagged) as the second business day on or after the 12th of the month."""

    dates: dict[tuple[int, int, str], date] = field(default_factory=dict)
    calendar: BondCalendar = field(default_factory=BondCalendar)
    approximate: bool = False

    def settlement_date(self, year: int, month: int, cls: str) -> date:
        key = (year, month, cls)
        if key in self.dates:
            return self.dates[key]
        self.approximate = True
        d = date(year, month, 12)
        while not self.calendar.is_business_day(d):
            d += timedelta(days=1)
        return self.calendar.add_business_days(d, 1)

    def notification_date(self, year: int, month: int, cls: str) -> date:
        return self.calendar.add_business_days(self.settlement_date(year, month, cls), -2)


@dataclass
class TBAContract:
    symbol: str
    program: str                      # FN | FH | GN | UMBS
    term_years: int
    coupon: float
    settlement_year: int
    settlement_month: int
    face: float = 1_000_000.0

    @property
    def cls(self) -> str:
        return PROGRAM_CLASS.get((self.program, self.term_years), "D")

    def eligible_pool(self, pool_program: str, pool_term: int, pool_coupon: float, pool_wam_months: int, max_wam_months: int = 360) -> bool:
        return pool_program == self.program and pool_term == self.term_years and abs(pool_coupon - self.coupon) < 1e-9 and pool_wam_months <= max_wam_months


@dataclass
class PoolAllocation:
    pool_symbol: str
    original_face: float
    factor: float
    coupon: float

    @property
    def current_face(self) -> float:
        return self.original_face * self.factor


def good_delivery(contract: TBAContract, allocations: list[PoolAllocation], variance: float = 0.0001, max_pools: int = 3) -> tuple[bool, str]:
    """Good delivery: total current face within +/- variance (0.01%) of the TBA face, pool count limit."""
    total = sum(a.current_face for a in allocations)
    if len(allocations) > max_pools:
        return False, f"too many pools ({len(allocations)} > {max_pools}) per million"
    if abs(total - contract.face) > contract.face * variance:
        return False, f"delivered face {total:,.0f} outside variance of {contract.face:,.0f}"
    if any(abs(a.coupon - contract.coupon) > 1e-9 for a in allocations):
        return False, "pool coupon mismatch"
    return True, "ok"


def deliver(contract: TBAContract, allocations: list[PoolAllocation], tba_price: float) -> list[dict]:
    """Convert a TBA forward into pool positions at the TBA price (price per 100 current face)."""
    ok, reason = good_delivery(contract, allocations)
    if not ok:
        raise ValueError(f"not good delivery: {reason}")
    return [{"pool": a.pool_symbol, "original_face": a.original_face, "factor": a.factor, "current_face": a.current_face, "price": tba_price, "cost": a.current_face * tba_price / 100.0} for a in allocations]


@dataclass
class DollarRoll:
    front: TBAContract
    back: TBAContract
    front_price: float
    back_price: float

    @property
    def drop(self) -> float:
        return self.front_price - self.back_price


def implied_roll_financing_rate(front_price: float, back_price: float, coupon: float, days: int, expected_paydown_fraction: float = 0.0, reinvest_rate: float = 0.0) -> float:
    """Financing rate implied by a dollar roll: give up one month of coupon and paydown, receive the drop.

    rate ~= (coupon carry + paydown effect - drop) / price * 360/days. Prepayment assumption matters:
    this is a scenario input, not an observation."""
    carry = coupon * days / 360.0 * 100.0
    paydown_gain = expected_paydown_fraction * (100.0 - front_price)
    drop = front_price - back_price
    return (carry + paydown_gain - drop) / front_price * 360.0 / days


def roll_breakeven_drop(front_price: float, coupon: float, financing_rate: float, days: int, expected_paydown_fraction: float = 0.0) -> float:
    carry = coupon * days / 360.0 * 100.0
    paydown_gain = expected_paydown_fraction * (100.0 - front_price)
    return carry + paydown_gain - front_price * financing_rate * days / 360.0


__all__ = ["PROGRAM_CLASS", "DollarRoll", "PoolAllocation", "TBAContract", "TBASettlementCalendar", "deliver", "good_delivery", "implied_roll_financing_rate", "roll_breakeven_drop"]
