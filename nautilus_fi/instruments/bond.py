"""The generic bond specification and its mapping onto a Nautilus instrument.

Representation choice
---------------------
Nautilus v2 forbids futures-style (multiplier) instruments on cash accounts, and cash accounting is
essential for a bond ledger (cash leaves when you buy). Bonds are therefore represented as ``Equity``
instruments quoted in **price per 100** with quantity in **par units of 100 face**::

    quantity (units) = face / 100          notional = units * price = face * price / 100

For factor-driven instruments (MBS/ABS/CLO/loans, inflation linkers) the quantity is *original* face
units and the Nautilus price is the *factor-adjusted* clean price (clean price x factor), so position
notional always equals current-face market value. Helpers ``face_to_units`` / ``units_to_face``
convert at the boundary; strategies work in face amounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cached_property
from typing import Any

from nautilus_fi.instruments.calendar import BondCalendar, get_calendar
from nautilus_fi.instruments.enums import (
    AssetType,
    BusinessDayConvention,
    CouponType,
    DayCount,
    Frequency,
    PrincipalType,
    Seniority,
    rating_to_score,
)
from nautilus_fi.instruments.schedule import CouponPeriod, generate_schedule

PAR_UNIT = 100.0  # one Nautilus quantity unit == 100 of face


def face_to_units(face: float) -> int:
    return int(round(face / PAR_UNIT))


def units_to_face(units: float) -> float:
    return float(units) * PAR_UNIT


@dataclass
class Identifiers:
    cusip: str | None = None
    isin: str | None = None
    figi: str | None = None
    bbg_id: str | None = None
    ticker: str | None = None
    other: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CallOption:
    """A call (or put) right: redeem on ``date`` at clean ``price`` per 100 current face."""

    date: date
    price: float = 100.0


@dataclass(frozen=True)
class StepCoupon:
    effective_date: date
    rate: float  # annual decimal


@dataclass
class FloatingTerms:
    index: str = "SOFR"
    spread_bp: float = 0.0
    reset_frequency: Frequency | None = None   # defaults to coupon frequency
    fixing_lag_days: int = 2                   # business days before period start (in advance)
    in_arrears: bool = False                   # compounded in arrears (daily fixings over the period)
    floor: float | None = None                 # annual decimal on the all-in coupon
    cap: float | None = None
    method: str = "IN_ADVANCE"                 # IN_ADVANCE | COMPOUNDED_IN_ARREARS | SIMPLE_AVERAGE_IN_ARREARS | INDEX
    lookback_days: int = 0                     # observation lookback for in-arrears compounding
    lockout_days: int = 0
    observation_shift: bool = False
    transition: object | None = None           # instruments.floating.BenchmarkTransition
    floor_on_index: bool = False               # floor applies to the index (loans) rather than the all-in coupon

    def __post_init__(self) -> None:
        if self.in_arrears and self.method == "IN_ADVANCE":
            self.method = "COMPOUNDED_IN_ARREARS"


@dataclass
class InflationTerms:
    index: str = "CPI-U"
    base_index: float = 100.0
    lag_months: int = 3
    deflation_floor: bool = True               # principal at maturity floored at par (TIPS)
    convention: str = "US_TIPS"                # instruments.linkers.CONVENTIONS key


@dataclass
class AmortizationTerms:
    """Principal profile for sinking, amortizing and factor-driven instruments."""

    sinking_schedule: list[tuple[date, float]] = field(default_factory=list)  # (date, fraction of original face)
    paydown_price: float = 100.0               # price at which principal is returned
    initial_factor: float = 1.0
    original_term_months: int | None = None    # level-pay scheduled amortization when set
    projected_cpr: float | None = None         # annual prepayment speed for projections
    projected_cdr: float | None = None         # annual default rate (reduces factor without cash)
    projected_severity: float = 0.0            # loss severity on defaulted principal


@dataclass
class BondSpec:
    """Reference data for any fixed income instrument the backtester can hold."""

    symbol: str                                # primary identifier (CUSIP/ISIN/FIGI), Nautilus symbol
    issuer: str
    issue_date: date
    maturity: date | None                      # None for perpetuals
    coupon_rate: float = 0.0                   # annual decimal (0.05 == 5%)
    coupon_type: CouponType = CouponType.FIXED
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCount = DayCount.THIRTY_360
    currency: str = "USD"
    asset_type: AssetType = AssetType.CORPORATE
    principal_type: PrincipalType = PrincipalType.BULLET
    name: str = ""
    sector: str = ""
    country: str = "US"
    seniority: Seniority = Seniority.SENIOR_UNSECURED
    rating_initial: str | None = None
    dated_date: date | None = None             # first accrual date (defaults to issue date)
    first_coupon_date: date | None = None
    end_of_month: bool = False
    business_day_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING
    calendar_name: str = "US_BOND"
    settlement_lag_days: int | None = None     # None -> default by asset type / currency
    min_piece: float = 2_000.0
    increment: float = 1_000.0
    amount_outstanding: float | None = None
    is_144a: bool = False
    is_benchmark: bool = False
    step_schedule: list[StepCoupon] = field(default_factory=list)
    floating: FloatingTerms | None = None
    inflation: InflationTerms | None = None
    amortization: AmortizationTerms | None = None
    call_schedule: list[CallOption] = field(default_factory=list)
    put_schedule: list[CallOption] = field(default_factory=list)
    make_whole: bool = False
    price_precision: int = 4
    identifiers: Identifiers = field(default_factory=Identifiers)
    tags: dict[str, Any] = field(default_factory=dict)
    terms_source: str = "TERMS"               # TERMS: cashflows from contract terms; VENDOR: terms-free (vendor prices, accrued, analytics, supplied cashflows)

    # ------------------------------------------------------------------ derived
    def __post_init__(self) -> None:
        if self.dated_date is None:
            self.dated_date = self.issue_date
        if self.maturity is None:
            self.principal_type = PrincipalType.PERPETUAL
        if self.coupon_type == CouponType.ZERO:
            self.frequency = Frequency.ZERO
        if self.coupon_type == CouponType.FLOATING and self.floating is None:
            self.floating = FloatingTerms()
        if self.coupon_type == CouponType.INFLATION_LINKED and self.inflation is None:
            self.inflation = InflationTerms()
        if self.principal_type in (PrincipalType.AMORTIZING, PrincipalType.SINKING) and self.amortization is None:
            self.amortization = AmortizationTerms()

    @property
    def terms_free(self) -> bool:
        return self.terms_source == "VENDOR"

    @property
    def calendar(self) -> BondCalendar:
        return get_calendar(self.calendar_name)

    @property
    def is_perpetual(self) -> bool:
        return self.maturity is None

    @property
    def is_callable(self) -> bool:
        return bool(self.call_schedule)

    @property
    def is_factor_driven(self) -> bool:
        return self.principal_type in (PrincipalType.AMORTIZING, PrincipalType.SINKING) or (
            self.coupon_type == CouponType.INFLATION_LINKED
        )

    @property
    def rating_score_initial(self) -> int | None:
        return rating_to_score(self.rating_initial)

    @cached_property
    def schedule(self) -> list[CouponPeriod]:
        """Coupon periods to maturity (perpetuals: 100 years of periods)."""
        end = self.maturity or (self.dated_date.replace(year=self.dated_date.year + 100))
        return generate_schedule(
            self.dated_date,
            end,
            self.frequency,
            first_coupon_date=self.first_coupon_date,
            calendar=self.calendar,
            convention=self.business_day_convention,
            end_of_month=self.end_of_month,
        )

    def coupon_rate_on(self, d: date) -> float:
        """Fixed/step coupon rate applicable to accrual on date ``d``."""
        rate = self.coupon_rate
        for step in sorted(self.step_schedule, key=lambda s: s.effective_date):
            if step.effective_date <= d:
                rate = step.rate
        return rate

    def current_period(self, d: date) -> CouponPeriod | None:
        for p in self.schedule:
            if p.accrual_start <= d < p.accrual_end:
                return p
        return None

    def previous_coupon_date(self, d: date) -> date:
        p = self.current_period(d)
        return p.accrual_start if p else self.dated_date

    def next_coupon_date(self, d: date) -> date | None:
        p = self.current_period(d)
        return p.accrual_end if p else None

    def is_alive(self, d: date) -> bool:
        if d < self.issue_date:
            return False
        if self.maturity is not None and d >= self.maturity:
            return False
        return True

    def next_call(self, d: date) -> CallOption | None:
        future = [c for c in sorted(self.call_schedule, key=lambda c: c.date) if c.date > d]
        return future[0] if future else None

    def settlement_lag(self) -> int:
        if self.settlement_lag_days is not None:
            return self.settlement_lag_days
        from nautilus_fi.execution.settlement import default_settlement_lag  # local import: avoid cycle

        return default_settlement_lag(self.asset_type, self.currency)

    def settle_date(self, trade_date: date) -> date:
        return self.calendar.add_business_days(trade_date, self.settlement_lag())

    # ------------------------------------------------------------------ Nautilus mapping
    def instrument_id(self, venue: str = "OTC"):
        from nautilus_trader.model import InstrumentId, Symbol, Venue

        return InstrumentId(Symbol(self.symbol), Venue(venue))

    def to_instrument(self, venue: str = "OTC", ts_init: int = 0):
        """Build the Nautilus ``Equity`` used to represent this bond (see module docstring)."""
        from nautilus_trader.model import Currency, Equity, Price, Quantity, Symbol

        increment_units = max(1, face_to_units(self.increment))
        min_units = max(increment_units, face_to_units(self.min_piece))
        tick = 10 ** (-self.price_precision)
        return Equity(
            instrument_id=self.instrument_id(venue),
            raw_symbol=Symbol(self.symbol),
            currency=Currency.from_str(self.currency),
            price_precision=self.price_precision,
            price_increment=Price(tick, self.price_precision),
            lot_size=Quantity(increment_units, 0),
            min_quantity=Quantity(min_units, 0),
            isin=self.identifiers.isin,
            ts_event=ts_init,
            ts_init=ts_init,
            info={
                "asset_type": self.asset_type.value,
                "coupon_type": self.coupon_type.value,
                "coupon_rate": self.coupon_rate,
                "maturity": self.maturity.isoformat() if self.maturity else None,
                "issuer": self.issuer,
                "sector": self.sector,
                "par_unit": PAR_UNIT,
            },
        )

    def round_face(self, face: float) -> float:
        """Round a face amount to the bond's increment, honouring the minimum piece."""
        if abs(face) < self.min_piece:
            return 0.0
        sign = 1.0 if face > 0 else -1.0
        return sign * (round(abs(face) / self.increment) * self.increment)

    def remaining_years(self, d: date) -> float | None:
        if self.maturity is None:
            return None
        return max((self.maturity - d).days, 0) / 365.25

    def age_years(self, d: date) -> float:
        return max((d - self.issue_date).days, 0) / 365.25

    def describe(self) -> str:
        mat = self.maturity.isoformat() if self.maturity else "PERP"
        return f"{self.issuer} {self.coupon_rate * 100:.3f}% {mat} [{self.asset_type.value}/{self.coupon_type.value}]"


def business_days_ago(cal: BondCalendar, d: date, n: int) -> date:
    return cal.add_business_days(d, -n)


__all__ = [
    "PAR_UNIT",
    "AmortizationTerms",
    "BondSpec",
    "CallOption",
    "FloatingTerms",
    "Identifiers",
    "InflationTerms",
    "StepCoupon",
    "business_days_ago",
    "face_to_units",
    "units_to_face",
]

_ = timedelta  # keep import for type users
