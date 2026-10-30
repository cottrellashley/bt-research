"""Fixed income instrument definitions, day counts, schedules and analytics."""

from nautilus_fi.instruments.analytics import BondAnalytics, accrued_interest, analyse
from nautilus_fi.instruments.bond import (
    PAR_UNIT,
    AmortizationTerms,
    BondSpec,
    CallOption,
    FloatingTerms,
    Identifiers,
    InflationTerms,
    StepCoupon,
    face_to_units,
    units_to_face,
)
from nautilus_fi.instruments.calendar import BondCalendar, us_bond_market_calendar
from nautilus_fi.instruments.cashflows import Cashflow, CashflowType, MarketInputs, project_cashflows
from nautilus_fi.instruments.curves import NelsonSiegelSvensson, SpreadCurve, ZeroCurve, bootstrap_par_curve
from nautilus_fi.instruments.daycount import year_fraction
from nautilus_fi.instruments.enums import (
    AssetType,
    BusinessDayConvention,
    CouponType,
    DayCount,
    Frequency,
    MarkTier,
    PrincipalType,
    Seniority,
    is_investment_grade,
    rating_to_score,
    score_to_rating,
)
from nautilus_fi.instruments.schedule import CouponPeriod, add_months, generate_schedule

__all__ = [
    "PAR_UNIT",
    "AmortizationTerms",
    "AssetType",
    "BondAnalytics",
    "BondCalendar",
    "BondSpec",
    "BusinessDayConvention",
    "CallOption",
    "Cashflow",
    "CashflowType",
    "CouponPeriod",
    "CouponType",
    "DayCount",
    "FloatingTerms",
    "Frequency",
    "Identifiers",
    "InflationTerms",
    "MarkTier",
    "MarketInputs",
    "NelsonSiegelSvensson",
    "PrincipalType",
    "Seniority",
    "SpreadCurve",
    "StepCoupon",
    "ZeroCurve",
    "accrued_interest",
    "add_months",
    "analyse",
    "bootstrap_par_curve",
    "face_to_units",
    "generate_schedule",
    "is_investment_grade",
    "project_cashflows",
    "rating_to_score",
    "score_to_rating",
    "units_to_face",
    "us_bond_market_calendar",
    "year_fraction",
]
