"""Leveraged / private loans (blueprint section 6): drawn and undrawn balances, commitment fees, PIK
toggles, amendments, transfer restrictions and delayed settlement compensation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from nautilus_fi.instruments.calendar import BondCalendar
from nautilus_fi.instruments.daycount import year_fraction
from nautilus_fi.instruments.enums import DayCount


@dataclass
class PIKTerms:
    pik_bp: float            # portion of the margin paid in kind
    cash_bp: float           # portion paid in cash
    toggle: bool = True      # borrower may elect PIK


@dataclass
class LoanTerms:
    commitment: float
    drawn: float
    index: str = "SOFR"
    spread_bp: float = 350.0
    floor: float | None = 0.0
    undrawn_fee_bp: float = 50.0
    pik: PIKTerms | None = None
    amortization_pct_per_year: float = 1.0
    soft_call_price: float | None = 101.0
    soft_call_end: date | None = None
    transfer: str = "ASSIGNMENT"          # ASSIGNMENT | PARTICIPATION
    borrower_consent: bool = True
    restricted: bool = False              # private-side information restriction
    delayed_comp_days: int = 7            # T+7 par convention
    day_count: DayCount = DayCount.ACT_360

    @property
    def undrawn(self) -> float:
        return max(self.commitment - self.drawn, 0.0)

    def all_in_rate(self, index_rate: float) -> float:
        base = max(index_rate, self.floor) if self.floor is not None else index_rate
        return base + self.spread_bp / 1e4

    def cash_and_pik(self, index_rate: float, elect_pik: bool) -> tuple[float, float]:
        """Split the all-in rate into cash and PIK components under a toggle election."""
        total = self.all_in_rate(index_rate)
        if self.pik is None or not elect_pik:
            return total, 0.0
        pik_part = self.pik.pik_bp / 1e4
        return total - pik_part, pik_part


def commitment_fee(terms: LoanTerms, start: date, end: date) -> float:
    """Fee on the undrawn commitment over [start, end) in currency."""
    yf = year_fraction(start, end, terms.day_count)
    return terms.undrawn * terms.undrawn_fee_bp / 1e4 * yf


def delayed_compensation(trade_date: date, settle_date: date, face: float, all_in_rate: float, purchase_price: float, cost_of_carry_rate: float, calendar: BondCalendar, terms: LoanTerms) -> dict:
    """LSTA-style delayed compensation when settlement exceeds the target (T+7 par).

    Buyer receives interest on the loan from the target settlement date to actual settlement and pays
    cost of carry on the purchase price for the same period. Returns the components (positive = to
    buyer). This is an approximation of the LSTA formula and should be flagged as such.
    """
    target = calendar.add_business_days(trade_date, terms.delayed_comp_days)
    if settle_date <= target:
        return {"target_settle": target, "days": 0, "interest_to_buyer": 0.0, "cost_of_carry": 0.0, "net_to_buyer": 0.0}
    yf = year_fraction(target, settle_date, terms.day_count)
    interest = face * all_in_rate * yf
    carry = face * purchase_price / 100.0 * cost_of_carry_rate * yf
    return {"target_settle": target, "days": (settle_date - target).days, "interest_to_buyer": interest, "cost_of_carry": carry, "net_to_buyer": interest - carry}


def scheduled_amortization(terms: LoanTerms, periods_per_year: int = 4) -> float:
    """Fraction of commitment repaid each period under straight-line amortization (e.g. 1% p.a.)."""
    return terms.amortization_pct_per_year / 100.0 / periods_per_year


def prepayment_price(terms: LoanTerms, d: date) -> float:
    if terms.soft_call_price is not None and terms.soft_call_end is not None and d <= terms.soft_call_end:
        return terms.soft_call_price
    return 100.0


__all__ = ["LoanTerms", "PIKTerms", "commitment_fee", "delayed_compensation", "prepayment_price", "scheduled_amortization"]
