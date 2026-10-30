"""Repo financing: rate, term, haircut, eligible collateral, specialness, substitution, margin and
rollover rules. Legal title moves to the cash lender but the repo seller keeps the economics, so
collateral coupons pass back during a funded position."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class RepoTerms:
    counterparty: str
    rate: float | None = None                 # fixed annual rate; None -> index + spread
    index: str | None = "SOFR"
    spread_bp: float = 10.0
    term_days: int = 1                        # 1 = overnight / open
    haircut: float = 0.02
    eligible: Callable[[str], bool] | None = None
    specialness_bp: float = 0.0               # reduces the repo rate for special collateral
    substitution_allowed: bool = True
    margin_threshold_pct: float = 0.005       # re-margin when collateral value moves beyond this
    rollover: str = "AUTO"                    # AUTO | MANUAL | FAIL
    rollover_fail_probability: float = 0.0
    day_basis: int = 360


@dataclass
class RepoPosition:
    repo_id: str
    instrument: str
    face: float
    currency: str
    collateral_value: float                  # dirty market value posted
    cash_borrowed: float
    start: date
    end: date
    rate: float
    terms: RepoTerms
    margin_posted: float = 0.0               # extra cash/collateral posted after margin calls
    accrued_interest: float = 0.0
    status: str = "OPEN"                     # OPEN | CLOSED | ROLLOVER_FAILED
    history: list[dict] = field(default_factory=list)


class RepoBook:
    def __init__(self, fixing: Callable[[str, date], float | None] | None = None) -> None:
        self.positions: dict[str, RepoPosition] = {}
        self._n = 0
        self._fixing = fixing

    def _rate(self, terms: RepoTerms, d: date) -> float:
        if terms.rate is not None:
            return terms.rate - terms.specialness_bp / 1e4
        base = self._fixing(terms.index, d) if (self._fixing and terms.index) else 0.0
        return (base or 0.0) + terms.spread_bp / 1e4 - terms.specialness_bp / 1e4

    def open(self, d: date, instrument: str, face: float, currency: str, dirty_value: float, terms: RepoTerms) -> RepoPosition:
        if terms.eligible is not None and not terms.eligible(instrument):
            raise ValueError(f"{instrument} is not eligible collateral for {terms.counterparty}")
        self._n += 1
        cash = dirty_value * (1.0 - terms.haircut)
        from datetime import timedelta

        pos = RepoPosition(f"REPO-{self._n}", instrument, face, currency, dirty_value, cash, d, d + timedelta(days=terms.term_days), self._rate(terms, d), terms)
        pos.history.append({"date": d, "event": "OPEN", "cash": cash, "rate": pos.rate})
        self.positions[pos.repo_id] = pos
        return pos

    def accrue(self, pos: RepoPosition, d_from: date, d_to: date) -> float:
        days = (d_to - d_from).days
        interest = (pos.cash_borrowed + pos.accrued_interest) * pos.rate * days / pos.terms.day_basis
        pos.accrued_interest += interest
        return interest

    def margin_call(self, pos: RepoPosition, d: date, new_dirty_value: float) -> float:
        """Collateral fell: post cash (positive) to restore the haircut; collateral rose: receive back (negative)."""
        required = pos.cash_borrowed / (1.0 - pos.terms.haircut)
        shortfall = required - new_dirty_value - pos.margin_posted
        if abs(shortfall) / max(required, 1.0) < pos.terms.margin_threshold_pct:
            return 0.0
        pos.margin_posted += shortfall
        pos.collateral_value = new_dirty_value
        pos.history.append({"date": d, "event": "MARGIN", "amount": shortfall})
        return shortfall

    def roll(self, pos: RepoPosition, d: date, fail: bool = False) -> bool:
        """Roll at term: settle accrued interest into the new balance; a failed roll closes the financing."""
        if fail or pos.terms.rollover == "FAIL":
            pos.status = "ROLLOVER_FAILED"
            pos.history.append({"date": d, "event": "ROLLOVER_FAILED"})
            return False
        from datetime import timedelta

        pos.cash_borrowed += pos.accrued_interest
        pos.accrued_interest = 0.0
        pos.rate = self._rate(pos.terms, d)
        pos.start, pos.end = d, d + timedelta(days=pos.terms.term_days)
        pos.history.append({"date": d, "event": "ROLL", "rate": pos.rate, "cash": pos.cash_borrowed})
        return True

    def close(self, pos: RepoPosition, d: date) -> float:
        """Repay cash plus accrued interest; margin posted comes back. Returns cash we pay."""
        repay = pos.cash_borrowed + pos.accrued_interest - pos.margin_posted
        pos.status = "CLOSED"
        pos.history.append({"date": d, "event": "CLOSE", "repay": repay})
        return repay

    def collateral_coupon_passthrough(self, pos: RepoPosition, coupon_cash: float) -> float:
        """Coupon on repoed collateral: received by the cash lender and manufactured back to us."""
        pos.history.append({"event": "COUPON_PASSTHROUGH", "amount": coupon_cash})
        return coupon_cash

    def financing_outstanding(self, currency: str | None = None) -> float:
        return sum(p.cash_borrowed + p.accrued_interest for p in self.positions.values() if p.status == "OPEN" and (currency is None or p.currency == currency))

    def encumbered(self, instrument: str) -> float:
        return sum(p.face for p in self.positions.values() if p.status == "OPEN" and p.instrument == instrument)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{k: v for k, v in p.__dict__.items() if k not in ("terms", "history")} | {"counterparty": p.terms.counterparty} for p in self.positions.values()])


__all__ = ["RepoBook", "RepoPosition", "RepoTerms"]
