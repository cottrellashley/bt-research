"""Securities lending for short positions: availability, fees, rebates, recall risk and manufactured
payments. Unlimited frictionless shorting is never an unspoken default."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class BorrowTerms:
    lender: str = "PRIME"
    fee_bp: float = 40.0
    rebate_bp: float = 0.0
    availability: Callable[[str, date], float] | None = None   # face available to borrow
    recall_notice_days: int = 3
    recall_probability_daily: float = 0.0
    day_basis: int = 360


@dataclass
class BorrowPosition:
    borrow_id: str
    instrument: str
    quantity: float
    currency: str
    start: date
    terms: BorrowTerms
    recall_date: date | None = None
    status: str = "OPEN"                 # OPEN | RECALLED | CLOSED
    fees_paid: float = 0.0
    manufactured_paid: float = 0.0
    history: list[dict] = field(default_factory=list)


class BorrowBook:
    def __init__(self) -> None:
        self.positions: dict[str, BorrowPosition] = {}
        self._n = 0

    def open(self, d: date, instrument: str, quantity: float, currency: str, terms: BorrowTerms) -> BorrowPosition:
        if terms.availability is not None:
            avail = terms.availability(instrument, d) - self.borrowed(instrument)
            if quantity > avail + 1e-9:
                raise ValueError(f"Borrow unavailable: {quantity:,.0f} requested, {max(avail, 0):,.0f} available for {instrument}")
        self._n += 1
        pos = BorrowPosition(f"BORROW-{self._n}", instrument, quantity, currency, d, terms)
        pos.history.append({"date": d, "event": "OPEN", "quantity": quantity})
        self.positions[pos.borrow_id] = pos
        return pos

    def daily_fee(self, pos: BorrowPosition, market_value: float, days: int = 1) -> float:
        fee = market_value * (pos.terms.fee_bp - pos.terms.rebate_bp) / 1e4 * days / pos.terms.day_basis
        pos.fees_paid += fee
        return fee

    def manufactured_payment(self, pos: BorrowPosition, coupon_per_100: float) -> float:
        amount = pos.quantity * coupon_per_100 / 100.0
        pos.manufactured_paid += amount
        pos.history.append({"event": "MANUFACTURED", "amount": amount})
        return amount

    def recall(self, pos: BorrowPosition, d: date) -> date:
        """Lender recalls: we must return the securities (cover) by the notice deadline."""
        from datetime import timedelta

        pos.status = "RECALLED"
        pos.recall_date = d + timedelta(days=pos.terms.recall_notice_days)
        pos.history.append({"date": d, "event": "RECALL", "deadline": pos.recall_date})
        return pos.recall_date

    def close(self, pos: BorrowPosition, d: date) -> None:
        pos.status = "CLOSED"
        pos.history.append({"date": d, "event": "CLOSE"})

    def borrowed(self, instrument: str) -> float:
        return sum(p.quantity for p in self.positions.values() if p.status in ("OPEN", "RECALLED") and p.instrument == instrument)

    def recalled_due(self, d: date) -> list[BorrowPosition]:
        return [p for p in self.positions.values() if p.status == "RECALLED" and p.recall_date is not None and p.recall_date <= d]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{k: v for k, v in p.__dict__.items() if k not in ("terms", "history")} for p in self.positions.values()])


__all__ = ["BorrowBook", "BorrowPosition", "BorrowTerms"]
