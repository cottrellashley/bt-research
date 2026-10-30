"""Cash and ownership states: trade-date exposure, unsettled receivables and payables, settled cash,
custody holdings, reservations for live orders and margin. A sale reduces price exposure before its
cash is usable; bridge financing exists only through an explicit policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from nautilus_fi.core.ledger import Account, Ledger


@dataclass
class PendingSettlement:
    trade_event_id: str
    instrument: str
    currency: str
    cash_amount: float            # signed: negative = we pay, positive = we receive
    quantity: float               # signed face/units delivered (+ receive securities)
    trade_date: date
    settle_date: date
    status: str = "PENDING"       # PENDING | SETTLED | FAILED
    fail_count: int = 0
    settled_on: date | None = None


class SettlementQueue:
    def __init__(self) -> None:
        self.items: list[PendingSettlement] = []

    def add(self, item: PendingSettlement) -> PendingSettlement:
        self.items.append(item)
        return item

    def due(self, d: date) -> list[PendingSettlement]:
        return [p for p in self.items if p.status == "PENDING" and p.settle_date <= d]

    def pending(self, currency: str | None = None, instrument: str | None = None) -> list[PendingSettlement]:
        return [p for p in self.items if p.status == "PENDING" and (currency is None or p.currency == currency) and (instrument is None or p.instrument == instrument)]

    def settle(self, item: PendingSettlement, d: date) -> None:
        item.status, item.settled_on = "SETTLED", d

    def fail(self, item: PendingSettlement, d: date, new_settle_date: date) -> None:
        """A settlement fail extends funding; the item stays pending with a new date."""
        item.fail_count += 1
        item.settle_date = new_settle_date
        item.status = "PENDING"

    def unsettled_cash(self, currency: str) -> tuple[float, float]:
        rec = sum(p.cash_amount for p in self.pending(currency) if p.cash_amount > 0)
        pay = -sum(p.cash_amount for p in self.pending(currency) if p.cash_amount < 0)
        return rec, pay

    def unsettled_quantity(self, instrument: str) -> float:
        return sum(p.quantity for p in self.pending(instrument=instrument))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.__dict__ for p in self.items])


@dataclass
class Reservation:
    ref: str
    kind: str                     # ORDER_CASH | ORDER_INVENTORY | MARGIN | FINANCING
    currency: str | None = None
    amount: float = 0.0
    instrument: str | None = None
    quantity: float = 0.0


class ReservationBook:
    """Cash, inventory and financing reservations released on fill, cancel or expiry."""

    def __init__(self) -> None:
        self._items: dict[str, Reservation] = {}

    def reserve_cash(self, ref: str, currency: str, amount: float) -> None:
        self._items[ref] = Reservation(ref, "ORDER_CASH", currency, amount)

    def reserve_inventory(self, ref: str, instrument: str, quantity: float) -> None:
        self._items[ref] = Reservation(ref, "ORDER_INVENTORY", None, 0.0, instrument, quantity)

    def reserve_margin(self, ref: str, currency: str, amount: float) -> None:
        self._items[ref] = Reservation(ref, "MARGIN", currency, amount)

    def release(self, ref: str) -> None:
        self._items.pop(ref, None)

    def reduce(self, ref: str, cash_amount: float = 0.0, quantity: float = 0.0) -> None:
        r = self._items.get(ref)
        if r is None:
            return
        r.amount = max(0.0, r.amount - cash_amount)
        r.quantity = max(0.0, r.quantity - quantity)
        if r.amount <= 1e-9 and r.quantity <= 1e-9:
            self._items.pop(ref, None)

    def reserved_cash(self, currency: str) -> float:
        return sum(r.amount for r in self._items.values() if r.currency == currency and r.kind in ("ORDER_CASH", "MARGIN", "FINANCING"))

    def reserved_inventory(self, instrument: str) -> float:
        return sum(r.quantity for r in self._items.values() if r.instrument == instrument)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class TieredCashRate:
    """Cash remuneration tiers per currency: list of (balance_threshold, annual_rate); negative balances pay ``borrow_rate``."""

    tiers: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    borrow_rate: dict[str, float] = field(default_factory=dict)
    day_basis: int = 360

    def rate(self, currency: str, balance: float) -> float:
        if balance < 0:
            return self.borrow_rate.get(currency, 0.0)
        tiers = sorted(self.tiers.get(currency, [(0.0, 0.0)]))
        rate = tiers[0][1]
        for threshold, r in tiers:
            if balance >= threshold:
                rate = r
        return rate

    def interest(self, currency: str, balance: float, days: int) -> float:
        return balance * self.rate(currency, balance) * days / self.day_basis


class CashView:
    """What is spendable: settled cash less reservations, plus bridge financing only if the policy allows."""

    def __init__(self, ledger: Ledger, queue: SettlementQueue, reservations: ReservationBook) -> None:
        self.ledger = ledger
        self.queue = queue
        self.reservations = reservations

    def settled(self, currency: str) -> float:
        return self.ledger.balance(Account.CASH_SETTLED, currency=currency)

    def unsettled(self, currency: str) -> tuple[float, float]:
        return self.queue.unsettled_cash(currency)

    def reserved(self, currency: str) -> float:
        return self.reservations.reserved_cash(currency)

    def collateral_posted(self, currency: str) -> float:
        return self.ledger.balance(Account.COLLATERAL_POSTED, currency=currency)

    def available(self, currency: str, bridge_allowance: float = 0.0) -> float:
        """Spendable now: settled cash less the payables already committed to pending settlements and
        less reservations, plus the policy's bridge allowance (0 by default). Unsettled sale proceeds
        never count (settle before spend)."""
        _, pay = self.unsettled(currency)
        return self.settled(currency) - pay - self.reserved(currency) + bridge_allowance

    def projected(self, currency: str) -> float:
        rec, pay = self.unsettled(currency)
        return self.settled(currency) + rec - pay - self.reserved(currency)

    def state(self, currency: str) -> dict:
        rec, pay = self.unsettled(currency)
        return {"currency": currency, "settled": self.settled(currency), "unsettled_receivable": rec, "unsettled_payable": pay, "reserved": self.reserved(currency), "collateral_posted": self.collateral_posted(currency)}


__all__ = ["CashView", "PendingSettlement", "Reservation", "ReservationBook", "SettlementQueue", "TieredCashRate"]
