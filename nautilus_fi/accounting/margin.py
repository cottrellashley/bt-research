"""The ledger's link to the margin venue: overlays and synthetic shorts in the book of record.

Nautilus keeps the margin account (collateral, realized P&L, daily funding) and the overlay
positions; this link mirrors them into the double-entry ledger so NAV, conservation and the
invariants cover cash bonds and overlays together:

* initial collateral is capital posted to the margin account (``COLLATERAL_POSTED``);
* every publication time the account's cash change (realized P&L, funding, expirations) is booked
  against realized P&L, and the overlays' unrealized P&L against derivative assets / liabilities;
* shorts carried as synthetic bond TRS pay a borrow fee, charged to the margin account;
* collateral moves between the venues by rule: topped up from settled cash when equity falls below
  ``min_ratio`` x initial margin, swept back above ``max_ratio`` (never below ``floor``). Transfers are
  booked immediately and applied to the margin account by its simulation module (in transit until then).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from nautilus_trader.model import Currency, Money, Venue

from nautilus_fi.core.events import EventSource, EventType
from nautilus_fi.core.ledger import Account, JournalEntry, Posting
from nautilus_fi.core.quantities import Money as LedgerMoney
from nautilus_fi.overlays.contracts import FinancedContract, OverlayBook


def _num(x) -> float:
    if x is None:
        return 0.0
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)


@dataclass
class MarginSettings:
    venue: str = "MARGIN"
    collateral: float = 0.0
    book: OverlayBook = field(default_factory=OverlayBook)
    min_ratio: float = 1.2
    target_ratio: float = 2.0
    max_ratio: float = 4.0
    floor: float | None = None                          # collateral never swept below this (None: the initial collateral)
    synthetic_shorts: bool | set[str] = False          # True: every bond; a set: those symbols
    short_suffix: str = "-TRS"
    borrow_fee_bp: float = 25.0
    short_financing_spread_bp: float = 0.0
    short_half_spread_bp: float = 10.0
    leverage: float = 10.0


class MarginLink:
    KEY = "MARGIN"

    def __init__(self, ledger, settings: MarginSettings) -> None:
        self.ledger = ledger
        self.s = settings
        self.venue = Venue(settings.venue)
        self.cache = None
        self.in_transit = 0.0
        self.queued: list[float] = []
        self.transfers: list[dict] = []
        self.fees: list[dict] = []
        self._last_fee_date: date | None = None
        self._initialised = False
        self.last: dict = {}
        self.otc_queued: list[float] = []                    # cash-venue side of transfers made between ledger passes

    def bind(self, cache) -> None:
        self.cache = cache

    # ------------------------------------------------------------------ state
    def margin_cash(self) -> float:
        acct = self.cache.account_for_venue(self.venue) if self.cache is not None else None
        if acct is None:
            return 0.0
        bal = acct.balance_total(Currency.from_str(self.ledger.base))
        return _num(bal) if bal is not None else 0.0

    def positions(self) -> list:
        if self.cache is None:
            return []
        return [p for p in self.cache.positions_open(self.venue, None, None, None, None) if abs(_num(p.signed_qty)) > 0]

    def units(self) -> dict[str, float]:
        return {p.instrument_id.symbol.value: _num(p.signed_qty) for p in self.positions()}

    def _mult(self, pos) -> float:
        inst = self.cache.instrument(pos.instrument_id)
        return _num(inst.multiplier) if inst is not None else 1.0

    def unrealized(self, d: date) -> float:
        u = 0.0
        for p in self.positions():
            px = self.s.book.price(p.instrument_id.symbol.value, d)
            if px is None:
                continue
            u += (px - _num(p.avg_px_open)) * _num(p.signed_qty) * self._mult(p)
        return u

    def notional(self, d: date) -> tuple[float, float, float]:
        gross = net = req = 0.0
        for p in self.positions():
            sym = p.instrument_id.symbol.value
            px = self.s.book.price(sym, d)
            if px is None:
                continue
            n = px * _num(p.signed_qty) * self._mult(p)
            gross += abs(n)
            net += n
            req += abs(n) * self.s.book.contracts[sym].margin_init if sym in self.s.book.contracts else 0.0
        return gross, net, req

    # ------------------------------------------------------------------ booking
    def _post(self, ts: int, postings: tuple, memo: str, payload: dict) -> None:
        ev = self.ledger._event(EventType.FUNDING, ts, None, payload, EventSource.FUNDING, observed=False)
        self.ledger.ledger.post(JournalEntry(ev.id, ts, tuple(p for p in postings if p.debit or p.credit), memo))

    def booked(self) -> float:
        return self.ledger.ledger.balance(Account.COLLATERAL_POSTED, self.KEY, self.ledger.base)

    def initialise(self, ts: int) -> None:
        if self._initialised:
            return
        self._initialised = True
        c = self.margin_cash()
        if abs(c) > 1e-9:
            ccy = self.ledger.base
            self._post(ts, (Posting.dr(Account.COLLATERAL_POSTED, LedgerMoney(c, ccy), self.KEY, "initial margin collateral"), Posting.cr(Account.CAPITAL, LedgerMoney(c, ccy), self.KEY, "initial margin collateral")), "margin collateral (capital)", {"kind": "MARGIN_COLLATERAL", "amount": c})

    def take_otc_adjustments(self) -> list[Money]:
        out = [Money(a, Currency.from_str(self.ledger.base)) for a in self.otc_queued if abs(a) > 1e-9]
        self.otc_queued = []
        return out

    def prefund(self, ts: int, d: date, initial_margin: float, available: float) -> float:
        """Top up collateral before new overlay orders are submitted: equity to ``target_ratio`` x the
        initial margin the targets need. Booked now; both venues apply it at their next pass."""
        equity = self.margin_cash() + self.in_transit + self.unrealized(d)
        move = min(max(self.s.target_ratio * initial_margin - equity, 0.0), max(available, 0.0))
        if move > 1e-6:
            ccy = self.ledger.base
            self._post(ts, (Posting.dr(Account.COLLATERAL_POSTED, LedgerMoney(move, ccy), self.KEY, "collateral prefund"), Posting.cr(Account.CASH_SETTLED, LedgerMoney(move, ccy), "", "collateral prefund")), "collateral prefund", {"kind": "COLLATERAL_PREFUND", "amount": move})
            self.queued.append(move)
            self.in_transit += move
            self.otc_queued.append(-move)
            self.transfers.append({"date": d, "amount": move, "equity_before": equity, "initial_margin": initial_margin, "kind": "PREFUND"})
        return move

    def take_margin_adjustments(self) -> list[Money]:
        out = [Money(a, Currency.from_str(self.ledger.base)) for a in self.queued if abs(a) > 1e-9]
        for a in self.queued:
            self.in_transit -= a
        self.queued = []
        return out

    def on_publish(self, ts: int, d: date, otc_available: float) -> list[Money]:
        """Book the margin account's P&L, charge borrow on shorts, apply the collateral rule. Returns
        the cash venue's adjustments (transfers)."""
        ccy = self.ledger.base
        self.initialise(ts)
        # realized P&L and funding: the account's cash change that the ledger has not booked yet
        diff = self.margin_cash() + self.in_transit - self.booked()
        if abs(diff) > 1e-6:
            self._post(ts, (Posting.dr(Account.COLLATERAL_POSTED, LedgerMoney(diff, ccy), self.KEY, "overlay realized P&L / funding"), Posting.cr(Account.REALIZED_PNL, LedgerMoney(diff, ccy), self.KEY, "overlay realized P&L / funding")), "overlay realized P&L", {"kind": "OVERLAY_REALIZED", "amount": diff})
        u = self.unrealized(d)
        ev = self.ledger._event(EventType.FUNDING, ts, None, {"kind": "OVERLAY_MTM", "unrealized": u}, EventSource.FUNDING, observed=False)
        self.ledger.policy.derivative_mark(ev.id, ts, self.KEY, ccy, u)
        # borrow fees on synthetic shorts
        days = (d - self._last_fee_date).days if self._last_fee_date is not None else 0
        self._last_fee_date = d
        if days > 0:
            for p in self.positions():
                sym = p.instrument_id.symbol.value
                c = self.s.book.contracts.get(sym)
                q = _num(p.signed_qty)
                if not isinstance(c, FinancedContract) or q >= 0 or c.borrow_fee_bp <= 0:
                    continue
                px = self.s.book.price(sym, d) or 0.0
                fee = abs(q) * px * self._mult(p) * c.borrow_fee_bp / 1e4 * days / 360.0
                if fee > 1e-9:
                    self._post(ts, (Posting.dr(Account.EXPENSE_BORROW, LedgerMoney(fee, ccy), sym, "borrow fee"), Posting.cr(Account.COLLATERAL_POSTED, LedgerMoney(fee, ccy), self.KEY, "borrow fee")), f"borrow fee {sym}", {"kind": "BORROW_FEE", "symbol": sym, "amount": fee})
                    self.queued.append(-fee)
                    self.in_transit -= fee
                    self.fees.append({"date": d, "symbol": sym, "fee": fee})
        # collateral rule
        gross, net, req = self.notional(d)
        equity = self.margin_cash() + self.in_transit + u
        floor = self.s.floor if self.s.floor is not None else self.s.collateral
        move = 0.0
        if req > 0 and equity < self.s.min_ratio * req:
            move = min(self.s.target_ratio * req - equity, max(otc_available, 0.0))
        elif equity > self.s.max_ratio * max(req, 1.0) and equity > floor:
            move = -min(equity - max(self.s.target_ratio * req, floor), max(self.margin_cash() + self.in_transit - req, 0.0))
        adjustments: list[Money] = []
        if abs(move) > 1e-6:
            self._post(ts, (Posting.dr(Account.COLLATERAL_POSTED, LedgerMoney(move, ccy), self.KEY, "collateral transfer"), Posting.cr(Account.CASH_SETTLED, LedgerMoney(move, ccy), "", "collateral transfer")), "collateral transfer", {"kind": "COLLATERAL_TRANSFER", "amount": move})
            self.queued.append(move)
            self.in_transit += move
            adjustments.append(Money(-move, Currency.from_str(ccy)))
            self.transfers.append({"date": d, "amount": move, "equity_before": equity, "initial_margin": req, "kind": "RULE"})
        self.last = {"margin_cash": self.margin_cash() + self.in_transit, "margin_unrealized": u, "margin_equity": self.margin_cash() + self.in_transit + u, "margin_required": req, "overlay_gross_notional": gross, "overlay_net_notional": net}
        return adjustments

    def transfers_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.transfers)


__all__ = ["MarginLink", "MarginSettings"]
