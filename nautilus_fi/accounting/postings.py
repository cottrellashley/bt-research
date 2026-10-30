"""Accounting policy: economic events -> balanced journal entries (blueprint section 17).

Securities are carried at market (clean) with accrued interest in its own account, so a coupon
receipt moves a receivable to cash and creates no profit, and a principal paydown reduces the asset
and increases cash without touching P&L beyond the price difference. Lots carry cost basis so
realized / unrealized categories can change under a lot policy without changing economic NAV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nautilus_fi.core.ledger import Account, JournalEntry, Ledger, Posting
from nautilus_fi.core.quantities import Money


class LotPolicy(str, Enum):
    AVERAGE = "AVERAGE"
    FIFO = "FIFO"
    LIFO = "LIFO"
    HIGH_COST = "HIGH_COST"


@dataclass
class Lot:
    lot_id: str
    instrument: str
    currency: str
    quantity: float            # face (current for bullets, original for factor instruments) or units; signed
    cost: float                # clean cost basis in currency (signed like quantity)
    carrying: float            # clean carrying value at market in currency (signed)
    opened_ns: int

    @property
    def unrealized(self) -> float:
        return self.carrying - self.cost


class PositionBook:
    def __init__(self, policy: LotPolicy = LotPolicy.FIFO) -> None:
        self.policy = policy
        self.lots: dict[str, list[Lot]] = {}
        self._n = 0

    def quantity(self, instrument: str) -> float:
        return sum(l.quantity for l in self.lots.get(instrument, []))

    def carrying(self, instrument: str) -> float:
        return sum(l.carrying for l in self.lots.get(instrument, []))

    def cost(self, instrument: str) -> float:
        return sum(l.cost for l in self.lots.get(instrument, []))

    def open(self, instrument: str, currency: str, quantity: float, cost: float, ts_ns: int) -> Lot:
        self._n += 1
        lot = Lot(f"{instrument}#{self._n}", instrument, currency, quantity, cost, cost, ts_ns)
        self.lots.setdefault(instrument, []).append(lot)
        return lot

    def _order(self, lots: list[Lot]) -> list[Lot]:
        if self.policy == LotPolicy.FIFO:
            return sorted(lots, key=lambda l: l.opened_ns)
        if self.policy == LotPolicy.LIFO:
            return sorted(lots, key=lambda l: -l.opened_ns)
        if self.policy == LotPolicy.HIGH_COST:
            return sorted(lots, key=lambda l: -(l.cost / l.quantity if l.quantity else 0.0))
        return lots

    def close(self, instrument: str, quantity: float) -> tuple[float, float]:
        """Remove ``quantity`` (same sign as the position) and return (cost_released, carrying_released),
        both signed like the lots (positive for longs, negative for shorts)."""
        lots = self.lots.get(instrument, [])
        if not lots:
            raise ValueError(f"No lots for {instrument}")
        total_qty = sum(l.quantity for l in lots)
        if abs(quantity) > abs(total_qty) * (1.0 + 1e-9) + 1e-6:
            raise ValueError(f"Cannot close {quantity} of {instrument}: only {total_qty} held")
        if abs(quantity) > abs(total_qty):                 # float rounding on a full close: close exactly what is held
            quantity = total_qty
        if self.policy == LotPolicy.AVERAGE:
            frac = quantity / total_qty
            cost_rel = sum(l.cost for l in lots) * frac
            carry_rel = sum(l.carrying for l in lots) * frac
            for l in lots:
                l.quantity -= l.quantity * frac
                l.cost -= l.cost * frac
                l.carrying -= l.carrying * frac
        else:
            remaining = quantity
            cost_rel = carry_rel = 0.0
            for l in self._order(lots):
                if abs(remaining) < 1e-12:
                    break
                take = l.quantity if abs(l.quantity) <= abs(remaining) else remaining
                frac = take / l.quantity if l.quantity else 0.0
                cost_rel += l.cost * frac
                carry_rel += l.carrying * frac
                l.quantity -= take
                l.cost -= l.cost * frac
                l.carrying -= l.carrying * frac
                remaining -= take
        self.lots[instrument] = [l for l in lots if abs(l.quantity) > 1e-9]
        return cost_rel, carry_rel

    def revalue(self, instrument: str, new_carrying: float) -> float:
        """Set total carrying value (pro-rata across lots); returns the change."""
        lots = self.lots.get(instrument, [])
        old = sum(l.carrying for l in lots)
        if not lots:
            return 0.0
        total_q = sum(l.quantity for l in lots)
        for l in lots:
            share = (l.quantity / total_q) if total_q else 1.0 / len(lots)
            l.carrying = new_carrying * share
        return new_carrying - old

    def scale(self, instrument: str, fraction_remaining: float) -> tuple[float, float]:
        """Pro-rata principal reduction (factor paydown, partial call): returns (cost_released, carrying_released)."""
        lots = self.lots.get(instrument, [])
        cost_rel = carry_rel = 0.0
        for l in lots:
            cost_rel += l.cost * (1 - fraction_remaining)
            carry_rel += l.carrying * (1 - fraction_remaining)
            l.quantity *= fraction_remaining
            l.cost *= fraction_remaining
            l.carrying *= fraction_remaining
        self.lots[instrument] = [l for l in lots if abs(l.quantity) > 1e-9]
        return cost_rel, carry_rel


@dataclass
class AccountingSettings:
    lot_policy: LotPolicy = LotPolicy.FIFO
    write_off_accrued_on_default: bool = True
    reclassify_defaulted_to_claim: bool = True


class StandardAccountingPolicy:
    """Posts balanced entries for each economic event and maintains the lot book."""

    def __init__(self, ledger: Ledger, settings: AccountingSettings | None = None) -> None:
        self.ledger = ledger
        self.settings = settings or AccountingSettings()
        self.book = PositionBook(self.settings.lot_policy)
        self.accrued: dict[tuple[str, str], float] = {}     # (instrument, ccy) -> accrued balance
        self.coupon_receivable: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------ helpers
    def _post(self, event_id: str, ts: int, postings: list[Posting], memo: str) -> JournalEntry:
        return self.ledger.post(JournalEntry(event_id, ts, tuple(p for p in postings if p.debit or p.credit), memo))

    @staticmethod
    def _pnl_postings(amount: float, ccy: str, key: str, memo: str) -> list[Posting]:
        """Realized P&L posting: credit for gains, debit for losses."""
        if abs(amount) < 1e-12:
            return []
        return [Posting.cr(Account.REALIZED_PNL, Money(amount, ccy), key, memo)]

    # ------------------------------------------------------------------ capital
    def capital_flow(self, event_id: str, ts: int, amount: Money, memo: str = "capital") -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.CASH_SETTLED, amount, "", memo), Posting.cr(Account.CAPITAL, amount, "", memo)], memo)

    # ------------------------------------------------------------------ trades and settlement
    def trade(self, event_id: str, ts: int, instrument: str, ccy: str, quantity: float, clean_value: float, accrued: float = 0.0, fees: float = 0.0, memo: str = "") -> JournalEntry:
        """Book a trade on trade date. ``quantity`` > 0 buys, < 0 sells. ``clean_value`` and ``accrued``
        are positive currency amounts for the traded quantity; fees are paid by us."""
        key = instrument
        held = self.book.quantity(instrument)
        postings: list[Posting] = []
        if quantity > 0 and held >= 0:  # open / add long
            self.book.open(instrument, ccy, quantity, clean_value, ts)
            postings += [Posting.dr(Account.SECURITIES, Money(clean_value, ccy), key, "purchase"), Posting.dr(Account.ACCRUED_INTEREST, Money(accrued, ccy), key, "accrued bought")]
            if fees:
                postings.append(Posting.dr(Account.EXPENSE_FEES, Money(fees, ccy), key, "fees"))
            postings.append(Posting.cr(Account.PAYABLE_UNSETTLED, Money(clean_value + accrued + fees, ccy), key, "trade payable"))
            self.accrued[(instrument, ccy)] = self.accrued.get((instrument, ccy), 0.0) + accrued
        elif quantity < 0 and held > 0:  # sell long (partial or full)
            cost_rel, carry_rel = self.book.close(instrument, -quantity)  # positive: same sign as the long position
            proceeds = clean_value + accrued - fees
            postings += [Posting.dr(Account.RECEIVABLE_UNSETTLED, Money(proceeds, ccy), key, "sale receivable")]
            if fees:
                postings.append(Posting.dr(Account.EXPENSE_FEES, Money(fees, ccy), key, "fees"))
            postings += [Posting.cr(Account.SECURITIES, Money(carry_rel, ccy), key, "derecognise"), Posting.cr(Account.ACCRUED_INTEREST, Money(accrued, ccy), key, "accrued sold")]
            unreal = carry_rel - cost_rel
            if abs(unreal) > 1e-12:
                postings.append(Posting.dr(Account.UNREALIZED_PNL, Money(unreal, ccy), key, "reverse unrealized"))
            postings += self._pnl_postings(clean_value - cost_rel, ccy, key, "realized on sale")
            self.accrued[(instrument, ccy)] = self.accrued.get((instrument, ccy), 0.0) - accrued
        elif quantity < 0:  # open / add short
            self.book.open(instrument, ccy, quantity, -clean_value, ts)
            postings += [Posting.dr(Account.RECEIVABLE_UNSETTLED, Money(clean_value + accrued - fees, ccy), key, "short sale receivable")]
            if fees:
                postings.append(Posting.dr(Account.EXPENSE_FEES, Money(fees, ccy), key, "fees"))
            postings += [Posting.cr(Account.SECURITIES_SHORT, Money(clean_value, ccy), key, "short"), Posting.cr(Account.ACCRUED_INTEREST, Money(accrued, ccy), key, "accrued owed")]
            self.accrued[(instrument, ccy)] = self.accrued.get((instrument, ccy), 0.0) - accrued
        else:  # buy to cover
            cost_rel, carry_rel = self.book.close(instrument, -quantity)  # negative: same sign as the short position
            cost_rel, carry_rel = -cost_rel, -carry_rel  # positive magnitudes of the short's (negative) cost/carrying
            postings += [Posting.dr(Account.SECURITIES_SHORT, Money(carry_rel, ccy), key, "cover"), Posting.dr(Account.ACCRUED_INTEREST, Money(accrued, ccy), key, "accrued paid")]
            if fees:
                postings.append(Posting.dr(Account.EXPENSE_FEES, Money(fees, ccy), key, "fees"))
            postings.append(Posting.cr(Account.PAYABLE_UNSETTLED, Money(clean_value + accrued + fees, ccy), key, "cover payable"))
            unreal = carry_rel - cost_rel
            if abs(unreal) > 1e-12:
                postings.append(Posting.dr(Account.UNREALIZED_PNL, Money(-unreal, ccy), key, "reverse unrealized"))
            postings += self._pnl_postings(cost_rel - clean_value, ccy, key, "realized on cover")
            self.accrued[(instrument, ccy)] = self.accrued.get((instrument, ccy), 0.0) + accrued
        return self._post(event_id, ts, postings, memo or f"trade {instrument} {quantity:+,.0f}")

    def settle(self, event_id: str, ts: int, instrument: str, ccy: str, amount: float, direction: str, memo: str = "") -> JournalEntry:
        """Move cash on settlement: ``direction`` 'PAY' clears a payable, 'RECEIVE' clears a receivable."""
        if direction == "PAY":
            postings = [Posting.dr(Account.PAYABLE_UNSETTLED, Money(amount, ccy), instrument, "settle payable"), Posting.cr(Account.CASH_SETTLED, Money(amount, ccy), "", "settle payable")]
        else:
            postings = [Posting.dr(Account.CASH_SETTLED, Money(amount, ccy), "", "settle receivable"), Posting.cr(Account.RECEIVABLE_UNSETTLED, Money(amount, ccy), instrument, "settle receivable")]
        return self._post(event_id, ts, postings, memo or f"settlement {instrument}")

    # ------------------------------------------------------------------ valuation
    def mark(self, event_id: str, ts: int, instrument: str, ccy: str, new_clean_value: float, memo: str = "mark") -> JournalEntry | None:
        """Revalue the position's clean carrying value to ``new_clean_value`` (signed; shorts negative)."""
        held = self.book.quantity(instrument)
        if abs(held) < 1e-12:
            return None
        delta = self.book.revalue(instrument, new_clean_value)
        if abs(delta) < 1e-12:
            return None
        account = Account.SECURITIES if held > 0 else Account.SECURITIES_SHORT
        if held > 0:
            postings = [Posting.dr(account, Money(delta, ccy), instrument, memo), Posting.cr(Account.UNREALIZED_PNL, Money(delta, ccy), instrument, memo)]
        else:  # short: carrying is negative; a more negative carrying is a larger liability and a loss
            postings = [Posting.cr(account, Money(-delta, ccy), instrument, memo), Posting.dr(Account.UNREALIZED_PNL, Money(-delta, ccy), instrument, memo)]
        return self._post(event_id, ts, postings, memo)

    def accrue(self, event_id: str, ts: int, instrument: str, ccy: str, delta_accrued: float, memo: str = "accrual") -> JournalEntry | None:
        if abs(delta_accrued) < 1e-12:
            return None
        self.accrued[(instrument, ccy)] = self.accrued.get((instrument, ccy), 0.0) + delta_accrued
        return self._post(event_id, ts, [Posting.dr(Account.ACCRUED_INTEREST, Money(delta_accrued, ccy), instrument, memo), Posting.cr(Account.INCOME_COUPON, Money(delta_accrued, ccy), instrument, memo)], memo)

    def set_accrued(self, event_id: str, ts: int, instrument: str, ccy: str, target_accrued: float) -> JournalEntry | None:
        """Bring the accrued balance to the contractual accrued amount (daily accrual posting)."""
        current = self.accrued.get((instrument, ccy), 0.0)
        return self.accrue(event_id, ts, instrument, ccy, target_accrued - current)

    # ------------------------------------------------------------------ cashflows
    def coupon_entitlement(self, event_id: str, ts: int, instrument: str, ccy: str, amount: float) -> JournalEntry:
        """Record date: the coupon becomes a receivable; accrued is released (no P&L)."""
        accrued = self.accrued.get((instrument, ccy), 0.0)
        release = min(accrued, amount) if amount >= 0 else amount
        catch_up = amount - release
        postings = [Posting.dr(Account.RECEIVABLE_COUPON, Money(amount, ccy), instrument, "entitled coupon"), Posting.cr(Account.ACCRUED_INTEREST, Money(release, ccy), instrument, "release accrued")]
        if abs(catch_up) > 1e-12:
            postings.append(Posting.cr(Account.INCOME_COUPON, Money(catch_up, ccy), instrument, "income catch-up"))
        self.accrued[(instrument, ccy)] = accrued - release
        self.coupon_receivable[(instrument, ccy)] = self.coupon_receivable.get((instrument, ccy), 0.0) + amount
        return self._post(event_id, ts, postings, f"coupon entitlement {instrument}")

    def coupon_payment(self, event_id: str, ts: int, instrument: str, ccy: str, amount: float, withholding: float = 0.0) -> JournalEntry:
        """Payment date: cash in, receivable (or accrued) out. Creates no profit by itself."""
        receivable = self.coupon_receivable.get((instrument, ccy), 0.0)
        postings = [Posting.dr(Account.CASH_SETTLED, Money(amount - withholding, ccy), "", "coupon cash")]
        if withholding:
            postings.append(Posting.dr(Account.EXPENSE_TAX, Money(withholding, ccy), instrument, "withholding tax"))
        if receivable > 1e-12:
            use = min(receivable, amount)
            postings.append(Posting.cr(Account.RECEIVABLE_COUPON, Money(use, ccy), instrument, "clear coupon receivable"))
            self.coupon_receivable[(instrument, ccy)] = receivable - use
            rest = amount - use
        else:
            rest = amount
        if rest > 1e-12:
            accrued = self.accrued.get((instrument, ccy), 0.0)
            release = min(accrued, rest)
            if release > 1e-12:
                postings.append(Posting.cr(Account.ACCRUED_INTEREST, Money(release, ccy), instrument, "release accrued"))
                self.accrued[(instrument, ccy)] = accrued - release
            if rest - release > 1e-12:
                postings.append(Posting.cr(Account.INCOME_COUPON, Money(rest - release, ccy), instrument, "income catch-up"))
        return self._post(event_id, ts, postings, f"coupon payment {instrument}")

    def principal_reduction(self, event_id: str, ts: int, instrument: str, ccy: str, fraction_of_position: float, paydown_value: float, settled: bool = True, memo: str = "principal") -> JournalEntry:
        """Scheduled / prepaid / called principal: asset down, cash (or receivable) up, price gap to P&L."""
        cost_rel, carry_rel = self.book.scale(instrument, 1.0 - fraction_of_position)
        cash_account = Account.CASH_SETTLED if settled else Account.RECEIVABLE_UNSETTLED
        postings = [Posting.dr(cash_account, Money(paydown_value, ccy), "" if settled else instrument, memo), Posting.cr(Account.SECURITIES, Money(carry_rel, ccy), instrument, memo)]
        unreal = carry_rel - cost_rel
        if abs(unreal) > 1e-12:
            postings.append(Posting.dr(Account.UNREALIZED_PNL, Money(unreal, ccy), instrument, "reverse unrealized"))
        postings += self._pnl_postings(paydown_value - cost_rel, ccy, instrument, "realized on paydown")
        return self._post(event_id, ts, postings, memo)

    def pik_capitalisation(self, event_id: str, ts: int, instrument: str, ccy: str, capitalised_value: float, added_quantity: float) -> JournalEntry:
        """Accrued interest becomes principal (no cash)."""
        self.book.open(instrument, ccy, added_quantity, capitalised_value, ts)
        accrued = self.accrued.get((instrument, ccy), 0.0)
        release = min(accrued, capitalised_value)
        postings = [Posting.dr(Account.SECURITIES, Money(capitalised_value, ccy), instrument, "PIK"), Posting.cr(Account.ACCRUED_INTEREST, Money(release, ccy), instrument, "PIK")]
        if capitalised_value - release > 1e-12:
            postings.append(Posting.cr(Account.INCOME_COUPON, Money(capitalised_value - release, ccy), instrument, "PIK income"))
        self.accrued[(instrument, ccy)] = accrued - release
        return self._post(event_id, ts, postings, f"PIK {instrument}")

    def default_nonaccrual(self, event_id: str, ts: int, instrument: str, ccy: str) -> list[JournalEntry]:
        """On default: write off accrued (policy) and reclassify the position to a contractual claim."""
        entries = []
        accrued = self.accrued.get((instrument, ccy), 0.0)
        if self.settings.write_off_accrued_on_default and abs(accrued) > 1e-12:
            entries.append(self._post(event_id, ts, [Posting.dr(Account.EXPENSE_WRITEOFF, Money(accrued, ccy), instrument, "accrued written off"), Posting.cr(Account.ACCRUED_INTEREST, Money(accrued, ccy), instrument, "accrued written off")], "default: accrued write-off"))
            self.accrued[(instrument, ccy)] = 0.0
        if self.settings.reclassify_defaulted_to_claim:
            carrying = self.book.carrying(instrument)
            if abs(carrying) > 1e-12:
                entries.append(self._post(event_id, ts, [Posting.dr(Account.CONTRACTUAL_CLAIM, Money(carrying, ccy), instrument, "claim"), Posting.cr(Account.SECURITIES, Money(carrying, ccy), instrument, "reclassify")], "default: reclassify to claim"))
        return entries

    def reclassify_to_claim(self, event_id: str, ts: int, instrument: str, ccy: str, value: float) -> JournalEntry | None:
        """Move a security bought after its default from SECURITIES to CONTRACTUAL_CLAIM (no P&L)."""
        if abs(value) < 1e-12:
            return None
        return self._post(event_id, ts, [Posting.dr(Account.CONTRACTUAL_CLAIM, Money(value, ccy), instrument, "claim"), Posting.cr(Account.SECURITIES, Money(value, ccy), instrument, "reclassify")], "reclassify to claim")

    def write_off_coupon_receivable(self, event_id: str, ts: int, instrument: str, ccy: str) -> JournalEntry | None:
        """A missed payment: an entitled-but-unpaid coupon receivable is written off to expense."""
        amount = self.coupon_receivable.get((instrument, ccy), 0.0)
        if abs(amount) < 1e-12:
            return None
        self.coupon_receivable[(instrument, ccy)] = 0.0
        return self._post(event_id, ts, [Posting.dr(Account.EXPENSE_WRITEOFF, Money(amount, ccy), instrument, "missed coupon"), Posting.cr(Account.RECEIVABLE_COUPON, Money(amount, ccy), instrument, "missed coupon")], "default: coupon receivable written off")

    def claim_mark(self, event_id: str, ts: int, instrument: str, ccy: str, new_value: float) -> JournalEntry | None:
        delta = self.book.revalue(instrument, new_value)
        if abs(delta) < 1e-12:
            return None
        return self._post(event_id, ts, [Posting.dr(Account.CONTRACTUAL_CLAIM, Money(delta, ccy), instrument, "claim mark"), Posting.cr(Account.UNREALIZED_PNL, Money(delta, ccy), instrument, "claim mark")], "claim revaluation")

    def recovery(self, event_id: str, ts: int, instrument: str, ccy: str, cash: float, fraction_of_claim: float = 1.0, claim_account: Account = Account.CONTRACTUAL_CLAIM, settled: bool = True) -> JournalEntry:
        """Cash (or a settlement receivable) against part or all of a claim; the gap to cost is realized."""
        cost_rel, carry_rel = self.book.scale(instrument, 1.0 - fraction_of_claim)
        cash_account = Account.CASH_SETTLED if settled else Account.RECEIVABLE_UNSETTLED
        postings = [Posting.dr(cash_account, Money(cash, ccy), "" if settled else instrument, "recovery"), Posting.cr(claim_account, Money(carry_rel, ccy), instrument, "claim released")]
        unreal = carry_rel - cost_rel
        if abs(unreal) > 1e-12:
            postings.append(Posting.dr(Account.UNREALIZED_PNL, Money(unreal, ccy), instrument, "reverse unrealized"))
        postings += self._pnl_postings(cash - cost_rel, ccy, instrument, "realized on recovery")
        return self._post(event_id, ts, postings, f"recovery {instrument}")

    def exchange(self, event_id: str, ts: int, old_instrument: str, ccy: str, new_positions: list[tuple[str, float, float]], cash: float = 0.0) -> JournalEntry:
        """Restructuring / exchange: old claim becomes cash plus new securities ``(instrument, quantity, value)``.
        Cost basis is allocated to the new positions pro rata to value; the difference is realized."""
        cost_rel, carry_rel = self.book.scale(old_instrument, 0.0)
        account = Account.CONTRACTUAL_CLAIM if self.ledger.balance(Account.CONTRACTUAL_CLAIM, old_instrument, ccy) > 1e-9 else Account.SECURITIES
        total_new = sum(v for _, _, v in new_positions) + cash
        postings = [Posting.cr(account, Money(carry_rel, ccy), old_instrument, "exchange out")]
        if cash:
            postings.append(Posting.dr(Account.CASH_SETTLED, Money(cash, ccy), "", "exchange cash"))
        for inst, qty, value in new_positions:
            basis = cost_rel * (value / total_new) if total_new else 0.0
            lot = self.book.open(inst, ccy, qty, basis, ts)
            lot.carrying = value
            postings.append(Posting.dr(Account.SECURITIES, Money(value, ccy), inst, "exchange in"))
            if abs(value - basis) > 1e-12:
                postings.append(Posting.cr(Account.UNREALIZED_PNL, Money(value - basis, ccy), inst, "unrealized on new security"))
        unreal_old = carry_rel - cost_rel
        if abs(unreal_old) > 1e-12:
            postings.append(Posting.dr(Account.UNREALIZED_PNL, Money(unreal_old, ccy), old_instrument, "reverse unrealized"))
        realized = cash - (cost_rel - sum(cost_rel * (v / total_new) if total_new else 0.0 for _, _, v in new_positions))
        postings += self._pnl_postings(realized, ccy, old_instrument, "realized on exchange")
        return self._post(event_id, ts, postings, f"exchange {old_instrument}")

    # ------------------------------------------------------------------ derivatives / margin
    def futures_variation_margin(self, event_id: str, ts: int, instrument: str, ccy: str, vm: float) -> JournalEntry:
        """Settled-to-market: daily P&L moves to cash; no carrying value remains (no duplicate P&L)."""
        return self._post(event_id, ts, [Posting.dr(Account.CASH_SETTLED, Money(vm, ccy), "", "variation margin"), Posting.cr(Account.REALIZED_PNL, Money(vm, ccy), instrument, "futures daily settlement")], f"VM {instrument}")

    def derivative_mark(self, event_id: str, ts: int, instrument: str, ccy: str, new_value: float) -> JournalEntry | None:
        """Collateralised-to-market: exposure stays on the balance sheet (asset if positive, liability if negative)."""
        current = self.ledger.balance(Account.DERIVATIVE_ASSETS, instrument, ccy) - self.ledger.balance(Account.DERIVATIVE_LIABILITIES, instrument, ccy)
        delta = new_value - current
        if abs(delta) < 1e-12:
            return None
        postings = []
        if current >= 0 and new_value >= 0:
            postings = [Posting.dr(Account.DERIVATIVE_ASSETS, Money(delta, ccy), instrument, "MTM")]
        elif current <= 0 and new_value <= 0:
            postings = [Posting.cr(Account.DERIVATIVE_LIABILITIES, Money(-delta, ccy), instrument, "MTM")]
        else:  # crossing zero: clear one side, open the other
            if current > 0:
                postings = [Posting.cr(Account.DERIVATIVE_ASSETS, Money(current, ccy), instrument, "MTM"), Posting.cr(Account.DERIVATIVE_LIABILITIES, Money(-new_value, ccy), instrument, "MTM")]
            else:
                postings = [Posting.dr(Account.DERIVATIVE_LIABILITIES, Money(-current, ccy), instrument, "MTM"), Posting.dr(Account.DERIVATIVE_ASSETS, Money(new_value, ccy), instrument, "MTM")]
        postings.append(Posting.cr(Account.UNREALIZED_PNL, Money(delta, ccy), instrument, "MTM"))
        return self._post(event_id, ts, postings, f"derivative MTM {instrument}")

    def collateral_posted(self, event_id: str, ts: int, agreement: str, ccy: str, amount: float) -> JournalEntry:
        """Initial / independent margin restricts liquidity; it is not a loss."""
        if amount >= 0:
            postings = [Posting.dr(Account.COLLATERAL_POSTED, Money(amount, ccy), agreement, "post collateral"), Posting.cr(Account.CASH_SETTLED, Money(amount, ccy), "", "post collateral")]
        else:
            postings = [Posting.dr(Account.CASH_SETTLED, Money(-amount, ccy), "", "collateral returned"), Posting.cr(Account.COLLATERAL_POSTED, Money(-amount, ccy), agreement, "collateral returned")]
        return self._post(event_id, ts, postings, f"collateral {agreement}")

    def collateral_received(self, event_id: str, ts: int, agreement: str, ccy: str, amount: float) -> JournalEntry:
        if amount >= 0:
            postings = [Posting.dr(Account.CASH_SETTLED, Money(amount, ccy), "", "collateral received"), Posting.cr(Account.COLLATERAL_RECEIVED, Money(amount, ccy), agreement, "collateral received")]
        else:
            postings = [Posting.dr(Account.COLLATERAL_RECEIVED, Money(-amount, ccy), agreement, "collateral returned"), Posting.cr(Account.CASH_SETTLED, Money(-amount, ccy), "", "collateral returned")]
        return self._post(event_id, ts, postings, f"collateral received {agreement}")

    # ------------------------------------------------------------------ financing
    def repo_open(self, event_id: str, ts: int, agreement: str, ccy: str, cash_borrowed: float) -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.CASH_SETTLED, Money(cash_borrowed, ccy), "", "repo cash"), Posting.cr(Account.FINANCING_PAYABLE, Money(cash_borrowed, ccy), agreement, "repo")], f"repo open {agreement}")

    def repo_interest(self, event_id: str, ts: int, agreement: str, ccy: str, interest: float) -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.EXPENSE_FINANCING, Money(interest, ccy), agreement, "repo interest"), Posting.cr(Account.FINANCING_PAYABLE, Money(interest, ccy), agreement, "repo interest")], f"repo interest {agreement}")

    def repo_close(self, event_id: str, ts: int, agreement: str, ccy: str, amount: float) -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.FINANCING_PAYABLE, Money(amount, ccy), agreement, "repo repay"), Posting.cr(Account.CASH_SETTLED, Money(amount, ccy), "", "repo repay")], f"repo close {agreement}")

    def borrow_fee(self, event_id: str, ts: int, instrument: str, ccy: str, fee: float) -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.EXPENSE_BORROW, Money(fee, ccy), instrument, "borrow fee"), Posting.cr(Account.CASH_SETTLED, Money(fee, ccy), "", "borrow fee")], f"borrow fee {instrument}")

    def manufactured_payment(self, event_id: str, ts: int, instrument: str, ccy: str, amount: float) -> JournalEntry:
        """Short position pays the coupon to the lender: clears the accrued owed, cash out."""
        self.accrued[(instrument, ccy)] = self.accrued.get((instrument, ccy), 0.0) + amount
        return self._post(event_id, ts, [Posting.dr(Account.ACCRUED_INTEREST, Money(amount, ccy), instrument, "manufactured payment"), Posting.cr(Account.CASH_SETTLED, Money(amount, ccy), "", "manufactured payment")], f"manufactured payment {instrument}")

    def cash_interest(self, event_id: str, ts: int, ccy: str, amount: float, memo: str = "cash interest") -> JournalEntry | None:
        if abs(amount) < 1e-12:
            return None
        if amount >= 0:
            postings = [Posting.dr(Account.CASH_SETTLED, Money(amount, ccy), "", memo), Posting.cr(Account.INCOME_INTEREST, Money(amount, ccy), "", memo)]
        else:
            postings = [Posting.dr(Account.EXPENSE_FINANCING, Money(-amount, ccy), "", "overdraft financing"), Posting.cr(Account.CASH_SETTLED, Money(-amount, ccy), "", "overdraft financing")]
        return self._post(event_id, ts, postings, memo)

    def fee(self, event_id: str, ts: int, ccy: str, amount: float, key: str = "", memo: str = "fee") -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.EXPENSE_FEES, Money(amount, ccy), key, memo), Posting.cr(Account.CASH_SETTLED, Money(amount, ccy), "", memo)], memo)

    def fee_income(self, event_id: str, ts: int, ccy: str, amount: float, key: str = "", memo: str = "fee income") -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.CASH_SETTLED, Money(amount, ccy), "", memo), Posting.cr(Account.INCOME_FEES, Money(amount, ccy), key, memo)], memo)

    def tax(self, event_id: str, ts: int, ccy: str, amount: float, key: str = "", memo: str = "tax") -> JournalEntry:
        return self._post(event_id, ts, [Posting.dr(Account.EXPENSE_TAX, Money(amount, ccy), key, memo), Posting.cr(Account.CASH_SETTLED, Money(amount, ccy), "", memo)], memo)


__all__ = ["AccountingSettings", "Lot", "LotPolicy", "PositionBook", "StandardAccountingPolicy"]
_ = field
