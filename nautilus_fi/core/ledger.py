"""Double-entry ledger (blueprint section 17).

Every economic event becomes a balanced journal entry. Balances by account, key (instrument,
agreement, counterparty) and currency give NAV as assets less liabilities; the equity side
(capital + income - expenses + realized + unrealized) must agree, which is the conservation test.
Corrections are reversals plus re-postings, never edits in place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from nautilus_fi.core.quantities import Money, currency_precision
from nautilus_fi.core.time import to_timestamp


class AccountClass(str, Enum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    INCOME = "INCOME"
    EXPENSE = "EXPENSE"


class Account(str, Enum):
    # assets
    CASH_SETTLED = "CASH_SETTLED"
    RECEIVABLE_UNSETTLED = "RECEIVABLE_UNSETTLED"      # sale proceeds / payments not yet received
    RECEIVABLE_COUPON = "RECEIVABLE_COUPON"            # entitled coupon not yet paid
    SECURITIES = "SECURITIES"                          # carrying value at market (clean)
    ACCRUED_INTEREST = "ACCRUED_INTEREST"
    DERIVATIVE_ASSETS = "DERIVATIVE_ASSETS"
    COLLATERAL_POSTED = "COLLATERAL_POSTED"            # initial margin / repo margin posted (restricted cash)
    CONTRACTUAL_CLAIM = "CONTRACTUAL_CLAIM"            # defaulted claims carried at recovery estimate
    # liabilities
    PAYABLE_UNSETTLED = "PAYABLE_UNSETTLED"
    SECURITIES_SHORT = "SECURITIES_SHORT"
    DERIVATIVE_LIABILITIES = "DERIVATIVE_LIABILITIES"
    COLLATERAL_RECEIVED = "COLLATERAL_RECEIVED"
    FINANCING_PAYABLE = "FINANCING_PAYABLE"            # repo borrowings
    BORROW_FEE_PAYABLE = "BORROW_FEE_PAYABLE"
    TAX_PAYABLE = "TAX_PAYABLE"
    # equity
    CAPITAL = "CAPITAL"                                # external subscriptions / withdrawals
    REALIZED_PNL = "REALIZED_PNL"
    UNREALIZED_PNL = "UNREALIZED_PNL"
    FX_TRANSLATION = "FX_TRANSLATION"
    # income / expense
    INCOME_COUPON = "INCOME_COUPON"
    INCOME_INTEREST = "INCOME_INTEREST"
    INCOME_FEES = "INCOME_FEES"                        # commitment / consent fees received
    EXPENSE_FINANCING = "EXPENSE_FINANCING"
    EXPENSE_BORROW = "EXPENSE_BORROW"
    EXPENSE_FEES = "EXPENSE_FEES"
    EXPENSE_TAX = "EXPENSE_TAX"
    EXPENSE_WRITEOFF = "EXPENSE_WRITEOFF"

    @property
    def klass(self) -> AccountClass:
        return ACCOUNT_CLASS[self]


ACCOUNT_CLASS = {
    Account.CASH_SETTLED: AccountClass.ASSET, Account.RECEIVABLE_UNSETTLED: AccountClass.ASSET, Account.RECEIVABLE_COUPON: AccountClass.ASSET,
    Account.SECURITIES: AccountClass.ASSET, Account.ACCRUED_INTEREST: AccountClass.ASSET, Account.DERIVATIVE_ASSETS: AccountClass.ASSET,
    Account.COLLATERAL_POSTED: AccountClass.ASSET, Account.CONTRACTUAL_CLAIM: AccountClass.ASSET,
    Account.PAYABLE_UNSETTLED: AccountClass.LIABILITY, Account.SECURITIES_SHORT: AccountClass.LIABILITY, Account.DERIVATIVE_LIABILITIES: AccountClass.LIABILITY,
    Account.COLLATERAL_RECEIVED: AccountClass.LIABILITY, Account.FINANCING_PAYABLE: AccountClass.LIABILITY, Account.BORROW_FEE_PAYABLE: AccountClass.LIABILITY,
    Account.TAX_PAYABLE: AccountClass.LIABILITY,
    Account.CAPITAL: AccountClass.EQUITY, Account.REALIZED_PNL: AccountClass.EQUITY, Account.UNREALIZED_PNL: AccountClass.EQUITY, Account.FX_TRANSLATION: AccountClass.EQUITY,
    Account.INCOME_COUPON: AccountClass.INCOME, Account.INCOME_INTEREST: AccountClass.INCOME, Account.INCOME_FEES: AccountClass.INCOME,
    Account.EXPENSE_FINANCING: AccountClass.EXPENSE, Account.EXPENSE_BORROW: AccountClass.EXPENSE, Account.EXPENSE_FEES: AccountClass.EXPENSE,
    Account.EXPENSE_TAX: AccountClass.EXPENSE, Account.EXPENSE_WRITEOFF: AccountClass.EXPENSE,
}


class UnbalancedEntry(ValueError):
    pass


@dataclass(frozen=True)
class Posting:
    account: Account
    currency: str
    debit: float = 0.0
    credit: float = 0.0
    key: str = ""              # instrument / agreement / counterparty / lot
    memo: str = ""

    def __post_init__(self) -> None:
        if self.debit < 0 or self.credit < 0:
            raise ValueError("Postings carry non-negative debit and credit amounts")
        if self.debit and self.credit:
            raise ValueError("A posting is either a debit or a credit")

    @staticmethod
    def dr(account: Account, amount: Money, key: str = "", memo: str = "") -> Posting:
        return Posting(account, amount.currency, debit=amount.amount, key=key, memo=memo) if amount.amount >= 0 else Posting(account, amount.currency, credit=-amount.amount, key=key, memo=memo)

    @staticmethod
    def cr(account: Account, amount: Money, key: str = "", memo: str = "") -> Posting:
        return Posting(account, amount.currency, credit=amount.amount, key=key, memo=memo) if amount.amount >= 0 else Posting(account, amount.currency, debit=-amount.amount, key=key, memo=memo)

    @property
    def signed(self) -> float:
        """Signed effect on the account's natural balance (+ increases assets/expenses, + increases liabilities/equity/income)."""
        if self.account.klass in (AccountClass.ASSET, AccountClass.EXPENSE):
            return self.debit - self.credit
        return self.credit - self.debit


@dataclass(frozen=True)
class JournalEntry:
    event_id: str
    ts_ns: int
    postings: tuple[Posting, ...]
    memo: str = ""
    entry_id: str = ""
    reverses: str | None = None

    def validate(self, tolerance: float = 1e-6) -> None:
        by_ccy: dict[str, float] = {}
        for p in self.postings:
            by_ccy[p.currency] = by_ccy.get(p.currency, 0.0) + p.debit - p.credit
        bad = {c: v for c, v in by_ccy.items() if abs(v) > max(tolerance, 10 ** (-currency_precision(c)) * 0.51)}
        if bad:
            raise UnbalancedEntry(f"Entry {self.entry_id or self.event_id} unbalanced by currency: {bad}")

    def reversed_at(self, ts_ns: int, event_id: str, memo: str = "reversal") -> JournalEntry:
        swapped = tuple(Posting(p.account, p.currency, debit=p.credit, credit=p.debit, key=p.key, memo=f"reverse: {p.memo}") for p in self.postings)
        return JournalEntry(event_id, ts_ns, swapped, memo, reverses=self.entry_id)


class Ledger:
    def __init__(self, base_currency: str = "USD") -> None:
        self.base_currency = base_currency
        self.entries: list[JournalEntry] = []
        self._balances: dict[tuple[Account, str, str], float] = {}
        self._seq = 0

    # ------------------------------------------------------------------ posting
    def post(self, entry: JournalEntry) -> JournalEntry:
        entry.validate()
        self._seq += 1
        entry = JournalEntry(entry.event_id, entry.ts_ns, entry.postings, entry.memo, entry_id=entry.entry_id or f"JE{self._seq:08d}", reverses=entry.reverses)
        for p in entry.postings:
            k = (p.account, p.key, p.currency)
            self._balances[k] = self._balances.get(k, 0.0) + p.signed
        self.entries.append(entry)
        return entry

    def reverse(self, entry_id: str, ts_ns: int, event_id: str, memo: str = "correction") -> JournalEntry:
        original = next((e for e in self.entries if e.entry_id == entry_id), None)
        if original is None:
            raise KeyError(entry_id)
        return self.post(original.reversed_at(ts_ns, event_id, memo))

    # ------------------------------------------------------------------ balances
    def balance(self, account: Account, key: str | None = None, currency: str | None = None) -> float:
        total = 0.0
        for (acc, k, c), v in self._balances.items():
            if acc == account and (key is None or k == key) and (currency is None or c == currency):
                total += v
        return total

    def balances_frame(self) -> pd.DataFrame:
        rows = [{"account": a.value, "class": a.klass.value, "key": k, "currency": c, "balance": v} for (a, k, c), v in self._balances.items() if abs(v) > 1e-9]
        return pd.DataFrame(rows, columns=["account", "class", "key", "currency", "balance"])

    def class_total(self, klass: AccountClass, currency: str, fx: Callable[[str], float] | None = None) -> float:
        total = 0.0
        for (acc, _k, c), v in self._balances.items():
            if acc.klass == klass:
                rate = 1.0 if c == currency else (fx(c) if fx else None)
                if rate is None:
                    raise ValueError(f"No FX rate to translate {c} into {currency}")
                total += v * rate
        return total

    def nav(self, fx: Callable[[str], float] | None = None) -> Money:
        """Assets less liabilities in the base currency (``fx(ccy)`` gives units of base per 1 ccy)."""
        a = self.class_total(AccountClass.ASSET, self.base_currency, fx)
        liab = self.class_total(AccountClass.LIABILITY, self.base_currency, fx)
        return Money(a - liab, self.base_currency)

    def equity_total(self, fx: Callable[[str], float] | None = None) -> Money:
        eq = self.class_total(AccountClass.EQUITY, self.base_currency, fx)
        inc = self.class_total(AccountClass.INCOME, self.base_currency, fx)
        exp = self.class_total(AccountClass.EXPENSE, self.base_currency, fx)
        return Money(eq + inc - exp, self.base_currency)

    def conservation_check(self, fx: Callable[[str], float] | None = None, tolerance: float = 1e-4) -> None:
        """Assets - liabilities must equal capital + retained P&L (balanced books)."""
        nav = self.nav(fx).amount
        eq = self.equity_total(fx).amount
        if abs(nav - eq) > tolerance:
            raise UnbalancedEntry(f"Ledger does not balance: NAV {nav:.6f} vs equity {eq:.6f}")

    def trial_balance(self) -> pd.DataFrame:
        df = self.balances_frame()
        if df.empty:
            return df
        return df.groupby(["class", "account", "currency"], as_index=False)["balance"].sum()

    def entries_frame(self) -> pd.DataFrame:
        rows = []
        for e in self.entries:
            for p in e.postings:
                rows.append({"entry_id": e.entry_id, "event_id": e.event_id, "ts": to_timestamp(e.ts_ns), "account": p.account.value, "key": p.key, "currency": p.currency, "debit": p.debit, "credit": p.credit, "memo": p.memo or e.memo, "reverses": e.reverses})
        return pd.DataFrame(rows)

    def cash(self, currency: str | None = None) -> float:
        return self.balance(Account.CASH_SETTLED, currency=currency or self.base_currency)

    def position_carrying(self, instrument: str, currency: str) -> float:
        return self.balance(Account.SECURITIES, key=instrument, currency=currency) - self.balance(Account.SECURITIES_SHORT, key=instrument, currency=currency)


__all__ = ["ACCOUNT_CLASS", "Account", "AccountClass", "JournalEntry", "Ledger", "Posting", "UnbalancedEntry"]
_ = field
