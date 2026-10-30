"""Non-negotiable invariants (blueprint section 20), runnable against any engine via callbacks:

* future information changes must not alter earlier signals, orders or as-known states;
* duplicate event delivery must not duplicate cash;
* payments and transformations reconcile balances;
* lot regrouping preserves economic NAV;
* a pinned replay reproduces results within declared tolerance;
* a rejected or unfilled order does not change ownership;
* a failed settlement does not create spendable cash;
* a single cashflow is realised by exactly one of the forecast engine and the environment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nautilus_fi.core.events import EventLog, EventType
from nautilus_fi.core.ledger import Ledger


@dataclass
class InvariantResult:
    name: str
    passed: bool
    detail: str = ""
    data: dict = field(default_factory=dict)


class InvariantSuite:
    def __init__(self) -> None:
        self.results: list[InvariantResult] = []

    def _add(self, r: InvariantResult) -> InvariantResult:
        self.results.append(r)
        return r

    # ------------------------------------------------------------------ temporal
    def temporal_isolation(self, run: Callable[[], tuple[pd.DataFrame, str]], perturb_future: Callable[[], None], cutoff: pd.Timestamp, key: str = "order_id") -> InvariantResult:
        """Run, perturb data after ``cutoff``, rerun: decisions/orders before the cutoff must be identical."""
        base, _ = run()
        perturb_future()
        again, _ = run()
        b = base[pd.to_datetime(base["ts"]) <= cutoff].sort_values(key).reset_index(drop=True) if len(base) else base
        a = again[pd.to_datetime(again["ts"]) <= cutoff].sort_values(key).reset_index(drop=True) if len(again) else again
        same = b.equals(a)
        return self._add(InvariantResult("temporal_isolation", same, "decisions before the cutoff unchanged" if same else "decisions before the cutoff changed after future data was perturbed", {"n_before": len(b)}))

    # ------------------------------------------------------------------ accounting
    def duplicate_delivery(self, log: EventLog, event) -> InvariantResult:
        try:
            log.append(event)
            return self._add(InvariantResult("duplicate_delivery", False, "duplicate event was appended"))
        except ValueError:
            return self._add(InvariantResult("duplicate_delivery", True, "duplicate event rejected"))

    def conservation(self, ledger: Ledger, fx=None) -> InvariantResult:
        try:
            ledger.conservation_check(fx)
            return self._add(InvariantResult("conservation", True, f"NAV {ledger.nav(fx).amount:,.2f} equals equity"))
        except Exception as e:  # noqa: BLE001
            return self._add(InvariantResult("conservation", False, str(e)))

    def lot_regrouping(self, nav_fifo: float, nav_average: float, tolerance: float = 1e-6) -> InvariantResult:
        ok = abs(nav_fifo - nav_average) <= tolerance
        return self._add(InvariantResult("lot_regrouping", ok, f"FIFO {nav_fifo:,.2f} vs AVERAGE {nav_average:,.2f}"))

    def pinned_replay(self, run: Callable[[], tuple[str, dict[str, str]]]) -> InvariantResult:
        chain1, out1 = run()
        chain2, out2 = run()
        ok = chain1 == chain2 and out1 == out2
        return self._add(InvariantResult("pinned_replay", ok, "event chain and output hashes reproduce" if ok else "replay differs", {"chain": chain1}))

    def unfilled_orders_do_not_change_ownership(self, orders: pd.DataFrame, positions_before: dict[str, float], positions_after: dict[str, float]) -> InvariantResult:
        unfilled = orders[orders["status"].isin(["REJECTED", "EXPIRED", "CANCELED"]) & (pd.to_numeric(orders.get("filled_qty", 0), errors="coerce").fillna(0) == 0)] if len(orders) else orders
        changed = [s for s in set(unfilled.get("instrument_id", [])) if positions_before.get(str(s), 0.0) != positions_after.get(str(s), 0.0)]
        return self._add(InvariantResult("unfilled_order_ownership", not changed, "no ownership change from unfilled orders" if not changed else f"ownership changed for {changed}"))

    def failed_settlement_not_spendable(self, settled_cash_before: float, settled_cash_after_fail: float) -> InvariantResult:
        ok = settled_cash_after_fail <= settled_cash_before + 1e-9
        return self._add(InvariantResult("failed_settlement_cash", ok, "failed settlement created no spendable cash" if ok else "cash appeared without settlement"))

    def single_realization(self, realized: pd.DataFrame, projected_applied: pd.DataFrame, keys: tuple[str, ...] = ("instrument", "date", "type")) -> InvariantResult:
        if realized.empty or projected_applied.empty:
            return self._add(InvariantResult("single_realization", True, "no overlap possible"))
        merged = realized.merge(projected_applied, on=list(keys), how="inner")
        return self._add(InvariantResult("single_realization", merged.empty, "no cashflow realised twice" if merged.empty else f"{len(merged)} cashflows realised by both engine and environment"))

    def payments_reconcile(self, log: EventLog, ledger: Ledger, currency: str = "USD", fx=None) -> InvariantResult:
        """Every coupon payment event has a ledger cash effect of the same amount (all currencies, translated
        to ``currency`` with ``fx`` when given)."""
        def rate(c: str) -> float:
            return 1.0 if c == currency else (float(fx(c)) if fx else 1.0)

        cash_events = sum(float(e.payload.get("amount", 0.0)) * rate(str(e.payload.get("currency", currency))) for e in log.events_for(types=[EventType.COUPON_PAYMENT]))
        booked = sum(p.debit * rate(p.currency) for e in ledger.entries for p in e.postings if p.account.value == "CASH_SETTLED" and "coupon" in (p.memo or e.memo).lower())
        ok = abs(cash_events - booked) < 0.01
        return self._add(InvariantResult("payments_reconcile", ok, f"events {cash_events:,.2f} vs ledger {booked:,.2f}"))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"invariant": r.name, "passed": r.passed, "detail": r.detail} for r in self.results])

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)


__all__ = ["InvariantResult", "InvariantSuite"]
