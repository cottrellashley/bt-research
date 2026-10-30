"""Policies for changes outside the manager's control: downgrades, index removals, paydowns.
Forced sales can be immediate, delayed, grandfathered or discretionary; the policy can dominate
performance and must be tested."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.portfolio.construction import OrderIntent


@dataclass
class ForcedSale:
    symbol: str
    face: float
    trigger: str                       # DOWNGRADE | INDEX_REMOVAL | MATURITY_BAND | PAYDOWN_REINVEST
    triggered_on: date
    due_by: date | None
    action: str                        # IMMEDIATE | DELAYED | GRANDFATHER | DISCRETIONARY


@dataclass
class ForcedActionPolicy:
    max_rating_score: int | None = 10                     # worst allowed (10 = BBB-): downgrade below forces action
    downgrade_action: str = "DELAYED"                     # IMMEDIATE | DELAYED | GRANDFATHER | DISCRETIONARY
    downgrade_grace_days: int = 30
    index_removal_action: str = "DELAYED"
    index_removal_grace_days: int = 5
    min_years_to_maturity: float | None = None
    reinvest_paydowns: bool = True
    pending: dict[str, ForcedSale] = field(default_factory=dict)

    def evaluate(self, d: date, holdings_face: dict[str, float], rating_scores: dict[str, int | None], removed_from_index: set[str] | None = None, years_to_maturity: dict[str, float] | None = None) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        from datetime import timedelta

        for sym, face in holdings_face.items():
            if face <= 0:
                continue
            score = rating_scores.get(sym)
            if self.max_rating_score is not None and score is not None and score > self.max_rating_score and sym not in self.pending:
                due = None if self.downgrade_action in ("GRANDFATHER", "DISCRETIONARY") else d + timedelta(days=0 if self.downgrade_action == "IMMEDIATE" else self.downgrade_grace_days)
                self.pending[sym] = ForcedSale(sym, face, "DOWNGRADE", d, due, self.downgrade_action)
            if removed_from_index and sym in removed_from_index and sym not in self.pending:
                due = None if self.index_removal_action in ("GRANDFATHER", "DISCRETIONARY") else d + timedelta(days=0 if self.index_removal_action == "IMMEDIATE" else self.index_removal_grace_days)
                self.pending[sym] = ForcedSale(sym, face, "INDEX_REMOVAL", d, due, self.index_removal_action)
            if self.min_years_to_maturity is not None and years_to_maturity and years_to_maturity.get(sym, 99) < self.min_years_to_maturity and sym not in self.pending:
                self.pending[sym] = ForcedSale(sym, face, "MATURITY_BAND", d, d, "IMMEDIATE")
        for sym, fs in list(self.pending.items()):
            if sym not in holdings_face or holdings_face[sym] <= 0:
                self.pending.pop(sym)
                continue
            if fs.action in ("GRANDFATHER", "DISCRETIONARY"):
                continue
            if fs.due_by is not None and d >= fs.due_by:
                intents.append(OrderIntent(sym, "SELL", holdings_face[sym], f"forced sale: {fs.trigger} ({fs.action}, triggered {fs.triggered_on})", priority=-1))
        return intents

    def clear(self, symbol: str) -> None:
        self.pending.pop(symbol, None)


__all__ = ["ForcedActionPolicy", "ForcedSale"]
