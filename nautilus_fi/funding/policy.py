"""Funding policy: liquidity is a portfolio constraint. Aggregate obligations by currency and
settlement window, check settled cash and collateral, and when a breach occurs apply explicit
actions in order (cancel buys, source financing, unwind hedges, liquidate by rule, declare failure).
Liquidation goes through the same execution model; cash is never injected unexplained."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd


class BreachAction(str, Enum):
    CANCEL_BUYS = "CANCEL_BUYS"
    SOURCE_REPO = "SOURCE_REPO"
    UNWIND_HEDGES = "UNWIND_HEDGES"
    LIQUIDATE_BY_RULE = "LIQUIDATE_BY_RULE"
    DECLARE_FAILURE = "DECLARE_FAILURE"


@dataclass
class FundingPolicy:
    allow_bridge: bool = False
    max_bridge: float = 0.0                        # bridge financing cap (base currency)
    bridge_rate_bp: float = 75.0                   # over cash rate
    settle_before_spend: bool = True               # unsettled sale proceeds cannot fund purchases
    breach_actions: tuple[BreachAction, ...] = (BreachAction.CANCEL_BUYS, BreachAction.SOURCE_REPO, BreachAction.LIQUIDATE_BY_RULE, BreachAction.DECLARE_FAILURE)
    liquidation_rule: str = "MOST_LIQUID_FIRST"    # MOST_LIQUID_FIRST | PRO_RATA | LARGEST_FIRST
    repo_eligible_share: float = 0.9               # share of eligible holdings that can be repoed
    horizon_days: int = 2                          # settlement window for obligations
    min_cash_buffer: float = 0.0
    allow_fx_conversion: bool = True               # foreign purchases may be funded from base cash at spot
    repo_haircut: float = 0.02
    failure_after_business_days: int = 5           # unresolved shortfall persisting this long is a funding default

    def bridge_allowance(self) -> float:
        return self.max_bridge if self.allow_bridge else 0.0


@dataclass
class Obligation:
    date: date
    currency: str
    amount: float                  # positive = we must pay
    kind: str                      # SETTLEMENT | MARGIN | COUPON_SHORT | REPO_REPAY | FEE
    ref: str = ""


@dataclass
class FundingAssessment:
    date: date
    currency: str
    settled_cash: float
    reserved: float
    receipts_in_window: float
    obligations_in_window: float
    bridge_allowance: float
    shortfall: float
    breached: bool
    detail: list[Obligation] = field(default_factory=list)


@dataclass
class FundingAction:
    kind: BreachAction
    currency: str
    amount: float = 0.0
    instrument: str | None = None
    quantity: float = 0.0
    reason: str = ""


class FundingAssessor:
    def __init__(self, policy: FundingPolicy) -> None:
        self.policy = policy

    def assess(self, d: date, currency: str, settled_cash: float, reserved: float, obligations: list[Obligation], receipts: list[Obligation]) -> FundingAssessment:
        from datetime import timedelta

        horizon = d + timedelta(days=self.policy.horizon_days)
        due = [o for o in obligations if d <= o.date <= horizon and o.currency == currency]
        inc = [r for r in receipts if d <= r.date <= horizon and r.currency == currency]
        pay = sum(o.amount for o in due)
        rec = sum(r.amount for r in inc) if not self.policy.settle_before_spend else sum(r.amount for r in inc if r.date <= d)
        bridge = self.policy.bridge_allowance()
        available = settled_cash - reserved + rec + bridge - self.policy.min_cash_buffer
        shortfall = max(0.0, pay - available)
        return FundingAssessment(d, currency, settled_cash, reserved, rec, pay, bridge, shortfall, shortfall > 1e-9, due)

    def plan(self, a: FundingAssessment, open_buy_orders: list[dict], holdings: list[dict], repo_capacity: float) -> list[FundingAction]:
        """Translate a breach into ordered actions. ``holdings`` rows: instrument, market_value, liquidity_rank, is_hedge, repo_eligible."""
        actions: list[FundingAction] = []
        remaining = a.shortfall
        if remaining <= 0:
            return actions
        for step in self.policy.breach_actions:
            if remaining <= 1e-9:
                break
            if step == BreachAction.CANCEL_BUYS:
                for o in sorted(open_buy_orders, key=lambda x: -x.get("cash", 0.0)):
                    if remaining <= 1e-9:
                        break
                    freed = min(o.get("cash", 0.0), remaining)
                    actions.append(FundingAction(step, a.currency, freed, o.get("instrument"), o.get("quantity", 0.0), "cancel resting buy to free reserved cash"))
                    remaining -= freed
            elif step == BreachAction.SOURCE_REPO:
                amt = min(repo_capacity * self.policy.repo_eligible_share, remaining)
                if amt > 0:
                    actions.append(FundingAction(step, a.currency, amt, reason="repo eligible holdings"))
                    remaining -= amt
            elif step == BreachAction.UNWIND_HEDGES:
                for h in [x for x in holdings if x.get("is_hedge")]:
                    if remaining <= 1e-9:
                        break
                    amt = min(abs(h.get("market_value", 0.0)), remaining)
                    actions.append(FundingAction(step, a.currency, amt, h["instrument"], h.get("quantity", 0.0), "unwind hedge"))
                    remaining -= amt
            elif step == BreachAction.LIQUIDATE_BY_RULE:
                rows = [x for x in holdings if not x.get("is_hedge") and x.get("market_value", 0.0) > 0]
                if self.policy.liquidation_rule == "MOST_LIQUID_FIRST":
                    rows = sorted(rows, key=lambda x: x.get("liquidity_rank", 99))
                elif self.policy.liquidation_rule == "LARGEST_FIRST":
                    rows = sorted(rows, key=lambda x: -x.get("market_value", 0.0))
                total = sum(x["market_value"] for x in rows) or 1.0
                for h in rows:
                    if remaining <= 1e-9:
                        break
                    amt = min(h["market_value"], remaining) if self.policy.liquidation_rule != "PRO_RATA" else min(h["market_value"], a.shortfall * h["market_value"] / total)
                    actions.append(FundingAction(step, a.currency, amt, h["instrument"], h.get("quantity", 0.0) * (amt / h["market_value"]), f"liquidate by {self.policy.liquidation_rule}"))
                    remaining -= amt
            elif step == BreachAction.DECLARE_FAILURE:
                actions.append(FundingAction(step, a.currency, remaining, reason="unresolved shortfall: funding default"))
                remaining = 0.0
        return actions

    @staticmethod
    def to_frame(actions: list[FundingAction]) -> pd.DataFrame:
        return pd.DataFrame([{**a.__dict__, "kind": a.kind.value} for a in actions])


__all__ = ["BreachAction", "FundingAction", "FundingAssessment", "FundingAssessor", "FundingPolicy", "Obligation"]
