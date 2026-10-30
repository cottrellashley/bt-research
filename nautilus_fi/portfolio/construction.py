"""Desired versus actual portfolios: signals become feasible target quantities under mandate and
funding constraints; live orders and reservations are netted so later decisions do not duplicate
pending orders or commit funding twice; only the unfilled remainder is cancelled or replaced."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.portfolio.constraints import ConstraintSet, Relaxation, Violation


@dataclass
class TargetPortfolio:
    weights: dict[str, float] = field(default_factory=dict)      # symbol -> target weight of NAV (signed)
    faces: dict[str, float] = field(default_factory=dict)        # explicit current-face targets override weights
    note: str = ""


@dataclass
class OrderIntent:
    symbol: str
    side: str
    face: float
    reason: str
    protocol: str = "RFQ"
    replaces: tuple[str, ...] = ()        # live order ids cancelled/replaced for the remainder
    priority: int = 0


@dataclass
class ConstructionResult:
    intents: list[OrderIntent]
    targets_face: dict[str, float]
    violations_pre: list[Violation]
    violations_post: list[Violation]
    relaxations: list[Relaxation]
    scaled_buys: float                   # 1.0 = no cash scaling applied
    diagnostics: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.__dict__ for i in self.intents])


class PortfolioConstructor:
    def __init__(self, constraints: ConstraintSet | None = None, cash_buffer: float = 0.01, settle_before_spend: bool = True) -> None:
        self.constraints = constraints or ConstraintSet()
        self.cash_buffer = cash_buffer
        self.settle_before_spend = settle_before_spend

    def build(
        self,
        target: TargetPortfolio,
        bonds: dict[str, BondSpec],
        nav: float,
        dirty_per_face: dict[str, float],
        holdings_face: dict[str, float],
        live_orders: list[dict],
        available_cash: float,
        position_rows: pd.DataFrame | None = None,
        protocol: str = "RFQ",
        forced: list[OrderIntent] | None = None,
    ) -> ConstructionResult:
        """``live_orders``: dicts with order_id, symbol, side, leaves_face. ``dirty_per_face``: value per 1 of face."""
        targets: dict[str, float] = {}
        for sym, w in target.weights.items():
            dv = dirty_per_face.get(sym)
            if sym in bonds and dv:
                targets[sym] = bonds[sym].round_face(w * nav / dv)
        for sym, face in target.faces.items():
            if sym in bonds:
                targets[sym] = bonds[sym].round_face(face)
        pending: dict[str, float] = {}
        live_ids: dict[str, list[str]] = {}
        for o in live_orders:
            sign = 1.0 if o["side"] == "BUY" else -1.0
            pending[o["symbol"]] = pending.get(o["symbol"], 0.0) + sign * o["leaves_face"]
            live_ids.setdefault(o["symbol"], []).append(o["order_id"])
        pre = self.constraints.evaluate(position_rows if position_rows is not None else pd.DataFrame(), nav)
        intents: list[OrderIntent] = list(forced or [])
        forced_syms = {i.symbol for i in intents}
        for sym in sorted(set(targets) | set(holdings_face)):
            if sym not in bonds or sym in forced_syms:
                continue
            delta = targets.get(sym, 0.0) - holdings_face.get(sym, 0.0)
            # live orders already cover part of the move: only trade the remainder, replacing stale ones
            delta_remaining = delta - pending.get(sym, 0.0)
            bond = bonds[sym]
            if abs(delta_remaining) < bond.min_piece:
                continue
            replaces = tuple(live_ids.get(sym, [])) if (pending.get(sym, 0.0) * delta < 0) else ()
            if replaces:
                delta_remaining = delta
            side = "BUY" if delta_remaining > 0 else "SELL"
            intents.append(OrderIntent(sym, side, abs(bond.round_face(delta_remaining)), "rebalance to target", protocol, replaces, priority=0 if side == "SELL" else 1))
        # funding: buys limited to available (settled) cash, sells first
        buys = [i for i in intents if i.side == "BUY"]
        cost = sum(i.face * dirty_per_face.get(i.symbol, 1.0) * 1.005 for i in buys)
        budget = max(available_cash * (1.0 - self.cash_buffer), 0.0)
        scale = min(1.0, budget / cost) if cost > 0 else 1.0
        if scale < 1.0:
            for i in buys:
                i.face = bonds[i.symbol].round_face(i.face * scale)
                i.reason += f"; scaled {scale:.0%} to settled cash"
            intents = [i for i in intents if i.face > 0]
        intents.sort(key=lambda i: (i.priority, i.symbol))
        post_rows = self._projected_rows(position_rows, intents, dirty_per_face, nav)
        post = self.constraints.evaluate(post_rows, nav)
        hard = self.constraints.hard_violations(post)
        diagnostics = {"n_targets": len(targets), "n_intents": len(intents), "hard_violations_post": len(hard)}
        if hard:
            # exclude the buys that create hard violations rather than silently relaxing
            bad_groups = {v.group for v in hard if v.group}
            kept = []
            for i in intents:
                if i.side == "BUY" and (i.symbol in bad_groups or bonds[i.symbol].issuer in bad_groups or bonds[i.symbol].sector in bad_groups):
                    i.reason += "; dropped: would breach hard constraint"
                    continue
                kept.append(i)
            intents = kept
            post = self.constraints.evaluate(self._projected_rows(position_rows, intents, dirty_per_face, nav), nav)
            diagnostics["dropped_for_constraints"] = True
        return ConstructionResult(intents, targets, pre, post, list(self.constraints.relaxations), scale, diagnostics)

    @staticmethod
    def _projected_rows(rows: pd.DataFrame | None, intents: list[OrderIntent], dirty_per_face: dict[str, float], nav: float) -> pd.DataFrame:
        if rows is None or rows.empty:
            return pd.DataFrame(rows) if rows is not None else pd.DataFrame()
        out = rows.copy()
        if "symbol" not in out:
            return out
        for i in intents:
            dv = dirty_per_face.get(i.symbol, 1.0)
            dw = (i.face * dv / nav) * (1.0 if i.side == "BUY" else -1.0) if nav else 0.0
            mask = out["symbol"] == i.symbol
            if mask.any():
                out.loc[mask, "weight"] = out.loc[mask, "weight"] + dw
        return out


__all__ = ["ConstructionResult", "OrderIntent", "PortfolioConstructor", "TargetPortfolio"]
