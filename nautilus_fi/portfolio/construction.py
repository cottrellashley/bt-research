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
    turnover: dict = field(default_factory=dict)         # this rebalance under every definition (fractions of NAV)
    dropped: list[OrderIntent] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.__dict__ for i in self.intents])

    def records(self, date=None) -> list[dict]:
        """Rows for the run's construction log: every pre / post violation plus a summary row."""
        out = []
        for stage, vs in (("pre", self.violations_pre), ("post", self.violations_post)):
            for v in vs:
                out.append({"date": date, "stage": stage, "constraint": v.constraint.name, "kind": v.constraint.kind.value, "group": v.group, "value": v.value, "bound": v.bound, "severity": v.severity})
        for i in self.dropped:
            out.append({"date": date, "stage": "dropped", "constraint": "", "kind": "", "group": i.symbol, "value": i.face, "bound": None, "severity": "", "reason": i.reason})
        for r in self.relaxations:
            out.append({"date": date, "stage": "relaxation", "constraint": r.constraint, "kind": "", "group": r.group, "value": r.to_bound, "bound": r.from_bound, "severity": "", "reason": r.reason})
        out.append({"date": date, "stage": "summary", "constraint": "", "kind": "", "group": None, "value": None, "bound": None, "severity": "", **{f"turnover_{k}": v for k, v in self.turnover.items()}, "n_intents": len(self.intents), "n_dropped": len(self.dropped), "scaled_buys": self.scaled_buys, "hard_post": sum(1 for v in self.violations_post if v.severity == "HARD")})
        return out


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
        candidate_rows: pd.DataFrame | None = None,
        benchmark_rows: pd.DataFrame | None = None,
        turnover_definition: str = "HALF_SUM",
    ) -> ConstructionResult:
        """``live_orders``: dicts with order_id, symbol, side, leaves_face. ``dirty_per_face``: value per 1 of face.
        ``candidate_rows``: constraint rows (attributes, exposures) for names that are not yet held, so new
        buys are checked like holdings; ``benchmark_rows``: benchmark weights with the same columns for
        benchmark-relative constraints; ``turnover_definition``: how TURNOVER constraints measure this rebalance."""
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
        pre = self.constraints.evaluate(position_rows if position_rows is not None else pd.DataFrame(), nav, cash=available_cash, benchmark_rows=benchmark_rows)
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
        # turnover budget: scale discretionary trades down to a hard TURNOVER maximum
        tov = self._turnover(intents, dirty_per_face, nav)
        for c in self.constraints.constraints:
            if c.kind.value == "TURNOVER" and c.hard and c.maximum is not None and tov.get(turnover_definition, 0.0) > c.maximum + 1e-12 and float(sum(holdings_face.values())) > 0:
                f = c.maximum / tov[turnover_definition]
                for i in intents:
                    if i.priority >= 0:                                     # forced sales keep their size
                        i.face = bonds[i.symbol].round_face(i.face * f)
                        i.reason += f"; scaled {f:.0%} to the turnover budget"
                intents = [i for i in intents if i.face > 0]
                self.constraints.relaxations.append(Relaxation(c.name, None, tov[turnover_definition], c.maximum, "trades scaled to the turnover budget"))
                tov = self._turnover(intents, dirty_per_face, nav)
        intents.sort(key=lambda i: (i.priority, i.symbol))
        cash_after = available_cash - sum(i.face * dirty_per_face.get(i.symbol, 1.0) * (1 if i.side == "BUY" else -1) for i in intents)
        post_rows = self._projected_rows(position_rows, intents, dirty_per_face, nav, candidate_rows, bonds)
        post = self.constraints.evaluate(post_rows, nav, tov.get(turnover_definition, 0.0), cash_after, benchmark_rows=benchmark_rows)
        dropped: list[OrderIntent] = []
        diagnostics = {"n_targets": len(targets), "n_intents": len(intents)}
        for _ in range(25):
            hard = self.constraints.hard_violations(post)
            culprits = self._culprits(hard, intents, post_rows, bonds)
            if not hard or not culprits:
                break
            for i in culprits:
                i.reason += "; dropped: would breach hard constraint"
            dropped += culprits
            intents = [i for i in intents if i not in culprits]
            tov = self._turnover(intents, dirty_per_face, nav)
            cash_after = available_cash - sum(i.face * dirty_per_face.get(i.symbol, 1.0) * (1 if i.side == "BUY" else -1) for i in intents)
            post_rows = self._projected_rows(position_rows, intents, dirty_per_face, nav, candidate_rows, bonds)
            post = self.constraints.evaluate(post_rows, nav, tov.get(turnover_definition, 0.0), cash_after, benchmark_rows=benchmark_rows)
            diagnostics["dropped_for_constraints"] = True
        diagnostics.update({"n_intents": len(intents), "hard_violations_post": len(self.constraints.hard_violations(post)), "n_dropped": len(dropped)})
        return ConstructionResult(intents, targets, pre, post, list(self.constraints.relaxations), scale, diagnostics, tov, dropped)

    @staticmethod
    def _turnover(intents: list[OrderIntent], dirty_per_face: dict[str, float], nav: float) -> dict[str, float]:
        if not nav:
            return {"ONE_WAY_BUYS": 0.0, "HALF_SUM": 0.0, "TWO_WAY": 0.0}
        b = sum(i.face * dirty_per_face.get(i.symbol, 1.0) for i in intents if i.side == "BUY") / nav
        s = sum(i.face * dirty_per_face.get(i.symbol, 1.0) for i in intents if i.side == "SELL") / nav
        return {"ONE_WAY_BUYS": b, "HALF_SUM": 0.5 * (b + s), "TWO_WAY": b + s}

    @staticmethod
    def _culprits(hard: list[Violation], intents: list[OrderIntent], rows: pd.DataFrame, bonds: dict[str, BondSpec]) -> list[OrderIntent]:
        """Buys to drop for hard violations: buys inside the breached group (issuer, sector, any group
        column, the name itself), or - for portfolio-level exposure breaches - the buy contributing most
        in the breaching direction."""
        out: list[OrderIntent] = []
        buys = [i for i in intents if i.side == "BUY"]
        if not buys or rows is None or rows.empty:
            return out
        attr = rows.set_index("symbol") if "symbol" in rows else rows
        for v in hard:
            c = v.constraint
            k = c.kind.value
            if k in ("ISSUER_WEIGHT", "GROUP_WEIGHT", "BENCHMARK_BAND", "CONCENTRATION", "RATING_FLOOR", "SENIORITY", "RESTRICTED") and v.group is not None:
                col = {"ISSUER_WEIGHT": "issuer", "SENIORITY": "seniority"}.get(k, c.group_col)
                for i in buys:
                    if i in out:
                        continue
                    by_name = k in ("CONCENTRATION", "RATING_FLOOR", "RESTRICTED") and i.symbol == v.group
                    by_group = bool(col) and i.symbol in attr.index and str(attr.at[i.symbol, col]) == str(v.group) and (k != "BENCHMARK_BAND" or v.value > v.bound)
                    by_issuer = k == "ISSUER_WEIGHT" and bonds.get(i.symbol) is not None and bonds[i.symbol].issuer == v.group
                    if by_name or by_group or by_issuer:
                        out.append(i)
            elif k in ("DURATION", "SPREAD_DURATION", "DTS", "KEY_RATE", "LEVERAGE"):
                col = {"DURATION": "duration", "SPREAD_DURATION": "spread_duration", "DTS": "dts", "KEY_RATE": c.group_col, "LEVERAGE": None}[k]
                direction = 1.0 if v.value > v.bound else -1.0
                best, best_c = None, 0.0
                for i in buys:
                    if i in out or i.symbol not in attr.index:
                        continue
                    contrib = float(attr.at[i.symbol, col]) if col and col in attr else 1.0
                    if contrib * direction > best_c:
                        best, best_c = i, contrib * direction
                if best is not None:
                    out.append(best)
        return out

    @staticmethod
    def _projected_rows(rows: pd.DataFrame | None, intents: list[OrderIntent], dirty_per_face: dict[str, float], nav: float, candidate_rows: pd.DataFrame | None = None, bonds: dict[str, BondSpec] | None = None) -> pd.DataFrame:
        """Post-trade rows: holdings plus a row for every name an intent touches (candidate rows,
        else attributes from the terms), weights moved by the intents."""
        base = rows.copy() if rows is not None and not rows.empty else pd.DataFrame(columns=["symbol", "weight"])
        if "symbol" not in base:
            return base
        have = set(base["symbol"])
        extra = []
        for i in intents:
            if i.symbol in have:
                continue
            have.add(i.symbol)
            if candidate_rows is not None and not candidate_rows.empty and "symbol" in candidate_rows and (candidate_rows["symbol"] == i.symbol).any():
                extra.append(candidate_rows[candidate_rows["symbol"] == i.symbol].iloc[[0]].assign(weight=0.0))
            elif bonds is not None and i.symbol in bonds:
                b = bonds[i.symbol]
                extra.append(pd.DataFrame([{"symbol": i.symbol, "issuer": b.issuer, "sector": b.sector, "country": b.country, "currency": b.currency, "seniority": b.seniority.value, "rating_score": b.rating_score_initial, "weight": 0.0, "duration": 0.0, "spread_duration": 0.0, "dts": 0.0, "liquidity_rank": 3}]))
        out = pd.concat([base, *extra], ignore_index=True) if extra else base.copy()
        if out.empty:
            return out
        out["weight"] = out["weight"].astype(float)
        for i in intents:
            dv = dirty_per_face.get(i.symbol, 1.0)
            dw = (i.face * dv / nav) * (1.0 if i.side == "BUY" else -1.0) if nav else 0.0
            out.loc[out["symbol"] == i.symbol, "weight"] += dw
        return out


__all__ = ["ConstructionResult", "OrderIntent", "PortfolioConstructor", "TargetPortfolio"]
