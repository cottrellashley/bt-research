"""Constraints with economic units: hard restrictions versus soft preferences, with every relaxation
recorded when a feasible solution needs one."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class ConstraintKind(str, Enum):
    ISSUER_WEIGHT = "ISSUER_WEIGHT"
    GROUP_WEIGHT = "GROUP_WEIGHT"           # sector / country / currency / rating bucket via ``group_col``
    SENIORITY = "SENIORITY"
    CONCENTRATION = "CONCENTRATION"         # single position weight
    DURATION = "DURATION"
    KEY_RATE = "KEY_RATE"
    DTS = "DTS"
    SPREAD_DURATION = "SPREAD_DURATION"
    LEVERAGE = "LEVERAGE"
    TURNOVER = "TURNOVER"
    LIQUIDITY = "LIQUIDITY"                 # min share in liquidity bucket <= rank
    CASH = "CASH"
    RATING_FLOOR = "RATING_FLOOR"           # max rating score (worst rating allowed)
    BENCHMARK_BAND = "BENCHMARK_BAND"
    RESTRICTED = "RESTRICTED"


@dataclass(frozen=True)
class Constraint:
    name: str
    kind: ConstraintKind
    unit: str                               # "weight", "years", "years*bp", "x NAV", "fraction/yr", "score"
    minimum: float | None = None
    maximum: float | None = None
    hard: bool = True
    group_col: str | None = None            # column used by GROUP_WEIGHT / LIQUIDITY / KEY_RATE
    group_value: str | None = None          # specific group (None = each group)
    scope: str = "*"


@dataclass
class Violation:
    constraint: Constraint
    group: str | None
    value: float
    bound: float
    severity: str                           # HARD | SOFT

    def describe(self) -> str:
        g = f"[{self.group}] " if self.group else ""
        return f"{self.severity} {self.constraint.name} {g}{self.value:.4g} vs bound {self.bound:.4g} ({self.constraint.unit})"


@dataclass
class Relaxation:
    constraint: str
    group: str | None
    from_bound: float
    to_bound: float
    reason: str


class ConstraintSet:
    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self.constraints = list(constraints or [])
        self.relaxations: list[Relaxation] = []

    def add(self, c: Constraint) -> Constraint:
        self.constraints.append(c)
        return c

    def relax(self, name: str, new_bound: float, reason: str, group: str | None = None) -> None:
        for i, c in enumerate(self.constraints):
            if c.name == name:
                old = c.maximum if c.maximum is not None else c.minimum
                self.relaxations.append(Relaxation(name, group, old if old is not None else float("nan"), new_bound, reason))
                self.constraints[i] = Constraint(c.name, c.kind, c.unit, c.minimum if c.maximum is not None else new_bound, new_bound if c.maximum is not None else c.maximum, c.hard, c.group_col, c.group_value, c.scope)

    @staticmethod
    def _check(c: Constraint, value: float, group: str | None, out: list[Violation]) -> None:
        if c.maximum is not None and value > c.maximum + 1e-12:
            out.append(Violation(c, group, value, c.maximum, "HARD" if c.hard else "SOFT"))
        if c.minimum is not None and value < c.minimum - 1e-12:
            out.append(Violation(c, group, value, c.minimum, "HARD" if c.hard else "SOFT"))

    def evaluate(self, rows: pd.DataFrame, nav: float, turnover: float = 0.0, cash: float = 0.0, benchmark_weights: dict[str, float] | None = None) -> list[Violation]:
        """``rows``: one row per position with columns weight (signed MV/NAV), issuer, sector, country,
        currency, rating_bucket, rating_score, seniority, duration, spread_duration, dts, liquidity_rank, krd_<tenor>..."""
        out: list[Violation] = []
        if rows is None:
            rows = pd.DataFrame()
        w = rows["weight"] if "weight" in rows else pd.Series(dtype=float)
        for c in self.constraints:
            if c.kind == ConstraintKind.ISSUER_WEIGHT and "issuer" in rows:
                for g, v in rows.groupby("issuer")["weight"].sum().items():
                    self._check(c, abs(v), str(g), out)
            elif c.kind == ConstraintKind.GROUP_WEIGHT and c.group_col in rows:
                for g, v in rows.groupby(c.group_col)["weight"].sum().items():
                    if c.group_value is None or str(g) == c.group_value:
                        self._check(c, v, str(g), out)
            elif c.kind == ConstraintKind.CONCENTRATION and len(w):
                for sym, v in zip(rows.get("symbol", rows.index), w):
                    self._check(c, abs(v), str(sym), out)
            elif c.kind in (ConstraintKind.DURATION, ConstraintKind.SPREAD_DURATION, ConstraintKind.DTS) and len(w):
                col = {"DURATION": "duration", "SPREAD_DURATION": "spread_duration", "DTS": "dts"}[c.kind.value]
                if col in rows:
                    self._check(c, float((w * rows[col].fillna(0.0)).sum()), None, out)
            elif c.kind == ConstraintKind.KEY_RATE and c.group_col in rows:
                self._check(c, float((w * rows[c.group_col].fillna(0.0)).sum()), c.group_col, out)
            elif c.kind == ConstraintKind.LEVERAGE and len(w):
                self._check(c, float(w.abs().sum()), None, out)
            elif c.kind == ConstraintKind.TURNOVER:
                self._check(c, turnover, None, out)
            elif c.kind == ConstraintKind.CASH:
                self._check(c, cash / nav if nav else 0.0, None, out)
            elif c.kind == ConstraintKind.RATING_FLOOR and "rating_score" in rows:
                for sym, score, v in zip(rows.get("symbol", rows.index), rows["rating_score"], w):
                    if score is not None and not pd.isna(score) and v > 0:
                        self._check(c, float(score), str(sym), out)
            elif c.kind == ConstraintKind.LIQUIDITY and "liquidity_rank" in rows and c.group_value is not None:
                share = float(w[rows["liquidity_rank"] <= int(c.group_value)].sum())
                self._check(c, share, f"rank<={c.group_value}", out)
            elif c.kind == ConstraintKind.SENIORITY and "seniority" in rows and c.group_value is not None:
                self._check(c, float(w[rows["seniority"] == c.group_value].sum()), c.group_value, out)
            elif c.kind == ConstraintKind.BENCHMARK_BAND and benchmark_weights and c.group_col in rows:
                for g, v in rows.groupby(c.group_col)["weight"].sum().items():
                    self._check(c, v - benchmark_weights.get(str(g), 0.0), str(g), out)
            elif c.kind == ConstraintKind.RESTRICTED and "symbol" in rows and c.group_value is not None:
                for sym, v in zip(rows["symbol"], w):
                    if sym == c.group_value and abs(v) > 1e-12:
                        out.append(Violation(c, str(sym), abs(v), 0.0, "HARD"))
        return out

    def hard_violations(self, violations: list[Violation]) -> list[Violation]:
        return [v for v in violations if v.severity == "HARD"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"name": c.name, "kind": c.kind.value, "unit": c.unit, "min": c.minimum, "max": c.maximum, "hard": c.hard, "group_col": c.group_col, "group_value": c.group_value} for c in self.constraints])


__all__ = ["Constraint", "ConstraintKind", "ConstraintSet", "Relaxation", "Violation"]
_ = field
