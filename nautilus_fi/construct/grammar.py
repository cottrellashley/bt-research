"""One serialisable constraint grammar for every construction method.

The optimiser enforces these constraints; heuristic methods (quantiles, rank weights, tilts) are
post-checked against them so every rebalance records the same violations whichever method built it.

Kinds (``min`` / ``max`` are fractions of NAV unless the field says otherwise):

* ``budget``            sum of weights in [min, max] (fully invested: min = max = 1)
* ``cash``              1 - sum of weights in [min, max]
* ``weight``            each weight in [min, max]
* ``active_weight``     each weight minus its benchmark weight in [min, max]
* ``group``             sum of weights per value of ``attribute`` (or only ``value``) in [min, max]
* ``active_group``      the same, relative to the benchmark (sector / rating / issuer bands)
* ``exposure``          sum of w x ``attribute`` (a numeric field: oasd, dts, krd_10 ...) in [min, max],
                        optionally per ``group_by`` value
* ``active_exposure``   the same relative to the benchmark (OASD within +/-0.25y, DTS within +/-5% ...);
                        ``relative=True`` expresses the band as a fraction of the benchmark's exposure
* ``turnover``          turnover under ``definition`` (HALF_SUM default) at most ``max``
* ``leverage``          sum of absolute weights at most ``max``
* ``cardinality``       number of holdings in [min, max] (mixed-integer)
* ``min_holding``       each holding zero or at least ``min`` (mixed-integer; ``min=None`` uses the
                        cross-section's minimum-piece weights)
* ``cvar``              conditional value at risk of the (active, when ``active=True``) loss over the
                        cross-section's scenarios at level ``beta``, at most ``max``

``hard=False`` makes a constraint soft: violations are allowed at ``penalty`` per unit in the
objective and are reported as relaxations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

KINDS = ("budget", "cash", "weight", "active_weight", "group", "active_group", "exposure", "active_exposure", "turnover", "leverage", "cardinality", "min_holding", "cvar")


@dataclass(frozen=True)
class ConstraintSpec:
    kind: str
    min: float | None = None
    max: float | None = None
    attribute: str | None = None
    value: str | None = None
    group_by: str | None = None
    definition: str = "HALF_SUM"
    relative: bool = False
    active: bool = True
    beta: float = 0.95
    hard: bool = True
    penalty: float = 100.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown constraint kind {self.kind!r}; known: {KINDS}")
        if self.kind in ("group", "active_group", "exposure", "active_exposure") and not self.attribute:
            raise ValueError(f"{self.kind} constraint needs an attribute")

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        parts = [self.kind]
        if self.attribute:
            parts.append(self.attribute)
        if self.value is not None:
            parts.append(str(self.value))
        if self.group_by:
            parts.append(f"by {self.group_by}")
        return ":".join(parts)

    @staticmethod
    def from_dict(d: dict) -> ConstraintSpec:
        d = dict(d)
        if "type" in d and "kind" not in d:
            d["kind"] = d.pop("type")
        return ConstraintSpec(**d)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def parse_constraints(items) -> list[ConstraintSpec]:
    return [c if isinstance(c, ConstraintSpec) else ConstraintSpec.from_dict(c) for c in (items or [])]


def _groups(data: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute not in data:
        raise KeyError(f"cross-section has no attribute {attribute!r} for a group constraint")
    return data[attribute].astype(str)


def check_constraints(weights: pd.Series, cs, constraints: list[ConstraintSpec], tol: float = 1e-6) -> list[dict]:
    """Violations of ``constraints`` by ``weights`` on the cross-section ``cs`` (any method)."""
    w = weights.reindex(cs.ids).fillna(0.0)
    wb = cs.benchmark.reindex(cs.ids).fillna(0.0)
    w0 = cs.current.reindex(cs.ids).fillna(0.0)
    out: list[dict] = []

    def chk(c: ConstraintSpec, value: float, group: str | None = None, lo: float | None = None, hi: float | None = None) -> None:
        lo = c.min if lo is None else lo
        hi = c.max if hi is None else hi
        if hi is not None and value > hi + tol:
            out.append({"constraint": c.label, "group": group, "value": float(value), "bound": float(hi), "side": "max", "severity": "HARD" if c.hard else "SOFT"})
        if lo is not None and value < lo - tol:
            out.append({"constraint": c.label, "group": group, "value": float(value), "bound": float(lo), "side": "min", "severity": "HARD" if c.hard else "SOFT"})

    for c in constraints:
        k = c.kind
        if k == "budget":
            chk(c, float(w.sum()))
        elif k == "cash":
            chk(c, float(1.0 - w.sum()))
        elif k in ("weight", "active_weight"):
            v = w - wb if k == "active_weight" else w
            for sym in v.index[(v > (c.max if c.max is not None else np.inf) + tol) | (v < (c.min if c.min is not None else -np.inf) - tol)]:
                chk(c, float(v[sym]), sym)
        elif k in ("group", "active_group"):
            g = _groups(cs.data, c.attribute)
            sums = (w - wb if k == "active_group" else w).groupby(g).sum()
            for grp, v in sums.items():
                if c.value is None or str(grp) == str(c.value):
                    chk(c, float(v), str(grp))
        elif k in ("exposure", "active_exposure"):
            x = cs.data[c.attribute].reindex(cs.ids).astype(float).fillna(0.0)
            a = (w - wb) if k == "active_exposure" else w
            keys = cs.data[c.group_by].astype(str) if c.group_by else pd.Series("ALL", index=cs.ids)
            for grp, idx in keys.groupby(keys).groups.items():
                val = float((a[idx] * x[idx]).sum())
                if k == "active_exposure" and c.relative:
                    bexp = float((wb[idx] * x[idx]).sum())
                    lo = None if c.min is None else c.min * abs(bexp)
                    hi = None if c.max is None else c.max * abs(bexp)
                    chk(c, val, None if grp == "ALL" else str(grp), lo, hi)
                else:
                    chk(c, val, None if grp == "ALL" else str(grp))
        elif k == "turnover":
            from nautilus_fi.domain.conventions import TurnoverDefinition, turnover

            chk(c, turnover(w0, w, TurnoverDefinition(c.definition)))
        elif k == "leverage":
            chk(c, float(w.abs().sum()))
        elif k == "cardinality":
            chk(c, float((w.abs() > 1e-9).sum()))
        elif k == "min_holding":
            m = pd.Series(c.min, index=cs.ids) if c.min is not None else (cs.min_weight.reindex(cs.ids).fillna(0.0) if cs.min_weight is not None else pd.Series(0.0, index=cs.ids))
            bad = (w.abs() > 1e-9) & (w.abs() < m - tol)
            for sym in w.index[bad]:
                out.append({"constraint": c.label, "group": sym, "value": float(w[sym]), "bound": float(m[sym]), "side": "min", "severity": "HARD" if c.hard else "SOFT"})
        elif k == "cvar" and cs.scenarios is not None:
            R = cs.scenarios.reindex(columns=cs.ids).fillna(0.0).to_numpy()
            a = (w - wb) if c.active else w
            loss = -(R @ a.to_numpy())
            q = np.quantile(loss, c.beta)
            chk(c, float(loss[loss >= q].mean()))
    return out


__all__ = ["KINDS", "ConstraintSpec", "check_constraints", "parse_constraints"]
