"""MRI v1 (``fi-mdl-macrocredit``) and MRI v2: index timing on a macro regime indicator (gap
analysis §5.3, SIG-C2, SIG-C4).

Conventions: the ``MRI_v1`` profile (the signal on close T earns close T -> T+1; daily excess
returns summed, in the legacy profile only). The components:

* ``PriceAdjustCost`` / ``cost``: the per-universe cost (5 / 20 / 10 / 25bp) applied as a signed
  adjustment to the execution price: a buy pays ``price x (1 + c)``, a sale receives
  ``price x (1 - c)``; in ``additive`` mode the adjustment is a log difference, for runs whose
  P&L sums log returns (SIG-C2, SIG-C4);
* ``additive_metrics``: the non-compounded P&L (returns summed);
* ``v2_signal``: MRI v2's OR rule across two regime signals, each input carried at most 5
  calendar days;
* ``decimal_guard``: inputs given in percent where decimals are expected fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance
from nautilus_fi.kernel.data import DataQualityError
from nautilus_fi.kernel.expr import Field, where
from nautilus_fi.kernel.mechanics import CostModel

PROFILE = "MRI_v1"
REQUIREMENTS = ("SIG-C2", "SIG-C4")
COST_BP = {"US_IG": 5.0, "US_HY": 20.0, "EU_IG": 10.0, "EU_HY": 25.0}       # to confirm: the review lists 5/20/10/25bp, mapped here in that order


@dataclass(frozen=True)
class PriceAdjustCost(CostModel):
    """A signed execution-price adjustment of ``bp``: buys at ``(1 + c)`` times the price cost
    ``c / (1 + c)`` of the value bought, sales at ``(1 - c)`` cost ``c`` of the value sold.
    ``additive``: the adjustment as a log difference (``log(1 + c)`` on buys, ``-log(1 - c)`` on
    sales), for P&L that sums log returns."""

    bp: float = 5.0
    additive: bool = False

    def _rates(self) -> tuple[float, float]:
        c = self.bp / 1e4
        if self.additive:
            return float(np.log1p(c)), float(-np.log1p(-c))
        return c / (1.0 + c), c

    def unit(self, ctx, t, trade, nav):
        b, s = self._rates()
        return np.full(ctx.N, 0.5 * (b + s))

    def total(self, ctx, t, trade, nav):
        tr = np.asarray(trade, dtype=np.float64)
        b, s = self._rates()
        return float(tr[tr > 0].sum() * b - tr[tr < 0].sum() * s)

    def __repr__(self) -> str:
        return f"PriceAdjustCost({self.bp:g}{', additive=True' if self.additive else ''})"


def cost(universe: str, additive: bool = False) -> PriceAdjustCost:
    """The universe's cost level (``COST_BP``) as a price adjustment."""
    key = universe.upper().replace(" ", "_")
    if key not in COST_BP:
        raise KeyError(f"no MRI cost level for {universe!r}; known: {sorted(COST_BP)}")
    return PriceAdjustCost(COST_BP[key], additive)


def additive_metrics(**kw):
    """Non-compounded P&L: annual return = summed returns per year, NAV = 1 + cumulative sum.
    For MRI's log-difference runs, give the source the log returns declared as simple
    (``FieldInfo(form="simple")``) so they are summed as they are."""
    from nautilus_fi.kernel.mechanics import Metrics

    return Metrics(pnl="ADDITIVE", **kw)


def v2_signal(a: str, b: str, a_threshold: float, b_threshold: float, a_above: bool = True, b_above: bool = True, carry_days: int = 5):
    """MRI v2's regime rule: on (1.0) when either signal is through its threshold, else off (0.0);
    each input carried at most ``carry_days`` calendar days, after which it counts as missing
    (and a missing input does not switch the regime on)."""
    fa, fb = Field(a).ffill(days=carry_days), Field(b).ffill(days=carry_days)
    on_a = (fa > a_threshold) if a_above else (fa < a_threshold)
    on_b = (fb > b_threshold) if b_above else (fb < b_threshold)
    return where(on_a | on_b, 1.0, 0.0)


def decimal_guard(values: pd.Series | pd.DataFrame, name: str, max_abs: float = 1.0, share: float = 0.5) -> pd.Series | pd.DataFrame:
    """Refuse inputs in the wrong unit: when more than ``share`` of the values exceed
    ``max_abs`` in magnitude, the series is in percent (or bp), not decimals."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) and float((v > max_abs).mean()) > share:
        raise DataQualityError(f"{name}: {float((v > max_abs).mean()):.0%} of values exceed {max_abs:g} in magnitude (median {np.median(v):.3g}): give decimals, not percent")
    return values


WINDOW_PRESETS: dict[str, dict] = {}                     # MRI v2's named windows: fill from its config (the review does not give the values)


_TOL = (Tolerance("returns", bp_per_year=5.0, min_corr=0.999), Tolerance("table:position", abs=1e-9))       # proposed; agree under §9.2
PARITY = [ParityCase(u, "mri", "fi-mdl-macrocredit stored backtest tables", _TOL) for u in ("US HY", "US IG", "EU IG", "EU HY")]

__all__ = ["COST_BP", "PARITY", "PROFILE", "REQUIREMENTS", "WINDOW_PRESETS", "PriceAdjustCost", "additive_metrics", "cost", "decimal_guard", "v2_signal"]
