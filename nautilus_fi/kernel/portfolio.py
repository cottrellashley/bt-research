"""Portfolio construction as objects, and an optimiser written as Python expressions.

Construction methods turn scores, the universe and the benchmark into weights:

* ``EqualWeight()``, ``TopN(50, weighting=ByField("amt_out"))``, ``Quantile(5, long=5, short=1,
  by="sector")``, ``RankWeighted(power=2)``, ``BenchmarkWeight()``, ``Tilt(0.5, max_active=0.01)``,
  ``Hold(expr)`` (weights straight from an expression), ``Pairs([...], exposure="dts")``;
* ``.cap(issuer=0.03)`` caps after weighting, ``.subject_to(*constraints)`` projects the weights onto
  constraints (the nearest feasible portfolio in L1), ``.checked(*constraints)`` only records
  violations;
* ``Optimise(objective, *constraints)`` solves an LP / MILP / QP with HiGHS.

Path-independent methods are evaluated for every rebalance date in one vectorised pass; methods
that depend on the current portfolio (the optimiser, buffers, projections) run per date.

The optimiser reads like the problem::

    Optimise(3e-4 * Alpha() - TrackingError(by=["sector", "rating"]) - Costs(),
             ActiveWeight("issuer") <= 0.005, ActiveWeight("sector").within(0.02),
             ActiveExposure("oasd").within(0.10), Turnover() <= 0.10, Names() <= 150)

Objective terms: ``Alpha(signal=None)`` (the strategy's signal or any expression), ``Linear(expr)``
(any per-instrument coefficient, e.g. carry), and penalties with negative coefficients:
``Costs()``, ``TrackingError(by, fields)`` (L1 of active bucket exposures), ``ActiveShare()``,
``Risk(model)`` (factor variance, QP; ``PCARisk`` builds one from history) and ``CVaR(beta)``.
Constraints are comparisons on measures: ``Weight``, ``ActiveWeight(group)``, ``GroupWeight``,
``Exposure``, ``ActiveExposure``, ``Turnover``, ``Leverage``, ``Budget``, ``CashWeight``, ``Names``,
``CVaRLimit``, ``LinearMeasure(expr)`` (custom), and ``MinHolding(w)``; ``.soft(penalty)`` makes one
soft. ``optimise.replace(Turnover() <= 0.05)`` swaps the constraint in the same slot.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import numpy as np
import pandas as pd

from nautilus_fi.benchmarks.benchmark import cap_weights
from nautilus_fi.construct.cross_section import CrossSection
from nautilus_fi.construct.grammar import ConstraintSpec, check_constraints
from nautilus_fi.construct.methods import HedgeOverlay
from nautilus_fi.construct.optimiser import (
    CVaRTerm,
    LinearRow,
    OptimiserSettings,
    RiskModel,
    TrackingTerm,
    optimise,
)
from nautilus_fi.construct.optimiser import Objective as _LegacyObjective
from nautilus_fi.domain.conventions import TurnoverDefinition
from nautilus_fi.kernel.expr import Cat, Expr, Field, _fmt, factorize, group_stats, grouped_rank, lift


# ====================================================================================== cross-sections
@dataclasses.dataclass
class KernelCross(CrossSection):
    """A cross-section that can evaluate kernel expressions at its date (``ctx`` row ``t``,
    panel columns ``cols`` of its ids)."""

    ctx: object = None
    t: int = -1
    cols: np.ndarray | None = None
    exits: str = "SELL"                        # held names outside the universe: SELL | EXCLUDE | HOLD_NOT_BUY
    lb_extra: np.ndarray | None = None         # per-name weight bounds from Bounds / Pin / NoSell / TradeLimit
    ub_extra: np.ndarray | None = None
    overlay_data: pd.DataFrame | None = None   # Hedge(joint=True): overlay exposures per unit notional (instrument x field)
    overlay_current: np.ndarray | None = None  # their current notionals
    overlay_cost_bp: float = 0.0
    overlay_max: float | None = None
    overlay_target: np.ndarray | None = None   # the solved notionals, for the checks

    def values(self, x) -> np.ndarray:
        return _cs_values(self, x)


def _cs_values(cs: CrossSection, x) -> np.ndarray:
    """Values of a field name or expression for the cross-section's ids."""
    if isinstance(x, str):
        x = Field(x)
    if type(x) is Field and x.lag_days is None and x.name in cs.data.columns:
        return pd.to_numeric(cs.data[x.name], errors="coerce").to_numpy(dtype=np.float64)
    ctx = getattr(cs, "ctx", None)
    if ctx is None:
        raise KeyError(f"{x!r} is not in the cross-section data ({sorted(cs.data.columns)[:20]}...) and no kernel context is attached")
    v = x.evaluate(ctx.full())
    if isinstance(v, Cat):
        raise TypeError(f"{x!r} is a category; numeric values are needed here")
    row = np.asarray(v[cs.t], dtype=np.float64)
    cols = np.asarray(cs.cols, dtype=np.int64)
    return np.where(cols >= 0, row[np.clip(cols, 0, None)], np.nan)          # ids outside the panel: NaN


@dataclasses.dataclass
class Batch:
    """Inputs of a construction for R decision rows over n instruments (all panel instruments in
    the vectorised path; one cross-section in the per-date path)."""

    scores: np.ndarray | None                 # (R, n) NaN outside the universe
    universe: np.ndarray                      # (R, n) bool
    benchmark: np.ndarray | None              # (R, n)
    current: np.ndarray | None = None         # (R, n) (per-date path only)
    ctx: object = None
    rows: np.ndarray | None = None            # panel rows of the R decisions
    cols: np.ndarray | None = None            # panel columns of the n instruments (None: all)
    cs: CrossSection | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.universe.shape

    @staticmethod
    def from_cs(cs: CrossSection) -> Batch:
        el = cs.eligible.to_numpy(dtype=bool)
        sc = None if cs.scores is None else np.where(el, cs.scores.to_numpy(dtype=np.float64), np.nan)[None, :]
        return Batch(sc, el[None, :], cs.benchmark.to_numpy(dtype=np.float64)[None, :], cs.current.to_numpy(dtype=np.float64)[None, :], getattr(cs, "ctx", None), None if getattr(cs, "t", -1) < 0 else np.array([cs.t]), getattr(cs, "cols", None), cs)

    def field(self, x) -> np.ndarray:
        if self.cs is not None:
            return _cs_values(self.cs, x)[None, :]
        e = Field(x) if isinstance(x, str) else x
        v = e.evaluate(self.ctx.full())
        return np.asarray(v, dtype=np.float64)[self.rows]

    def groups(self, by) -> Cat:
        R, n = self.shape
        if by is None:
            return Cat(np.zeros(n, dtype=np.int32), ["ALL"])
        if self.cs is not None:
            if isinstance(by, (list, tuple)):
                from nautilus_fi.kernel.expr import combine_codes

                return combine_codes([self.groups(b) for b in by], R, n)
            if isinstance(by, str) and by in self.cs.data.columns:
                return factorize(self.cs.data[by].tolist())
            ctx = self.ctx
            if ctx is None:
                raise KeyError(f"group {by!r} is not in the cross-section data")
            g = ctx.groups(by)
            cols = np.asarray(self.cs.cols, dtype=np.int64)
            row = g.broadcast(ctx.T, ctx.N)[self.cs.t]
            return Cat(np.where(cols >= 0, row[np.clip(cols, 0, None)], -1).astype(np.int32), g.categories)
        g = self.ctx.groups(by)
        codes = g.codes if g.codes.ndim == 1 else g.codes[self.rows]
        return Cat(codes, g.categories)


def _normalise(raw: np.ndarray, total: float | np.ndarray = 1.0) -> np.ndarray:
    raw = np.where(np.isfinite(raw), raw, 0.0)
    s = raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, raw / np.where(s > 0, s, 1.0), 0.0) * total


# ====================================================================================== weighting schemes
class Weighting:
    def apply(self, sel: np.ndarray, B: Batch) -> np.ndarray:
        raise NotImplementedError

    def names(self) -> set[str]:
        return set()


class Equal(Weighting):
    def apply(self, sel, B):
        return _normalise(sel.astype(np.float64))

    def __repr__(self) -> str:
        return "Equal()"


class ByField(Weighting):
    """Weights proportional to a field or expression (market value, amount outstanding)."""

    def __init__(self, x) -> None:
        self.x = Field(x) if isinstance(x, str) else lift(x)

    def names(self):
        return self.x.names()

    def apply(self, sel, B):
        v = np.clip(np.nan_to_num(B.field(self.x)), 0.0, None)
        return _normalise(np.where(sel, v, 0.0))

    def __repr__(self) -> str:
        return f"ByField({self.x.name!r})" if type(self.x) is Field else f"ByField({self.x!r})"


class InverseField(ByField):
    """Weights proportional to 1 / x (inverse DTS, inverse duration: risk-balanced legs)."""

    def apply(self, sel, B):
        v = B.field(self.x)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.isfinite(v) & (v > 0), 1.0 / v, 0.0)
        return _normalise(np.where(sel, inv, 0.0))

    def __repr__(self) -> str:
        return f"InverseField({self.x.name!r})" if type(self.x) is Field else f"InverseField({self.x!r})"


class ByBenchmark(Weighting):
    """Benchmark weights inside the selection (equal weights where the selection holds none)."""

    def apply(self, sel, B):
        if B.benchmark is None:
            return Equal().apply(sel, B)
        bw = np.where(sel, np.nan_to_num(B.benchmark), 0.0)
        eq = _normalise(sel.astype(np.float64))
        out = _normalise(bw)
        return np.where(bw.sum(axis=1, keepdims=True) > 0, out, eq)

    def __repr__(self) -> str:
        return "ByBenchmark()"


class ByScore(Weighting):
    """Weights proportional to the score's percentile rank within the selection, to a power."""

    def __init__(self, power: float = 1.0) -> None:
        self.power = float(power)

    def apply(self, sel, B):
        if B.scores is None:
            return Equal().apply(sel, B)
        R, n = sel.shape
        x = np.where(sel, B.scores, np.nan)
        r = grouped_rank(x, Cat(np.zeros(n, dtype=np.int32), ["ALL"]), R, n)
        return _normalise(np.where(sel, np.nan_to_num(r) ** self.power, 0.0))

    def __repr__(self) -> str:
        return f"ByScore({self.power:g})"


def as_weighting(w) -> Weighting:
    if isinstance(w, Weighting):
        return w
    if w in (None, "equal"):
        return Equal()
    if w == "benchmark":
        return ByBenchmark()
    if isinstance(w, (str, Expr)):
        return ByField(w)
    raise TypeError(f"weighting must be a Weighting, 'equal', 'benchmark' or a field, not {w!r}")


# ====================================================================================== base
class Construction:
    """Base class. Implement ``batch(B) -> (R, n) weights`` (vectorised) and/or
    ``weights(cs) -> (Series, diagnostics)``."""

    path_dependent = False                     # needs the current portfolio: run per date
    needs_scores = True

    def batch(self, B: Batch) -> np.ndarray | None:
        return None

    def weights(self, cs: CrossSection) -> tuple[pd.Series, dict]:
        out = self.batch(Batch.from_cs(cs))
        if out is None:
            raise NotImplementedError(f"{type(self).__name__} implements neither batch nor weights")
        w = out[0]
        nz = np.abs(w) > 1e-15
        return pd.Series(w[nz], index=cs.ids[nz]), {}

    def constraints(self) -> tuple:
        return ()

    def names(self) -> set[str]:
        out: set[str] = set()
        for c in self.constraints():
            out |= c.names()
        return out

    def check(self, weights: pd.Series, cs: CrossSection) -> list[dict]:
        return check_all(self.constraints(), weights, cs)

    def overlays(self) -> Hedge | None:
        return None

    # ------------------------------------------------------------------ combinators
    def cap(self, issuer: float | None = None, name: float | None = None, by: str | None = None, max_weight: float | None = None) -> Construction:
        out: Construction = self
        if issuer is not None or by is not None:
            out = Capped(out, by or "issuer", float(issuer if issuer is not None else max_weight))
        if name is not None:
            out = Capped(out, None, float(name))
        return out

    def subject_to(self, *constraints: Constraint, settings: OptimiserSettings | None = None) -> Construction:
        return Projected(self, tuple(constraints), settings)

    def checked(self, *constraints: Constraint) -> Construction:
        return Checked(self, tuple(constraints))

    def replace(self, *constraints: Constraint) -> Construction:
        raise TypeError(f"{type(self).__name__} has no constraints to replace; use .subject_to(...) or Optimise(...)")

    def with_(self, **changes) -> Construction:
        """A changed copy by constructor argument: ``TopN(50).with_(n=30)``."""
        kw = self._kw()
        unknown = set(changes) - set(kw)
        if unknown:
            raise TypeError(f"{type(self).__name__} has no argument(s) {sorted(unknown)}; arguments: {sorted(kw)}")
        return type(self)(**{**kw, **changes})

    def _kw(self) -> dict:
        import inspect

        params = inspect.signature(type(self).__init__).parameters
        return {k: getattr(self, k) for k, p in params.items() if k != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and hasattr(self, k)}


# ====================================================================================== heuristics
class EqualWeight(Construction):
    """Equal weights over the universe, or over the ``top`` names by score."""

    needs_scores = False

    def __init__(self, top: int | None = None, invested: float = 1.0) -> None:
        self.top, self.invested = top, float(invested)

    def batch(self, B):
        sel = B.universe
        if self.top is not None and B.scores is not None:
            sel = _top_mask(B.scores, self.top, B, None)
        return _normalise(sel.astype(np.float64), self.invested)

    def __repr__(self) -> str:
        return f"EqualWeight({'' if self.top is None else f'top={self.top}'}{'' if self.invested == 1.0 else (', ' if self.top is not None else '') + f'invested={self.invested:g}'})"


def _top_mask(scores: np.ndarray, n: int, B: Batch, by, bottom: bool = False) -> np.ndarray:
    R, m = scores.shape
    x = scores if not bottom else -scores
    g = B.groups(by)
    G = max(len(g.categories), 1)
    if G == 1:
        from nautilus_fi.kernel.expr import row_rank

        r = row_rank(-x, "first")
        return np.nan_to_num(r, nan=np.inf) <= n
    pct = grouped_rank(-x, g, R, m, "first")
    codes = g.broadcast(R, m)
    ok = np.isfinite(pct) & (codes >= 0)
    flat = (np.arange(R, dtype=np.int64)[:, None] * G + codes)[ok]
    cnt = np.bincount(flat, minlength=R * G).reshape(R, G)
    out = np.zeros((R, m), dtype=bool)
    out[ok] = np.round(pct[ok] * cnt.reshape(-1)[flat]) <= n
    return out


class TopN(Construction):
    """The ``n`` best names by score (per group with ``by=``), weighted by ``weighting``.
    ``buffer``: a holding stays while it ranks within ``n + buffer`` (fewer trades)."""

    def __init__(self, n: int, weighting=None, by=None, invested: float = 1.0, buffer: int = 0) -> None:
        self.n, self.weighting, self.by, self.invested, self.buffer = int(n), as_weighting(weighting), by, float(invested), int(buffer)
        self.path_dependent = type(self).path_dependent or self.buffer > 0          # a subclass may run per date anyway

    def names(self):
        return self.weighting.names() | _by_names(self.by)

    def batch(self, B):
        if B.scores is None:
            raise ValueError("TopN needs a signal")
        if self.buffer and B.current is None:
            return None
        sel = _top_mask(B.scores, self.n, B, self.by)
        if self.buffer and B.current is not None:
            keep = (B.current != 0) & _top_mask(B.scores, self.n + self.buffer, B, self.by)
            R, _m = sel.shape
            out = np.zeros_like(sel)
            for r in range(R):
                k = keep[r]
                slots = self.n - int(k.sum())
                cand = np.where(k, np.nan, B.scores[r])
                if slots > 0:
                    order = np.argsort(-np.nan_to_num(cand, nan=-np.inf), kind="stable")[:slots]
                    order = order[np.isfinite(cand[order])]
                    k = k.copy()
                    k[order] = True
                out[r] = k
            sel = out
        return self.weighting.apply(sel, B) * self.invested

    def __repr__(self) -> str:
        parts = [str(self.n)]
        if not isinstance(self.weighting, Equal):
            parts.append(f"weighting={self.weighting!r}")
        if self.by is not None:
            parts.append(f"by={_fmt(self.by)}")
        if self.invested != 1.0:
            parts.append(f"invested={self.invested:g}")
        if self.buffer:
            parts.append(f"buffer={self.buffer}")
        return f"TopN({', '.join(parts)})"


class Quantile(Construction):
    """Score bins (``n`` per date, within ``by`` groups): hold bin ``long`` (default the top) and
    short bin ``short`` (default none), each leg weighted by ``weighting`` and scaled to
    ``invested`` / ``short_gross``."""

    def __init__(self, n: int = 5, long: int | None = None, short: int | None = None, by=None, weighting=None, invested: float = 1.0, short_gross: float | None = None, buffer: int = 0, keep_weights: bool = False) -> None:
        """``buffer``: a holding stays while within ``buffer`` bins of its leg (fewer trades);
        ``keep_weights``: names that stay keep their drifted weight, new names share the rest
        (drift-aware: no trades just to reset weights)."""
        self.n, self.long, self.short, self.by = int(n), long, short, by
        self.weighting, self.invested, self.short_gross = as_weighting(weighting), float(invested), short_gross
        self.buffer, self.keep_weights = int(buffer), bool(keep_weights)
        self.path_dependent = type(self).path_dependent or self.buffer > 0 or self.keep_weights

    def names(self):
        return self.weighting.names() | _by_names(self.by)

    def bins(self, B: Batch) -> np.ndarray:
        if B.scores is None:
            raise ValueError("Quantile needs a signal")
        R, m = B.shape
        pct = grouped_rank(B.scores, B.groups(self.by), R, m, "first")
        return np.where(np.isfinite(pct), np.clip(np.ceil(pct * self.n), 1, self.n), 0).astype(np.int32)

    def batch(self, B):
        if (self.buffer or self.keep_weights) and B.current is None:
            return None
        b = self.bins(B)
        top = self.long or self.n
        long_sel = b == top
        short_sel = (b == self.short) if self.short is not None else None
        if self.buffer and B.current is not None:
            long_sel |= (B.current > 0) & (b >= top - self.buffer) & (b > 0)
            if short_sel is not None:
                short_sel |= (B.current < 0) & (b <= self.short + self.buffer) & (b > 0)
        w = self._leg(long_sel, B, self.invested, B.current if self.keep_weights else None, 1.0)
        if short_sel is not None:
            w = w - self._leg(short_sel, B, self.short_gross if self.short_gross is not None else self.invested, B.current if self.keep_weights else None, -1.0)
        return w

    def _leg(self, sel, B, gross, current, sign):
        base = self.weighting.apply(sel, B) * gross
        if current is None:
            return base
        kept = np.where(sel & (np.sign(current) == sign), np.abs(current), 0.0)
        room = np.clip(gross - kept.sum(axis=1, keepdims=True), 0.0, None)
        new = sel & (kept == 0)
        fresh = self.weighting.apply(new, B)
        return np.where(kept > 0, kept, fresh * room)

    def __repr__(self) -> str:
        parts = [str(self.n)]
        if self.buffer:
            parts.append(f"buffer={self.buffer}")
        if self.keep_weights:
            parts.append("keep_weights=True")
        if self.long is not None:
            parts.append(f"long={self.long}")
        if self.short is not None:
            parts.append(f"short={self.short}")
        if self.by is not None:
            parts.append(f"by={_fmt(self.by)}")
        if not isinstance(self.weighting, Equal):
            parts.append(f"weighting={self.weighting!r}")
        if self.invested != 1.0:
            parts.append(f"invested={self.invested:g}")
        if self.short_gross is not None:
            parts.append(f"short_gross={self.short_gross:g}")
        return f"Quantile({', '.join(parts)})"


class RankWeighted(Construction):
    """Weights rising with the score's rank (``power``); ``long_short``: centred ranks, gross
    ``2 x invested``; ``top``: only the best ``top`` names."""

    def __init__(self, power: float = 1.0, long_short: bool = False, top: int | None = None, invested: float = 1.0) -> None:
        self.power, self.long_short, self.top, self.invested = float(power), bool(long_short), top, float(invested)

    def batch(self, B):
        if B.scores is None:
            raise ValueError("RankWeighted needs a signal")
        x = B.scores
        if self.top is not None:
            x = np.where(_top_mask(x, self.top, B, None), x, np.nan)
        R, m = x.shape
        r = grouped_rank(x, Cat(np.zeros(m, dtype=np.int32), ["ALL"]), R, m)
        ok = np.isfinite(r)
        if not self.long_short:
            return _normalise(np.where(ok, r ** self.power, 0.0), self.invested)
        mean = np.nanmean(np.where(ok, r, np.nan), axis=1, keepdims=True)
        v = np.where(ok, (r - mean) * 2.0, 0.0)
        v = np.sign(v) * np.abs(v) ** self.power
        s = np.abs(v).sum(axis=1, keepdims=True)
        return np.where(s > 0, v / np.where(s > 0, s, 1.0) * 2.0 * self.invested, 0.0)

    def __repr__(self) -> str:
        parts = [] if self.power == 1.0 else [f"power={self.power:g}"]
        if self.long_short:
            parts.append("long_short=True")
        if self.top is not None:
            parts.append(f"top={self.top}")
        return f"RankWeighted({', '.join(parts)})"


class BenchmarkWeight(Construction):
    """The benchmark's weights inside the universe, renormalised (index replication)."""

    needs_scores = False

    def batch(self, B):
        if B.benchmark is None:
            raise ValueError("BenchmarkWeight needs a benchmark with constituents")
        return _normalise(np.where(B.universe, np.nan_to_num(B.benchmark), 0.0))

    def __repr__(self) -> str:
        return "BenchmarkWeight()"


class Tilt(Construction):
    """Benchmark weights scaled by ``1 + strength x z(score)`` (z within ``by`` groups), active
    weights clipped to ``max_active`` and renormalised."""

    def __init__(self, strength: float = 0.5, max_active: float | None = None, by=None) -> None:
        self.strength, self.max_active, self.by = float(strength), max_active, by

    def names(self):
        return _by_names(self.by)

    def batch(self, B):
        if B.benchmark is None:
            raise ValueError("Tilt needs a benchmark with constituents")
        wb = np.nan_to_num(B.benchmark)
        mem = wb > 0
        R, m = wb.shape
        sc = np.where(mem, B.scores if B.scores is not None else 0.0, np.nan)
        fin = np.isfinite(sc)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(np.where(fin, sc, np.nan), axis=1, keepdims=True)
        sc = np.where(mem & ~fin, np.nan_to_num(mu), sc)
        g = B.groups(self.by)
        cnt, mean, ssd, valid, flat = group_stats(np.nan_to_num(sc), mem, g.codes, max(len(g.categories), 1))
        sd = np.sqrt(ssd / np.maximum(cnt, 1))
        z = np.zeros((R, m))
        sdf = sd.reshape(-1)[flat]
        z[valid] = np.where(sdf > 0, (np.nan_to_num(sc)[valid] - mean.reshape(-1)[flat]) / np.where(sdf > 0, sdf, 1.0), 0.0)
        tot = wb.sum(axis=1, keepdims=True)
        w = _normalise(np.clip(wb * (1.0 + self.strength * z), 0.0, None), tot)
        if self.max_active is not None:
            for _ in range(20):
                w2 = _normalise(np.clip(wb + np.clip(w - wb, -self.max_active, self.max_active), 0.0, None), tot)
                if np.allclose(w2, w, atol=1e-12):
                    break
                w = w2
        return w

    def __repr__(self) -> str:
        parts = [f"{self.strength:g}"]
        if self.max_active is not None:
            parts.append(f"max_active={self.max_active:g}")
        if self.by is not None:
            parts.append(f"by={_fmt(self.by)}")
        return f"Tilt({', '.join(parts)})"


class Hold(Construction):
    """Target weights straight from an expression (a model's output, a published allocation);
    ``normalise``: scale each date's weights to sum to ``invested``."""

    needs_scores = False

    def __init__(self, x, normalise: bool = False, invested: float = 1.0) -> None:
        self.x = Field(x) if isinstance(x, str) else lift(x)
        self.normalise, self.invested = bool(normalise), float(invested)

    def names(self):
        return self.x.names()

    def batch(self, B):
        v = np.where(B.universe, np.nan_to_num(B.field(self.x)), 0.0)
        return _normalise(v, self.invested) if self.normalise else v

    def __repr__(self) -> str:
        return f"Hold({self.x!r}{', normalise=True' if self.normalise else ''})"


class Pairs(Construction):
    """Long-short pairs: each long leg ``gross_per_pair`` of NAV, the short leg sized to neutralise
    ``exposure`` (``notional``, ``oasd``, ``oad``, ``dts``, ``dv01`` ...)."""

    needs_scores = False

    def __init__(self, pairs: Iterable[tuple[str, str]], exposure: str = "dts", gross_per_pair: float = 0.1) -> None:
        self.pairs, self.exposure, self.gross_per_pair = tuple((str(a), str(b)) for a, b in pairs), exposure, float(gross_per_pair)

    def names(self):
        return set() if self.exposure == "notional" else {self.exposure}

    def weights(self, cs):
        w = pd.Series(0.0, index=cs.ids)
        x = None if self.exposure == "notional" else pd.Series(_cs_values(cs, self.exposure), index=cs.ids)
        ratios = []
        for long_, short_ in self.pairs:
            if long_ not in w.index or short_ not in w.index:
                continue
            ratio = 1.0
            if x is not None:
                xl, xs = x[long_], x[short_]
                if not (np.isfinite(xl) and np.isfinite(xs)) or xs == 0:
                    continue
                ratio = float(xl / xs)
            w[long_] += self.gross_per_pair
            w[short_] -= self.gross_per_pair * ratio
            ratios.append(ratio)
        return w[w != 0.0], {"pairs_used": len(ratios)}

    def __repr__(self) -> str:
        return f"Pairs({len(self.pairs)} pairs, exposure={self.exposure!r}, gross_per_pair={self.gross_per_pair:g})"


# ====================================================================================== wrappers
class Capped(Construction):
    def __init__(self, method: Construction, by: str | None, cap: float) -> None:
        self.method, self.by, self.cap_ = method, by, float(cap)
        self.path_dependent = method.path_dependent
        self.needs_scores = method.needs_scores

    def names(self):
        return self.method.names() | ({self.by} if self.by else set())

    def constraints(self):
        return self.method.constraints()

    def _apply(self, w: np.ndarray, B: Batch) -> np.ndarray:
        R, m = w.shape
        g = B.groups(self.by) if self.by is not None else None
        codes = None if g is None else g.broadcast(R, m)
        out = w.copy()
        for r in range(R):
            nz = np.nonzero(w[r] > 0)[0]
            if not len(nz) or (w[r] < 0).any():
                continue
            total = float(w[r, nz].sum())
            s = pd.Series(w[r, nz], index=nz)
            grp = None if codes is None else pd.Series(codes[r, nz], index=nz)
            try:
                out[r, nz] = cap_weights(s, grp, self.cap_).to_numpy() * total
            except ValueError:
                pass                                   # infeasible cap for this date: left uncapped (the check records it)
        return out

    def batch(self, B):
        w = self.method.batch(B)
        return None if w is None else self._apply(w, B)

    def weights(self, cs):
        w, diag = self.method.weights(cs)
        B = Batch.from_cs(cs)
        v = w.reindex(cs.ids).fillna(0.0).to_numpy()[None, :]
        out = self._apply(v, B)[0]
        nz = np.abs(out) > 1e-15
        return pd.Series(out[nz], index=cs.ids[nz]), diag

    def __repr__(self) -> str:
        return f"{self.method!r}.cap({'name' if self.by is None else ('issuer' if self.by == 'issuer' else 'by=' + repr(self.by) + ', max_weight')}={self.cap_:g})"

    def _kw(self):
        return {"method": self.method, "by": self.by, "cap": self.cap_}


class Checked(Construction):
    """The method's weights unchanged; ``constraints`` only recorded as violations."""

    def __init__(self, method: Construction, constraints: tuple) -> None:
        self.method, self.constraints_ = method, tuple(constraints)
        self.path_dependent = method.path_dependent
        self.needs_scores = method.needs_scores

    def constraints(self):
        return self.constraints_ + tuple(self.method.constraints())

    def names(self):
        return self.method.names() | super().names()

    def batch(self, B):
        return self.method.batch(B)

    def weights(self, cs):
        return self.method.weights(cs)

    def replace(self, *constraints):
        return Checked(self.method, _replace(self.constraints_, constraints))

    def __repr__(self) -> str:
        return f"{self.method!r}.checked({', '.join(repr(c) for c in self.constraints_)})"

    def _kw(self):
        return {"method": self.method, "constraints": self.constraints_}


class Projected(Construction):
    """The feasible portfolio nearest (L1) to the method's weights: heuristics that respect a
    constraint set."""

    path_dependent = True

    def __init__(self, method: Construction, constraints: tuple, settings: OptimiserSettings | None = None) -> None:
        self.method, self.constraints_, self.settings = method, tuple(constraints), settings or OptimiserSettings()
        self.needs_scores = method.needs_scores

    def constraints(self):
        return self.constraints_

    def names(self):
        return self.method.names() | super().names()

    def weights(self, cs):
        target, diag = self.method.weights(cs)
        cons = list(self.constraints_)
        if not any(isinstance(c.measure, (Budget, CashWeight)) for c in cons):
            cons.append(Budget() == float(target.sum()))
        res = _solve(cs, _LegacyObjective(alpha=0.0, cost=0.0, active_l1=1.0, anchor=target), cons, long_only=bool((target >= 0).all()), settings=self.settings)
        return res.weights[res.weights.abs() > 1e-12], {**diag, **_opt_diag(res), "projection_l1": float((res.weights.reindex(cs.ids).fillna(0) - target.reindex(cs.ids).fillna(0)).abs().sum())}

    def replace(self, *constraints):
        return Projected(self.method, _replace(self.constraints_, constraints), self.settings)

    def __repr__(self) -> str:
        return f"{self.method!r}.subject_to({', '.join(repr(c) for c in self.constraints_)})"

    def _kw(self):
        return {"method": self.method, "constraints": self.constraints_, "settings": self.settings}


class Legacy(Construction):
    """A ``construct.ConstructionSpec`` (the original construction layer) as a kernel construction."""

    path_dependent = True

    def __init__(self, spec) -> None:
        self.spec = spec

    def names(self):
        from nautilus_fi.kernel.strategy import legacy_names

        return legacy_names(construction=self.spec)

    def weights(self, cs):
        out = self.spec.build(cs)
        return out.weights, {**out.diagnostics, "_violations": out.violations}

    def check(self, weights, cs):
        return []

    def overlays(self):
        return None

    def __repr__(self) -> str:
        return f"Legacy({self.spec.method.name}, {len(self.spec.constraints)} constraints)"


# ====================================================================================== optimiser: objective
class Term:
    """An objective term; combine with numbers, ``+`` and ``-`` into an ``Objective``."""

    penalty = False

    def __mul__(self, k) -> Objective:
        return Objective(((float(k), self),))

    __rmul__ = __mul__

    def __neg__(self) -> Objective:
        return Objective(((-1.0, self),))

    def __add__(self, other) -> Objective:
        return Objective(((1.0, self),)) + other

    def __radd__(self, other):
        return Objective(((1.0, self),)) if other == 0 else NotImplemented

    def __sub__(self, other) -> Objective:
        return Objective(((1.0, self),)) - other

    def names(self) -> set[str]:
        return set()


class Objective:
    def __init__(self, terms: tuple = ()) -> None:
        self.terms = tuple((float(k), t) for k, t in terms)

    def _as(self, other) -> Objective:
        if isinstance(other, Objective):
            return other
        if isinstance(other, Term):
            return Objective(((1.0, other),))
        raise TypeError(f"cannot add {type(other).__name__} to an objective")

    def __add__(self, other) -> Objective:
        return Objective(self.terms + self._as(other).terms)

    def __sub__(self, other) -> Objective:
        return Objective(self.terms + tuple((-k, t) for k, t in self._as(other).terms))

    def __mul__(self, k) -> Objective:
        return Objective(tuple((k * c, t) for c, t in self.terms))

    __rmul__ = __mul__

    def __neg__(self) -> Objective:
        return self * -1.0

    def names(self) -> set[str]:
        return set().union(*[t.names() for _, t in self.terms]) if self.terms else set()

    def __repr__(self) -> str:
        out = ""
        for i, (k, t) in enumerate(self.terms):
            sign = "-" if k < 0 else "+"
            mag = abs(k)
            body = repr(t) if mag == 1.0 else f"{mag:g} * {t!r}"
            out += (("- " if sign == "-" else "") + body) if i == 0 else f" {sign} {body}"
        return out or "0"

    def compile(self, cs: CrossSection) -> tuple[_LegacyObjective, np.ndarray | None, dict]:
        n = len(cs.ids)
        vec = np.zeros(n)
        linear = False
        cost = 0.0
        tracking = None
        l1 = 0.0
        risk = None
        cvar = None
        extras: dict = {}
        for k, t in self.terms:
            if t.penalty and k > 0:
                raise ValueError(f"{t!r} is a penalty: give it a negative coefficient (maximise alpha - penalty)")
            if isinstance(t, (Alpha, Linear)):
                vec += k * t.vector(cs)
                linear = True
            elif isinstance(t, Costs):
                cost += -k
                if t.model is not None:
                    extras["cost_model"] = t.model
            elif isinstance(t, TrackingError):
                tracking = TrackingTerm(t.fields, t.by, -k, t.scale)
            elif isinstance(t, ActiveShare):
                l1 += -k
            elif isinstance(t, Risk):
                rm = t.model(cs) if callable(t.model) and not isinstance(t.model, RiskModel) else t.model
                risk = dataclasses.replace(rm, aversion=-k, active=t.active)
            elif isinstance(t, CVaR):
                cvar = CVaRTerm(t.beta, -k, t.active)
                extras["scenarios"] = t.scenarios
            else:
                raise TypeError(f"unknown objective term {t!r}")
        return _LegacyObjective(alpha=1.0 if linear else 0.0, cost=cost, tracking=tracking, active_l1=l1, risk=risk, cvar=cvar), (vec if linear else None), extras


class Alpha(Term):
    """``scores . w``: the strategy's signal (or any expression), times ``scale``."""

    def __init__(self, signal=None, scale: float = 1.0) -> None:
        self.signal = None if signal is None else (Field(signal) if isinstance(signal, str) else lift(signal))
        self.scale = float(scale)

    def names(self):
        return set() if self.signal is None else self.signal.names()

    def vector(self, cs) -> np.ndarray:
        if self.signal is None:
            if cs.scores is None:
                raise ValueError("Alpha() needs the strategy's signal (or Alpha(expr))")
            v = cs.scores.to_numpy(dtype=np.float64)
        else:
            v = _cs_values(cs, self.signal)
        return np.nan_to_num(v) * self.scale

    def __repr__(self) -> str:
        args = ([] if self.signal is None else [repr(self.signal)]) + ([] if self.scale == 1.0 else [f"scale={self.scale:g}"])
        return f"Alpha({', '.join(args)})"


class Linear(Term):
    """Any per-instrument coefficient (an expression, or ``fn(cs) -> array``) times weight."""

    def __init__(self, x, name: str | None = None) -> None:
        self.x = x if callable(x) and not isinstance(x, Expr) else (Field(x) if isinstance(x, str) else lift(x))
        self.name = name or (repr(self.x) if isinstance(self.x, Expr) else getattr(x, "__name__", "fn"))

    def names(self):
        return self.x.names() if isinstance(self.x, Expr) else set()

    def vector(self, cs) -> np.ndarray:
        v = _cs_values(cs, self.x) if isinstance(self.x, Expr) else np.asarray(self.x(cs), dtype=np.float64)
        return np.nan_to_num(v)

    def __repr__(self) -> str:
        return f"Linear({self.name})"


class Costs(Term):
    """Transaction costs per unit traded: the execution cost model's linear part (or ``model``)."""

    penalty = True

    def __init__(self, model=None) -> None:
        self.model = model

    def __repr__(self) -> str:
        return "Costs()" if self.model is None else f"Costs({self.model!r})"


class TrackingError(Term):
    """L1 norm of active bucket exposures (``fields`` by ``by`` groups), each field scaled by the
    benchmark's exposure: the linear tracking proxy of index replication."""

    penalty = True

    def __init__(self, by=("sector",), fields=("oasd", "dts"), scale: dict | None = None) -> None:
        self.by = tuple([by] if isinstance(by, str) else by)
        self.fields = tuple([fields] if isinstance(fields, str) else fields)
        self.scale = scale

    def names(self):
        return set(self.by) | set(self.fields)

    def __repr__(self) -> str:
        return f"TrackingError(by={_fmt(list(self.by))}, fields={_fmt(list(self.fields))})"


class ActiveShare(Term):
    penalty = True

    def __repr__(self) -> str:
        return "ActiveShare()"


class Risk(Term):
    """Factor-model variance of the (active) portfolio: a QP. ``model``: a ``RiskModel`` or
    ``fn(cs) -> RiskModel`` (``PCARisk`` estimates one from history)."""

    penalty = True

    def __init__(self, model, active: bool = True) -> None:
        self.model, self.active = model, bool(active)

    def names(self):
        return getattr(self.model, "names", lambda: set())()

    def __repr__(self) -> str:
        return f"Risk({self.model!r}{'' if self.active else ', active=False'})"


class CVaR(Term):
    """Conditional value at risk of the (active) loss over scenarios (``HistoricalScenarios``)."""

    penalty = True

    def __init__(self, beta: float = 0.95, scenarios=None, active: bool = True) -> None:
        self.beta, self.scenarios, self.active = float(beta), scenarios or HistoricalScenarios(), bool(active)

    def names(self):
        return self.scenarios.names()

    def __repr__(self) -> str:
        return f"CVaR({self.beta:g})"


class HistoricalScenarios:
    """The last ``window`` rows of a return field as scenarios (known at the decision date)."""

    def __init__(self, window: int = 250, field: str = "xr") -> None:
        self.window, self.field = int(window), field

    def names(self):
        return {self.field}

    def __call__(self, cs) -> pd.DataFrame:
        ctx = getattr(cs, "ctx", None)
        if ctx is None:
            raise ValueError("HistoricalScenarios needs a kernel cross-section (run through the kernel engine)")
        X = ctx.field(self.field)
        lo = max(0, cs.t - self.window + 1)
        block = np.nan_to_num(np.asarray(X[lo:cs.t + 1], dtype=np.float64)[:, cs.cols])
        return pd.DataFrame(block, columns=cs.ids)

    def __repr__(self) -> str:
        return f"HistoricalScenarios({self.window}, {self.field!r})"


class PCARisk:
    """A statistical factor model from the last ``window`` rows of a return field: ``factors``
    principal components plus specific variance (per-period units)."""

    def __init__(self, window: int = 250, field: str = "xr", factors: int = 5) -> None:
        self.window, self.field, self.factors = int(window), field, int(factors)

    def names(self):
        return {self.field}

    def __call__(self, cs) -> RiskModel:
        X = HistoricalScenarios(self.window, self.field)(cs).to_numpy()
        X = X - X.mean(axis=0, keepdims=True)
        S = max(X.shape[0], 1)
        k = max(1, min(self.factors, min(X.shape) - 1))
        _U, _s, Vt = np.linalg.svd(X, full_matrices=False)
        L = Vt[:k].T
        f = X @ L
        F = (f.T @ f) / S
        resid = X - f @ L.T
        spec = np.maximum(resid.var(axis=0), 1e-12)
        cols = [f"pc{i + 1}" for i in range(k)]
        return RiskModel(pd.DataFrame(L, index=cs.ids, columns=cols), pd.DataFrame(F, index=cols, columns=cols), pd.Series(spec, index=cs.ids))

    def __repr__(self) -> str:
        return f"PCARisk({self.window}, {self.field!r}, factors={self.factors})"


# ====================================================================================== optimiser: constraints
def _bound(x):
    """A constraint bound: a number, or a value resolved at each rebalance (an expression such as a
    regime-scaled target, or ``fn(cs) -> number``)."""
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, Expr) or callable(x):
        return x
    return float(x)


def _resolve_bound(b, cs):
    if b is None or isinstance(b, float):
        return b
    if isinstance(b, Expr):
        v = np.asarray(_cs_values(cs, b), dtype=np.float64)
        fin = v[np.isfinite(v)]
        return float(fin[0]) if len(fin) else None                   # a date series broadcast: one value per date
    v = b(cs)
    return None if v is None else float(v)


class Measure:
    """The left-hand side of a constraint; compare it with numbers or ``.within(...)``. Bounds may
    be numbers, expressions or ``fn(cs)``, resolved at each rebalance (bounds that change by date)."""

    kind = ""

    def __le__(self, x) -> Constraint:
        return Constraint(self, None, _bound(x))

    def __ge__(self, x) -> Constraint:
        return Constraint(self, _bound(x), None)

    def __eq__(self, x) -> Constraint:                 # type: ignore[override]
        return Constraint(self, _bound(x), _bound(x))

    def __hash__(self) -> int:
        return hash(repr(self))

    def within(self, lower, upper=None) -> Constraint:
        """``within(b)``: the symmetric band [-b, b]; ``within(lo, hi)``: [lo, hi]."""
        if upper is None:
            if isinstance(lower, (int, float, np.integer, np.floating)):
                return Constraint(self, -abs(float(lower)), abs(float(lower)))
            return Constraint(self, _NegBound(lower), _bound(lower))
        return Constraint(self, _bound(lower), _bound(upper))

    def where(self, predicate) -> Subset:
        """This measure over the names where ``predicate`` holds: ``ActiveWeight("issuer").where(
        Field("sector") == "BANKING") <= 0`` keeps Banking issuers at or below their index weight."""
        return Subset(self, predicate)

    def spec_kw(self) -> dict:
        return {}

    def names(self) -> set[str]:
        kw = self.spec_kw()
        return {v for k, v in kw.items() if k in ("attribute", "group_by") and v}

    @property
    def slot(self) -> tuple:
        return (self.kind, tuple(sorted((k, str(v)) for k, v in self.spec_kw().items())))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Weight(Measure):
    kind = "weight"


class ActiveWeight(Measure):
    """Per name (no group), or per group of an attribute (``"issuer"``, ``"sector"``), relative to the benchmark."""

    def __init__(self, group: str | None = None, value=None) -> None:
        self.group, self.value = group, value
        self.kind = "active_weight" if group is None else "active_group"

    def spec_kw(self):
        return {} if self.group is None else {"attribute": self.group, "value": None if self.value is None else str(self.value)}

    def __repr__(self) -> str:
        if self.group is None:
            return "ActiveWeight()"
        return f"ActiveWeight({self.group!r}{'' if self.value is None else ', ' + repr(self.value)})"


class GroupWeight(Measure):
    kind = "group"

    def __init__(self, group: str, value=None) -> None:
        self.group, self.value = group, value

    def spec_kw(self):
        return {"attribute": self.group, "value": None if self.value is None else str(self.value)}

    def __repr__(self) -> str:
        return f"GroupWeight({self.group!r}{'' if self.value is None else ', ' + repr(self.value)})"


class Exposure(Measure):
    kind = "exposure"

    def __init__(self, field: str, by: str | None = None) -> None:
        self.field, self.by_ = field, by

    def by(self, group: str) -> Exposure:
        return type(self)(self.field, group) if type(self) is Exposure else ActiveExposure(self.field, group, self.relative)

    def spec_kw(self):
        return {"attribute": self.field, "group_by": self.by_}

    def names(self):
        return {self.field} | ({self.by_} if self.by_ else set())

    def __repr__(self) -> str:
        return f"Exposure({self.field!r}{'' if self.by_ is None else ', by=' + repr(self.by_)})"


class ActiveExposure(Exposure):
    kind = "active_exposure"

    def __init__(self, field: str, by: str | None = None, relative: bool = False) -> None:
        super().__init__(field, by)
        self.relative = bool(relative)

    def spec_kw(self):
        return {**super().spec_kw(), "relative": self.relative}

    def __repr__(self) -> str:
        return f"ActiveExposure({self.field!r}{'' if self.by_ is None else ', by=' + repr(self.by_)}{', relative=True' if self.relative else ''})"


class Turnover(Measure):
    kind = "turnover"

    def __init__(self, definition: TurnoverDefinition | str = TurnoverDefinition.HALF_SUM) -> None:
        self.definition = definition if isinstance(definition, TurnoverDefinition) else TurnoverDefinition(str(definition).upper())

    def spec_kw(self):
        return {"definition": self.definition.value}

    def __repr__(self) -> str:
        return "Turnover()" if self.definition == TurnoverDefinition.HALF_SUM else f"Turnover({self.definition.value!r})"


class Leverage(Measure):
    kind = "leverage"


class Budget(Measure):
    kind = "budget"


class CashWeight(Measure):
    kind = "cash"


class Names(Measure):
    kind = "cardinality"


class CVaRLimit(Measure):
    kind = "cvar"

    def __init__(self, beta: float = 0.95, scenarios=None, active: bool = True) -> None:
        self.beta, self.scenarios, self.active = float(beta), scenarios or HistoricalScenarios(), bool(active)

    def spec_kw(self):
        return {"beta": self.beta, "active": self.active}

    def names(self):
        return self.scenarios.names()

    def __repr__(self) -> str:
        return f"CVaRLimit({self.beta:g})"


class LinearMeasure(Measure):
    """``coef . w`` for any per-instrument coefficient (an expression or ``fn(cs) -> array``):
    custom constraints such as a carry floor or a liquidity budget."""

    kind = "linear"

    def __init__(self, x, name: str | None = None) -> None:
        self.x = x if callable(x) and not isinstance(x, Expr) else (Field(x) if isinstance(x, str) else lift(x))
        self.name = name or (repr(self.x) if isinstance(self.x, Expr) else getattr(x, "__name__", "fn"))

    def coef(self, cs) -> np.ndarray:
        v = _cs_values(cs, self.x) if isinstance(self.x, Expr) else np.asarray(self.x(cs), dtype=np.float64)
        return np.nan_to_num(v)

    def names(self):
        return self.x.names() if isinstance(self.x, Expr) else set()

    @property
    def slot(self):
        return ("linear", self.name)

    def __repr__(self) -> str:
        return f"LinearMeasure({self.name})"


class _MinHold(Measure):
    kind = "min_holding"

    def __repr__(self) -> str:
        return "MinHolding"


class _NegBound:
    """``-b`` for a bound resolved per rebalance (``within(expr)``)."""

    def __init__(self, b) -> None:
        self.b = b

    def __call__(self, cs):
        v = _resolve_bound(self.b, cs)
        return None if v is None else -abs(v)

    def __repr__(self) -> str:
        return f"-{self.b!r}"


class Constraint:
    def __init__(self, measure: Measure, lo, hi, hard: bool = True, penalty: float = 100.0, strict: bool = False) -> None:
        self.measure, self.lo, self.hi, self.hard, self.penalty, self.is_strict = measure, lo, hi, bool(hard), float(penalty), bool(strict)

    def soft(self, penalty: float = 100.0) -> Constraint:
        return Constraint(self.measure, self.lo, self.hi, False, penalty)

    def strict(self) -> Constraint:
        """A limit the relaxation ladder never moves (a mandate rule)."""
        c = Constraint(self.measure, self.lo, self.hi, self.hard, self.penalty, True)
        c.min_benchmark_weight = getattr(self, "min_benchmark_weight", None)
        return c

    def skip_below(self, benchmark_weight: float) -> Constraint:
        """Group constraints: skip groups the benchmark holds less than ``benchmark_weight`` of."""
        c = Constraint(self.measure, self.lo, self.hi, self.hard, self.penalty, self.is_strict)
        c.min_benchmark_weight = float(benchmark_weight)
        return c

    def at(self, cs) -> Constraint:
        """This constraint with its bounds resolved for the cross-section's date."""
        lo, hi = _resolve_bound(self.lo, cs), _resolve_bound(self.hi, cs)
        if lo is self.lo and hi is self.hi:
            return self
        c = Constraint(self.measure, lo, hi, self.hard, self.penalty, self.is_strict)
        c.min_benchmark_weight = getattr(self, "min_benchmark_weight", None)
        return c

    def linear_rows(self, cs) -> list:
        if hasattr(self.measure, "rows"):
            return self.measure.rows(cs, self.lo, self.hi, self.hard, self.penalty)
        r = self.linear_row(cs)
        return [] if r is None else [r]

    def bounds(self, cs):
        return self.measure.bounds(cs) if hasattr(self.measure, "bounds") else None

    @property
    def slot(self) -> tuple:
        return self.measure.slot

    def names(self) -> set[str]:
        return self.measure.names()

    def to_spec(self) -> ConstraintSpec | None:
        m = self.measure
        if isinstance(m, (LinearMeasure, TEV)) or hasattr(m, "rows") or hasattr(m, "bounds"):
            return None
        kw = {k: v for k, v in m.spec_kw().items() if k not in ("value",) or v is not None}
        if "value" in kw and kw["value"] is None:
            kw.pop("value")
        return ConstraintSpec(m.kind, self.lo, self.hi, hard=self.hard, penalty=self.penalty, strict=self.is_strict, min_benchmark_weight=getattr(self, "min_benchmark_weight", None), **kw)

    def linear_row(self, cs) -> LinearRow | None:
        if not isinstance(self.measure, LinearMeasure):
            return None
        return LinearRow(self.measure.coef(cs), self.lo, self.hi, repr(self.measure), self.hard, self.penalty)

    def __repr__(self) -> str:
        m = self.measure
        f = (lambda v: f"{v:g}" if isinstance(v, float) else repr(v))
        if isinstance(m, _MinHold):
            body = f"MinHolding({'' if self.lo is None else f(self.lo)})"
        elif isinstance(m, _MinTrade):
            body = f"MinTrade({f(self.lo)})"
        elif hasattr(m, "bounds") or isinstance(m, _RelCap):
            body = repr(m)
        elif self.lo is not None and self.hi is not None and isinstance(self.lo, float) and isinstance(self.hi, float) and self.lo == self.hi:
            body = f"{m!r} == {f(self.lo)}"
        elif self.lo is not None and self.hi is not None and isinstance(self.lo, float) and isinstance(self.hi, float) and self.lo == -self.hi:
            body = f"{m!r}.within({f(self.hi)})"
        elif self.lo is not None and self.hi is not None:
            body = f"{m!r}.within({f(self.lo)}, {f(self.hi)})"
        elif self.hi is not None:
            body = f"{m!r} <= {f(self.hi)}"
        else:
            body = f"{m!r} >= {f(self.lo)}"
        return body + ("" if self.hard else f".soft({self.penalty:g})") + (".strict()" if self.is_strict else "")


def MinHolding(min_weight: float | None = None) -> Constraint:
    """Each holding zero or at least ``min_weight`` of NAV (None: the minimum piece at the price)."""
    return Constraint(_MinHold(), None if min_weight is None else float(min_weight), None)


class _MinTrade(Measure):
    kind = "min_trade"

    def __repr__(self) -> str:
        return "MinTrade"


def MinTrade(min_weight: float) -> Constraint:
    """Each trade zero or at least ``min_weight`` of NAV (semi-continuous trades, a MILP); closing a
    position smaller than the minimum is allowed."""
    return Constraint(_MinTrade(), float(min_weight), None)


def _replace(old: tuple, new: tuple) -> tuple:
    slots = {c.slot for c in new}
    return tuple(c for c in old if c.slot not in slots) + tuple(new)


def check_all(constraints, weights: pd.Series, cs: CrossSection) -> list[dict]:
    if not constraints:
        return []
    initial = float(cs.current.abs().sum()) < 1e-9
    specs = []
    out: list[dict] = []
    wv = weights.reindex(cs.ids).fillna(0.0).to_numpy()
    for c in constraints:
        c = c.at(cs)
        if isinstance(c.measure, TEV):
            sig = c.measure.sigma(cs, wv)
            if c.hi is not None and sig > c.hi * (1 + 1e-3) + 1e-9:
                out.append({"constraint": repr(c.measure), "group": None, "value": sig, "bound": c.hi, "side": "max", "severity": "HARD" if c.hard else "SOFT"})
            continue
        b = c.bounds(cs)
        if b is not None:
            lb, ub = b
            for i in np.nonzero(((wv < lb - 1e-6) if lb is not None else False) | ((wv > ub + 1e-6) if ub is not None else False))[0]:
                out.append({"constraint": repr(c.measure), "group": str(cs.ids[i]), "value": float(wv[i]), "bound": float(lb[i] if lb is not None and wv[i] < lb[i] else ub[i]), "side": "min" if lb is not None and wv[i] < lb[i] else "max", "severity": "HARD"})
            continue
        if hasattr(c.measure, "rows"):
            for row in c.linear_rows(cs):
                v = float(row.coef @ wv)
                if row.hi is not None and v > row.hi + 1e-6:
                    out.append({"constraint": row.name, "group": None, "value": v, "bound": row.hi, "side": "max", "severity": "HARD" if row.hard else "SOFT"})
                if row.lo is not None and v < row.lo - 1e-6:
                    out.append({"constraint": row.name, "group": None, "value": v, "bound": row.lo, "side": "min", "severity": "HARD" if row.hard else "SOFT"})
            continue
        spec = c.to_spec()
        if spec is not None:
            if (spec.kind == "turnover" and initial) or (spec.kind == "cvar" and cs.scenarios is None):
                continue
            specs.append(spec)
        else:
            row = c.linear_row(cs)
            v = float(row.coef @ weights.reindex(cs.ids).fillna(0.0).to_numpy())
            if row.hi is not None and v > row.hi + 1e-6:
                out.append({"constraint": row.name, "group": None, "value": v, "bound": row.hi, "side": "max", "severity": "HARD" if row.hard else "SOFT"})
            if row.lo is not None and v < row.lo - 1e-6:
                out.append({"constraint": row.name, "group": None, "value": v, "bound": row.lo, "side": "min", "severity": "HARD" if row.hard else "SOFT"})
    return check_constraints(weights, cs, specs) + out


def _opt_diag(res) -> dict:
    out = {"optimiser": res.summary(), "binding": res.binding[:50], "relaxations": res.relaxations, "ok": res.ok, "message": res.message}
    if getattr(res, "overlays", None) is not None:
        out["overlays"] = res.overlays                          # solved with the book (Hedge(joint=True))
    return out


def _solve(cs, legacy_objective, constraints, long_only, settings, extras: dict | None = None, extra_rows=()):
    constraints = [c.at(cs) for c in constraints if not isinstance(c.measure, TEV)]
    specs = [s for s in (c.to_spec() for c in constraints) if s is not None]
    rows = [r for c in constraints for r in c.linear_rows(cs)] + list(extra_rows)
    lb = ub = None
    for c in constraints:
        b = c.bounds(cs)
        if b is not None:
            lo_, hi_ = b
            if lo_ is not None:
                lb = lo_ if lb is None else np.maximum(lb, lo_)
            if hi_ is not None:
                ub = hi_ if ub is None else np.minimum(ub, hi_)
    if lb is not None or ub is not None:
        cs = dataclasses.replace(cs, lb_extra=lb, ub_extra=ub)
    need_scen = any(s.kind == "cvar" for s in specs) or (extras or {}).get("scenarios") is not None
    if need_scen and cs.scenarios is None:
        prov = (extras or {}).get("scenarios") or next((c.measure.scenarios for c in constraints if isinstance(c.measure, CVaRLimit)), None)
        if prov is not None:
            cs = dataclasses.replace(cs, scenarios=prov(cs))
    return optimise(cs, legacy_objective, specs, long_only, settings, rows)


# ====================================================================================== optimiser: risk limits, bounds and rows
def _risk_model(model, cs) -> RiskModel:
    return model(cs) if callable(model) and not isinstance(model, RiskModel) else model


def _risk_matvec(rm: RiskModel, ids, a: np.ndarray) -> np.ndarray:
    L = rm.loadings.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    F = rm.factor_cov.to_numpy(dtype=float)
    spec = rm.specific_var.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    return L @ (F @ (L.T @ a)) + spec * a


class TEV(Measure):
    """Ex-ante tracking error (``active=True``: of weights minus the benchmark's) or volatility of a
    factor risk model (a ``RiskModel`` or ``fn(cs) -> RiskModel``, e.g. ``PCARisk``), annualised by
    ``periods_per_year``. ``TEV(model) <= 0.02`` is a hard limit, solved by outer linearisation
    (cutting planes on the convex TE function) around the LP / QP."""

    kind = "tev"

    def __init__(self, model, active: bool = True, periods_per_year: float = 252.0) -> None:
        self.model, self.active, self.periods_per_year = model, bool(active), float(periods_per_year)

    def names(self):
        return getattr(self.model, "names", lambda: set())()

    def _a(self, cs, w: np.ndarray) -> np.ndarray:
        return w - (cs.benchmark.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float) if self.active else 0.0)

    def sigma(self, cs, w: np.ndarray, rm: RiskModel | None = None) -> float:
        rm = rm or _risk_model(self.model, cs)
        a = self._a(cs, w)
        return float(np.sqrt(max(a @ _risk_matvec(rm, cs.ids, a), 0.0) * self.periods_per_year))

    def cut(self, cs, w: np.ndarray, limit: float, rm: RiskModel, name: str, hard: bool, penalty: float) -> LinearRow | None:
        a = self._a(cs, w)
        Sa = _risk_matvec(rm, cs.ids, a)
        sig = float(np.sqrt(max(a @ Sa, 0.0) * self.periods_per_year))
        if sig <= limit * (1 + 1e-3) or sig <= 0:
            return None
        g = self.periods_per_year * Sa / sig                              # gradient of the annualised TE in weights
        base = float(g @ (cs.benchmark.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float))) if self.active else 0.0
        return LinearRow(g, None, limit + base, name, hard, penalty)

    def __repr__(self) -> str:
        return f"TEV({self.model!r}{'' if self.active else ', active=False'})"


def TargetTE(model, target: float, periods_per_year: float = 252.0) -> Constraint:
    """A tracking-error budget the alpha objective spends: ``TEV(model) <= target``."""
    return TEV(model, True, periods_per_year) <= float(target)


def _values(v, cs, default=np.nan) -> np.ndarray:
    """Per-name values on the cross-section: a number, a dict / Series by id, an expression,
    ``fn(cs)``, or ``"current"`` (the current weights)."""
    n = len(cs.ids)
    if v is None:
        return np.full(n, default)
    if isinstance(v, str) and v == "current":
        return cs.current.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.full(n, float(v))
    if isinstance(v, dict):
        v = pd.Series(v, dtype=float)
    if isinstance(v, pd.Series):
        return v.reindex(cs.ids.astype(str)).to_numpy(dtype=float)
    if isinstance(v, Expr):
        return np.asarray(_cs_values(cs, v), dtype=float)
    return np.asarray(v(cs), dtype=float)


def _mask(names, cs) -> np.ndarray:
    if names is None:
        return np.ones(len(cs.ids), dtype=bool)
    if isinstance(names, Expr):
        return np.nan_to_num(np.asarray(_cs_values(cs, names), dtype=float)) > 0
    return cs.ids.astype(str).isin([str(x) for x in names])


class _Bounds(Measure):
    kind = "bounds"

    def __init__(self, lo=None, hi=None, names=None, trade=None, label: str = "Bounds") -> None:
        self.lo_, self.hi_, self.names_, self.trade, self.label = lo, hi, names, trade, label

    def names(self):
        out = set()
        for v in (self.lo_, self.hi_, self.names_, self.trade):
            if isinstance(v, Expr):
                out |= v.names()
        return out

    def bounds(self, cs):
        m = _mask(self.names_, cs)
        lb = ub = None
        if self.lo_ is not None:
            v = _values(self.lo_, cs)
            lb = np.where(m & np.isfinite(v), v, -np.inf)
        if self.hi_ is not None:
            v = _values(self.hi_, cs)
            ub = np.where(m & np.isfinite(v), v, np.inf)
        if self.trade is not None:
            cap = np.abs(_values(self.trade, cs))
            w0 = cs.current.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
            lo2, hi2 = np.where(m & np.isfinite(cap), w0 - cap, -np.inf), np.where(m & np.isfinite(cap), w0 + cap, np.inf)
            lb = lo2 if lb is None else np.maximum(lb, lo2)
            ub = hi2 if ub is None else np.minimum(ub, hi2)
        return lb, ub

    def __repr__(self) -> str:
        return self.label


def Bounds(lo=None, hi=None, names=None) -> Constraint:
    """Per-name weight bounds: numbers, ``{id: w}`` / Series, expressions (factor units:
    ``hi=0.5 / Field("dts")``) or ``fn(cs)``; ``names``: ids or a predicate the bounds apply to."""
    return Constraint(_Bounds(lo, hi, names, None, f"Bounds(lo={_fmt(lo)}, hi={_fmt(hi)})"), None, None)


def Pin(weights) -> Constraint:
    """Hold these names at exactly these weights (``{id: w}``)."""
    return Constraint(_Bounds(weights, weights, list(pd.Series(weights).index), None, f"Pin({len(pd.Series(weights))} names)"), None, None)


def NoSell(names) -> Constraint:
    """These names (ids or a predicate) may not be reduced."""
    return Constraint(_Bounds("current", None, names, None, "NoSell"), None, None)


def NoBuy(names) -> Constraint:
    """These names may not be increased (hold, never buy: grandfathered names)."""
    return Constraint(_Bounds(None, "current", names, None, "NoBuy"), None, None)


def TradeLimit(max_trade, names=None) -> Constraint:
    """Each name trades at most ``max_trade`` of NAV (a number, or capacity such as
    ``Field("adv") * 0.1 / nav``); a band that needs a larger trade binds at the capacity."""
    return Constraint(_Bounds(None, None, names, max_trade, f"TradeLimit({_fmt(max_trade)})"), None, None)


def _group_values(cs, group) -> np.ndarray:
    if group in cs.data.columns:
        return cs.data[group].astype(str).to_numpy()
    ctx = getattr(cs, "ctx", None)
    if ctx is not None:
        cat = ctx.groups(group)
        codes = cat.broadcast(ctx.T, ctx.N)[cs.t][cs.cols]
        labels = np.array([str(x) for x in cat.categories] + ["<none>"])
        return labels[np.where(codes >= 0, codes, len(labels) - 1)]
    raise KeyError(f"group {group!r} is not on the cross-section")


class Subset(Measure):
    """A measure restricted to the names where ``predicate`` holds (``Measure.where``)."""

    kind = "subset"

    def __init__(self, inner: Measure, predicate) -> None:
        self.inner, self.predicate = inner, (Field(predicate) if isinstance(predicate, str) else lift(predicate))

    def names(self):
        return self.inner.names() | self.predicate.names()

    @property
    def slot(self):
        return ("subset", repr(self.inner), repr(self.predicate))

    def rows(self, cs, lo, hi, hard, penalty) -> list:
        mask = np.nan_to_num(np.asarray(_cs_values(cs, self.predicate), dtype=float)) > 0
        wb = cs.benchmark.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
        inner = self.inner
        active = isinstance(inner, (ActiveWeight, ActiveExposure))
        out = []
        if isinstance(inner, (GroupWeight, ActiveWeight)) and getattr(inner, "group", None) is not None:
            g = _group_values(cs, inner.group)
            for grp in np.unique(g[mask]):
                if inner.value is not None and str(grp) != str(inner.value):
                    continue
                sel = mask & (g == grp)
                base = float(wb[sel].sum()) if active else 0.0
                out.append(LinearRow(sel.astype(float), None if lo is None else lo + base, None if hi is None else hi + base, f"{inner!r}[{grp}] where {self.predicate!r}", hard, penalty))
        elif isinstance(inner, Exposure):
            x = np.nan_to_num(np.asarray(_cs_values(cs, Field(inner.field)), dtype=float)) * mask
            base = float(wb @ x) if active else 0.0
            out.append(LinearRow(x, None if lo is None else lo + base, None if hi is None else hi + base, f"{inner!r} where {self.predicate!r}", hard, penalty))
        elif isinstance(inner, (Weight, ActiveWeight)):
            coef = mask.astype(float)
            base = float(wb[mask].sum()) if active else 0.0
            out.append(LinearRow(coef, None if lo is None else lo + base, None if hi is None else hi + base, f"total {inner!r} where {self.predicate!r}", hard, penalty))
        else:
            raise TypeError(f"{inner!r}.where(...) supports group weights, weights and exposures")
        return out

    def __repr__(self) -> str:
        return f"{self.inner!r}.where({self.predicate!r})"


class _RelCap(Measure):
    kind = "relcap"

    def __init__(self, group: str, absolute: float, relative: float, floor: float | None = None) -> None:
        self.group, self.absolute, self.relative, self.floor = group, float(absolute), float(relative), floor

    def names(self):
        return {self.group}

    @property
    def slot(self):
        return ("relcap", self.group)

    def rows(self, cs, lo, hi, hard, penalty) -> list:
        g = _group_values(cs, self.group)
        wb = cs.benchmark.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
        out = []
        for grp in np.unique(g):
            sel = g == grp
            cap = min(self.absolute, self.relative * float(wb[sel].sum()))
            if self.floor is not None:
                cap = max(cap, float(self.floor))
            out.append(LinearRow(sel.astype(float), None, cap, f"{self!r}[{grp}]", hard, penalty))
        return out

    def __repr__(self) -> str:
        return f"GroupWeight({self.group!r}).cap(absolute={self.absolute:g}, relative={self.relative:g}{'' if self.floor is None else f', floor={self.floor:g}'})"


class CurrencyExposure(Measure):
    """Exposure to a currency, counting cash in the base currency: ``sum(w where currency == c)``
    plus the cash weight when ``c`` is the base currency."""

    kind = "currency"

    def __init__(self, currency: str, base: str = "USD", field: str = "currency") -> None:
        self.currency, self.base, self.field = str(currency), str(base), field

    def names(self):
        return {self.field}

    @property
    def slot(self):
        return ("currency", self.currency)

    def rows(self, cs, lo, hi, hard, penalty) -> list:
        ccy = _group_values(cs, self.field)
        if self.currency == self.base:                                   # 1 - sum(w) + sum(w in base) = 1 - sum(w not in base)
            coef = -(ccy != self.currency).astype(float)
            return [LinearRow(coef, None if lo is None else lo - 1.0, None if hi is None else hi - 1.0, repr(self), hard, penalty)]
        return [LinearRow((ccy == self.currency).astype(float), lo, hi, repr(self), hard, penalty)]

    def __repr__(self) -> str:
        return f"CurrencyExposure({self.currency!r}{'' if self.base == 'USD' else f', base={self.base!r}'})"


class AnnualTurnoverBudget:
    """A per-rebalance turnover bound from an annual target: a ``month_end_share`` of the monthly
    budget at month ends and the rest spread over ``events_per_month`` interim events (Erin Beta+).
    Use as a bound: ``Turnover() <= AnnualTurnoverBudget(1.2)``."""

    def __init__(self, annual: float, month_end_share: float = 0.4, events_per_month: int = 3) -> None:
        self.annual, self.share, self.events = float(annual), float(month_end_share), int(events_per_month)

    def __call__(self, cs) -> float:
        monthly = self.annual / 12.0
        dates = cs.ctx.panel.dates
        t = cs.t
        month_end = t + 1 >= len(dates) or dates[t + 1].month != dates[t].month
        return monthly * self.share if month_end else monthly * (1.0 - self.share) / max(self.events, 1)

    def __repr__(self) -> str:
        return f"AnnualTurnoverBudget({self.annual:g}, month_end_share={self.share:g})"


class IndexTurnover:
    """A turnover bound that follows the index: half the sum of the benchmark's weight changes over
    the last ``window`` rows, plus a discretionary ``allowance``: ``Turnover() <= IndexTurnover(ix, allowance=0.02)``."""

    def __init__(self, benchmark, window: int = 21, allowance: float = 0.0) -> None:
        self.benchmark, self.window, self.allowance = benchmark, int(window), float(allowance)

    def __call__(self, cs) -> float:
        W = self.benchmark.weights(cs.ctx)
        t = cs.t
        k = max(0, t - self.window)
        return float(0.5 * np.abs(W[t] - W[k]).sum() + self.allowance)

    def __repr__(self) -> str:
        return f"IndexTurnover({self.benchmark!r}, allowance={self.allowance:g})"


def _cap(self, absolute: float, relative: float, floor: float | None = None) -> Constraint:
    """Each group at most ``min(absolute, relative x its index weight)`` (``floor``: at least this
    cap, for names outside the index)."""
    return Constraint(_RelCap(self.group, absolute, relative, floor), None, None)


GroupWeight.cap = _cap



class Optimise(Construction):
    """An LP / MILP / QP over trades, solved by HiGHS: maximise the objective subject to the
    constraints (fully invested by default when long-only)."""

    path_dependent = True
    handles_exits = True                       # applies the strategy's exit policy to held names outside the universe

    def __init__(self, objective=None, *constraints: Constraint, long_only: bool = True, settings: OptimiserSettings | None = None, constraints_: Iterable[Constraint] | None = None, tie_break=None, tie_tolerance: float = 1e-6, tev_iterations: int = 30) -> None:
        """``tie_break``: a second objective (``-ActiveShare()``, ``-Risk(model)``, ``-Linear(issuer_dts)``)
        minimised at no loss in the first stage's objective (within ``tie_tolerance``, relative).
        ``TEV(model) <= x`` limits are solved by cutting planes, at most ``tev_iterations`` re-solves."""
        if isinstance(objective, Constraint):
            constraints = (objective, *constraints)
            objective = None
        if isinstance(tie_break, Term):
            tie_break = Objective(((1.0, tie_break),))
        self.tie_break, self.tie_tolerance, self.tev_iterations = tie_break, float(tie_tolerance), int(tev_iterations)
        if isinstance(objective, Term):
            objective = Objective(((1.0, objective),))
        self.objective = objective if objective is not None else Objective(((1.0, Alpha()), (-1.0, Costs())))
        self.constraints_ = tuple(constraints) + tuple(constraints_ or ())
        for c in self.constraints_:
            if not isinstance(c, Constraint):
                raise TypeError(f"Optimise constraints must be Constraint objects (e.g. Turnover() <= 0.1), not {c!r}")
        self.long_only = bool(long_only)
        self.settings = settings or OptimiserSettings()
        self.needs_scores = any(isinstance(t, Alpha) and t.signal is None for _, t in self.objective.terms)

    def constraints(self):
        return self.constraints_

    def names(self):
        return self.objective.names() | super().names()

    def weights(self, cs):
        legacy, vec, extras = self.objective.compile(cs)
        if vec is not None:
            cs = dataclasses.replace(cs, scores=pd.Series(vec, index=cs.ids))
        model = extras.get("cost_model")
        if model is not None and getattr(cs, "ctx", None) is not None:
            unit = model.linear(cs.ctx, cs.t, cs.nav)[cs.cols]
            cs = dataclasses.replace(cs, cost_bp=pd.Series(unit * 1e4, index=cs.ids))
        res, cuts, tev = self._solve_with_limits(cs, legacy, extras, ())
        diag = _opt_diag(res)
        if tev:
            diag["tev"] = tev
        if self.tie_break is not None and res.ok and vec is not None:
            w1 = res.weights.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
            score = float(vec @ w1)
            keep = LinearRow(vec, score - self.tie_tolerance * max(abs(score), 1e-12), None, "tie-break: first-stage objective kept", True, 0.0)
            legacy2, vec2, extras2 = self.tie_break.compile(cs)
            cs2 = dataclasses.replace(cs, scores=pd.Series(vec2 if vec2 is not None else np.zeros(len(cs.ids)), index=cs.ids))
            res2, _, _ = self._solve_with_limits(cs2, legacy2, extras2, (*cuts, keep))
            if res2.ok:
                diag["tie_break"] = {"first_stage_score": score, "second_stage_score": float(vec @ res2.weights.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)), "objective": repr(self.tie_break)}
                res = res2
        return res.weights[res.weights.abs() > 1e-12], diag

    def _solve_with_limits(self, cs, legacy, extras, extra_rows):
        """Solve, then add a cutting plane for each TE / risk limit the solution breaches, until
        every limit holds (or ``tev_iterations`` re-solves)."""
        tevs = [c.at(cs) for c in self.constraints_ if isinstance(c.measure, TEV)]
        cuts = list(extra_rows)
        res = _solve(cs, legacy, self.constraints_, self.long_only, self.settings, extras, cuts)
        if not tevs:
            return res, cuts, None
        models = [_risk_model(c.measure.model, cs) for c in tevs]
        integer = any(c.measure.kind in ("cardinality", "min_holding", "min_trade") for c in self.constraints_)
        if len(tevs) == 1 and tevs[0].hi is not None and legacy.risk is None and not integer:
            return self._bisect_tev(cs, legacy, extras, cuts, tevs[0], models[0], res)
        it = 0
        while res.ok and it < self.tev_iterations:
            w = res.weights.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
            new = [c.measure.cut(cs, w, c.hi, rm, f"{c.measure!r} cut {it}", c.hard, c.penalty) for c, rm in zip(tevs, models, strict=True) if c.hi is not None]
            new = [r for r in new if r is not None]
            if not new:
                break
            cuts += new
            res = _solve(cs, legacy, self.constraints_, self.long_only, self.settings, extras, cuts)
            it += 1
        w = res.weights.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
        tev = {repr(c.measure): {"limit": c.hi, "value": c.measure.sigma(cs, w, rm), "cuts": it} for c, rm in zip(tevs, models, strict=True)}
        return res, cuts, tev

    def _bisect_tev(self, cs, legacy, extras, cuts, c, rm, res0):
        """One TE limit: the risk-aversion multiplier of ``a' S a`` (a QP) bisected to the smallest
        value whose solution meets the limit; on the Lagrangian path of the constrained problem."""
        m = c.measure
        sig = lambda r: m.sigma(cs, r.weights.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float), rm)
        limit = float(c.hi)
        if not res0.ok or sig(res0) <= limit * (1 + 1e-3):
            return res0, cuts, {repr(m): {"limit": limit, "value": sig(res0) if res0.ok else float("nan"), "aversion": 0.0}}
        solve = lambda lam: _solve(cs, dataclasses.replace(legacy, risk=dataclasses.replace(rm, aversion=lam, active=m.active)), self.constraints_, self.long_only, self.settings, extras, cuts)
        lo, hi = 0.0, 1.0
        best = solve(hi)
        while best.ok and sig(best) > limit and hi < 1e18:
            lo, hi = hi, hi * 10.0
            best = solve(hi)
        n = 0
        while best.ok and n < 40 and (hi - lo) > 1e-4 * hi:
            mid = np.sqrt(lo * hi) if lo > 0 else hi / 10.0
            r = solve(mid)
            if r.ok and sig(r) <= limit:
                hi, best = mid, r
            else:
                lo = mid
            n += 1
        return best, cuts, {repr(m): {"limit": limit, "value": sig(best) if best.ok else float("nan"), "aversion": hi, "solves": n}}

    # ------------------------------------------------------------------ variants
    def _same(self, objective, constraints, **kw) -> Optimise:
        base = {"long_only": self.long_only, "settings": self.settings, "tie_break": self.tie_break, "tie_tolerance": self.tie_tolerance, "tev_iterations": self.tev_iterations}
        return Optimise(objective, *constraints, **{**base, **kw})

    def replace(self, *constraints: Constraint) -> Optimise:
        """Swap constraints in the same slot (``Turnover() <= 0.05`` replaces the turnover limit)."""
        return self._same(self.objective, _replace(self.constraints_, constraints))

    def add(self, *constraints: Constraint) -> Optimise:
        return self._same(self.objective, (*self.constraints_, *constraints))

    def without(self, *measures) -> Optimise:
        """Drop constraints by measure class (``Turnover``) or slot of an example constraint."""
        def drop(c):
            for m in measures:
                if isinstance(m, type) and isinstance(c.measure, m):
                    return True
                if isinstance(m, Constraint) and m.slot == c.slot:
                    return True
            return False
        return self._same(self.objective, [c for c in self.constraints_ if not drop(c)])

    def with_objective(self, objective) -> Optimise:
        return self._same(objective, self.constraints_)

    def with_settings(self, **changes) -> Optimise:
        return self._same(self.objective, self.constraints_, settings=dataclasses.replace(self.settings, **changes))

    def with_(self, **changes) -> Optimise:
        kw = {"objective": self.objective, "long_only": self.long_only, "settings": self.settings, "tie_break": self.tie_break, "tie_tolerance": self.tie_tolerance, "tev_iterations": self.tev_iterations, **changes}
        cons = changes.get("constraints", self.constraints_)
        kw.pop("constraints", None)
        return Optimise(kw.pop("objective"), *cons, **kw)

    def __repr__(self) -> str:
        cons = ", ".join(repr(c) for c in self.constraints_)
        extra = ("" if self.long_only else ", long_only=False") + ("" if self.tie_break is None else f", tie_break={self.tie_break!r}")
        return f"Optimise({self.objective!r}{', ' + cons if cons else ''}{extra})"


class Stratified(Construction):
    """The rule-based index constructor: cells from ``cells`` (``(field, edges)`` bucketed, or a
    category name: maturity x rating, sector ...), each cell given its benchmark weight and filled
    with its ``per_cell`` best names by the signal (the pecking order), weighted ``weighting``
    within the cell. Combine with ``.cap(issuer=...)`` for issuer caps and ``.subject_to(...)`` for
    duration / KRD matching."""

    needs_scores = True

    def __init__(self, cells=(("years_to_maturity", (3.0, 5.0, 7.0, 10.0)), ("rating_score", (4.5, 7.5, 10.5))), per_cell: int = 3, weighting=None) -> None:
        self.cells, self.per_cell, self.weighting = tuple(cells), int(per_cell), as_weighting(weighting)

    def names(self):
        out = set()
        for c in self.cells:
            out.add(c[0] if isinstance(c, (tuple, list)) else c)
        return out | self.weighting.names()

    def _codes(self, cs) -> np.ndarray:
        keys = []
        for c in self.cells:
            if isinstance(c, (tuple, list)):
                v = np.asarray(_cs_values(cs, Field(c[0])), dtype=float)
                keys.append(np.where(np.isfinite(v), np.digitize(v, c[1]), -1).astype(str))
            else:
                keys.append(_group_values(cs, c))
        return np.array(["|".join(t) for t in zip(*keys, strict=True)]) if keys else np.full(len(cs.ids), "ALL")

    def weights(self, cs):
        codes = self._codes(cs)
        wb = cs.benchmark.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
        sc = cs.scores.reindex(cs.ids).to_numpy(dtype=float) if cs.scores is not None else np.zeros(len(cs.ids))
        el = cs.eligible.reindex(cs.ids).fillna(False).to_numpy(dtype=bool)
        out = np.zeros(len(cs.ids))
        filled, empty = 0, []
        for cell in np.unique(codes):
            m = codes == cell
            target = float(wb[m].sum())
            if target <= 0:
                continue
            cand = np.nonzero(m & el & np.isfinite(sc))[0]
            if not len(cand):
                empty.append(cell)
                continue
            pick = cand[np.argsort(-sc[cand], kind="stable")[: self.per_cell]]
            sel = np.zeros(len(cs.ids), dtype=bool)
            sel[pick] = True
            ww = self.weighting.apply(sel[None, :], Batch.from_cs(cs))[0]
            out += ww / ww.sum() * target if ww.sum() > 0 else 0.0
            filled += 1
        if empty and out.sum() > 0:                                      # cells without a candidate: their weight spread pro rata
            out *= float(wb.sum()) / out.sum()
        w = pd.Series(out, index=cs.ids)
        return w[w.abs() > 1e-15], {"method": "Stratified", "cells_filled": filled, "cells_empty": len(empty)}

    def __repr__(self) -> str:
        return f"Stratified(cells={list(self.cells)!r}, per_cell={self.per_cell})"


class ZScoreRule(Construction):
    """A z-score state machine per name over the signal's path: enter short when ``z >= entry`` and
    long when ``z <= -entry`` (mean reversion), exit when ``|z| <= exit``, stop out when ``|z| >= stop``
    (and stay out until z returns inside ``exit``), and close after ``max_hold`` rows. Positions are
    ``size`` each (or ``size / count`` with ``normalise``). ``z``: the z-score expression (default
    the strategy's signal is not enough: the path is needed, so give it)."""

    needs_scores = False

    def __init__(self, z, entry: float = 2.0, exit: float = 0.5, stop: float | None = 4.0, max_hold: int | None = None, size: float = 0.1, normalise: bool = False, sign: float = -1.0) -> None:
        self.z = Field(z) if isinstance(z, str) else lift(z)
        self.entry, self.exit, self.stop, self.max_hold = float(entry), float(exit), stop, max_hold
        self.size, self.normalise, self.sign = float(size), bool(normalise), float(sign)

    def names(self):
        return self.z.names()

    def positions(self, ctx) -> np.ndarray:
        key = f"{ctx.token}|zrule:{self.z.key}:{self.entry}:{self.exit}:{self.stop}:{self.max_hold}"
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        Z = np.asarray(self.z.evaluate(ctx.full()), dtype=float)
        T, N = Z.shape
        pos = np.zeros((T, N))
        cur = np.zeros(N)
        age = np.zeros(N, dtype=np.int64)
        locked = np.zeros(N, dtype=bool)
        for t in range(T):
            z = Z[t]
            fin = np.isfinite(z)
            az = np.abs(np.where(fin, z, 0.0))
            locked &= ~(fin & (az <= self.exit))
            stopped = (cur != 0) & fin & ((az >= self.stop) if self.stop is not None else False)
            expired = (cur != 0) & ((age >= self.max_hold) if self.max_hold else False)
            close = (cur != 0) & (~fin | (az <= self.exit)) | stopped | expired
            locked |= (stopped | expired) & ~(fin & (az <= self.exit))   # out until z is back inside the exit band
            cur = np.where(close, 0.0, cur)
            age = np.where(cur != 0, age + 1, 0)                          # rows held, this one included
            enter = (cur == 0) & fin & (az >= self.entry) & ~locked & ~close
            if self.stop is not None:
                enter &= az < self.stop
            cur = np.where(enter, self.sign * np.sign(z), cur)
            age = np.where(enter, 1, age)
            pos[t] = cur
        ctx.cache.put(key, pos)
        return pos

    def batch(self, B):
        if B.ctx is None or B.rows is None:
            return None
        P = self.positions(B.ctx)[B.rows]
        if B.cols is not None:
            P = P[:, B.cols]
        P = np.where(B.universe | (P != 0), P, 0.0)
        if self.normalise:
            k = np.abs(P).sum(axis=1, keepdims=True)
            return np.where(k > 0, P / np.where(k > 0, k, 1.0), 0.0) * self.size
        return P * self.size

    def __repr__(self) -> str:
        return f"ZScoreRule({self.z!r}, entry={self.entry:g}, exit={self.exit:g}, stop={self.stop}, max_hold={self.max_hold})"


_ACTION_RANK = {"sell": 0, "exclude": 1, "underweight": 2, "softexclude": 3, "hold": 4, "noaction": 5, "buymax": 6, "buy": 7}


class Rules(Construction):
    """Per-name actions on top of a construction, by precedence (sell > exclude > underweight >
    softexclude > hold > noaction > buymax > buy, as paperport and Erin Beta+ apply them):
    ``Rules(base, [(Field("rating_score") > 13, "sell"), (Field("esg_flag") > 0, "exclude"),
    (Field("watch") > 0, "softexclude"), (Field("liquidity") < 1, ("buymax", 0.01))])``.
    sell / exclude: zero; underweight: at most the benchmark weight; softexclude: no more than now;
    hold / noaction: the current weight; buymax x: at most x; buy x: x. The other names absorb the
    difference pro rata (long-only). The construction log counts each action."""

    path_dependent = True

    def __init__(self, base: Construction, rules) -> None:
        self.base = as_construction(base) if not isinstance(base, Construction) else base
        self.rules = tuple((lift(p) if not isinstance(p, Expr) else p, a) for p, a in rules)
        self.needs_scores = self.base.needs_scores
        for _, a in self.rules:
            name = a[0] if isinstance(a, tuple) else a
            if name not in _ACTION_RANK:
                raise ValueError(f"unknown rule action {name!r}; known: {sorted(_ACTION_RANK)}")

    def constraints(self):
        return self.base.constraints()

    def names(self):
        return self.base.names() | set().union(*[p.names() for p, _ in self.rules]) if self.rules else self.base.names()

    def weights(self, cs):
        w, diag = self.base.weights(cs)
        ids = cs.ids
        wv = w.reindex(ids).fillna(0.0).to_numpy(dtype=float)
        w0 = cs.current.reindex(ids).fillna(0.0).to_numpy(dtype=float)
        wb = cs.benchmark.reindex(ids).fillna(0.0).to_numpy(dtype=float)
        rank = np.full(len(ids), 99)
        act = np.array([""] * len(ids), dtype=object)
        arg = np.full(len(ids), np.nan)
        for pred, a in self.rules:
            name, x = (a if isinstance(a, tuple) else (a, np.nan))
            hit = np.nan_to_num(np.asarray(_cs_values(cs, pred), dtype=float)) > 0
            better = hit & (_ACTION_RANK[name] < rank)
            rank = np.where(better, _ACTION_RANK[name], rank)
            act = np.where(better, name, act)
            arg = np.where(better, x, arg)
        out = wv.copy()
        fixed = act != ""
        out = np.where((act == "sell") | (act == "exclude"), 0.0, out)
        out = np.where(act == "underweight", np.minimum(out, wb), out)
        out = np.where(act == "softexclude", np.minimum(out, np.maximum(w0, 0.0)), out)
        out = np.where((act == "hold") | (act == "noaction"), w0, out)
        out = np.where(act == "buymax", np.minimum(out, arg), out)
        out = np.where(act == "buy", arg, out)
        free = ~fixed & (out > 0)
        if (wv >= 0).all() and free.any():
            room = float(wv.sum()) - float(out[fixed].sum())
            tot = float(out[free].sum())
            out[free] *= max(room, 0.0) / tot if tot > 0 else 0.0
        counts = {k: int((act == k).sum()) for k in _ACTION_RANK if (act == k).any()}
        res = pd.Series(out, index=ids)
        return res[res.abs() > 1e-15], {**{k: v for k, v in diag.items() if not k.startswith("_")}, **{f"rule_{k}": v for k, v in counts.items()}}

    def __repr__(self) -> str:
        return f"Rules({self.base!r}, {len(self.rules)} rules)"


class MaxIR(Construction):
    """The highest information ratio along the tracking-error frontier: ``base`` solved under
    ``TEV(model) <= level`` for each level, keeping the one with the best ex-ante alpha per unit
    of TE (``active=False``: the best Sharpe, on volatility). MaxSharpe is ``MaxIR(..., active=False)``."""

    path_dependent = True
    handles_exits = True

    def __init__(self, base: Optimise, model, levels=(0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05), active: bool = True, periods_per_year: float = 252.0) -> None:
        self.base, self.model, self.levels, self.active, self.periods_per_year = base, model, tuple(float(x) for x in levels), bool(active), float(periods_per_year)
        self.needs_scores = base.needs_scores

    def constraints(self):
        return self.base.constraints()

    def names(self):
        return self.base.names() | getattr(self.model, "names", lambda: set())()

    def weights(self, cs):
        _, vec, _ = self.base.objective.compile(cs)
        alpha = vec if vec is not None else np.nan_to_num(cs.scores.to_numpy(dtype=float)) if cs.scores is not None else np.zeros(len(cs.ids))
        wb = cs.benchmark.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float) if self.active else 0.0
        rows, best = [], None
        for lv in self.levels:
            w, d = self.base.add(TEV(self.model, self.active, self.periods_per_year) <= lv).weights(cs)
            wv = w.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
            te = TEV(self.model, self.active, self.periods_per_year).sigma(cs, wv)
            ir = float(alpha @ (wv - wb)) / te if te > 1e-12 else float("-inf")
            rows.append({"level": lv, "te": te, "ratio": ir, "ok": d.get("ok", True)})
            if d.get("ok", True) and (best is None or ir > best[0]):
                best = (ir, w, d, lv)
        if best is None:
            w, d = self.base.weights(cs)
            return w, {**d, "frontier": rows}
        return best[1], {**best[2], "frontier": rows, "chosen_te": best[3], "ratio": best[0]}

    def __repr__(self) -> str:
        return f"MaxIR({self.base!r}, levels={list(self.levels)}{'' if self.active else ', active=False'})"


def MaxSharpe(base: Optimise, model, levels=(0.02, 0.04, 0.06, 0.08, 0.1, 0.15), periods_per_year: float = 252.0) -> MaxIR:
    return MaxIR(base, model, levels, active=False, periods_per_year=periods_per_year)


class Sparse(Construction):
    """Hit a holdings band with reweighted L1: ``base`` re-solved with a penalty ``lam / (|w| + eps)``
    on each weight, ``lam`` doubled while there are too many names and halved while too few (an
    adaptive-lambda IRL1, not a cardinality MILP)."""

    path_dependent = True
    handles_exits = True

    def __init__(self, base: Optimise, names=(40, 60), iterations: int = 12, eps: float = 1e-4, lam: float | None = None) -> None:
        self.base, self.band, self.iterations, self.eps, self.lam = base, (int(names[0]), int(names[1])), int(iterations), float(eps), lam
        self.needs_scores = base.needs_scores

    def constraints(self):
        return self.base.constraints()

    def names(self):
        return self.base.names()

    def weights(self, cs):
        _, vec, _ = self.base.objective.compile(cs)
        scale = float(np.nanmax(np.abs(vec))) if vec is not None and np.isfinite(vec).any() and np.nanmax(np.abs(vec)) > 0 else 1.0
        lam = self.lam if self.lam is not None else 1e-3 * scale
        c = np.ones(len(cs.ids))
        lo, hi = self.band
        best, trail = None, []
        w, d = self.base.weights(cs)
        for _ in range(self.iterations):
            k = int((w.abs() > 1e-9).sum())
            trail.append({"lambda": lam, "names": k})
            gap = 0 if lo <= k <= hi else (k - hi if k > hi else lo - k)
            if best is None or gap < best[0]:
                best = (gap, w, d)
            if gap == 0:
                break
            wv = w.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
            c = 1.0 / (np.abs(wv) + self.eps)
            lam = lam * 2.0 if k > hi else lam / 2.0
            pen = Linear(lambda cs_, c=c, lam=lam: -lam * c / c.max(), name="sparsity")
            w, d = self.base.with_objective(self.base.objective + pen).weights(cs)
        return best[1], {**best[2], "sparsity": trail}

    def __repr__(self) -> str:
        return f"Sparse({self.base!r}, names={self.band})"


# ====================================================================================== overlays
class Hedge:
    """Overlay notionals (fractions of NAV) that bring the portfolio's ``fields`` exposures to
    ``targets`` (absolute, or relative to the benchmark with ``active=True``) using overlay
    instruments from the source (futures, CDX, swaps: returns per unit notional + exposures)."""

    def __init__(self, instruments=("TY",), fields=("oad",), targets: dict | None = None, active: bool = False, max_gross: float | None = None, financed: Iterable[str] = (), cost_bp: float = 0.5, cost_per_contract: float = 0.0, contract_notional: dict | None = None, tolerance: float = 0.0, lot: float | None = None, joint: bool = False) -> None:
        """``cost_per_contract`` / ``contract_notional`` (per instrument): exchange and tick costs per
        contract traded; ``tolerance``: leave the hedge unchanged while every target is within this of
        its exposure; ``lot``: round notionals (fraction of NAV) to this size. ``joint``: the
        notionals are decision variables of the portfolio's optimiser (an ``Optimise``), solved with
        the cash book: its exposure limits on ``fields`` count the overlays' exposure, each unit
        traded costs ``cost_bp``, and ``max_gross`` caps each instrument; ``targets`` are then the
        optimiser's constraints, not sizing targets."""
        self.joint = bool(joint)
        self.cost_per_contract = float(cost_per_contract)
        self.contract_notional = dict(contract_notional or {})
        self.tolerance = float(tolerance)
        self.lot = lot
        self.instruments = tuple(instruments)
        self.fields = tuple(fields)
        self.targets = dict(targets or {})
        self.active, self.max_gross = bool(active), max_gross
        self.financed = tuple(financed)
        self.cost_bp = float(cost_bp)
        self._impl = HedgeOverlay(self.instruments, self.fields, self.targets, self.active, self.max_gross)

    def names(self):
        return set(self.fields)

    def size(self, weights: pd.Series, cs: CrossSection, overlay_data: pd.DataFrame) -> tuple[pd.Series, dict]:
        return self._impl.size(weights, cs, overlay_data)

    def __repr__(self) -> str:
        return f"Hedge({list(self.instruments)!r}, fields={list(self.fields)!r}, targets={self.targets!r}{', active=True' if self.active else ''}{', joint=True' if self.joint else ''})"


class KRDWaterfall(Hedge):
    """Key-rate hedges sized longest bucket first: each ``(field, instrument)`` pair, from the
    longest key rate to the shortest, neutralises what is left in its bucket after the longer
    hedges (whose own KRDs spill into shorter buckets), to ``targets[field]`` (a number or
    ``fn(cs)``: a regime-adjusted target)."""

    def __init__(self, buckets, targets: dict | None = None, **kw) -> None:
        self.buckets = tuple(buckets)
        super().__init__(instruments=tuple(i for _, i in self.buckets), fields=tuple(f for f, _ in self.buckets), targets={}, **kw)
        self.bucket_targets = dict(targets or {})

    def size(self, weights, cs, overlay_data):
        w = weights.reindex(cs.ids).fillna(0.0)
        book = {f: float((w * cs.data[f].reindex(cs.ids).fillna(0.0)).sum()) if f in cs.data else 0.0 for f, _ in self.buckets}
        notion: dict = {}
        spill = dict.fromkeys(book, 0.0)
        for f, inst in self.buckets:
            tgt = self.bucket_targets.get(f, 0.0)
            tgt = float(tgt(cs)) if callable(tgt) else float(tgt)
            per = float(overlay_data.at[inst, f]) if (inst in overlay_data.index and f in overlay_data.columns) else 0.0
            need = tgt - book[f] - spill[f]
            n = need / per if per else 0.0
            notion[inst] = notion.get(inst, 0.0) + n
            for g, _ in self.buckets:
                if g in overlay_data.columns and inst in overlay_data.index:
                    spill[g] += n * float(np.nan_to_num(overlay_data.at[inst, g]))
        return pd.Series(notion), {"method": "KRDWaterfall", "residual": {f: book[f] + spill[f] for f in book}}

    def __repr__(self) -> str:
        return f"KRDWaterfall({list(self.buckets)!r})"


def _by_names(by) -> set[str]:
    if by is None:
        return set()
    if isinstance(by, str):
        return {by}
    if isinstance(by, Expr):
        return by.names()
    return set().union(*[_by_names(b) for b in by])


def as_construction(x) -> Construction:
    if isinstance(x, Construction):
        return x
    if type(x).__name__ == "ConstructionSpec":
        return Legacy(x)
    raise TypeError(f"portfolio must be a construction (TopN, Quantile, Optimise ...), not {type(x).__name__}")


__all__ = [
    "TEV",
    "ActiveExposure",
    "ActiveShare",
    "ActiveWeight",
    "Alpha",
    "AnnualTurnoverBudget",
    "Batch",
    "BenchmarkWeight",
    "Bounds",
    "Budget",
    "ByBenchmark",
    "ByField",
    "ByScore",
    "CVaR",
    "CVaRLimit",
    "Capped",
    "CashWeight",
    "Checked",
    "Constraint",
    "Construction",
    "Costs",
    "CurrencyExposure",
    "Equal",
    "EqualWeight",
    "Exposure",
    "GroupWeight",
    "Hedge",
    "HistoricalScenarios",
    "Hold",
    "IndexTurnover",
    "InverseField",
    "KernelCross",
    "Legacy",
    "Leverage",
    "Linear",
    "LinearMeasure",
    "MaxIR",
    "MaxSharpe",
    "Measure",
    "MinHolding",
    "MinTrade",
    "Names",
    "NoBuy",
    "NoSell",
    "Objective",
    "Optimise",
    "PCARisk",
    "Pairs",
    "Pin",
    "Projected",
    "Quantile",
    "RankWeighted",
    "Risk",
    "Sparse",
    "Subset",
    "TargetTE",
    "Term",
    "Tilt",
    "TopN",
    "TrackingError",
    "TradeLimit",
    "Turnover",
    "Weight",
    "Weighting",
    "as_construction",
    "as_weighting",
    "check_all",
]
