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
        self.path_dependent = self.buffer > 0

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

    def __init__(self, n: int = 5, long: int | None = None, short: int | None = None, by=None, weighting=None, invested: float = 1.0, short_gross: float | None = None) -> None:
        self.n, self.long, self.short, self.by = int(n), long, short, by
        self.weighting, self.invested, self.short_gross = as_weighting(weighting), float(invested), short_gross

    def names(self):
        return self.weighting.names() | _by_names(self.by)

    def bins(self, B: Batch) -> np.ndarray:
        if B.scores is None:
            raise ValueError("Quantile needs a signal")
        R, m = B.shape
        pct = grouped_rank(B.scores, B.groups(self.by), R, m, "first")
        return np.where(np.isfinite(pct), np.clip(np.ceil(pct * self.n), 1, self.n), 0).astype(np.int32)

    def batch(self, B):
        b = self.bins(B)
        w = self.weighting.apply(b == (self.long or self.n), B) * self.invested
        if self.short is not None:
            w = w - self.weighting.apply(b == self.short, B) * (self.short_gross if self.short_gross is not None else self.invested)
        return w

    def __repr__(self) -> str:
        parts = [str(self.n)]
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
class Measure:
    """The left-hand side of a constraint; compare it with numbers or ``.within(...)``."""

    kind = ""

    def __le__(self, x) -> Constraint:
        return Constraint(self, None, float(x))

    def __ge__(self, x) -> Constraint:
        return Constraint(self, float(x), None)

    def __eq__(self, x) -> Constraint:                 # type: ignore[override]
        return Constraint(self, float(x), float(x))

    def __hash__(self) -> int:
        return hash(repr(self))

    def within(self, lower: float, upper: float | None = None) -> Constraint:
        """``within(b)``: the symmetric band [-b, b]; ``within(lo, hi)``: [lo, hi]."""
        if upper is None:
            return Constraint(self, -abs(float(lower)), abs(float(lower)))
        return Constraint(self, float(lower), float(upper))

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


class Constraint:
    def __init__(self, measure: Measure, lo: float | None, hi: float | None, hard: bool = True, penalty: float = 100.0) -> None:
        self.measure, self.lo, self.hi, self.hard, self.penalty = measure, lo, hi, bool(hard), float(penalty)

    def soft(self, penalty: float = 100.0) -> Constraint:
        return Constraint(self.measure, self.lo, self.hi, False, penalty)

    @property
    def slot(self) -> tuple:
        return self.measure.slot

    def names(self) -> set[str]:
        return self.measure.names()

    def to_spec(self) -> ConstraintSpec | None:
        m = self.measure
        if isinstance(m, LinearMeasure):
            return None
        kw = {k: v for k, v in m.spec_kw().items() if k not in ("value",) or v is not None}
        if "value" in kw and kw["value"] is None:
            kw.pop("value")
        return ConstraintSpec(m.kind, self.lo, self.hi, hard=self.hard, penalty=self.penalty, **kw)

    def linear_row(self, cs) -> LinearRow | None:
        if not isinstance(self.measure, LinearMeasure):
            return None
        return LinearRow(self.measure.coef(cs), self.lo, self.hi, repr(self.measure), self.hard, self.penalty)

    def __repr__(self) -> str:
        m = self.measure
        if isinstance(m, _MinHold):
            body = f"MinHolding({'' if self.lo is None else f'{self.lo:g}'})"
        elif self.lo is not None and self.hi is not None and self.lo == self.hi:
            body = f"{m!r} == {self.lo:g}"
        elif self.lo is not None and self.hi is not None and self.lo == -self.hi:
            body = f"{m!r}.within({self.hi:g})"
        elif self.lo is not None and self.hi is not None:
            body = f"{m!r}.within({self.lo:g}, {self.hi:g})"
        elif self.hi is not None:
            body = f"{m!r} <= {self.hi:g}"
        else:
            body = f"{m!r} >= {self.lo:g}"
        return body + ("" if self.hard else f".soft({self.penalty:g})")


def MinHolding(min_weight: float | None = None) -> Constraint:
    """Each holding zero or at least ``min_weight`` of NAV (None: the minimum piece at the price)."""
    return Constraint(_MinHold(), None if min_weight is None else float(min_weight), None)


def _replace(old: tuple, new: tuple) -> tuple:
    slots = {c.slot for c in new}
    return tuple(c for c in old if c.slot not in slots) + tuple(new)


def check_all(constraints, weights: pd.Series, cs: CrossSection) -> list[dict]:
    if not constraints:
        return []
    initial = float(cs.current.abs().sum()) < 1e-9
    specs = []
    out: list[dict] = []
    for c in constraints:
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
    return {"optimiser": res.summary(), "binding": res.binding[:50], "relaxations": res.relaxations, "ok": res.ok, "message": res.message}


def _solve(cs, legacy_objective, constraints, long_only, settings, extras: dict | None = None):
    specs = [s for s in (c.to_spec() for c in constraints) if s is not None]
    rows = [r for r in (c.linear_row(cs) for c in constraints) if r is not None]
    need_scen = any(s.kind == "cvar" for s in specs) or (extras or {}).get("scenarios") is not None
    if need_scen and cs.scenarios is None:
        prov = (extras or {}).get("scenarios") or next((c.measure.scenarios for c in constraints if isinstance(c.measure, CVaRLimit)), None)
        if prov is not None:
            cs = dataclasses.replace(cs, scenarios=prov(cs))
    return optimise(cs, legacy_objective, specs, long_only, settings, rows)


class Optimise(Construction):
    """An LP / MILP / QP over trades, solved by HiGHS: maximise the objective subject to the
    constraints (fully invested by default when long-only)."""

    path_dependent = True

    def __init__(self, objective=None, *constraints: Constraint, long_only: bool = True, settings: OptimiserSettings | None = None, constraints_: Iterable[Constraint] | None = None) -> None:
        if isinstance(objective, Constraint):
            constraints = (objective, *constraints)
            objective = None
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
        res = _solve(cs, legacy, self.constraints_, self.long_only, self.settings, extras)
        return res.weights[res.weights.abs() > 1e-12], _opt_diag(res)

    # ------------------------------------------------------------------ variants
    def replace(self, *constraints: Constraint) -> Optimise:
        """Swap constraints in the same slot (``Turnover() <= 0.05`` replaces the turnover limit)."""
        return Optimise(self.objective, *_replace(self.constraints_, constraints), long_only=self.long_only, settings=self.settings)

    def add(self, *constraints: Constraint) -> Optimise:
        return Optimise(self.objective, *self.constraints_, *constraints, long_only=self.long_only, settings=self.settings)

    def without(self, *measures) -> Optimise:
        """Drop constraints by measure class (``Turnover``) or slot of an example constraint."""
        def drop(c):
            for m in measures:
                if isinstance(m, type) and isinstance(c.measure, m):
                    return True
                if isinstance(m, Constraint) and m.slot == c.slot:
                    return True
            return False
        return Optimise(self.objective, *[c for c in self.constraints_ if not drop(c)], long_only=self.long_only, settings=self.settings)

    def with_objective(self, objective) -> Optimise:
        return Optimise(objective, *self.constraints_, long_only=self.long_only, settings=self.settings)

    def with_settings(self, **changes) -> Optimise:
        return Optimise(self.objective, *self.constraints_, long_only=self.long_only, settings=dataclasses.replace(self.settings, **changes))

    def with_(self, **changes) -> Optimise:
        kw = {"objective": self.objective, "long_only": self.long_only, "settings": self.settings, **changes}
        cons = changes.get("constraints", self.constraints_)
        kw.pop("constraints", None)
        return Optimise(kw.pop("objective"), *cons, **kw)

    def __repr__(self) -> str:
        cons = ", ".join(repr(c) for c in self.constraints_)
        extra = "" if self.long_only else ", long_only=False"
        return f"Optimise({self.objective!r}{', ' + cons if cons else ''}{extra})"


# ====================================================================================== overlays
class Hedge:
    """Overlay notionals (fractions of NAV) that bring the portfolio's ``fields`` exposures to
    ``targets`` (absolute, or relative to the benchmark with ``active=True``) using overlay
    instruments from the source (futures, CDX, swaps: returns per unit notional + exposures)."""

    def __init__(self, instruments=("TY",), fields=("oad",), targets: dict | None = None, active: bool = False, max_gross: float | None = None, financed: Iterable[str] = (), cost_bp: float = 0.5) -> None:
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
        return f"Hedge({list(self.instruments)!r}, fields={list(self.fields)!r}, targets={self.targets!r}{', active=True' if self.active else ''})"


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
    "ActiveExposure",
    "ActiveShare",
    "ActiveWeight",
    "Alpha",
    "Batch",
    "BenchmarkWeight",
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
    "Equal",
    "EqualWeight",
    "Exposure",
    "GroupWeight",
    "Hedge",
    "HistoricalScenarios",
    "Hold",
    "InverseField",
    "KernelCross",
    "Legacy",
    "Leverage",
    "Linear",
    "LinearMeasure",
    "Measure",
    "MinHolding",
    "Names",
    "Objective",
    "Optimise",
    "PCARisk",
    "Pairs",
    "Projected",
    "Quantile",
    "RankWeighted",
    "Risk",
    "Term",
    "Tilt",
    "TopN",
    "TrackingError",
    "Turnover",
    "Weight",
    "Weighting",
    "as_construction",
    "as_weighting",
    "check_all",
]
