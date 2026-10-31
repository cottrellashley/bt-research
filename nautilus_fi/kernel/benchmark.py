"""Benchmarks as expressions: ``Index("LUACTRUU")`` is both a universe (its members on each date)
and a benchmark (weights through time, returns, exposures).

* ``Index(name)``: constituents and weights from the source (``source.constituents(name)``:
  reconstitution rows effective after their close) or a daily weight field
  (``Index(name, weight_field="index_weight")``); weights drift with returns between
  reconstitutions or stay static (``drift="STATIC"``).
* Derived indices re-apply rules at every parent reconstitution, then drift like the parent:
  ``.exclude(sector="FIN")``, ``.only(currency="USD")``, ``.where(Field("rating_score") <= 10)``,
  ``.cap(issuer=0.03)`` / ``.cap(name=0.02)`` (pro-rata redistribution), ``.reweight("equal")``.
* ``Blend((Index("A"), 0.6), (Index("B"), 0.4), rebalance="BME")``: a composite reset on a calendar.
* ``ReturnSeries("index.tr")``: a published return series only (no weights).

Row ``t`` of a weight matrix holds close-of-day weights: they earn the period ending ``t + 1``.
Membership on a decision date is therefore the composition that earns the next period, which
index providers publish ahead of the reconstitution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nautilus_fi.benchmarks.benchmark import BenchmarkWeightsUnavailable, cap_weights
from nautilus_fi.domain.calendars import RebalanceCalendar, parse_calendar
from nautilus_fi.domain.conventions import WeightDrift
from nautilus_fi.kernel.expr import EvalContext, Expr, _fmt, alive_mask, lift


def _drift(v) -> WeightDrift:
    return v if isinstance(v, WeightDrift) else WeightDrift(str(v).upper())


def drift_weights(recon: dict[int, np.ndarray], R: np.ndarray, alive: np.ndarray, drift: WeightDrift) -> np.ndarray:
    """Close-of-day weights from reconstitution vectors (row -> weights effective after its close):
    drifting with returns (``DRIFT``) or held at the reconstitution values (``STATIC``), dead names
    dropped and the rest renormalised."""
    T, N = R.shape
    out = np.zeros((T, N))
    cur = np.zeros(N)
    anchor = None
    for t in range(T):
        if t > 0 and cur.any():
            if drift == WeightDrift.DRIFT:
                cur = cur * (1.0 + np.nan_to_num(R[t]))
            elif anchor is not None:
                cur = anchor.copy()
            cur = np.where(alive[t], cur, 0.0)
            s = cur.sum()
            cur = cur / s if s > 0 else cur
        if t in recon:
            nxt = alive[t + 1] if t + 1 < T else alive[t]
            cur = np.where(nxt, recon[t], 0.0)
            s = cur.sum()
            cur = cur / s if s > 0 else cur
            anchor = cur.copy()
        out[t] = cur
    return out


class Benchmark(Expr):
    """A benchmark is a membership predicate (``kind="bool"``) with weights and returns."""

    kind = "bool"
    has_weights = True

    # -------------------------------------------------------------- interface
    def weights(self, ctx: EvalContext) -> np.ndarray:
        key = f"{ctx.token}|bw:{self.key}|{ctx.return_field}"
        hit = ctx.cache.get(key)
        if hit is None:
            hit = self._weights(ctx)
            ctx.cache.put(key, hit)
        return hit

    def _weights(self, ctx: EvalContext) -> np.ndarray:
        raise BenchmarkWeightsUnavailable(f"{self!r}: constituent weights are not available")

    def recon(self, ctx: EvalContext) -> tuple[dict[int, np.ndarray], WeightDrift]:
        """Reconstitution rows and their weight vectors (the basis derived indices transform)."""
        raise BenchmarkWeightsUnavailable(f"{self!r}: no reconstitutions")

    def returns(self, ctx: EvalContext) -> np.ndarray:
        """Benchmark return per row (period ending at the row); row 0 is 0."""
        key = f"{ctx.token}|br:{self.key}|{ctx.return_field}"
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        W = self.weights(ctx)
        R = np.nan_to_num(np.asarray(ctx.field(ctx.return_field), dtype=np.float64))
        out = np.zeros(ctx.T)
        out[1:] = (W[:-1] * R[1:]).sum(axis=1)
        ctx.cache.put(key, out)
        return out

    def exposures(self, ctx: EvalContext, fields) -> pd.DataFrame:
        W = self.weights(ctx)
        return pd.DataFrame({f: (W * np.nan_to_num(ctx.field(f))).sum(axis=1) for f in fields if f in ctx.panel}, index=ctx.panel.dates)

    def _eval(self, ctx):
        return self.weights(ctx) > 0

    # -------------------------------------------------------------- derived
    def exclude(self, **attr_values) -> Benchmark:
        return Filtered(self, _attr_pred(attr_values), negate=True, label=f"exclude({_kw(attr_values)})")

    def only(self, **attr_values) -> Benchmark:
        return Filtered(self, _attr_pred(attr_values), negate=False, label=f"only({_kw(attr_values)})")

    def where(self, predicate, other=None) -> Benchmark:            # type: ignore[override]
        if other is not None:
            raise TypeError("Benchmark.where(predicate) keeps members where the predicate holds")
        return Filtered(self, lift(predicate), negate=False, label=f"where({predicate!r})")

    def cap(self, issuer: float | None = None, name: float | None = None, by: str | None = None, max_weight: float | None = None) -> Benchmark:
        """Cap a group's total weight (``issuer=0.03`` -> issuers at 3%, ``by=`` another attribute)
        or single names (``name=0.02``), redistributing pro rata."""
        out: Benchmark = self
        if issuer is not None or by is not None:
            out = Capped(out, by or "issuer", float(issuer if issuer is not None else max_weight))
        if name is not None:
            out = Capped(out, None, float(name))
        return out

    def reweight(self, how: str = "equal") -> Benchmark:
        return Reweighted(self, how)


def _kw(d: dict) -> str:
    return ", ".join(f"{k}={_fmt(list(v) if isinstance(v, (list, tuple, set)) else v)}" for k, v in d.items())


def _attr_pred(attr_values: dict) -> Expr:
    from nautilus_fi.kernel.expr import Field

    pred = None
    for k, v in attr_values.items():
        vals = list(v) if isinstance(v, (list, tuple, set)) else [v]
        p = Field(k).isin(vals)
        pred = p if pred is None else (pred | p)
    if pred is None:
        raise ValueError("give at least one attribute=value")
    return pred


class Index(Benchmark):
    """A published index: constituents from the source (or a daily weight field)."""

    def __init__(self, name: str, drift: str | WeightDrift = "DRIFT", weight_field: str | None = None) -> None:
        self.name = str(name)
        self.drift = _drift(drift)
        self.weight_field = weight_field

    def __repr__(self) -> str:
        extra = ("" if self.drift == WeightDrift.DRIFT else ", drift='STATIC'") + ("" if self.weight_field is None else f", weight_field={self.weight_field!r}")
        return f"Index({self.name!r}{extra})"

    def _own_indices(self):
        return set() if self.weight_field is not None else {self.name}

    def _own_names(self):
        return {self.weight_field} if self.weight_field is not None else set()

    def _constituents(self, ctx: EvalContext) -> pd.DataFrame:
        if ctx.source is None:
            raise BenchmarkWeightsUnavailable(f"{self!r} needs a source with constituents")
        return ctx.source.constituents(self.name, None, ctx.panel.dates[-1] if ctx.T else None)

    def recon(self, ctx):
        key = f"{ctx.token}|recon:{self.key}"
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        T, N = ctx.T, ctx.N
        rows: dict[int, np.ndarray] = {}
        if self.weight_field is not None:
            W = np.nan_to_num(np.asarray(ctx.field(self.weight_field), dtype=np.float64))
            member = W > 0
            change = np.concatenate([[True], (member[1:] != member[:-1]).any(axis=1)])
            for t in np.nonzero(change)[0]:
                rows[int(t)] = W[t].copy()
        else:
            c = self._constituents(ctx)
            d = pd.to_datetime(c["date"]).dt.normalize()
            pos = ctx.panel.id_pos(c["instrument_id"].astype(str))
            w = c["weight"].to_numpy(dtype=float)
            dates = ctx.panel.dates
            before = d < dates[0]
            if before.any():
                last = d[before].max()
                sel = (d == last).to_numpy() & (pos >= 0)
                v = np.zeros(N)
                v[pos[sel]] = w[sel]
                rows[0] = v
            for day in sorted(d[~before].unique()):
                i = int(dates.searchsorted(day, side="right")) - 1     # a non-trading date: effective from the previous panel date
                if i < 0 or i >= T:
                    continue
                sel = (d == day).to_numpy() & (pos >= 0)
                v = np.zeros(N)
                v[pos[sel]] = w[sel]
                rows[i] = v
        out = (rows, self.drift)
        ctx.cache.put(key, out)
        return out

    def _weights(self, ctx):
        if self.weight_field is not None and self.drift == WeightDrift.DRIFT:
            W = np.nan_to_num(np.asarray(ctx.field(self.weight_field), dtype=np.float64))
            s = W.sum(axis=1, keepdims=True)
            return np.where(s > 0, W / np.where(s > 0, s, 1.0), 0.0)        # published daily weights (already drifted)
        rows, drift = self.recon(ctx)
        R = np.asarray(ctx.field(ctx.return_field), dtype=np.float64)
        return drift_weights(rows, R, alive_mask(ctx), drift)


class _Derived(Benchmark):
    def __init__(self, parent: Benchmark) -> None:
        self.parent = parent
        self._children = (parent,)

    def _transform(self, ctx: EvalContext, t: int, w: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recon(self, ctx):
        key = f"{ctx.token}|recon:{self.key}"
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        rows, drift = self.parent.recon(ctx)
        out = ({t: self._transform(ctx, t, w) for t, w in rows.items()}, drift)
        ctx.cache.put(key, out)
        return out

    def _weights(self, ctx):
        rows, drift = self.recon(ctx)
        R = np.asarray(ctx.field(ctx.return_field), dtype=np.float64)
        return drift_weights(rows, R, alive_mask(ctx), drift)


class Filtered(_Derived):
    def __init__(self, parent: Benchmark, predicate: Expr, negate: bool, label: str) -> None:
        super().__init__(parent)
        self.predicate, self.negate, self.label = predicate, negate, label
        self._children = (parent, predicate)

    def __repr__(self) -> str:
        return f"{self.parent!r}.{self.label}"

    def _transform(self, ctx, t, w):
        m = np.asarray(self.predicate.evaluate(ctx.full()), dtype=bool)[t]
        keep = ~m if self.negate else m
        v = np.where(keep, w, 0.0)
        s = v.sum()
        return v / s if s > 0 else v


class Capped(_Derived):
    def __init__(self, parent: Benchmark, by: str | None, cap: float) -> None:
        super().__init__(parent)
        self.by, self.cap_ = by, float(cap)

    def __repr__(self) -> str:
        if self.by is None:
            return f"{self.parent!r}.cap(name={self.cap_:g})"
        return f"{self.parent!r}.cap({'issuer' if self.by == 'issuer' else 'by=' + repr(self.by) + ', max_weight'}={self.cap_:g})"

    def _own_names(self):
        return {self.by} if self.by else set()

    def _transform(self, ctx, t, w):
        nz = np.nonzero(w > 0)[0]
        if not len(nz):
            return w
        ws = pd.Series(w[nz], index=ctx.panel.ids[nz])
        groups = None
        if self.by is not None:
            g = ctx.groups(self.by)
            codes = g.broadcast(ctx.T, ctx.N)[t][nz]
            groups = pd.Series(codes, index=ws.index)
        capped = cap_weights(ws, groups, self.cap_)
        out = np.zeros(ctx.N)
        out[nz] = capped.to_numpy() * w[nz].sum()
        return out


class Reweighted(_Derived):
    def __init__(self, parent: Benchmark, how: str = "equal") -> None:
        if how != "equal":
            raise ValueError("reweight: 'equal' (use .cap() for capped market weights)")
        super().__init__(parent)
        self.how = how

    def __repr__(self) -> str:
        return f"{self.parent!r}.reweight({self.how!r})"

    def _transform(self, ctx, t, w):
        m = w > 0
        return np.where(m, 1.0 / max(int(m.sum()), 1), 0.0)


class Blend(Benchmark):
    """Weighted composite of benchmarks, reset to the target mix on ``rebalance`` dates."""

    def __init__(self, *components: tuple[Benchmark, float], rebalance: str | RebalanceCalendar = "BME") -> None:
        if not components:
            raise ValueError("Blend needs (benchmark, weight) pairs")
        self.components = tuple((b, float(w)) for b, w in components)
        self.rebalance = parse_calendar(rebalance)
        self._children = tuple(b for b, _ in self.components)

    def __repr__(self) -> str:
        return f"Blend({', '.join(f'({b!r}, {w:g})' for b, w in self.components)}, rebalance={str(self.rebalance)!r})"

    def _mix(self, ctx) -> np.ndarray:
        rets = np.column_stack([b.returns(ctx) for b, _ in self.components])
        target = np.array([w for _, w in self.components])
        target = target / target.sum()
        days = list(ctx.panel.dates.date)
        due = set(self.rebalance.dates(days, first_decision=True))
        c = np.zeros((ctx.T, len(target)))
        cur = target.copy()
        for t, d in enumerate(days):
            if t > 0:
                cur = cur * (1.0 + rets[t])
                cur = cur / cur.sum()
            if d in due:
                cur = target.copy()
            c[t] = cur
        return c

    def _weights(self, ctx):
        c = self._mix(ctx)
        out = np.zeros((ctx.T, ctx.N))
        for k, (b, _) in enumerate(self.components):
            out += c[:, [k]] * b.weights(ctx)
        return out

    def returns(self, ctx):
        c = self._mix(ctx)
        rets = np.column_stack([b.returns(ctx) for b, _ in self.components])
        out = np.zeros(ctx.T)
        out[1:] = (c[:-1] * rets[1:]).sum(axis=1)
        return out


class ReturnSeries(Benchmark):
    """A benchmark known only by its published return series (a source date series)."""

    has_weights = False

    def __init__(self, series: str) -> None:
        self.series = str(series)

    def __repr__(self) -> str:
        return f"ReturnSeries({self.series!r})"

    def _own_names(self):
        return {self.series}

    def returns(self, ctx):
        v = ctx.series_values(self.series)[:, 0].copy()
        v[0] = 0.0
        return np.nan_to_num(v)

    def _eval(self, ctx):
        raise BenchmarkWeightsUnavailable(f"{self!r} has no constituents: it cannot define a universe")


def as_benchmark(b) -> Benchmark | None:
    if b is None or isinstance(b, Benchmark):
        return b
    if isinstance(b, str):
        return Index(b)
    raise TypeError(f"benchmark must be an Index / Blend / ReturnSeries, not {type(b).__name__}")


__all__ = ["Benchmark", "BenchmarkWeightsUnavailable", "Blend", "Capped", "Filtered", "Index", "ReturnSeries", "Reweighted", "as_benchmark", "drift_weights"]
