"""Benchmarks as expressions: ``Index("LUACTRUU")`` is both a universe (its members on each date)
and a benchmark (weights through time, returns, exposures).

* ``Index(name)``: constituents and weights from the source (``source.constituents(name)``:
  reconstitution rows effective after their close), drifting with returns between
  reconstitutions or held flat (``drift="STATIC"``); or the provider's published daily weights
  (``Index(name, weight_field="index_weight")``), used as published either way.
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
from nautilus_fi.kernel.expr import EvalContext, Expr, _fmt, alive_mask, lift, shift_rows


def _drift(v) -> WeightDrift:
    return v if isinstance(v, WeightDrift) else WeightDrift(str(v).upper())


def drift_weights(recon: dict[int, np.ndarray], R: np.ndarray, alive: np.ndarray, drift: WeightDrift, redemptions: str = "pro_rata") -> np.ndarray:
    """Close-of-day weights from reconstitution vectors (row -> weights effective after its close):
    drifting with returns (``DRIFT``) or held at the reconstitution values (``STATIC``), dead names
    dropped and the rest renormalised. A reconstitution keeps the members that exist at its close;
    one that stops existing the next day earns nothing on its last period and is then dropped.
    ``redemptions="cash"``: a redeemed member's weight is held as (zero-return) cash until the next
    reconstitution instead of being spread pro rata over the rest."""
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
            if redemptions == "cash":
                total = cur.sum()
                cur = np.where(alive[t], cur, 0.0)
                cur = cur / total if total > 0 else cur                     # weights of a whole that includes the cash
            else:
                cur = np.where(alive[t], cur, 0.0)
                s = cur.sum()
                cur = cur / s if s > 0 else cur
        if t in recon:
            cur = np.where(alive[t], recon[t], 0.0)                 # what exists at the close (not whether it survives the next day)
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

    def decision_lag(self, ctx: EvalContext) -> int:
        """Rows by which the weights a decision may see trail the index's own holdings: the
        availability lag of a published weight field (constituent files are published ahead: 0)."""
        return max((c.decision_lag(ctx) for c in self.children() if isinstance(c, Benchmark)), default=0)

    def known_weights(self, ctx: EvalContext) -> np.ndarray:
        """The weights as a decision on each row may know them (``weights`` shifted by
        ``decision_lag``). Returns and analytics use ``weights``, the holdings themselves."""
        lag = self.decision_lag(ctx)
        return self.weights(ctx) if lag == 0 else shift_rows(self.weights(ctx), lag, 0.0)

    def returns(self, ctx: EvalContext) -> np.ndarray:
        """Benchmark return per row (period ending at the row); row 0 is 0."""
        key = f"{ctx.token}|br:{self.key}|{ctx.return_field}"
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        W = self.weights(ctx)
        R = np.nan_to_num(np.asarray(ctx.realised(ctx.return_field), dtype=np.float64))
        out = np.zeros(ctx.T)
        out[1:] = (W[:-1] * R[1:]).sum(axis=1)
        ctx.cache.put(key, out)
        return out

    def return_gaps(self, ctx: EvalContext) -> np.ndarray:
        """Rows (bool) whose benchmark return is incomplete: a member it holds that exists has no
        realised return. ``returns`` counts such a member as earning zero; the engine applies the
        strategy's missing-return policy to these rows."""
        W = self.weights(ctx)
        R = np.asarray(ctx.realised(ctx.return_field), dtype=np.float64)
        out = np.zeros(ctx.T, dtype=bool)
        out[1:] = ((W[:-1] > 0) & alive_mask(ctx)[1:] & ~np.isfinite(R[1:])).any(axis=1)
        return out

    def exposures(self, ctx: EvalContext, fields) -> pd.DataFrame:
        W = self.weights(ctx)
        return pd.DataFrame({f: (W * np.nan_to_num(ctx.field(f))).sum(axis=1) for f in fields if f in ctx.panel}, index=ctx.panel.dates)

    def _eval(self, ctx):
        return self.known_weights(ctx) > 0

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

    def reweight(self, how: str = "equal", field: str | None = None, schedule=None) -> Benchmark:
        return Reweighted(self, how, field, schedule)


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

    def __init__(self, name: str, drift: str | WeightDrift = "DRIFT", weight_field: str | None = None, redemptions: str = "pro_rata") -> None:
        if redemptions not in ("pro_rata", "cash"):
            raise ValueError("redemptions must be 'pro_rata' or 'cash'")
        self.name = str(name)
        self.drift = _drift(drift)
        self.weight_field = weight_field
        self.redemptions = redemptions

    def __repr__(self) -> str:
        extra = ("" if self.drift == WeightDrift.DRIFT else ", drift='STATIC'") + ("" if self.weight_field is None else f", weight_field={self.weight_field!r}")
        return f"Index({self.name!r}{extra})"

    def _own_indices(self):
        return set() if self.weight_field is not None else {self.name}

    def _own_names(self):
        return {self.weight_field} if self.weight_field is not None else set()

    def decision_lag(self, ctx):
        return ctx.lag_of(self.weight_field) if self.weight_field is not None else 0

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
            W = np.nan_to_num(np.asarray(ctx.realised(self.weight_field), dtype=np.float64))
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
        if self.weight_field is not None:                                # published daily weights: the provider's own, used as published
            W = np.nan_to_num(np.asarray(ctx.realised(self.weight_field), dtype=np.float64))
            s = W.sum(axis=1, keepdims=True)
            return np.where(s > 0, W / np.where(s > 0, s, 1.0), 0.0)
        rows, drift = self.recon(ctx)
        R = np.asarray(ctx.realised(ctx.return_field), dtype=np.float64)
        return drift_weights(rows, R, alive_mask(ctx), drift, self.redemptions)


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
        R = np.asarray(ctx.realised(ctx.return_field), dtype=np.float64)
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
    """The parent's members reweighted ``"equal"`` or by ``"market_value"`` (``field``, e.g. amount
    outstanding x price), at the parent's reconstitutions and on ``schedule`` (its own calendar,
    e.g. "BME"), drifting in between."""

    def __init__(self, parent: Benchmark, how: str = "equal", field: str | None = None, schedule=None) -> None:
        if how not in ("equal", "market_value"):
            raise ValueError("reweight: 'equal' or 'market_value' (with field=)")
        if how == "market_value" and not field:
            raise ValueError("reweight('market_value') needs field= (e.g. 'amt_out')")
        super().__init__(parent)
        self.how, self.field, self.schedule = how, field, (parse_calendar(schedule) if schedule is not None else None)

    def _own_names(self):
        return {self.field} if self.field else set()

    def __repr__(self) -> str:
        extra = ("" if self.field is None else f", field={self.field!r}") + ("" if self.schedule is None else f", schedule={str(self.schedule)!r}")
        return f"{self.parent!r}.reweight({self.how!r}{extra})"

    def recon(self, ctx):
        if self.schedule is None:
            return super().recon(ctx)
        key = f"{ctx.token}|recon:{self.key}"
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        rows, drift = self.parent.recon(ctx)
        W = self.parent.weights(ctx)                                     # the parent's composition on its own calendar dates
        due = set(ctx.panel.dates.get_indexer(pd.to_datetime(self.schedule.dates(list(ctx.panel.dates.date), True))))
        allrows = sorted(set(rows) | {int(t) for t in due if t >= 0})
        out = ({t: self._transform(ctx, t, rows.get(t, W[t])) for t in allrows}, drift)
        ctx.cache.put(key, out)
        return out

    def _transform(self, ctx, t, w):
        m = w > 0
        if self.how == "equal":
            return np.where(m, 1.0 / max(int(m.sum()), 1), 0.0)
        v = np.nan_to_num(np.asarray(ctx.field(self.field, 0)[t], dtype=np.float64))
        v = np.where(m, np.clip(v, 0.0, None), 0.0)
        return v / v.sum() if v.sum() > 0 else np.where(m, 1.0 / max(int(m.sum()), 1), 0.0)


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

    @property
    def has_weights(self) -> bool:                  # type: ignore[override]
        return all(b.has_weights for b, _ in self.components)

    def return_gaps(self, ctx):
        out = np.zeros(ctx.T, dtype=bool)
        for b, _ in self.components:
            out |= np.asarray(b.return_gaps(ctx), dtype=bool)
        return out

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
        v = ctx.series_values(self.series, 0)[:, 0].copy()
        v[0] = 0.0
        return np.nan_to_num(v)

    def return_gaps(self, ctx):
        out = ~np.isfinite(ctx.series_values(self.series, 0)[:, 0])
        out[0] = False
        return out

    def _eval(self, ctx):
        raise BenchmarkWeightsUnavailable(f"{self!r} has no constituents: it cannot define a universe")


class PairedIndex(Benchmark):
    """Two universes for one index (Erin Beta+'s ``_STATS`` / ``RU`` pair): decisions see the
    ``construction`` index (next month's composition, published ahead), returns and analytics use
    the ``returns`` index (what the index actually holds this month)."""

    def __init__(self, construction: Benchmark, returns: Benchmark) -> None:
        self.construction, self.returns_ix = as_benchmark(construction), as_benchmark(returns)
        self._children = (self.construction, self.returns_ix)

    def __repr__(self) -> str:
        return f"PairedIndex({self.construction!r}, {self.returns_ix!r})"

    @property
    def drift(self):
        return getattr(self.returns_ix, "drift", WeightDrift.DRIFT)

    def _weights(self, ctx):
        return self.returns_ix.weights(ctx)

    def known_weights(self, ctx):
        return self.construction.known_weights(ctx)

    def returns(self, ctx):
        return self.returns_ix.returns(ctx)

    def return_gaps(self, ctx):
        return self.returns_ix.return_gaps(ctx)


class BenchmarkStat(Expr):
    """A benchmark-weighted average of a field per date, broadcast (``@bench.oas``): filters such
    as ``Field("oas_bp") > BenchmarkStat("oas_bp", Index("X")) * 1.5``. Uses the weights a decision
    may know."""

    def __init__(self, field: str, benchmark) -> None:
        self.field, self.benchmark = str(field), as_benchmark(benchmark)
        self._children = (self.benchmark,)

    def _params(self):
        return [self.field]

    def _own_names(self):
        return {self.field}

    def __repr__(self) -> str:
        return f"BenchmarkStat({self.field!r}, {self.benchmark!r})"

    def _eval(self, ctx):
        W = self.benchmark.known_weights(ctx)
        X = np.nan_to_num(np.asarray(ctx.field(self.field), dtype=np.float64))
        s = W.sum(axis=1)
        v = np.where(s > 0, (W * X).sum(axis=1) / np.where(s > 0, s, 1.0), np.nan)
        return np.broadcast_to(v[:, None], (ctx.T, ctx.N))


def replication_report(benchmark, published: pd.Series, source, start=None, end=None, tolerance_bp: float = 20.0, return_field: str = "tr") -> dict:
    """The benchmark rebuilt from constituents against the provider's published returns: the
    annualised tracking difference and error, and whether the difference is within
    ``tolerance_bp`` a year (the non-circular replication test: ``published`` must come from the
    provider, not from this formula)."""
    from nautilus_fi.kernel.data import DataRequest, as_source
    from nautilus_fi.kernel.expr import EvalContext

    src = as_source(source)
    cat = src.catalog()
    names = sorted(set(benchmark.names()) | {return_field} | ({"alive"} if "alive" in cat else set()))
    p = src.load(DataRequest(tuple(n for n in names if n in cat and cat[n].entity == "asset"), start, end))
    ctx = EvalContext(p, src.attributes(p.ids), None, src, cat, return_field=return_field)
    rebuilt = pd.Series(benchmark.returns(ctx), index=p.dates).iloc[1:]
    pub = published.reindex(rebuilt.index)
    both = rebuilt.notna() & pub.notna()
    d = (rebuilt - pub)[both]
    years = max((d.index[-1] - d.index[0]).days / 365.0, 1e-9) if len(d) > 1 else float("nan")
    td = float(((1 + rebuilt[both]).prod() ** (1 / years) - (1 + pub[both]).prod() ** (1 / years)) * 1e4) if len(d) > 1 else float("nan")
    te = float(d.std() * np.sqrt(252) * 1e4) if len(d) > 1 else float("nan")
    return {"dates": int(both.sum()), "tracking_difference_bp": td, "tracking_error_bp": te, "within_tolerance": bool(np.isfinite(td) and abs(td) <= tolerance_bp), "tolerance_bp": tolerance_bp, "worst_day_bp": float(d.abs().max() * 1e4) if len(d) else float("nan")}


def as_benchmark(b) -> Benchmark | None:
    if b is None or isinstance(b, Benchmark):
        return b
    if isinstance(b, str):
        return Index(b)
    raise TypeError(f"benchmark must be an Index / Blend / ReturnSeries, not {type(b).__name__}")


__all__ = ["Benchmark", "BenchmarkStat", "BenchmarkWeightsUnavailable", "Blend", "Capped", "Filtered", "Index", "PairedIndex", "ReturnSeries", "Reweighted", "as_benchmark", "drift_weights", "replication_report"]
