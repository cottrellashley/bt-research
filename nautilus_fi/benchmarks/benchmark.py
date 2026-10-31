"""Benchmarks as objects: constituents and weights through time, not a label and one return series.

* ``ConstituentBenchmark`` / ``PublishedIndex``: vendor constituents and weights per reconstitution
  date (effective after that date's close), constituent returns from a panel, weights that drift
  with returns between reconstitutions or stay static (``WeightDrift``), membership, removals,
  aggregate exposures (OASD, DTS, key rates) and a reconstruction check against published returns.
* ``CustomBenchmark``: rules applied to a parent at each of its reconstitutions - attribute and
  field filters (sector, rating, maturity ...), issuer caps, single-name caps, reweighting.
* ``CompositeBenchmark``: weighted blend of benchmarks, rebalanced on a calendar.
* ``ReturnSeriesBenchmark``: only an aggregate return series (weights are unavailable and anything
  needing them raises ``BenchmarkWeightsUnavailable``).

Row ``t`` of a weight matrix holds close-of-day weights: they earn the period ending ``t + 1``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from nautilus_fi.data.panel import Panel
from nautilus_fi.domain.calendars import RebalanceCalendar, parse_calendar
from nautilus_fi.domain.conventions import WeightDrift


class BenchmarkWeightsUnavailable(ValueError):
    pass


def cap_weights(weights: pd.Series, groups: pd.Series | None, cap: float, max_iter: int = 200, tol: float = 1e-12) -> pd.Series:
    """Cap each group's total weight at ``cap`` and redistribute the excess pro rata to the uncapped
    groups (index issuer-capping). ``groups=None`` caps single names."""
    w = weights.astype(float).clip(lower=0.0)
    if w.sum() <= 0:
        return w
    w = w / w.sum()
    g = groups.reindex(w.index) if groups is not None else pd.Series(w.index, index=w.index)
    n_groups = g.nunique()
    if cap * n_groups < 1.0 - 1e-9:
        raise ValueError(f"cap {cap} infeasible for {n_groups} groups (cap x groups < 1)")
    capped: set = set()
    for _ in range(max_iter):
        totals = w.groupby(g).sum()
        over = totals[totals > cap + tol]
        if over.empty:
            break
        for grp, tot in over.items():
            mask = g == grp
            w[mask] *= cap / tot
            capped.add(grp)
        free = ~g.isin(capped)
        excess = 1.0 - w.sum()
        if free.any() and w[free].sum() > 0:
            w[free] += excess * w[free] / w[free].sum()
        else:
            break
    return w / w.sum()


class Benchmark:
    """Interface. Subclasses define ``name`` (and ``drift`` where weights drift)."""

    # -------------------------------------------------------------- interface
    def weight_matrix(self, panel: Panel, return_field: str = "tr") -> np.ndarray:
        raise BenchmarkWeightsUnavailable(f"{self.name}: constituent weights are not available")

    def returns(self, panel: Panel, return_field: str = "tr") -> pd.Series:
        w = self.weight_matrix(panel, return_field)
        r = np.nan_to_num(panel[return_field].astype(np.float64))
        out = np.zeros(len(panel.dates))
        out[1:] = (w[:-1] * r[1:]).sum(axis=1)
        return pd.Series(out, index=panel.dates, name=self.name)

    def weights_at(self, panel: Panel, d, return_field: str = "tr") -> pd.Series:
        i = panel.date_pos(d)
        w = self._cached(panel, return_field)[i]
        nz = np.nonzero(w)[0]
        return pd.Series(w[nz], index=panel.ids[nz], name=self.name)

    def members(self, panel: Panel, d, return_field: str = "tr") -> set[str]:
        return set(self.weights_at(panel, d, return_field).index)

    def removals(self, panel: Panel, d_prev, d, return_field: str = "tr") -> set[str]:
        return self.members(panel, d_prev, return_field) - self.members(panel, d, return_field)

    def exposures(self, panel: Panel, d, fields: list[str], return_field: str = "tr") -> dict[str, float]:
        i = panel.date_pos(d)
        w = self._cached(panel, return_field)[i]
        return {f: float(np.nansum(w * panel.asof_row(f, i))) for f in fields if f in panel}

    def exposure_series(self, panel: Panel, fields: list[str], return_field: str = "tr") -> pd.DataFrame:
        w = self._cached(panel, return_field)
        return pd.DataFrame({f: np.nansum(w * np.nan_to_num(panel[f].astype(np.float64)), axis=1) for f in fields if f in panel}, index=panel.dates)

    def _cached(self, panel: Panel, return_field: str) -> np.ndarray:
        key = (id(panel), return_field, len(panel.dates), len(panel.ids))
        cache = getattr(self, "_wcache", None)
        if cache is None or cache[0] != key:
            self._wcache = (key, self.weight_matrix(panel, return_field))
        return self._wcache[1]


@dataclass
class ConstituentBenchmark(Benchmark):
    """``constituents``: columns date (reconstitution date; weights effective after its close),
    instrument_id, weight. Missing members in the panel are dropped and weights renormalised."""

    name: str
    constituents: pd.DataFrame
    drift: WeightDrift = WeightDrift.DRIFT
    published_returns: pd.Series | None = None

    def __post_init__(self) -> None:
        c = self.constituents.copy()
        c["date"] = pd.to_datetime(c["date"]).dt.normalize()
        c["instrument_id"] = c["instrument_id"].astype(str)
        self.constituents = c
        self._recon = {d: g.set_index("instrument_id")["weight"].astype(float) for d, g in c.groupby("date")}

    def reconstitutions(self) -> list[pd.Timestamp]:
        return sorted(self._recon)

    def reconstitution_weights(self, d) -> pd.Series:
        return self._recon[pd.Timestamp(d).normalize()]

    def _recon_vector(self, panel: Panel, w: pd.Series) -> np.ndarray:
        v = np.zeros(len(panel.ids))
        pos = panel.id_pos(w.index)
        ok = pos >= 0
        v[pos[ok]] = w.to_numpy()[ok]
        return v

    def weight_matrix(self, panel: Panel, return_field: str = "tr") -> np.ndarray:
        T, N = len(panel.dates), len(panel.ids)
        out = np.zeros((T, N))
        r = panel.get(return_field)
        alive = panel["alive"] > 0 if "alive" in panel else (np.isfinite(r) if r is not None else np.ones((T, N), dtype=bool))
        recon_at = {}
        before = [d for d in self._recon if d < panel.dates[0]]
        if before:                                        # the latest reconstitution before the window holds on day one
            recon_at[0] = self._recon_vector(panel, self._recon[max(before)])
        for d, w in self._recon.items():
            i = panel.date_pos(d)
            if 0 <= i < T and panel.dates[i] == d:
                recon_at[i] = self._recon_vector(panel, w)
            elif i >= 0:                                  # reconstitution on a non-panel date: effective from the previous panel date
                recon_at.setdefault(i, self._recon_vector(panel, w))
        anchor = None
        current = np.zeros(N)
        for t in range(T):
            if t > 0 and current.any():
                if self.drift == WeightDrift.DRIFT and r is not None:
                    current = current * (1.0 + np.nan_to_num(r[t].astype(np.float64)))
                else:
                    current = anchor.copy() if anchor is not None else current
                current = np.where(alive[t], current, 0.0)
                s = current.sum()
                current = current / s if s > 0 else current
            if t in recon_at:
                current = np.where(alive[t + 1] if t + 1 < T else alive[t], recon_at[t], 0.0)
                s = current.sum()
                current = current / s if s > 0 else current
                anchor = current.copy()
            elif anchor is not None and self.drift == WeightDrift.STATIC:
                a = np.where(alive[t], anchor, 0.0)
                s = a.sum()
                current = a / s if s > 0 else a
            out[t] = current
        return out

    def reconstruction_check(self, panel: Panel, return_field: str = "tr") -> pd.DataFrame:
        """Reconstructed vs published index returns (daily difference, cumulative gap)."""
        if self.published_returns is None:
            raise ValueError(f"{self.name}: no published returns to check against")
        rec = self.returns(panel, return_field)
        pub = pd.Series(self.published_returns).reindex(rec.index)
        diff = rec - pub
        return pd.DataFrame({"reconstructed": rec, "published": pub, "difference": diff, "cumulative_gap": (1 + rec).cumprod() - (1 + pub.fillna(0)).cumprod()})


PublishedIndex = ConstituentBenchmark


@dataclass
class CustomBenchmark(Benchmark):
    """Rules on a parent benchmark, re-applied at each parent reconstitution.

    ``include`` / ``exclude``: ``{attribute: [values]}`` on static ``attributes``; ``field_rules``:
    ``[(field, op, value)]`` on panel fields at the reconstitution date (``("rating_score", "<=", 10)``,
    ``("years_to_maturity", ">=", 3)``); ``predicate``: any callable on the reconstitution cross-section
    (attributes joined with panel fields) returning a boolean mask; ``issuer_cap`` / ``max_weight``:
    caps with pro-rata redistribution; ``reweight``: ``"parent"`` (renormalise parent weights) or
    ``"equal"``."""

    name: str
    parent: ConstituentBenchmark
    attributes: pd.DataFrame | None = None
    include: dict = field(default_factory=dict)
    exclude: dict = field(default_factory=dict)
    field_rules: list = field(default_factory=list)
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None
    issuer_cap: float | None = None
    issuer_col: str = "issuer"
    max_weight: float | None = None
    reweight: str = "parent"
    drift: WeightDrift = WeightDrift.DRIFT

    _OPS: ClassVar[dict] = {"<=": np.less_equal, "<": np.less, ">=": np.greater_equal, ">": np.greater, "==": np.equal, "!=": np.not_equal}

    def rules(self) -> dict:
        return {"parent": self.parent.name, "include": self.include, "exclude": self.exclude, "field_rules": self.field_rules, "predicate": getattr(self.predicate, "__name__", None), "issuer_cap": self.issuer_cap, "max_weight": self.max_weight, "reweight": self.reweight, "drift": self.drift.value}

    def _apply(self, panel: Panel, d, w: pd.Series) -> pd.Series:
        cs = pd.DataFrame(index=w.index)
        if self.attributes is not None:
            cs = cs.join(self.attributes, how="left")
        i = panel.date_pos(d)
        pos = panel.id_pos(w.index)
        for f in {fr[0] for fr in self.field_rules}:
            row = panel.asof_row(f, i) if f in panel else np.full(len(panel.ids), np.nan)
            cs[f] = np.where(pos >= 0, row[np.clip(pos, 0, None)], np.nan)
        keep = pd.Series(True, index=w.index)
        for col, vals in self.include.items():
            keep &= cs[col].isin(vals)
        for col, vals in self.exclude.items():
            keep &= ~cs[col].isin(vals)
        for f, op, v in self.field_rules:
            keep &= pd.Series(self._OPS[op](cs[f].to_numpy(dtype=float), v), index=w.index).fillna(False)
        if self.predicate is not None:
            keep &= self.predicate(cs).reindex(w.index).fillna(False).astype(bool)
        out = w[keep]
        if self.reweight == "equal" and len(out):
            out = pd.Series(1.0 / len(out), index=out.index)
        out = out / out.sum() if out.sum() > 0 else out
        if self.issuer_cap is not None and len(out):
            out = cap_weights(out, cs.loc[out.index, self.issuer_col] if self.issuer_col in cs else None, self.issuer_cap)
        if self.max_weight is not None and len(out):
            out = cap_weights(out, None, self.max_weight)
        return out

    def constituents_frame(self, panel: Panel) -> pd.DataFrame:
        rows = []
        for d in self.parent.reconstitutions():
            w = self._apply(panel, d, self.parent.reconstitution_weights(d))
            rows.append(pd.DataFrame({"date": d, "instrument_id": w.index, "weight": w.to_numpy()}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "instrument_id", "weight"])

    def weight_matrix(self, panel: Panel, return_field: str = "tr") -> np.ndarray:
        return ConstituentBenchmark(self.name, self.constituents_frame(panel), self.drift).weight_matrix(panel, return_field)


@dataclass
class ReturnSeriesBenchmark(Benchmark):
    name: str
    series: pd.Series

    def returns(self, panel: Panel | None = None, return_field: str = "tr") -> pd.Series:
        s = pd.Series(self.series, dtype=float)
        s.index = pd.to_datetime(s.index)
        return s.reindex(panel.dates) if panel is not None else s


@dataclass
class CompositeBenchmark(Benchmark):
    """Blend of benchmarks with target weights, reset on ``rebalance`` dates and drifting between."""

    name: str
    components: list[tuple[Benchmark, float]]
    rebalance: RebalanceCalendar | str = "BME"

    def _comp_weights(self, panel: Panel, return_field: str) -> tuple[np.ndarray, list[pd.Series]]:
        cal = parse_calendar(self.rebalance)
        rets = [b.returns(panel, return_field).reindex(panel.dates).fillna(0.0) for b, _ in self.components]
        target = np.array([w for _, w in self.components], dtype=float)
        target = target / target.sum()
        days = list(panel.dates.date)
        due = set(cal.dates(days, first_decision=True))
        c = np.zeros((len(days), len(self.components)))
        cur = target.copy()
        for t, d in enumerate(days):
            if t > 0:
                cur = cur * (1.0 + np.array([r.iloc[t] for r in rets]))
                cur = cur / cur.sum()
            if d in due:
                cur = target.copy()
            c[t] = cur
        return c, rets

    def returns(self, panel: Panel, return_field: str = "tr") -> pd.Series:
        c, rets = self._comp_weights(panel, return_field)
        R = np.column_stack([r.to_numpy() for r in rets])
        out = np.zeros(len(panel.dates))
        out[1:] = (c[:-1] * R[1:]).sum(axis=1)
        return pd.Series(out, index=panel.dates, name=self.name)

    def weight_matrix(self, panel: Panel, return_field: str = "tr") -> np.ndarray:
        c, _ = self._comp_weights(panel, return_field)
        out = np.zeros((len(panel.dates), len(panel.ids)))
        for k, (b, _) in enumerate(self.components):
            out += c[:, [k]] * b.weight_matrix(panel, return_field)
        return out


def benchmark_from_panel_weights(name: str, panel: Panel, weight_field: str = "index_weight", reconstitution_dates: list | None = None, drift: WeightDrift = WeightDrift.DRIFT) -> ConstituentBenchmark:
    """Build a constituent benchmark from a daily weight field, keeping the reconstitution rows
    (dates where membership changes, or the given dates)."""
    W = panel[weight_field]
    if reconstitution_dates is None:
        member = W > 0
        change = np.concatenate([[True], (member[1:] != member[:-1]).any(axis=1)])
        idx = np.nonzero(change)[0]
    else:
        idx = [panel.date_pos(d) for d in reconstitution_dates]
    rows = []
    for i in idx:
        nz = np.nonzero(W[i] > 0)[0]
        rows.append(pd.DataFrame({"date": panel.dates[i], "instrument_id": panel.ids[nz], "weight": W[i][nz]}))
    return ConstituentBenchmark(name, pd.concat(rows, ignore_index=True), drift)


__all__ = ["Benchmark", "BenchmarkWeightsUnavailable", "CompositeBenchmark", "ConstituentBenchmark", "CustomBenchmark", "PublishedIndex", "ReturnSeriesBenchmark", "benchmark_from_panel_weights", "cap_weights"]
