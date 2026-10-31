"""Construction methods and the construction spec shared by the panel and event engines.

A method turns a ``CrossSection`` (scores, attributes, exposures, benchmark and current weights)
into target weights. ``ConstructionSpec`` adds the constraint grammar, an optional derivative
overlay sized against the resulting exposures, and the post-check that records violations for
every method (the optimiser enforces constraints; heuristics are checked against them).

Methods: ``EqualWeight``, ``Quantile`` (bins, optionally within groups, long-only or long-short),
``RankWeights``, ``BenchmarkPlusTilts``, ``PairsSizing`` (notional, DV01, OASD or DTS neutral),
``Optimise`` (LP / MILP / QP via ``construct.optimiser``). Overlay: ``HedgeOverlay`` (DV01 / KRD /
DTS hedges with futures, CDX or swaps by least squares on exposures per unit notional).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.benchmarks.benchmark import cap_weights
from nautilus_fi.construct.cross_section import CrossSection
from nautilus_fi.construct.grammar import ConstraintSpec, check_constraints, parse_constraints
from nautilus_fi.construct.optimiser import Objective, OptimiserSettings, optimise
from nautilus_fi.domain.conventions import turnover_all


@dataclass
class ConstructionOutput:
    weights: pd.Series
    overlays: pd.Series | None
    diagnostics: dict
    violations: list[dict]
    ok: bool = True


class ConstructionMethod:
    name = "method"

    def weights(self, cs: CrossSection, spec: ConstructionSpec) -> tuple[pd.Series, dict]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"method": self.name, **{k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, str, bool, type(None), tuple, list))}}


def _scores(cs: CrossSection) -> pd.Series:
    if cs.scores is None:
        raise ValueError("this construction method needs scores")
    return cs.scores[cs.eligible].dropna()


@dataclass
class EqualWeight(ConstructionMethod):
    top: int | None = None
    invested: float = 1.0
    name: str = "EqualWeight"

    def weights(self, cs, spec):
        if cs.scores is not None:
            pick = _scores(cs).sort_values(ascending=False)
            pick = pick.index[: self.top] if self.top else pick.index
        else:
            pick = cs.ids[cs.eligible.to_numpy()]
            pick = pick[: self.top] if self.top else pick
        w = pd.Series(self.invested / len(pick), index=pick) if len(pick) else pd.Series(dtype=float)
        return w, {"n_selected": len(pick)}


@dataclass
class Quantile(ConstructionMethod):
    """Hold bin ``long_bin`` (default: the top) and short ``short_bin`` (default none) of ``n_bins``
    score bins; ``by`` bins within groups (sector-neutral quintiles); ``weighting``: ``equal``,
    ``benchmark`` (benchmark weights inside the bin) or a field name (e.g. market value)."""

    n_bins: int = 5
    long_bin: int | None = None
    short_bin: int | None = None
    by: str | None = None
    weighting: str = "equal"
    invested: float = 1.0
    short_gross: float | None = None
    name: str = "Quantile"

    def bins(self, cs: CrossSection) -> pd.Series:
        sc = _scores(cs)
        if self.by:
            g = cs.data.loc[sc.index, self.by].astype(str)
            pct = sc.groupby(g).rank(pct=True, method="first")
        else:
            pct = sc.rank(pct=True, method="first")
        return np.ceil(pct * self.n_bins).clip(1, self.n_bins).astype(int)

    def _leg(self, cs, members: pd.Index, gross: float) -> pd.Series:
        if not len(members):
            return pd.Series(dtype=float)
        if self.weighting == "equal":
            raw = pd.Series(1.0, index=members)
        elif self.weighting == "benchmark":
            raw = cs.benchmark.reindex(members).fillna(0.0)
            raw = raw if raw.sum() > 0 else pd.Series(1.0, index=members)
        else:
            raw = cs.data.loc[members, self.weighting].astype(float).clip(lower=0.0).fillna(0.0)
        return raw / raw.sum() * gross if raw.sum() > 0 else raw

    def weights(self, cs, spec):
        b = self.bins(cs)
        lb = self.long_bin or self.n_bins
        w = self._leg(cs, b.index[b == lb], self.invested)
        if self.short_bin is not None:
            short = self._leg(cs, b.index[b == self.short_bin], self.short_gross if self.short_gross is not None else self.invested)
            w = w.sub(short, fill_value=0.0)
        return w, {"bin_counts": b.value_counts().sort_index().to_dict()}


@dataclass
class RankWeights(ConstructionMethod):
    power: float = 1.0
    long_short: bool = False
    top: int | None = None
    invested: float = 1.0
    name: str = "RankWeights"

    def weights(self, cs, spec):
        sc = _scores(cs).sort_values(ascending=False)
        if self.top:
            sc = sc.iloc[: self.top]
        r = sc.rank(pct=True)
        if self.long_short:
            v = (r - r.mean()) * 2.0
            v = np.sign(v) * np.abs(v) ** self.power
            w = v / v.abs().sum() * 2.0 * self.invested if v.abs().sum() else v
        else:
            v = r ** self.power
            w = v / v.sum() * self.invested
        return w, {}


@dataclass
class BenchmarkPlusTilts(ConstructionMethod):
    """Benchmark weights scaled by ``1 + strength x z(score)``, active weights clipped to
    ``max_active`` and renormalised (z-scores within ``by`` groups when given)."""

    strength: float = 0.5
    max_active: float | None = None
    by: str | None = None
    name: str = "BenchmarkPlusTilts"

    def weights(self, cs, spec):
        from nautilus_fi.construct.signals import zscore

        wb = cs.benchmark[cs.benchmark > 0]
        sc = cs.scores.reindex(wb.index) if cs.scores is not None else pd.Series(0.0, index=wb.index)
        z = zscore(sc.fillna(sc.mean() if sc.notna().any() else 0.0), cs.data.loc[wb.index, self.by] if self.by else None).fillna(0.0)
        w = (wb * (1.0 + self.strength * z)).clip(lower=0.0)
        w = w / w.sum() * wb.sum() if w.sum() > 0 else wb
        if self.max_active is not None:
            for _ in range(20):
                active = (w - wb).clip(-self.max_active, self.max_active)
                w2 = (wb + active).clip(lower=0.0)
                w2 = w2 / w2.sum() * wb.sum()
                if np.allclose(w2, w):
                    break
                w = w2
        return w, {"tilt_strength": self.strength}


@dataclass
class PairsSizing(ConstructionMethod):
    """Long-short pairs sized so the short leg neutralises the long leg's ``exposure`` (``notional``,
    ``oasd``, ``oad``, ``dts``, ``dv01`` ...); ``gross_per_pair`` of long notional per pair."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    exposure: str = "dts"
    gross_per_pair: float = 0.1
    name: str = "PairsSizing"

    def weights(self, cs, spec):
        w = pd.Series(0.0, index=cs.ids)
        used = []
        for long_, short_ in self.pairs:
            if long_ not in w.index or short_ not in w.index:
                continue
            if self.exposure == "notional":
                ratio = 1.0
            else:
                xl, xs = cs.data.at[long_, self.exposure], cs.data.at[short_, self.exposure]
                if not (np.isfinite(xl) and np.isfinite(xs)) or xs == 0:
                    continue
                ratio = float(xl / xs)
            w[long_] += self.gross_per_pair
            w[short_] -= self.gross_per_pair * ratio
            used.append((long_, short_, ratio))
        return w[w != 0.0], {"pairs_used": len(used), "hedge_ratios": [u[2] for u in used]}


@dataclass
class Optimise(ConstructionMethod):
    objective: Objective = field(default_factory=Objective)
    settings: OptimiserSettings = field(default_factory=OptimiserSettings)
    name: str = "Optimise"

    def weights(self, cs, spec):
        res = optimise(cs, self.objective, spec.constraints, spec.long_only, self.settings)
        return res.weights, {"optimiser": res.summary(), "binding": res.binding[:50], "duals": dict(list(res.duals.items())[:50]), "relaxations": res.relaxations, "ok": res.ok, "message": res.message}

    def to_dict(self) -> dict:
        return {"method": self.name, "objective": self.objective.to_dict(), "time_limit": self.settings.time_limit, "mip_rel_gap": self.settings.mip_rel_gap}


@dataclass
class HedgeOverlay:
    """Overlay notionals (fractions of NAV) bringing the portfolio's ``fields`` exposures to
    ``targets`` (absolute, or relative to the benchmark when ``active=True``) with ``instruments``
    whose exposures per unit notional come from ``overlay_data`` (rows: instruments). Least squares
    when instruments and fields differ in number."""

    instruments: tuple[str, ...] = ("TY",)
    fields: tuple[str, ...] = ("oad",)
    targets: dict = field(default_factory=dict)
    active: bool = False
    max_gross: float | None = None

    def size(self, weights: pd.Series, cs: CrossSection, overlay_data: pd.DataFrame) -> tuple[pd.Series, dict]:
        w = weights.reindex(cs.ids).fillna(0.0)
        port = {f: float((w * cs.data[f].reindex(cs.ids).astype(float).fillna(0.0)).sum()) for f in self.fields}
        bench = {f: float((cs.benchmark * cs.data[f].reindex(cs.ids).astype(float).fillna(0.0)).sum()) for f in self.fields}
        tgt = np.array([self.targets.get(f, 0.0) + (bench[f] if self.active else 0.0) for f in self.fields])
        gap = tgt - np.array([port[f] for f in self.fields])
        E = overlay_data.reindex(list(self.instruments))[list(self.fields)].astype(float).fillna(0.0).to_numpy().T      # fields x instruments
        notional, *_ = np.linalg.lstsq(E, gap, rcond=None)
        out = pd.Series(notional, index=list(self.instruments))
        if self.max_gross is not None and out.abs().sum() > self.max_gross:
            out *= self.max_gross / out.abs().sum()
        after = {f: port[f] + float(E[i] @ out.to_numpy()) for i, f in enumerate(self.fields)}
        return out, {"exposure_before": port, "exposure_after": after, "target": dict(zip(self.fields, tgt.tolist()))}


@dataclass
class ConstructionSpec:
    method: ConstructionMethod
    constraints: list = field(default_factory=list)
    long_only: bool = True
    overlay: HedgeOverlay | None = None
    issuer_cap: float | None = None            # heuristic methods: cap issuers after weighting
    issuer_col: str = "issuer"
    name: str = ""

    def __post_init__(self) -> None:
        self.constraints = parse_constraints(self.constraints)

    def to_dict(self) -> dict:
        return {"name": self.name, "method": self.method.to_dict(), "constraints": [c.to_dict() for c in self.constraints], "long_only": self.long_only, "issuer_cap": self.issuer_cap, "overlay": None if self.overlay is None else {"instruments": list(self.overlay.instruments), "fields": list(self.overlay.fields), "targets": self.overlay.targets, "active": self.overlay.active}}

    def build(self, cs: CrossSection, overlay_data: pd.DataFrame | None = None) -> ConstructionOutput:
        w, diag = self.method.weights(cs, self)
        w = pd.Series(w, dtype=float)
        ok = bool(diag.get("ok", True))
        if self.issuer_cap is not None and not isinstance(self.method, Optimise) and len(w) and (w > 0).all() and self.issuer_col in cs.data:
            total = float(w.sum())
            w = cap_weights(w, cs.data.loc[w.index, self.issuer_col], self.issuer_cap) * total
        w = w[w.abs() > 1e-12]
        overlays = None
        if self.overlay is not None:
            if overlay_data is None:
                raise ValueError("an overlay needs overlay_data (instrument exposures per unit notional)")
            overlays, odiag = self.overlay.size(w, cs, overlay_data)
            diag["overlay"] = odiag
        initial = float(cs.current.abs().sum()) < 1e-9                  # building from cash: turnover limits do not apply
        violations = check_constraints(w, cs, [c for c in self.constraints if (c.kind != "cvar" or cs.scenarios is not None) and not (initial and c.kind == "turnover")])
        diag.update({"method": self.method.name, "n_names": int((w.abs() > 0).sum()), "gross": float(w.abs().sum()), "net": float(w.sum()), **{f"turnover_{k}": v for k, v in turnover_all(cs.current[cs.current != 0], w).items()}, "n_violations": len(violations), "n_hard_violations": sum(1 for v in violations if v["severity"] == "HARD")})
        return ConstructionOutput(w, overlays, diag, violations, ok)


__all__ = ["BenchmarkPlusTilts", "ConstraintSpec", "ConstructionMethod", "ConstructionOutput", "ConstructionSpec", "EqualWeight", "HedgeOverlay", "Optimise", "PairsSizing", "Quantile", "RankWeights"]
