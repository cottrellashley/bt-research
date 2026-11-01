"""Run a ``ResearchSpec`` inside the event engine.

``SpecStrategy`` is a ``FixedIncomeStrategy`` whose decisions come from the spec's ``Decider`` and
``ConstructionSpec`` - the same objects the panel engine runs - through a ``ContextView`` built only
from the restricted ``DecisionContext``: generic fields published by the environment actor (vendor
analytics, signals, benchmark weights), mark-derived fields (OAS, OAD, OASD, DTS, prices, half
spreads), positions and point-in-time ratings. Orders then go through the portfolio constructor,
the execution simulator, settlement, funding and the ledger like any other strategy; overlay
notionals go to the margin venue when the run has one.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from nautilus_fi.backtest.strategy import FixedIncomeStrategy, FixedIncomeStrategyConfig
from nautilus_fi.construct.cross_section import CrossSection
from nautilus_fi.domain.spec import DecisionView, ResearchSpec, as_target
from nautilus_fi.universe.universe import UniverseFilter

MARK_FIELDS = ("oas_bp", "oad", "oasd", "dts", "ytm", "price_clean", "price_dirty", "accrued", "half_spread_bp", "factor", "quality", "rating_score", "years_to_maturity", "amt_out")


class ContextView(DecisionView):
    def __init__(self, strategy: FixedIncomeStrategy, universe_from_benchmark: bool = True) -> None:
        self._s = strategy
        ctx = strategy.ctx
        self.date = pd.Timestamp(strategy.today)
        self.nav = float(strategy.nav() or 0.0)
        bw = strategy.benchmark_weights()
        self._bench = pd.Series(bw, dtype=float)
        if universe_from_benchmark and bw:
            members = [sym for sym in bw if ctx.mark(sym) is not None and ctx.can_trade(sym)[0]]
            self._universe = pd.Index(sorted(members))
        else:
            self._universe = pd.Index(sorted(b.symbol for b in strategy.universe(UniverseFilter())))
        nav = self.nav or 1.0
        self._holdings = pd.Series({sym: face * (strategy.dirty_value_per_face(sym) or 0.0) / nav for sym, face in strategy.holdings_face().items()}, dtype=float)
        bonds = strategy.services.bonds
        ids = self.ids
        self._attrs = pd.DataFrame({"issuer": [bonds[i].issuer if i in bonds else None for i in ids], "sector": [bonds[i].sector if i in bonds else None for i in ids], "country": [bonds[i].country if i in bonds else None for i in ids], "currency": [bonds[i].currency if i in bonds else None for i in ids], "seniority": [bonds[i].seniority.value if i in bonds else None for i in ids], "asset_type": [bonds[i].asset_type.value if i in bonds else None for i in ids], "coupon": [bonds[i].coupon_rate if i in bonds else np.nan for i in ids]}, index=ids)

    @property
    def universe(self) -> pd.Index:
        return self._universe

    @property
    def holdings(self) -> pd.Series:
        return self._holdings

    @property
    def benchmark(self) -> pd.Series:
        return self._bench

    @property
    def attributes(self) -> pd.DataFrame:
        return self._attrs

    def has(self, name: str) -> bool:
        return name in MARK_FIELDS or self._s.ctx.field_row(name) is not None or name in self._s.ctx.field_names()

    def field(self, name: str) -> pd.Series:
        ctx = self._s.ctx
        ids = self.ids
        row = ctx.field_row(name)
        if row is not None:
            pid, vals = row
            pos = pid.get_indexer(ids)
            out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
            return pd.Series(out.astype(float), index=ids, name=name)
        if name in ctx.field_names():
            sl = ctx.field_slice(name)
            return pd.Series([sl.get(i, np.nan) for i in ids], index=ids, dtype=float, name=name)
        if name not in MARK_FIELDS:
            raise KeyError(f"field {name!r} is not published to the context")
        bonds = self._s.services.bonds
        vals = []
        for i in ids:
            mk = ctx.mark(i)
            if name == "rating_score":
                vals.append(self._s.rating_score(i))
                continue
            if name == "years_to_maturity":
                b = bonds.get(i)
                vals.append(b.remaining_years(self._s.today) if b is not None else np.nan)
                continue
            if name == "amt_out":
                b = bonds.get(i)
                vals.append(b.amount_outstanding if b is not None else np.nan)
                continue
            if mk is None:
                vals.append(np.nan)
                continue
            if name == "oas_bp":
                vals.append(mk.z_spread_bp)
            elif name == "oad":
                vals.append(mk.mod_duration)
            elif name == "oasd":
                vals.append(mk.spread_duration)
            elif name == "dts":
                vals.append(mk.spread_duration * mk.z_spread_bp)
            elif name == "ytm":
                vals.append(mk.ytm)
            elif name == "price_clean":
                vals.append(mk.clean_mid)
            elif name == "price_dirty":
                vals.append(mk.clean_mid + mk.accrued)
            elif name == "accrued":
                vals.append(mk.accrued)
            elif name == "half_spread_bp":
                vals.append((mk.clean_ask - mk.clean_bid) / 2.0 / mk.clean_mid * 1e4 if mk.clean_mid else np.nan)
            elif name == "factor":
                vals.append(mk.factor)
            elif name == "quality":
                vals.append(mk.quality)
        return pd.Series(vals, index=ids, dtype=float, name=name)


class SpecStrategyConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, fields: tuple[str, ...] | None = None, min_face: float | None = None, universe_from_benchmark: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fields = fields
        self.min_face = min_face
        self.universe_from_benchmark = universe_from_benchmark


class SpecStrategy(FixedIncomeStrategy):
    def __new__(cls, spec: ResearchSpec | None = None, config: SpecStrategyConfig | None = None, *args, **kwargs):  # the PyO3 base takes the config only
        return super().__new__(cls, config or SpecStrategyConfig(rebalance=spec.rebalance if spec is not None else "BMS"))

    def __init__(self, spec: ResearchSpec, config: SpecStrategyConfig | None = None) -> None:
        cfg = config or SpecStrategyConfig(rebalance=spec.rebalance)
        super().__init__(cfg)
        self.spec = spec
        self.spec_log: list[dict] = []

    def view(self) -> ContextView:
        return ContextView(self, self.fi_config.universe_from_benchmark)

    def on_decision(self, d: date) -> None:
        spec = self.spec
        view = self.view()
        if spec.universe is not None:
            view._universe = spec.universe.apply(view, view.universe)
        target = as_target(spec.decider.decide(view), spec.construction is not None)
        if target.skip:
            self.spec_log.append({"date": d, "action": "skip", "note": target.note})
            return
        overlays = target.overlays
        if target.weights is not None:
            weights = pd.Series(target.weights, dtype=float)
            diag = {"method": "Target.weights"}
        else:
            names = list(self.fi_config.fields or [f for f in MARK_FIELDS if f not in ("factor", "quality")])
            names += [f for f in self.ctx.field_names() if f not in names and not f.startswith("benchmark.")]
            data = view.fields(names).join(view.attributes, how="left")
            cs = CrossSection.build(view.date, view.universe, data, view.holdings, view.benchmark, target.scores, view.field("half_spread_bp"), self.fi_config.min_face, view.field("price_dirty"), view.nav)
            ov_data = self.overlay_exposures() if hasattr(self, "overlay_exposures") else None
            out = spec.construction.build(cs, ov_data if ov_data is not None and len(ov_data) else None)
            weights = out.weights
            diag = out.diagnostics
            if out.overlays is not None:
                overlays = out.overlays
            for v in out.violations:
                self.construction_log.append({"date": d, "stage": "spec", **v})
        self.spec_log.append({"date": d, "action": "rebalance", **{k: v for k, v in diag.items() if not isinstance(v, (dict, list))}})
        longs = {k: float(v) for k, v in weights.items() if v > 0}
        shorts = {k: float(v) for k, v in weights.items() if v < 0}
        self.rebalance_to_weights(longs)
        if shorts or overlays is not None:
            if not getattr(self.services, "margin_venue", None):
                self.log.warning("negative weights / overlays need a margin venue (BacktestSettings.margin); ignored")
                return
            book = {f"{k}{self.services.short_suffix}": v for k, v in shorts.items()}
            if overlays is not None:
                book.update({str(k): float(v) for k, v in overlays.items()})
            self.rebalance_overlays(book)


__all__ = ["MARK_FIELDS", "ContextView", "SpecStrategy", "SpecStrategyConfig"]
