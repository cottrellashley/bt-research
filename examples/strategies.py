"""Example strategies built on ``FixedIncomeStrategy``.

They are deliberately simple: the point of the examples is the execution / accounting realism, not
alpha. Both work in face amounts and conventional prices and never touch future data.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from nautilus_fi.backtest.strategy import FixedIncomeStrategy, FixedIncomeStrategyConfig
from nautilus_fi.instruments.enums import AssetType
from nautilus_fi.universe.universe import UniverseFilter


class EqualWeightCorporatesConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, invested_fraction: float = 0.92, max_names: int = 25, min_years: float = 1.5, max_years: float = 12.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.invested_fraction = invested_fraction
        self.max_names = max_names
        self.min_years = min_years
        self.max_years = max_years


class EqualWeightCorporates(FixedIncomeStrategy):
    """Monthly equal-weight investment-grade corporate portfolio (a plumbing baseline)."""

    def on_decision(self, d: date) -> None:
        cfg: EqualWeightCorporatesConfig = self.fi_config
        members = self.universe(UniverseFilter(asset_types={AssetType.CORPORATE}, max_rating_score=10, min_years_to_maturity=cfg.min_years, max_years_to_maturity=cfg.max_years))
        members = sorted(members, key=lambda b: (b.amount_outstanding or 0), reverse=True)[: cfg.max_names]
        if not members:
            return
        w = cfg.invested_fraction / len(members)
        self.rebalance_to_weights({b.symbol: w for b in members})


class CarryValueConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, n_long: int = 10, invested_fraction: float = 0.9, min_years: float = 2.0, max_years: float = 12.0, max_rating_score: int = 10, min_quality: float = 0.5, protocol: str = "RFQ", **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_long = n_long
        self.invested_fraction = invested_fraction
        self.min_years = min_years
        self.max_years = max_years
        self.max_rating_score = max_rating_score
        self.min_quality = min_quality
        self.protocol = protocol


class CarryValueStrategy(FixedIncomeStrategy):
    """Buy the cheapest corporates versus their sector/rating peers (spread residual), equal weight.

    Signals use only received marks (point-in-time). Bonds whose latest mark is a model or stale price
    below ``min_quality`` are excluded from the *signal* to avoid evaluated-price circularity, and the
    return is measured from the next feasible execution (orders rest until the t+1 window).
    """

    def on_decision(self, d: date) -> None:
        cfg: CarryValueConfig = self.fi_config
        members = self.universe(UniverseFilter(asset_types={AssetType.CORPORATE}, max_rating_score=cfg.max_rating_score, min_years_to_maturity=cfg.min_years, max_years_to_maturity=cfg.max_years))
        rows = []
        for b in members:
            mk = self.mark(b.symbol)
            if mk is None or mk.quality < cfg.min_quality or not np.isfinite(mk.z_spread_bp) or not np.isfinite(mk.mod_duration):
                continue
            score = self.rating_score(b.symbol) or 10
            rows.append((b.symbol, mk.z_spread_bp, mk.mod_duration, f"{b.sector}|{min(score, 10) // 4}", mk.ytm))
        if len(rows) < 3:
            return
        # residual spread vs peer group median, lightly duration adjusted
        groups: dict[str, list[float]] = {}
        for sym, z, dur, grp, _ in rows:
            groups.setdefault(grp, []).append(z / max(dur, 1.0))
        med = {g: float(np.median(v)) for g, v in groups.items()}
        ranked = sorted(rows, key=lambda r: r[1] / max(r[2], 1.0) - med[r[3]], reverse=True)
        picks = [r[0] for r in ranked[: cfg.n_long]]
        w = cfg.invested_fraction / len(picks)
        if cfg.log_decisions:
            self.log.info(f"[{d}] picks: {picks}")
        self.rebalance_to_weights({s: w for s in picks}, protocol=cfg.protocol)


class LadderTreasuriesConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, invested_fraction: float = 0.95, **kwargs) -> None:
        super().__init__(**kwargs)
        self.invested_fraction = invested_fraction


class LadderTreasuries(FixedIncomeStrategy):
    """Equal-weight ladder of Treasuries, bills and linkers (government-only sleeve)."""

    def on_decision(self, d: date) -> None:
        cfg: LadderTreasuriesConfig = self.fi_config
        members = self.universe(UniverseFilter(asset_types={AssetType.SOVEREIGN, AssetType.BILL, AssetType.LINKER}))
        if not members:
            return
        w = cfg.invested_fraction / len(members)
        self.rebalance_to_weights({b.symbol: w for b in members})


class EqualWeightEverythingConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, invested_fraction: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        self.invested_fraction = invested_fraction


class EqualWeightEverything(FixedIncomeStrategy):
    """Equal weight across every alive bond of every asset type: exercises floaters, amortizers,
    linkers, bills, callables and defaults in the ledger."""

    def on_decision(self, d: date) -> None:
        cfg: EqualWeightEverythingConfig = self.fi_config
        members = self.universe(UniverseFilter(min_years_to_maturity=0.1))
        if not members:
            return
        w = cfg.invested_fraction / len(members)
        self.rebalance_to_weights({b.symbol: w for b in members})


class BuyAndHoldConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, targets: dict[str, float] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.targets = targets or {}


class BuyAndHold(FixedIncomeStrategy):
    """Buy fixed face amounts on the first decision and hold to redemption (ledger test harness)."""

    def on_decision(self, d: date) -> None:
        if getattr(self, "_done", False):
            return
        self._done = True
        for symbol, face in self.fi_config.targets.items():
            self.order_face(symbol, face)


__all__ = ["BuyAndHold", "BuyAndHoldConfig", "CarryValueConfig", "CarryValueStrategy", "EqualWeightCorporates", "EqualWeightCorporatesConfig", "EqualWeightEverything", "EqualWeightEverythingConfig", "LadderTreasuries", "LadderTreasuriesConfig"]
