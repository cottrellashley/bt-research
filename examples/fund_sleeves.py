"""Fund-style sleeves covering the product types a fixed income platform has to backtest.

Each sleeve is a plain ``FixedIncomeStrategy``: a point-in-time universe filter, a weighting rule and
(where the mandate needs it) typed constraints and a forced-action policy. They are archetypes of
common fund mandates (AAA CLO, mezzanine CLO, agency MBS, securitized income, ultrashort, IG
corporate, EM hard currency, multi-sector income, core-plus, short duration flexible, high yield,
developed-market hedged, money market), not models of any particular fund.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import ClassVar

from nautilus_fi.backtest import BacktestSettings, FixedIncomeStrategy, FixedIncomeStrategyConfig
from nautilus_fi.core.manifest import DeployabilityClaim
from nautilus_fi.instruments import AssetType
from nautilus_fi.instruments.enums import rating_to_score
from nautilus_fi.portfolio.constraints import Constraint, ConstraintKind
from nautilus_fi.portfolio.forced_actions import ForcedActionPolicy
from nautilus_fi.universe import UniverseFilter

S = rating_to_score
SECURITIZED = {AssetType.ABS, AssetType.CMBS, AssetType.CLO, AssetType.MBS}


def _tag(name: str) -> Callable[[object, date], bool]:
    return lambda b, d: bool(b.tags.get(name))


def _not_tag(name: str) -> Callable[[object, date], bool]:
    return lambda b, d: not b.tags.get(name)


@dataclass
class Bucket:
    name: str
    filt: UniverseFilter
    weight: float


class SleeveConfig(FixedIncomeStrategyConfig):
    def __init__(self, *, invested_fraction: float = 0.95, max_names: int | None = None, rank_by: str | None = None, buckets: list[Bucket] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.invested_fraction = invested_fraction
        self.max_names = max_names
        self.rank_by = rank_by
        self.buckets = list(buckets or [])


class BucketSleeve(FixedIncomeStrategy):
    """Equal weight within buckets, bucket weights renormalised over non-empty buckets."""

    BUCKETS: ClassVar[list[Bucket]] = []

    def __init__(self, config: SleeveConfig | None = None) -> None:   # Nautilus Strategy.__new__ accepts a single config argument
        super().__init__(config or SleeveConfig())
        self.sleeve_config: SleeveConfig = config or SleeveConfig()
        self.buckets = self.sleeve_config.buckets or self.BUCKETS

    def _rank(self, members):
        key = self.sleeve_config.rank_by
        if key is None:
            return members
        def score(b):
            mk = self.mark(b.symbol)
            v = getattr(mk, key, None) if mk else None
            return -(v if v is not None and not math.isnan(v) else -1e9)
        return sorted(members, key=score)

    def on_decision(self, d: date) -> None:
        weights: dict[str, float] = {}
        chosen: dict[str, list] = {}
        for bucket in self.buckets:
            members = self._rank(self.universe(bucket.filt))
            if self.sleeve_config.max_names:
                members = members[: self.sleeve_config.max_names]
            if members:
                chosen[bucket.name] = members
        total = sum(b.weight for b in self.buckets if b.name in chosen)
        if total <= 0:
            return
        for bucket in self.buckets:
            members = chosen.get(bucket.name)
            if not members:
                continue
            w = self.sleeve_config.invested_fraction * bucket.weight / total / len(members)
            for b in members:
                weights[b.symbol] = weights.get(b.symbol, 0.0) + w
        self.rebalance_to_weights(weights)


@dataclass
class SleeveSpec:
    name: str
    archetype: str
    buckets: list[Bucket]
    settings: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    asset_families: tuple[str, ...] = ()

    def make(self, strategy_id: str, rebalance: str = "M"):
        from nautilus_trader.model import StrategyId

        cfg = SleeveConfig(strategy_id=StrategyId(strategy_id), rebalance=rebalance, buckets=self.buckets, **self.config)
        return BucketSleeve(cfg)

    def backtest_settings(self, starting_cash: float = 100e6) -> BacktestSettings:
        return BacktestSettings(starting_cash=starting_cash, hypothesis=f"{self.archetype} sleeve: mandate-driven equal weight within buckets", owner="fund-sleeves-demo", deployability=DeployabilityClaim.RETROSPECTIVE_RECONSTRUCTION, **self.settings)


ISSUER_CAP = Constraint("issuer_cap_5pct", ConstraintKind.ISSUER_WEIGHT, "weight", maximum=0.05, hard=False)
NAME_CAP = Constraint("name_cap_10pct", ConstraintKind.CONCENTRATION, "weight", maximum=0.10, hard=False)
IG_FLOOR = Constraint("ig_floor", ConstraintKind.RATING_FLOOR, "score", maximum=S("BBB-"), hard=False)

SLEEVES: dict[str, SleeveSpec] = {
    "CLO_AAA": SleeveSpec("CLO_AAA", "AAA CLO ETF", [Bucket("clo_aaa", UniverseFilter(asset_types={AssetType.CLO}, max_rating_score=S("AA-")), 1.0)], {"constraints": [NAME_CAP]}, {"invested_fraction": 0.97}, ("CLO",)),
    "CLO_MEZZ": SleeveSpec("CLO_MEZZ", "B-BBB CLO ETF", [Bucket("clo_mezz", UniverseFilter(asset_types={AssetType.CLO}, min_rating_score=S("A-"), max_rating_score=S("B-")), 1.0)], {}, {"invested_fraction": 0.95}, ("CLO",)),
    "AGENCY_MBS": SleeveSpec("AGENCY_MBS", "Mortgage-backed securities ETF", [Bucket("mbs", UniverseFilter(asset_types={AssetType.MBS}), 1.0)], {}, {"invested_fraction": 0.96}, ("MBS",)),
    "SECURITIZED_INCOME": SleeveSpec("SECURITIZED_INCOME", "Securitized income (ABS / CMBS / CLO / MBS)", [Bucket("securitized", UniverseFilter(asset_types=SECURITIZED, max_rating_score=S("BBB-")), 1.0)], {"constraints": [NAME_CAP]}, {"invested_fraction": 0.95}, ("ABS", "CMBS", "CLO", "MBS")),
    "ULTRASHORT": SleeveSpec("ULTRASHORT", "Short duration income (ultrashort)", [Bucket("short_ig", UniverseFilter(asset_types={AssetType.CORPORATE, AssetType.SOVEREIGN, AssetType.BILL, AssetType.AGENCY}, currencies={"USD"}, max_years_to_maturity=2.5, max_rating_score=S("BBB-"), custom=_not_tag("em_hard_currency")), 1.0)], {"constraints": [Constraint("duration_cap_1y", ConstraintKind.DURATION, "years", maximum=1.0, hard=False)]}, {"invested_fraction": 0.97}, ("CORPORATE", "BILL", "SOVEREIGN")),
    "CORPORATE_IG": SleeveSpec("CORPORATE_IG", "Corporate bond ETF (investment grade)", [Bucket("ig", UniverseFilter(asset_types={AssetType.CORPORATE}, currencies={"USD"}, max_rating_score=S("BBB-"), min_years_to_maturity=1.0), 1.0)], {"constraints": [ISSUER_CAP, IG_FLOOR], "forced_actions": ForcedActionPolicy(max_rating_score=S("BBB-"), downgrade_action="DELAYED", downgrade_grace_days=30)}, {"invested_fraction": 0.96}, ("CORPORATE",)),
    "EM_HARD_CURRENCY": SleeveSpec("EM_HARD_CURRENCY", "Emerging markets debt, hard currency", [Bucket("em", UniverseFilter(asset_types={AssetType.SOVEREIGN, AssetType.CORPORATE}, currencies={"USD"}, custom=_tag("em_hard_currency")), 1.0)], {}, {"invested_fraction": 0.95}, ("SOVEREIGN",)),
    "MULTI_SECTOR_INCOME": SleeveSpec("MULTI_SECTOR_INCOME", "Multi-sector income", [
        Bucket("securitized", UniverseFilter(asset_types=SECURITIZED, max_rating_score=S("BB-")), 0.35),
        Bucket("hy", UniverseFilter(asset_types={AssetType.CORPORATE}, currencies={"USD"}, min_rating_score=S("BB+"), max_rating_score=S("CCC+")), 0.25),
        Bucket("ig", UniverseFilter(asset_types={AssetType.CORPORATE}, currencies={"USD"}, max_rating_score=S("BBB-")), 0.20),
        Bucket("em", UniverseFilter(asset_types={AssetType.SOVEREIGN}, custom=_tag("em_hard_currency")), 0.10),
        Bucket("loans", UniverseFilter(asset_types={AssetType.LOAN}), 0.10),
    ], {"constraints": [NAME_CAP]}, {"invested_fraction": 0.95}, ("ABS", "CMBS", "CLO", "MBS", "CORPORATE", "SOVEREIGN", "LOAN")),
    "CORE_PLUS": SleeveSpec("CORE_PLUS", "Flexible core-plus bond", [
        Bucket("govt", UniverseFilter(asset_types={AssetType.SOVEREIGN, AssetType.LINKER}, currencies={"USD"}, custom=_not_tag("em_hard_currency")), 0.30),
        Bucket("ig", UniverseFilter(asset_types={AssetType.CORPORATE}, currencies={"USD"}, max_rating_score=S("BBB-")), 0.35),
        Bucket("mbs", UniverseFilter(asset_types={AssetType.MBS}), 0.20),
        Bucket("hy", UniverseFilter(asset_types={AssetType.CORPORATE}, currencies={"USD"}, min_rating_score=S("BB+")), 0.10),
        Bucket("em", UniverseFilter(asset_types={AssetType.SOVEREIGN}, custom=_tag("em_hard_currency")), 0.05),
    ], {"constraints": [Constraint("duration_band", ConstraintKind.DURATION, "years", minimum=2.0, maximum=7.0, hard=False)]}, {"invested_fraction": 0.96}, ("SOVEREIGN", "LINKER", "CORPORATE", "MBS")),
    "SHORT_DURATION_FLEX": SleeveSpec("SHORT_DURATION_FLEX", "Short duration flexible bond", [
        Bucket("short_credit", UniverseFilter(asset_types={AssetType.CORPORATE, AssetType.SOVEREIGN, AssetType.ABS, AssetType.CLO, AssetType.LOAN}, currencies={"USD"}, max_years_to_maturity=3.5), 0.80),
        Bucket("bills", UniverseFilter(asset_types={AssetType.BILL}), 0.20),
    ], {}, {"invested_fraction": 0.97}, ("CORPORATE", "ABS", "CLO", "LOAN", "BILL")),
    "HIGH_YIELD": SleeveSpec("HIGH_YIELD", "High-yield bond", [Bucket("hy", UniverseFilter(asset_types={AssetType.CORPORATE, AssetType.LOAN}, currencies={"USD"}, min_rating_score=S("BB+"), max_rating_score=S("CCC+")), 1.0)], {"constraints": [ISSUER_CAP], "forced_actions": ForcedActionPolicy(max_rating_score=S("CCC+"), downgrade_action="DELAYED", downgrade_grace_days=20)}, {"invested_fraction": 0.95}, ("CORPORATE", "LOAN", "DISTRESSED")),
    "DEVELOPED_WORLD_HEDGED": SleeveSpec("DEVELOPED_WORLD_HEDGED", "Developed world bond, USD hedged", [
        Bucket("ust", UniverseFilter(asset_types={AssetType.SOVEREIGN}, currencies={"USD"}, custom=_not_tag("em_hard_currency")), 0.50),
        Bucket("eur_credit", UniverseFilter(asset_types={AssetType.CORPORATE}, currencies={"EUR"}, max_rating_score=S("BBB-")), 0.50),
    ], {}, {"invested_fraction": 0.95}, ("SOVEREIGN", "CORPORATE", "FX")),
    "MONEY_MARKET": SleeveSpec("MONEY_MARKET", "Government / prime money market", [Bucket("mm", UniverseFilter(asset_types={AssetType.BILL}, max_years_to_maturity=397 / 365), 1.0)], {}, {"invested_fraction": 0.98}, ("BILL", "MONEY_MARKET")),
}

__all__ = ["SLEEVES", "Bucket", "BucketSleeve", "SleeveConfig", "SleeveSpec"]
