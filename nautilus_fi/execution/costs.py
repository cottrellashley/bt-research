"""Transaction cost models: half spread as a function of bond, size, side, regime and protocol.

Defaults follow the stylised facts of Edwards, Harris and Piwowar (costs fall sharply with trade size,
are lower for highly rated, recently issued and short-maturity bonds), Guo-Lehalle-Xu (impact beyond
normal size, buy/sell asymmetry) and Meli-Todorova (portfolio trades cost ~40% less than single-name
RFQ, most in illiquid bonds). Costs scale with a stress indicator so March-2020-style regimes bite.
Calibrate the tables on your own TRACE / RFQ history before trusting absolute levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.execution.assumptions import ExecutionAssumptions
from nautilus_fi.execution.liquidity import TYPICAL_SIZE, LiquidityStats
from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.enums import AssetType, rating_bucket

# base half spread in bp of price for a typical institutional size, by asset type
BASE_HALF_SPREAD_BP = {
    AssetType.SOVEREIGN: 1.0, AssetType.BILL: 0.5, AssetType.AGENCY: 4.0, AssetType.SUPRANATIONAL: 5.0,
    AssetType.CORPORATE: 18.0, AssetType.MUNICIPAL: 35.0, AssetType.COVERED: 8.0, AssetType.PREFERRED: 40.0,
    AssetType.LOAN: 50.0, AssetType.LINKER: 3.0, AssetType.MBS: 6.0, AssetType.CMBS: 30.0, AssetType.ABS: 25.0,
    AssetType.CLO: 45.0, AssetType.OTHER: 30.0,
}
RATING_MULTIPLIER = {"AAA-AA": 0.75, "A": 0.9, "BBB": 1.15, "BB": 1.6, "B": 2.2, "CCC-D": 3.5, "NR": 1.5}


class CostModel:
    def half_spread_bp(self, bond: BondSpec, stats: LiquidityStats, face: float, side: str, d: date, stress: float, protocol: str, rating_score: int | None, assumptions: ExecutionAssumptions, duration: float | None = None) -> float:
        raise NotImplementedError


class ZeroCostModel(CostModel):
    def half_spread_bp(self, *args, **kwargs) -> float:
        return 0.0


@dataclass
class ParametricCostModel(CostModel):
    base_bp: dict = field(default_factory=lambda: dict(BASE_HALF_SPREAD_BP))
    rating_multiplier: dict = field(default_factory=lambda: dict(RATING_MULTIPLIER))
    size_elasticity: float = 0.25           # cost ~ (typical / size) ** elasticity (falls with size)
    size_mult_floor: float = 0.6
    size_mult_cap: float = 3.0
    duration_slope: float = 0.05            # +5% per year of duration
    off_the_run_mult: float = 1.2           # issued more than 2 years ago
    buy_mult: float = 1.10                  # buy/sell asymmetry (dealers need to source bonds)
    stress_beta: float = 1.5                # half spread *= 1 + beta * stress
    impact_bp_per_sqrt_adv: float = 8.0     # impact = coef * sqrt(size / max(adv, typical)) beyond typical size
    portfolio_trade_discount: float = 0.40  # Meli-Todorova
    low_quality_mult: float = 1.3           # mark quality < 0.4 (model prices): wider dealer quotes

    def half_spread_bp(self, bond, stats, face, side, d, stress, protocol, rating_score, assumptions, duration=None) -> float:
        typical = TYPICAL_SIZE.get(bond.asset_type, 1_000_000.0)
        hs = self.base_bp.get(bond.asset_type, 30.0)
        hs *= self.rating_multiplier.get(rating_bucket(rating_score), 1.0) if not bond.asset_type.is_government else 1.0
        dur = duration if duration is not None else (bond.remaining_years(d) or 5.0) * 0.8
        hs *= 1.0 + self.duration_slope * min(dur, 20.0)
        if bond.age_years(d) > 2.0 and not bond.asset_type.is_government:
            hs *= self.off_the_run_mult
        size_mult = (typical / max(face, 1.0)) ** self.size_elasticity
        hs *= min(max(size_mult, self.size_mult_floor), self.size_mult_cap)
        if side == "BUY":
            hs *= self.buy_mult
        if stats.quote_score < 0.4 and stats.days_since_trade > 5:
            hs *= self.low_quality_mult
        stress_eff = max(0.0, stress) * assumptions.stress_multiplier
        hs *= 1.0 + self.stress_beta * stress_eff
        if face > typical:
            ref = max(stats.adv, typical)
            hs += self.impact_bp_per_sqrt_adv * ((face - typical) / ref) ** 0.5 * (1.0 + stress_eff)
        if protocol == "PORTFOLIO_TRADE" and assumptions.allow_portfolio_trade_discount:
            hs *= 1.0 - self.portfolio_trade_discount
        return hs * assumptions.cost_multiplier


@dataclass
class VendorBidAskCostModel(CostModel):
    """Use the mark's own bid/ask (BVAL, ICE, CP+) when it exists; fall back to the parametric model
    and always add impact for size beyond normal."""

    fallback: ParametricCostModel = field(default_factory=ParametricCostModel)
    mark_half_spread_bp: dict = field(default_factory=dict)   # (symbol, date) -> half spread bp from the mark

    def half_spread_bp(self, bond, stats, face, side, d, stress, protocol, rating_score, assumptions, duration=None) -> float:
        base = self.mark_half_spread_bp.get((bond.symbol, d))
        if base is None or base <= 0:
            return self.fallback.half_spread_bp(bond, stats, face, side, d, stress, protocol, rating_score, assumptions, duration)
        typical = TYPICAL_SIZE.get(bond.asset_type, 1_000_000.0)
        hs = base * (1.0 + self.fallback.stress_beta * max(0.0, stress) * assumptions.stress_multiplier * 0.5)
        if side == "BUY":
            hs *= self.fallback.buy_mult
        if face > typical:
            ref = max(stats.adv, typical)
            hs += self.fallback.impact_bp_per_sqrt_adv * ((face - typical) / ref) ** 0.5
        if protocol == "PORTFOLIO_TRADE" and assumptions.allow_portfolio_trade_discount:
            hs *= 1.0 - self.fallback.portfolio_trade_discount
        return hs * assumptions.cost_multiplier


__all__ = ["BASE_HALF_SPREAD_BP", "RATING_MULTIPLIER", "CostModel", "ParametricCostModel", "VendorBidAskCostModel", "ZeroCostModel"]
