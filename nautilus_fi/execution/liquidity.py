"""Fill probability and tradeable size per bond-day.

Inputs mirror what desks actually look at: has it traded recently (TRACE count / volume), size versus
typical institutional size, side (buying a bond dealers do not hold is harder; selling distressed
paper in a selloff is hardest), quality of the mark (BVAL score / composite presence) and market
stress. The default model is the DNR-style volume gate: a bond is tradeable on a day only if recent
volume clears a threshold, otherwise the order carries to the next day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from nautilus_fi.execution.assumptions import ExecutionAssumptions
from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.enums import AssetType

# typical institutional trade size by asset type (face)
TYPICAL_SIZE = {
    AssetType.SOVEREIGN: 25_000_000.0, AssetType.BILL: 25_000_000.0, AssetType.AGENCY: 5_000_000.0,
    AssetType.SUPRANATIONAL: 5_000_000.0, AssetType.CORPORATE: 2_000_000.0, AssetType.MUNICIPAL: 1_000_000.0,
    AssetType.COVERED: 3_000_000.0, AssetType.PREFERRED: 1_000_000.0, AssetType.LOAN: 2_000_000.0,
    AssetType.LINKER: 10_000_000.0, AssetType.MBS: 5_000_000.0, AssetType.CMBS: 2_000_000.0,
    AssetType.ABS: 2_000_000.0, AssetType.CLO: 2_000_000.0, AssetType.OTHER: 1_000_000.0,
}
# "always liquid" floors: tradeable size available even on no-print days (dealer inventory / RFQ)
LIQUIDITY_FLOOR = {
    AssetType.SOVEREIGN: 250_000_000.0, AssetType.BILL: 250_000_000.0, AssetType.AGENCY: 10_000_000.0,
    AssetType.SUPRANATIONAL: 10_000_000.0, AssetType.LINKER: 50_000_000.0, AssetType.MBS: 10_000_000.0,
}


@dataclass(slots=True)                                   # one per bond-day: slots halve the memory
class LiquidityStats:
    volume_5d: float = 0.0
    trade_count_20d: int = 0
    days_since_trade: int = 9_999
    quote_score: float = 0.0              # 0..1 (vendor score / 10, or 0.9 when a composite exists)
    amount_outstanding: float | None = None
    composite_available: bool = False

    @property
    def adv(self) -> float:
        return self.volume_5d / 5.0


@dataclass
class LiquidityDecision:
    p_fill: float
    max_face: float
    reason: str = ""


class LiquidityModel:
    def evaluate(self, bond: BondSpec, stats: LiquidityStats, side: str, d: date, stress: float, order_face: float, assumptions: ExecutionAssumptions) -> LiquidityDecision:
        raise NotImplementedError


class AlwaysLiquid(LiquidityModel):
    def evaluate(self, bond, stats, side, d, stress, order_face, assumptions) -> LiquidityDecision:
        return LiquidityDecision(1.0, float("inf"), "always")


@dataclass
class VolumeGatedLiquidityModel(LiquidityModel):
    """DNR-style volume gate with size, side, quality and stress adjustments."""

    volume_threshold: float = 250_000.0        # 5-day public volume needed for an "active" day
    cap_fraction: float = 0.20                 # fill at most this share of 5-day volume per day
    base_p_active: float = 0.85
    base_p_inactive: float = 0.10              # chance a dealer shows an axe on a no-print bond
    buy_penalty: float = 0.85                  # buys harder than sells in normal markets
    stress_sell_penalty: float = 0.6           # p *= (1 - penalty * stress) for sells in stress
    stress_buy_penalty: float = 0.25
    oversize_penalty: float = 0.6              # order > 3x typical size
    hy_penalty: float = 0.9

    def evaluate(self, bond, stats, side, d, stress, order_face, assumptions) -> LiquidityDecision:
        stress_eff = max(0.0, stress) * assumptions.stress_multiplier
        typical = TYPICAL_SIZE.get(bond.asset_type, 1_000_000.0)
        active = stats.volume_5d >= self.volume_threshold or stats.days_since_trade <= 1
        if bond.asset_type in LIQUIDITY_FLOOR:
            p = 0.98
            max_face = max(LIQUIDITY_FLOOR[bond.asset_type], self.cap_fraction * stats.volume_5d)
            reason = "govt/agency floor"
        elif active:
            p = min(1.0, self.base_p_active * (0.6 + 0.4 * min(1.0, stats.trade_count_20d / 10.0)))
            max_face = max(self.cap_fraction * stats.volume_5d, typical if stats.quote_score >= 0.6 else 0.0)
            reason = "active"
        else:
            p = self.base_p_inactive * (0.5 + stats.quote_score)
            max_face = typical * (0.5 + stats.quote_score) if stats.quote_score > 0.3 else typical * 0.25
            reason = "inactive"
        if side == "BUY":
            p *= self.buy_penalty
            p *= max(0.0, 1.0 - self.stress_buy_penalty * min(stress_eff, 1.5))
        else:
            p *= max(0.0, 1.0 - self.stress_sell_penalty * min(stress_eff, 1.5))
        if order_face > 3.0 * typical:
            p *= self.oversize_penalty
        if bond.asset_type == AssetType.CORPORATE and (bond.tags.get("high_yield") or (bond.rating_score_initial or 0) > 10):
            p *= self.hy_penalty
        if stress_eff > 0:
            max_face *= max(0.25, 1.0 - 0.5 * min(stress_eff, 1.5))
        return LiquidityDecision(min(max(p, 0.0), 1.0), max_face, reason)


__all__ = ["LIQUIDITY_FLOOR", "TYPICAL_SIZE", "AlwaysLiquid", "LiquidityDecision", "LiquidityModel", "LiquidityStats", "VolumeGatedLiquidityModel"]
