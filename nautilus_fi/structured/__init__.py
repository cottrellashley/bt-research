"""Structured products (blueprint section 7): collateral pools, deal waterfalls with triggers and
reinvestment, tranche allocation that conserves deal cash, external engine adapters, TBA and rolls."""

from nautilus_fi.structured.collateral import CollateralPool, PoolAssumptions, PoolCashflow, RemittanceReport
from nautilus_fi.structured.engine import (
    DealCashflowEngine,
    EngineSnapshot,
    ExternalEngineAdapter,
    InternalWaterfallEngine,
    TrancheCashflow,
)
from nautilus_fi.structured.tba import (
    DollarRoll,
    PoolAllocation,
    TBAContract,
    TBASettlementCalendar,
    implied_roll_financing_rate,
)
from nautilus_fi.structured.waterfall import Deal, DealState, Tranche, WaterfallResult, run_waterfall_period

__all__ = [
    "CollateralPool",
    "Deal",
    "DealCashflowEngine",
    "DealState",
    "DollarRoll",
    "EngineSnapshot",
    "ExternalEngineAdapter",
    "InternalWaterfallEngine",
    "PoolAllocation",
    "PoolAssumptions",
    "PoolCashflow",
    "RemittanceReport",
    "TBAContract",
    "TBASettlementCalendar",
    "Tranche",
    "TrancheCashflow",
    "WaterfallResult",
    "implied_roll_financing_rate",
    "run_waterfall_period",
]
