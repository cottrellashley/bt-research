"""Stress tests and uncertainty analysis (blueprint section 21): static shocks versus dynamic stress
versus historical replay versus synthetic paths, stress dimensions, joint mechanisms, reverse stress
and uncertainty categories kept visible rather than averaged away."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class StressKind(str, Enum):
    STATIC_SHOCK = "STATIC_SHOCK"            # positions fixed, revalue
    DYNAMIC = "DYNAMIC"                      # decisions, cashflows, margin and execution evolve
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    SYNTHETIC = "SYNTHETIC"


class StressDimension(str, Enum):
    MARKET = "MARKET"
    CREDIT_BEHAVIOR = "CREDIT_BEHAVIOR"
    LIQUIDITY = "LIQUIDITY"
    FUNDING = "FUNDING"
    DATA_OPERATIONS = "DATA_OPERATIONS"
    MODEL = "MODEL"


@dataclass(frozen=True)
class StressScenario:
    name: str
    kind: StressKind
    dimension: StressDimension
    curve_shocks_bp: dict[str, float] = field(default_factory=dict)      # curve name or "*" -> parallel bp
    twist_bp: dict[str, float] = field(default_factory=dict)             # tenor -> bp (non-parallel)
    spread_shock_bp: float = 0.0
    spread_shock_by_bucket: dict[str, float] = field(default_factory=dict)
    fx_shocks: dict[str, float] = field(default_factory=dict)            # pair -> pct
    vol_multiplier: float = 1.0
    default_multiplier: float = 1.0
    recovery_delay_months: int = 0
    severity_add: float = 0.0
    cpr_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    fill_probability_multiplier: float = 1.0
    quote_response_multiplier: float = 1.0
    delay_add_business_days: int = 0
    venue_loss: tuple[str, ...] = ()
    haircut_add: float = 0.0
    repo_rollover_fail: bool = False
    borrow_recall: bool = False
    cash_rate_add_bp: float = 0.0
    intraday_margin_call: bool = False
    stale_feed_days: int = 0
    publication_delay_days: int = 0
    missing_fixings: bool = False
    corrected_reports: bool = False
    alternative_model: str | None = None
    held_fixed: tuple[str, ...] = ()
    rationale: str = ""

    def execution_overrides(self) -> dict:
        return {"cost_multiplier": self.cost_multiplier, "fill_probability_multiplier": self.fill_probability_multiplier, "min_delay_add": self.delay_add_business_days}


STANDARD_SCENARIOS: list[StressScenario] = [
    StressScenario("rates +100bp parallel", StressKind.STATIC_SHOCK, StressDimension.MARKET, curve_shocks_bp={"*": 100}, rationale="duration"),
    StressScenario("rates -100bp parallel", StressKind.STATIC_SHOCK, StressDimension.MARKET, curve_shocks_bp={"*": -100}, cpr_multiplier=1.6, rationale="duration and refinancing wave"),
    StressScenario("bear steepener", StressKind.STATIC_SHOCK, StressDimension.MARKET, twist_bp={"2": 0, "10": 60, "30": 90}),
    StressScenario("spreads +200bp", StressKind.STATIC_SHOCK, StressDimension.MARKET, spread_shock_bp=200, rationale="credit beta"),
    StressScenario("USD +8%", StressKind.STATIC_SHOCK, StressDimension.MARKET, fx_shocks={"EURUSD": -0.08, "GBPUSD": -0.08}),
    StressScenario("default clustering", StressKind.DYNAMIC, StressDimension.CREDIT_BEHAVIOR, default_multiplier=3.0, recovery_delay_months=6, severity_add=0.15),
    StressScenario("March 2020 liquidity", StressKind.DYNAMIC, StressDimension.LIQUIDITY, spread_shock_bp=150, cost_multiplier=3.0, fill_probability_multiplier=0.5, quote_response_multiplier=0.4, delay_add_business_days=2, rationale="dealer balance sheet withdrawal"),
    StressScenario("venue loss + failed new issues", StressKind.DYNAMIC, StressDimension.LIQUIDITY, venue_loss=("PORTFOLIO_TRADE", "ALL_TO_ALL"), fill_probability_multiplier=0.7),
    StressScenario("funding squeeze", StressKind.DYNAMIC, StressDimension.FUNDING, haircut_add=0.05, repo_rollover_fail=True, cash_rate_add_bp=150, borrow_recall=True, intraday_margin_call=True),
    StressScenario("stale feeds and delayed publications", StressKind.DYNAMIC, StressDimension.DATA_OPERATIONS, stale_feed_days=5, publication_delay_days=3, missing_fixings=True),
    StressScenario("alternative prepayment model", StressKind.DYNAMIC, StressDimension.MODEL, alternative_model="psa-150", rationale="structural alternative, not a parameter tweak"),
    StressScenario("bid marks instead of mid", StressKind.STATIC_SHOCK, StressDimension.MODEL, alternative_model="mark-policy-bid"),
]


def joint_scenarios(a: StressScenario, b: StressScenario, name: str | None = None) -> StressScenario:
    """Combine two scenarios (wider spreads plus higher haircuts, falling rates plus refinancing)."""
    merged = {**a.__dict__}
    for k, v in b.__dict__.items():
        if k in ("name", "kind", "dimension", "rationale"):
            continue
        cur = merged[k]
        if isinstance(v, dict):
            merged[k] = {**cur, **v}
        elif isinstance(v, bool):
            merged[k] = cur or v
        elif isinstance(v, (int, float)) and k.endswith(("multiplier",)):
            merged[k] = cur * v
        elif isinstance(v, (int, float)):
            merged[k] = cur + v
        elif isinstance(v, tuple):
            merged[k] = tuple(set(cur) | set(v))
        elif v is not None:
            merged[k] = v
    merged["name"] = name or f"{a.name} + {b.name}"
    merged["kind"] = StressKind.DYNAMIC if StressKind.DYNAMIC in (a.kind, b.kind) else a.kind
    merged["dimension"] = a.dimension
    merged["rationale"] = f"joint: {a.rationale}; {b.rationale}"
    return StressScenario(**merged)


def reverse_stress(run: Callable[[float], dict], grid: list[float], failure: Callable[[dict], bool]) -> dict:
    """Find the smallest shock level at which ``failure(result)`` is true (threshold search over a grid)."""
    rows = []
    threshold = None
    for level in grid:
        res = run(level)
        failed = failure(res)
        rows.append({"level": level, "failed": failed, **{k: v for k, v in res.items() if isinstance(v, (int, float))}})
        if failed and threshold is None:
            threshold = level
    return {"threshold": threshold, "path": pd.DataFrame(rows)}


def uncertainty_envelope(results: dict[str, dict[str, float]], metric: str) -> pd.DataFrame:
    """Separate result columns per uncertainty category (contract data, timing, marks, execution,
    parameters, model structure, numerical) instead of one averaged interval."""
    rows = []
    for category, by_variant in results.items():
        vals = pd.Series(by_variant, dtype=float)
        rows.append({"category": category, "n_variants": len(vals), f"{metric}_min": vals.min(), f"{metric}_median": vals.median(), f"{metric}_max": vals.max(), "range": vals.max() - vals.min()})
    return pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)


__all__ = ["STANDARD_SCENARIOS", "StressDimension", "StressKind", "StressScenario", "joint_scenarios", "reverse_stress", "uncertainty_envelope"]
