"""Execution assumption sets. Run every strategy under at least GROSS, NET_SPREAD and NET_FULL so the
report shows where the alpha disappears (gross -> spread -> spread + delays/partials)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ExecutionAssumptions:
    name: str = "NET_FULL"
    fill_at_mid: bool = False                 # ignore bid/ask and impact (GROSS)
    apply_fill_probability: bool = True       # volume-gated / probabilistic fills
    apply_size_cap: bool = True               # cap fills at a fraction of observed volume
    min_delay_business_days: int = 0          # operational lag on top of the natural t+1 window
    max_order_age_business_days: int = 5      # orders expire (GTD) after this many business days
    cost_multiplier: float = 1.0              # scale half spreads (stress tests: x2, x3)
    stress_multiplier: float = 1.0            # scale the stress index effect on costs / fill probability
    allow_portfolio_trade_discount: bool = True
    seed: int = 42

    def with_(self, **changes) -> ExecutionAssumptions:
        return replace(self, **changes)


GROSS = ExecutionAssumptions(name="GROSS", fill_at_mid=True, apply_fill_probability=False, apply_size_cap=False, max_order_age_business_days=5)
NET_SPREAD = ExecutionAssumptions(name="NET_SPREAD", fill_at_mid=False, apply_fill_probability=False, apply_size_cap=False)
NET_FULL = ExecutionAssumptions(name="NET_FULL")
NET_FULL_LAGGED = NET_FULL.with_(name="NET_FULL_LAGGED", min_delay_business_days=1)

PRESETS = {a.name: a for a in (GROSS, NET_SPREAD, NET_FULL, NET_FULL_LAGGED)}

__all__ = ["GROSS", "NET_FULL", "NET_FULL_LAGGED", "NET_SPREAD", "PRESETS", "ExecutionAssumptions"]
