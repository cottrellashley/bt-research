"""Standard robustness runs: gross/net/net+delay ladder, cost stress, capacity sweep."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from nautilus_fi.backtest.builder import BacktestSettings, FixedIncomeBacktest
from nautilus_fi.backtest.environment import MarketEnvironment
from nautilus_fi.backtest.result import FixedIncomeResult, compare
from nautilus_fi.backtest.strategy import FixedIncomeStrategy
from nautilus_fi.execution.assumptions import GROSS, NET_FULL, NET_SPREAD, ExecutionAssumptions


def run_execution_ladder(
    env: MarketEnvironment,
    strategy_factory: Callable[[], FixedIncomeStrategy],
    assumptions: list[ExecutionAssumptions] | None = None,
    settings: BacktestSettings | None = None,
    **kwargs,
) -> dict[str, FixedIncomeResult]:
    """Run the same strategy under GROSS, NET_SPREAD and NET_FULL (or a custom ladder)."""
    results: dict[str, FixedIncomeResult] = {}
    for a in assumptions or [GROSS, NET_SPREAD, NET_FULL]:
        results[a.name] = FixedIncomeBacktest(env, strategy_factory, a, settings=settings, name=a.name, **kwargs).run()
    return results


def run_cost_stress(
    env: MarketEnvironment,
    strategy_factory: Callable[[], FixedIncomeStrategy],
    multipliers: tuple[float, ...] = (1.0, 2.0, 3.0),
    base: ExecutionAssumptions = NET_FULL,
    settings: BacktestSettings | None = None,
    **kwargs,
) -> dict[str, FixedIncomeResult]:
    results: dict[str, FixedIncomeResult] = {}
    for m in multipliers:
        a = base.with_(name=f"{base.name}_COSTx{m:g}", cost_multiplier=m, stress_multiplier=max(1.0, m / 1.5))
        results[a.name] = FixedIncomeBacktest(env, strategy_factory, a, settings=settings, name=a.name, **kwargs).run()
    return results


def run_capacity_sweep(
    env: MarketEnvironment,
    strategy_factory: Callable[[], FixedIncomeStrategy],
    aum_levels: tuple[float, ...] = (50e6, 250e6, 1e9),
    assumptions: ExecutionAssumptions = NET_FULL,
    settings: BacktestSettings | None = None,
    **kwargs,
) -> dict[str, FixedIncomeResult]:
    """Re-run at several AUM levels; fill rates and delay costs reveal capacity."""
    results: dict[str, FixedIncomeResult] = {}
    base = settings or BacktestSettings()
    for aum in aum_levels:
        s = BacktestSettings(**{**base.__dict__, "starting_cash": aum})
        name = f"AUM_{aum / 1e6:g}mm"
        results[name] = FixedIncomeBacktest(env, strategy_factory, assumptions, settings=s, name=name, **kwargs).run()
    return results


def ladder_table(results: dict[str, FixedIncomeResult]) -> pd.DataFrame:
    return compare(results)


__all__ = ["ladder_table", "run_capacity_sweep", "run_cost_stress", "run_execution_ladder"]
