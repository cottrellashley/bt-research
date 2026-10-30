"""Cost stress (x1, x2, x3 half spreads with stress scaling) and a capacity sweep."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from nautilus_trader.model import StrategyId

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies import CarryValueConfig, CarryValueStrategy

from nautilus_fi.backtest import (
    AnalyticsConfig,
    BacktestSettings,
    compare,
    run_capacity_sweep,
    run_cost_stress,
)
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


def main() -> None:
    market = generate_synthetic_market(SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 12, 29), seed=7))
    env = market.build_environment(analytics=AnalyticsConfig(every_n_days=1))

    def make_strategy():
        return CarryValueStrategy(CarryValueConfig(strategy_id=StrategyId("CARRY-001"), rebalance="M", n_long=10))

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("=== cost stress ===")
        print(compare(run_cost_stress(env, make_strategy, (1.0, 2.0, 3.0), settings=BacktestSettings(starting_cash=100e6))).round(4).to_string())
        print("\n=== capacity sweep ===")
        print(compare(run_capacity_sweep(env, make_strategy, (50e6, 250e6, 1e9))).round(4).to_string())


if __name__ == "__main__":
    main()
