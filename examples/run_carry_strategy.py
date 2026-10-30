"""Run the carry/value corporate strategy gross, net of spread and net of spread + delays.

Usage: python examples/run_carry_strategy.py [--quick]
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from nautilus_trader.model import StrategyId

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies import CarryValueConfig, CarryValueStrategy

from nautilus_fi.backtest import AnalyticsConfig, BacktestSettings, compare, run_execution_ladder
from nautilus_fi.execution import GROSS, NET_FULL, NET_SPREAD
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


def main(quick: bool = False) -> None:
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 12, 29) if quick else date(2024, 12, 31), seed=7)
    t0 = time.time()
    market = generate_synthetic_market(cfg)
    env = market.build_environment(analytics=AnalyticsConfig(every_n_days=1))
    print(f"environment: {len(env.bonds)} bonds, {len(env.dates)} days, {len(env.marks)} marks, built in {time.time() - t0:.1f}s")
    print(env.marks_frame()["tier"].value_counts().to_string(), "\n")

    def make_strategy():
        return CarryValueStrategy(CarryValueConfig(strategy_id=StrategyId("CARRY-001"), rebalance="M", n_long=10, log_decisions=False))

    t1 = time.time()
    results = run_execution_ladder(env, make_strategy, [GROSS, NET_SPREAD, NET_FULL], settings=BacktestSettings(starting_cash=100e6))
    print(f"three backtests in {time.time() - t1:.1f}s\n")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(compare(results).round(4).to_string())
        net = results["NET_FULL"]
        print("\nNET_FULL cashflows by type:")
        print(net.cashflows.groupby("type")["amount"].agg(["count", "sum"]).round(0).to_string())
        print("\nNET_FULL fill decisions:")
        print(net.decisions.groupby("reason").size().to_string())


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
