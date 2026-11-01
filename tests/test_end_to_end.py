"""End-to-end: synthetic market -> environment -> Nautilus backtest under the execution ladder."""

from datetime import date

import pandas as pd
import pytest
from nautilus_trader.model import StrategyId
from strategies import (
    EqualWeightCorporates,
    EqualWeightCorporatesConfig,
    EqualWeightEverything,
    EqualWeightEverythingConfig,
)

from nautilus_fi.backtest import (
    AnalyticsConfig,
    BacktestSettings,
    FixedIncomeBacktest,
    compare,
    run_execution_ladder,
)
from nautilus_fi.execution import GROSS, NET_FULL, NET_SPREAD
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


@pytest.fixture(scope="module")
def env():
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 9, 29), n_treasuries=2, n_bills=1, n_ig=8, n_hy=3, n_frn=2, n_mbs=1, n_linkers=1, seed=11)
    mkt = generate_synthetic_market(cfg)
    return mkt.build_environment(analytics=AnalyticsConfig(every_n_days=1)), mkt


def make_corp():
    return EqualWeightCorporates(EqualWeightCorporatesConfig(strategy_id=StrategyId("EQW-001"), rebalance="BMS", max_names=6))


def test_environment_records_mark_tiers(env):
    e, mkt = env
    tiers = e.marks_frame()["tier"].value_counts()
    assert tiers.get("TRADE", 0) > 0 and (tiers.get("EVALUATED", 0) + tiers.get("MODEL", 0)) > 0
    assert any(a.action == "MATURITY" for a in e.actions)        # the bill matures inside the window
    assert any(a.action == "DEFAULT" for a in e.actions) and any(a.action == "RECOVERY" for a in e.actions)


def test_execution_timing_and_ladder(env):
    e, _ = env
    results = run_execution_ladder(e, make_corp, [GROSS, NET_SPREAD, NET_FULL], settings=BacktestSettings(starting_cash=50e6))
    for name, r in results.items():
        assert not r.nav.empty and r.nav["nav"].notna().all()
        fills = r.fills[r.fills["protocol"] != "EXPIRATION"]
        assert len(fills) > 0
        # fills only happen at execution windows, strictly after the decision day
        assert set(pd.to_datetime(fills["ts"]).dt.time.unique()) == {e.schedule.window_time}
        assert (fills["delay_days"] >= 1).all()
    g, s, f = results["GROSS"], results["NET_SPREAD"], results["NET_FULL"]
    assert abs(g.stats()["spread_cost_bp"]) < 0.01                                              # gross fills at mid (price rounding only)
    assert s.fills[s.fills["protocol"] != "EXPIRATION"]["spread_cost"].sum() > 0              # spread costs paid
    assert f.stats()["fill_rate"] < 1.0 or (f.fills_report["status"] == "EXPIRED").any()   # gating / expiry bites
    table = compare(results)
    assert set(table.columns) == {"GROSS", "NET_SPREAD", "NET_FULL"}
    assert table.loc["total_return", "GROSS"] >= table.loc["total_return", "NET_SPREAD"] - 1e-3


def test_all_asset_types_cashflows(env):
    e, mkt = env
    strat = EqualWeightEverything(EqualWeightEverythingConfig(strategy_id=StrategyId("ALL-001"), rebalance="BMS"))
    res = FixedIncomeBacktest(e, strat, NET_FULL, settings=BacktestSettings(starting_cash=200e6)).run()
    kinds = set(res.cashflows["type"])
    assert {"COUPON", "ACCRUED_PAID"} <= kinds
    assert "PRINCIPAL" in kinds                                   # MBS factor paydowns
    coupons = res.cashflows[res.cashflows["type"] == "COUPON"]
    assert (coupons["amount"] > 0).all()
    # redemption events: maturity of the bill at par and recovery of the defaulted bond
    redemptions = res.fills[res.fills["protocol"] == "EXPIRATION"]
    if len(redemptions):
        assert redemptions["clean_price"].between(30, 101).all()
    # NAV identity: nav = cash + market value, every day
    nav = res.nav
    assert ((nav["cash"] + nav["market_value"] + nav["coupon_receivable"] - nav["nav"]).abs() < 1e-6).all()
    assert nav["n_positions"].max() >= 8


def test_determinism(env):
    e, _ = env
    a = FixedIncomeBacktest(e, make_corp, NET_FULL, settings=BacktestSettings(starting_cash=50e6)).run()
    b = FixedIncomeBacktest(e, make_corp, NET_FULL, settings=BacktestSettings(starting_cash=50e6)).run()
    assert a.nav["nav"].iloc[-1] == pytest.approx(b.nav["nav"].iloc[-1])
    assert len(a.fills) == len(b.fills)
