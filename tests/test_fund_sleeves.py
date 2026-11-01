"""Fund-style sleeves on the extended synthetic market: structured tranches, loans, hybrids, EUR
credit with FX funding, CPI linkers, commercial paper and money market paper all run through the
same ledger, fill model and manifest."""

from datetime import date

import pandas as pd
import pytest
from fund_sleeves import SLEEVES

from nautilus_fi.backtest import AnalyticsConfig, FixedIncomeBacktest
from nautilus_fi.execution import NET_FULL, NET_SPREAD
from nautilus_fi.instruments import AssetType
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


@pytest.fixture(scope="module")
def market_env():
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 7, 31), n_treasuries=2, n_bills=1, n_ig=5, n_hy=3, n_frn=1, n_mbs=1, n_linkers=2, seed=5, stress_episode=False)
    mkt = generate_synthetic_market(cfg)
    return mkt, mkt.build_environment(analytics=AnalyticsConfig(every_n_days=1))


def _run(env, name, assumptions=NET_SPREAD, cash=100e6):
    spec = SLEEVES[name]
    return FixedIncomeBacktest(env, spec.make(f"{name}-T"), assumptions, settings=spec.backtest_settings(cash), name=name).run()


def _identity(res):
    nav = res.nav
    assert not nav.empty
    assert ((nav["cash"] + nav["market_value"] + nav["coupon_receivable"] - nav["nav"]).abs() < 1e-4).all()
    assert nav["ledger_balanced"].all()
    assert res.status == "COMPLETED" and res.outcome == "SOLVENT"


def test_market_covers_every_product_family(market_env):
    _, env = market_env
    types = {b.asset_type for b in env.bonds.values()}
    assert {AssetType.CLO, AssetType.ABS, AssetType.CMBS, AssetType.LOAN, AssetType.PREFERRED, AssetType.MUNICIPAL, AssetType.LINKER, AssetType.MBS, AssetType.BILL, AssetType.SOVEREIGN, AssetType.CORPORATE} <= types
    assert {"EUR", "USD"} <= {b.currency for b in env.bonds.values()}
    assert env.validation_problems == [] and not env.unsupported_symbols
    assert len(env.ois_curves) == len(env.dates)
    assert env.fx("EUR", env.dates[3]) is not None
    assert any(b.inflation and b.inflation.index in env.cpi and s in env.cpi_base for s, b in env.bonds.items() if b.asset_type == AssetType.LINKER)


def test_clo_sleeve_receives_waterfall_payments(market_env):
    _, env = market_env
    res = _run(env, "CLO_AAA")
    _identity(res)
    coupons = res.cashflows[(res.cashflows["type"] == "COUPON")]
    assert not coupons.empty and (coupons["note"] == "deal waterfall interest").all()
    paid = set(coupons["symbol"])
    assert paid and all(env.bonds[s].asset_type == AssetType.CLO for s in paid)
    assert all(env.is_structured(s) for s in paid)
    prin = res.cashflows[res.cashflows["type"] == "PRINCIPAL"]
    assert not prin.empty                                    # AAA amortises after the reinvestment period
    h = res.holdings
    assert (h["factor"] <= 1.0 + 1e-12).all() and h["factor"].min() < 1.0


def test_multi_sector_sleeve_loans_settle_t_plus_7(market_env):
    _, env = market_env
    res = _run(env, "MULTI_SECTOR_INCOME")
    _identity(res)
    fills = res.fills[res.fills["protocol"] != "EXPIRATION"]
    loans = fills[fills["symbol"].map(lambda s: env.bonds[s].asset_type == AssetType.LOAN)]
    assert not loans.empty
    lags = [env.calendar.business_days_between(r.date, r.settle_date) for r in loans.itertuples()]
    assert min(lags) >= 7
    corp = fills[fills["symbol"].map(lambda s: env.bonds[s].asset_type == AssetType.CORPORATE)]
    assert corp.empty or max(env.calendar.business_days_between(r.date, r.settle_date) for r in corp.itertuples()) <= 2
    assert res.nav["unsettled_cash"].abs().max() > 0        # the settlement queue carried unsettled cash
    types = {env.bonds[s].asset_type for s in fills["symbol"]}
    assert {AssetType.LOAN, AssetType.CORPORATE} <= types and (types & {AssetType.CLO, AssetType.ABS, AssetType.CMBS, AssetType.MBS})


def test_unhedged_sleeve_funds_eur_bonds_at_spot(market_env):
    _, env = market_env
    res = _run(env, "DEVELOPED_WORLD_UNHEDGED")
    _identity(res)
    eur = res.fills[res.fills["symbol"].map(lambda s: env.bonds[s].currency == "EUR")]
    assert not eur.empty
    fx_events = res.events[res.events["type"] == "FX_FIXING"] if "type" in res.events else pd.DataFrame()
    assert len(fx_events) >= len(eur[eur["side"] == "BUY"])
    assert (res.reconciliation["difference"].abs() < 0.05).all()   # Nautilus multi-currency cash agrees with the ledger
    inv = res.invariants()
    assert inv["passed"].all(), inv.to_string()                      # conservation and payment reconciliation hold across currencies


def test_money_market_sleeve_matures_paper_at_par(market_env):
    _, env = market_env
    res = _run(env, "MONEY_MARKET")
    _identity(res)
    red = res.fills[res.fills["protocol"] == "EXPIRATION"]
    assert not red.empty and (red["clean_price"] - 100.0).abs().max() < 1e-6
    assert res.cashflows[res.cashflows["type"] == "COUPON"].empty
    assert res.nav["duration"].max() < 1.0


def test_high_yield_sleeve_handles_default_and_forced_sales(market_env):
    _, env = market_env
    res = _run(env, "HIGH_YIELD", NET_FULL)
    _identity(res)
    assert not res.decisions.empty
    inv = res.invariants()
    assert not inv.empty and inv["passed"].all()
    assert set(res.support["status"]) <= {"VALIDATED", "SUPPORTED_WITH_APPROXIMATION", "EXPERIMENTAL"}
    assert res.manifest.output_hashes.get("event_chain")


def test_manifest_records_assumptions_and_support(market_env):
    _, env = market_env
    res = _run(env, "CORE_PLUS")
    m = res.manifest
    assert m.run_mode.value == "HISTORICAL_REPLAY" and m.computational_status.value == "COMPLETED"
    keys = {a.get("name") for a in m.assumptions}
    assert {"execution.reference_policy", "funding.settle_before_spend", "curves.max_stale_days"} <= keys
    assert {"nav", "fills", "cashflows", "event_chain"} <= set(m.output_hashes)
    assert not res.support.empty and "point_in_time_replay" in set(res.support["feature"])
    assert res.bundle().to_markdown()
