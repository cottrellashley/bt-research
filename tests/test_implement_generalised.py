"""The event engine generalised: fields, benchmarks, vendor analytics, terms-free bonds, construction
logs, the margin venue (overlays and synthetic shorts), repo-financed leverage, streaming, incremental
fill processing and Paper / Implement reconciliation."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from strategies import (
    BuyAndHold,
    BuyAndHoldConfig,
    EqualWeightCorporates,
    EqualWeightCorporatesConfig,
    EqualWeightEverything,
    EqualWeightEverythingConfig,
)

from nautilus_fi.accounting.margin import MarginSettings
from nautilus_fi.backtest import (
    BacktestSettings,
    FixedIncomeBacktest,
    FixedIncomeStrategy,
    FixedIncomeStrategyConfig,
    MarketEnvironmentBuilder,
)
from nautilus_fi.backtest.spec_strategy import SpecStrategy, SpecStrategyConfig
from nautilus_fi.construct import BenchmarkPlusTilts, ConstructionSpec, EqualWeight
from nautilus_fi.core.registry import UnsupportedCombination
from nautilus_fi.domain import AnalyticsSource, Conventions, MissingReturnPolicy, Timing
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec, UniverseRule
from nautilus_fi.execution import GROSS, NET_FULL, NET_SPREAD
from nautilus_fi.execution.process import ProcessConfig
from nautilus_fi.funding.policy import FundingPolicy
from nautilus_fi.instruments import BondSpec
from nautilus_fi.overlays import OverlayBook, index_trs, treasury_future
from nautilus_fi.paper import PaperEngine, PaperSettings
from nautilus_fi.paper.reconcile import paper_data_from_environment, reconcile
from nautilus_fi.portfolio import Constraint, ConstraintKind, ForcedActionPolicy
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel, to_market_inputs


def _small_market(months=6, n_ig=10):
    end = date(2023, 1 + months - 1, 28)
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=end, n_treasuries=3, n_bills=0, n_ig=n_ig, n_hy=0, n_frn=0, n_mbs=0, n_linkers=0, n_frn_arrears=0, n_loans=0, n_hybrids=0, n_munis=0, n_em_sov=0, n_eur_corp=0, n_cp=0, clo_deal=False, abs_deal=False, cmbs_deal=False, default_one_hy=False)
    return generate_synthetic_market(cfg).build_environment()


@pytest.fixture(scope="module")
def market():
    return _small_market()


@pytest.fixture(scope="module")
def panel_env():
    """An environment built from the panel universe: vendor analytics (compared with computed ones),
    benchmark constituents, a signal field."""
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=80, start="2015-01-02", end="2015-08-31", seed=4))
    inp = to_market_inputs(s)
    b = MarketEnvironmentBuilder(inp["reference"], s.dates[0], s.dates[-1], strict_dependencies=False)
    b.with_evaluated_prices(inp["evaluated_prices"]).with_curves(inp["curves"]).with_ratings(inp["ratings"]).with_fields(inp["fields"]).with_benchmark_constituents(inp["benchmark_constituents"], "IDX")
    b.with_analytics(inp["analytics"], AnalyticsSource.COMPARE)
    return b.build(), s


def test_vendor_analytics_fields_and_benchmark_reach_the_environment(panel_env):
    env, s = panel_env
    assert env.analytics_source == "COMPARE" and len(env.analytics_discrepancies) > 100
    d = env.dates[30]
    sym = next(k for k in env.bonds if (k, d) in env.analytics_by_key)
    i, j = s.panel.date_pos(d), s.panel.id_pos([sym])[0]
    assert env.analytics(sym, d).z_spread_bp == pytest.approx(float(s.panel["oas_bp"][i, j]), rel=1e-6)    # vendor value used
    assert "signal.value" in env.field_panel and "benchmark.weight" in env.field_panel
    w = env.field_panel["benchmark.weight"]
    assert np.allclose(w[w.sum(axis=1) > 0].sum(axis=1), 1.0) and env.benchmark is not None


def test_spec_strategy_reads_published_fields_and_benchmark(panel_env):
    env, _ = panel_env
    spec = ResearchSpec("tilt", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(BenchmarkPlusTilts(0.5, max_active=0.01)), Conventions(missing_returns=MissingReturnPolicy.ZERO), calendar="BME")
    strat = SpecStrategy(spec, SpecStrategyConfig(rebalance="BME"))
    res = FixedIncomeBacktest(env, strat, NET_SPREAD, settings=BacktestSettings(starting_cash=200e6, conventions=spec.conventions, universe_from_benchmark=True)).run()
    assert res.status == "COMPLETED" and res.outcome == "SOLVENT" and res.extra["unprocessed_fills"] == 0
    rebal = [r for r in strat.spec_log if r["action"] == "rebalance"]
    assert len(rebal) >= 7 and all(r["method"] == "BenchmarkPlusTilts" for r in rebal)
    held = set(res.holdings["symbol"])
    members = set(env.field_panel.ids[(env.field_panel["benchmark.weight"] > 0).any(axis=0)])
    assert held <= members                                                       # the universe came from the benchmark
    assert res.manifest.conventions["name"] == "canonical"


def test_terms_free_bonds_with_vendor_prices_and_supplied_cashflows():
    days = pd.bdate_range("2024-01-02", "2024-04-30")
    ref = pd.DataFrame({"symbol": ["V1"], "issuer": ["ACME"], "issue_date": [None], "maturity": ["2030-06-15"], "coupon_rate": [0.05], "terms_source": ["VENDOR"], "min_piece": 1000, "increment": 1000})
    ev = pd.DataFrame({"date": days, "instrument_id": "V1", "mid": 100.0, "bid": 99.9, "ask": 100.1, "accrued": [5.0 * (i % 125) / 250 for i in range(len(days))]})
    an = pd.DataFrame({"date": days, "instrument_id": "V1", "oas_bp": 110.0, "oad": 5.0, "oasd": 5.0})
    cf = pd.DataFrame({"symbol": ["V1"], "pay_date": ["2024-03-15"], "interest_per_100": [2.5]})
    b = MarketEnvironmentBuilder(ref, days[0], days[-1])
    env = b.with_evaluated_prices(ev).with_analytics(an, "VENDOR").with_supplied_cashflows(cf).build()
    assert env.bonds["V1"].terms_free and env.accrued("V1", date(2024, 2, 1)) == pytest.approx(ev.set_index("date").loc["2024-02-01", "accrued"])
    res = FixedIncomeBacktest(env, BuyAndHold(BuyAndHoldConfig(targets={"V1": 1_000_000})), GROSS, settings=BacktestSettings(starting_cash=2e6)).run()
    coupons = res.cashflows[res.cashflows["type"] == "COUPON"]
    assert res.status == "COMPLETED" and coupons["amount"].sum() == pytest.approx(25_000.0)
    assert res.invariants()["passed"].all()
    bad = pd.DataFrame({"symbol": ["V2"], "issuer": ["X"], "issue_date": [None], "maturity": ["2030-01-01"], "coupon_rate": [0.04], "terms_source": ["VENDOR"]})
    with pytest.raises(UnsupportedCombination):
        MarketEnvironmentBuilder(bad, days[0], days[-1]).with_evaluated_prices(ev.assign(instrument_id="V2").drop(columns="accrued")).build()   # needs vendor accrued and cashflows


def test_construction_log_records_violations_and_turnover_budget(market):
    corp = {s: b.sector for s, b in market.bonds.items() if b.asset_type.value == "CORPORATE"}
    cons = [Constraint("sector cap", ConstraintKind.GROUP_WEIGHT, "weight", maximum=0.3, group_col="sector"), Constraint("turnover", ConstraintKind.TURNOVER, "per rebalance", maximum=0.25)]
    res = FixedIncomeBacktest(market, EqualWeightCorporates(EqualWeightCorporatesConfig(rebalance="M", max_names=9)), NET_SPREAD, settings=BacktestSettings(starting_cash=50e6, constraints=cons)).run()
    log = res.construction
    assert not log.empty and "summary" in set(log["stage"]) and len(res.fills) > 0
    dropped = log[log["stage"] == "dropped"]
    if len(dropped):                                                             # buys that would breach the sector cap are dropped and logged
        assert dropped["reason"].str.contains("hard constraint").all()
    summ = log[log["stage"] == "summary"]
    post = log[(log["stage"] == "post") & (log["severity"] == "HARD")]
    for _, row in summ[summ["hard_post"] > 0].iterrows():                       # breaches dropping buys cannot cure are recorded, not hidden
        assert (post["date"] == row["date"]).sum() == row["hard_post"]
    assert (summ["turnover_HALF_SUM"].iloc[1:] <= 0.25 + 1e-3).all()            # after the initial build, trades scale to the budget
    assert res.stats()["turnover_definition"] == "HALF_SUM" and len(corp) > 0


def test_bme_calendar_in_the_event_engine(market):
    strat = EqualWeightCorporates(EqualWeightCorporatesConfig(rebalance="BME", start_with_decision=False, max_names=5))
    FixedIncomeBacktest(market, strat, NET_SPREAD, settings=BacktestSettings(starting_cash=20e6)).run()
    dates = sorted({r["date"] for r in strat.decision_log})
    assert dates and all(market.calendar.add_business_days(d, 1).month != d.month for d in dates)


def test_cash_venue_short_is_rejected_but_margin_venue_carries_it(market):
    corp = [s for s, b in market.bonds.items() if b.asset_type.value == "CORPORATE"]

    class Shorter(FixedIncomeStrategy):
        def on_decision(self, d):
            if not getattr(self, "_done", False):
                self._done = True
                self.order_face(corp[0], -1_000_000)

    res = FixedIncomeBacktest(market, Shorter(FixedIncomeStrategyConfig(rebalance="D")), NET_SPREAD, settings=BacktestSettings(process_config=ProcessConfig(allow_shorts=True))).run()
    assert res.fills.empty                                                       # Nautilus refuses short sales on a cash account

    dates = pd.to_datetime(market.dates)
    book = OverlayBook()
    book.add(index_trs("IG-TRS", pd.Series(100 * np.cumprod(1 + np.full(len(dates), 0.0002)), index=dates), 0.05, 50.0, exposures=pd.DataFrame({"oad": 7.0}, index=dates)))
    book.add(treasury_future("TYM3", pd.Series(110 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, len(dates))), index=dates), date(2023, 4, 20), oad=6.5))

    class Overlayed(FixedIncomeStrategy):
        def on_decision(self, d):
            self.rebalance_to_weights({s: 0.2 for s in corp[:4]})
            targets = {"IG-TRS": 0.5, corp[5] + "-TRS": -0.05}
            if d < date(2023, 4, 15):
                targets["TYM3"] = -0.3
            self.rebalance_overlays(targets)

    for collateral in (10e6, 0.5e6):
        bt = FixedIncomeBacktest(market, Overlayed(FixedIncomeStrategyConfig(rebalance="M")), NET_SPREAD, settings=BacktestSettings(starting_cash=100e6, margin=MarginSettings(collateral=collateral, book=book, synthetic_shorts={corp[5]})))
        r = bt.run()
        nav = r.nav
        assert r.status == "COMPLETED" and r.invariants()["passed"].all()
        assert (nav["cash"] + nav["market_value"] + nav["coupon_receivable"] + nav["margin_equity"] - nav["nav"]).abs().max() < 1e-4
        pos = r.positions_report
        sides = dict(zip(pos["instrument_id"].astype(str), pos["side"].astype(str)))
        assert sides["IG-TRS.MARGIN"] == "LONG" and sides[f"{corp[5]}-TRS.MARGIN"] == "SHORT" and sides["TYM3.MARGIN"] == "FLAT"
        if collateral < 1e6:
            assert len(r.extra["margin_transfers"]) >= 1 and not r.extra["borrow_fees"].empty
            final_small = r.nav["nav"].iloc[-1]
        else:
            final_big = r.nav["nav"].iloc[-1]
    assert final_small == pytest.approx(final_big, rel=1e-6)                   # where collateral sits does not change NAV


def test_repo_financed_leverage(market):
    res = FixedIncomeBacktest(market, EqualWeightCorporates(EqualWeightCorporatesConfig(rebalance="M", invested_fraction=1.4, max_names=10)), NET_SPREAD, settings=BacktestSettings(starting_cash=50e6, funding_policy=FundingPolicy(repo_for_leverage=True))).run()
    assert res.outcome == "SOLVENT" and (res.nav["gross_exposure"] / res.nav["nav"]).max() > 1.2 and res.nav["repo_balance"].max() > 0
    assert res.cashflows[res.cashflows["type"] == "FINANCING"]["amount"].sum() < 0 and res.invariants()["passed"].all()


def test_streaming_incremental_scan_and_reproducible_hashes():
    env = generate_synthetic_market(SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 7, 31), n_treasuries=2, n_bills=1, n_ig=6, n_hy=3, n_frn=1, n_mbs=1, n_linkers=1, seed=11)).build_environment()
    mk = lambda: EqualWeightEverything(EqualWeightEverythingConfig(rebalance="M"))
    a = FixedIncomeBacktest(env, mk, NET_FULL, settings=BacktestSettings(starting_cash=100e6)).run()
    b = FixedIncomeBacktest(env, mk, NET_FULL, settings=BacktestSettings(starting_cash=100e6, stream_days=21)).run()
    c = FixedIncomeBacktest(env, mk, NET_FULL, settings=BacktestSettings(starting_cash=100e6, order_scan="full")).run()
    assert a.extra["unprocessed_fills"] == 0
    assert a.nav["nav"].iloc[-1] == pytest.approx(b.nav["nav"].iloc[-1]) == pytest.approx(c.nav["nav"].iloc[-1])
    assert a.manifest.output_hashes["event_chain"] == b.manifest.output_hashes["event_chain"] == c.manifest.output_hashes["event_chain"]
    assert a.manifest.output_hashes["fills"] == b.manifest.output_hashes["fills"]


def test_paper_and_implement_reconcile(market):
    spec = ResearchSpec("wide", FieldSignal("oas_bp"), ConstructionSpec(EqualWeight(top=6, invested=0.95)), Conventions(timing=Timing(1), missing_returns=MissingReturnPolicy.ZERO), calendar="BME", universe=UniverseRule(field_rules=[("oasd", ">", 1.0)]))
    data = paper_data_from_environment(market)
    paper = PaperEngine(spec, data, PaperSettings(record_daily_weights=True, initial_nav=50e6)).run()
    impl = FixedIncomeBacktest(market, SpecStrategy(spec, SpecStrategyConfig(rebalance="BME", universe_from_benchmark=False)), NET_SPREAD, settings=BacktestSettings(starting_cash=50e6, conventions=spec.conventions)).run()
    rec = reconcile(paper, impl, data.panel)
    s = rec.summary
    parts = s["holdings_gap"] + s["cash"] + s["execution"] + s["residual"]
    assert parts == pytest.approx(s["shortfall_vs_gross"], abs=1e-12) and s["daily_correlation"] > 0.5
    assert abs(s["residual"]) < abs(s["shortfall_vs_gross"]) + 0.02


def test_forced_sale_on_index_removal(panel_env):
    env, _ = panel_env
    w = env.field_panel["benchmark.weight"]
    ids = env.field_panel.ids
    first = int(np.argmax(w.sum(axis=1) > 0))
    leavers = [ids[j] for j in np.nonzero(w[first + 1] > 0)[0] if (w[first + 1:, j] == 0).any()]
    assert leavers
    target = leavers[0]

    class HoldForever(FixedIncomeStrategy):
        def on_decision(self, d):
            self.rebalance_to_target_faces({target: 500_000.0})

    res = FixedIncomeBacktest(env, HoldForever(FixedIncomeStrategyConfig(rebalance="W")), NET_SPREAD, settings=BacktestSettings(starting_cash=5e6, forced_actions=ForcedActionPolicy(max_rating_score=None, index_removal_action="IMMEDIATE"))).run()
    sells = res.fills[(res.fills["symbol"] == target) & (res.fills["side"] == "SELL")]
    assert res.status == "COMPLETED" and len(sells) >= 1                             # sold once it left the index, despite its target


def test_bond_spec_terms_source_validation():
    with pytest.raises(ValueError):
        from nautilus_fi.data.reference import bond_from_row

        bond_from_row({"symbol": "X", "issuer": "Y", "issue_date": None, "maturity": "2030-01-01", "coupon_rate": 0.05})
    assert not BondSpec("A", "B", date(2020, 1, 1), date(2030, 1, 1)).terms_free


def test_swap_contract_prices_carry_and_exposures():
    from nautilus_fi.instruments.curves import ZeroCurve
    from nautilus_fi.instruments.derivatives import InterestRateSwap
    from nautilus_fi.overlays import swap_contract

    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    curves = {d: ZeroCurve.flat(d, 0.04 + 0.0005 * i) for i, d in enumerate(days)}
    par = InterestRateSwap("IRS5Y", 10_000_000, "USD", date(2024, 1, 2), date(2029, 1, 2), 0.0, pay_fixed=True)
    par.fixed_rate = par.value(days[0], curves[days[0]])["par_rate"]
    c = swap_contract(par, curves)
    assert c.prices.iloc[0] == pytest.approx(100.0, abs=1e-6)                    # an at-market swap is worth par
    assert c.prices.iloc[-1] > c.prices.iloc[0]                                   # payer gains as rates rise
    assert (c.exposures["dv01"] > 0).all() and c.exposures["oad"].iloc[0] > 3.0
