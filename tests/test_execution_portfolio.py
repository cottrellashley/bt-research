"""Execution process, portfolio construction, constraints, forced actions, risk, attribution, returns."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nautilus_fi.accounting.attribution import AttributionInputs, attribute_instrument, attribute_portfolio
from nautilus_fi.accounting.returns import compute_returns, money_weighted_return, time_weighted_returns
from nautilus_fi.execution import (
    GROSS,
    NET_FULL,
    LiquidityStats,
    ParametricCostModel,
    VolumeGatedLiquidityModel,
)
from nautilus_fi.execution.process import (
    ExecutionContext,
    LiquidityBudget,
    OrderProcessSimulator,
    OrderRequest,
    OrderState,
    ProcessConfig,
    VenueType,
)
from nautilus_fi.instruments import BondSpec
from nautilus_fi.instruments.calendar import us_bond_market_calendar
from nautilus_fi.portfolio import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
    ForcedActionPolicy,
    PortfolioConstructor,
    TargetPortfolio,
    compute_risk,
    full_revaluation,
)

CAL = us_bond_market_calendar()
NS = 1_000_000_000


def bond(sym="B", issuer="ACME") -> BondSpec:
    return BondSpec(sym, issuer, date(2020, 1, 15), date(2030, 1, 15), 0.05, amount_outstanding=1e9, rating_initial="BBB")


def ctx(d=date(2024, 3, 5), hour=10, ref=99.0, stats=None, stress=0.0, borrow=None, reoffer=None, alloc=0.0) -> ExecutionContext:
    now = int(pd.Timestamp(f"{d} {hour:02d}:00", tz="UTC").value)
    return ExecutionContext(d, now, ref, "PRIOR_CLOSE", None if ref is None else ref - 0.1, None if ref is None else ref + 0.1, stats or LiquidityStats(volume_5d=5e6, trade_count_20d=15, days_since_trade=0, quote_score=0.8), stress, 9, 5.0, borrow_available=borrow, new_issue_reoffer=reoffer, new_issue_allocation=alloc)


def req(order_id="O1", side="BUY", face=2e6, limit=None, created=date(2024, 3, 4), protocol="RFQ", short=False) -> OrderRequest:
    return OrderRequest(order_id, "B", side, face, limit, int(pd.Timestamp(f"{created} 16:45", tz="UTC").value), created, protocol, is_short=short)


def sim(assumptions=NET_FULL, **cfg) -> OrderProcessSimulator:
    return OrderProcessSimulator(VolumeGatedLiquidityModel(), ParametricCostModel(), assumptions, ProcessConfig(**cfg))


def test_process_eligibility_delay_and_reference():
    s = sim(restricted={"B"})
    assert s.evaluate(bond(), req(), ctx(), CAL).state == OrderState.REJECTED
    s = sim()
    assert s.evaluate(bond(), req(), ctx(hour=20), CAL).reason == "outside trading hours"
    s = sim(approval_delay_business_days=2)
    out = s.evaluate(bond(), req(created=date(2024, 3, 4)), ctx(d=date(2024, 3, 5)), CAL)
    assert out.state == OrderState.CREATED and "approval" in out.reason
    assert sim().evaluate(bond(), req(short=True), ctx(), CAL).state == OrderState.REJECTED
    assert sim(allow_shorts=True).evaluate(bond(), req(side="SELL", short=True), ctx(borrow=1e6), CAL).reason.startswith("borrow unavailable")
    assert sim().evaluate(bond(), req(), ctx(ref=None), CAL).reason == "no reference price"


def test_rfq_quotes_limit_and_cost_decomposition():
    from nautilus_fi.execution import NET_SPREAD

    s = sim(NET_SPREAD, n_dealers=6, quote_dispersion_bp=2.0)      # deterministic: no fill-probability draw
    out = s.evaluate(bond(), req(), ctx(), CAL)
    assert out.state in (OrderState.FILLED, OrderState.PARTIALLY_FILLED) and out.venue == VenueType.RFQ
    assert any(q.responded for q in out.quotes) and out.fills[0][0] > 99.0           # buys pay above the reference
    assert out.cost is not None and out.cost.half_spread_bp > 0 and "ref 99.0000" in out.cost.explain()
    tight = s.evaluate(bond(), req(order_id="O2", limit=99.01), ctx(), CAL)
    assert tight.state == OrderState.QUOTED and "outside limit" in tight.reason and not tight.fills
    gross = sim(GROSS).evaluate(bond(), req(order_id="O3"), ctx(), CAL)
    assert gross.fills[0][0] == pytest.approx(99.0) and gross.cost.total_bp == 0.0
    big = sim(NET_SPREAD).evaluate(bond(), req(order_id="O4", face=20e6), ctx(), CAL)
    assert big.cost is None or big.cost.impact_bp >= 0
    gated = sim(NET_FULL)
    outcomes = [gated.evaluate(bond(), req(order_id=f"P{k}"), ctx(stats=LiquidityStats(volume_5d=0.0, trade_count_20d=0, days_since_trade=40, quote_score=0.2)), CAL) for k in range(20)]
    assert sum(1 for o in outcomes if o.fills) < 20 and all(o.p_fill < 0.5 for o in outcomes)   # inactive bonds rarely fill


def test_liquidity_budget_is_shared_and_released():
    b = LiquidityBudget()
    b.set_capacity("B", date(2024, 3, 5), 3e6)
    assert b.offer("O1", "B", date(2024, 3, 5), 2e6) == 2e6 and b.offer("O2", "B", date(2024, 3, 5), 2e6) == 1e6
    assert b.available("B", date(2024, 3, 5)) == 0.0
    b.settle("O1", 0.5e6)                                  # O1 only filled 0.5mm: the rest is released
    assert b.available("B", date(2024, 3, 5)) == pytest.approx(1.5e6)
    b.release_all_offers()
    assert b.available("B", date(2024, 3, 5)) == pytest.approx(2.5e6)
    s = sim()
    first = s.evaluate(bond(), req(order_id="A", face=4e6), ctx(), CAL)
    second = s.evaluate(bond(), req(order_id="B2", face=4e6), ctx(), CAL)
    assert first.offered_face + second.offered_face <= s.budget.capacity[("B", date(2024, 3, 5))] + 1e-6


def test_primary_allocation_and_no_response_in_stress():
    s = sim()
    out = s.evaluate(bond(), req(protocol="NEW_ISSUE", face=5e6), ctx(reoffer=99.75, alloc=0.3), CAL)
    assert out.venue == VenueType.PRIMARY and out.fills == [(99.75, 1.5e6)] and out.state == OrderState.PARTIALLY_FILLED
    stressed = sim(dealer_response_base=0.05).evaluate(bond(), req(order_id="S"), ctx(stress=1.5, stats=LiquidityStats(volume_5d=5e6, trade_count_20d=15, days_since_trade=0, quote_score=0.2)), CAL)
    assert stressed.state in (OrderState.ROUTED, OrderState.QUOTED) and not stressed.fills


def test_constraints_construction_and_forced_actions():
    cs = ConstraintSet([
        Constraint("issuer<=5%", ConstraintKind.ISSUER_WEIGHT, "weight", maximum=0.05),
        Constraint("sector FIN<=30%", ConstraintKind.GROUP_WEIGHT, "weight", maximum=0.30, hard=False, group_col="sector"),
        Constraint("duration 3-6", ConstraintKind.DURATION, "years", minimum=3.0, maximum=6.0),
        Constraint("no CCC", ConstraintKind.RATING_FLOOR, "score", maximum=16),
    ])
    rows = pd.DataFrame({"symbol": ["A", "B"], "issuer": ["ACME", "ACME"], "sector": ["FIN", "FIN"], "weight": [0.6, 0.4], "duration": [4.0, 8.0], "rating_score": [9, 17]})
    v = cs.evaluate(rows, nav=100e6)
    kinds = {x.constraint.name for x in v}
    assert "issuer<=5%" in kinds and "no CCC" in kinds and "duration 3-6" not in kinds
    cs.relax("sector FIN<=30%", 0.35, "feasibility")
    assert cs.relaxations and cs.constraints[1].maximum == 0.35
    bonds = {"A": bond("A", "ACME"), "B": bond("B", "ACME"), "C": bond("C", "OTHER")}
    pc = PortfolioConstructor(ConstraintSet([Constraint("issuer<=50%", ConstraintKind.ISSUER_WEIGHT, "weight", maximum=0.5)]))
    res = pc.build(TargetPortfolio(weights={"A": 0.3, "C": 0.3}), bonds, nav=10e6, dirty_per_face={"A": 1.0, "B": 1.0, "C": 1.0},
                   holdings_face={"B": 2e6}, live_orders=[{"order_id": "L1", "symbol": "A", "side": "BUY", "leaves_face": 1e6}], available_cash=2.5e6,
                   position_rows=pd.DataFrame({"symbol": ["B"], "issuer": ["ACME"], "weight": [0.2]}))
    by = {i.symbol: i for i in res.intents}
    assert by["B"].side == "SELL" and by["B"].face == 2e6 and res.intents[0].side == "SELL"       # sells first
    assert by["A"].face <= 2e6 and by["C"].face <= 3e6 and res.scaled_buys < 1.0                   # live order netted, cash-scaled
    fp = ForcedActionPolicy(max_rating_score=10, downgrade_action="DELAYED", downgrade_grace_days=10)
    assert fp.evaluate(date(2024, 3, 1), {"X": 1e6}, {"X": 12}) == []
    intents = fp.evaluate(date(2024, 3, 12), {"X": 1e6}, {"X": 12})
    assert len(intents) == 1 and intents[0].side == "SELL" and "DOWNGRADE" in intents[0].reason
    gf = ForcedActionPolicy(downgrade_action="GRANDFATHER")
    assert gf.evaluate(date(2024, 3, 1), {"X": 1e6}, {"X": 12}) == [] and gf.evaluate(date(2025, 3, 1), {"X": 1e6}, {"X": 12}) == []


def test_risk_and_full_revaluation():
    rows = pd.DataFrame({"symbol": ["A", "B", "S"], "market_value": [50e6, 30e6, -20e6], "duration": [5.0, 2.0, 7.0], "spread_duration": [4.5, 1.8, 6.5], "z_spread_bp": [150, 80, 300], "convexity": [30, 5, 60], "currency": ["USD", "EUR", "USD"], "rating_bucket": ["BBB", "A", "BB"], "recovery": [0.4, 0.4, 0.3], "krd_5": [4.0, 0.5, 6.0]})
    rr = compute_risk(rows, nav=60e6, liquidity_daily_capacity={"A": 5e6, "B": 1e6, "S": 2e6})
    assert rr.dv01 == pytest.approx((50e6 * 5 + 30e6 * 2 - 20e6 * 7) * 1e-4)
    assert rr.key_rate_dv01["5"] == pytest.approx((50e6 * 4 + 30e6 * 0.5 - 20e6 * 6) * 1e-4)
    assert rr.fx_delta["EUR"] == 30e6 and rr.jump_to_default == pytest.approx(-20e6 * 0.7) and rr.funding_need == 20e6
    assert rr.standalone_total > rr.netted_total and rr.days_to_liquidate_p90 >= 10
    fr = full_revaluation([{"v": 100.0, "dur": 5.0}], lambda p, s: p["v"] * (1 - p["dur"] * s.get("bp", 0) / 1e4), [{"name": "+100bp", "bp": 100}, {"name": "-50bp", "bp": -50}])
    assert fr.loc[0, "pnl"] == pytest.approx(-5.0) and fr.loc[1, "pnl"] == pytest.approx(2.5)


def test_attribution_reconciles_and_returns_handle_flows():
    x = AttributionInputs("A", date(2024, 3, 5), 1e6, 99.0, 98.6, 1.0, 1.05, 0.0, 5.0, 4.5, 10.0, 5.0, 1.0, 1.0, execution_cost=500.0)
    comp = attribute_instrument(x)
    assert comp["total"] == pytest.approx(((98.6 + 1.05) - (99.0 + 1.0)) * 1e4 - 500.0)
    assert sum(comp[k] for k in ("carry", "rates", "spread", "optionality", "fx", "defaults_recoveries", "execution", "financing_fees", "model_change", "residual")) == pytest.approx(comp["total"])
    assert comp["rates"] == pytest.approx(-5.0 * 10 / 1e4 * 99.0 * 1e4) and comp["carry"] == pytest.approx(0.05 * 1e4)
    rep = attribute_portfolio([x], ledger_pnl=comp["total"] - 1_000.0, financing_fees=-1_000.0)
    assert abs(rep.residual_vs_ledger) < 1e-6 and rep.by_component()["financing_fees"] == -1_000.0
    rep2 = attribute_portfolio([x], ledger_pnl=comp["total"] + 777.0)
    assert rep2.notes and rep2.rows.iloc[-1]["symbol"] == "UNEXPLAINED"
    idx = pd.to_datetime(["2024-01-02", "2024-07-02", "2025-01-02"])
    nav = pd.Series([100.0, 112.0, 113.12], index=idx)
    flows = pd.Series([0.0, 10.0, 0.0], index=idx)
    twr = time_weighted_returns(nav, flows)
    assert twr.iloc[1] == pytest.approx(112 / 110 - 1) and twr.iloc[2] == pytest.approx(0.01)
    rs = compute_returns(nav, flows)
    assert rs.twr_cumulative.iloc[-1] == pytest.approx((112 / 110) * 1.01 - 1) and rs.mwr_annual is not None
    assert money_weighted_return(pd.DatetimeIndex(["2024-01-01", "2025-01-01"]), np.array([-100.0, 0.0]), 110.0) == pytest.approx(0.10, abs=1e-3)
    gross = compute_returns(nav, flows, denominator="GROSS", gross_exposure=pd.Series([200.0, 200.0, 200.0], index=idx))
    assert gross.denominator == "GROSS" and gross.twr_daily.iloc[1] == pytest.approx(2.0 / 200.0)
    assert 0.0 < rs.mwr_annual < 0.1
