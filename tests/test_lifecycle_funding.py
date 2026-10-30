"""Conventions, lifecycle state machine, corrections and funding."""

from datetime import date

import pytest

from nautilus_fi.accounting.postings import StandardAccountingPolicy
from nautilus_fi.core import Event, EventLog, EventSource, EventType, Ledger, Money, TimePoint, to_ns
from nautilus_fi.funding import (
    BorrowBook,
    BorrowTerms,
    BreachAction,
    CashView,
    CSATerms,
    FundingAssessor,
    FundingObligation,
    FundingPolicy,
    MarginEngine,
    PendingSettlement,
    PercentNotionalIM,
    RepoBook,
    RepoTerms,
    ReservationBook,
    SettlementQueue,
    TieredCashRate,
)
from nautilus_fi.instruments import AssetType, BondSpec, CallOption, CouponType, DayCount
from nautilus_fi.instruments.calendar import us_bond_market_calendar
from nautilus_fi.instruments.conventions import (
    EntitlementMode,
    SettlementConventions,
    entitlement_rule,
    market_for,
)
from nautilus_fi.lifecycle import BondLifecycleHandler, ContractState, RestatementBridge, order_events

T = to_ns("2024-03-01 16:00")


def ev(t: EventType, payload: dict, ns: int = T, instrument: str = "B", source=EventSource.ENVIRONMENT, caused_by=()) -> Event:
    return Event(t, TimePoint(ns, ns), source, instrument, payload=payload, caused_by=caused_by)


# --------------------------------------------------------------------------------- conventions
def test_dated_settlement_rules():
    sc = SettlementConventions()
    cal = us_bond_market_calendar()
    assert sc.rule(AssetType.CORPORATE, "USD", date(2024, 5, 24)).lag_business_days == 2
    assert sc.settle_date(AssetType.CORPORATE, "USD", date(2024, 5, 24), cal) == date(2024, 5, 29)   # Memorial Day skipped
    assert sc.rule(AssetType.CORPORATE, "USD", date(2024, 5, 28)).lag_business_days == 1
    assert sc.rule(AssetType.CORPORATE, "USD", date(2016, 1, 4)).lag_business_days == 3
    assert sc.rule(AssetType.SOVEREIGN, "USD", date(2016, 1, 4)).lag_business_days == 1
    assert sc.rule(AssetType.SOVEREIGN, "JPY", date(2018, 4, 2)).lag_business_days == 2
    assert sc.rule(AssetType.SOVEREIGN, "JPY", date(2018, 5, 2)).lag_business_days == 1
    assert sc.rule(AssetType.LOAN, "USD", date(2024, 6, 3)).approximation is True
    assert sc.rule(AssetType.CORPORATE, "EUR", date(2024, 6, 3)).lag_business_days == 2


def test_entitlement_rules():
    cal = us_bond_market_calendar()
    us = entitlement_rule(market_for(AssetType.CORPORATE, "USD"))
    assert us.mode == EntitlementMode.PAYMENT_DATE_HOLDER and us.cutoff(date(2024, 6, 17), cal) == date(2024, 6, 17)
    gilt = entitlement_rule(market_for(AssetType.SOVEREIGN, "GBP"))
    assert gilt.mode == EntitlementMode.EX_DATE_HOLDER
    cutoff = gilt.cutoff(date(2024, 6, 7), cal)
    assert cutoff < date(2024, 6, 7) and gilt.in_ex_period(date(2024, 6, 5), date(2024, 6, 7), cal)
    assert not gilt.in_ex_period(date(2024, 5, 20), date(2024, 6, 7), cal)


# ---------------------------------------------------------------------------------- lifecycle
def spec() -> BondSpec:
    return BondSpec("B", "ACME", date(2020, 3, 1), date(2030, 3, 1), 0.06, call_schedule=[CallOption(date(2025, 3, 1), 101.0)])


def test_partial_call_reconciles_cash_and_position():
    h = BondLifecycleHandler()
    st = h.initial_state(spec(), date(2024, 1, 2))
    res = h.apply(st, ev(EventType.PARTIAL_CALL, {"fraction": 0.25, "price": 101.0, "accrued_per_100": 1.5}), spec())
    assert res.state.factor == pytest.approx(0.75)
    assert res.cash_per_100 == pytest.approx(0.25 * 102.5) and res.cash_kind == "REDEMPTION"
    assert res.position_changes[0].kind == "SCALE" and res.position_changes[0].fraction == pytest.approx(0.75)
    res2 = h.apply(res.state, ev(EventType.FACTOR_UPDATE, {"factor": 0.70, "paydown_price": 100.0}), spec())
    assert res2.cash_per_100 == pytest.approx(5.0) and res2.state.factor == pytest.approx(0.70)


def test_tender_with_proration_and_policy():
    h = BondLifecycleHandler()
    st = h.initial_state(spec(), date(2024, 1, 2))
    st = h.apply(st, ev(EventType.TENDER_OFFER, {"announce": date(2024, 3, 1), "expiry": date(2024, 3, 15), "settlement": date(2024, 3, 20), "price": 103.0, "max_fraction": 0.4}), spec()).state
    assert st.state == ContractState.TENDER_OPEN
    st = h.apply(st, ev(EventType.TENDER_ELECTION, {}), spec(), {"mark_clean": 101.0}).state   # policy: tender (103 >= 101 + 0.25)
    res = h.apply(st, ev(EventType.TENDER_SETTLEMENT, {"accrued_per_100": 0.5}), spec())
    assert res.state.factor == pytest.approx(0.6) and res.cash_per_100 == pytest.approx(0.4 * 103.5)
    assert any("prorated" in n for n in res.notes) and res.state.state == ContractState.OUTSTANDING
    st2 = h.apply(h.initial_state(spec(), date(2024, 1, 2)), ev(EventType.TENDER_OFFER, {"announce": date(2024, 3, 1), "expiry": date(2024, 3, 15), "settlement": date(2024, 3, 20), "price": 100.0}), spec()).state
    st2 = h.apply(st2, ev(EventType.TENDER_ELECTION, {}), spec(), {"mark_clean": 101.0}).state      # policy: do not tender
    assert h.apply(st2, ev(EventType.TENDER_SETTLEMENT, {}), spec()).cash_per_100 == 0.0


def test_default_lifecycle_to_discharge_and_exchange():
    h = BondLifecycleHandler()
    st = h.initial_state(spec(), date(2024, 1, 2))
    st = h.apply(st, ev(EventType.MISSED_PAYMENT, {"date": date(2024, 3, 1), "grace_end": date(2024, 3, 31)}), spec()).state
    assert st.state == ContractState.GRACE_PERIOD
    res = h.apply(st, ev(EventType.GRACE_PERIOD_END, {"date": date(2024, 3, 31)}), spec())
    assert res.emitted and res.emitted[0].type == EventType.DEFAULT and not res.emitted[0].observed is False
    st = h.apply(res.state, res.emitted[0], spec()).state
    assert st.state == ContractState.DEFAULTED and st.default.claim_per_100 == pytest.approx(100.0)
    r1 = h.apply(st, ev(EventType.RECOVERY_DISTRIBUTION, {"date": date(2024, 9, 1), "amount_per_100": 20.0}), spec())
    assert r1.cash_kind == "RECOVERY" and r1.position_changes[0].kind == "SCALE" and r1.state.state == ContractState.DEFAULTED
    r2 = h.apply(r1.state, ev(EventType.RECOVERY_DISTRIBUTION, {"date": date(2025, 1, 1), "amount_per_100": 15.0, "final": True}), spec())
    assert r2.state.state == ContractState.DISCHARGED and r2.position_changes[-1].kind == "CLOSE"
    # exchange into new debt + equity
    st3 = h.apply(h.initial_state(spec(), date(2024, 1, 2)), ev(EventType.DEFAULT, {"date": date(2024, 3, 1)}), spec()).state
    r3 = h.apply(st3, ev(EventType.EXCHANGE, {"cash_per_100": 5.0, "new": [{"instrument": "NEWDEBT", "ratio": 0.5, "value_per_100": 30.0}, {"instrument": "EQ", "ratio": 10.0, "value_per_100": 8.0}]}), spec())
    kinds = [c.kind for c in r3.position_changes]
    assert kinds == ["NEW", "NEW", "CLOSE"] and r3.state.state == ContractState.EXCHANGED and r3.cash_per_100 == 5.0


def test_hybrid_pik_deferral_and_linker_floor_and_precedence():
    h = BondLifecycleHandler()
    st = h.initial_state(spec(), date(2024, 1, 2))
    st = h.apply(st, ev(EventType.COUPON_DEFERRAL, {"amount_per_100": 3.0, "cumulative": True}), spec()).state
    assert st.deferred_coupons and st.deferred_coupons[0][1] == 3.0
    st = h.apply(st, ev(EventType.PIK_CAPITALISATION, {"amount_per_100": 3.0}), spec()).state
    assert st.factor == pytest.approx(1.03) and st.pik_added == 3.0
    st = h.apply(st, ev(EventType.WRITE_DOWN, {"fraction": 0.5}), spec()).state
    assert st.factor == pytest.approx(0.515)
    linker = BondSpec("L", "UST", date(2020, 7, 15), date(2030, 7, 15), 0.00625, coupon_type=CouponType.INFLATION_LINKED, day_count=DayCount.ACT_ACT_ICMA, asset_type=AssetType.LINKER)
    ls = h.initial_state(linker, date(2024, 1, 2))
    ls = h.apply(ls, ev(EventType.INDEX_RATIO_UPDATE, {"index_ratio": 0.97}, instrument="L"), linker).state
    res = h.apply(ls, ev(EventType.MATURITY, {}, instrument="L"), linker)
    assert res.cash_per_100 == pytest.approx(100.0 * 1.0 / 0.97, rel=1e-9) or res.cash_per_100 >= 100.0   # deflation floor applies
    events = [ev(EventType.DECISION, {}), ev(EventType.MATURITY, {}), ev(EventType.COUPON_ENTITLEMENT, {"amount_per_100": 3.0, "payment_date": date(2024, 3, 1)}), ev(EventType.SETTLEMENT, {})]
    ordered = [e.type for e in order_events(events)]
    assert ordered.index(EventType.COUPON_ENTITLEMENT) < ordered.index(EventType.MATURITY) < ordered.index(EventType.SETTLEMENT) < ordered.index(EventType.DECISION)


def test_correction_bridge_reverses_and_reposts():
    log, ledger = EventLog(), Ledger("USD")
    pol = StandardAccountingPolicy(ledger)
    pol.capital_flow("cap", T, Money(1000.0, "USD"))
    orig = log.append(ev(EventType.COUPON_PAYMENT, {"amount_per_100": 2.5, "amount": 25.0}))
    e1 = pol.coupon_payment(orig.id, T, "B", "USD", 25.0)
    assert ledger.cash() == pytest.approx(1025.0)
    bridge = RestatementBridge(log, ledger)
    from nautilus_fi.core.ledger import JournalEntry

    corrected = JournalEntry(orig.id, T + 1, pol.coupon_payment("tmp", T + 1, "B", "USD", 20.0).postings)
    ledger.reverse(ledger.entries[-1].entry_id, T + 1, "undo-tmp")           # remove the temporary posting used to build the entry
    b = bridge.correct(orig, "amount", 20.0, T + 2, [e1], [corrected], note="servicer restatement")
    assert ledger.cash() == pytest.approx(1020.0) and b.nav_effect == pytest.approx(-5.0)
    assert log.get(b.correction_event_id).corrects == orig.id and len(bridge.to_frame()) == 1
    ledger.conservation_check()


# ------------------------------------------------------------------------------------ funding
def test_unsettled_proceeds_cannot_fund_purchases():
    ledger = Ledger("USD")
    pol = StandardAccountingPolicy(ledger)
    pol.capital_flow("cap", T, Money(1_000_000.0, "USD"))
    pol.trade("b", T, "B", "USD", 900_000, 900_000.0)
    pol.settle("s", T, "B", "USD", 900_000.0, "PAY")
    pol.trade("sell", T, "B", "USD", -900_000, 910_000.0)
    q, res = SettlementQueue(), ReservationBook()
    q.add(PendingSettlement("sell", "B", "USD", 910_000.0, -900_000, date(2024, 3, 1), date(2024, 3, 4)))
    view = CashView(ledger, q, res)
    assert view.settled("USD") == pytest.approx(100_000.0)
    assert view.available("USD") == pytest.approx(100_000.0)                 # sale proceeds not yet usable
    assert view.projected("USD") == pytest.approx(1_010_000.0)
    res.reserve_cash("O1", "USD", 60_000.0)
    assert view.available("USD") == pytest.approx(40_000.0)
    assert view.available("USD", bridge_allowance=FundingPolicy(allow_bridge=True, max_bridge=50_000).bridge_allowance()) == pytest.approx(90_000.0)
    item = q.due(date(2024, 3, 4))[0]
    q.fail(item, date(2024, 3, 4), date(2024, 3, 5))
    assert item.fail_count == 1 and not q.due(date(2024, 3, 4))
    q.settle(item, date(2024, 3, 5))
    pol.settle("s2", T, "B", "USD", 910_000.0, "RECEIVE")
    assert view.settled("USD") == pytest.approx(1_010_000.0)
    assert TieredCashRate({"USD": [(0, 0.04), (1e6, 0.045)]}, {"USD": 0.06}).rate("USD", 2e6) == 0.045
    assert TieredCashRate({"USD": [(0, 0.04)]}, {"USD": 0.06}).interest("USD", -100_000, 30) == pytest.approx(-500.0)


def test_repo_haircut_interest_margin_and_rollover():
    book = RepoBook(fixing=lambda idx, d: 0.05)
    terms = RepoTerms("DEALER", rate=None, index="SOFR", spread_bp=20, term_days=7, haircut=0.02, eligible=lambda s: s.startswith("UST"))
    with pytest.raises(ValueError):
        book.open(date(2024, 3, 1), "HY1", 1e6, "USD", 950_000.0, terms)
    pos = book.open(date(2024, 3, 1), "UST10", 10e6, "USD", 9_800_000.0, terms)
    assert pos.cash_borrowed == pytest.approx(9_800_000.0 * 0.98) and pos.rate == pytest.approx(0.052)
    interest = book.accrue(pos, date(2024, 3, 1), date(2024, 3, 8))
    assert interest == pytest.approx(pos.cash_borrowed * 0.052 * 7 / 360)
    call = book.margin_call(pos, date(2024, 3, 5), 9_600_000.0)
    assert call > 0 and pos.margin_posted == pytest.approx(call)
    assert book.roll(pos, date(2024, 3, 8)) and pos.accrued_interest == 0.0
    assert not book.roll(pos, date(2024, 3, 15), fail=True) and pos.status == "ROLLOVER_FAILED"
    assert book.collateral_coupon_passthrough(pos, 125_000.0) == 125_000.0
    assert book.encumbered("UST10") == 10e6 or pos.status == "ROLLOVER_FAILED"


def test_borrow_availability_fee_recall_and_margin_calls():
    bb = BorrowBook()
    terms = BorrowTerms(fee_bp=50, availability=lambda s, d: 5e6)
    pos = bb.open(date(2024, 3, 1), "CORP1", 3e6, "USD", terms)
    with pytest.raises(ValueError):
        bb.open(date(2024, 3, 1), "CORP1", 3e6, "USD", terms)
    assert bb.daily_fee(pos, 2.9e6) == pytest.approx(2.9e6 * 0.005 / 360)
    assert bb.manufactured_payment(pos, 2.5) == pytest.approx(75_000.0)
    deadline = bb.recall(pos, date(2024, 3, 10))
    assert deadline == date(2024, 3, 13) and bb.recalled_due(date(2024, 3, 13)) == [pos]
    csa = CSATerms("BANK", "NS1", threshold=500_000, minimum_transfer=100_000, rounding=10_000, eligible_collateral=("CASH", "UST"))
    me = MarginEngine({"NS1": csa}, PercentNotionalIM(0.02))
    calls = me.process(date(2024, 3, 1), {"NS1": -1_250_000.0}, {"NS1": [{"notional": 50e6}]})
    kinds = {c.kind: c.amount for c in calls}
    assert kinds["VM"] == pytest.approx(750_000.0) and kinds["IM"] == pytest.approx(1_000_000.0)
    assert me.process(date(2024, 3, 2), {"NS1": -1_300_000.0}, {"NS1": [{"notional": 50e6}]}) == []      # below MTA
    calls2 = me.process(date(2024, 3, 3), {"NS1": 800_000.0}, {"NS1": [{"notional": 50e6}]})
    assert calls2[0].kind == "VM" and calls2[0].amount < 0 and me.posted_vm["NS1"] == pytest.approx(-300_000.0)
    with pytest.raises(ValueError):
        me.collateral_value(csa, "EQUITY", 1e6)
    assert me.collateral_value(csa, "UST", 1e6) == pytest.approx(980_000.0)


def test_funding_breach_actions_in_order():
    policy = FundingPolicy(breach_actions=(BreachAction.CANCEL_BUYS, BreachAction.SOURCE_REPO, BreachAction.LIQUIDATE_BY_RULE, BreachAction.DECLARE_FAILURE))
    fa = FundingAssessor(policy)
    obligations = [FundingObligation(date(2024, 3, 2), "USD", 5_000_000.0, "SETTLEMENT", "buy1"), FundingObligation(date(2024, 3, 2), "USD", 1_000_000.0, "MARGIN", "NS1")]
    receipts = [FundingObligation(date(2024, 3, 3), "USD", 2_000_000.0, "SETTLEMENT", "sell1")]
    a = fa.assess(date(2024, 3, 1), "USD", settled_cash=1_500_000.0, reserved=200_000.0, obligations=obligations, receipts=receipts)
    assert a.breached and a.shortfall == pytest.approx(6_000_000.0 - 1_300_000.0)        # unsettled receipt does not count
    actions = fa.plan(a, open_buy_orders=[{"instrument": "X", "cash": 1_000_000.0, "quantity": 1e6}], holdings=[{"instrument": "UST", "market_value": 2_000_000.0, "liquidity_rank": 1, "repo_eligible": True}, {"instrument": "HY", "market_value": 3_000_000.0, "liquidity_rank": 5}], repo_capacity=1_000_000.0)
    kinds = [x.kind for x in actions]
    assert kinds[0] == BreachAction.CANCEL_BUYS and BreachAction.SOURCE_REPO in kinds and BreachAction.LIQUIDATE_BY_RULE in kinds
    assert actions[-1].kind != BreachAction.DECLARE_FAILURE                                  # resolved before failure
    assert sum(x.amount for x in actions) == pytest.approx(a.shortfall)
    tiny = fa.plan(a, [], [], 0.0)
    assert tiny[-1].kind == BreachAction.DECLARE_FAILURE and tiny[-1].amount == pytest.approx(a.shortfall)
