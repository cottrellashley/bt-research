"""Core: typed quantities, time model, event log, double-entry ledger and accounting fixtures,
registry / support matrix, assumptions, manifest, restricted context."""

from datetime import date

import pytest

from nautilus_fi.accounting.postings import AccountingSettings, LotPolicy, StandardAccountingPolicy
from nautilus_fi.core import (
    Account,
    Event,
    EventLog,
    EventSource,
    EventType,
    Ledger,
    Money,
    Notional,
    NotionalKind,
    Observation,
    Price,
    PriceKind,
    PrincipalBasis,
    QuoteBasis,
    TimePoint,
    UnitError,
    market_value,
    to_ns,
)
from nautilus_fi.core.assumptions import AssumptionSet
from nautilus_fi.core.context import DecisionContext, InformationBoundaryError
from nautilus_fi.core.ledger import UnbalancedEntry
from nautilus_fi.core.manifest import CounterfactualSpec, ExperimentManifest, RunMode, hash_frame
from nautilus_fi.core.quantities import Compounding, Rate
from nautilus_fi.core.registry import (
    Capability,
    DependencyGraph,
    ModelRegistration,
    ModelRegistry,
    Node,
    SupportMatrix,
    SupportStatus,
    UnsupportedCombination,
)
from nautilus_fi.core.time import DataVintage

T0 = to_ns("2024-01-02 16:00")


# --------------------------------------------------------------------------------------- quantities
def test_money_requires_same_currency():
    assert (Money(100, "USD") + Money(5, "USD")).amount == 105
    with pytest.raises(UnitError):
        Money(1, "USD") + Money(1, "EUR")
    assert str(Money(1234.567, "USD").rounded()) == "1,234.57 USD"


def test_price_conversions_need_explicit_inputs():
    bill = Price(0.0525, QuoteBasis.DISCOUNT_YIELD, PriceKind.NOT_APPLICABLE)
    with pytest.raises(UnitError):
        bill.to_pct_of_par()
    assert bill.to_pct_of_par(days_to_maturity=182, year_basis=360).value == pytest.approx(100 * (1 - 0.0525 * 182 / 360))
    dirty = Price(101.5, kind=PriceKind.DIRTY)
    with pytest.raises(UnitError):
        dirty.to_pct_of_par()
    assert dirty.to_pct_of_par(accrued_pct=1.5).value == pytest.approx(100.0)
    assert Price(0.985, QuoteBasis.DECIMAL).to_pct_of_par().value == pytest.approx(98.5)
    assert Price(110.5, QuoteBasis.POINTS).to_pct_of_par(conversion_factor=0.9).value == pytest.approx(99.45)


def test_market_value_enforces_principal_basis():
    px = Price(98.0)
    orig = Notional(1_000_000, NotionalKind.ORIGINAL, "USD")
    cur = Notional(800_000, NotionalKind.CURRENT, "USD")
    assert market_value(px, orig, factor=0.8).amount == pytest.approx(784_000)
    assert market_value(px, cur).amount == pytest.approx(784_000)           # factor not applied twice
    adj = Price(78.4, principal_basis=PrincipalBasis.ORIGINAL)
    assert market_value(adj, orig, factor=0.8).amount == pytest.approx(784_000)
    with pytest.raises(UnitError):
        market_value(adj, cur, factor=0.8)                                   # would understate value
    real = Price(98.0, principal_basis=PrincipalBasis.INDEXED)
    assert market_value(real, orig, index_ratio=1.12, accrued_pct_current=0.5).amount == pytest.approx(1_000_000 * 1.12 * 98.5 / 100)
    with pytest.raises(UnitError):
        market_value(Price(0.05, QuoteBasis.YIELD), orig)


def test_rate_compounding_conversion():
    r = Rate(0.05, Compounding.SEMI_ANNUAL)
    cont = r.to_compounding(Compounding.CONTINUOUS)
    assert cont.value == pytest.approx(2 * __import__("math").log(1.025))
    assert cont.to_compounding(Compounding.SEMI_ANNUAL).value == pytest.approx(0.05)


def test_observation_and_timepoint_ordering():
    with pytest.raises(UnitError):
        Observation("X", "mid", 99.0, economic_ns=10, available_ns=5)
    tp = TimePoint.derived([TimePoint(1, 5), TimePoint(2, 9)], computation_delay_ns=1, operational_delay_ns=2)
    assert tp.available_ns == 12 and tp.economic_ns <= tp.available_ns
    assert not tp.visible_at(11) and tp.visible_at(12)
    assert DataVintage(100).admits(99) and not DataVintage(100).admits(101)


# ---------------------------------------------------------------------------------------- event log
def test_event_log_dedupe_lineage_and_chain():
    log = EventLog()
    e1 = log.append(Event(EventType.COUPON_ENTITLEMENT, TimePoint(T0, T0), EventSource.LIFECYCLE, "BOND", payload={"amount": 25.0}))
    e2 = log.append(Event(EventType.COUPON_PAYMENT, TimePoint(T0 + 1, T0 + 1), EventSource.LIFECYCLE, "BOND", payload={"amount": 25.0}, caused_by=(e1.id,)))
    with pytest.raises(ValueError):
        log.append(Event(EventType.COUPON_PAYMENT, TimePoint(T0 + 1, T0 + 1), EventSource.LIFECYCLE, "BOND", payload={"amount": 25.0}, caused_by=(e1.id,), seq=e2.seq))
    with pytest.raises(ValueError):
        log.append(Event(EventType.MARK, TimePoint(T0, T0), EventSource.ENVIRONMENT, "BOND", caused_by=("nope",)))
    assert [e.type for e in log.lineage(e2.id)] == [EventType.COUPON_ENTITLEMENT, EventType.COUPON_PAYMENT]
    assert len(log.chain_hash) == 64 and len(log.to_frame()) == 2


# ------------------------------------------------------------------------------------------ ledger
def make_policy(cash: float = 0.0) -> StandardAccountingPolicy:
    ledger = Ledger("USD")
    pol = StandardAccountingPolicy(ledger, AccountingSettings(lot_policy=LotPolicy.FIFO))
    if cash:
        pol.capital_flow("cap", T0, Money(cash, "USD"))
    return pol


def test_blueprint_1030_fixture_coupon_and_paydown_create_no_profit():
    """1,000 principal at 100 plus 30 accrued = 1,030; receive 30 coupon and 100 principal -> still 1,030."""
    pol = make_policy(1_030.0)
    pol.trade("t", T0, "BOND", "USD", quantity=1_000, clean_value=1_000.0, accrued=30.0)
    pol.settle("s", T0, "BOND", "USD", 1_030.0, "PAY")
    led = pol.ledger
    assert led.nav().amount == pytest.approx(1_030.0)
    pol.coupon_payment("c", T0 + 1, "BOND", "USD", 30.0)
    pol.principal_reduction("p", T0 + 1, "BOND", "USD", fraction_of_position=0.1, paydown_value=100.0)
    assert led.nav().amount == pytest.approx(1_030.0)
    assert led.cash() == pytest.approx(130.0)
    assert led.balance(Account.SECURITIES, "BOND") == pytest.approx(900.0)
    assert led.balance(Account.ACCRUED_INTEREST, "BOND") == pytest.approx(0.0)
    assert led.balance(Account.REALIZED_PNL) == pytest.approx(0.0) and led.balance(Account.INCOME_COUPON) == pytest.approx(0.0)
    led.conservation_check()


def test_trade_settlement_mark_and_sale_realize_correctly():
    pol = make_policy(2_000_000.0)
    pol.trade("b1", T0, "BOND", "USD", 1_000_000, 980_000.0, accrued=5_000.0, fees=100.0)
    led = pol.ledger
    assert led.balance(Account.PAYABLE_UNSETTLED, "BOND") == pytest.approx(985_100.0)
    assert led.nav().amount == pytest.approx(2_000_000.0 - 100.0)            # only the fee leaves NAV
    pol.settle("s1", T0 + 1, "BOND", "USD", 985_100.0, "PAY")
    assert led.cash() == pytest.approx(2_000_000.0 - 985_100.0)
    pol.mark("m1", T0 + 2, "BOND", "USD", 990_000.0)
    assert led.balance(Account.UNREALIZED_PNL) == pytest.approx(10_000.0)
    pol.set_accrued("a1", T0 + 2, "BOND", "USD", 8_000.0)                    # accrual income 3,000
    assert led.balance(Account.INCOME_COUPON) == pytest.approx(3_000.0)
    pol.trade("s2", T0 + 3, "BOND", "USD", -400_000, 398_000.0, accrued=3_200.0)   # sell 40% at 99.5
    assert led.balance(Account.REALIZED_PNL) == pytest.approx(398_000.0 - 392_000.0)   # vs cost 392,000
    assert led.balance(Account.UNREALIZED_PNL) == pytest.approx(10_000.0 * 0.6)
    assert led.balance(Account.ACCRUED_INTEREST, "BOND") == pytest.approx(8_000.0 - 3_200.0)
    led.conservation_check()
    assert pol.book.quantity("BOND") == pytest.approx(600_000)


def test_short_sale_cover_and_manufactured_payment():
    pol = make_policy(1_000_000.0)
    pol.trade("ss", T0, "BOND", "USD", -500_000, 490_000.0, accrued=2_000.0)
    led = pol.ledger
    assert led.balance(Account.SECURITIES_SHORT, "BOND") == pytest.approx(490_000.0)
    pol.settle("st", T0 + 1, "BOND", "USD", 492_000.0, "RECEIVE")
    pol.mark("mk", T0 + 2, "BOND", "USD", -495_000.0)                         # price up: loss for the short
    assert led.balance(Account.UNREALIZED_PNL) == pytest.approx(-5_000.0)
    pol.manufactured_payment("mp", T0 + 3, "BOND", "USD", 12_500.0)
    pol.trade("cv", T0 + 4, "BOND", "USD", 500_000, 495_000.0, accrued=0.0)   # cover at 99
    assert led.balance(Account.REALIZED_PNL) == pytest.approx(490_000.0 - 495_000.0)
    assert abs(led.balance(Account.UNREALIZED_PNL)) < 1e-6
    led.conservation_check()


def test_default_claim_recovery_and_exchange():
    pol = make_policy(1_000_000.0)
    pol.trade("b", T0, "HY", "USD", 500_000, 450_000.0, accrued=10_000.0)
    pol.settle("s", T0, "HY", "USD", 460_000.0, "PAY")
    pol.mark("m", T0 + 1, "HY", "USD", 150_000.0)                            # distressed
    pol.default_nonaccrual("d", T0 + 2, "HY", "USD")
    led = pol.ledger
    assert led.balance(Account.ACCRUED_INTEREST, "HY") == pytest.approx(0.0) and led.balance(Account.EXPENSE_WRITEOFF) == pytest.approx(10_000.0)
    assert led.balance(Account.CONTRACTUAL_CLAIM, "HY") == pytest.approx(150_000.0) and led.balance(Account.SECURITIES, "HY") == pytest.approx(0.0)
    nav_before = led.nav().amount
    pol.exchange("x", T0 + 3, "HY", "USD", [("NEWDEBT", 200_000, 120_000.0), ("EQUITY", 1_000, 20_000.0)], cash=20_000.0)
    assert led.nav().amount == pytest.approx(nav_before + 10_000.0)          # 160k received for 150k carrying
    assert led.balance(Account.SECURITIES, "NEWDEBT") == pytest.approx(120_000.0)
    led.conservation_check()
    pol2 = make_policy(1_000_000.0)
    pol2.trade("b", T0, "HY", "USD", 500_000, 450_000.0)
    pol2.settle("s", T0, "HY", "USD", 450_000.0, "PAY")
    pol2.default_nonaccrual("d", T0 + 1, "HY", "USD")
    pol2.recovery("r", T0 + 2, "HY", "USD", cash=175_000.0)
    assert pol2.ledger.balance(Account.REALIZED_PNL) == pytest.approx(175_000.0 - 450_000.0)
    pol2.ledger.conservation_check()


def test_futures_vm_and_collateral_do_not_duplicate_pnl():
    pol = make_policy(1_000_000.0)
    pol.collateral_posted("im", T0, "CME", "USD", 50_000.0)
    led = pol.ledger
    assert led.cash() == pytest.approx(950_000.0) and led.nav().amount == pytest.approx(1_000_000.0)
    pol.futures_variation_margin("vm", T0 + 1, "TYZ4", "USD", -12_000.0)
    assert led.nav().amount == pytest.approx(988_000.0) and led.balance(Account.REALIZED_PNL) == pytest.approx(-12_000.0)
    pol.derivative_mark("irs", T0 + 1, "IRS1", "USD", -30_000.0)
    assert led.balance(Account.DERIVATIVE_LIABILITIES, "IRS1") == pytest.approx(30_000.0)
    pol.derivative_mark("irs2", T0 + 2, "IRS1", "USD", 10_000.0)
    assert led.balance(Account.DERIVATIVE_ASSETS, "IRS1") == pytest.approx(10_000.0) and led.balance(Account.DERIVATIVE_LIABILITIES, "IRS1") == pytest.approx(0.0)
    led.conservation_check()


def test_repo_financing_and_reversal():
    pol = make_policy(100_000.0)
    e = pol.repo_open("r1", T0, "REPO-1", "USD", 900_000.0)
    pol.repo_interest("r2", T0 + 1, "REPO-1", "USD", 1_250.0)
    led = pol.ledger
    assert led.cash() == pytest.approx(1_000_000.0) and led.nav().amount == pytest.approx(100_000.0 - 1_250.0)
    led.reverse(e.entry_id, T0 + 2, "corr")
    assert led.balance(Account.FINANCING_PAYABLE, "REPO-1") == pytest.approx(1_250.0)
    led.conservation_check()
    with pytest.raises(UnbalancedEntry):
        from nautilus_fi.core.ledger import JournalEntry, Posting

        led.post(JournalEntry("bad", T0, (Posting.dr(Account.CASH_SETTLED, Money(1, "USD")),)))


# --------------------------------------------------------------------------------- registry etc.
def test_support_matrix_and_dependency_validation():
    sm = SupportMatrix()
    sm.declare(Capability("CORPORATE", "partial_call", SupportStatus.VALIDATED, validation_case="partial call reconciliation"))
    sm.declare(Capability("CLO", "waterfall_triggers", SupportStatus.SUPPORTED_WITH_APPROXIMATION, approximation="simplified OC/IC tests"))
    assert sm.status("CORPORATE", "partial_call").status == SupportStatus.VALIDATED
    assert sm.status("CORPORATE", "partial_call", jurisdiction="JP").status == SupportStatus.UNSUPPORTED
    with pytest.raises(UnsupportedCombination):
        sm.check([("TBA", "dollar_roll")])
    g = DependencyGraph()
    g.add_node(Node("FRN1", "product"))
    g.add_node(Node("fixings:SOFR", "data", units="rate/ACT360", available=False, detail="no SOFR history loaded"))
    g.require("FRN1", "fixings:SOFR", "rate/ACT360", "floating coupon needs fixings")
    g.require("FRN1", "model:exercise", "", "callable needs exercise model")
    problems = g.validate()
    assert any("unavailable" in p for p in problems) and any("missing dependency" in p for p in problems)
    with pytest.raises(UnsupportedCombination):
        g.assert_valid()
    reg = ModelRegistry()
    reg.register(ModelRegistration("hw1f-tree", "1.0", "pricing", ("callable",), training_cutoff="2019-12-31"))
    assert reg.get("hw1f-tree@1.0").training_cutoff == "2019-12-31" and len(reg.to_frame()) == 1


def test_assumption_precedence_and_export():
    a = AssumptionSet()
    a.default("cpr", 0.06, "annual", source="system default")
    a.product_default("MBS", "cpr", 0.08, "annual")
    a.instrument_override("POOL1", "cpr", 0.12, "annual", source="dealer model")
    assert a.value("cpr") == 0.06
    assert a.value("cpr", {"product:MBS"}) == 0.08
    assert a.value("cpr", {"product:MBS", "instrument:POOL1"}) == 0.12
    assert a.value("missing", default=1.0) == 1.0
    assert set(a.export_resolved()["level"]) == {"SYSTEM_DEFAULT", "PRODUCT_DEFAULT", "INSTRUMENT_OVERRIDE"}
    assert a.used()["value"].tolist()[-1] == 0.12


def test_manifest_validation_and_hashing():
    m = ExperimentManifest(start="2024-01-02", end="2024-06-28", run_mode=RunMode.HISTORICAL_COUNTERFACTUAL)
    assert any("CounterfactualSpec" in p for p in m.validate())
    m.counterfactual = CounterfactualSpec(changed_drivers=("prepayments",), recompute=("pool_balances", "coupons", "marks"))
    assert any("reinvestment" in p for p in m.validate())
    m.counterfactual = CounterfactualSpec(changed_drivers=("prepayments",), recompute=("pool_balances", "coupons", "marks"), frozen=("reinvestment",))
    assert m.validate() == []
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2]})
    assert hash_frame(df) == hash_frame(df.copy()) and hash_frame(df) != hash_frame(pd.DataFrame({"a": [1, 3]}))
    m.add_data("prices", df, knowledge_time="2024-07-01")
    assert m.data_versions[0].rows == 2 and len(m.config_hash) == 16 and "python" in m.environment


def test_decision_context_enforces_boundary():
    ctx = DecisionContext("PM", DataVintage(None))
    ctx._set_now(T0)
    with pytest.raises(InformationBoundaryError):
        ctx._publish("mark", "BOND", object(), available_ns=T0 + 1)
    ctx._publish("fixing", "SOFR", 0.053, available_ns=T0)
    assert ctx.fixing("SOFR") == 0.053 and ctx.today == date(2024, 1, 2)
    with pytest.raises(InformationBoundaryError):
        _ = ctx.env
    with pytest.raises(InformationBoundaryError):
        ctx._set_now(T0 - 1)
    ctx._bind(restricted={"BAD"})
    assert ctx.can_trade("BAD") == (False, "restricted list") and ctx.can_trade("GOOD")[0]
