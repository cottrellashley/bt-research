"""Product packages: floaters, linkers, loans, hybrids, distressed, money market, tax, funds,
structured waterfalls, TBA, derivatives and FX (blueprint sections 6-7 critical validation cases)."""

from datetime import date, timedelta

import pytest

from nautilus_fi.instruments import (
    AssetType,
    BondSpec,
    CouponType,
    DayCount,
    FloatingTerms,
    Frequency,
    InflationTerms,
    MarketInputs,
    ZeroCurve,
    project_cashflows,
)
from nautilus_fi.instruments.calendar import us_bond_market_calendar
from nautilus_fi.instruments.cashflows import coupon_rate_for_period, factor_at
from nautilus_fi.instruments.derivatives import (
    CreditDefaultSwap,
    FXForward,
    InterestRateSwap,
    TreasuryFuture,
    hazard_from_spread,
)
from nautilus_fi.instruments.distressed import (
    ExchangeLeg,
    ExchangeTerms,
    claim_at_default,
    claim_value_per_100,
)
from nautilus_fi.instruments.floating import (
    BenchmarkTransition,
    MissingFixing,
    compounded_in_arrears,
    index_ratio_rate,
    term_rate_in_advance,
)
from nautilus_fi.instruments.funds import FundInstrument
from nautilus_fi.instruments.fx import FXRates, HedgeProgram, hedged_return_decomposition
from nautilus_fi.instruments.hybrids import (
    DeferralType,
    HybridTerms,
    IssuerCallPolicy,
    expected_redemption,
    trigger_breached,
)
from nautilus_fi.instruments.linkers import (
    UK_GILT_8M,
    US_TIPS,
    index_ratio,
    redemption_per_100,
    reference_cpi,
)
from nautilus_fi.instruments.loans import LoanTerms, PIKTerms, commitment_fee, delayed_compensation
from nautilus_fi.instruments.money_market import (
    AmortizedCostPolicy,
    bond_equivalent_yield,
    certificate_of_deposit,
    discount_yield_to_price,
    liquidity_fee,
    money_market_yield,
    price_to_discount_yield,
)
from nautilus_fi.instruments.tax import TaxProfile
from nautilus_fi.instruments.terms_history import TermsHistory
from nautilus_fi.structured import (
    CollateralPool,
    Deal,
    DealState,
    InternalWaterfallEngine,
    PoolAllocation,
    PoolAssumptions,
    RemittanceReport,
    TBAContract,
    TBASettlementCalendar,
    Tranche,
    implied_roll_financing_rate,
    run_waterfall_period,
)
from nautilus_fi.structured.tba import deliver, good_delivery

CAL = us_bond_market_calendar()


# ------------------------------------------------------------------------------------- floaters
def daily_sofr(start: date, end: date, rate: float = 0.05) -> dict[date, float]:
    out, d = {}, start
    while d <= end:
        if CAL.is_business_day(d):
            out[d] = rate
        d += timedelta(days=1)
    return out


def test_compounded_in_arrears_weighting_lookback_and_knowledge():
    fix = daily_sofr(date(2024, 1, 2), date(2024, 4, 30), 0.05)
    det = compounded_in_arrears(fix, date(2024, 2, 1), date(2024, 5, 1), CAL)
    # flat 5% compounded daily with weekend weighting annualises slightly above 5%
    assert 0.0500 < det.rate < 0.0504 and det.status == "KNOWN"
    lb = compounded_in_arrears(fix, date(2024, 2, 1), date(2024, 5, 1), CAL, lookback_days=5)
    assert abs(lb.rate - det.rate) < 1e-6
    partial = compounded_in_arrears(fix, date(2024, 2, 1), date(2024, 5, 1), CAL, known_as_of=date(2024, 3, 15))
    assert partial.status == "PARTIAL" and partial.missing > 0
    with pytest.raises(MissingFixing):
        compounded_in_arrears({}, date(2024, 2, 1), date(2024, 5, 1), CAL)
    assert index_ratio_rate({date(2024, 2, 1): 1.10, date(2024, 5, 1): 1.1125}, date(2024, 2, 1), date(2024, 5, 1)).rate == pytest.approx((1.1125 / 1.10 - 1) * 360 / 90)
    adv = term_rate_in_advance({date(2024, 1, 26): 0.052, date(2024, 1, 30): 0.053}, date(2024, 2, 1), CAL, 2, known_as_of=date(2024, 1, 29))
    assert adv.status == "PROJECTED" and adv.rate == 0.052                      # the 30 Jan fixing is not yet knowable
    with pytest.raises(MissingFixing):
        term_rate_in_advance({date(2024, 1, 30): 0.053}, date(2024, 2, 1), CAL, 2, known_as_of=date(2024, 1, 29))


def test_floater_coupon_crossing_holidays_and_benchmark_transition():
    frn = BondSpec("FRN", "BANK", date(2023, 6, 20), date(2026, 6, 20), 0.0, coupon_type=CouponType.FLOATING, frequency=Frequency.QUARTERLY, day_count=DayCount.ACT_360,
                   floating=FloatingTerms(index="LIBOR3M", spread_bp=100, method="IN_ADVANCE", transition=BenchmarkTransition("LIBOR3M", "SOFR", date(2023, 7, 1), spread_adjustment_bp=26.161, lookback_days=2)))
    period = next(p for p in frn.schedule if p.accrual_start == date(2024, 6, 20))   # crosses 4 July and Juneteenth-free window
    fix = daily_sofr(date(2024, 5, 1), date(2024, 10, 1), 0.053)
    mi = MarketInputs(daily_fixings={"SOFR": fix}, known_as_of=date(2024, 12, 31))
    rate = coupon_rate_for_period(frn, period, mi)
    assert rate == pytest.approx(mi.last_determination.rate + (100 + 26.161) / 1e4, abs=1e-9)
    assert mi.last_determination.status == "KNOWN" and mi.last_determination.method.value == "COMPOUNDED_IN_ARREARS"
    pre = next(p for p in frn.schedule if p.accrual_start == date(2023, 6, 20))      # before the transition: LIBOR in advance
    mi2 = MarketInputs(fixings={"LIBOR3M": {date(2023, 6, 16): 0.055}}, daily_fixings={"SOFR": fix})
    assert coupon_rate_for_period(frn, pre, mi2) == pytest.approx(0.055 + 0.01)


# ------------------------------------------------------------------------------------- linkers
def test_tips_reference_cpi_interpolation_known_vs_projected_and_floor():
    cpi = {(2024, 1): 308.417, (2024, 2): 310.326, (2024, 3): 312.332, (2024, 4): 313.548}
    ref, status = reference_cpi(cpi, date(2024, 5, 16), US_TIPS)          # lag 3: Feb -> Mar, day 16 of 31
    assert ref == pytest.approx(310.326 + 15 / 31 * (312.332 - 310.326)) and status == "KNOWN"
    _, status2 = reference_cpi(cpi, date(2024, 7, 16), US_TIPS, known_as_of=date(2024, 5, 20))   # needs April (published mid-May) and May
    assert status2 == "PROJECTED"
    ratio, _ = index_ratio(cpi, date(2024, 5, 16), 280.0, US_TIPS)
    assert ratio == pytest.approx(round((310.326 + 15 / 31 * (312.332 - 310.326)) / 280.0, 5))
    assert redemption_per_100(0.97, US_TIPS) == 100.0 and redemption_per_100(0.97, UK_GILT_8M) == 97.0
    ref8, _ = reference_cpi(cpi, date(2024, 10, 10), UK_GILT_8M)              # 8-month lag, no interpolation: Feb
    assert ref8 == 310.326
    tips = BondSpec("TIPS", "UST", date(2024, 1, 15), date(2034, 1, 15), 0.0125, coupon_type=CouponType.INFLATION_LINKED, day_count=DayCount.ACT_ACT_ICMA, asset_type=AssetType.LINKER, inflation=InflationTerms(convention="US_TIPS"))
    mi = MarketInputs(cpi=cpi, cpi_base=300.0, known_as_of=date(2024, 6, 1))
    assert factor_at(tips, date(2024, 5, 16), mi) == pytest.approx(ratio * 280.0 / 300.0, rel=1e-4)


# -------------------------------------------------------------------------------- loans / hybrids
def test_loan_fees_pik_and_delayed_compensation_and_terms_history():
    lt = LoanTerms(commitment=10e6, drawn=6e6, spread_bp=350, floor=0.005, undrawn_fee_bp=50, pik=PIKTerms(pik_bp=200, cash_bp=150))
    assert lt.undrawn == 4e6 and commitment_fee(lt, date(2024, 1, 1), date(2024, 4, 1)) == pytest.approx(4e6 * 0.005 * 91 / 360)
    assert lt.all_in_rate(0.002) == pytest.approx(0.005 + 0.035)                  # floor binds on the index
    cash, pik = lt.cash_and_pik(0.05, elect_pik=True)
    assert cash + pik == pytest.approx(0.085) and pik == pytest.approx(0.02)
    dc = delayed_compensation(date(2024, 3, 1), date(2024, 3, 25), 5e6, 0.085, 98.0, 0.053, CAL, lt)
    assert dc["days"] > 0 and dc["interest_to_buyer"] > dc["cost_of_carry"] > 0
    assert delayed_compensation(date(2024, 3, 1), date(2024, 3, 8), 5e6, 0.085, 98.0, 0.053, CAL, lt)["net_to_buyer"] == 0.0
    base = BondSpec("LN", "CO", date(2022, 1, 1), date(2028, 1, 1), 0.08)
    th = TermsHistory(base)
    th.amend(effective_date=date(2024, 6, 1), announce_date=date(2024, 5, 15), coupon_rate=0.075, maturity=date(2029, 1, 1), note="repricing + extension")
    assert th.as_of(date(2024, 5, 31)).coupon_rate == 0.08 and th.as_of(date(2024, 6, 1)).coupon_rate == 0.075
    assert th.known_as_of(date(2024, 5, 10), date(2024, 6, 1)).coupon_rate == 0.08       # not yet announced
    assert th.known_as_of(date(2024, 5, 20), date(2024, 6, 1)).maturity == date(2029, 1, 1)
    assert len(th.pending(date(2024, 5, 20))) == 1


def test_hybrid_no_assumed_redemption_and_triggers():
    ht = HybridTerms(first_call=date(2026, 6, 15), reset_spread_bp=300, deferral=DeferralType.NON_CUMULATIVE, trigger_level=0.05125)
    assert expected_redemption(ht, date(2024, 1, 2), benchmark_rate=0.04, issuer_new_issue_rate=0.075) is None     # reset 7% < new issue 7.5%: no call
    assert expected_redemption(ht, date(2024, 1, 2), benchmark_rate=0.04, issuer_new_issue_rate=0.06) == date(2026, 6, 15)
    assert expected_redemption(ht, date(2024, 1, 2), 0.04, 0.075, IssuerCallPolicy(regulatory_value_lost=True)) == date(2026, 6, 15)
    assert trigger_breached(ht, 0.05) and not trigger_breached(ht, 0.06)


def test_distressed_claim_and_exchange_units():
    spec = BondSpec("HY", "ZOMBIE", date(2021, 8, 15), date(2028, 8, 15), 0.08)
    claim = claim_at_default(spec, date(2024, 2, 20))
    assert claim.principal_per_100 == 100.0 and claim.prepetition_interest_per_100 == pytest.approx(8 * 5 / 360, abs=1e-6)
    claim.recovery_estimate_pct, claim.expected_distribution = 0.4, date(2025, 2, 20)
    v = claim_value_per_100(claim, date(2024, 2, 20), 0.15)
    assert 30 < v < 40
    ex = ExchangeTerms("HY", [ExchangeLeg("NEWDEBT", "DEBT", 0.5, 80.0), ExchangeLeg("EQ", "EQUITY", 10.0, 1.5), ExchangeLeg("CASH", "CASH", 5.0)], consent_fee_per_100=1.0)
    assert ex.value_per_100({"NEWDEBT": 80.0, "EQ": 1.5}) == pytest.approx(0.5 * 80 + 15 + 5 + 1)
    assert ex.new_positions(1_000_000) == [("NEWDEBT", 5_000.0, "DEBT"), ("EQ", 100_000.0, "EQUITY")] and ex.cash(1_000_000) == pytest.approx(60_000.0)


# --------------------------------------------------------------------------- money market / tax / funds
def test_money_market_yields_amortized_cost_and_liquidity_fee():
    px = discount_yield_to_price(0.0525, 182)
    assert price_to_discount_yield(px, 182) == pytest.approx(0.0525)
    assert money_market_yield(px, 182) > 0.0525 and bond_equivalent_yield(px, 182) > money_market_yield(px, 182)
    cd = certificate_of_deposit("CD1", "BANK", date(2024, 1, 10), date(2024, 7, 10), 0.054)
    flows = project_cashflows(cd, date(2024, 1, 10))
    assert len(flows) == 2 and flows[0].amount == pytest.approx(5.4 * 182 / 360, rel=1e-6)
    pol = AmortizedCostPolicy()
    assert pol.amortized_price(99.0, date(2024, 1, 1), date(2024, 7, 1), date(2024, 4, 1)) == pytest.approx(99.5, abs=0.01)
    assert pol.breaches(0.9940, 1.0) and not pol.breaches(0.9970, 1.0)
    assert liquidity_fee(0.06, 15.0) == pytest.approx(0.0015) and liquidity_fee(0.04, 15.0) == 0.0 and liquidity_fee(0.06, 0.5) == 0.0
    tp = TaxProfile("US", "US_INDIVIDUAL", ordinary_rate=0.37, state_rate=0.05, withholding={"BR": 0.15}, treaty_rates={"DE": 0.0})
    assert tp.after_tax_coupon(100.0, AssetType.MUNICIPAL) == pytest.approx(95.0)
    assert tp.after_tax_coupon(100.0, AssetType.CORPORATE) == pytest.approx(58.0)
    assert tp.after_tax_coupon(100.0, AssetType.SOVEREIGN, "BR") == pytest.approx(58.0 * 0.85)
    assert tp.taxable_equivalent_yield(0.03) == pytest.approx(0.03 / (1 - 0.42))
    f = FundInstrument("JAAA", "ETF", nav={date(2024, 1, 2): 50.0, date(2024, 1, 31): 50.4}, market_price={date(2024, 1, 2): 50.1, date(2024, 1, 31): 50.3}, distributions={date(2024, 1, 15): 0.2})
    assert f.premium_discount(date(2024, 1, 2)) == pytest.approx(0.002)
    assert f.total_return(date(2024, 1, 2), date(2024, 1, 31), "NAV") == pytest.approx((50.4 + 0.2) / 50.0 - 1)
    assert f.total_return(date(2024, 1, 2), date(2024, 1, 31), "MARKET") == pytest.approx((50.3 + 0.2) / 50.1 - 1)


# ---------------------------------------------------------------------------------------- structured
def make_deal() -> Deal:
    return Deal("CLO-1", [
        Tranche("A", 300e6, index="SOFR", spread_bp=150, seniority=1, oc_trigger=1.20, ic_trigger=1.20),
        Tranche("B", 60e6, index="SOFR", spread_bp=300, seniority=2, oc_trigger=1.10, ic_trigger=1.10, deferrable=True),
        Tranche("E", 40e6, seniority=3, kind="EQUITY"),
    ], senior_fee_bp=20, subordinate_fee_bp=30, reinvestment_end=date(2025, 12, 31), payment_frequency=4)


def test_waterfall_conserves_cash_and_pays_by_seniority():
    deal, state = make_deal(), DealState.initial(make_deal())
    res = run_waterfall_period(deal, state, date(2024, 4, 15), collateral_balance_end=400e6, interest_collected=9e6, principal_collected=10e6, losses=0.0, fixings={"SOFR": 0.05})
    assert abs(res.conservation_error) < 1e-6
    assert res.interest_paid["A"] == pytest.approx(300e6 * 0.065 / 4) and res.interest_paid["B"] == pytest.approx(60e6 * 0.08 / 4)
    assert res.reinvested == pytest.approx(10e6) and res.equity_cash > 0 and all(v["oc_pass"] and v["ic_pass"] for v in res.tests.values())


def test_trigger_breach_redirects_interest_to_senior_principal():
    deal, state = make_deal(), DealState.initial(make_deal())
    res = run_waterfall_period(deal, state, date(2024, 4, 15), collateral_balance_end=330e6, interest_collected=9e6, principal_collected=0.0, losses=70e6, fixings={"SOFR": 0.05})
    assert not res.tests["B"]["oc_pass"] and res.diverted_interest > 0
    assert res.principal_paid["A"] == pytest.approx(res.diverted_interest) and state.balances["A"] < 300e6
    assert res.equity_cash == 0.0 and abs(res.conservation_error) < 1e-6
    assert state.balances["E"] == 0.0 and state.balances["B"] < 60e6          # losses hit equity then mezz


def test_engine_uses_realized_remittance_then_projection_and_tranche_factors():
    deal, state = make_deal(), DealState.initial(make_deal())
    pool = CollateralPool(400e6, 0.09, 84, amortizing=False)
    rem = RemittanceReport(date(2024, 4, 15), date(2024, 4, 25), 400e6, 9e6, 0.0, 8e6, 1e6, 0.4e6, 0.0, 392e6, 0.09)
    results, flows, snap = InternalWaterfallEngine().project(deal, state, pool, date(2024, 1, 15), 3, PoolAssumptions(cpr=0.2, cdr=0.03, severity=0.4), {"SOFR": 0.05}, remittances=[rem])
    assert [r.available_interest for r in results][0] == pytest.approx(9e6)
    sources = {f.source for f in flows if f.period_end == date(2024, 4, 15)}
    assert sources == {"REALIZED"} and {f.source for f in flows if f.period_end > date(2024, 4, 15)} == {"PROJECTED"}
    assert all(abs(r.conservation_error) < 1e-6 for r in results) and snap.engine.endswith("internal_waterfall")
    a_flows = [f for f in flows if f.tranche == "A"]
    assert all(0 < f.factor_end <= 1.0 for f in a_flows) and a_flows[-1].factor_end <= a_flows[0].factor_end


def test_tba_delivery_and_dollar_roll():
    cal = TBASettlementCalendar(dates={(2024, 6, "A"): date(2024, 6, 13)}, calendar=CAL)
    assert cal.settlement_date(2024, 6, "A") == date(2024, 6, 13) and cal.notification_date(2024, 6, "A") == date(2024, 6, 11)
    approx = cal.settlement_date(2024, 7, "A")
    assert cal.approximate and CAL.is_business_day(approx)
    tba = TBAContract("FNCL 5.5 JUN", "FN", 30, 0.055, 2024, 6, face=1_000_000)
    assert tba.cls == "A"
    pools = [PoolAllocation("FN1", 600_000, 0.95, 0.055), PoolAllocation("FN2", 450_000, 0.9555, 0.055)]
    ok, _ = good_delivery(tba, pools)
    assert ok
    legs = deliver(tba, pools, 99.5)
    assert sum(l["current_face"] for l in legs) == pytest.approx(1_000_000, rel=1e-4) and legs[0]["cost"] == pytest.approx(570_000 * 0.995)
    with pytest.raises(ValueError):
        deliver(tba, [PoolAllocation("FN3", 1_000_000, 0.9, 0.055)], 99.5)
    r = implied_roll_financing_rate(99.5, 99.1, 0.055, 30, expected_paydown_fraction=0.01)
    assert 0.0 < r < 0.10


# --------------------------------------------------------------------------------------- derivatives
def test_futures_swap_cds_and_fx_forward():
    fut = TreasuryFuture("TYZ4", date(2024, 12, 19), date(2024, 12, 31), "UST10", 0.82)
    assert fut.invoice_price(110.5, 1.2) == pytest.approx(110.5 * 0.82 + 1.2)
    assert fut.variation_margin(110.0, 110.5, 10) == pytest.approx(5_000.0)
    assert fut.contract_dv01(0.08) == pytest.approx(0.08 / 0.82 * 1000)
    ctd, irr = fut.cheapest_to_deliver([("A", 90.0, 0.82, 0.0), ("B", 95.0, 0.86, 0.0)], 110.0, {}, {}, 90)
    assert ctd in ("A", "B")
    curve = ZeroCurve.flat(date(2024, 1, 2), 0.04)
    swap = InterestRateSwap("IRS", 100e6, "USD", date(2024, 1, 2), date(2029, 1, 2), fixed_rate=0.04, pay_fixed=True)
    v = swap.value(date(2024, 1, 2), curve)
    import math

    assert abs(v["par_rate"] - (math.exp(0.04) - 1.0)) < 0.001 and v["dv01"] > 0                # flat continuous 4% -> simple annual par ~4.08%; payer gains when rates rise
    assert v["npv"] == pytest.approx((v["par_rate"] - 0.04) * v["annuity"] * 100e6, rel=0.02)
    at_par = InterestRateSwap("IRS2", 100e6, "USD", date(2024, 1, 2), date(2029, 1, 2), fixed_rate=v["par_rate"], pay_fixed=True)
    assert abs(at_par.value(date(2024, 1, 2), curve)["npv"]) < 1.0
    h = hazard_from_spread(100.0, 0.4, curve, 5.0)
    assert 0.014 < h < 0.019
    cds = CreditDefaultSwap("CDS", "ACME", 10e6, "USD", date(2024, 1, 2), date(2029, 6, 20), running_spread_bp=100, recovery=0.4, buy_protection=True)
    base = cds.value(date(2024, 1, 2), curve, par_spread_bp=100.0)
    assert abs(base["npv"]) < 1_000 and base["cs01"] > 0 and abs(base["par_spread_bp"] - 100.0) < 0.5
    wide = cds.value(date(2024, 1, 2), curve, par_spread_bp=200.0)
    assert wide["npv"] > 300_000
    ce = cds.credit_event(0.30, date(2024, 7, 1))
    assert ce["payout_to_us"] == pytest.approx(7e6) and cds.terminated
    idx = CreditDefaultSwap("CDX", "CDX.IG", 125e6, "USD", date(2024, 3, 20), date(2029, 6, 20), 100, 0.4, True, index_series="IG42", constituents=125)
    ce2 = idx.credit_event(0.25, date(2024, 7, 1))
    assert ce2["affected_notional"] == pytest.approx(1e6) and idx.factor == pytest.approx(1 - 1 / 125)
    fwd = FXForward("F", "EURUSD", -10e6, 1.10, date(2024, 4, 2))
    val = fwd.value(date(2024, 1, 2), 1.08, curve)
    assert val["npv_quote"] > 0                                                           # short EUR gains when the forward falls
    assert fwd.settlement_cashflows(1.09) == {"EUR": -10e6, "USD": 11e6}


def test_fx_rates_and_hedge_program():
    fx = FXRates()
    fx.add_spot("EURUSD", {date(2024, 1, 2): 1.10}); fx.add_spot("GBPUSD", {date(2024, 1, 2): 1.27})
    fx.add_points("EURUSD", 1, {date(2024, 1, 2): 15.0})
    assert fx.rate("USDEUR", date(2024, 1, 3)) == pytest.approx(1 / 1.10) and fx.rate("EURGBP", date(2024, 1, 3)) == pytest.approx(1.10 / 1.27)
    assert fx.convert(100.0, "EUR", "USD", date(2024, 1, 5)) == pytest.approx(110.0)
    assert fx.forward_rate("EURUSD", date(2024, 1, 2), 1) == pytest.approx(1.1015)
    hp = HedgeProgram("USD", hedge_ratio=1.0, tenor_months=1)
    new = hp.required_hedges(date(2024, 1, 2), {"EUR": 50e6, "USD": 100e6}, fx)
    assert len(new) == 1 and new[0].notional_base == pytest.approx(-50e6) and new[0].pair == "EURUSD"
    assert hp.required_hedges(date(2024, 1, 3), {"EUR": 50.5e6}, fx) == []                 # within threshold
    assert len(hp.required_hedges(date(2024, 1, 3), {"EUR": 60e6}, fx)) == 1
    dec = hedged_return_decomposition(0.01, -0.03, 0.002)
    assert dec["unhedged"] == pytest.approx(1.01 * 0.97 - 1) and dec["hedged"] > dec["unhedged"]
