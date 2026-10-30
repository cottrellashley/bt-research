"""Bond math against hand-computed values."""

from datetime import date

import numpy as np
import pytest

from nautilus_fi.instruments import (
    AmortizationTerms,
    AssetType,
    BondSpec,
    CallOption,
    CashflowType,
    CouponType,
    DayCount,
    FloatingTerms,
    Frequency,
    MarketInputs,
    NelsonSiegelSvensson,
    PrincipalType,
    ZeroCurve,
    accrued_interest,
    analyse,
    bootstrap_par_curve,
    face_to_units,
    generate_schedule,
    project_cashflows,
    units_to_face,
    year_fraction,
)
from nautilus_fi.instruments.analytics import price_from_spread, price_from_yield
from nautilus_fi.instruments.calendar import us_bond_market_calendar
from nautilus_fi.instruments.cashflows import factor_at


def corp() -> BondSpec:
    return BondSpec(symbol="CORP1", issuer="ACME", issue_date=date(2020, 3, 15), maturity=date(2030, 3, 15), coupon_rate=0.05, day_count=DayCount.THIRTY_360)


def test_day_counts():
    assert year_fraction(date(2024, 3, 15), date(2024, 5, 1), DayCount.THIRTY_360) == pytest.approx(46 / 360)
    assert year_fraction(date(2024, 1, 31), date(2024, 2, 29), DayCount.THIRTY_E_360) == pytest.approx(29 / 360)
    assert year_fraction(date(2024, 1, 1), date(2024, 7, 1), DayCount.ACT_360) == pytest.approx(182 / 360)
    assert year_fraction(date(2024, 2, 15), date(2024, 5, 1), DayCount.ACT_ACT_ICMA, date(2024, 2, 15), date(2024, 8, 15), 2) == pytest.approx(76 / (2 * 182))
    assert year_fraction(date(2023, 7, 1), date(2024, 7, 1), DayCount.ACT_ACT_ISDA) == pytest.approx(184 / 365 + 182 / 366)


def test_schedule_backward_with_stub():
    periods = generate_schedule(date(2020, 1, 20), date(2025, 3, 15), Frequency.SEMI_ANNUAL)
    assert periods[-1].accrual_end == date(2025, 3, 15)
    assert periods[0].accrual_start == date(2020, 1, 20)
    assert periods[0].is_stub and periods[0].ref_start == date(2019, 9, 15)
    assert all(not p.is_stub for p in periods[1:])
    assert len(periods) == 11


def test_accrued_interest_conventions():
    assert accrued_interest(corp(), date(2024, 5, 1)) == pytest.approx(5 * 46 / 360, abs=1e-9)
    ust = BondSpec(symbol="UST", issuer="UST", issue_date=date(2024, 2, 15), maturity=date(2034, 2, 15), coupon_rate=0.0425, day_count=DayCount.ACT_ACT_ICMA, asset_type=AssetType.SOVEREIGN)
    assert accrued_interest(ust, date(2024, 5, 1)) == pytest.approx(2.125 * 76 / 182, abs=1e-9)


def test_par_identity_and_yield_roundtrip():
    b = corp()
    assert price_from_yield(b, date(2024, 3, 15), 0.05) == pytest.approx(100.0, abs=1e-8)
    px = price_from_yield(b, date(2024, 5, 1), 0.0475)
    a = analyse(b, date(2024, 5, 1), px)
    assert a.ytm == pytest.approx(0.0475, abs=1e-8)
    assert 4.0 < a.mod_duration < 5.5 and a.convexity > 0 and a.dv01 > 0
    assert a.mac_duration == pytest.approx(a.mod_duration * (1 + a.ytm / 2))


def test_cashflows_exclude_completed_period_on_coupon_date():
    b = BondSpec(symbol="C", issuer="X", issue_date=date(2022, 6, 1), maturity=date(2032, 6, 1), coupon_rate=0.06)
    flows = project_cashflows(b, date(2024, 6, 1))
    assert flows[0].date > date(2024, 6, 3)  # the coupon paid 2024-06-03 belongs to the previous holder
    assert sum(1 for f in flows if f.type == CashflowType.REDEMPTION) == 1


def test_yield_to_worst_picks_the_call():
    b = BondSpec(symbol="CALL1", issuer="HYCO", issue_date=date(2022, 6, 1), maturity=date(2032, 6, 1), coupon_rate=0.06, call_schedule=[CallOption(date(2027, 6, 1), 102.0), CallOption(date(2029, 6, 1), 101.0)])
    a = analyse(b, date(2024, 6, 1), 104.0)
    assert a.ytw < a.ytm and a.ytw_date == date(2027, 6, 1)


def test_bootstrap_reprices_par_bonds():
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    pars = [0.052, 0.051, 0.049, 0.045, 0.043, 0.042, 0.0425, 0.043, 0.046, 0.045]
    zc = bootstrap_par_curve(date(2024, 5, 1), tenors, pars)
    for t, c in zip(tenors, pars):
        assert zc.par_yield(t) == pytest.approx(c, abs=1e-8)
    assert 0 < zc.df(10.0) < 1


def test_nss_fit_and_zero_curve_bumps():
    tenors = np.array([0.5, 1, 2, 5, 10, 30])
    nss = NelsonSiegelSvensson(0.045, -0.01, -0.008, 0.0, 1.8, 12.0)
    fit = NelsonSiegelSvensson.fit(tenors, nss.yield_(tenors))
    assert np.max(np.abs(fit.yield_(tenors) - nss.yield_(tenors))) < 1e-5
    zc = ZeroCurve.flat(date(2024, 1, 1), 0.04)
    bumped = zc.bumped(5.0, 1.0)
    assert bumped.zero(5.0) == pytest.approx(0.0401) and bumped.zero(30.0) == pytest.approx(0.04)


def test_spreads_and_key_rate_durations_sum_to_spread_duration():
    b = corp()
    curve = ZeroCurve.flat(date(2024, 5, 1), 0.045)
    a = analyse(b, date(2024, 5, 1), 98.5, curve=curve, krd_tenors=[0.5, 2, 3, 5, 7, 10, 20, 30])
    assert a.z_spread_bp > 0 and a.g_spread_bp > 0
    assert sum(a.key_rate_durations.values()) == pytest.approx(a.spread_duration, rel=0.02)
    # matrix pricing primitive inverts the z-spread
    assert price_from_spread(b, date(2024, 5, 1), curve, a.z_spread_bp) == pytest.approx(98.5, abs=1e-6)


def test_floater_cashflows_and_discount_margin():
    f = BondSpec(symbol="FRN", issuer="BANK", issue_date=date(2023, 6, 20), maturity=date(2028, 6, 20), coupon_type=CouponType.FLOATING, frequency=Frequency.QUARTERLY, day_count=DayCount.ACT_360, floating=FloatingTerms(index="SOFR", spread_bp=150))
    mi = MarketInputs(fixings={"SOFR": {date(2024, 3, 18): 0.0531, date(2024, 6, 18): 0.0533}})
    flows = project_cashflows(f, date(2024, 7, 1), mi)
    assert flows[0].rate == pytest.approx(0.0531 + 0.015)  # fixing 2 business days before the period start (Juneteenth holiday)
    dm_par = analyse(f, date(2024, 7, 1), 100.0, inputs=mi).discount_margin_bp
    dm_prem = analyse(f, date(2024, 7, 1), 100.5, inputs=mi).discount_margin_bp
    assert 140 < dm_par < 160 and dm_prem < dm_par


def test_amortizing_factor_projection_and_cashflows():
    m = BondSpec(symbol="POOL", issuer="FNMA", issue_date=date(2019, 4, 1), maturity=date(2049, 4, 1), coupon_rate=0.055, frequency=Frequency.MONTHLY, asset_type=AssetType.MBS, principal_type=PrincipalType.AMORTIZING, amortization=AmortizationTerms(original_term_months=360, projected_cpr=0.08))
    mi = MarketInputs(realized_factors={date(2024, 4, 1): 0.82})
    assert factor_at(m, date(2024, 4, 15), mi) == pytest.approx(0.82)
    assert factor_at(m, date(2024, 5, 1), mi) < 0.82
    flows = project_cashflows(m, date(2024, 4, 15), mi)
    principal = sum(c.amount for c in flows if c.type != CashflowType.COUPON)
    assert principal == pytest.approx(82.0, abs=1e-6)
    a = analyse(m, date(2024, 4, 15), 99.0, inputs=mi)
    assert a.factor == pytest.approx(0.82) and 5 < a.wal < 12


def test_zero_coupon_and_linker():
    z = BondSpec(symbol="BILL", issuer="UST", issue_date=date(2024, 1, 4), maturity=date(2024, 7, 5), coupon_type=CouponType.ZERO, asset_type=AssetType.BILL, day_count=DayCount.ACT_360)
    a = analyse(z, date(2024, 1, 5), 97.4)
    assert 0.05 < a.ytm < 0.056 and a.accrued == 0.0
    l = BondSpec(symbol="TIPS", issuer="UST", issue_date=date(2022, 7, 15), maturity=date(2032, 7, 15), coupon_rate=0.00625, coupon_type=CouponType.INFLATION_LINKED, day_count=DayCount.ACT_ACT_ICMA, asset_type=AssetType.LINKER)
    a = analyse(l, date(2024, 7, 15), 98.0, inputs=MarketInputs(index_ratios={date(2024, 7, 15): 1.12}))
    assert a.factor == pytest.approx(1.12) and 0.0 < a.ytm < 0.05


def test_nautilus_mapping_and_units():
    b = corp()
    inst = b.to_instrument("OTC")
    assert str(inst.id) == "CORP1.OTC" and int(inst.lot_size) == 10 and int(inst.min_quantity) == 20
    assert face_to_units(1_000_000) == 10_000 and units_to_face(10_000) == 1_000_000
    assert b.round_face(1_234_567) == 1_235_000 and b.round_face(500) == 0.0
    cal = us_bond_market_calendar()
    assert not cal.is_business_day(date(2024, 7, 4)) and cal.add_business_days(date(2024, 7, 3), 1) == date(2024, 7, 5)
    assert b.settlement_lag() == 1 and b.settle_date(date(2024, 7, 3)) == date(2024, 7, 5)
