"""Ledger accounting on a hand-built environment: maturity, call, default/recovery, bill, MBS paydowns,
coupon entitlement on the fill day."""

from datetime import date

import pandas as pd
import pytest
from nautilus_trader.model import StrategyId
from strategies import BuyAndHold, BuyAndHoldConfig

from nautilus_fi.backtest import BacktestSettings, FixedIncomeBacktest, MarketEnvironmentBuilder
from nautilus_fi.execution import GROSS
from nautilus_fi.instruments import (
    AmortizationTerms,
    AssetType,
    BondSpec,
    CallOption,
    CouponType,
    DayCount,
    Frequency,
    PrincipalType,
)

START, END = date(2024, 1, 2), date(2024, 6, 28)


def bonds():
    return {
        "MAT5": BondSpec("MAT5", "ACME", date(2019, 3, 15), date(2024, 3, 15), 0.05, day_count=DayCount.THIRTY_360),
        "BILL": BondSpec("BILL", "UST", date(2023, 12, 1), date(2024, 4, 5), 0.0, coupon_type=CouponType.ZERO, asset_type=AssetType.BILL, day_count=DayCount.ACT_360, min_piece=100, increment=100),
        "CALLB": BondSpec("CALLB", "HYCO", date(2021, 5, 1), date(2027, 5, 1), 0.06, call_schedule=[CallOption(date(2024, 5, 1), 102.0)]),
        "DFLT": BondSpec("DFLT", "ZOMBIE", date(2021, 8, 15), date(2028, 8, 15), 0.08, rating_initial="B"),
        "POOL": BondSpec("POOL", "FNMA", date(2020, 1, 1), date(2050, 1, 1), 0.05, frequency=Frequency.MONTHLY, asset_type=AssetType.MBS, principal_type=PrincipalType.AMORTIZING, amortization=AmortizationTerms(original_term_months=360, projected_cpr=0.06), min_piece=1000, increment=1),
        "ENT": BondSpec("ENT", "ENTCO", date(2020, 7, 3), date(2026, 1, 3), 0.04),
    }


@pytest.fixture(scope="module")
def env():
    b = bonds()
    cal = __import__("nautilus_fi.instruments.calendar", fromlist=["us_bond_market_calendar"]).us_bond_market_calendar()
    dates = cal.business_days(START, END)
    prices = []
    for d in dates:
        for s in b:
            mid = {"MAT5": 99.8, "BILL": 99.5, "CALLB": 103.0, "DFLT": 30.0 if d >= date(2024, 2, 20) else 80.0, "POOL": 97.0, "ENT": 98.0}[s]
            prices.append({"date": pd.Timestamp(d), "instrument_id": s, "mid": mid, "bid": mid - 0.1, "ask": mid + 0.1, "score": 9.0, "source": "TEST"})
    curves = [{"date": pd.Timestamp(d), "curve": "GOVT", "tenor": t, "rate": 0.04, "kind": "zero"} for d in dates for t in (0.25, 1, 5, 10, 30)]
    factors = [{"date": pd.Timestamp(m), "instrument_id": "POOL", "factor": f} for m, f in ((date(2023, 12, 1), 0.80), (date(2024, 1, 1), 0.80), (date(2024, 2, 1), 0.78), (date(2024, 3, 1), 0.76), (date(2024, 4, 1), 0.75), (date(2024, 5, 1), 0.74), (date(2024, 6, 1), 0.73))]
    actions = [
        {"instrument_id": "CALLB", "action": "CALL", "effective_date": pd.Timestamp("2024-05-01"), "price": 102.0, "announce_ts": pd.Timestamp("2024-04-01 15:00", tz="UTC"), "note": "called"},
        {"instrument_id": "DFLT", "action": "DEFAULT", "effective_date": pd.Timestamp("2024-02-20"), "price": 30.0, "announce_ts": pd.Timestamp("2024-02-20 08:00", tz="UTC"), "note": "missed coupon"},
        {"instrument_id": "DFLT", "action": "RECOVERY", "effective_date": pd.Timestamp("2024-04-15"), "price": 35.0, "announce_ts": pd.Timestamp("2024-04-10 08:00", tz="UTC"), "note": "workout"},
    ]
    builder = MarketEnvironmentBuilder(b, START, END)
    builder.with_evaluated_prices(pd.DataFrame(prices)).with_curves(pd.DataFrame(curves)).with_factors(pd.DataFrame(factors)).with_corporate_actions(pd.DataFrame(actions))
    return builder.build()


@pytest.fixture(scope="module")
def result(env):
    targets = {"MAT5": 1_000_000, "BILL": 2_000_000, "CALLB": 1_000_000, "DFLT": 1_000_000, "POOL": 1_000_000, "ENT": 1_000_000}
    strat = BuyAndHold(BuyAndHoldConfig(strategy_id=StrategyId("BH-001"), rebalance="M", targets=targets))
    return FixedIncomeBacktest(env, strat, GROSS, settings=BacktestSettings(starting_cash=20e6)).run()


def _cf(result, symbol, kind):
    cf = result.cashflows
    return cf[(cf["symbol"] == symbol) & (cf["type"] == kind)]


def _redemption(result, symbol):
    f = result.fills
    return f[(f["symbol"] == symbol) & (f["protocol"] == "EXPIRATION")]


def test_fills_happen_next_window(result):
    buys = result.fills[result.fills["protocol"] != "EXPIRATION"]
    assert set(buys["symbol"]) == {"MAT5", "BILL", "CALLB", "DFLT", "POOL", "ENT"}
    assert set(pd.to_datetime(buys["ts"]).dt.date) == {date(2024, 1, 3)}
    assert (buys["delay_days"] == 1).all()


def test_maturity_pays_final_coupon_and_par(result):
    coupon = _cf(result, "MAT5", "COUPON")
    assert len(coupon) == 1 and coupon["date"].iloc[0] == date(2024, 3, 15)
    assert coupon["amount"].iloc[0] == pytest.approx(1_000_000 * 0.05 / 2)
    red = _redemption(result, "MAT5")
    assert len(red) == 1 and red["clean_price"].iloc[0] == pytest.approx(100.0) and red["date"].iloc[0] == date(2024, 3, 15)
    assert red["accrued"].iloc[0] == 0.0


def test_bill_matures_at_par_without_coupons(result):
    assert _cf(result, "BILL", "COUPON").empty
    red = _redemption(result, "BILL")
    assert len(red) == 1 and red["clean_price"].iloc[0] == pytest.approx(100.0) and red["face"].iloc[0] == pytest.approx(2_000_000)
    assert red["date"].iloc[0] == date(2024, 4, 5)


def test_call_pays_coupon_and_call_price(result):
    coupon = _cf(result, "CALLB", "COUPON")
    assert len(coupon) == 1 and coupon["date"].iloc[0] == date(2024, 5, 1) and coupon["amount"].iloc[0] == pytest.approx(30_000)
    red = _redemption(result, "CALLB")
    assert len(red) == 1 and red["clean_price"].iloc[0] == pytest.approx(102.0) and red["date"].iloc[0] == date(2024, 5, 1)
    assert _cf(result, "CALLB", "ACCRUED_AT_REDEMPTION")["amount"].abs().sum() < 1e-9  # coupon date == call date


def test_default_stops_accrual_and_recovery_closes(result):
    coupons = _cf(result, "DFLT", "COUPON")
    assert len(coupons) == 1 and coupons["date"].iloc[0] == date(2024, 2, 15)   # paid: before the Feb 20 default
    assert coupons["amount"].iloc[0] == pytest.approx(40_000)
    assert _cf(result, "DFLT", "ACCRUED_AT_REDEMPTION").empty
    red = _redemption(result, "DFLT")
    assert len(red) == 1 and red["clean_price"].iloc[0] == pytest.approx(35.0) and red["date"].iloc[0] == date(2024, 4, 15) and red["accrued"].iloc[0] == 0.0
    assert not _cf(result, "DFLT", "DEFAULT").empty


def test_mbs_paydowns_follow_realized_factors(result):
    prin = _cf(result, "POOL", "PRINCIPAL").sort_values("date")
    assert len(prin) == 5  # Feb..Jun factor drops (Jan factor equal to Dec)
    orig = 1_000_000 / 0.80  # bought 1mm current face at factor 0.80
    assert prin["amount"].iloc[0] == pytest.approx((0.80 - 0.78) * orig, rel=1e-3)
    assert prin["amount"].sum() == pytest.approx((0.80 - 0.73) * orig, rel=1e-3)
    coupons = _cf(result, "POOL", "COUPON")
    assert len(coupons) >= 5 and (coupons["amount"].diff().dropna() <= 1e-6).all()  # coupon on a shrinking balance


def test_no_coupon_to_buyer_on_payment_date(result):
    ent = _cf(result, "ENT", "COUPON")
    assert date(2024, 1, 3) not in set(ent["date"])
    assert ent.empty  # next coupon (Jul 3) is after the window


def test_nav_identity_and_closed_positions(result):
    nav = result.nav
    assert ((nav["cash"] + nav["market_value"] + nav["coupon_receivable"] - nav["nav"]).abs() < 1e-6).all()   # NAV = cash + dirty value + entitled coupons
    closed = result.positions_report[result.positions_report["side"] == "FLAT"]
    assert set(closed["instrument_id"].astype(str)) == {"MAT5.OTC", "BILL.OTC", "CALLB.OTC", "DFLT.OTC"}
