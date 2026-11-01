"""Conventions, rebalance calendars, panels and the bitemporal field store."""

import itertools
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nautilus_fi.data.field_store import FieldStore
from nautilus_fi.data.panel import Panel
from nautilus_fi.domain import (
    PROFILES,
    Annualisation,
    Conventions,
    ExcessReturnMethod,
    Explicit,
    IndexCalendar,
    MissingReturnPolicy,
    MissingReturnsError,
    MonthEnd,
    MonthStart,
    Timing,
    TurnoverDefinition,
    Weekly,
    apply_missing_policy,
    compound_excess,
    excess_return_series,
    parse_calendar,
    performance_stats,
    turnover,
)
from nautilus_fi.domain.conventions import detect_bad_dates, periods_per_year, years_spanned
from nautilus_fi.instruments.calendar import us_bond_market_calendar

DAYS = us_bond_market_calendar().business_days(date(2024, 1, 2), date(2024, 12, 31))


def test_timing_and_profiles():
    assert Timing(2).trade_offset_days == 1 and Timing(1).trade_offset_days == 0
    with pytest.raises(ValueError):
        Timing(0)
    assert Conventions.profile("MRI_v1").timing.return_lag_days == 1 and Conventions.profile("MRI_v1").excess_return == ExcessReturnMethod.DAILY_SUM
    assert Conventions.profile("erin_beta_plus").benchmark_drift.value == "STATIC" and not Conventions.profile("paperport").confirmed
    assert {"canonical", "official", "fi_lib_research", "EUR_IG", "MRI_v1", "erin_beta_plus", "fi_lib_backtesting", "paperport"} <= set(PROFILES)
    c = Conventions().with_(timing=1, turnover=TurnoverDefinition.TWO_WAY)
    assert c.timing.return_lag_days == 1 and c.to_dict()["turnover"] == "TWO_WAY" and "lag 1d" in c.describe()
    with pytest.raises(KeyError):
        Conventions.profile("nope")


def test_turnover_definitions():
    before, after = {"A": 0.5, "B": 0.5}, {"A": 0.2, "B": 0.5, "C": 0.3}
    assert turnover(before, after, TurnoverDefinition.ONE_WAY_BUYS) == pytest.approx(0.3)
    assert turnover(before, after, TurnoverDefinition.HALF_SUM) == pytest.approx(0.3)
    assert turnover(before, after, TurnoverDefinition.TWO_WAY) == pytest.approx(0.6)
    assert turnover({}, {"A": 1.0}, TurnoverDefinition.ONE_WAY_BUYS) == pytest.approx(1.0)


def test_annualisation_bases_differ_and_are_stated():
    idx = pd.DatetimeIndex(pd.to_datetime(DAYS[1:]))
    r = pd.Series(0.0002, index=idx)
    assert periods_per_year(idx, Annualisation.TRADING_252) == 252
    assert years_spanned(idx, Annualisation.CALENDAR_365) == pytest.approx(365 / 365, abs=0.02)
    a = performance_stats(r, Conventions(annualisation=Annualisation.TRADING_252))
    b = performance_stats(r, Conventions(annualisation=Annualisation.CALENDAR_365))
    assert a["annualisation"] == "TRADING_252" and b["annualisation"] == "CALENDAR_365"
    assert a["annual_return"] != b["annual_return"]
    with pytest.raises(ValueError):
        performance_stats(r, Conventions(), benchmark=r.iloc[:-5])          # a short benchmark is not silently zero-filled


def test_excess_return_methods():
    tr = np.array([-0.05, -0.08, 0.06])
    tsy = np.array([0.02, 0.03, -0.01])
    daily_sum = compound_excess(tr, tsy, ExcessReturnMethod.DAILY_SUM)
    comp = compound_excess(tr, tsy, ExcessReturnMethod.COMPOUNDED)
    geo = compound_excess(tr, tsy, ExcessReturnMethod.GEOMETRIC)
    assert daily_sum == pytest.approx(float((tr - tsy).sum()))
    assert comp == pytest.approx(float(np.prod(1 + tr) - np.prod(1 + tsy)))
    assert abs(daily_sum - comp) > 0.003 and abs(comp - geo) > 1e-4          # stress-sized returns separate the definitions
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    t = pd.Series(0.001, index=idx)
    s = pd.Series(0.0004, index=idx)
    m = excess_return_series(t, s, Conventions(), "ME")
    assert len(m) == 3 and (m > 0).all()
    v = excess_return_series(t, s, Conventions(), "ME", vendor_excess=pd.Series(0.0005, index=idx))
    assert v.iloc[0] == pytest.approx((1.0005) ** len(idx[idx.month == 1]) - 1)


def test_missing_return_policies():
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"A": [0.01, np.nan, 0.01, 0.0, 0.01, 0.01], "B": [0.01, 0.01, 0.01, 0.0, 0.01, 0.01], "C": [0.0, 0.01, 0.01, 0.0, 0.01, 0.01]}, index=idx)
    with pytest.raises(MissingReturnsError):
        apply_missing_policy(df, MissingReturnPolicy.FAIL)
    z, rep = apply_missing_policy(df, MissingReturnPolicy.ZERO)
    assert z.loc[idx[1], "A"] == 0.0 and rep.filled_zero == 1
    assert idx[3] in detect_bad_dates(df, 0.5)
    refill = df.fillna(0.02).replace(0.0, 0.03)
    r, rep = apply_missing_policy(df, MissingReturnPolicy.REFILL, refill=refill)
    assert r.loc[idx[1], "A"] == 0.02 and r.loc[idx[3], "B"] == 0.03 and rep.refilled >= 4
    d, rep = apply_missing_policy(df, MissingReturnPolicy.DROP_DATE)
    assert idx[1] not in d.index and idx[3] not in d.index


def test_rebalance_calendars():
    bme = MonthEnd().dates(DAYS)
    bms = MonthStart().dates(DAYS)
    assert len(bme) == 12 and all(d.month != (DAYS[DAYS.index(d) + 1].month if DAYS.index(d) + 1 < len(DAYS) else 0) for d in bme)
    assert bms[0] == DAYS[0] and len(bms) == 12
    wed = Weekly(2).dates(DAYS)
    assert all(d.weekday() <= 2 for d in wed) and 50 <= len(wed) <= 53
    assert date(2024, 11, 27) in wed                                         # Wednesday before Thanksgiving
    assert parse_calendar("W-WED") == Weekly(2, name="W-WED") and parse_calendar("BME") == MonthEnd()
    five = parse_calendar(5).dates(DAYS, first_decision=True)
    assert all(DAYS.index(b) - DAYS.index(a) == 5 for a, b in itertools.pairwise(five))
    hol = Explicit((date(2024, 7, 4), date(2024, 12, 25)))                   # an instruction on a holiday acts on the next trading day
    assert hol.dates(DAYS) == [date(2024, 7, 5), date(2024, 12, 26)]
    ahead = Explicit((date(2024, 7, 4), date(2024, 12, 25)), "on_or_before")  # dates published ahead may move earlier
    assert ahead.dates(DAYS) == [date(2024, 7, 3), date(2024, 12, 24)]
    idx_cal = IndexCalendar("LUACTRUU", [date(2024, 3, 31), date(2024, 6, 30)])
    assert idx_cal.dates(DAYS) == [date(2024, 3, 28), date(2024, 6, 28)] and idx_cal.name == "INDEX:LUACTRUU"
    with pytest.raises(ValueError):
        parse_calendar("fortnightly")


def test_panel_point_in_time_access():
    df = pd.DataFrame({"date": list(pd.bdate_range("2024-01-01", periods=4)) * 2, "instrument_id": ["A"] * 4 + ["B"] * 4, "oas_bp": [100, 101, 102, 103, 200, 201, 202, 203], "signal.x": [1, 2, 3, 4, 5, 6, 7, 8]})
    p = Panel.from_long(df, ["oas_bp", "signal.x"])
    assert p["oas_bp"].shape == (4, 2) and p.frame("oas_bp").loc["2024-01-03", "B"] == 202
    p.lags["signal.x"] = 1
    assert list(p.asof_row("signal.x", 2)) == [2, 6]                         # a one-day lag: the decision on day 2 sees day 1
    assert np.isnan(p.asof_row("signal.x", 0)).all()
    h = p.history("oas_bp", 3, 2)
    assert list(h.index.day) == [3, 4] and h.loc[h.index[-1], "A"] == 103
    sub = p.subset(ids=["B"], start="2024-01-02")
    assert sub["oas_bp"].shape == (3, 1)
    assert p.date_pos(date(2024, 1, 6)) == 3                                  # a weekend maps to the previous date


def test_field_store_is_bitemporal_and_never_overwrites(tmp_path):
    fs = FieldStore(tmp_path)
    first = pd.DataFrame({"instrument_id": ["A", "B"], "date": ["2023-12-29", "2024-01-02"], "oas": [100.0, 120.0]})
    fs.write("oas", first, knowledge_time="2024-01-03T08:00:00Z", source="vendor")
    revised = pd.DataFrame({"instrument_id": ["A"], "date": ["2023-12-29"], "oas": [105.0]})
    fs.write("oas", revised, knowledge_time="2024-01-10T08:00:00Z", source="vendor")
    with pytest.raises(FileExistsError):
        fs.write("oas", revised, knowledge_time="2024-01-10T08:00:00Z")
    latest = fs.read("oas")
    asof = fs.read("oas", as_of="2024-01-05")
    firstk = fs.read("oas", mode="first")
    a_latest = latest.loc[latest["entity"] == "A", "value"].iloc[0]
    assert a_latest == 105.0 and asof.loc[asof["entity"] == "A", "value"].iloc[0] == 100.0 and firstk.loc[firstk["entity"] == "A", "value"].iloc[0] == 100.0
    assert {p.parent.name for _, p in fs.snapshots("oas")} == {"year=2023", "year=2024"}     # partitioned by year
    assert fs.read("oas", start="2024-01-01")["entity"].tolist() == ["B"]
    panel = fs.panel(["oas"], as_of="2024-01-05")
    assert panel.frame("oas").loc["2023-12-29", "A"] == 100.0
    assert len(fs.describe()) == 3
