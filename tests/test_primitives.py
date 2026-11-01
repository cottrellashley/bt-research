"""Platform primitives from section 4 of the 2026-10-30 gap analysis (timing, returns, costs,
optimiser, construction, benchmarks, metrics, validation, provenance, proposals)."""

from __future__ import annotations

import itertools
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

import nautilus_fi as nfi
from nautilus_fi.kernel.engine import run


@pytest.fixture(scope="module")
def syn():
    return nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2016-06-30", seed=7)


def _value(**kw):
    base = {"universe": nfi.Index("SYNTH_IG"), "signal": nfi.Field("signal.value"), "portfolio": nfi.Quantile(5), "benchmark": nfi.Index("SYNTH_IG"), "missing": nfi.ZeroFill()}
    base.update(kw)
    return nfi.Strategy("value", **base)


DAYS = [d.date() for d in pd.bdate_range("2024-01-01", "2024-12-31") if d.date() not in {date(2024, 1, 1), date(2024, 7, 4), date(2024, 12, 25)}]


# ====================================================================================== 4.3 timing and schedules
def test_43_calendars_yearly_biweekly_nth_day_and_anchored_weekly():
    assert nfi.YearEnd().dates(DAYS) == [date(2024, 12, 31)] and nfi.YearStart().dates(DAYS) == [date(2024, 1, 2)]
    assert nfi.Strategy(schedule="Y").schedule == nfi.YearEnd()
    biweekly = nfi.Weekly(2, every=2).dates(DAYS)
    assert all(b.toordinal() - a.toordinal() == 14 for a, b in itertools.pairwise(biweekly)) and 25 <= len(biweekly) <= 27
    second = nfi.NthBusinessDay(2).dates(DAYS)
    assert second[0] == date(2024, 1, 3) and len(second) == 12 and nfi.NthBusinessDay(-1).dates(DAYS) == nfi.MonthEnd().dates(DAYS)
    anchored = nfi.AnchoredWeekly().dates(DAYS)
    jan = [d for d in anchored if d.month == 2]
    assert date(2024, 1, 31) in anchored and date(2024, 2, 7) in jan and date(2024, 2, 21) in jan
    assert date(2024, 2, 28) not in anchored or date(2024, 2, 29) in anchored                   # skip within 7 days of the next anchor
    assert all((nxt - d).days >= 7 or d in nfi.MonthEnd().dates(DAYS) for d, nxt in itertools.pairwise(anchored) if nxt in nfi.MonthEnd().dates(DAYS))


def test_43_legacy_names_warn_and_same_bar_trading_must_be_said():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nfi.Strategy(schedule="M")
    assert any("BMS" in str(x.message) for x in w)
    with pytest.raises(ValueError, match="SameClose"):
        nfi.NextClose(0)
    assert nfi.SameClose().lag == 0


def test_43_index_dates_score_drop_and_interim_rebalances(syn):
    idx = nfi.IndexDates("SYNTH_IG").resolve(syn)
    r = run(_value(schedule=nfi.IndexDates("SYNTH_IG")), syn)
    assert len(r.trades) >= len(idx.targets) - 2
    drop = run(_value(portfolio=nfi.TopN(20), schedule=nfi.Triggered(0.15, "score_drop", base=nfi.Weekly(2))), syn)
    acts = drop.construction["action"]
    assert (acts == "trigger_skip").any() and (acts == "rebalance").sum() > 1
    mixed = _value(portfolio=nfi.TopN(20), schedule=nfi.WithInterim(nfi.MonthEnd(), nfi.Triggered(0.1, "score_drop", base=nfi.Weekly(2)), nfi.TopN(10)))
    m = run(mixed, syn)
    fixed = set(pd.to_datetime(nfi.MonthEnd().dates(list(syn.data.panel.dates.date), first_decision=True)))
    interim = m.construction[(m.construction["action"] == "rebalance") & ~m.construction["date"].isin(fixed)]
    assert len(interim) >= 1 and (interim["n_names"] <= 10).all()


def test_43_lag_stress_table_and_gate(syn, tmp_path):
    from nautilus_fi.governance import GateFailure
    from nautilus_fi.research import Hypothesis

    h = Hypothesis("H-LAG", "x", "SYNTH_IG", "BME", "signal.value", "SYNTH_IG", 1e8, "long", "1m", "hs", "sharpe < -10", ("2015-01-01", "2015-12-31"), ("2016-01-01", "2016-06-30"), ("2030-01-01", "2030-12-31"))
    sess = nfi.Session(tmp_path, source=syn, hypothesis=h)
    st = _value(hypothesis="H-LAG")
    table = sess.lag_stress(st)
    assert list(table.index) == [0, 1, 2, 3] and table.loc[1, "ratio_to_own_lag"] == pytest.approx(1.0) and sess.n_trials() == 0
    r = sess.run(st)
    with pytest.raises(GateFailure, match="lag_stress"):
        sess.promote(r, "CANDIDATE", min_dsr=None, max_pbo=None, thresholds={"min_lag_stress": 5.0})
    rec = sess.promote(r, "CANDIDATE", min_dsr=None, max_pbo=None, thresholds={"min_lag_stress": 0.1})
    assert rec.checks["lag_stress"]["ok"]


# ====================================================================================== 4.8 costs, execution and financing
def test_48_cost_models_per_change_buy_only_duration_and_table(syn):
    from nautilus_fi.kernel.data import DataRequest
    from nautilus_fi.kernel.expr import EvalContext

    p = syn.load(DataRequest(("oasd", "price_clean", "rating_score", "half_spread_bp"), None, None))
    ctx = EvalContext(p, syn.attributes(p.ids), None, syn, syn.catalog())._clone(audit=[])
    t, nav = 100, 1e8
    trade = np.zeros(ctx.N)
    trade[:3], trade[3:5] = 0.01, -0.01
    assert nfi.PerChange(250.0).total(ctx, t, trade, nav) == pytest.approx(5 * 250.0 / nav)
    buy = nfi.BuyOnly(nfi.HalfSpread(fallback_bp=10.0), min_trade_bp=15.0)
    small = trade.copy()
    small[0] = 0.001                                                   # 10bp of NAV: below the threshold, free
    hs = np.nan_to_num(p["half_spread_bp"][t], nan=10.0) / 1e4
    assert buy.total(ctx, t, small, nav) == pytest.approx(float((0.01 * hs[1:3]).sum())) and len(ctx.audit) == 1
    dur = nfi.DurationSpreadCost(3.0).unit(ctx, t, trade, nav)
    assert dur[0] == pytest.approx(p["oasd"][t][0] * 3.0 / 1e4 * p["price_clean"][t][0] / 100.0)
    table = pd.DataFrame({"half_spread_bp": [5.0, 8.0, 20.0, 30.0]}, index=pd.MultiIndex.from_tuples([(0, "FIN"), (1, "FIN"), (2, "FIN"), (3, "FIN")]))
    tc = nfi.TableCost(table, ("rating_bucket", "sector"), default_bp=12.0).unit(ctx, t, trade, nav)
    fin = (ctx.attributes["sector"] == "FIN").to_numpy()
    assert set(np.round(tc[~fin] * 1e4, 6)) == {12.0} and set(np.round(tc[fin] * 1e4, 6)) <= {5.0, 8.0, 20.0, 30.0}


def test_48_additive_pnl_borrow_series_tolerance_ticks_and_breakeven(syn):
    add = run(_value(metrics=nfi.Metrics(pnl="ADDITIVE")), syn)
    s = add.stats()
    assert s["pnl"] == "ADDITIVE" and s["annual_return"] == pytest.approx(add.returns.iloc[1:].sum() / s["years"])
    import pandas as _pd

    rate = _pd.Series(0.05, index=syn.data.panel.dates)
    src = nfi.PanelSource(syn.data.panel, syn.data.attributes, {**syn._series, "borrow": rate}, {"SYNTH_IG": syn.data.membership}, syn.data.overlays)
    ls = _value(portfolio=nfi.Quantile(5, short=1), financing=nfi.Cash("cash", short_fee_bp=0))
    cheap, dear = run(ls, src), run(ls.with_(financing=nfi.Financing("cash", 0.0, 0.0, "borrow")), src)
    assert dear.daily["financing_return"].sum() < cheap.daily["financing_return"].sum()
    tol = run(_value(execution=nfi.Execution(1, nfi.HalfSpread(), None, price_tolerance_bp=5.0)), syn)
    assert tol.diagnostics["price_tolerance_skips"] > 0
    hedged = _value(overlay=nfi.Hedge(["TY"], fields=["oad"], targets={"oad": 0.0}))
    plain = run(hedged, syn)
    ticked = run(hedged.with_(overlay=nfi.Hedge(["TY"], fields=["oad"], targets={"oad": 0.0}, cost_per_contract=50.0, contract_notional={"TY": 100_000.0})), syn)
    assert ticked.costs.sum() > plain.costs.sum()
    lazy = run(hedged.with_(overlay=nfi.Hedge(["TY"], fields=["oad"], targets={"oad": 0.0}, tolerance=0.5)), syn)
    assert lazy.costs.sum() < plain.costs.sum()                    # the hedge is not traded while within tolerance
    be = nfi.Session(source=syn).breakeven(_value(portfolio=nfi.TopN(30)), multipliers=(0.0, 5.0, 20.0, 60.0), baseline=_value(portfolio=nfi.BenchmarkWeight()))
    assert be["table"].shape[0] == 4 and (be["breakeven_multiplier"] is None or be["breakeven_multiplier"] > 0)


def test_43_ffill_by_calendar_days():
    from nautilus_fi.data.panel import Panel
    from nautilus_fi.kernel.expr import EvalContext

    d = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-09", "2024-01-15"])
    p = Panel(d, pd.Index(["A"]), {"x": np.array([[1.0], [np.nan], [np.nan], [np.nan], [np.nan]])})
    v = nfi.Field("x").ffill(days=5).evaluate(EvalContext(p))
    assert v[:, 0].tolist()[:3] == [1.0, 1.0, 1.0] and np.isnan(v[3, 0]) and np.isnan(v[4, 0])          # 8 days old by the 9th
    assert nfi.Field("x").ffill(days=7).lookback() == 6


# ====================================================================================== 4.1 / 4.2 / 4.9 returns and data contracts
def test_41_bond_returns_from_prices_coupons_and_write_offs():
    from nautilus_fi.kernel import returns as R

    d = pd.bdate_range("2024-01-01", periods=6)
    clean = pd.DataFrame({"A": [99.0, 99.1, 99.2, 99.1, 99.3, 99.4], "D": [80.0, 70.0, 60.0, 40.0, 40.0, 41.0]}, index=d)
    acc = pd.DataFrame({"A": [2.40, 2.42, 2.44, 0.02, 0.04, 0.06], "D": [1.0, 1.02, 1.04, 0.0, 0.0, 0.0]}, index=d)
    dft = pd.DataFrame(False, index=d, columns=clean.columns)
    dft.loc[d[3], "D"] = True
    tr = R.total_return_from_prices(clean, acc, defaulted=dft)
    coupon = 2.44 + 0.02 - 0.02
    assert tr.loc[d[3], "A"] == pytest.approx((99.1 + 0.02 + coupon) / (99.2 + 2.44) - 1)
    assert tr.loc[d[3], "D"] == pytest.approx((40.0 + 0.0) / (60.0 + 1.04) - 1)                   # defaulted accrued is a loss


def test_41_mtd_decumulation_alignment_resampling_and_forward_returns():
    from nautilus_fi.kernel import returns as R

    d = pd.bdate_range("2024-01-25", "2024-03-05")
    rng = np.random.default_rng(0)
    tr, ir = pd.Series(rng.normal(3e-4, 3e-3, len(d)), index=d), pd.Series(rng.normal(1e-4, 2e-3, len(d)), index=d)
    out = R.decumulate_mtd(R.mtd_from_daily(tr) * 100, R.mtd_from_daily(ir) * 100)
    assert np.abs(out["tr"] - tr).max() < 1e-12 and np.abs(out["tr"] - out["ir"] - out["xr"]).max() < 1e-15
    a, b, rep = R.align_legs(tr, ir.iloc[3:], "outer")
    assert len(a) == len(d) and rep["only_total"] == 3 and b.iloc[0] == 0.0
    with pytest.raises(ValueError, match="disagree"):
        R.align_legs(tr, ir.iloc[3:], "strict")
    m = R.resample_returns(tr, "ME")
    assert m.loc["2024-02-29"] == pytest.approx(float((1 + tr["2024-02"]).prod() - 1)) and len(R.common_dates(tr, ir.iloc[3:])) == len(d) - 3
    frame = pd.DataFrame({"A": tr, "B": tr.where(tr.index < d[20])})
    fwd = R.forward_returns(frame, horizons=(5,), lag=1)
    row = fwd[(fwd["instrument_id"] == "B") & (fwd["date"] == d[16])].iloc[0]
    assert bool(row["exit"]) and fwd[(fwd["instrument_id"] == "A") & (fwd["date"] == d[0])]["fwd_tr"].iloc[0] == pytest.approx(float((1 + tr.iloc[2:7]).prod() - 1))


def test_42_spread_proxy_curves_fx_and_returns_from_an_expression(syn):
    from nautilus_fi.kernel import returns as R

    d = syn.data.panel.dates
    sp = R.spread_return(syn.data.panel.frame("oas_bp"), syn.data.panel.frame("oasd"))
    assert np.allclose(sp["total"], sp["carry"] + sp["spread"], equal_nan=True)
    st = _value(returns=nfi.SpreadReturn(), benchmark=None, execution=nfi.NextClose(cost=nfi.NoCost()))
    r = run(st, syn)
    xr = run(_value(returns="xr", benchmark=None, execution=nfi.NextClose(cost=nfi.NoCost())), syn)
    assert r.gross.iloc[1:].corr(xr.gross.iloc[1:]) > 0.7                 # the proxy tracks the vendor excess return
    y = pd.DataFrame({2.0: 0.04, 10.0: 0.045}, index=d[:30])
    y.iloc[15:, :] += 0.001
    cm = R.constant_maturity_returns(y)
    assert cm[10.0].iloc[15] < -0.005 < cm[10.0].iloc[14] and R.carry_rolldown(y, 10.0)["rolldown_3m"].iloc[0] > 0
    spot = pd.Series(np.linspace(1.10, 1.20, 30), index=d[:30])
    loc = pd.Series(1e-4, index=d[:30])
    assert np.allclose(R.fx_hedged_returns(loc, spot, spot).iloc[1:], 1e-4) and R.unhedged_returns(loc, spot).iloc[1:].mean() > 1e-4


def test_49_cds_futures_dv01_and_pairs():
    from nautilus_fi.kernel import returns as R

    d = pd.bdate_range("2024-03-01", periods=60)
    s = pd.Series(100.0, index=d)
    s.iloc[30:] = 110.0
    s.loc[pd.Timestamp("2024-03-20")] = 105.0                         # roll day jump: not a return
    c = R.cds_returns(s, 4.5, coupon_bp=100.0)
    assert c["mtm"].iloc[30] == pytest.approx(-4.5 * 0.001) and c["mtm"].loc["2024-03-20"] == 0.0 and len(c["premium"]) >= 1
    px = pd.DataFrame({"H": np.linspace(110, 111, 60), "M": np.linspace(109, 110, 60)}, index=d)
    cf = R.continuous_futures(px, {"H": d[40], "M": d[59] + pd.Timedelta(days=30)}, roll_days=5)
    assert cf["contract"].iloc[34] == "H" and cf["contract"].iloc[35] == "M" and np.isfinite(cf["returns"].iloc[35])
    assert abs(cf["level"].iloc[35] - cf["level"].iloc[34]) < 0.1                                  # back-adjusted: no jump
    rng = np.random.default_rng(1)
    dy = pd.Series(rng.normal(0, 5, 300), index=pd.bdate_range("2023-01-02", periods=300))
    fr = -0.0007 * dy + rng.normal(0, 1e-4, 300)
    assert R.empirical_dv01(fr, dy, 120).iloc[-1] == pytest.approx(0.0007, rel=0.05)
    a = pd.Series(np.cumsum(rng.normal(0, 1, 300)) + 100, index=dy.index)
    b = 0.5 * a + rng.normal(0, 0.02, 300)
    pp = R.pair_spread_pnl(a, b, window=60)
    assert pp["beta"].iloc[-1] == pytest.approx(2.0, rel=0.2) and pp["pnl_per_unit"].iloc[100:].std() < a.diff().iloc[100:].std()


def test_42_attribution_adds_up_by_leg_and_group(syn):
    from nautilus_fi.kernel.analytics import attribution

    st = _value(overlay=nfi.Hedge(("TY",), ("oad",), targets={"oad": 0.0}, active=True))
    res = run(st, syn, "2015-03-01", "2015-12-31")
    a = attribution(res, syn, by="sector")
    t = a["total"]
    assert list(t.columns) == ["rates", "carry", "spread", "fx", "residual", "cash", "financing", "overlay", "costs", "net"]
    assert np.abs(t.drop(columns="net").sum(axis=1) - t["net"]).max() < 1e-12 and (t["overlay"] != 0).any() and (t["costs"] < 0).any()
    g = a["groups"].groupby("date")[["rates", "carry", "spread"]].sum()
    assert np.abs(g - t[["rates", "carry", "spread"]]).to_numpy().max() < 1e-12 and set(a["groups"]["group"]) == set(syn.data.attributes["sector"])
    assert abs(a["summary"].at["rates", "sum"]) > abs(a["summary"].at["residual", "sum"])          # the synthetic book is mostly rates
    act = attribution(res, syn, active=True)["total"]
    assert np.abs(act[["rates", "carry", "spread", "fx", "residual"]].sum(axis=1) - act["active"]).max() < 1e-12


def test_49_fx_hedge_programme_ratio_and_monthly_roll():
    from nautilus_fi.kernel.returns import fx_hedged_returns, unhedged_returns

    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    S = pd.Series(1.10 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.005, len(idx)))), index=idx)
    F = S * 1.002                                                                   # a 20bp premium per forward
    flat = pd.Series(0.0, index=idx)
    monthly = (1 + fx_hedged_returns(flat, S, F, roll="ME").fillna(0.0)).groupby(idx.month).prod()
    assert np.allclose(monthly, 1.002)                                              # the spot move hedged, the premium earned per roll
    assert np.allclose(fx_hedged_returns(flat, S, F, ratio=0.0, roll="ME").iloc[1:], unhedged_returns(flat, S).iloc[1:])
    half = fx_hedged_returns(flat, S, F, ratio=0.5)
    assert np.allclose(half, 0.5 * fx_hedged_returns(flat, S, F) + 0.5 * unhedged_returns(flat, S), equal_nan=True)


def test_49_hedges_solved_with_the_book(syn):
    from nautilus_fi.kernel.engine import prepare

    band = nfi.ActiveExposure("oad").within(0.05)
    st = _value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05, band), execution=nfi.SameClose(), overlay=nfi.Hedge(("TY",), ("oad",), joint=True, cost_bp=0.5, max_gross=2.0))
    res = run(st, syn, "2015-03-01", "2015-12-31")
    ov = res.diagnostics["overlays"].pivot(index="date", columns="instrument_id", values="notional")["TY"]
    prep = prepare([st], syn)
    ty = pd.Series(np.asarray(prep.overlays["oad"], dtype=float)[:, list(prep.overlays.ids).index("TY")], index=prep.overlays.dates)
    days = pd.to_datetime(res.trades["trade_date"])
    book = res.exposures["active_oad"].reindex(days)
    total = book + ov.reindex(days) * ty.reindex(days)
    assert (book.abs() > 0.1).any() and (total.abs() <= 0.05 + 1e-4).all() and res.violations.empty      # the hedge, not the book, holds the band
    assert "joint=True" in repr(st.overlay) and (ov.abs() <= 2.0 + 1e-9).all()
    seq = run(st.with_(overlay=nfi.Hedge(("TY",), ("oad",), targets={"oad": 0.0}, active=True)), syn, "2015-03-01", "2015-12-31")
    assert (seq.exposures["active_oad"].reindex(days).abs() <= 0.05 + 1e-4).all()                       # sized after the solve: the book holds it


def test_41_declared_forms_availability_windows_and_dated_categories(syn):
    d = syn.data.panel
    tr = d.frame("tr").astype(float)
    level = (1 + tr.fillna(0.0)).cumprod() * 100
    fields = {k: d.frame(k) for k in ("signal.value", "alive", "index_weight", "half_spread_bp")}
    base = nfi.FrameSource({**fields, "tr": tr}, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership})
    priced = nfi.FrameSource({**fields, "tr": level.where(tr.notna())}, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership}, info=[nfi.FieldInfo("tr", form="price")])
    a, b = run(_value(), base), run(_value(), priced)
    assert np.abs(a.gross - b.gross).iloc[2:].max() < 1e-10
    late = nfi.FrameSource({**fields, "tr": tr, "esg": d.frame("signal.value")}, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership}, info=[nfi.FieldInfo("esg", start="2016-01-04")])
    with pytest.raises(nfi.MissingDataError, match="Available"):
        run(_value(universe=nfi.Index("SYNTH_IG") & (nfi.Field("esg") > 0)), late)
    ok = run(_value(universe=nfi.Index("SYNTH_IG") & (~nfi.Available("esg") | (nfi.Field("esg") > 0))), late)
    assert ok.daily["n_names"].iloc[5] > 0
    sector = pd.DataFrame("IND", index=d.dates, columns=d.ids)
    sector.loc["2016-01-04":, d.ids[:10]] = "FIN"                     # reclassified mid-sample
    dated = nfi.FrameSource({**fields, "tr": tr}, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership}, categories={"sector_t": sector})
    r = run(_value(portfolio=nfi.Quantile(5, by="sector_t")), dated)
    assert set(r.groups["sector_t"]["weight"].columns) == {"FIN", "IND"}


# ====================================================================================== 4.7 optimiser
@pytest.fixture(scope="module")
def cross(syn):
    """A kernel cross-section on a mid-sample date, with the benchmark and the value signal."""
    from nautilus_fi.kernel.engine import _cross, prepare

    st = _value(portfolio=nfi.Optimise(nfi.Alpha() - nfi.Costs(), nfi.Weight() <= 0.05), fields=("xr", "currency"))
    prep = prepare([st], syn)
    ctx = prep.ctx
    t = int(ctx.panel.dates.searchsorted(pd.Timestamp("2016-01-29")))
    U = np.asarray(st.universe.evaluate(ctx), dtype=bool)
    Wk = st.benchmark.known_weights(ctx)
    S = np.asarray(st.signal.evaluate(ctx.within(U, "u")), dtype=float)
    w = np.where(Wk[t - 21] > 0, Wk[t - 21], 0.0)
    return _cross(t, st.resolved(), ctx, S[t], U[t], Wk[t], w, 1e8, extra_names=("oasd", "sector", "issuer", "currency"))


def _w(res):
    return res[0]


def test_47_tracking_error_limits_by_cutting_planes_and_maxir(cross):
    model = nfi.PCARisk(120, "xr", 5)
    base = nfi.Optimise(nfi.Alpha() - nfi.Costs(), nfi.Weight() <= 0.05)
    free_w, _ = base.weights(cross)
    te = nfi.TEV(model)
    loose = te.sigma(cross, free_w.reindex(cross.ids).fillna(0.0).to_numpy())
    limit = loose / 2
    w, d = base.add(te <= limit).weights(cross)
    tight = te.sigma(cross, w.reindex(cross.ids).fillna(0.0).to_numpy())
    assert limit * 0.98 <= tight <= limit * 1.002 and d["tev"][repr(te)]["aversion"] > 0            # the budget is spent, not undershot
    k, _ = base.add(te <= limit, nfi.MinTrade(0.001)).weights(cross)                                 # with integers: cutting planes
    assert te.sigma(cross, k.reindex(cross.ids).fillna(0.0).to_numpy()) <= limit * 1.3
    levels = (loose / 4, loose / 2, loose)
    _, dd = nfi.MaxIR(base, model, levels=levels).weights(cross)
    assert dd["chosen_te"] in levels and len(dd["frontier"]) == 3


def test_47_min_trade_tie_break_bounds_pins_and_trade_limits(cross):
    w0 = cross.current.reindex(cross.ids).fillna(0.0)
    base = nfi.Optimise(nfi.Alpha() - nfi.Costs(), nfi.Weight() <= 0.05, nfi.Turnover() <= 0.3)
    w, _ = base.add(nfi.MinTrade(0.004)).weights(cross)
    dw = (w.reindex(cross.ids).fillna(0.0) - w0).abs()
    closing = (w.reindex(cross.ids).fillna(0.0).abs() < 1e-9) & (w0.abs() < 0.004)
    assert ((dw < 1e-7) | (dw >= 0.004 - 1e-7) | closing).all()
    a, _ = base.weights(cross)
    tb, d = nfi.Optimise(nfi.Alpha() - nfi.Costs(), nfi.Weight() <= 0.05, nfi.Turnover() <= 0.3, tie_break=-nfi.ActiveShare()).weights(cross)
    score = cross.scores.reindex(cross.ids).fillna(0.0)
    act = lambda x: float((x.reindex(cross.ids).fillna(0.0) - cross.benchmark.reindex(cross.ids).fillna(0.0)).abs().sum())
    assert "tie_break" in d and float(score @ tb.reindex(cross.ids).fillna(0.0)) >= float(score @ a.reindex(cross.ids).fillna(0.0)) - 1e-6 and act(tb) <= act(a) + 1e-9
    held = list(w0[w0 > 0].index[:2])
    pinned, _ = base.add(nfi.Pin({held[0]: 0.02}), nfi.NoSell([held[1]]), nfi.TradeLimit(0.01)).weights(cross)
    pv = pinned.reindex(cross.ids).fillna(0.0)
    assert pv[held[0]] == pytest.approx(0.02, abs=1e-7) and pv[held[1]] >= w0[held[1]] - 1e-9 and ((pv - w0).abs() <= 0.01 + 1e-7).all()


def test_47_dated_bounds_subsets_relative_caps_and_currency(cross):
    base = nfi.Optimise(nfi.Alpha() - nfi.Costs(), nfi.Weight() <= 0.05)
    regime = lambda cs: 0.2 if cs.date.year == 2016 else 0.5
    w, _ = base.add(nfi.ActiveExposure("oasd").within(regime)).weights(cross)
    oasd = cross.data["oasd"].reindex(cross.ids).fillna(0.0)
    act = float(((w.reindex(cross.ids).fillna(0.0) - cross.benchmark.reindex(cross.ids).fillna(0.0)) * oasd).sum())
    assert abs(act) <= 0.2 + 1e-6
    fin = nfi.ActiveWeight("issuer").where(nfi.Field("sector") == "FIN") <= 0.0
    w2, _ = base.add(fin).weights(cross)
    sector, issuer = cross.data["sector"], cross.data["issuer"]
    gap = (w2.reindex(cross.ids).fillna(0.0) - cross.benchmark.reindex(cross.ids).fillna(0.0)).groupby(issuer).sum()
    fin_issuers = set(issuer[sector == "FIN"])
    assert all(gap[i] <= 1e-6 for i in fin_issuers)
    floor = 1.3 / issuer.nunique()                                     # caps must leave room to be fully invested
    w3, d3 = base.add(nfi.GroupWeight("issuer").cap(absolute=0.08, relative=3.0, floor=floor)).weights(cross)
    by = w3.reindex(cross.ids).fillna(0.0).groupby(issuer).sum()
    bw = cross.benchmark.reindex(cross.ids).fillna(0.0).groupby(issuer).sum()
    assert not [r for r in d3["relaxations"] if r.get("constraint") == "all limits"]
    assert (by <= np.maximum(np.minimum(0.08, 3.0 * bw), floor) + 1e-6).all() and (by > floor + 1e-6).any()
    w4, _ = base.add(nfi.CashWeight() <= 0.1, nfi.CurrencyExposure("USD") >= 1.0).weights(cross)
    assert w4.sum() <= 1.0 + 1e-6


def test_47_relaxation_ladder_strict_limits_conflicts_and_push(cross, syn):
    impossible = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05, nfi.Exposure("oasd") <= 0.5)
    _, d = impossible.weights(cross)
    rel = [r for r in d["relaxations"] if r.get("constraint") == "all limits"]
    assert rel and rel[0]["tier"] in ("grandfathered", "widened x1.5", "widened x2", "softened")
    assert any(r.get("constraint") == "conflict" for r in d["relaxations"])
    strict = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05, (nfi.Exposure("oasd") <= 0.5).strict())
    _, ds = strict.weights(cross)
    assert ds["ok"] is False                                           # a strict limit is never relaxed: the book is kept
    st = _value(portfolio=impossible.with_settings(on_infeasible="push"))
    r = run(st, syn, end="2015-04-30")
    assert r.diagnostics["pushed_rebalances"] > 5 and (r.construction["action"] == "pushed").any()


def test_47_sparsity_targets_and_turnover_budgets(cross):
    w, d = nfi.Sparse(nfi.Optimise(nfi.Alpha() - 0.002 * nfi.ActiveShare(), nfi.Weight() <= 0.2), names=(8, 15)).weights(cross)
    k = int((w.abs() > 1e-9).sum())
    assert 8 <= k <= 15 or len(d["sparsity"]) >= 2
    budget = nfi.AnnualTurnoverBudget(1.2, month_end_share=0.4)
    assert budget(cross) == pytest.approx(1.2 / 12 * 0.4)                # 2016-01-29 is a month end
    assert nfi.IndexTurnover(nfi.Index("SYNTH_IG"), allowance=0.01)(cross) >= 0.01


# ====================================================================================== 4.4 universe and benchmark / 4.6 construction
def test_44_paired_universes_redemptions_mv_reweight_and_stats(syn):
    from nautilus_fi.kernel.data import DataRequest
    from nautilus_fi.kernel.expr import EvalContext

    p = syn.load(DataRequest(("tr", "alive", "amt_out", "price_clean", "oas_bp"), None, None))
    ctx = EvalContext(p, syn.attributes(p.ids), None, syn, syn.catalog())
    ix = nfi.Index("SYNTH_IG")
    from nautilus_fi.domain.conventions import WeightDrift
    from nautilus_fi.kernel.benchmark import drift_weights

    alive = np.array([[True, True], [True, True], [True, False], [True, False]])
    R = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, np.nan], [0.0, np.nan]])
    assert drift_weights({0: np.array([0.5, 0.5])}, R, alive, WeightDrift.DRIFT)[2].tolist() == [1.0, 0.0]
    assert drift_weights({0: np.array([0.5, 0.5])}, R, alive, WeightDrift.DRIFT, "cash")[2].tolist() == [0.5, 0.0]   # the redeemed half waits in cash
    mv = ix.reweight("market_value", field="amt_out", schedule="BME")
    W = mv.weights(ctx)
    t = int(p.dates.searchsorted(pd.Timestamp("2015-06-30")))
    m = W[t] > 0
    amt = np.nan_to_num(p["amt_out"][t])
    assert np.allclose(W[t][m] / W[t][m].sum(), amt[m] / amt[m].sum(), atol=1e-6)
    paired = nfi.PairedIndex(ix.exclude(sector="FIN"), ix)
    assert (paired.known_weights(ctx) > 0).sum() < (paired.weights(ctx) > 0).sum() and np.allclose(paired.returns(ctx), ix.returns(ctx))
    stat = nfi.BenchmarkStat("oas_bp", ix).evaluate(ctx)
    wide = run(_value(universe=ix & (nfi.Field("oas_bp") > nfi.BenchmarkStat("oas_bp", ix)) & (nfi.Field("oas_bp") < nfi.Field("oas_bp").cs_quantile(0.9)), benchmark=None), syn)
    assert np.isfinite(stat[50, 0]) and wide.daily["n_names"].iloc[5] > 0
    rep = nfi.replication_report(ix, syn.data.index["tr"], syn)
    assert rep["within_tolerance"] and abs(rep["tracking_difference_bp"]) < 1.0              # synthetic index: built with the same formula


def test_44_group_constraints_skip_small_benchmark_groups(syn):
    from nautilus_fi.kernel.engine import _cross, prepare

    st = _value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05, nfi.ActiveWeight("sector").within(0.0)))
    prep = prepare([st], syn)
    ctx = prep.ctx
    t = int(ctx.panel.dates.searchsorted(pd.Timestamp("2016-01-29")))
    U = np.asarray(st.universe.evaluate(ctx), dtype=bool)
    Wk = st.benchmark.known_weights(ctx)
    S = np.asarray(st.signal.evaluate(ctx.within(U, "u")), dtype=float)
    cs = _cross(t, st.resolved(), ctx, S[t], U[t], Wk[t], np.zeros(ctx.N), 1e8)
    bsec = cs.benchmark.groupby(cs.data["sector"]).sum()
    small = bsec[bsec > 0].idxmin()
    exact, _ = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.2, nfi.ActiveWeight("sector").within(0.0)).weights(cs)
    skip, _ = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.2, nfi.ActiveWeight("sector").within(0.0).skip_below(float(bsec[small]) + 1e-9)).weights(cs)
    g = lambda w: w.reindex(cs.ids).fillna(0.0).groupby(cs.data["sector"]).sum()
    assert abs(g(exact)[small] - bsec[small]) < 1e-6 and abs(g(skip).drop(small) - bsec.drop(small)).max() < 1e-6


def test_46_constructions_buffers_strata_zscore_rules_and_waterfall(syn):
    plain = run(_value(), syn)
    buf = run(_value(portfolio=nfi.Quantile(5, buffer=1)), syn)
    keep = run(_value(portfolio=nfi.Quantile(5, keep_weights=True)), syn)
    assert buf.stats()["turnover"] < plain.stats()["turnover"] and keep.stats()["turnover"] < plain.stats()["turnover"]
    strat = run(_value(portfolio=nfi.Stratified(per_cell=2)), syn)
    assert strat.construction["cells_filled"].iloc[-1] > 3 and abs(strat.weights.groupby("date")["weight"].sum().iloc[-1] - 1.0) < 1e-9
    from nautilus_fi.data.panel import Panel
    from nautilus_fi.kernel.expr import EvalContext

    path = np.array([0.0, 2.1, 1.0, 0.2, -2.5, -1.0, -5.0, -1.0, 0.1, 2.2, 2.0, 2.0, 2.0])[:, None]
    zc = EvalContext(Panel(pd.bdate_range("2024-01-01", periods=len(path)), pd.Index(["A"]), {"z": path}))
    pos = nfi.ZScoreRule("z", entry=2.0, exit=0.5, stop=4.0, max_hold=2).positions(zc)[:, 0]
    assert pos.tolist() == [0, -1, -1, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0]          # enter, exit at 0.2, stop at -5 (out until inside), max hold 2
    zr = run(_value(portfolio=nfi.ZScoreRule(nfi.Field("oas_bp").zscore(), entry=1.5, exit=0.3, size=0.02, sign=1.0), financing=nfi.Cash("cash", short_fee_bp=0)), syn)
    assert (zr.weights["weight"] > 0).any()
    rules = nfi.Rules(nfi.TopN(30), [(nfi.Field("rating_score") > 7, "sell"), (nfi.Field("oasd") > 8, ("buymax", 0.01))])
    rr = run(_value(portfolio=rules), syn)
    last = rr.weights[rr.weights["date"] == rr.weights["date"].max()].set_index("instrument_id")["weight"]
    rating = syn.data.panel.frame("rating_score").loc[rr.weights["date"].max()]
    assert (rating.reindex(last.index) <= 7).all() and "rule_sell" in rr.construction.columns
    wf = run(_value(overlay=nfi.KRDWaterfall([("oad", "TY"), ("oasd", "CDX_IG")], targets={"oad": 0.0, "oasd": lambda cs: 1.0})), syn)
    assert "overlays" in wf.diagnostics and (wf.diagnostics["overlays"]["notional"] < 0).any()


# ====================================================================================== 4.10 metrics and reporting
def test_410_relative_capture_windows_tables_timing_and_breakdowns(syn):
    from nautilus_fi.kernel import analytics as A

    r = run(_value(), syn)
    rd = A.relative_drawdown(r)
    assert rd["drawdown"].max() <= 1e-12 and A.relative_drawdown(r, "additive")["drawdown"].min() <= 0
    cap = A.capture(r)
    assert {"up_capture", "down_capture", "jensen_alpha", "beta", "appraisal_ratio"} <= set(cap) and "beta" in r.stats()
    w = A.windows(r, presets=("YTD", "1Y", "SI"), oos_start="2016-01-01", trailing=(63,))
    assert {"YTD", "1Y", "SI", "OOS", "last_63"} <= set(w.index) and w.loc["SI", "total_return"] == pytest.approx(r.stats()["total_return"])
    yt = A.period_table(r)
    assert "cumulative" in yt.index and yt.loc["2015", "portfolio"] == pytest.approx(float((1 + r.returns["2015"].iloc[1:]).prod() - 1) if r.returns.index[0].year == 2015 else 0.0, abs=1e-12)
    assert len(A.period_table(r, "ME")) == 19
    ts = A.timing_stats(r)
    assert 0 <= ts["time_neutral"] <= 1 and len(ts["rolling_sharpe"]) == len(r.returns) - 1
    regimes = pd.Series(np.where(r.returns.index.year == 2015, "calm", "stress"), index=r.returns.index)
    b = A.breakdowns(r, regimes, drawdown_threshold=-0.001)
    assert set(b["by_regime"].index) == {"calm", "stress"} and "whipsaws" in b
    hr = A.hit_rates(r, source=syn)
    assert 0 <= hr["active_hit_rate"] <= 1 and 0 <= hr["trade_hit_rate"] <= 1
    ws = A.window_stats(r)
    assert len(ws) == len(r.trades) - (1 if ws["end"].iloc[-1] == ws["start"].iloc[-1] else 0) or len(ws) > 10


def test_410_risk_attribution_blotter_coverage_and_writers(syn, tmp_path):
    import sqlite3

    from nautilus_fi.kernel import analytics as A
    from nautilus_fi.kernel.engine import _cross, prepare

    r = run(_value(), syn)
    split = A.carry_spread_split(r, syn)
    assert np.allclose(split["carry"] + split["spread"] + split["residual"], split["active"])
    fc = A.factor_contributions(r, syn, fields=("oasd",))
    assert "oasd" in fc.columns and len(fc) > 5
    bl = A.trade_blotter(r)
    assert len(bl) > 0 and set(bl["side"]) <= {"BUY", "SELL"} and bl["cost"].sum() == pytest.approx(r.trades["cost"].sum(), rel=1e-6)
    cov = A.data_coverage(r)
    assert (cov["weight_plus_cash"].iloc[1:] - 1.0).abs().max() < 1e-9
    st = _value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05))
    prep = prepare([st], syn)
    t = int(prep.ctx.panel.dates.searchsorted(pd.Timestamp("2016-01-29")))
    U = np.asarray(st.universe.evaluate(prep.ctx), dtype=bool)
    Wk = st.benchmark.known_weights(prep.ctx)
    cs = _cross(t, st.resolved(), prep.ctx, np.zeros(prep.ctx.N), U[t], Wk[t], np.zeros(prep.ctx.N), 1e8, extra_names=("oas_bp",))
    rm = nfi.PCARisk(120, "xr", 5)(cs) if hasattr(nfi, "PCARisk") else None
    w, _ = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05).weights(dataclasses_replace(cs, scores=pd.Series(np.random.default_rng(0).normal(size=len(cs.ids)), index=cs.ids)))
    risk = A.ex_ante_risk(w, cs.benchmark, rm, limit=0.02, carry=cs.data["oas_bp"], groups=cs.data.get("sector"))
    assert risk["tev"] > 0 and abs(risk["contribution"].sum() - risk["tev"]) < 1e-9 and "active_carry" in risk and risk["budget_utilisation"] > 0
    j = A.write_json(r, tmp_path / "r.json", as_of="2026-10-31", stale_after_days=3)
    import json

    payload = json.loads(j.read_text())
    assert payload["stale"] and payload["data_end"] == "2016-06-30" and payload["warnings"]
    con = sqlite3.connect(tmp_path / "out.db")
    assert A.write_sql(r, con, as_of="2016-07-01") == 1 and con.execute("select stale from BACKTEST_RESULTS").fetchone()[0] == 0


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


def test_410_signal_extras(syn):
    from nautilus_fi.kernel import analytics as A
    from nautilus_fi.kernel import returns as R

    d = syn.data.panel
    scores = d.frame("signal.value").iloc[::21]
    fw = R.forward_returns(d.frame("tr").astype(float), horizons=(21,), lag=1, alive=d.frame("alive"), dates=scores.index)
    fwd = fw.pivot_table(index="date", columns="instrument_id", values="fwd_tr")
    base = -scores
    ex = A.signal_extras(scores, fwd, baseline=base, issuers=syn.data.attributes["issuer"], dts=d.frame("dts").iloc[::21], control=syn.data.attributes[["coupon"]].assign(coupon=lambda x: x["coupon"] * 100), tolerance={"coupon": 0.5})
    assert ex["ic_uplift"] > 0 and {"spread_mean", "spread_median", "spread_winsorised_mean", "monotonicity", "monotonic_increasing"} <= set(ex) and ex["control_matches"] > 0
    neutral = A.signal_extras(scores, fwd, groups=syn.data.attributes["sector"])
    assert np.isfinite(neutral["mean_ic"])


# ====================================================================================== 4.11 research and validation
def test_411_walk_forward_gated_selection_events_and_placebos(syn, tmp_path):
    from nautilus_fi.kernel import validation as V
    from nautilus_fi.research.walkforward import walk_forward_splits

    sess = nfi.Session(tmp_path, source=syn)
    seen = []

    def builder(params, train):
        seen.append(train)
        return _value(portfolio=nfi.TopN(params["n"])).with_(name=f"top{params['n']}")

    splits = walk_forward_splits(date(2015, 1, 2), date(2016, 6, 30), dev_months=6, val_months=4, test_months=4, step_months=4)
    wf = V.walk_forward(sess, builder, [{"n": 10}, {"n": 30}], splits, label_horizon=21, embargo=5)
    assert len(wf["folds"]) == len(splits) and len(wf["oos_returns"]) > 50
    val_starts = [pd.Timestamp(s.dev_end) for s in splits]
    assert all(tr[1] < vs - pd.Timedelta(days=21) for tr, vs in zip(seen[1::2][: len(splits)], val_starts, strict=False))     # purged training windows
    stages = {t["stage"] for t in sess.ledger.trials_list()}
    assert {"VALIDATION", "OOS"} <= stages
    rs = [sess.run(_value(portfolio=nfi.TopN(n)).with_(name=f"t{n}"), record=False) for n in (10, 20, 40)]
    pick = V.select_gated(rs, primary="sharpe", tolerance=0.5, pnl="annual_return", floor=0.0)
    assert pick in rs
    with pytest.raises(V.SelectionFailure):
        V.select_gated(rs, primary="sharpe", tolerance=0.0, pnl="annual_return", floor=10.0)
    ev = pd.Series([np.nan] * 10 + ["A"] * 5 + [np.nan] * 5 + ["B"] * 5 + [np.nan] * 5, index=pd.bdate_range("2024-01-01", periods=30))
    folds = V.event_splits(ev, n_splits=2, embargo=2)
    assert all(not set(f["test"]) & set(f["train"]) for f in folds) and len(folds) == 2
    pl = V.placebos(sess, _value(portfolio=nfi.TopN(20)), n=2)
    assert len(pl["table"]) == 5 and 0 <= pl["p_value"] <= 1 and pl["table"].loc[pl["table"]["kind"] == "anti", "sharpe"].iloc[0] < pl["real"]


def test_411_labels_preregistration_and_sweep_gates(syn, tmp_path):
    from nautilus_fi.governance import TrialLedger
    from nautilus_fi.kernel import validation as V
    from nautilus_fi.research import Hypothesis

    r = run(_value(), syn)
    lab = V.label_periods(r, "2015-12-31", "2016-03-31")
    assert set(lab["stats"].index) == {"IS", "VAL", "OOS"}
    led = TrialLedger(tmp_path)
    led.record("PAPER", "early", "x", {}, ("2015-06-01", "2016-02-01"), {}, None)
    led.register(Hypothesis("H-PRE", "x", "u", "BME", "s", "b", 1e8, "c", "1m", "hs", "sharpe < 0", ("2015-01-01", "2015-06-30"), ("2015-07-01", "2015-12-31"), ("2016-01-01", "2016-06-30")))
    pre = V.preregistered(r, led, "H-PRE", subperiods=[("2015-01-01", "2015-06-30"), ("2015-07-01", "2015-12-31")], shadow_start="2015-07-01")
    assert pre["contaminated"] and len(pre["subperiods"]) == 2 and pre["shadow"]["verdict"] in ("PASS", "FAIL") and pre["shadow"]["stage"] == "final"
    ok, why = V.eligible(r, max_turnover=0.01, max_gap_days=40)
    assert not ok and any("turnover" in w for w in why) and V.eligible(r, max_gap_days=10)[1]
    board = V.leaderboard([r, run(_value(portfolio=nfi.TopN(40)), syn)], metric="sharpe", max_turnover=100)
    assert board["eligible"].all() and board["sharpe"].is_monotonic_decreasing


# ====================================================================================== 4.12 reproducibility and provenance
def test_412_saved_configs_idempotent_trials_quality_inputs_and_regime_records(syn, tmp_path):
    from nautilus_fi.governance import TrialLedger
    from nautilus_fi.kernel.strategy import StrategyError

    st = _value(universe=nfi.Index("SYNTH_IG") & (nfi.Field("oasd") > 1), portfolio=nfi.Optimise(nfi.Alpha() - nfi.Costs(), nfi.Weight() <= 0.05), schedule=nfi.Triggered(0.1, "score_drop", base=nfi.Weekly(2)), metrics=nfi.Metrics(annualisation=nfi.Trading252(), pnl="ADDITIVE"))
    st.save(tmp_path / "s.json")
    back = nfi.Strategy.load(tmp_path / "s.json")
    assert back.fingerprint == st.fingerprint
    with pytest.raises(StrategyError, match="custom code"):
        _value(signal=nfi.Field("signal.value").apply(lambda v: v * 2)).to_code()
    led = TrialLedger(tmp_path / "led")
    a = led.record("PAPER", "s", "h", {}, ("2020-01-01", "2020-02-01"), {}, None, session="x", data_version="v")
    b = led.record("PAPER", "s", "h", {}, ("2020-01-01", "2020-02-01"), {}, None, session="x", data_version="v")
    assert a.trial_id == b.trial_id and len(list((tmp_path / "led" / "trials").glob("*.json"))) == 1
    idx, problems = led.rebuild_index()
    assert len(idx) == 1 and not problems and (tmp_path / "led" / "index.parquet").exists()
    sess = nfi.Session(tmp_path / "s", source=syn, keep_inputs=True, settings=nfi.RunSettings(record_fields=("cash",)))
    r = sess.run(_value(fields=("cash",)))
    t = sess.ledger.trials_list()[-1]
    assert t["summary"]["quality_failed_rebalances"] == 0 and t["summary"]["code_version"] and pathlib_exists(t["summary"]["inputs"])
    assert "record_cash" in r.construction.columns and r.quality()["n_rebalances"] == len(r.trades)


def pathlib_exists(p) -> bool:
    from pathlib import Path

    return Path(p).exists()


# ====================================================================================== 4.13 engine parity
def _syn_book(syn, as_of, weights: pd.Series, nav: float):
    """Bonds, bid/ask prices and face holdings for synthetic names at ``as_of``."""
    from nautilus_fi.instruments.bond import BondSpec

    panel = syn.data.panel
    t = int(panel.dates.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    clean = pd.Series(panel["price_clean"][t], index=panel.ids)
    half = pd.Series(panel["half_spread_bp"][t], index=panel.ids).fillna(5.0) / 1e4 * clean
    a = syn.data.attributes
    bonds = {s: BondSpec(s, str(a.at[s, "issuer"]), pd.Timestamp(a.at[s, "issue_date"]).date(), pd.Timestamp(a.at[s, "maturity"]).date(), float(a.at[s, "coupon"]), sector=str(a.at[s, "sector"]), min_piece=1000.0, increment=1000.0) for s in panel.ids}
    prices = pd.DataFrame({"bid": clean - half, "ask": clean + half, "mid": clean, "price_date": pd.Timestamp(as_of).date()}).dropna()
    face = (weights * nav / (clean.reindex(weights.index) / 100.0)).round(-3)
    return bonds, prices, face[face > 0]


def test_413_constant_exposure_holds_the_traded_weights(syn):
    drift = _value(schedule="BMS", execution=nfi.NextClose(1, nfi.HalfSpread()))
    const = drift.with_(hold="constant")
    a, b = run(drift, syn, "2015-03-01", "2015-08-31"), run(const, syn, "2015-03-01", "2015-08-31")
    trade_dates = set(pd.to_datetime(b.trades["trade_date"]))
    W = pd.DataFrame(b.daily_weights, index=b.returns.index, columns=b.ids)
    Wd = pd.DataFrame(a.daily_weights, index=a.returns.index, columns=a.ids)
    checked = 0
    for d0, d1 in zip(W.index[:-1], W.index[1:], strict=True):
        if d1 in trade_dates or d0 < min(trade_dates):
            continue
        same = W.loc[d1].ne(0) == W.loc[d0].ne(0)                               # no name redeemed in between
        if same.all():
            assert np.allclose(W.loc[d1], W.loc[d0])                            # held constant between rebalances
            checked += 1
    assert checked > 50 and not np.allclose(Wd.iloc[30], Wd.iloc[31])         # the drift run's weights move
    gross = (W.shift(1).fillna(0.0) * pd.DataFrame(syn.data.panel["tr"], index=syn.data.panel.dates, columns=syn.data.panel.ids).reindex(index=W.index, columns=W.columns).fillna(0.0)).sum(axis=1)
    assert np.allclose(b.gross.iloc[1:], (gross + b.daily["cash_return"]).iloc[1:], atol=2e-6)   # each day earns the held weights' returns
    assert abs(a.stats()["annual_return"] - b.stats()["annual_return"]) > 1e-6
    assert nfi.Strategy.from_code(const.to_code()).fingerprint == const.fingerprint and "hold=" in const.to_code() and "constant" in const.describe()
    with pytest.raises(Exception, match="constant"):
        nfi.Strategy(hold="sideways")
    from nautilus_fi.kernel.implement import run_implement

    with pytest.raises(ValueError, match="Paper fidelity"):
        run_implement(const, syn, "2015-03-01", "2015-04-30")
    with pytest.raises(MemoryError, match="validation tier"):                    # sized before the environment is built
        run_implement(drift, syn, "2015-03-01", "2015-04-30", memory_limit_gb=1e-6)


def test_413_decide_at_matches_the_backtest_and_proposes_from_a_strategy(syn):
    from nautilus_fi.propose import propose_from_strategy

    st = _value(portfolio=nfi.Quantile(5), schedule="BME")
    res = run(st, syn, "2015-06-01", "2016-04-08")
    as_of = pd.Timestamp("2016-03-31")
    held = res.weights[res.weights["date"] == as_of].set_index("instrument_id")["weight"]
    d = nfi.decide_at(st, syn, as_of, current=held, nav=float(res.nav.loc[as_of]))
    traded = res.weights[res.weights["date"] == pd.Timestamp("2016-04-01")].set_index("instrument_id")["weight"]
    assert d.action == "rebalance" and d.scheduled and d.as_of == as_of
    assert set(d.weights.index) == set(traded.index) and np.allclose(d.weights.reindex(traded.index), traded, atol=1e-9)
    assert not nfi.decide_at(st, syn, "2016-03-30", current=held).scheduled
    nav = 50e6
    bonds, prices, face = _syn_book(syn, as_of, held, nav)
    cash = nav - float((face * prices["mid"].reindex(face.index) / 100.0).sum())
    p = propose_from_strategy(st, syn, as_of, face, prices, bonds, cash, coverage_fields=("oasd",))
    assert p.spec_hash == st.fingerprint and p.extra["decision"].action == "rebalance" and len(p.lines) > 0
    assert "severity" in p.validation and not (p.validation["check"] == "schedule").any()
    assert not p.validation["check"].str.startswith("coverage:").any()
    after = p.holdings_after.set_index("symbol")["weight"]
    assert abs(after.sum() - d.weights.reindex(after.index).fillna(0).sum()) < 0.01   # within rounding of the strategy's own target
    capped = st.with_(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.02))
    q = propose_from_strategy(capped, syn, as_of, face, prices, bonds, cash, min_trade_face=0.0)
    assert (q.holdings_after["weight"] <= 0.02 + 0.002).all()                  # the cap holds after rounding (or is reported)
    book = res.weights[res.weights["date"] == res.weights.loc[res.weights["date"] <= as_of, "date"].max()].set_index("instrument_id")["weight"]
    trig = st.with_(schedule=nfi.Triggered(1e9, "score_drop", base="BME"))
    skip = nfi.decide_at(trig, syn, as_of, current=book, last_decision="2016-02-29")
    assert skip.action == "trigger_skip" and skip.log["value"] != 0.0 and skip.weights.equals(skip.current)   # both decisions' scores read
    from nautilus_fi.propose import apply_feedback

    off = propose_from_strategy(st, syn, "2016-03-30", face, prices, bonds, cash)
    assert "schedule" in set(off.validation["check"])
    again = apply_feedback(off, [{"symbol": off.lines["symbol"].iloc[0], "action": "reject"}])
    assert "schedule" in set(again.validation["check"]) and again.extra["decision"] is off.extra["decision"]


# ====================================================================================== 4.14 live proposals
def _small_book():
    from nautilus_fi.instruments.bond import BondSpec, Identifiers

    bonds = {s: BondSpec(s, iss, date(2020, 1, 15), date(2030, 1, 15), 0.05, sector=sec, min_piece=1000, increment=1000, identifiers=Identifiers(cusip=f"12345{i}AB1")) for i, (s, iss, sec) in enumerate([("A", "I1", "FIN"), ("B", "I1", "FIN"), ("C", "I2", "IND"), ("D", "I3", "UTL")])}
    prices = pd.DataFrame({"bid": [99.5, 101.0, 98.0, 100.0], "ask": [99.8, 101.3, 98.4, 100.4], "accrued": [1.2, 0.5, 2.0, 0.1], "price_date": [date(2024, 6, 3)] * 3 + [date(2024, 5, 20)]}, index=list(bonds))
    return bonds, prices


def test_414_minimums_caps_cash_target_flows_mv_and_severities():
    from nautilus_fi.propose import build_proposal, carry_analytics, severity_of

    bonds, prices = _small_book()
    as_of = date(2024, 6, 3)
    hold = {"A": 300_000, "C": 300_000}
    base = build_proposal({"A": 0.3046, "B": 0.55, "C": 0.1}, hold, prices, bonds, 400_000.0, as_of, min_trade_bps=50, max_position_weight=0.4, cash_target=0.02)
    lines = base.lines.set_index("symbol")
    assert "A" not in lines.index and "below_min_trade" in set(base.validation["check"])          # a 2,000 face trade is under 50bp of NAV
    assert base.holdings_after.set_index("symbol").at["B", "weight"] <= 0.4 + 1e-9                 # capped
    assert base.validation.set_index("check").at["cash_target", "severity"] == "warning" and base.ok   # 19.5% cash against a 2% target
    full = build_proposal({"A": 0.3, "B": 0.4, "C": 0.3}, hold, prices, bonds, 400_000.0, as_of, cash_target=0.02)
    assert full.holdings_after["weight"].sum() == pytest.approx(0.98, abs=0.003) and full.validation.set_index("check").at["cash_target", "status"] == "PASS"
    mv = build_proposal({"A": 0.3}, hold, prices, bonds, 400_000.0, as_of, reported_nav=2_000_000.0)
    assert not mv.ok and set(mv.blockers["check"]) == {"mv_reconciliation"}
    low = build_proposal({"A": 0.5, "B": 0.49}, hold, prices, bonds, 400_000.0, as_of, cash_bounds=(0.02, 0.10))
    assert "cash_min" in set(low.blockers["check"])
    inflow = build_proposal({"A": 0.1, "B": 0.5}, hold, prices, bonds, 400_000.0, as_of, cash_flow=+200_000.0)
    assert (inflow.lines["side"] == "BUY").all() and "cash_flow_mode" in set(inflow.validation["check"])
    outflow = build_proposal({"A": 0.1, "B": 0.5}, hold, prices, bonds, 400_000.0, as_of, cash_flow=-200_000.0)
    assert (outflow.lines["side"] == "SELL").all()
    stale = build_proposal({"A": 0.3, "D": 0.2}, hold, prices, bonds, 400_000.0, as_of, restricted={"D"})
    sev = stale.validation.set_index("check")["severity"]
    assert sev["stale_price"] == "approval_required" and sev["restricted_list"] == "blocker" and not stale.ok
    assert severity_of("constraint:group:sector", "WARN") == "approval_required" and severity_of("constraint:x", "FAIL") == "blocker" and severity_of("anything", "PASS") == "info"
    assert severity_of("stale_price", "WARN", {"stale_price": "warning"}) == "warning"
    an = pd.DataFrame({"signal": [1.0, np.nan, 0.5], "oasd": [5.0, 6.0, np.nan]}, index=["A", "B", "C"])
    cov = build_proposal({"A": 0.2, "B": 0.4, "C": 0.1}, hold, prices, bonds, 400_000.0, as_of, analytics=an, coverage_min={"oasd": 0.5})
    c = cov.validation.set_index("check")
    assert c.at["coverage:signal", "severity"] == "approval_required" and "coverage:oasd" not in c.index
    today = pd.DataFrame({"oasd": [5.0, np.nan]}, index=["A", "B"])
    filled, rep = carry_analytics(today, pd.DataFrame({"oasd": [4.9, 6.1]}, index=["A", "B"]), "2024-06-03", "2024-05-31")
    assert filled.at["B", "oasd"] == 6.1 and list(rep["symbol"]) == ["B"]
    _, none = carry_analytics(today, pd.DataFrame({"oasd": [4.9, 6.1]}, index=["A", "B"]), "2024-06-03", "2024-05-28")
    assert none.empty                                                                             # older than a business day: not carried
    car = build_proposal({"A": 0.3}, hold, prices, bonds, 400_000.0, as_of, carried=rep)
    assert car.validation.set_index("check").at["analytics_carried", "severity"] == "warning"


def test_414_headerless_tsv_futures_lines_and_pm_feedback(tmp_path):
    from nautilus_fi.propose import (
        QOE_TSV,
        FutureTerms,
        add_futures,
        apply_feedback,
        build_proposal,
        contracts_from_overlays,
        write_orders,
    )

    bonds, prices = _small_book()
    as_of = date(2024, 6, 3)
    p = build_proposal({"A": 0.3, "B": 0.25, "C": 0.1}, {"A": 200_000, "C": 300_000}, prices, bonds, 600_000.0, as_of, portfolio="FI")
    f = write_orders(p, tmp_path / "orders.tsv", QOE_TSV)
    raw = f.read_text().splitlines()
    assert len(raw) == len(p.lines) and raw[0].split("\t")[0].startswith("FI-20240603") and len(raw[0].split("\t")) == len(QOE_TSV.columns)
    ty = FutureTerms("TYU4", 100_000, first_notice=date(2024, 6, 6), security_id="TYU4 Comdty", margin_init=2_000)
    n = contracts_from_overlays({"TYU4": -0.5}, p.nav, {"TYU4": ty}, {"TYU4": 110.0})
    assert n["TYU4"] == pytest.approx(-0.5 * p.nav / 110_000)
    fp = add_futures(p, {"TYU4": 4.4}, {"TYU4": ty}, {"TYU4": 110.0}, current={"TYU4": 1})
    fl = fp.lines[fp.lines["instrument"] == "FUTURE"].iloc[0]
    assert fl["face"] == 3 and fl["side"] == "BUY" and fl["cash"] == 0.0 and fl["security_id"] == "TYU4 Comdty"
    sev = fp.validation.set_index("check")["severity"]
    assert sev["futures_lot"] == "warning" and sev["futures_roll"] == "approval_required"
    late = add_futures(p, {"TYU4": 4}, {"TYU4": FutureTerms("TYU4", 100_000, first_notice=date(2024, 6, 3))}, {"TYU4": 110.0})
    assert "futures_expiry" in set(late.blockers["check"])
    poor = add_futures(p, {"TYU4": 400}, {"TYU4": ty}, {"TYU4": 110.0})
    assert "futures_margin" in set(poor.blockers["check"])
    fb = apply_feedback(fp, [{"symbol": "B", "action": "reject", "comment": "issuer on watch"}, {"symbol": "C", "action": "resize", "face": 50_000}, {"symbol": "TYU4", "action": "resize", "face": 2, "side": "BUY"}, {"check": "futures_roll", "action": "approve"}])
    lines = fb.lines.set_index("symbol")
    assert "B" not in lines.index and lines.at["C", "face"] == 50_000 and lines.at["TYU4", "face"] == 2
    v = fb.validation
    assert (v["check"] == "pm_feedback").sum() == 4 and (v.loc[v["check"] == "futures_roll", "severity"] == "approved").all()
    d = apply_feedback(p, [{"symbol": "A", "action": "defer"}])
    assert d.extra["deferred"] == ["A"] and "A" not in set(d.lines["symbol"])
    with pytest.raises(ValueError, match="unknown feedback action"):
        apply_feedback(p, [{"symbol": "A", "action": "shrug"}])


def test_414_external_books_units_and_fills():
    from nautilus_fi.propose import (
        book_to_face,
        build_proposal,
        lines_in_book_units,
        positions_after_fills,
        reconcile_fills,
    )

    bonds, prices = _small_book()
    book = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "quantity": [100_000, 200_000, 50, 1_015_000.0], "unit": ["face", "notional", "units", "market_value"], "factor": [np.nan, 0.5, np.nan, np.nan], "unit_size": [np.nan, np.nan, 1000, np.nan]})
    face = book_to_face(book, prices=prices.assign(mid=[99.65, 101.15, 98.2, 100.2]))
    assert face["A"] == 100_000 and face["B"] == 100_000 and face["C"] == 50_000 and face["D"] == pytest.approx(1_015_000 / 1.003)
    with pytest.raises(ValueError, match="pool factor"):
        book_to_face(pd.DataFrame({"symbol": ["B"], "quantity": [1.0], "unit": ["notional"]}))
    p = build_proposal({"A": 0.3, "B": 0.25}, {"A": 200_000}, prices.assign(factor=[1.0, 0.5, 1.0, 1.0]), bonds, 600_000.0, date(2024, 6, 3))
    u = lines_in_book_units(p, {"B": "notional"}).set_index("symbol")
    assert u.at["B", "quantity"] == pytest.approx(u.at["B", "face"] / 0.5) and u.at["B", "unit"] == "notional"
    oid = p.lines.set_index("symbol")["order_id"]
    buy_b = p.lines.set_index("symbol").at["B", "face"]
    fills = pd.DataFrame([{"order_id": oid["B"], "symbol": "B", "side": "BUY", "face": buy_b / 2, "price": 101.3}, {"order_id": oid["B"], "symbol": "B", "side": "BUY", "face": buy_b / 2, "price": 101.9}, {"order_id": "X-1", "symbol": "C", "side": "BUY", "face": 5_000, "price": 98.4}])
    rec = reconcile_fills(p, fills, price_tolerance_bp=25).set_index("order_id")
    assert rec.at[oid["B"], "status"] == "FILLED" and rec.at[oid["B"], "avg_price"] == pytest.approx(101.6)
    assert rec.at[oid["B"], "slippage_bp"] == pytest.approx((101.6 / 101.3 - 1) * 1e4) and rec.at[oid["B"], "price_status"] == "PRICE_BREAK"
    assert rec.at[oid["A"], "status"] == "UNFILLED" and rec.at["X-1", "status"] == "UNEXPECTED"
    after = positions_after_fills({"A": 200_000}, fills)
    assert after["B"] == buy_b and after["C"] == 5_000


# ====================================================================================== 6 operational interfaces
def test_6_target_book_reconciliation_store_and_record_links(syn, tmp_path):
    from nautilus_fi.governance import attach
    from nautilus_fi.governance.records import RecordRegistry
    from nautilus_fi.propose import (
        TARGET_BOOK_COLUMNS,
        BookStore,
        FileBookStore,
        build_proposal,
        read_target_book,
        reconcile_book,
        reconcile_returns,
        target_book,
        write_target_book,
    )

    bonds, prices = _small_book()
    p = build_proposal({"A": 0.3, "B": 0.25}, {"A": 200_000, "C": 300_000}, prices, bonds, 600_000.0, date(2024, 6, 3), portfolio="FI")
    tb = target_book(p, strategy="value", record_id="r1")
    assert tuple(tb.columns) == TARGET_BOOK_COLUMNS and set(tb["symbol"]) == {"A", "B", "C"}
    row = tb.set_index("symbol")
    assert row.at["C", "target_face"] == 0 and row.at["C", "trade_face"] == -300_000 and row.at["B", "security_id"] == bonds["B"].identifiers.cusip
    path = write_target_book(tb, tmp_path / "tb.csv")
    back = read_target_book(path)
    assert np.allclose(back["target_face"], tb["target_face"]) and list(back.columns) == list(TARGET_BOOK_COLUMNS)
    path.write_text(path.read_text().replace("FI", "XX"))
    with pytest.raises(ValueError, match="hash"):
        read_target_book(path)
    st = _value(schedule="BME")
    d = nfi.decide_at(st, syn, "2016-03-31")
    db = target_book(d, portfolio="FI")
    assert db["target_face"].isna().all() and db["target_weight"].sum() == pytest.approx(d.weights.sum())
    model = pd.Series({"A": 100.0, "B": 50.0, "C": 0.0})
    actual = pd.DataFrame({"symbol": ["A", "B", "B", "C", "D"], "face": [100.0, 40.0, 5.0, 10.0, 0.0], "kind": ["POSITION", "POSITION", "ADJUSTMENT", "POSITION", "POSITION"]})
    r = reconcile_book(model, actual).set_index("symbol")
    assert r.at["A", "status"] == "MATCH" and r.at["B", "status"] == "BREAK" and r.at["B", "actual"] == 40.0 and r.at["C", "status"] == "BLOCK"
    assert r.attrs["blocked"] and r.at["B", "action"] == "rebase_model"
    idx = pd.bdate_range("2024-01-01", periods=50)
    ext = pd.Series(np.random.default_rng(1).normal(0, 0.003, 50), index=idx)
    stats = reconcile_returns(ext + 0.00005, ext)
    assert stats["n"] == 50 and stats["mean_bp"] == pytest.approx(0.5) and stats["share_beyond_tolerance"] == 0.0
    store = FileBookStore(tmp_path / "store")
    assert isinstance(store, BookStore)
    store.write_positions("FI", "2024-06-03", pd.Series({"A": 200_000.0, "C": 300_000.0}), cash=600_000.0)
    assert store.positions("FI", "2024-06-03")["C"] == 300_000 and store.cash("FI", "2024-06-03") == 600_000
    ref = store.save_target(tb)
    assert ref == "FI/2024-06-03/target_book"
    with pytest.raises(FileExistsError):
        store.save_target(tb)
    cfg = tmp_path / "universe.sql"
    cfg.write_text("select * from bonds")
    sess = nfi.Session(tmp_path / "s", source=syn)
    a = sess.run(_value(), start="2015-01-02", end="2015-12-31")
    b = sess.run(_value(), start="2015-01-02", end="2016-03-31")
    links = {"portfolio_id": "PP-7", "port_id": "U12345-7", "owner": "ashley", "attachments": [attach(cfg)]}
    ra = sess.promote(a, "EXPLORATORY", links=links)
    rb = sess.promote(b, "EXPLORATORY", links={**links, "config_version": "2"})
    assert ra.version == 1 and rb.version == 2 and ra.links["attachments"][0]["name"] == "universe.sql" and "Strategy(" in ra.config
    reg = RecordRegistry(tmp_path / "s" / "records")
    assert [x["version"] for x in reg.find(port_id="U12345-7")] == [1, 2] and len(reg.find(name=a.name, version=2)) == 1 and reg.verify_chain()
    with pytest.raises(ValueError, match="unknown record link"):
        sess.promote(b, "EXPLORATORY", links={"port": "x"})
