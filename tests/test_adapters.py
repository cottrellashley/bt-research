"""Legacy-framework adapters (gap analysis §5): components on synthetic data, and the parity
harness that must confirm them against each framework's real outputs."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import nautilus_fi as nfi
from nautilus_fi.kernel.engine import _cross, prepare, run


@pytest.fixture(scope="module")
def syn():
    return nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2016-06-30", seed=7)


def _value(**kw):
    base = {"universe": nfi.Index("SYNTH_IG"), "signal": nfi.Field("signal.value"), "portfolio": nfi.Quantile(5), "benchmark": nfi.Index("SYNTH_IG"), "missing": nfi.ZeroFill()}
    base.update(kw)
    return nfi.Strategy("value", **base)


# ====================================================================================== registry and parity
def test_registry_lists_every_framework_unconfirmed():
    from nautilus_fi.adapters import ADAPTERS, status

    rows = {r["adapter"]: r for r in status()}
    assert set(rows) == set(ADAPTERS) and rows["systematicpm"]["status"] == "not captured"
    assert all(r["status"] == "unconfirmed" for k, r in rows.items() if k != "systematicpm")
    assert [r["adapter"] for r in status()][:6] == ["fi_lib_backtesting", "erin", "mri", "lta_ucits", "fi_lib_research", "assetiq"]
    assert rows["erin"]["parity"] == ("147HYC", "147IGC", "147SHY", "JTJSHY") and "ERIN-05" in rows["erin"]["requirements"]


def test_parity_harness_passes_matches_fails_drift_and_never_confirms(syn, tmp_path):
    from nautilus_fi.adapters import (
        ParityCase,
        ParityFailure,
        Tolerance,
        compare,
        load_golden,
        profile_confirmation,
        run_parity,
    )
    from nautilus_fi.domain.conventions import PROFILES

    st = _value()
    res = run(st, syn, "2015-03-01", "2015-12-31")
    golden = pd.DataFrame({"date": res.returns.index, "net": res.returns.to_numpy()})
    golden.to_csv(tmp_path / "golden.csv", index=False)
    g, sha = load_golden(tmp_path / "golden.csv")
    case = ParityCase("self", "erin", "golden.csv", (Tolerance("returns", bp_per_year=1.0, min_corr=0.9999),), ("2015-03-01", "2015-12-31"), build=lambda: st)
    assert compare(res, g, case, sha).passed
    stats = ParityCase("self", "erin", "golden.json", (Tolerance("turnover", abs=1e-9), Tolerance("sharpe", abs=1e-9)))
    assert compare(res, {"turnover": res.stats()["turnover"], "sharpe": res.stats()["sharpe"]}, stats, sha).passed
    assert not compare(res, {"turnover": res.stats()["turnover"] + 0.1}, stats, sha).passed                # and a missing statistic fails
    drift = compare(res, g + 0.0002, case, sha)                                  # 5bp a day off: fails the 1bp/yr band
    assert not drift.passed and "returns: annualised difference (bp)" in set(drift.failures()["metric"])
    rerun = run_parity(case, syn, g, sha)
    assert rerun.passed and "PASSED" in rerun.to_markdown()
    with pytest.raises(ParityFailure, match="failed"):
        profile_confirmation("erin_beta_plus", [drift])
    with pytest.raises(ParityFailure, match="missing"):
        profile_confirmation("erin_beta_plus", [rerun], required=["147HYC"])
    evidence = profile_confirmation("erin_beta_plus", [rerun])
    assert evidence["cases"][0]["golden_sha256"] == sha and not PROFILES["erin_beta_plus"].confirmed     # evidence only: the flag is a reviewed change


# ====================================================================================== Erin Beta+
def test_erin_orphan_refill_end_to_end_through_the_kernel():
    from nautilus_fi.adapters import erin

    days = pd.bdate_range("2024-01-01", "2024-02-29")
    ids = [f"B{i}" for i in range(10)]
    rng = np.random.default_rng(3)
    raw = pd.DataFrame(rng.normal(0.0002, 0.002, (len(days), 10)), index=days, columns=ids)
    bad, lost = pd.Timestamp("2024-01-19"), pd.Timestamp("2024-02-09")          # two Fridays the vendor published as zeros
    raw.loc[bad] = 0.0
    raw.loc[lost] = 0.0
    raw.loc[bad, "B9"] = 0.001                                                      # 90% zero: still bad
    orphan = pd.Timestamp("2024-01-20")                                             # Saturday: the late file for the 19th
    late = pd.DataFrame([rng.normal(0.0, 0.002, 10)], index=[orphan], columns=ids)
    data = pd.concat([raw, late]).sort_index()
    refill, report = erin.orphan_refill(data, days)
    assert report.set_index("bad_date").loc[bad, "orphan_date"] == orphan and not report.set_index("bad_date").loc[lost, "resolved"]
    assert np.allclose(refill.loc[bad], late.loc[orphan]) and refill.drop(bad).isna().all().all()
    src = nfi.FrameSource({"tr": raw, "tr_orphan": refill.reindex(days)}, name="erin_test")
    st = nfi.Strategy("ew", universe=nfi.All(), portfolio=nfi.EqualWeight(), schedule="BMS", execution=nfi.NextClose(1, nfi.NoCost())).under("erin_beta_plus", refill="tr_orphan")
    from nautilus_fi.domain.conventions import MissingReturnsError

    with pytest.raises(MissingReturnsError, match="no .tr_orphan. value for any live name"):
        run(st, src, "2024-01-02", "2024-02-29")                                    # the unresolved bad date has nothing to refill from
    res = run(st, src, "2024-01-02", "2024-02-06")
    assert res.missing["refilled_bad_dates"] == 10 and abs(res.gross.loc[bad] - float(late.loc[orphan].mean())) < 1e-6


def test_erin_composite_score_rules():
    from nautilus_fi.adapters import erin
    from nautilus_fi.kernel.data import DataQualityError

    d = pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-15", "2024-01-16"])
    a = pd.DataFrame({"X": [60, np.nan, np.nan, 70.0], "Y": [40, 50, np.nan, np.nan], "Z": [np.nan, np.nan, np.nan, np.nan]}, index=d)
    b = pd.DataFrame({"X": [80, np.nan, np.nan, 90.0], "Y": [np.nan, 30, np.nan, np.nan], "Z": [np.nan, np.nan, np.nan, np.nan]}, index=d)
    c = pd.DataFrame({"X": [70, 20, np.nan, np.nan], "Y": [20, np.nan, 10.0, np.nan], "Z": [np.nan, np.nan, np.nan, np.nan]}, index=d)
    s = erin.composite_score_2({"a": a, "b": b, "c": c})
    assert s.at[d[0], "X"] == pytest.approx(70.0) and s.at[d[0], "Y"] == pytest.approx(30.0)        # all, then one null: averaged
    assert s.at[d[1], "X"] == pytest.approx(70.0) and s.at[d[1], "Y"] == pytest.approx(40.0)   # two nulls: carried from 3 days before
    assert s.at[d[2], "Y"] == 19.0                                                  # two nulls, last valid 10 days before: 19
    assert s.at[d[2], "X"] == 50.0 and s.at[d[3], "Y"] == 50.0 and s.at[d[0], "Z"] == 50.0      # no row: 50
    assert s.at[d[3], "X"] == pytest.approx(80.0)
    w = erin.composite_score_2({"a": a, "b": b, "c": c}, weights={"a": 3.0, "b": 1.0, "c": 0.0})
    assert w.at[d[0], "X"] == pytest.approx((3 * 60 + 80) / 4)
    solo = erin.composite_score_2({"a": a, "b": b, "c": c}, standalone="c")
    assert solo.at[d[1], "X"] == 20.0 and solo.at[d[3], "X"] == 19.0                 # standalone null, last valid 11 days ago: 19
    with pytest.raises(DataQualityError, match="0-100"):
        erin.composite_score_2({"a": a / 100.0 * 1000})


def test_erin_mri_mapping_staleness_overrides_and_dated_dts_band(syn):
    from nautilus_fi.adapters import erin

    sc = pd.Series([-3.0, -1.0, 0.0, 1.0, 5.0], index=pd.to_datetime(["2016-01-04", "2016-02-01", "2016-03-01", "2016-04-01", "2016-05-02"]))
    knots = [(-2.0, 0.8), (0.0, 1.0), (2.0, 1.2)]
    r = erin.mri_ratio(sc, sc.index, knots)
    assert list(np.round(r, 6)) == [0.8, 0.9, 1.0, 1.1, 1.2]                         # clamped at both ends
    with pytest.raises(erin.StaleInputError, match="days old"):
        erin.mri_ratio(sc, ["2016-01-20"], knots)
    o = erin.mri_ratio(sc, sc.index, knots, overrides={"2016-03-01": 1.05, "2016-05-01": None})
    assert list(np.round(o, 6)) == [0.8, 0.9, 1.05, 1.05, 1.2]
    month_starts = pd.bdate_range("2015-01-01", "2016-06-30", freq="BMS")
    ratio = pd.Series(np.where(np.arange(len(month_starts)) % 2 == 0, 0.9, 1.1), index=month_starts)
    days = syn.data.panel.dates
    series = ratio.reindex(days, method="ffill").bfill()                            # changes at month starts: a month-end decision sees its month's ratio
    src = nfi.CompositeSource(syn, nfi.FrameSource({"tr": pd.DataFrame(index=days)}, series={"erin.mri_ratio": series}, name="mri"))
    st = _value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05, *erin.mri_dts_band(band=0.02)), execution=nfi.SameClose())
    res = run(st, src, "2015-03-01", "2016-03-31")
    ex = res.exposures
    got = (ex["dts"] / (ex["dts"] - ex["active_dts"])).reindex(pd.to_datetime(res.trades["trade_date"]))
    want = series.reindex(got.index)
    assert set(np.round(want, 2)) == {0.9, 1.1} and (got - want).abs().max() < 0.02 + 0.01


def test_erin_filter_dsl_actions_audit_and_analyst_scores(syn):
    from nautilus_fi.adapters import erin

    e = erin.parse_filter("rating_score > 7 AND NOT (oas_bp < @bench.oas_bp OR oasd >= @universe.oasd.p90)", nfi.Index("SYNTH_IG"))
    assert e.kind == "bool" and "oas_bp" in e.names() and "oasd" in e.names()
    with pytest.raises(ValueError):
        erin.parse_filter("rating_score >")
    with pytest.raises(ValueError, match="percentile"):
        erin.parse_filter("oasd > @universe.oasd")
    fs = erin.filters(["SELL: rating_score > 7", ("EXCLUDE", "oasd > @universe.oasd.p90"), "TICKER_UNDERWEIGHT: oas_bp > @bench.oas_bp"], nfi.Index("SYNTH_IG"))
    assert [r.action for r in fs.rules] == ["SELL", "EXCLUDE", "TICKER_UNDERWEIGHT"] and len(fs.constraints) == 2
    st = fs.apply(_value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05), execution=nfi.NextClose(1)))
    res = run(st, syn, "2015-03-01", "2015-08-31")
    ctx = prepare([st], syn).ctx_for(st.resolved())
    rating = np.asarray(nfi.Field("rating_score").evaluate(ctx), dtype=float)
    for dec, trade in zip(res.trades["decision_date"], res.trades["trade_date"], strict=True):
        names = res.weights.loc[res.weights["date"] == trade, "instrument_id"]
        assert np.nanmax(rating[ctx.panel.dates.get_loc(dec), ctx.panel.ids.get_indexer(names)]) <= 7 + 1e-9   # SELL names are not held
    with pytest.raises(Exception, match="optimiser constraints"):
        fs.apply(_value(portfolio=nfi.TopN(30)))
    audit = erin.filter_audit(fs, syn, "2015-03-01", "2015-06-30", universe=nfi.Index("SYNTH_IG"))
    assert set(audit["rule"]) == {"sell_1", "exclude_2", "ticker_underweight_3"} and (audit.groupby("rule")["n_hit"].sum() > 0).all()
    recs = pd.DataFrame({"date": ["2024-01-02", "2024-01-10", "2024-01-03"], "instrument_id": ["A", "A", "B"], "recommendation": ["Buy", "strong sell", 1]})
    sc = erin.analyst_scores(recs, pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-20"]), ["A", "B"])
    assert sc.at[pd.Timestamp("2024-01-05"), "A"] == 1 and sc.at[pd.Timestamp("2024-01-12"), "A"] == -2 and np.isnan(sc.at[pd.Timestamp("2024-01-20"), "B"])
    with pytest.raises(Exception, match="unknown recommendation"):
        erin.analyst_scores(pd.DataFrame({"date": ["2024-01-02"], "instrument_id": ["A"], "recommendation": ["MAYBE"]}), ["2024-01-05"])


def test_erin_lp_schedule_costs_and_gates(syn):
    from nautilus_fi.adapters import erin

    lp = erin.lp_construction(groups=("sector",), group_penalty=5.0, sparsity=(20, 30), buy_turnover=0.3)
    st = _value(portfolio=lp, execution=nfi.NextClose(1, erin.costs()), schedule="BMS")
    res = run(st, syn, "2015-03-01", "2015-09-30")
    n = res.weights.groupby("date")["instrument_id"].count()
    assert n.max() <= 30 and len(res.trades) >= 6
    audit = res.diagnostics["cost_audit"]
    assert (audit["trade"] > 1e-4).all() and len(audit)                                 # buys only, above 1bp of NAV
    ctx = prepare([st], syn).ctx_for(st.resolved())
    t = int(ctx.panel.dates.searchsorted(pd.Timestamp("2016-01-29")))
    unit = erin.costs().model.unit(ctx, t, None, 1e8) * 1e4
    d, px, r = (np.asarray(ctx.field(f)[t], dtype=float) for f in ("oasd", "price_clean", "rating_score"))
    ok = np.isfinite(d) & np.isfinite(px) & np.isfinite(r) & (d > 0)
    assert np.allclose((unit / (d * px / 100.0))[ok], np.where(r[ok] > 10.5, 10.0, 3.0))   # IG 3bp / HY 10bp x OASD x price
    assert erin.costs().total(ctx, t, -np.full(ctx.N, 0.01), 1e8) == 0.0                     # sales are free
    assert erin.schedule().dates([d.date() for d in pd.bdate_range("2024-01-01", "2024-03-31")])
    ok, why = erin.eligible(res)
    assert isinstance(ok, bool) and isinstance(why, list)
    tuned = erin.apply_tuning(_value(portfolio=nfi.TopN(40)), {"portfolio.n": 25, "execution.lag": 2})
    assert tuned.portfolio.n == 25 and tuned.execution.lag == 2


# ====================================================================================== fi-lib-backtesting
def test_fi_lib_backtesting_constraint_grammar_and_settings(syn):
    from nautilus_fi.adapters import fi_lib_backtesting as fl

    st = _value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05))
    cs = _cs(syn, st, ("oad",))
    b = float((cs.benchmark * cs.data["oad"]).sum())
    oad = lambda w: float((w.reindex(cs.ids).fillna(0.0) * cs.data["oad"]).sum())
    for kind, lo, hi, want in (("absolute", b - 0.3, b - 0.2, (b - 0.3, b - 0.2)), ("offset", -0.1, 0.1, (b - 0.1, b + 0.1)), ("ratio", 0.95, 1.05, (0.95 * b, 1.05 * b))):
        w, _ = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05, *fl.constraint("oad", kind, lo, hi)).weights(cs)
        assert want[0] - 1e-6 <= oad(w) <= want[1] + 1e-6, kind
    g = fl.group_limit("sector", -0.02, 0.02, min_benchmark_weight=0.05)
    assert getattr(g, "min_benchmark_weight", None) == 0.05
    with pytest.raises(ValueError, match="group limits"):
        fl.constraint("oad", "offset", -0.1, 0.1, min_benchmark_weight=0.05)
    assert fl.relaxation(push=True).on_infeasible == "push" and fl.relaxation().relax_tiers == (1.5, 2.0)
    assert fl.score_drop(0.2).measure == "score_drop" and type(fl.score_drop(0.2, fixed="BQE")).__name__ == "WithInterim"
    assert fl.benchmark("SYNTH_IG").drift.value == "DRIFT"


# ====================================================================================== MRI
def test_mri_price_adjust_cost_v2_rule_and_decimal_guard():
    from nautilus_fi.adapters import mri
    from nautilus_fi.kernel.data import DataQualityError

    c = mri.cost("US HY")
    assert c.bp == 20.0 and c.total(None, 0, np.array([1.0]), 1.0) == pytest.approx(0.002 / 1.002) and c.total(None, 0, np.array([-1.0]), 1.0) == pytest.approx(0.002)
    a = mri.PriceAdjustCost(20.0, additive=True)
    assert a.total(None, 0, np.array([1.0, -1.0]), 1.0) == pytest.approx(np.log1p(0.002) - np.log1p(-0.002))
    with pytest.raises(KeyError):
        mri.cost("JP IG")
    days = pd.bdate_range("2024-01-01", "2024-02-29")
    sa = pd.Series(np.nan, index=days)
    sa.iloc[0] = 2.0                                                                  # through its threshold once, then silent
    sb = pd.Series(0.0, index=days)
    sb.iloc[30] = 5.0
    src = nfi.FrameSource({"px": pd.DataFrame({"IDX": 100.0}, index=days)}, series={"a": sa, "b": sb}, name="mri")
    sig = mri.v2_signal("a", "b", 1.0, 3.0)
    prep = prepare([nfi.Strategy("t", universe=nfi.All(), signal=sig, portfolio=nfi.EqualWeight(), returns="px")], src)
    v = np.asarray(sig.evaluate(prep.ctx), dtype=float)[:, 0]
    on = pd.Series(v, index=prep.ctx.panel.dates)
    assert on.iloc[0] == 1.0 and on.iloc[3] == 1.0 and on.iloc[5] == 0.0 and on.iloc[30] == 1.0   # carried 5 calendar days, not more
    with pytest.raises(DataQualityError, match="percent"):
        mri.decimal_guard(pd.Series([4.5, 5.0, 3.9]), "ust_10y")
    assert mri.decimal_guard(pd.Series([0.045, 0.05]), "ust_10y") is not None
    assert mri.additive_metrics().pnl == "ADDITIVE"


# ====================================================================================== fi-lib-research, EUR IG
def test_fi_lib_research_engine_inputs_and_alignment(syn):
    from nautilus_fi.adapters import fi_lib_research as fr

    s = fr.engine(_value(), cost_per_change=100.0)
    assert s.hold == "constant" and s.metrics.pnl == "ADDITIVE" and s.profile == "fi_lib_research" and "PerChange" in repr(s.execution.cost)
    assert fr.input_field("idx", "level").form == "price" and fr.input_field("r", "log_return").form == "log"
    with pytest.raises(ValueError, match="input kind"):
        fr.input_field("x", "vibes")
    a = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    b = pd.Series([0.001, 0.002], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    with pytest.raises(ValueError, match="disagree on dates"):
        fr.align(a, b, "strict")
    res = run(s, syn, "2015-03-01", "2015-08-31")
    assert res.stats()["annual_return"] == pytest.approx(res.returns.sum() / ((res.returns.index[-1] - res.returns.index[0]).days / 365.25), rel=0.01)


def test_eur_ig_tilt_calibration_ensemble_and_blend():
    from nautilus_fi.adapters import eur_ig_lgbm as eu

    assert list(eu.symmetric_tilt(np.array([-1.0, -0.33, 0.0, 0.42, 2.0]), k=0.5, clip=0.4, step=0.1)) == [0.6, 0.8, 1.0, 1.2, 1.4]
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    z = pd.Series(rng.normal(size=600), index=idx)
    y = pd.Series((rng.uniform(size=600) < 1 / (1 + np.exp(-1.5 * z))).astype(float), index=idx)
    p = eu.oof_calibrate(z, y, folds=5)
    assert p.between(0, 1).all() and abs(p.mean() - 0.5) < 0.05 and p.corr(z, method="spearman") > 0.95
    assert eu.centred_score(0.75) == 0.5
    ens = eu.seed_ensemble(lambda seed: np.full(3, float(seed)), seeds=range(10))
    assert np.allclose(ens, 4.5)
    preds = pd.DataFrame({"m1": z + rng.normal(0, 0.1, 600), "m2": rng.normal(size=600)}, index=idx)
    w = eu.fit_blend(preds, z, idx[399])
    preds2 = preds.copy()
    preds2.iloc[400:] = 0.0
    assert w["m1"] > 0.9 and w.sum() == pytest.approx(1.0) and w.equals(eu.fit_blend(preds2, z, idx[399]))   # the test rows are never read


# ====================================================================================== composite, assetiq
def test_composite_statistics_and_investible_filter(syn):
    from nautilus_fi.adapters import composite

    x = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 300))
    t = composite.nw_tstats(x)
    assert set(t) == {13, 26} and all(np.isfinite(v) for v in t.values())
    point, lo, hi = composite.bootstrap(x, n_boot=300)
    assert lo < point < hi
    oas = pd.DataFrame(syn.data.panel["oas_bp"], index=syn.data.panel.dates, columns=syn.data.panel.ids)
    cap = float(np.nanpercentile(oas.loc["2015-03-02"], 60))
    st = _value(universe=nfi.Index("SYNTH_IG") & composite.investible(cap), portfolio=nfi.TopN(15), execution=nfi.SameClose(nfi.NoCost()))
    res = run(st, syn, "2015-03-01", "2015-08-31")
    ctx = prepare([st], syn).ctx_for(st.resolved())
    known = np.asarray(nfi.Field("oas_bp").evaluate(ctx), dtype=float)
    for d in pd.to_datetime(res.trades["decision_date"]):
        names = res.weights.loc[res.weights["date"] == d, "instrument_id"]
        assert len(names) and np.nanmax(known[ctx.panel.dates.get_loc(d), ctx.panel.ids.get_indexer(names)]) <= cap      # nothing above the cap


def _cs(syn, st, extra):
    prep = prepare([st], syn, extra_fields=tuple(extra))
    ctx = prep.ctx
    t = int(ctx.panel.dates.searchsorted(pd.Timestamp("2016-01-29")))
    U = np.asarray(st.universe.evaluate(ctx), dtype=bool)
    Wk = st.benchmark.known_weights(ctx)
    S = np.asarray(st.signal.evaluate(ctx.within(U, "u")), dtype=float)
    return _cross(t, st.resolved(), ctx, S[t], U[t], Wk[t], np.zeros(ctx.N), 1e8, extra_names=tuple(extra))


def test_assetiq_factor_space_bands_copula_and_tev_band(syn):
    from nautilus_fi.adapters import assetiq

    w = pd.Series({"a": 0.5, "b": 0.5})
    dts = pd.Series({"a": 2.0, "b": 4.0})
    assert assetiq.from_factor(assetiq.to_factor(w, dts), dts).equals(w)
    t = assetiq.tilt_bands(pd.Series([-2.0, -0.5, 0.5, 2.0, np.nan]), [(-np.inf, -1.0, -0.02), (1.0, np.inf, 0.02)])
    assert list(t.iloc[:4]) == [-0.02, 0.0, 0.0, 0.02] and np.isnan(t.iloc[4])
    rng = np.random.default_rng(2)
    base = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=500)
    sc = assetiq.kde_copula_scenarios(pd.DataFrame(base, columns=["x", "y"]), n=3000)
    assert sc.shape == (3000, 2) and abs(sc.corr(method="spearman").iloc[0, 1] - 0.79) < 0.06
    model = nfi.PCARisk(120, "xr", 3)
    st = _value(portfolio=nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05))
    cs = _cs(syn, st, ("xr",))
    base_opt = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05)
    free, _ = base_opt.weights(cs)
    te_free = nfi.TEV(model).sigma(cs, free.reindex(cs.ids).fillna(0.0).to_numpy())
    band = assetiq.tev_band(base_opt, model, floor=0.4 * te_free, budget=0.8 * te_free, points=3)
    w2, _ = band.weights(cs)
    te2 = nfi.TEV(model).sigma(cs, w2.reindex(cs.ids).fillna(0.0).to_numpy())
    assert 0.4 * te_free * (1 - 2e-3) <= te2 <= 0.8 * te_free * (1 + 2e-3) and len(band.levels) == 3
    with pytest.raises(ValueError, match="floor"):
        assetiq.tev_band(base_opt, model, floor=0.05, budget=0.01)


# ====================================================================================== lta-ucits
def test_lta_ucits_limits_screens_flows_and_paired_t(syn):
    from nautilus_fi.adapters import lta_ucits as lu

    assert "relcap" == lu.issuer_limit().measure.kind and lu.banks_at_index(nfi.Field("sector") == "FIN").hi == 0.0
    uni, nobuy = lu.esg_screen()
    assert uni.kind == "bool" and isinstance(nobuy, nfi.Constraint) and "esg_state" in uni.names()
    opt = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.05)
    assert len(lu.flow_construction(opt, +1e6).constraints()) == 2 and len(lu.flow_construction(opt, 0.0).constraints()) == 1
    days = pd.bdate_range("2020-01-01", "2022-12-30")
    frames = {n: pd.DataFrame({"X": 1.0, "Y": 2.0}, index=days) for n in ("tr", "a", "b")}
    src = nfi.FrameSource(frames, info=[nfi.FieldInfo("tr"), nfi.FieldInfo("a", start="2020-01-01"), nfi.FieldInfo("b", start="2021-06-01")], name="starts")
    two = nfi.Strategy("ab", universe=nfi.All(), signal=nfi.Field("a") + nfi.Field("b"))
    assert lu.common_start(two, src) == pd.Timestamp("2021-06-01") and lu.common_start(nfi.Strategy("a", universe=nfi.All(), signal=nfi.Field("a")), src) == pd.Timestamp("2020-01-01")
    a = pd.Series(np.random.default_rng(5).normal(0.0005, 0.002, 500), index=pd.bdate_range("2020-01-01", periods=500))
    pt = lu.paired_t(a, a - 0.0004)
    assert pt["mean_difference"] == pytest.approx(0.0004) and pt["t_stat"] > 10 and pt["share_positive"] == 1.0
    st = _value(portfolio=lu.two_stage(nfi.Alpha(), nfi.Weight() <= 0.05, lu.issuer_limit(0.06, 3.0)), execution=nfi.NextClose(1))
    res = run(st, syn, "2015-03-01", "2015-06-30")
    iss = syn.data.attributes["issuer"]
    by_issuer = res.weights.assign(issuer=res.weights["instrument_id"].map(iss)).groupby(["date", "issuer"])["weight"].sum()
    assert res.diagnostics["relaxed_rebalances"] == 0 and by_issuer.max() <= 0.06 + 1e-6      # (4% x 3 is infeasible on 30 synthetic issuers)


# ====================================================================================== credit-pairs, globalrates
def test_credit_pairs_adf_fdr_sizing_and_imm():
    from nautilus_fi.adapters import credit_pairs as cp

    assert cp._mackinnon_p(-2.86) == pytest.approx(0.05, abs=0.003) and cp._mackinnon_p(-3.43) == pytest.approx(0.01, abs=0.002)
    rng = np.random.default_rng(11)
    walk = pd.Series(np.cumsum(rng.normal(size=500)))
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.5 * ar[i - 1] + rng.normal()
    assert cp.adf(walk)["pvalue"] > 0.1 and cp.adf(pd.Series(ar))["pvalue"] < 0.01
    fdr = cp.bh_fdr(pd.Series([0.001, 0.008, 0.039, 0.041, 0.6], index=list("abcde")), q=0.05)
    assert list(fdr["reject"]) == [True, True, False, False, False]
    assert cp.dv01_neutral(10e6, 4.5e-4, 9e-4) == -5e6 and cp.etf_shares(1_000_000, 101.3, lot=10) == 9870
    assert cp.imm_dates("2024-01-01", "2024-12-31") == [date(2024, 3, 20), date(2024, 6, 20), date(2024, 9, 20), date(2024, 12, 20)]


def test_globalrates_country_calendars():
    from nautilus_fi.adapters import globalrates as gr

    de, uk, us = gr.country_calendar("DE"), gr.country_calendar("GB"), gr.country_calendar("US")
    assert de.name == "TARGET" and not de.is_business_day(date(2024, 5, 1)) and de.is_business_day(date(2024, 5, 27))
    assert not uk.is_business_day(date(2022, 6, 3)) and not uk.is_business_day(date(2021, 12, 28)) and uk.is_business_day(date(2024, 5, 1))
    assert not us.is_business_day(date(2024, 7, 4))
    with pytest.raises(KeyError):
        gr.country_calendar("JP")
