"""The research kernel: data seam, expressions, benchmarks, mechanics, construction, strategies,
the vectorised engine, sessions, the agent API and Implement fidelity."""

from __future__ import annotations

import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest

import nautilus_fi as nfi
from nautilus_fi.kernel.data import DataRequest
from nautilus_fi.kernel.expr import EvalContext


@pytest.fixture(scope="module")
def src():
    return nfi.SyntheticSource(n_bonds=300, start="2015-01-02", end="2016-12-30", seed=11)


@pytest.fixture(scope="module")
def ctx(src):
    p = src.load(DataRequest(("oas_bp", "oasd", "tr", "alive", "rating_score", "signal.value", "index_weight", "half_spread_bp", "amt_out"), None, None))
    return EvalContext(p, src.attributes(p.ids), src.series(["cash", "index.tr"]), src, src.catalog())


def _frame(ctx, name):
    return pd.DataFrame(np.asarray(ctx.panel[name], dtype=float), index=ctx.panel.dates, columns=ctx.panel.ids)


# ====================================================================================== data seam
def test_catalog_search_and_suggestions(src):
    cat = src.catalog()
    assert "oas_bp" in cat and cat["oas_bp"].entity == "asset" and cat["sector"].entity == "attribute" and cat["cash"].entity == "series"
    assert any(f.name == "oasd" for f in cat.search("spread duration"))
    with pytest.raises(nfi.UnknownFieldError) as e:
        cat["oas"]
    assert "oas_bp" in str(e.value)
    assert "SYNTH_IG" in cat.indices


def test_frame_callable_composite_mapped_sources(src):
    p = src.load(DataRequest(("oas_bp", "oasd"), "2015-06-01", "2015-08-31"))
    wide = {"oas": pd.DataFrame(p["oas_bp"], index=p.dates, columns=p.ids)}
    fs = nfi.FrameSource(wide, attributes=src.attributes(p.ids)[["sector"]], lags={"oas": 1})
    assert fs.catalog()["oas"].lag == 1
    sub = fs.load(DataRequest(("oas",), "2015-07-01", "2015-07-31", ids=tuple(p.ids[:5])))
    assert sub["oas"].shape == (len(sub.dates), 5) and sub.dates[0] >= pd.Timestamp("2015-07-01")
    calls = []

    def fetch(field, start, end, ids):
        calls.append(field)
        long = p.to_long(["oasd"]).rename(columns={"oasd": "value"})
        return long[(long["date"] >= start) & (long["date"] <= end)]

    cs = nfi.CallableSource(fetch, {"dur": "spread duration"}, name="warehouse")
    comp = nfi.CompositeSource(fs, cs)
    both = comp.load(DataRequest(("oas", "dur"), "2015-06-01", "2015-08-31"))
    assert set(both.fields) == {"oas", "dur"} and calls == ["dur"]
    np.testing.assert_allclose(np.nan_to_num(both["dur"]), np.nan_to_num(p["oasd"]))
    mapping = {f"X{i}": k for i, k in enumerate(p.ids)}
    ms = nfi.MappedSource(fs, mapping)
    m = ms.load(DataRequest(("oas",), "2015-06-01", "2015-06-30", ids=("X0", "X1", "NOPE")))
    assert list(m.ids) == ["X0", "X1"] and "NOPE" in ms.unmapped


def test_cached_source_slices_memmaps_and_pins_snapshot(src, tmp_path):
    class Live(nfi.PanelSource):
        def version(self):
            return None                                   # a warehouse that cannot say what changed

    live = Live(src.raw_panel, src.attributes())
    c = nfi.CachedSource(live, root=tmp_path)
    a = c.load(DataRequest(("oas_bp", "tr"), "2015-01-01", "2016-12-31"))
    b = c.load(DataRequest(("oas_bp",), "2015-06-01", "2015-12-31"))
    assert c.stats["hits"] == 1 and c.stats["misses"] == 2
    np.testing.assert_array_equal(np.nan_to_num(b["oas_bp"]), np.nan_to_num(a["oas_bp"][(a.dates >= "2015-06-01") & (a.dates <= "2015-12-31")]))
    v1 = c.version()
    assert v1.startswith("snapshot:") and nfi.CachedSource(live, root=tmp_path).version() == v1
    fresh = nfi.CachedSource(live, root=tmp_path)
    d = fresh.load(DataRequest(("oas_bp",), "2015-06-01", "2015-12-31"))
    assert fresh.stats["disk_hits"] == 1 and isinstance(d["oas_bp"], np.ndarray)


def test_field_store_source_reads_as_of(tmp_path):
    from nautilus_fi.data.field_store import FieldStore

    store = FieldStore(tmp_path)
    df = pd.DataFrame({"instrument_id": ["A", "B"], "date": pd.to_datetime(["2020-01-02", "2020-01-02"]), "oas": [100.0, 200.0]})
    store.write("oas", df, knowledge_time="2020-01-02T22:00:00Z")
    store.write("oas", df.assign(oas=[110.0, 210.0]), knowledge_time="2020-01-05T22:00:00Z")
    fss = nfi.FieldStoreSource(store, lags={"oas": 1})
    first = fss.load(DataRequest(("oas",), "2020-01-01", "2020-01-10", as_of="2020-01-03"))
    latest = fss.load(DataRequest(("oas",), "2020-01-01", "2020-01-10"))
    assert first["oas"][0].tolist() == [100.0, 200.0] and latest["oas"][0].tolist() == [110.0, 210.0]
    assert fss.catalog()["oas"].lag == 1 and fss.catalog().supports_as_of


def test_sql_source_on_sqlite():
    from nautilus_fi.data.loaders import DatasetMapping, SQLLoader

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE an (d TEXT, id TEXT, oas REAL, dur REAL)")
    conn.executemany("INSERT INTO an VALUES (?,?,?,?)", [("2021-01-04", "A", 1.0, 5.0), ("2021-01-04", "B", 2.0, 6.0), ("2021-01-05", "A", 1.5, 5.1)])
    m = DatasetMapping("analytics", "an", {"date": "d", "instrument_id": "id", "oas_bp": "oas", "oasd": "dur"})
    sql = nfi.SQLSource(SQLLoader(lambda: conn), [m])
    assert {"oas_bp", "oasd"} <= set(sql.catalog().names("asset"))
    p = sql.load(DataRequest(("oas_bp", "oasd"), "2021-01-01", "2021-01-31"))
    assert p["oas_bp"].shape == (2, 2) and p["oasd"][1][0] == 5.1


# ====================================================================================== expressions
def test_cross_sectional_transforms_match_pandas(ctx):
    oas = _frame(ctx, "oas_bp")
    sec = ctx.attributes["sector"]
    z = nfi.Field("oas_bp").zscore(by="sector").evaluate(ctx)
    ref = oas.T.groupby(sec).transform(lambda x: (x - x.mean()) / x.std(ddof=0)).T
    assert np.nanmax(np.abs(z - ref.to_numpy())) < 1e-9
    r = nfi.Field("oas_bp").rank(by="sector").evaluate(ctx)
    assert np.nanmax(np.abs(r - oas.T.groupby(sec).rank(pct=True).T.to_numpy())) < 1e-12
    d = nfi.Field("oas_bp").demean(by="sector").evaluate(ctx)
    assert np.nanmax(np.abs(d - oas.T.groupby(sec).transform(lambda x: x - x.mean()).T.to_numpy())) < 1e-9
    n = nfi.Field("oas_bp").neutralise("oasd", by="sector").evaluate(ctx)
    t = 200
    ok = np.isfinite(n[t])
    assert abs(np.corrcoef(n[t][ok], ctx.panel["oasd"][t][ok])[0, 1]) < 1e-8
    top = nfi.Field("oas_bp").top(5, by="sector").evaluate(ctx)
    assert top[t].sum() == 5 * sec.nunique()
    b = nfi.Field("rating_score").bucket([4, 7, 10]).evaluate(ctx)
    assert b.categories == ["<4", "4-7", "7-10", ">=10"]


def test_time_series_transforms_match_pandas(ctx):
    oas = _frame(ctx, "oas_bp")
    m = nfi.Field("oas_bp").rolling(21).mean().evaluate(ctx)
    assert np.nanmax(np.abs(m - oas.rolling(21).mean().to_numpy())) < 1e-9
    s = nfi.Field("oas_bp").rolling(21, min_periods=5).std().evaluate(ctx)
    assert np.nanmax(np.abs(s - oas.rolling(21, min_periods=5).std().to_numpy())) < 1e-6
    lg = nfi.Field("oas_bp").lag(3).evaluate(ctx)
    np.testing.assert_array_equal(np.nan_to_num(lg[10]), np.nan_to_num(oas.iloc[7].to_numpy()))
    ch = nfi.Field("oas_bp").diff(5).evaluate(ctx)
    assert np.nanmax(np.abs(ch - oas.diff(5).to_numpy())) < 1e-9


def test_predicates_categories_and_nan(ctx):
    u = ((nfi.Field("rating_score") <= 10) & (nfi.Attribute("sector") != "FIN") & nfi.Field("oasd").notna()).evaluate(ctx)
    sec = ctx.attributes["sector"].to_numpy()
    assert not u[:, sec == "FIN"].any()
    assert not u[~np.isfinite(ctx.panel["oasd"])].any()
    isin = nfi.Attribute("sector").isin(["FIN", "UTL"]).evaluate(ctx)
    assert isin[0].sum() == int(np.isin(sec, ["FIN", "UTL"]).sum())


def test_lookahead_structural_and_empirical(ctx):
    assert nfi.Field("tr").lead(1).lookahead() == 1 and nfi.Field("tr").lag(1).lookahead() < 0
    assert nfi.Field("tr").lag(2).lead(1).lookahead() < 0
    with pytest.raises(nfi.LookaheadError):
        nfi.check_lookahead(nfi.Field("tr").lead(1), ctx)
    with pytest.raises(nfi.LookaheadError, match="uses later data"):
        nfi.check_lookahead(nfi.Field("oas_bp").apply(lambda x: x - np.nanmean(x, axis=0)), ctx)
    rep = nfi.check_lookahead(nfi.Field("oas_bp").apply(lambda x: np.log(x)), ctx)
    assert rep["empirical"] and all(c["rows_changed"] == 0 for c in rep["cuts"])
    cs = nfi.Field("oas_bp").cs(lambda s: s.rank(pct=True), name="pyrank")
    rows = np.arange(0, ctx.T, 25)
    np.testing.assert_allclose(np.nan_to_num(cs.evaluate(ctx.at_rows(rows))[rows]), np.nan_to_num(nfi.Field("oas_bp").rank().evaluate(ctx)[rows]))


def test_availability_lag_is_applied_and_cannot_be_undercut(src):
    p = src.load(DataRequest(("oas_bp", "tr", "alive"), "2015-01-01", "2015-12-31"))
    fs = nfi.PanelSource(p, src.attributes(p.ids), fields=[nfi.FieldInfo("oas_bp", lag=1)])
    c = EvalContext(fs.load(DataRequest(("oas_bp",), None, None)), None, None, fs, fs.catalog())
    v = nfi.Field("oas_bp").evaluate(c)
    np.testing.assert_array_equal(np.nan_to_num(v[5]), np.nan_to_num(p["oas_bp"][4]))
    with pytest.raises(nfi.LookaheadError):
        nfi.Field("oas_bp", lag=0).evaluate(c)


def test_custom_code_changes_the_fingerprint():
    a = nfi.Strategy(signal=nfi.Field("x").apply(lambda v: v * 2.0))
    b = nfi.Strategy(signal=nfi.Field("x").apply(lambda v: v * 3.0))
    assert a.fingerprint != b.fingerprint
    assert nfi.Strategy(signal=nfi.Field("x").rank()).fingerprint == nfi.Strategy(signal=nfi.Field("x").rank()).fingerprint


# ====================================================================================== benchmarks
def test_index_reproduces_published_returns_and_derived_rules(src, ctx):
    ix = nfi.Index("SYNTH_IG")
    assert np.abs(ix.returns(ctx) - src.data.index["tr"].to_numpy()).max() < 1e-12
    assert np.abs(ix.weights(ctx) - np.nan_to_num(ctx.panel["index_weight"])).max() < 1e-8
    d = ix.exclude(sector="FIN").cap(issuer=0.03)
    W = d.weights(ctx)
    fin = (ctx.attributes["sector"] == "FIN").to_numpy()
    assert W[:, fin].max() == 0.0
    rows, _ = d.recon(ctx)
    iss = ctx.attributes["issuer"]
    for t, w in list(rows.items())[:5]:
        assert pd.Series(w, index=ctx.panel.ids).groupby(iss).sum().max() <= 0.03 + 1e-9
    bl = nfi.Blend((ix, 0.5), (d, 0.5))
    assert np.isfinite(bl.returns(ctx)).all()


# ====================================================================================== mechanics
def test_cost_algebra_and_impact(ctx):
    hs = np.nan_to_num(np.asarray(ctx.panel["half_spread_bp"][50], dtype=float), nan=np.nanmedian(ctx.panel["half_spread_bp"][50])) / 1e4
    m = nfi.HalfSpread() * 2 + nfi.FixedCost(1.0)
    np.testing.assert_allclose(m.unit(ctx, 50, None, 1e8), 2 * hs + 1e-4)
    imp = nfi.SqrtImpact(10.0)
    small, big = imp.unit(ctx, 50, np.full(ctx.N, 1e-4), 1e8), imp.unit(ctx, 50, np.full(ctx.N, 1e-2), 1e8)
    assert np.nanmean(big) > np.nanmean(small) > 0


def test_execution_timing_and_conventions_profiles(src):
    base = nfi.Strategy("q", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.Quantile(5), missing=nfi.ZeroFill())
    same = base.with_(execution=nfi.SameClose(), name="same")
    s = nfi.Session(source=src)
    a, b = s.run(base), s.run(same)
    assert a.trades["trade_date"].iloc[0] > b.trades["trade_date"].iloc[0]
    mri = base.under("MRI_v1")
    assert mri.execution.lag == 0 and mri.metrics.excess.method.value == "DAILY_SUM" and mri.conventions().name == "MRI_v1" and not mri.conventions().confirmed


def test_missing_returns_fail_then_zero_fill():
    sd = nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2015-12-31", missing_rate=0.02, seed=3)
    st = nfi.Strategy("q", signal=nfi.Field("signal.value"), portfolio=nfi.TopN(30))
    from nautilus_fi.domain.conventions import MissingReturnsError
    from nautilus_fi.kernel.engine import run

    with pytest.raises(MissingReturnsError, match="ZeroFill"):
        run(st, sd)
    r = run(st.with_(missing=nfi.ZeroFill()), sd)
    assert r.missing["zero_filled"] > 0


def test_participation_spreads_trades_over_days(src):
    st = nfi.Strategy("q", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.TopN(20), missing=nfi.ZeroFill(), capital=5e9, execution=nfi.NextClose(1, participation=nfi.Participation(0.05)))
    from nautilus_fi.kernel.engine import run

    r = run(st, src)
    assert (r.trades["fill_ratio"] < 1).any() and len(r.trades) > r.diagnostics["n_decisions"]


def test_long_short_pays_borrow_and_cash_earns(src):
    from nautilus_fi.kernel.engine import run

    ls = nfi.Strategy("ls", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.Quantile(5, short=1), missing=nfi.ZeroFill(), financing=nfi.Cash("cash", short_fee_bp=100))
    r = run(ls, src)
    assert r.daily["financing_return"].sum() < 0 and r.daily["cash_return"].sum() > 0
    assert r.weights.groupby("date")["weight"].sum().abs().max() < 1e-9          # dollar neutral on every trade date (legs drift in between)
    no = run(ls.with_(financing=nfi.NoCash(short_fee_bp=0)), src)
    assert no.daily["cash_return"].abs().sum() == 0


# ====================================================================================== construction
def test_vectorised_and_per_date_paths_agree(src):
    from nautilus_fi.kernel.engine import prepare, simulate

    class PerDate(nfi.Quantile):
        path_dependent = True

    a = nfi.Strategy("vec", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").zscore(by="sector"), portfolio=nfi.Quantile(5, by="sector", weighting=nfi.ByField("amt_out")), missing=nfi.ZeroFill())
    b = a.with_(portfolio=PerDate(5, by="sector", weighting=nfi.ByField("amt_out")), name="row")
    prep = prepare([a, b], src)
    ra, rb = simulate(a, prep), simulate(b, prep)
    assert ra.diagnostics["vectorised_targets"] and not rb.diagnostics["vectorised_targets"]
    np.testing.assert_allclose(ra.returns.to_numpy(), rb.returns.to_numpy(), atol=1e-12)


def test_optimiser_api_enforces_constraints(src):
    from nautilus_fi.kernel.engine import run

    opt = nfi.Optimise(3e-4 * nfi.Alpha() - nfi.TrackingError(by=["sector"]) - nfi.Costs(), nfi.ActiveWeight("issuer") <= 0.01, nfi.ActiveWeight("sector").within(0.02), nfi.ActiveExposure("oasd").within(0.1), nfi.Turnover() <= 0.1, nfi.Weight() <= 0.03)
    st = nfi.Strategy("ip", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").zscore(by="sector"), portfolio=opt, benchmark=nfi.Index("SYNTH_IG"), missing=nfi.ZeroFill())
    r = run(st, src)
    hard = r.violations[r.violations["severity"] == "HARD"] if len(r.violations) else r.violations
    assert len(hard) == 0, hard.head()
    sec = r.groups["sector"]                      # bands hold against the benchmark the decision saw (it drifts a day before the trade)
    target = sec["weight"].loc[pd.to_datetime(r.trades["trade_date"])].to_numpy()
    seen = sec["benchmark_weight"].loc[pd.to_datetime(r.trades["decision_date"])].to_numpy()
    assert np.abs(target - seen).max() <= 0.02 + 1e-6
    assert (r.trades["HALF_SUM"].iloc[1:] <= 0.1 + 1e-6).all()
    assert r.weights.groupby("date")["weight"].max().max() <= 0.03 + 1e-9
    v = opt.replace(nfi.Turnover() <= 0.05)
    assert sum(isinstance(c.measure, nfi.Turnover) for c in v.constraints()) == 1 and "Turnover() <= 0.05" in repr(v)
    assert not any(isinstance(c.measure, nfi.Turnover) for c in v.without(nfi.Turnover).constraints())
    with pytest.raises(ValueError, match="penalty"):
        nfi.Optimise(nfi.Alpha() + nfi.Costs()).weights(_tiny_cs())


def _tiny_cs():
    from nautilus_fi.construct.cross_section import CrossSection

    ids = pd.Index(["A", "B", "C"])
    return CrossSection(pd.Timestamp("2020-01-02"), ids, pd.DataFrame({"oasd": [3.0, 5.0, 7.0], "sector": ["X", "X", "Y"], "issuer": ["a", "b", "c"]}, index=ids), pd.Series(0.0, index=ids), pd.Series([0.4, 0.3, 0.3], index=ids), pd.Series([1.0, 0.0, -1.0], index=ids), cost_bp=pd.Series(5.0, index=ids))


def test_custom_linear_constraint_and_projection():
    cs = _tiny_cs()
    w, _ = nfi.Optimise(nfi.Alpha(), nfi.Weight() <= 0.6, nfi.LinearMeasure(nfi.Field("oasd")) <= 4.0).weights(cs)
    assert float((w * cs.data["oasd"].reindex(w.index)).sum()) <= 4.0 + 1e-9
    p, _ = nfi.EqualWeight().subject_to(nfi.Weight() <= 0.5, nfi.GroupWeight("sector", "Y") <= 0.2).weights(cs)
    assert abs(p.sum() - 1.0) < 1e-9 and p.max() <= 0.5 + 1e-9 and p.get("C", 0.0) <= 0.2 + 1e-9
    capped, _ = nfi.TopN(3, weighting=nfi.ByField("oasd")).cap(name=0.4).weights(cs)
    assert capped.max() <= 0.4 + 1e-9 and abs(capped.sum() - 1) < 1e-9


def test_hedge_overlay_neutralises_duration(src):
    from nautilus_fi.kernel.engine import run

    st = nfi.Strategy("hedged", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.TopN(40), missing=nfi.ZeroFill(), overlay=nfi.Hedge(["TY"], fields=["oad"], targets={"oad": 0.0}))
    r = run(st, src)
    ov = r.diagnostics["overlays"]
    assert (ov["notional"] < 0).all() and r.daily["overlay_return"].abs().sum() > 0


# ====================================================================================== strategy
def test_strategy_identity_variants_and_description(src):
    s = nfi.Strategy("value", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").zscore(by="sector"), portfolio=nfi.TopN(50), benchmark=nfi.Index("SYNTH_IG"))
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.name = "x"
    t = s.set("portfolio.n", 25).set("execution.lag", 2)
    assert t.portfolio.n == 25 and t.execution.lag == 2 and s.portfolio.n == 50
    d = t.diff(s)
    assert set(d.components()) == {"portfolio", "execution"} and "TopN(25)" in str(d)
    assert t.fingerprint != s.fingerprint and s.with_(name="renamed").fingerprint == s.fingerprint
    text = s.describe()
    for word in ("universe", "signal", "TopN(50)", "NextClose", "Cash('cash')", "fingerprint"):
        assert word in text
    issues = nfi.Strategy(signal=nfi.Field("oas")).validate(src)
    assert any("oas_bp" in i for i in issues)
    req = s.requirements(src.catalog())
    assert {"signal.value", "tr"} <= req.fields and "sector" in req.attributes and "SYNTH_IG" in req.indices


def test_from_spec_matches_the_legacy_paper_engine_exactly(src):
    from nautilus_fi.benchmarks import ConstituentBenchmark
    from nautilus_fi.construct import ConstructionSpec
    from nautilus_fi.construct import Quantile as LegacyQuantile
    from nautilus_fi.domain import Conventions, MissingReturnPolicy
    from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
    from nautilus_fi.kernel.engine import run
    from nautilus_fi.paper import PaperData, PaperEngine, PaperSettings

    d = src.data
    spec = ResearchSpec("legacy", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(LegacyQuantile(5)), Conventions().with_(missing_returns=MissingReturnPolicy.ZERO), calendar="BME")
    legacy = PaperEngine(spec, PaperData(d.panel, d.attributes, d.cash, ConstituentBenchmark("IDX", d.membership)), PaperSettings(start="2015-03-02")).run()
    kernel = run(nfi.Strategy.from_spec(spec).with_(benchmark=nfi.Index("SYNTH_IG")), src, start="2015-03-02")
    native = run(nfi.Strategy("native", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").winsor(0.01).zscore(by="sector"), portfolio=nfi.Quantile(5), benchmark=nfi.Index("SYNTH_IG"), missing=nfi.ZeroFill()), src, start="2015-03-02")
    idx = legacy.returns.index
    assert np.abs(legacy.returns - kernel.returns.reindex(idx)).max() < 1e-12
    assert np.abs(legacy.returns - native.returns.reindex(idx)).max() < 1e-12


def test_triggered_schedule_skips_small_changes(src):
    from nautilus_fi.kernel.engine import run

    st = nfi.Strategy("trig", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").rolling(63).mean(), portfolio=nfi.TopN(40), missing=nfi.ZeroFill(), schedule=nfi.Triggered(0.2, "turnover"))
    r = run(st, src)
    skips = (r.construction["action"] == "trigger_skip").sum()
    assert skips > 0 and len(r.trades) < r.diagnostics["n_decisions"]


# ====================================================================================== results, sessions, API
def test_compare_explain_and_unified_stats(src):
    sess = nfi.Session(source=src)
    base = nfi.Strategy("value", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").zscore(by="sector"), portfolio=nfi.TopN(60), benchmark=nfi.Index("SYNTH_IG"), missing=nfi.ZeroFill())
    idea = base.with_(signal=base.signal.neutralise("oasd"), name="neutral")
    cmp = sess.compare(base, idea)
    e = cmp.explain()
    text = e.text()
    assert "What changed" in text and "signal" in text and "Where the difference came from" in text and "by sector" in text
    d = e.decomposition
    assert abs(d["net"] - (d["holdings"] + d["cash_financing"] + d["costs"] + d["residual"])) < 1e-12
    st = cmp[0].stats()
    for k in ("annual_return", "sharpe", "turnover", "cost_drag", "tracking_error", "information_ratio", "max_drawdown", "avg_names"):
        assert k in st
    same = sess.compare(base, base.with_(name="again"))
    assert "identical" in same.explain().text()


def test_session_counts_distinct_trials_caches_and_deflates(src, tmp_path):
    base = nfi.Strategy("value", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.TopN(60), missing=nfi.ZeroFill())
    sess = nfi.Session(tmp_path, source=src, name="thread-1")
    sess.run(base)
    sess.run(base)
    sw = sess.sweep(base, {"portfolio.n": [30, 60, 120]})
    assert sess.n_trials() == 3 and len(sw.table()) == 3
    assert sw.selection()["deflated_sharpe"]["ledger_trials"] == 3
    again = nfi.Session(tmp_path, source=src, name="thread-1")
    r = again.run(base)
    assert again.log[-1]["cached"] and again.n_trials() == 3 and r.name == "value"


def test_parallel_sweep_matches_sequential(src):
    base = nfi.Strategy("value", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value").zscore(by="sector"), portfolio=nfi.TopN(60), missing=nfi.ZeroFill())
    par = nfi.Session(source=src).sweep(base, {"portfolio.n": [20, 40, 80]}, workers=2)
    seq = nfi.Session(source=src).sweep(base, {"portfolio.n": [20, 40, 80]})
    assert par.mode.endswith("processes") and np.allclose(par.table()["sharpe"].sort_index(), seq.table()["sharpe"].sort_index())


def test_ledger_is_safe_with_concurrent_writers(tmp_path):
    from nautilus_fi.governance import TrialLedger

    def write(k):
        led = TrialLedger(tmp_path)
        for i in range(10):
            led.record("PAPER", f"w{k}", f"h{k}{i}", {"i": i}, ("2020-01-01", "2020-12-31"), {"sharpe": 0.1 * i}, pd.Series(np.random.default_rng(i).normal(0, 0.01, 50), index=pd.bdate_range("2020-01-01", periods=50)), session=f"s{k}")

    threads = [threading.Thread(target=write, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    led = TrialLedger(tmp_path)
    assert len(led.trials_list()) == 40 and led.n_trials(None, ("PAPER",), session="s2") == 10


def test_promote_through_the_gates(src, tmp_path):
    from nautilus_fi.governance import GateFailure
    from nautilus_fi.research import Hypothesis

    h = Hypothesis("H-KERNEL", "wide spreads mean revert", "SYNTH_IG", "month end", "signal.value", "SYNTH_IG", 1e8, "long only", "1m", "half spread", "Sharpe < 0", ("2015-01-01", "2015-12-31"), ("2016-01-01", "2016-06-30"), ("2030-01-01", "2030-12-31"))
    sess = nfi.Session(tmp_path, source=src, hypothesis=h)
    st = nfi.Strategy("value", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.TopN(60), missing=nfi.ZeroFill())
    r = sess.run(st)
    rec = sess.promote(r, "EXPLORATORY")
    assert rec.level == "EXPLORATORY" and rec.spec_hash == st.fingerprint
    cand = sess.promote(r, "CANDIDATE")
    assert cand.level == "CANDIDATE" and cand.n_trials == 1
    legacy = sess.run(st.under("MRI_v1"))                      # an unconfirmed legacy profile cannot be a candidate
    with pytest.raises(GateFailure, match="conventions_confirmed"):
        sess.promote(legacy, "CANDIDATE")


def test_api_vocabulary_and_evaluate(src):
    nfi.session(source=src)
    text = nfi.vocabulary()
    for word in ("Optimise", "Index(", "NextClose", "sweep", "explain"):
        assert word in text
    rep = nfi.evaluate(nfi.Field("signal.value").zscore(by="sector"), universe=nfi.Index("SYNTH_IG"), horizons=(5, 21))
    assert rep.summary().loc[21, "mean_ic"] > 0
    assert any(t["stage"] == "EVALUATE" for t in nfi.current_session().ledger.trials_list())


def test_implement_fidelity_runs_the_same_strategy():
    sd = nfi.SyntheticSource(n_bonds=40, start="2023-01-03", end="2023-05-31", seed=5)
    st = nfi.Strategy("wide", universe=nfi.Index("SYNTH_IG") & (nfi.Field("oasd") > 1), signal=nfi.Field("oas_bp").zscore(), portfolio=nfi.TopN(8), missing=nfi.ZeroFill(), capital=50e6)
    sess = nfi.Session(source=sd)
    paper = sess.run(st)
    impl = sess.run(st, fidelity="implement")
    assert impl.fidelity == "implement" and impl.stats()["implement_outcome"] == "SOLVENT"
    shared = {"annual_return", "annual_vol", "sharpe", "max_drawdown", "turnover", "cost_drag"}
    assert shared <= set(impl.stats()) and shared <= set(paper.stats())
    rec = nfi.reconcile(paper, impl, source=sd)
    assert {"holdings_gap", "execution", "residual"} <= set(rec.summary) and rec.summary["daily_correlation"] > 0.8
    assert "fidelities differ" in impl.explain(paper).text()
