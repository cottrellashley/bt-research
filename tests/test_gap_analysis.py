"""Regression tests for the 2026-10-30 gap analysis: each one fails on revision 10dd477.

IDs follow the analysis: D1-D10 correctness defects, G1-G14 governance bypasses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nautilus_fi as nfi
from nautilus_fi.domain.conventions import MissingReturnsError
from nautilus_fi.kernel.engine import run


@pytest.fixture(scope="module")
def syn():
    return nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2016-06-30", seed=7)


def _relabelled(src, fields=(), name="relabelled", **kw):
    """The synthetic data as a plain PanelSource with some FieldInfo overrides (availability lags)."""
    d = src.data
    return nfi.PanelSource(d.panel, d.attributes, src._series, {src.INDEX: d.membership}, d.overlays, list(fields), name=name, **kw)


def _value(src_index="SYNTH_IG", **kw):
    base = {"universe": nfi.Index(src_index), "signal": nfi.Field("signal.value"), "portfolio": nfi.Quantile(5), "benchmark": nfi.Index(src_index), "missing": nfi.ZeroFill()}
    base.update(kw)
    return nfi.Strategy("value", **base)


def _tiny(n=6, days=80, seed=0, **extra):
    """A small frame source: daily returns, an always-on signal, a cash series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=days)
    ids = [f"B{i}" for i in range(n)]
    tr = pd.DataFrame(rng.normal(2e-4, 3e-3, (days, n)), index=dates, columns=ids)
    sig = pd.DataFrame(rng.normal(size=(days, n)), index=dates, columns=ids)
    fields = {"tr": tr, "signal.value": sig, **extra.pop("fields", {})}
    series = extra.pop("series", {"cash": pd.Series(1e-4, index=dates)})
    return nfi.FrameSource(fields, series=series, **extra), dates, ids


# ====================================================================================== D1
def test_d1_realised_returns_ignore_the_availability_lag(syn):
    lag0 = _relabelled(syn, name="lag0")
    lag1 = _relabelled(syn, [nfi.FieldInfo("tr", lag=1)], name="lag1")
    st = _value()
    a, b = run(st, lag0), run(st, lag1)
    assert np.abs(a.gross - b.gross).max() < 1e-15
    assert np.abs(a.benchmark_returns - b.benchmark_returns).max() < 1e-15
    p = syn.data.panel
    from nautilus_fi.kernel.data import DataRequest
    from nautilus_fi.kernel.expr import EvalContext

    c = EvalContext(lag1.load(DataRequest(("tr",), None, None)), None, None, lag1, lag1.catalog())
    np.testing.assert_array_equal(np.nan_to_num(nfi.Field("tr").evaluate(c)[10]), np.nan_to_num(p["tr"][9]))     # a signal still sees yesterday's return


# ====================================================================================== D2
def test_d2_cash_band_alone_keeps_the_book_fully_invested(syn):
    st = nfi.Strategy("opt", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.Optimise(nfi.CashWeight() <= 0.01, nfi.Weight() <= 0.05), missing=nfi.ZeroFill(), financing=nfi.Cash("cash"))
    r = run(st, syn, end="2015-06-30")
    invested = r.weights.groupby("date")["weight"].sum()
    assert invested.max() <= 1.0 + 1e-6 and invested.min() >= 0.99 - 1e-6
    lev = st.with_(portfolio=nfi.Optimise(nfi.CashWeight() <= 0.01, nfi.Leverage() <= 1.3, nfi.Weight() <= 0.05))
    assert run(lev, syn, end="2015-06-30").weights.groupby("date")["weight"].sum().max() > 1.01      # leverage needs saying so


# ====================================================================================== D3
def test_d3_excess_return_compounds_each_leg(syn):
    st = _value(metrics=nfi.Metrics(excess=nfi.Compounded("tsy_r", vendor=None)))
    r = run(st, syn)
    g, ts = r.gross, r.daily["treasury_leg"]
    monthly = r.excess("ME")
    legs = pd.DataFrame({"g": g, "t": ts})
    expect = legs.groupby(pd.Grouper(freq="ME")).apply(lambda m: float((1 + m["g"]).prod() - (1 + m["t"]).prod()))
    assert np.abs(monthly.reindex(expect.index) - expect).max() < 1e-12
    s = r.stats()
    y = s["years"]
    gt, gs = float((1 + g.iloc[1:]).prod()), float((1 + ts.iloc[1:]).prod())
    assert abs(s["excess_annual_return"] - (gt ** (1 / y) - gs ** (1 / y))) < 1e-12
    legacy = run(st.under("MRI_v1"), syn)                          # DAILY_SUM only under the profile that sums
    lg = legacy.gross.iloc[1:] - legacy.daily["treasury_leg"].iloc[1:]
    assert abs(legacy.stats()["excess_annual_return"] - lg.sum() / legacy.stats()["years"]) < 1e-12


# ====================================================================================== D4
def test_d4_missing_benchmark_returns_follow_the_policy():
    src, dates, ids = _tiny()
    tr = src.raw_panel["tr"].copy()
    tr[30, 2] = np.nan                                              # a benchmark member with no return
    cons = pd.DataFrame({"date": dates[0], "instrument_id": ids, "weight": 1.0 / len(ids)})
    src = nfi.FrameSource({"tr": pd.DataFrame(tr, index=dates, columns=ids), "signal.value": src.raw_panel.frame("signal.value")}, series={"cash": pd.Series(1e-4, index=dates)}, constituents={"IX": cons})
    st = nfi.Strategy("b", universe=nfi.Index("IX"), signal=nfi.Field("signal.value"), portfolio=nfi.TopN(2), benchmark=nfi.Index("IX"), missing=nfi.Fail(), execution=nfi.NextClose(cost=nfi.NoCost()))
    with pytest.raises(MissingReturnsError, match="benchmark"):
        run(st, src)
    r = run(st.with_(missing=nfi.ZeroFill()), src)
    assert r.missing["benchmark_zero_filled"] == 1 and r.stats()["data_fills"] >= 1


def test_d4_missing_cash_follows_the_policy():
    src, _, _ = _tiny(series={"cash": pd.Series(1e-4, index=pd.bdate_range("2021-01-04", periods=80)).where(lambda x: x.index != pd.Timestamp("2021-02-15"))})
    st = nfi.Strategy("c", signal=nfi.Field("signal.value"), portfolio=nfi.TopN(2, invested=0.5), missing=nfi.Fail(), execution=nfi.NextClose(cost=nfi.NoCost()))
    with pytest.raises(MissingReturnsError, match="cash"):
        run(st, src)
    assert run(st.with_(missing=nfi.ZeroFill()), src).missing["cash_zero_filled"] == 1


def test_d4_off_calendar_series_compound_into_the_next_trading_day():
    src, dates, _ = _tiny()
    cal = pd.date_range(dates[0], dates[-1], freq="D")                # the benchmark publishes every calendar day
    pub = pd.Series(np.random.default_rng(1).normal(3e-4, 2e-3, len(cal)), index=cal)
    src = nfi.FrameSource({"tr": src.raw_panel.frame("tr"), "signal.value": src.raw_panel.frame("signal.value")}, series={"cash": pd.Series(1e-4, index=dates), "bench.tr": pub})
    r = run(nfi.Strategy("rs", signal=nfi.Field("signal.value"), portfolio=nfi.TopN(2), benchmark=nfi.ReturnSeries("bench.tr"), missing=nfi.ZeroFill(), execution=nfi.NextClose(cost=nfi.NoCost())), src)
    total = float((1 + r.benchmark_returns.iloc[1:]).prod())
    expect = float((1 + pub[(pub.index > r.returns.index[0]) & (pub.index <= r.returns.index[-1])]).prod())
    assert abs(total - expect) < 1e-12 and r.missing["snapped_observations"] > 0


def test_d4_returns_that_stop_without_an_exit_are_missing_data():
    src, dates, ids = _tiny()
    tr = src.raw_panel.frame("tr")
    tr.loc[dates[40]:, "B0"] = np.nan                               # B0's data just stops: no alive field, no maturity
    st = nfi.Strategy("e", signal=nfi.Field("signal.value"), portfolio=nfi.EqualWeight(), missing=nfi.Fail(), execution=nfi.NextClose(cost=nfi.NoCost()))
    plain = nfi.FrameSource({"tr": tr, "signal.value": src.raw_panel.frame("signal.value")}, series={"cash": pd.Series(1e-4, index=dates)})
    with pytest.raises(MissingReturnsError, match="no maturity, call or exit"):
        run(st, plain)
    attrs = pd.DataFrame({"maturity": [dates[39]] + [pd.Timestamp("2040-01-01")] * 5}, index=ids)
    matured = nfi.FrameSource({"tr": tr, "signal.value": src.raw_panel.frame("signal.value")}, attributes=attrs, series={"cash": pd.Series(1e-4, index=dates)})
    r = run(st, matured)
    assert r.missing["redeemed"] >= 1 and r.missing["unexplained_exits"] == 0


def test_d4_percent_scale_returns_and_insolvency_stop_the_run():
    src, dates, _ = _tiny()
    pct = nfi.FrameSource({"tr": src.raw_panel.frame("tr") * 100, "signal.value": src.raw_panel.frame("signal.value")}, series={"cash": pd.Series(1e-4, index=dates)})
    st = nfi.Strategy("p", signal=nfi.Field("signal.value"), portfolio=nfi.TopN(2), missing=nfi.ZeroFill(), execution=nfi.NextClose(cost=nfi.NoCost()))
    with pytest.raises(nfi.DataQualityError, match="percentages"):
        run(st, pct)
    crash = src.raw_panel.frame("tr")
    crash.iloc[30] = -1.5
    bust = nfi.FrameSource({"tr": crash, "signal.value": src.raw_panel.frame("signal.value")}, series={"cash": pd.Series(1e-4, index=dates)})
    with pytest.raises(nfi.InsolvencyError):
        run(st, bust)


def test_d4_duplicate_rows_and_conflicting_panels_raise():
    long = pd.DataFrame({"date": ["2021-01-04", "2021-01-04", "2021-01-05"], "instrument_id": ["A", "A", "A"], "tr": [0.01, 0.02, 0.0]})
    with pytest.raises(nfi.DataQualityError, match="share a"):
        nfi.FrameSource(long)
    from nautilus_fi.data.panel import Panel
    from nautilus_fi.kernel.data import align

    d = pd.bdate_range("2021-01-04", periods=3)
    a = Panel(d, pd.Index(["A"]), {"x": np.array([[1.0], [2.0], [3.0]])})
    b = Panel(d, pd.Index(["A"]), {"x": np.array([[1.0], [2.5], [np.nan]])})
    with pytest.raises(nfi.DataQualityError, match="disagree"):
        align([a, b])
    c = Panel(d, pd.Index(["A"]), {"x": np.array([[np.nan], [2.0], [3.0]])})
    assert np.isfinite(align([c, a])["x"]).all()                    # agreeing sources fill each other's gaps


def test_d4_half_spread_fallbacks_are_counted(syn):
    from nautilus_fi.kernel.data import MissingDataError

    hs = syn.data.panel.frame("half_spread_bp").copy()
    hs.iloc[:, :40] = np.nan
    fields = {k: syn.data.panel.frame(k) for k in ("tr", "signal.value", "alive", "index_weight")} | {"half_spread_bp": hs}
    src = nfi.FrameSource(fields, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership})
    st = _value(benchmark=None)
    with pytest.raises(MissingDataError, match="fallback_bp"):
        run(st, src, end="2015-06-30")
    r = run(st.with_(execution=nfi.NextClose(cost=nfi.HalfSpread(fallback_bp="median"))), src, end="2015-06-30")
    assert r.missing["cost_filled"] > 0 and r.stats()["data_fills"] >= r.missing["cost_filled"]


def test_d4_drop_dates_removes_the_date_for_portfolio_and_benchmark():
    sd = nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2015-12-31", n_bad_dates=3, seed=4)
    st = _value(missing=nfi.DropDates(0.5))
    r = run(st, sd)
    bad = pd.to_datetime(r.missing["bad_dates"])
    assert len(bad) == 3 and not r.returns.index.isin(bad).any() and not r.benchmark_returns.index.isin(bad).any()
    assert r.missing["dropped"] == 3 and len(r.returns) == len(sd.data.panel.dates) - 3


def test_d4_evaluation_excludes_windows_with_missing_returns():
    from nautilus_fi.paper.evaluate import EvaluationSettings, evaluate_signal

    sd = nfi.SyntheticSource(n_bonds=150, start="2015-01-02", end="2015-12-31", missing_rate=0.01, seed=2)
    rep = evaluate_signal(sd.data.panel, "signal.value", settings=EvaluationSettings(horizons=(21,), return_field="tr", n_boot=50))
    assert rep.excluded.sum() > 0
    zero = evaluate_signal(sd.data.panel, "signal.value", settings=EvaluationSettings(horizons=(21,), return_field="tr", n_boot=50, missing="zero"))
    assert not zero.ic[21].equals(rep.ic[21])


# ====================================================================================== D5
def test_d5_index_leavers_are_sold_despite_a_turnover_cap(syn):
    st = nfi.Strategy("opt", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.Optimise(nfi.Weight() <= 0.03, nfi.Turnover() <= 0.02), missing=nfi.ZeroFill())
    r = run(st, syn)
    member = nfi.Index("SYNTH_IG")
    from nautilus_fi.kernel.engine import prepare

    prep = prepare([st], syn)
    M = member.known_weights(prep.ctx) > 0
    dates, ids = prep.ctx.panel.dates, prep.ctx.panel.ids
    w = r.weights.merge(r.trades[["trade_date", "decision_date"]], left_on="date", right_on="trade_date")
    dec = dates.get_indexer(w["decision_date"])
    col = ids.get_indexer(w["instrument_id"])
    assert M[dec, col].all()                                      # nothing held after a rebalance left the index at its decision
    assert r.trades["forced_sells"].sum() > 0
    hold = run(st.with_(exits="HOLD_NOT_BUY"), syn)
    assert hold.trades["forced_sells"].sum() == 0 and not hold.returns.equals(r.returns)


def test_d5_an_empty_signal_keeps_the_book(syn):
    sig = syn.data.panel.frame("signal.value").copy()
    sig.loc["2015-06-30"] = np.nan                                  # a month-end outage
    fields = {k: syn.data.panel.frame(k) for k in ("tr", "alive", "index_weight", "half_spread_bp")} | {"signal.value": sig}
    src = nfi.FrameSource(fields, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership})
    r = run(_value(benchmark=None), src)
    ab = r.construction[r.construction["action"] == "aborted"]
    assert len(ab) == 1 and ab["date"].iloc[0] == pd.Timestamp("2015-06-30") and r.diagnostics["aborted_rebalances"] == 1
    assert r.daily.loc["2015-07-06", "n_names"] > 0                # still invested after the outage


# ====================================================================================== D6
def test_d6_relaxed_constraints_are_counted_or_refused(syn):
    from nautilus_fi.construct.optimiser import OptimisationError

    impossible = nfi.Optimise(nfi.Weight() <= 0.05, nfi.Exposure("oasd") <= 0.5)       # a duration cap no long-only book meets
    st = nfi.Strategy("relax", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=impossible, missing=nfi.ZeroFill())
    r = run(st, syn, end="2015-04-30")
    assert r.stats()["relaxed_rebalances"] > 0 and (r.construction["action"] == "rebalance_relaxed").all()
    assert "oasd" in r.construction["relaxed_constraints"].iloc[0]
    with pytest.raises(OptimisationError, match="infeasible|Infeasible"):
        run(st.with_(portfolio=impossible.with_settings(on_infeasible="fail")), syn, end="2015-04-30")


def test_d4_drop_dates_kernel_matches_the_legacy_engine():
    from nautilus_fi.benchmarks import ConstituentBenchmark
    from nautilus_fi.construct import ConstructionSpec
    from nautilus_fi.construct import Quantile as LegacyQuantile
    from nautilus_fi.domain import Conventions, MissingReturnPolicy
    from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
    from nautilus_fi.paper import PaperData, PaperEngine, PaperSettings

    sd = nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2015-12-31", n_bad_dates=3, seed=4)
    d = sd.data
    spec = ResearchSpec("legacy", FieldSignal("signal.value"), ConstructionSpec(LegacyQuantile(5)), Conventions().with_(missing_returns=MissingReturnPolicy.DROP_DATE), calendar="BME")
    legacy = PaperEngine(spec, PaperData(d.panel, d.attributes, d.cash, ConstituentBenchmark("IDX", d.membership)), PaperSettings(start="2015-02-02")).run()
    kernel = run(nfi.Strategy.from_spec(spec).with_(benchmark=nfi.Index("SYNTH_IG")), sd, start="2015-02-02")
    assert legacy.returns.index.equals(kernel.returns.index) and np.abs(legacy.returns - kernel.returns).max() < 1e-12


# ====================================================================================== governance (G1-G14) and D7.7 / D8
def _h(hid, criteria="sharpe < -10", holdout=("2030-01-01", "2030-12-31")):
    from nautilus_fi.research import Hypothesis

    return Hypothesis(hid, "wide spreads mean revert", "SYNTH_IG", "month end", "signal.value", "SYNTH_IG", 1e8, "long only", "1m", "half spread", criteria, ("2015-01-01", "2015-12-31"), ("2016-01-01", "2016-06-30"), holdout)


def _base():
    return nfi.Strategy("value", universe=nfi.Index("SYNTH_IG"), signal=nfi.Field("signal.value"), portfolio=nfi.TopN(60), missing=nfi.ZeroFill())


LENIENT = {"min_dsr": None, "max_pbo": None}


def test_g1_the_search_counts_whatever_hypothesis_it_was_filed_under(syn, tmp_path):
    search = nfi.Session(tmp_path, source=syn)                       # 12 variants, no hypothesis
    sw = search.sweep(_base(), {"portfolio.n": [10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120]})
    fresh = nfi.Session(tmp_path, source=syn, name="fresh", hypothesis=_h("H-FRESH"))
    r = fresh.run(_base().set("portfolio.n", 20).with_(hypothesis="H-FRESH"))           # the winner, refiled
    rec = fresh.promote(r, "CANDIDATE", **LENIENT)
    distinct = len({tuple(np.round(x.returns.to_numpy(), 14)) for x in sw.results})      # variants holding the same names earn the same
    assert rec.n_trials == 13 and rec.dsr["ledger_trials"] == distinct > 1


def test_g2_relabelling_an_unconfirmed_profile_does_not_confirm_it(syn, tmp_path):
    from nautilus_fi.governance import GateFailure

    st = _base().under("paperport").with_(profile="kernel", name="relabelled", hypothesis="H2")
    assert not st.conventions().confirmed
    sess = nfi.Session(tmp_path, source=syn, hypothesis=_h("H2"))
    r = sess.run(st)
    with pytest.raises(GateFailure, match="conventions_confirmed"):
        sess.promote(r, "CANDIDATE", **LENIENT)


def test_g3_registration_is_stamped_and_frozen(tmp_path):
    from nautilus_fi.governance import TrialLedger

    led = TrialLedger(tmp_path)
    h = _h("H3")
    h.registered_utc = "2000-01-01T00:00:00+00:00"                 # a backdated registration
    reg = led.register(h)
    assert reg.registered_utc > "2026" and led.registry.frozen("H3")
    import json

    path = tmp_path / "hypotheses.json"
    data = json.loads(path.read_text())
    data["hypotheses"]["H3"]["failure_criteria"] = "sharpe < -100"   # edited after registration
    path.write_text(json.dumps(data))
    led.registry.reload()
    assert not led.registry.frozen("H3")


def test_g4_official_needs_the_same_strategys_implement_run_and_a_reconciliation(syn, tmp_path):
    from types import SimpleNamespace

    from nautilus_fi.governance.records import RecordLevel, RecordRegistry

    sess = nfi.Session(tmp_path, source=syn, hypothesis=_h("H4"))
    r = sess.run(_base().with_(hypothesis="H4"))
    other = SimpleNamespace(status="COMPLETED", outcome="SOLVENT", manifest=SimpleNamespace(output_hashes={"event_chain": "x"}, code_version="v"), stats=dict)
    from nautilus_fi.kernel.session import _SpecProxy

    checks = RecordRegistry(tmp_path / "rec", **LENIENT).checks(RecordLevel.OFFICIAL, r, _SpecProxy(r.fingerprint, "H4"), sess.ledger, other, {}, "some-other-fingerprint")
    assert not checks["implement_matches_paper"]["ok"] and not checks["reconciled"]["ok"] and not checks["reconciliation_tolerance"]["ok"]


def test_g5_thresholds_are_enforced_by_default(syn, tmp_path):
    from nautilus_fi.governance import GateFailure

    sess = nfi.Session(tmp_path, source=syn, hypothesis=_h("H5"))
    short = sess.run(_base().with_(hypothesis="H5"), end="2015-06-30")
    with pytest.raises(GateFailure) as e:
        sess.promote(short, "CANDIDATE")
    assert "track_record" in e.value.failed
    weak = sess.run(_base().with_(signal=-nfi.Field("signal.value"), portfolio=nfi.Quantile(5, short=1), financing=nfi.NoCash(short_fee_bp=0), name="anti", hypothesis="H5"))       # no skill, no carry
    with pytest.raises(GateFailure) as e:
        sess.promote(weak, "CANDIDATE")
    assert "dsr_threshold" in e.value.failed


def test_g6_the_holdout_is_seen_once(syn, tmp_path):
    from nautilus_fi.research.experiment import HoldoutViolation

    sess = nfi.Session(tmp_path, source=syn, hypothesis=_h("H6", holdout=("2016-03-01", "2016-06-30")))
    st = _base().with_(hypothesis="H6")
    r = sess.run(st)                                                # touches the holdout
    with pytest.raises(HoldoutViolation):
        sess.run(st.set("portfolio.n", 30))
    from nautilus_fi.governance import GateFailure

    with pytest.raises(GateFailure, match="holdout_guard"):        # and a result that saw it is not a candidate
        sess.promote(r, "CANDIDATE", **LENIENT)


def test_g7_g8_one_record_per_result_and_an_anchored_chain(syn, tmp_path):
    from nautilus_fi.governance import GateFailure
    from nautilus_fi.governance.records import RecordRegistry

    sess = nfi.Session(tmp_path, source=syn, hypothesis=_h("H7"))
    r = sess.run(_base().with_(hypothesis="H7"))
    sess.promote(r, "EXPLORATORY")
    with pytest.raises(GateFailure, match="not_already_recorded"):
        sess.promote(r, "EXPLORATORY")
    sess.promote(r.__class__(**{**r.__dict__, "_stats": None}), "CANDIDATE", min_dsr=0.5)
    reg = RecordRegistry(tmp_path / "records")
    assert reg.verify_chain()
    lines = reg.path.read_text().splitlines()
    reg.path.write_text("\n".join(lines[:-1]) + "\n")                 # delete the last record
    assert not reg.verify_chain()
    with pytest.raises(GateFailure, match="chain"):
        sess.promote(r, "EXPLORATORY")


def test_g9_concurrent_registration_cannot_overwrite(tmp_path):
    import threading

    from nautilus_fi.governance import TrialLedger

    errors, done = [], []

    def go(text):
        h = _h("H9")
        h.rationale = text
        try:
            TrialLedger(tmp_path).register(h)
            done.append(text)
        except ValueError as e:
            errors.append(e)

    threads = [threading.Thread(target=go, args=(f"writer {k}",)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(done) == 1 and len(errors) == 5
    assert TrialLedger(tmp_path).hypothesis("H9").rationale == done[0]


def test_g10_an_unregistered_hypothesis_refuses_the_run(syn, tmp_path):
    sess = nfi.Session(tmp_path, source=syn)
    with pytest.raises(KeyError, match="not registered"):
        sess.run(_base().with_(hypothesis="NOT-REGISTERED"))
    assert sess.n_trials() == 0


def test_g11_an_unreadable_trial_file_fails_the_count(syn, tmp_path):
    from nautilus_fi.governance.ledger import LedgerError

    sess = nfi.Session(tmp_path, source=syn)
    sess.run(_base())
    (tmp_path / "ledger" / "trials" / "20261030-corrupt.json").write_text("{not json")
    with pytest.raises(LedgerError):
        sess.ledger.n_trials(None)


def test_g12_failure_criteria_are_evaluated(syn, tmp_path):
    from nautilus_fi.governance import GateFailure

    sess = nfi.Session(tmp_path, source=syn, hypothesis=_h("H12", "sharpe < 5"))
    r = sess.run(_base().with_(hypothesis="H12"))
    with pytest.raises(GateFailure, match="failure_criteria"):
        sess.promote(r, "CANDIDATE", **LENIENT)
    vague = nfi.Session(tmp_path / "v", source=syn, hypothesis=_h("H12V", "if it stops working"))
    with pytest.raises(GateFailure, match="machine-checkable"):
        vague.promote(vague.run(_base().with_(hypothesis="H12V")), "CANDIDATE", **LENIENT)


def test_g13_one_trial_definition_for_counts_and_deflation(tmp_path):
    from nautilus_fi.governance import TrialLedger

    led = TrialLedger(tmp_path)
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 60), index=pd.bdate_range("2020-01-01", periods=60))
    for stage in ("PAPER", "IMPLEMENT", "VALIDATION"):
        led.record(stage, "s", "same-spec", {}, ("2020-01-01", "2020-03-24"), {}, rets, data_version="v1")
    led.record("PAPER", "s2", "other-spec", {}, ("2020-01-01", "2020-03-24"), {}, rets * 1.1, data_version="v1")
    assert led.n_trials(None) == 2 and led.deflated_sharpe(None, rets)["ledger_trials"] == 2
    led.record("IMPLEMENT", "s2", "other-spec", {}, ("2020-01-01", "2020-03-24"), {}, rets * 0.9, data_version="v1")
    picked = {t["spec_hash"]: t["stage"] for t in led.counted()}
    assert picked == {"same-spec": "VALIDATION", "other-spec": "PAPER"}                # an Implement run checks a trial, it is not the draw


def test_g14_d77_official_needs_point_in_time_inputs_and_lookahead_checks(syn, tmp_path):
    from nautilus_fi.governance import GateFailure
    from nautilus_fi.governance.records import RecordLevel, RecordRegistry
    from nautilus_fi.kernel.session import _SpecProxy

    unknown = _relabelled(syn, name="unknown-vintage")                 # cannot say whether values were revised
    sess = nfi.Session(tmp_path, source=unknown, hypothesis=_h("H14"))
    r = sess.run(_base().with_(hypothesis="H14"))
    checks = RecordRegistry(tmp_path / "rec", **LENIENT).checks(RecordLevel.OFFICIAL, r, _SpecProxy(r.fingerprint, "H14"), sess.ledger)
    assert not checks["point_in_time_inputs"]["ok"]
    unchecked = nfi.Session(tmp_path / "u", source=syn, hypothesis=_h("H14U"), settings=nfi.RunSettings(check_lookahead=False))
    with pytest.raises(GateFailure, match="lookahead_checked"):
        unchecked.promote(unchecked.run(_base().with_(hypothesis="H14U")), "CANDIDATE", **LENIENT)


def test_d8_cache_keys_include_settings_and_trials_include_data(syn, tmp_path):
    sess = nfi.Session(tmp_path, source=syn)
    a = sess.run(_base())
    b = sess.run(_base(), settings=nfi.RunSettings(first_decision=False))
    assert not sess.log[-1]["cached"] and not a.returns.equals(b.returns)
    other = nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2016-06-30", seed=8)     # refreshed data
    sess.run(_base(), source=other)
    assert sess.n_trials() == 2                                     # same strategy and window, new data: a new trial
    frames = {k: syn.data.panel.frame(k) for k in ("tr", "signal.value", "alive", "index_weight", "half_spread_bp")}
    plain = nfi.FrameSource(frames, syn.data.attributes, syn._series, {"SYNTH_IG": syn.data.membership})
    plain._version = None
    plain.version = lambda: None                                    # an unversioned source
    sess.run(_base(), source=plain)
    assert not list((tmp_path / "results").glob(f"{sess.log[-1]['key']}.pkl"))


# ====================================================================================== D7
def test_d7_benchmark_reconstitution_uses_what_exists_at_its_close():
    from nautilus_fi.domain.conventions import WeightDrift
    from nautilus_fi.kernel.benchmark import drift_weights

    alive = np.array([[True, True], [True, True], [True, False], [True, False]])
    R = np.array([[0.0, 0.0], [0.01, 0.01], [0.01, np.nan], [0.01, np.nan]])
    W = drift_weights({1: np.array([0.5, 0.5])}, R, alive, WeightDrift.DRIFT)
    assert W[1, 1] == 0.5                                           # B is a member at the close of row 1: its exit tomorrow is not known


def test_d7_as_of_reads_are_strictly_before_the_cutoff(tmp_path):
    from nautilus_fi.data import field_store
    from nautilus_fi.data.field_store import FieldStore

    fs = FieldStore(tmp_path)
    fs.write("oas", pd.DataFrame({"instrument_id": ["A"], "date": ["2024-01-02"], "oas": [100.0]}), knowledge_time="2024-01-02T22:00:00Z")
    fs.write("oas", pd.DataFrame({"instrument_id": ["A"], "date": ["2024-01-02"], "oas": [101.0]}), knowledge_time="2024-01-03T22:00:00Z")
    assert fs.read("oas", as_of="2024-01-03T22:00:00Z")["value"].iloc[0] == 100.0     # published at the cutoff: not yet known
    assert fs.read("oas", as_of="2024-01-03T22:00:00Z", inclusive=True)["value"].iloc[0] == 101.0
    assert field_store.AS_OF_INCLUSIVE is False


def test_d7_walk_forward_purge_counts_the_execution_lag():
    from nautilus_fi.research.walkforward import purged_train_indices

    td = pd.bdate_range("2020-01-01", periods=120)
    test_start, test_end = td[80], td[100]
    plain = purged_train_indices(td[:80], test_start, test_end, 5, trading_days=td)
    lagged = purged_train_indices(td[:80], test_start, test_end, 5, trading_days=td, execution_lag_days=1)
    assert len(lagged) == len(plain) - 1 and td[lagged].max() + pd.offsets.BDay(6) < test_start


def test_d7_unbounded_carries_load_all_history(syn):
    from nautilus_fi.kernel.engine import prepare

    st = _value(signal=nfi.Field("signal.value").ffill())
    prep = prepare([st], syn, start="2016-01-04")
    assert prep.panel.dates[0] == syn.data.panel.dates[0]
    assert "all history" in str(st.requirements(syn.catalog()))


def test_d7_explicit_dates_on_holidays_act_after_them():
    from datetime import date

    days = [date(2024, 7, 3), date(2024, 7, 5), date(2024, 7, 8)]
    assert nfi.Explicit((date(2024, 7, 4),)).dates(days) == [date(2024, 7, 5)]


def test_d7_off_calendar_field_dates_do_not_become_trading_rows():
    dates = pd.bdate_range("2024-01-01", periods=30)
    tr = pd.DataFrame(1e-4, index=dates, columns=["A", "B"])
    rating = pd.DataFrame(5.0, index=[pd.Timestamp("2024-01-13"), pd.Timestamp("2024-01-31")], columns=["A", "B"])     # a Saturday month-end file
    src = nfi.FrameSource({"tr": tr, "rating_score": rating})
    p = src.raw_panel
    assert p.dates.equals(dates) and src.snapped_observations == 2
    assert p.frame("rating_score").loc["2024-01-15", "A"] == 5.0     # the Saturday value is known from Monday
    long = pd.DataFrame({"date": list(dates) * 2 + [pd.Timestamp("2024-01-13")], "instrument_id": ["A"] * 30 + ["B"] * 30 + ["A"], "tr": [1e-4] * 60 + [np.nan], "rating_score": [np.nan] * 60 + [7.0]})
    assert nfi.FrameSource(long).raw_panel.dates.equals(dates)


def test_d7_decider_lookahead_is_checked(syn):
    from nautilus_fi.domain.spec import Decider, Target
    from nautilus_fi.kernel.expr import LookaheadError

    class Peek(Decider):
        def decide(self, view):
            fut = view._p.values["tr"][min(view._i + 5, len(view._p.dates) - 1)]          # reads five rows ahead
            return Target(scores=pd.Series(np.nan_to_num(fut[view._cand]), index=view.ids))

    st = nfi.Strategy("peek", decider=Peek(), portfolio=nfi.TopN(20), missing=nfi.ZeroFill())
    with pytest.raises(LookaheadError, match="decider Peek"):
        run(st, syn)


# ====================================================================================== D9 / D10
def test_d9_erin_profile_needs_a_refill_field_and_counts_zeros_only(syn):
    from nautilus_fi.kernel.strategy import StrategyError

    with pytest.raises(StrategyError, match="refill"):
        _value().under("erin_beta_plus")
    st = _value().under("erin_beta_plus", refill="tr")
    assert st.missing.threshold == 0.8 and st.missing.zeros_only and st.missing.refill == "tr"
    from nautilus_fi.domain.conventions import detect_bad_dates

    d = pd.bdate_range("2024-01-01", periods=3)
    r = pd.DataFrame([[0.01, 0.02, 0.0, 0.01, 0.0], [np.nan] * 4 + [0.01], [0.0] * 4 + [0.01]], index=d)
    assert detect_bad_dates(r, 0.8, zeros_only=True) == [d[2]]          # missing values are gaps, not a bad date


def test_d9_published_weights_are_used_as_published_and_derived_drift_is_reported(syn):
    from nautilus_fi.kernel.data import DataRequest
    from nautilus_fi.kernel.expr import EvalContext

    p = syn.load(DataRequest(("tr", "alive", "index_weight"), None, None))
    ctx = EvalContext(p, syn.attributes(p.ids), syn.series(["cash"]), syn, syn.catalog())
    W = nfi.Index("SYNTH_IG", drift="STATIC", weight_field="index_weight").weights(ctx)
    pub = np.nan_to_num(p["index_weight"])
    pub = pub / np.where(pub.sum(axis=1, keepdims=True) > 0, pub.sum(axis=1, keepdims=True), 1)
    assert np.abs(W - pub).max() < 1e-7                                 # never frozen between membership changes (published as float32)
    st = _value(benchmark=nfi.Index("SYNTH_IG").exclude(sector="FIN").cap(issuer=0.05)).under("erin_beta_plus", refill="tr")
    assert st.conventions().benchmark_drift.value == "STATIC"


def test_d9_start_labelled_returns_realise_in_the_next_period(syn):
    d = syn.data
    tr = d.panel.frame("tr")
    start = tr.shift(-1)                                            # the same returns, stamped with the period start
    fields = {k: d.panel.frame(k) for k in ("signal.value", "alive", "index_weight", "half_spread_bp")}
    a = nfi.FrameSource({**fields, "tr": tr}, d.attributes, syn._series, {"SYNTH_IG": d.membership})
    b = nfi.FrameSource({**fields, "tr": start}, d.attributes, syn._series, {"SYNTH_IG": d.membership}, info=[nfi.FieldInfo("tr", label="start")])
    ra, rb = run(_value(), a), run(_value(), b)
    assert np.abs(ra.gross - rb.gross).max() < 1e-15


def test_d10_rounding_respects_the_minimum_piece_and_reference_defaults_are_reported():
    from datetime import date

    from nautilus_fi.data.reference import bond_from_row
    from nautilus_fi.instruments.bond import BondSpec

    b = BondSpec("X", "I", date(2020, 1, 1), date(2030, 1, 1), 0.05, min_piece=5_000, increment=2_000)
    assert b.round_face(5_000) == 5_000 and b.round_face(6_500) == 7_000 and b.round_face(4_000) == 0.0 and b.tradable(8_900, "down") == 7_000
    row = {"symbol": "Y", "issuer": "I", "issue_date": "2020-01-01", "maturity": "2030-01-01", "coupon_rate": 0.05, "min_piece": float("nan")}
    spec = bond_from_row(row)
    assert spec.min_piece == 2_000.0 and spec.tags["reference_defaults"] == ["min_piece", "increment"]


def test_d10_proposals_size_at_the_traded_side_and_round_once():
    from datetime import date

    from nautilus_fi.instruments.bond import BondSpec
    from nautilus_fi.propose import build_proposal

    bonds = {"A": BondSpec("A", "I", date(2020, 1, 1), date(2030, 1, 1), 0.05, min_piece=2_000, increment=1_000)}
    prices = pd.DataFrame({"bid": [96.0], "ask": [100.0]}, index=["A"])
    prop = build_proposal({"A": 0.5}, {}, prices, bonds, cash=1_000_000.0, as_of=date(2024, 6, 3), nav=1_000_000.0)
    buy = prop.lines.iloc[0]
    assert buy["price_clean"] == 100.0 and -buy["cash"] <= 500_000.0 and buy["face"] == 500_000.0      # at ask: never more than the target cash
    small = build_proposal({"A": 0.001}, {}, prices, bonds, cash=1_000_000.0, as_of=date(2024, 6, 3), nav=1_000_000.0)
    assert small.lines.empty and "below_min_piece" in set(small.validation["check"])                 # reported, not dropped silently


def test_d10_carry_includes_roll_down_at_constant_yield():
    from datetime import date

    from nautilus_fi.accounting.attribution import AttributionInputs, attribute_instrument

    x = AttributionInputs("A", date(2024, 3, 5), 1e6, 95.0, 95.02, 1.0, 1.0137, 0.0, None, None, None, None, ytm_prev=0.06, carry_days=1)
    comp = attribute_instrument(x)
    assert comp["carry"] == pytest.approx(0.06 * 96.0 / 365.25 * 1e4) and comp["carry"] > (1.0137 - 1.0) * 1e4
    assert sum(comp[k] for k in ("carry", "rates", "spread", "optionality", "fx", "defaults_recoveries", "execution", "financing_fees", "model_change", "residual")) == pytest.approx(comp["total"])


# ====================================================================================== section 4.13
def test_s413_implement_results_fill_the_same_schema():
    sd = nfi.SyntheticSource(n_bonds=40, start="2023-01-03", end="2023-05-31", seed=5)
    st = nfi.Strategy("wide", universe=nfi.Index("SYNTH_IG") & (nfi.Field("oasd") > 1), signal=nfi.Field("oas_bp").zscore(), portfolio=nfi.TopN(8), missing=nfi.ZeroFill(), capital=50e6)
    sess = nfi.Session(source=sd)
    paper, impl = sess.run(st), sess.run(st, fidelity="implement")
    assert {"oasd", "oad"} <= set(impl.exposures.columns) and len(impl.exposures) == len(impl.returns)
    assert (impl.construction["action"] == "rebalance").any()
    assert impl.excess_returns is not None and "excess_annual_return" in impl.stats()
    assert abs(impl.exposures["oasd"].iloc[-1] - paper.exposures["oasd"].iloc[-1]) < 1.0


def test_s7_triggered_field_matches_the_legacy_engine():
    from nautilus_fi.benchmarks import ConstituentBenchmark
    from nautilus_fi.construct import ConstructionSpec
    from nautilus_fi.construct import Quantile as LegacyQuantile
    from nautilus_fi.domain import Conventions, MissingReturnPolicy
    from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
    from nautilus_fi.paper import PaperData, PaperEngine, PaperSettings

    sd = nfi.SyntheticSource(n_bonds=120, start="2015-01-02", end="2015-12-31", seed=6)
    d = sd.data
    trig = nfi.Triggered(0.15, "signal", field="oas_bp", base=nfi.MonthEnd())
    spec = ResearchSpec("legacy", FieldSignal("signal.value"), ConstructionSpec(LegacyQuantile(5)), Conventions().with_(missing_returns=MissingReturnPolicy.ZERO), calendar=trig)
    legacy = PaperEngine(spec, PaperData(d.panel, d.attributes, d.cash, ConstituentBenchmark("IDX", d.membership)), PaperSettings(start="2015-02-02")).run()
    kernel = run(nfi.Strategy.from_spec(spec).with_(benchmark=nfi.Index("SYNTH_IG")), sd, start="2015-02-02")
    skips = (legacy.construction["action"] == "trigger_skip").sum()
    assert skips > 0 and skips == (kernel.construction["action"] == "trigger_skip").sum()
    assert np.abs(legacy.returns - kernel.returns.reindex(legacy.returns.index)).max() < 1e-12
