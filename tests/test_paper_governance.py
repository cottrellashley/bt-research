"""Panel engine (Paper), signal evaluation (Evaluate), governance (ledger, walk-forward, records)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nautilus_fi.benchmarks import ConstituentBenchmark
from nautilus_fi.construct import ConstructionSpec, EqualWeight, HedgeOverlay, Quantile
from nautilus_fi.data.panel import Panel
from nautilus_fi.domain import Conventions, MissingReturnPolicy, MissingReturnsError, Timing, Triggered
from nautilus_fi.domain.spec import Decider, FieldSignal, ResearchSpec, Target, UniverseRule
from nautilus_fi.governance import GateFailure, RecordLevel, RecordRegistry, TrialLedger, walk_forward
from nautilus_fi.paper import PaperData, PaperEngine, PaperSettings
from nautilus_fi.paper.evaluate import EvaluationSettings, evaluate_signal
from nautilus_fi.research import HoldoutViolation, Hypothesis, walk_forward_splits
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


@pytest.fixture(scope="module")
def synth():
    return generate_synthetic_panel(SyntheticPanelConfig(n_bonds=250, start="2015-01-02", end="2017-12-29", seed=8, signal_noise=1.5))


def _data(s, **kw):
    return PaperData(s.panel, s.attributes, s.cash, ConstituentBenchmark("IDX", s.membership), s.overlays, **kw)


class Replicate(Decider):
    def decide(self, view):
        return Target(weights=view.benchmark)


class Oracle(Decider):
    """Weights proportional to the return of the *next* row: only usable without a trade lag."""

    def __init__(self, panel):
        self.panel = panel

    def decide(self, view):
        i = self.panel.date_pos(view.date)
        if i + 1 >= len(self.panel.dates):
            return Target(skip=True)
        nxt = pd.Series(self.panel["tr"][i + 1], index=self.panel.ids).reindex(view.universe).dropna()
        top = nxt.sort_values().index[-10:]
        return Target(weights=pd.Series(0.1, index=top))


def test_replication_tracks_the_index_exactly_without_costs(synth):
    res = PaperEngine(ResearchSpec("rep", Replicate(), conventions=Conventions(timing=Timing(1))), _data(synth, cost_field=None), PaperSettings(default_cost_bp=0.0)).run()
    assert (res.returns - res.benchmark_returns).iloc[2:].abs().max() < 1e-9
    costly = PaperEngine(ResearchSpec("rep", Replicate(), conventions=Conventions(timing=Timing(1))), _data(synth)).run()
    assert costly.costs.sum() > 0 and costly.nav.iloc[-1] < res.nav.iloc[-1]


def test_timing_lag_prevents_lookahead(synth):
    """A decision can never earn the return of its own information date, and with a two-day lag an
    oracle for tomorrow's return is worthless."""
    d = _data(synth, cost_field=None)
    lag1 = PaperEngine(ResearchSpec("o1", Oracle(synth.panel), conventions=Conventions(timing=Timing(1)), calendar="D"), d).run()
    lag2 = PaperEngine(ResearchSpec("o2", Oracle(synth.panel), conventions=Conventions(timing=Timing(2)), calendar="D"), d).run()
    assert lag1.stats()["annual_return"] > 0.5 and lag2.stats()["annual_return"] < lag1.stats()["annual_return"] / 5


def test_quintile_run_records_turnover_costs_exposures(synth):
    spec = ResearchSpec("q5", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(Quantile(5)), Conventions(missing_returns=MissingReturnPolicy.ZERO), universe=UniverseRule(field_rules=[("oasd", ">", 1.0)]))
    res = PaperEngine(spec, _data(synth)).run()
    s = res.stats()
    assert s["n_rebalances"] == 36 and s["turnover_annual"] > 0 and s["cost_drag_annual"] > 0
    assert {"ONE_WAY_BUYS", "HALF_SUM", "TWO_WAY"} <= set(res.trades.columns)
    assert "active_oasd" in res.exposures and res.excess("ME").notna().all()
    assert res.spec_hash == spec.hash and len(res.construction) == 37      # day one + 36 month ends; the last cannot trade inside the window


def test_missing_returns_policies_and_redemptions():
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=120, start="2015-01-02", end="2015-12-31", seed=2, missing_rate=0.01, n_bad_dates=2))
    spec = ResearchSpec("rep", Replicate(), conventions=Conventions(timing=Timing(1)))
    with pytest.raises(MissingReturnsError):
        PaperEngine(spec, _data(s)).run()                                                # FAIL is the canonical default
    z = PaperEngine(ResearchSpec("rep", Replicate(), conventions=Conventions(timing=Timing(1), missing_returns=MissingReturnPolicy.ZERO)), _data(s)).run()
    assert z.missing["zero_filled"] > 0
    s.panel.with_field("tr_vendor_b", s.true_tr)
    r = PaperEngine(ResearchSpec("rep", Replicate(), conventions=Conventions(timing=Timing(1), missing_returns=MissingReturnPolicy.REFILL)), _data(s, refill_field="tr_vendor_b")).run()
    assert r.missing["refilled"] > 0 and len(r.missing["bad_dates"]) == 2


def test_leverage_shorts_and_overlays(synth):
    class LongShort(Decider):
        def decide(self, view):
            sc = view.field("signal.value").reindex(view.universe).dropna().sort_values()
            w = pd.Series(0.0, index=sc.index)
            w[sc.index[-20:]] = 0.075
            w[sc.index[:10]] = -0.05
            return Target(weights=w, overlays=pd.Series({"TY": -0.5}))
    res = PaperEngine(ResearchSpec("ls", LongShort()), _data(synth), PaperSettings(borrow_fee_bp=50.0)).run()
    assert res.daily["gross_exposure"].max() > 1.5 and res.daily["cash_weight"].min() < 0.0 + 1.0
    assert not res.overlays.empty and (res.overlays["notional"] < 0).any()
    hedged = ResearchSpec("hedged", FieldSignal("signal.value"), ConstructionSpec(EqualWeight(top=20), overlay=HedgeOverlay(("TY",), ("oad",), {"oad": 0.0})))
    h = PaperEngine(hedged, _data(synth)).run()
    unhedged = PaperEngine(ResearchSpec("u", FieldSignal("signal.value"), ConstructionSpec(EqualWeight(top=20))), _data(synth)).run()
    rates = synth.treasury["r_10"].reindex(h.returns.index)
    assert abs(h.returns.corr(rates)) < abs(unhedged.returns.corr(rates))


def test_triggered_calendar_skips_small_changes(synth):
    spec = ResearchSpec("trig", FieldSignal("signal.value"), ConstructionSpec(EqualWeight(top=30)), calendar=Triggered(threshold=0.6, measure="turnover"))
    res = PaperEngine(spec, _data(synth)).run()
    assert (res.construction["action"] == "trigger_skip").any() and (res.construction["action"] == "rebalance").any()


def test_signal_evaluation(synth):
    rep = evaluate_signal(synth.panel, "signal.value", Conventions(), EvaluationSettings(horizons=(5, 21), neutralise_groups=("sector",), neutralise_exposures=("oasd",)), synth.attributes)
    s = rep.summary()
    assert s.loc[21, "mean_ic"] > 0.05 and s.loc[21, "ic_tstat_nw"] > 3 and s.loc[21, "monotonicity"] > 0.5 and "mean_ic_neutral" in s
    synth.panel.with_field("noise", np.random.default_rng(0).normal(size=synth.panel["tr"].shape))
    assert abs(evaluate_signal(synth.panel, "noise", Conventions(), EvaluationSettings(horizons=(21,))).summary().loc[21, "mean_ic"]) < 0.03
    oracle = pd.DataFrame(np.vstack([synth.panel["xr"][1:], np.full((1, len(synth.panel.ids)), np.nan)]), index=synth.panel.dates, columns=synth.panel.ids)
    lag2 = evaluate_signal(synth.panel, oracle, Conventions(timing=Timing(2)), EvaluationSettings(horizons=(1,)))
    lag1 = evaluate_signal(synth.panel, oracle, Conventions(timing=Timing(1)), EvaluationSettings(horizons=(1,)))
    assert lag1.summary().loc[1, "mean_ic"] > 0.9 and abs(lag2.summary().loc[1, "mean_ic"]) < 0.2   # forward windows start after the trade


# ------------------------------------------------------------------ governance
def _hyp(hid="H1"):
    return Hypothesis(hid, "idiosyncratic spreads mean-revert", "IG", "BME", "signal.value", "IDX", 100e6, "+/-2% sectors", "1m", "half spread", "IR < 0.3", ("2015-01-01", "2016-12-31"), ("2017-01-01", "2017-06-30"), ("2017-07-01", "2017-12-31"))


def test_ledger_dsr_pbo_and_holdout(tmp_path, synth):
    led = TrialLedger(tmp_path)
    with pytest.raises(KeyError):
        led.record("PAPER", "x", "h", {}, ("2015-01-01", "2015-12-31"), {}, hypothesis_id="UNREGISTERED")
    led.register(_hyp())
    with pytest.raises(ValueError):
        led.register(_hyp())
    data = _data(synth)
    for k in (3, 5, 10):
        spec = ResearchSpec(f"q{k}", FieldSignal("signal.value"), ConstructionSpec(Quantile(k)), hypothesis_id="H1", params={"bins": k})
        PaperEngine(spec, data, PaperSettings(end="2016-12-31")).run(ledger=led)
    assert led.n_trials("H1") == 3 and led.sharpe_variance("H1") is not None
    dsr = led.deflated_sharpe("H1", pd.Series(np.random.default_rng(1).normal(0.0004, 0.003, 500)))
    assert dsr["var_trial_sr_source"] == "supplied" and dsr["ledger_trials"] == 3
    assert 0 <= led.pbo("H1", n_splits=4)["pbo"] <= 1
    spec = ResearchSpec("q5", FieldSignal("signal.value"), ConstructionSpec(Quantile(5)), hypothesis_id="H1")
    PaperEngine(spec, data).run(ledger=led)                                              # covers the holdout: viewed now
    with pytest.raises(HoldoutViolation):
        PaperEngine(spec, data).run(ledger=led)
    assert led.holdout_viewed("H1") and len(led.trials("H1")) == 4


def test_walk_forward_records_every_trial(tmp_path, synth):
    led = TrialLedger(tmp_path)
    h = _hyp("WF")
    h.holdout_period = ("2030-01-01", "2030-12-31")
    led.register(h)
    mk = lambda p: ResearchSpec(f"q{p['bins']}", FieldSignal("signal.value"), ConstructionSpec(Quantile(p["bins"])), params=p)
    splits = walk_forward_splits(date(2015, 1, 1), date(2017, 12, 31), 12, 6, 6, 6)
    wf = walk_forward(mk, [{"bins": 3}, {"bins": 5}], _data(synth), splits, label_horizon_days=21, embargo_days=5, ledger=led, hypothesis_id="WF")
    assert len(wf.folds) == len(splits) and (wf.folds["purged_val"] > 0).all() and len(wf.oos_returns) > 100
    stages = led.trials("WF")["stage"].value_counts()
    assert stages["VALIDATION"] == 2 * len(splits) and stages["OOS"] == len(splits)


def test_record_gates_and_hash_chain(tmp_path, synth):
    led = TrialLedger(tmp_path / "ledger")
    h = _hyp("REC")
    h.holdout_period = ("2030-01-01", "2030-12-31")
    led.register(h)
    spec = ResearchSpec("q5", FieldSignal("signal.value"), ConstructionSpec(Quantile(5)), hypothesis_id="REC", conventions=Conventions(missing_returns=MissingReturnPolicy.ZERO))
    res = PaperEngine(spec, _data(synth)).run(ledger=led)
    reg = RecordRegistry(tmp_path / "records")
    rec = reg.submit(RecordLevel.EXPLORATORY, res, spec, led)
    assert rec.level == "EXPLORATORY" and (tmp_path / "records" / f"{rec.record_id}.md").exists()
    cand = reg.submit(RecordLevel.CANDIDATE, res, spec, led)
    assert cand.checks["trial_in_ledger"]["ok"] and cand.prev_hash == rec.hash
    with pytest.raises(GateFailure) as e:
        reg.submit(RecordLevel.OFFICIAL, res, spec, led)
    assert {"implement_completed", "reconciled"} <= set(e.value.failed)
    unconfirmed = ResearchSpec("q5u", FieldSignal("signal.value"), ConstructionSpec(Quantile(5)), hypothesis_id="REC", conventions=Conventions.profile("paperport").with_(missing_returns=MissingReturnPolicy.ZERO))
    r2 = PaperEngine(unconfirmed, _data(synth)).run(ledger=led)
    with pytest.raises(GateFailure) as e2:
        reg.submit(RecordLevel.CANDIDATE, r2, unconfirmed, led)
    assert "conventions_confirmed" in e2.value.failed
    assert reg.verify_chain() and len(reg.records()) == 2


def test_panel_without_alive_field_treats_trailing_nans_as_redemptions():
    dates = pd.bdate_range("2024-01-01", periods=6)
    tr = np.array([[np.nan, np.nan], [0.01, 0.01], [0.01, 0.01], [0.01, np.nan], [0.01, np.nan], [0.01, np.nan]])
    p = Panel(dates, pd.Index(["A", "B"]), {"tr": tr})

    class Both(Decider):
        def decide(self, view):
            return Target(weights=pd.Series(0.5, index=["A", "B"]))
    res = PaperEngine(ResearchSpec("both", Both(), conventions=Conventions(timing=Timing(1)), calendar="BME"), PaperData(p, alive_field=None, cost_field=None, excess_field=None, treasury_field=None)).run()
    assert res.missing["redeemed"] == 1 and res.daily["cash_weight"].iloc[-1] > 0.4
