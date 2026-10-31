"""Benchmarks, signal transforms, the constraint grammar, construction methods and the optimiser."""

import numpy as np
import pandas as pd
import pytest

from nautilus_fi.benchmarks import (
    BenchmarkWeightsUnavailable,
    CompositeBenchmark,
    ConstituentBenchmark,
    CustomBenchmark,
    ReturnSeriesBenchmark,
    cap_weights,
)
from nautilus_fi.construct import (
    BenchmarkPlusTilts,
    ConstraintSpec,
    ConstructionSpec,
    CrossSection,
    CVaRTerm,
    EqualWeight,
    HedgeOverlay,
    Objective,
    Optimise,
    OptimiserSettings,
    PairsSizing,
    Quantile,
    RankWeights,
    RiskModel,
    TrackingTerm,
    check_constraints,
    neutralise,
    optimise,
    parse_constraints,
    zscore,
)
from nautilus_fi.domain import WeightDrift
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


@pytest.fixture(scope="module")
def synth():
    return generate_synthetic_panel(SyntheticPanelConfig(n_bonds=300, start="2015-01-02", end="2016-06-30", seed=5))


def _cs(n=60, seed=1, current=None):
    rng = np.random.default_rng(seed)
    ids = pd.Index([f"B{i}" for i in range(n)])
    data = pd.DataFrame({"sector": rng.choice(["FIN", "IND", "UTL"], n), "issuer": [f"I{i // 2}" for i in range(n)], "oasd": rng.uniform(1, 10, n), "oas_bp": rng.uniform(50, 300, n)}, index=ids)
    data["dts"] = data["oasd"] * data["oas_bp"]
    wb = pd.Series(rng.uniform(0.5, 1.5, n), index=ids)
    wb /= wb.sum()
    return CrossSection(pd.Timestamp("2020-01-31"), ids, data, current if current is not None else pd.Series(0.0, index=ids), wb, pd.Series(rng.normal(size=n), index=ids), cost_bp=pd.Series(10.0, index=ids))


# ------------------------------------------------------------------ benchmarks
def test_constituent_benchmark_reconstructs_the_index(synth):
    b = ConstituentBenchmark("IDX", synth.membership, WeightDrift.DRIFT)
    assert (b.returns(synth.panel) - synth.index["tr"]).abs().max() < 1e-12
    assert np.abs(b.weight_matrix(synth.panel) - synth.panel["index_weight"]).max() < 1e-6
    static = ConstituentBenchmark("IDXS", synth.membership, WeightDrift.STATIC).returns(synth.panel)
    assert (static - b.returns(synth.panel)).abs().max() > 0                   # the drift convention matters
    d0, d1 = synth.membership["date"].unique()[2:4]
    assert b.removals(synth.panel, d0, d1) == b.members(synth.panel, d0) - b.members(synth.panel, d1)
    ex = b.exposures(synth.panel, d1, ["oasd", "dts"])
    assert ex["oasd"] > 0 and ex["dts"] > ex["oasd"]


def test_custom_capped_and_composite_benchmarks(synth):
    parent = ConstituentBenchmark("IDX", synth.membership)
    c = CustomBenchmark("EX-FIN 3-10y capped", parent, synth.attributes, exclude={"sector": ["FIN"]}, field_rules=[("years_to_maturity", ">=", 3)], issuer_cap=0.03)
    cf = c.constituents_frame(synth.panel)
    assert not synth.attributes.loc[cf["instrument_id"], "sector"].eq("FIN").any()
    iss = cf.assign(issuer=synth.attributes.loc[cf["instrument_id"], "issuer"].to_numpy()).groupby(["date", "issuer"])["weight"].sum()
    assert iss.max() <= 0.03 + 1e-9 and cf.groupby("date")["weight"].sum().round(9).eq(1.0).all()
    assert c.rules()["issuer_cap"] == 0.03
    comp = CompositeBenchmark("60/40", [(parent, 0.6), (c, 0.4)])
    r = comp.returns(synth.panel)
    assert r.abs().sum() > 0 and comp.weight_matrix(synth.panel).sum(axis=1)[5] == pytest.approx(1.0)
    rs = ReturnSeriesBenchmark("agg", synth.index["tr"])
    assert rs.returns(synth.panel).equals(synth.index["tr"].reindex(synth.panel.dates))
    with pytest.raises(BenchmarkWeightsUnavailable):
        rs.weight_matrix(synth.panel)


def test_cap_weights():
    w = pd.Series([0.5, 0.3, 0.1, 0.1], index=list("ABCD"))
    g = pd.Series(["X", "X", "Y", "Z"], index=w.index)
    capped = cap_weights(w, g, 0.4)
    assert capped.groupby(g).sum().max() <= 0.4 + 1e-12 and capped.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cap_weights(w, g, 0.2)


# ------------------------------------------------------------------ signals and grammar
def test_signal_transforms():
    x = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], index=list("abcdef"))
    g = pd.Series(["A", "A", "A", "B", "B", "B"], index=x.index)
    z = zscore(x, g)
    assert z.groupby(g).mean().abs().max() < 1e-12
    n = neutralise(x, groups=g)
    assert abs(n.groupby(g).mean()).max() < 1e-9
    with pytest.raises(ValueError):
        ConstraintSpec("bogus")
    with pytest.raises(ValueError):
        ConstraintSpec("group")
    assert parse_constraints([{"type": "weight", "max": 0.05}])[0].kind == "weight"


# ------------------------------------------------------------------ optimiser
def test_lp_respects_bands_and_reports_binding():
    cs = _cs()
    cons = [{"kind": "active_group", "attribute": "sector", "min": -0.02, "max": 0.02}, {"kind": "active_exposure", "attribute": "oasd", "min": -0.1, "max": 0.1}, {"kind": "weight", "max": 0.05}]
    r = optimise(cs, Objective(alpha=0.001, tracking=TrackingTerm(("oasd", "dts"), ("sector",))), cons)
    assert r.ok and r.weights.sum() == pytest.approx(1.0) and r.weights.max() <= 0.05 + 1e-9
    assert not check_constraints(r.weights, cs, parse_constraints(cons))
    assert r.binding and r.duals and r.turnover["ONE_WAY_BUYS"] == pytest.approx(1.0)


def test_turnover_budget_and_initial_build_exemption():
    cs = _cs()
    base = optimise(cs, Objective(alpha=0.001), [{"kind": "weight", "max": 0.05}, {"kind": "turnover", "max": 0.05}])
    assert base.ok and base.turnover["ONE_WAY_BUYS"] == pytest.approx(1.0)              # building from cash is not limited
    assert any("initial build" in r["reason"] for r in base.relaxations)
    nxt = optimise(_cs(seed=1, current=base.weights).__class__(cs.date, cs.ids, cs.data, base.weights, cs.benchmark, pd.Series(np.random.default_rng(9).normal(size=len(cs.ids)), index=cs.ids), cost_bp=cs.cost_bp), Objective(alpha=0.01), [{"kind": "weight", "max": 0.05}, {"kind": "turnover", "max": 0.05}])
    assert nxt.turnover["HALF_SUM"] <= 0.05 + 1e-7


@pytest.mark.parametrize("strategy", ["full", "lp_round"])
def test_mixed_integer_min_holding_and_cardinality(strategy):
    cs = _cs(n=80)
    r = optimise(cs, Objective(alpha=0.001, tracking=TrackingTerm(("oasd", "dts"), ("sector",))), [{"kind": "weight", "max": 0.06}, {"kind": "cardinality", "max": 25}, {"kind": "min_holding", "min": 0.01}], settings=OptimiserSettings(mip_strategy=strategy))
    held = r.weights[r.weights > 1e-12]
    assert r.ok and len(held) <= 25 and held.min() >= 0.01 - 1e-9 and r.weights.sum() == pytest.approx(1.0)


def test_qp_risk_model_lowers_tracking_error():
    cs = _cs()
    L = pd.DataFrame({"dur": cs.data["oasd"] / 10, "fin": (cs.data["sector"] == "FIN").astype(float)}, index=cs.ids)
    F = pd.DataFrame([[4e-4, 1e-4], [1e-4, 2e-4]], index=L.columns, columns=L.columns)
    rm = RiskModel(L, F, pd.Series(1e-4, index=cs.ids), aversion=2000.0)
    loose = optimise(cs, Objective(alpha=0.001), [{"kind": "weight", "max": 0.05}])
    tight = optimise(cs, Objective(alpha=0.001, risk=rm), [{"kind": "weight", "max": 0.05}])
    from nautilus_fi.construct.optimiser import _variance

    assert tight.ok and tight.problem["quadratic"]
    assert _variance(rm, cs.ids, (tight.weights - cs.benchmark).to_numpy()) < _variance(rm, cs.ids, (loose.weights - cs.benchmark).to_numpy())


def test_cvar_soft_and_infeasible():
    cs = _cs()
    cs.scenarios = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (150, len(cs.ids))), columns=cs.ids)
    r = optimise(cs, Objective(alpha=0.001, cvar=CVaRTerm(0.95, 1.0)), [{"kind": "weight", "max": 0.05}, {"kind": "cvar", "max": 0.01}])
    assert r.ok and "cvar" in r.components
    soft = optimise(cs, Objective(alpha=1.0), [{"kind": "weight", "max": 0.05}, {"kind": "active_exposure", "attribute": "oasd", "min": -1e-6, "max": 1e-6, "hard": False, "penalty": 1e-4}])
    assert soft.ok and soft.relaxations
    impossible = [{"kind": "weight", "max": 0.001}]
    keep = optimise(cs, Objective(), impossible, settings=OptimiserSettings(on_infeasible="keep"))
    assert not keep.ok and keep.weights.equals(cs.current)
    conflict = optimise(cs, Objective(), [{"kind": "weight", "max": 0.05}, {"kind": "active_exposure", "attribute": "oasd", "min": 50.0}])
    assert conflict.ok and conflict.message.startswith("relaxed")


# ------------------------------------------------------------------ methods
def test_construction_methods():
    cs = _cs(n=50)
    q = ConstructionSpec(Quantile(5)).build(cs)
    assert len(q.weights) == 10 and q.weights.sum() == pytest.approx(1.0)
    ls = ConstructionSpec(Quantile(5, short_bin=1, invested=0.5), long_only=False).build(cs)
    assert ls.weights.sum() == pytest.approx(0.0) and (ls.weights < 0).sum() == 10
    by = Quantile(5, by="sector").bins(cs)
    assert by.groupby(cs.data.loc[by.index, "sector"]).nunique().min() >= 3
    rw = ConstructionSpec(RankWeights(top=20)).build(cs)
    assert rw.weights.sum() == pytest.approx(1.0) and rw.weights.idxmax() == cs.scores.idxmax()
    tilt = ConstructionSpec(BenchmarkPlusTilts(1.0, max_active=0.002)).build(cs)
    assert (tilt.weights - cs.benchmark).abs().max() <= 0.002 + 1e-6
    eq = ConstructionSpec(EqualWeight(top=10), [{"kind": "weight", "max": 0.05}], issuer_cap=0.15).build(cs)
    assert eq.diagnostics["n_hard_violations"] == len([v for v in eq.violations if v["severity"] == "HARD"]) and eq.violations      # 10% names breach a 5% max: recorded
    pairs = ConstructionSpec(PairsSizing([("B0", "B1"), ("B2", "B3")], "dts", 0.1), long_only=False).build(cs)
    dts = cs.data["dts"]
    assert abs(pairs.weights["B0"] * dts["B0"] + pairs.weights["B1"] * dts["B1"]) < 1e-9
    opt = ConstructionSpec(Optimise(Objective(alpha=0.001)), [{"kind": "weight", "max": 0.05}]).build(cs)
    assert opt.ok and opt.diagnostics["optimiser"]["status"] == "Optimal"


def test_hedge_overlay_neutralises_duration():
    cs = _cs(n=40)
    spec = ConstructionSpec(EqualWeight(top=10), overlay=HedgeOverlay(("TY",), ("oasd",), {"oasd": 0.0}))
    out = spec.build(cs, pd.DataFrame({"oasd": [6.5]}, index=["TY"]))
    assert out.diagnostics["overlay"]["exposure_after"]["oasd"] == pytest.approx(0.0, abs=1e-9) and out.overlays["TY"] < 0
