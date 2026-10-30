"""Research integrity, stress scenarios, invariants, tolerances and report bundle."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nautilus_fi.accounting.postings import StandardAccountingPolicy
from nautilus_fi.core import Event, EventLog, EventSource, EventType, Ledger, Money, TimePoint, to_ns
from nautilus_fi.core.manifest import ExperimentManifest
from nautilus_fi.reports import ReportBundle, explain_order, explain_payment
from nautilus_fi.research import (
    STANDARD_SCENARIOS,
    ExperimentRecord,
    ExperimentRegistry,
    HoldoutViolation,
    Hypothesis,
    StressDimension,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    joint_scenarios,
    min_track_record_length,
    newey_west_sharpe_se,
    probability_of_backtest_overfitting,
    purged_train_indices,
    reverse_stress,
    sharpe_ratio,
    uncertainty_envelope,
    walk_forward_splits,
)
from nautilus_fi.validation import TOLERANCES, InvariantSuite, within


def test_experiment_registry_and_holdout_guard(tmp_path):
    reg = ExperimentRegistry(tmp_path / "registry.json")
    h = Hypothesis("H1", "cheap bonds mean revert", "USD IG", "monthly 16:45", "spread residual", "AGG", 100e6, "issuer<=3%", "1m", "NET_FULL", "net IR < 0", ("2015-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31"))
    reg.register(h)
    reg.record(ExperimentRecord("H1", "abc", {"n_long": 10}, ("2015-01-01", "2019-12-31"), {"sharpe": 0.8}, "COMPLETED"))
    reg.record(ExperimentRecord("H1", "abd", {"n_long": 20}, ("2015-01-01", "2019-12-31"), {"sharpe": 1.1}, "COMPLETED"))
    assert reg.n_trials("H1") == 2
    reg.record(ExperimentRecord("H1", "fin", {"n_long": 20}, ("2022-01-01", "2024-12-31"), {"sharpe": 0.4}, "COMPLETED", note="final"))
    with pytest.raises(HoldoutViolation):
        reg.record(ExperimentRecord("H1", "again", {"n_long": 25}, ("2022-01-01", "2024-12-31"), {"sharpe": 0.9}, "COMPLETED"))
    reloaded = ExperimentRegistry(tmp_path / "registry.json")
    assert len(reloaded.records) == 3 and "H1" in reloaded.holdout_viewed and len(reloaded.variants_frame("H1")) == 3


def test_walk_forward_and_purging():
    splits = walk_forward_splits(date(2015, 1, 1), date(2024, 12, 31), 36, 12, 12, 12)
    assert splits and splits[0].dev_end == date(2018, 1, 1) and splits[0].test_start == date(2019, 1, 1) and splits[-1].test_end <= date(2024, 12, 31)
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    keep = purged_train_indices(dates, pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-30"), label_horizon_days=20, embargo_days=5)
    kept = dates[keep]
    assert kept.max() == dates.max() and not ((kept >= pd.Timestamp("2020-05-12")) & (kept <= pd.Timestamp("2020-07-05"))).any()


def test_dependence_aware_statistics():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.004, 1000))
    sr = sharpe_ratio(r)
    se = newey_west_sharpe_se(r)
    assert 0 < se < 2
    point, lo, hi = block_bootstrap_ci(r, n_boot=200, block_len=10)
    assert lo < point < hi and point == pytest.approx(sr)
    dsr = deflated_sharpe_ratio(1.5, n_trials=50, n_obs=1000, skew=-0.5, kurt=5.0)
    assert 0 <= dsr["dsr"] <= 1 and dsr["expected_max_sr_annual"] > 0 and dsr["dsr"] < deflated_sharpe_ratio(1.5, 1, 1000)["dsr"]
    assert min_track_record_length(1.0) > min_track_record_length(2.0)
    perf = rng.normal(0.0, 0.01, size=(400, 10))
    perf[:, 3] += 0.001           # one genuinely better variant
    res = probability_of_backtest_overfitting(perf, n_splits=4)
    assert 0 <= res["pbo"] <= 1 and res["n_combinations"] == 6
    noise = probability_of_backtest_overfitting(rng.normal(0.0, 0.01, size=(400, 10)), n_splits=4)
    assert noise["pbo"] >= 0.3          # pure noise: in-sample winner is usually not better out of sample


def test_stress_scenarios_reverse_and_envelope():
    liq = next(s for s in STANDARD_SCENARIOS if s.dimension == StressDimension.LIQUIDITY)
    fund = next(s for s in STANDARD_SCENARIOS if s.dimension == StressDimension.FUNDING)
    joint = joint_scenarios(liq, fund)
    assert joint.cost_multiplier == liq.cost_multiplier and joint.haircut_add == fund.haircut_add and joint.repo_rollover_fail
    rs = reverse_stress(lambda lvl: {"drawdown": -0.02 * lvl}, [1, 2, 3, 4, 5], lambda r: r["drawdown"] < -0.05)
    assert rs["threshold"] == 3 and len(rs["path"]) == 5
    env = uncertainty_envelope({"marks": {"mid": 0.05, "bid": 0.04}, "execution": {"x1": 0.05, "x3": 0.01}}, "return")
    assert env.iloc[0]["category"] == "execution"


def test_invariants_tolerances_and_report_bundle(tmp_path):
    suite = InvariantSuite()
    log, ledger = EventLog(), Ledger("USD")
    pol = StandardAccountingPolicy(ledger)
    pol.capital_flow("cap", to_ns("2024-01-02"), Money(1000.0, "USD"))
    ev = log.append(Event(EventType.COUPON_PAYMENT, TimePoint(to_ns("2024-01-03"), to_ns("2024-01-03")), EventSource.LIFECYCLE, "B", payload={"amount": 25.0}))
    pol.coupon_payment(ev.id, to_ns("2024-01-03"), "B", "USD", 25.0)
    suite.duplicate_delivery(log, Event(EventType.COUPON_PAYMENT, TimePoint(to_ns("2024-01-03"), to_ns("2024-01-03")), EventSource.LIFECYCLE, "B", payload={"amount": 25.0}, seq=ev.seq))
    suite.conservation(ledger)
    suite.lot_regrouping(1_000_000.0, 1_000_000.0)
    suite.pinned_replay(lambda: ("abc", {"nav": "h1"}))
    suite.unfilled_orders_do_not_change_ownership(pd.DataFrame({"status": ["EXPIRED"], "filled_qty": [0], "instrument_id": ["B"]}), {"B": 0.0}, {"B": 0.0})
    suite.failed_settlement_not_spendable(100.0, 100.0)
    suite.single_realization(pd.DataFrame({"instrument": ["B"], "date": ["2024-01-03"], "type": ["COUPON"]}), pd.DataFrame({"instrument": ["B"], "date": ["2024-02-03"], "type": ["COUPON"]}))
    suite.payments_reconcile(log, ledger)
    calls = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02", "2024-01-05"]), "order_id": ["a", "b"]})
    state = {"perturbed": False}
    suite.temporal_isolation(lambda: (calls if not state["perturbed"] else calls.assign(order_id=["a", "zz"]), "x"), lambda: state.update(perturbed=True), pd.Timestamp("2024-01-03"))
    assert suite.all_passed, suite.to_frame()
    assert within("cash_posting", 100.004, 100.0, "USD") and not within("cash_posting", 100.01, 100.0, "USD") and within("identifier", 1, 1)
    assert TOLERANCES["oas_bp"].absolute == 5.0
    m = ExperimentManifest(start="2024-01-02", end="2024-06-28", hypothesis="test")
    bundle = ReportBundle(m, performance=pd.DataFrame({"total_return": [0.01]}), headline={"net return": "1.0%"}, unknowns=["execution calibration: assumed (no RFQ history)"])
    out = bundle.export(tmp_path / "run")
    md = (out / "report.md").read_text()
    assert "Unknown / unsupported" in md and "Reproducibility" in md and (out / "manifest.json").exists()
    assert len(explain_order(log, "none")) == 0
    ex = explain_payment(log, ledger, ev.id)
    assert len(ex["lineage"]) == 1 and len(ex["postings"]) == 2
