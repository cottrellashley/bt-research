"""Market state, calibration recipes, behaviour models, pricing engines and mark policy."""

from datetime import date

import pytest

from nautilus_fi.instruments import BondSpec, CallOption, ZeroCurve
from nautilus_fi.market import (
    CalibrationStatus,
    CurveCalibrator,
    CurveKind,
    CurveRef,
    MarketState,
    MissingMarketData,
)
from nautilus_fi.pricing import (
    DefaultModel,
    DiscountingEngine,
    EngineRegistry,
    EvidenceClassification,
    EvidenceType,
    FourValues,
    HullWhiteTreeEngine,
    IssuerCallPolicy,
    MarkCandidate,
    MarkPolicy,
    MarkSelector,
    PSAModel,
    RefinanceIncentiveModel,
    attribute_by_evidence,
    cpr_to_smm,
)

AS_OF = date(2024, 1, 2)


def test_market_state_lookups_and_failed_curve_is_never_reused():
    ms = MarketState(AS_OF)
    ms.add_curve(CurveRef("USD-OIS", CurveKind.DISCOUNT_OIS, "USD"), ZeroCurve.flat(AS_OF, 0.045, "USD-OIS"))
    ms.add_curve(CurveRef("USD-GOVT", CurveKind.SOVEREIGN, "USD"), ZeroCurve.flat(AS_OF, 0.042, "USD-GOVT"), status="FAILED")
    assert ms.discount("USD").zero(5.0) == pytest.approx(0.045)
    assert ms.projection("SOFR", "USD").name == "USD-OIS"            # explicit single-curve fallback
    with pytest.raises(MissingMarketData):
        ms.sovereign("USD")                                             # failed calibration refuses
    with pytest.raises(MissingMarketData):
        ms.discount("EUR")
    assert ms.requirements_met([CurveRef("x", CurveKind.DISCOUNT_OIS, "EUR")])
    shocked = ms.shocked({"USD-OIS": 50.0})
    assert shocked.discount("USD").zero(5.0) == pytest.approx(0.05) and ms.discount("USD").zero(5.0) == pytest.approx(0.045)


def test_calibration_recipes_and_statuses():
    cal = CurveCalibrator()
    quotes = [{"tenor": t, "quote": q, "source": "UST"} for t, q in zip([0.5, 1, 2, 5, 10, 30], [0.052, 0.05, 0.046, 0.043, 0.044, 0.046])]
    curve, rec = cal.bootstrap_par("GOVT", AS_OF, quotes)
    assert curve is not None and rec.status == CalibrationStatus.OK and rec.max_residual_bp < 1e-6 and rec.elapsed_ms >= 0
    bad, rec2 = cal.bootstrap_par("GOVT", AS_OF, quotes[:1])
    assert bad is None and rec2.status == CalibrationStatus.FAILED
    _, rec3 = cal.fit_nss("GOVT-NSS", AS_OF, quotes + [{"tenor": 7, "quote": 0.12}])   # a wild quote breaks the fit tolerance
    assert rec3.status == CalibrationStatus.FAILED and "residual" in rec3.message
    carried, rec4 = CurveCalibrator.carry_forward(curve, date(2024, 1, 4))
    assert carried is not None and rec4.status == CalibrationStatus.STALE and rec4.age_days == 2
    none, rec5 = CurveCalibrator.carry_forward(curve, date(2024, 1, 20))
    assert none is None and rec5.status == CalibrationStatus.FAILED
    hz, rec6 = cal.credit_from_cds("ACME", AS_OF, [{"tenor": 5, "quote": 150.0}], curve)
    assert rec6.status == CalibrationStatus.OK and 0.02 < hz[5.0] < 0.03


def test_behaviour_models():
    psa = PSAModel(150)
    assert psa.cpr(15) == pytest.approx(0.5 * 0.06 * 1.5) and psa.cpr(60) == pytest.approx(0.09)
    refi = RefinanceIncentiveModel()
    low, high = refi.cpr(36, 0.05, 0.07), refi.cpr(36, 0.07, 0.05)
    assert high > low and high > 0.2 and cpr_to_smm(0.12) == pytest.approx(1 - 0.88 ** (1 / 12))
    assert refi.cpr(36, 0.07, 0.05, burnout=0.8) < high
    pol = IssuerCallPolicy(threshold_bp=25)
    assert pol.exercise(0.06, 0.05, 100.0, 102.0) and not pol.exercise(0.06, 0.059, 100.0, 102.0)
    assert not pol.exercise(0.06, 0.03, 108.0, 104.0, make_whole=True) and pol.exercise(0.06, 0.03, 103.0, 104.0, make_whole=True)
    dm = DefaultModel()
    assert dm.default_probability("BB", 5) == pytest.approx(1 - 0.991 ** 5) and dm.expected_recovery("SUBORDINATED") == 0.25
    assert dm.recovery_date(date(2024, 1, 15)) == date(2025, 1, 15)


def test_discounting_engine_flags_optionality_and_hw_tree_prices_callable():
    curve = ZeroCurve.flat(AS_OF, 0.05)
    bullet = BondSpec("B", "ACME", date(2022, 1, 15), date(2032, 1, 15), 0.05)
    callable_ = BondSpec("C", "ACME", date(2022, 1, 15), date(2032, 1, 15), 0.06, call_schedule=[CallOption(date(2026, 1, 15), 100.0), CallOption(date(2028, 1, 15), 100.0)])
    disc = DiscountingEngine()
    r = disc.price(bullet, date(2024, 1, 16), curve, 0.0)
    assert r.applicable and not r.flags and 95 < r.clean < 105
    rc = disc.price(callable_, date(2024, 1, 16), curve, 0.0)
    assert rc.flags and "oas" in rc.unavailable
    hw = HullWhiteTreeEngine(a=0.03, sigma=0.01, steps_per_year=12)
    assert hw.supports(callable_) and not hw.supports(bullet)
    tree = hw.price(callable_, date(2024, 1, 16), curve, 0.0)
    straight = disc.price(callable_, date(2024, 1, 16), curve, 0.0).clean
    assert tree.clean < straight and tree.diagnostics["option_value"] > 0 and tree.clean > 95
    oas = hw.oas(callable_, date(2024, 1, 16), curve, tree.clean)
    assert abs(oas["oas_bp"]) < 2.0 and 0 < oas["effective_duration"] < 7
    bullet_like = HullWhiteTreeEngine(a=0.03, sigma=0.0001, steps_per_year=12)
    far_call = BondSpec("D", "ACME", date(2022, 1, 15), date(2032, 1, 15), 0.02, call_schedule=[CallOption(date(2031, 12, 1), 150.0)])
    assert abs(bullet_like.price(far_call, date(2024, 1, 16), curve).clean - disc.price(far_call, date(2024, 1, 16), curve).clean) < 0.5
    assert EngineRegistry().choose(callable_).model_id == "hw1f-trinomial" and EngineRegistry().choose(bullet).model_id == "discounting"


def test_mark_policy_selection_and_circularity():
    ts = 1
    cands = [
        MarkCandidate(99.0, EvidenceType.EVALUATED, "BVAL", ts, 98.8, 99.2, quality=0.8, lineage=("BVAL",)),
        MarkCandidate(99.1, EvidenceType.TRANSACTION, "TRACE", ts, size=2_000_000, age_days=0, lineage=("TRACE",)),
        MarkCandidate(93.0, EvidenceType.TRANSACTION, "TRACE", ts, size=25_000, age_days=0, lineage=("TRACE",)),   # small distressed print
        MarkCandidate(99.3, EvidenceType.MODEL, "MATRIX", ts, quality=0.3, age_days=0, lineage=("BVAL", "MATRIX")),
        MarkCandidate(100.5, EvidenceType.COMPOSITE, "CP+", ts, 100.3, 100.7, age_days=4),                            # stale composite
    ]
    sel = MarkSelector(MarkPolicy()).select(cands)
    assert sel.chosen.evidence == EvidenceType.TRANSACTION and sel.chosen.clean == 99.1
    reasons = [r for _, r in sel.excluded]
    assert any("stale" in r for r in reasons) and any("outlier" in r or "small distressed" in r for r in reasons)
    assert sel.dispersion_pct is not None and len(sel.retained) >= 2
    bid_sel = MarkSelector(MarkPolicy(side="POSITION_AWARE", preference=("EVALUATED",))).select(cands, position_sign=1)
    assert bid_sel.chosen.clean == 98.8 and bid_sel.chosen.side == "BID"
    fv = FourValues(signal_input=cands[3], accounting_mark=cands[0], execution_reference=cands[1])
    assert fv.shared_lineage() == {"BVAL"} and fv.circularity_flags()
    ec = EvidenceClassification.from_mark(cands[3], timestamps_real=False, exec_evidence="ASSUMED", product_status="VALIDATED")
    assert ec.valuation_evidence == "MODEL" and ec.temporal_authenticity == "ASSUMED_TIME" and ec.data_completeness == "LOW"
    import pandas as pd

    rows = pd.DataFrame({"evidence": ["TRANSACTION", "MODEL", "MODEL"], "market_value": [60.0, 30.0, 10.0], "pnl": [1.0, 2.0, -1.0]})
    shares = attribute_by_evidence(rows, ("market_value", "pnl"))
    assert shares.loc["MODEL", "market_value_share"] == pytest.approx(0.4) and shares.loc["MODEL", "pnl_share"] == pytest.approx(0.75)
