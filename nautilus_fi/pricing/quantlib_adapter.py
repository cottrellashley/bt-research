"""Optional QuantLib bridge for analytics the pure-Python layer does not attempt.

* callable bond OAS / effective duration with a Hull-White lattice (``callable_bond_oas``)
* piecewise log-linear discount curve bootstrapping from deposit + bond helpers (``ql_bootstrap``)

QuantLib is an optional dependency (``pip install nautilus-fi[quantlib]``). Everything else in the
package works without it.
"""

from __future__ import annotations

from datetime import date

try:  # pragma: no cover - import guard
    import QuantLib as ql

    HAVE_QUANTLIB = True
except Exception:  # pragma: no cover
    ql = None
    HAVE_QUANTLIB = False

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.curves import ZeroCurve
from nautilus_fi.instruments.enums import DayCount, Frequency


def _require() -> None:
    if not HAVE_QUANTLIB:
        raise ImportError("QuantLib is not installed; pip install QuantLib (or nautilus-fi[quantlib])")


def _ql_date(d: date):
    return ql.Date(d.day, d.month, d.year)


def _ql_daycount(dc: DayCount):
    return {
        DayCount.THIRTY_360: ql.Thirty360(ql.Thirty360.USA),
        DayCount.THIRTY_E_360: ql.Thirty360(ql.Thirty360.European),
        DayCount.ACT_ACT_ICMA: ql.ActualActual(ql.ActualActual.ISMA),
        DayCount.ACT_ACT_ISDA: ql.ActualActual(ql.ActualActual.ISDA),
        DayCount.ACT_360: ql.Actual360(),
        DayCount.ACT_365F: ql.Actual365Fixed(),
    }[dc]


def _ql_frequency(f: Frequency):
    return {Frequency.ANNUAL: ql.Annual, Frequency.SEMI_ANNUAL: ql.Semiannual, Frequency.QUARTERLY: ql.Quarterly, Frequency.MONTHLY: ql.Monthly, Frequency.ZERO: ql.Once}[f]


def ql_curve(curve: ZeroCurve):
    """Convert a ``ZeroCurve`` (continuous zeros) into a QuantLib ``YieldTermStructureHandle``."""
    _require()
    today = _ql_date(curve.as_of)
    ql.Settings.instance().evaluationDate = today
    dates = [today] + [today + ql.Period(int(round(t * 365.25)), ql.Days) for t in curve.tenors]
    rates = [float(curve.zero_rates[0])] + [float(z) for z in curve.zero_rates]
    ts = ql.ZeroCurve(dates, rates, ql.Actual365Fixed(), ql.NullCalendar(), ql.Linear(), ql.Continuous, ql.Annual)
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)


def callable_bond_oas(bond: BondSpec, settle: date, clean_price: float, curve: ZeroCurve, hw_sigma: float = 0.01, hw_a: float = 0.03, grid_steps: int = 200) -> dict:
    """OAS (bp), effective duration and convexity for a callable fixed-rate bond via Hull-White tree."""
    _require()
    if bond.maturity is None:
        raise ValueError("perpetual callable bonds not supported by this adapter")
    handle = ql_curve(curve.with_as_of(settle))
    ql.Settings.instance().evaluationDate = _ql_date(settle)
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    schedule = ql.Schedule(_ql_date(bond.dated_date), _ql_date(bond.maturity), ql.Period(_ql_frequency(bond.frequency)), cal, ql.Following, ql.Following, ql.DateGeneration.Backward, bond.end_of_month)
    call_sched = ql.CallabilitySchedule()
    for c in bond.call_schedule:
        if c.date > settle:
            call_sched.append(ql.Callability(ql.BondPrice(c.price, ql.BondPrice.Clean), ql.Callability.Call, _ql_date(c.date)))
    cb = ql.CallableFixedRateBond(bond.settlement_lag(), 100.0, schedule, [bond.coupon_rate], _ql_daycount(bond.day_count), ql.Following, 100.0, _ql_date(bond.issue_date), call_sched)
    model = ql.HullWhite(handle, hw_a, hw_sigma)
    engine = ql.TreeCallableFixedRateBondEngine(model, grid_steps, handle)
    cb.setPricingEngine(engine)
    oas = cb.OAS(clean_price, handle, _ql_daycount(bond.day_count), ql.Compounded, _ql_frequency(bond.frequency))
    eff_dur = cb.effectiveDuration(oas, handle, _ql_daycount(bond.day_count), ql.Compounded, _ql_frequency(bond.frequency))
    eff_cvx = cb.effectiveConvexity(oas, handle, _ql_daycount(bond.day_count), ql.Compounded, _ql_frequency(bond.frequency))
    return {"oas_bp": oas * 1e4, "effective_duration": eff_dur, "effective_convexity": eff_cvx, "model_clean_price": cb.cleanPrice()}


__all__ = ["HAVE_QUANTLIB", "callable_bond_oas", "ql_curve"]
