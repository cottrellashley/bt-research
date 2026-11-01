"""fi-lib-backtesting (``fiq_backtester``): the framework's behaviour as named components over the
kernel, and its parity cases (gap analysis §5.2, LIB-05 to LIB-24).

Conventions: the ``fi_lib_backtesting`` profile (benchmark weights drift between
reconstitutions, round-trip turnover, missing returns zero-filled). The components:

* ``benchmark`` / ``replication``: provider weights snapped at reconstitution (BME + 1 business
  day: the constituent files' dates), drift in between, and the replication check against the
  index's published returns within +/-20bp a year (LIB-05);
* ``constraint``: the constraint grammar: absolute, ratio and offset limits on any metric,
  optionally by group, with a group minimum benchmark weight (LIB-06, LIB-07);
* ``turnover_budget``: an annual turnover budget; sales forced by exits and cash flows are
  reclaimed (outside the budget) (LIB-09);
* ``relaxation``: the relaxation cascade with an elastic fallback, or push to the next day
  (LIB-11);
* ``rules``: the rule-based constructor (LIB-12);
* ``score_drop``: the score-drop trigger with interim rebalances (LIB-14);
* ``futures_returns`` / ``cds_returns``: futures roll and CDS return models (LIB-17, LIB-18);
* ``fx_hedged``: FX-hedged benchmark returns (LIB-19);
* ``external_book``: a live rebalance against an external book (LIB-24).
"""

from __future__ import annotations

from nautilus_fi.adapters.parity import ParityCase, Tolerance

PROFILE = "fi_lib_backtesting"
REQUIREMENTS = ("LIB-05", "LIB-06", "LIB-07", "LIB-09", "LIB-11", "LIB-12", "LIB-14", "LIB-17", "LIB-18", "LIB-19", "LIB-24")


def benchmark(code: str, weight_field: str | None = None, redemptions: str = "pro_rata"):
    """The index with provider weights at each constituent file's date (the reconstitution,
    BME + 1bd) and drifting with returns in between."""
    from nautilus_fi.kernel.benchmark import Index

    return Index(code, drift="DRIFT", weight_field=weight_field, redemptions=redemptions)


def replication(code: str, published, source, tolerance_bp: float = 20.0):
    """The drifted index against its published returns: tracking difference within
    ``tolerance_bp`` a year (the framework's acceptance check)."""
    from nautilus_fi.kernel.benchmark import replication_report

    return replication_report(benchmark(code), published, source, tolerance_bp=tolerance_bp)


KINDS = ("absolute", "ratio", "offset")


def constraint(metric: str, kind: str = "offset", lo: float | None = None, hi: float | None = None, by: str | None = None, min_benchmark_weight: float | None = None, hard: bool = True) -> list:
    """One limit of the grammar on any metric (a field: ``oad``, ``dts``, ``oas_bp`` ...).
    ``absolute``: the portfolio's exposure within [lo, hi]; ``offset``: the active exposure
    (portfolio minus benchmark) within [lo, hi]; ``ratio``: portfolio / benchmark within [lo, hi]
    (so 0.9..1.1 is +/-10%). ``by``: per group (``sector``); ``min_benchmark_weight``: groups
    below it in the index are skipped (LIB-07)."""
    from nautilus_fi.kernel.portfolio import ActiveExposure, Constraint, Exposure

    if kind not in KINDS:
        raise ValueError(f"unknown constraint kind {kind!r}; known: {KINDS}")
    if kind == "absolute":
        m = Exposure(metric, by)
        lo_, hi_ = lo, hi
    elif kind == "offset":
        m = ActiveExposure(metric, by)
        lo_, hi_ = lo, hi
    else:
        m = ActiveExposure(metric, by, relative=True)
        lo_, hi_ = None if lo is None else lo - 1.0, None if hi is None else hi - 1.0
    c = Constraint(m, lo_, hi_, hard=hard)
    if min_benchmark_weight is not None:
        if by is None:
            raise ValueError("min_benchmark_weight applies to group limits: give by=")
        c = c.skip_below(float(min_benchmark_weight))
    return [c]


def group_limit(group: str, lo: float | None = None, hi: float | None = None, min_benchmark_weight: float | None = None, hard: bool = True):
    """Active weight per group (``sector``, ``rating_bucket``) within [lo, hi]; groups below
    ``min_benchmark_weight`` in the index are skipped."""
    from nautilus_fi.kernel.portfolio import ActiveWeight, Constraint

    c = Constraint(ActiveWeight(group), lo, hi, hard=hard)
    return c.skip_below(float(min_benchmark_weight)) if min_benchmark_weight is not None else c


def turnover_budget(annual: float, month_end_share: float = 0.4, events_per_month: int = 3):
    """The annual budget spread over rebalances; forced exits and cash-flow trades are outside it
    (the kernel reports them as forced sells)."""
    from nautilus_fi.kernel.portfolio import AnnualTurnoverBudget

    return AnnualTurnoverBudget(annual, month_end_share, events_per_month)


def relaxation(push: bool = False, tiers: tuple[float, ...] = (1.5, 2.0), elastic: bool = True):
    """Optimiser settings for the cascade: grandfather breaches, widen limits by ``tiers``, then
    soften them (the elastic fallback), each step recorded; ``push``: keep the book and retry the
    next day instead."""
    from nautilus_fi.construct.optimiser import OptimiserSettings

    return OptimiserSettings(on_infeasible="push" if push else "relax", relax_tiers=tuple(tiers), relax_soft=elastic)


def rules(base, rules_):
    """The rule-based constructor: ``base`` adjusted by ordered rules."""
    from nautilus_fi.kernel.portfolio import Rules

    return Rules(base, rules_)


def score_drop(threshold: float, base="BME", interim_portfolio=None, fixed=None):
    """Rebalance between the fixed dates when the book's score has dropped by ``threshold``
    (an interim rebalance, optionally with its own construction)."""
    from nautilus_fi.domain.calendars import Triggered, WithInterim, parse_calendar

    trig = Triggered(threshold, "score_drop", base=parse_calendar(base))
    return WithInterim(parse_calendar(fixed), trig, interim_portfolio) if fixed is not None else trig


def futures_returns(*args, **kw):
    """Continuous futures returns, rolled before first notice (``returns.continuous_futures``)."""
    from nautilus_fi.kernel.returns import continuous_futures

    return continuous_futures(*args, **kw)


def cds_returns(*args, **kw):
    """CDS returns from spreads, risky duration and the running coupon, rolled on IMM dates."""
    from nautilus_fi.kernel.returns import cds_returns as _cds

    return _cds(*args, **kw)


def fx_hedged(*args, **kw):
    """Benchmark returns hedged into the base currency with rolled forwards."""
    from nautilus_fi.kernel.returns import fx_hedged_returns

    return fx_hedged_returns(*args, **kw)


def external_book(strategy, source, as_of, book, prices, bonds, cash, factors=None, unit_size=None, **kw):
    """A live rebalance against an external book held in face, notional, units or market value."""
    from nautilus_fi.propose.live import book_to_face
    from nautilus_fi.propose.proposal import propose_from_strategy

    face = book_to_face(book, factors=factors, unit_size=unit_size, prices=prices)
    return propose_from_strategy(strategy, source, as_of, face, prices, bonds, cash, **kw)


_TOL = (Tolerance("returns", bp_per_year=20.0, min_corr=0.995),)              # the framework's own replication band; agree under §9.2
PARITY = [
    ParityCase("US IG 2015-2025", "fi_lib_backtesting", "audit/returns_audit.md", _TOL, ("2015-01-01", "2025-12-31")),
    ParityCase("US HY 2016-2025", "fi_lib_backtesting", "audit/returns_audit.md", _TOL, ("2016-01-01", "2025-12-31")),
    ParityCase("EM corporates", "fi_lib_backtesting", "backtest_run_tests/run_em_corp_backtest.py", _TOL),
]

__all__ = ["KINDS", "PARITY", "PROFILE", "REQUIREMENTS", "benchmark", "cds_returns", "constraint", "external_book", "futures_returns", "fx_hedged", "group_limit", "relaxation", "replication", "rules", "score_drop", "turnover_budget"]
