"""The agent-facing API: a handful of functions over a default session.

::

    import nautilus_fi as nfi

    nfi.session("~/research/credit-value", source=company_data)    # where trials and results live
    nfi.catalog().search("spread duration")                          # what data exist
    base = nfi.Strategy("value", universe=nfi.Index("LUACTRUU"), signal=nfi.Field("oas_bp").zscore(by="sector"),
                        portfolio=nfi.TopN(100), benchmark=nfi.Index("LUACTRUU"))
    idea = base.with_(signal=base.signal.neutralise("oasd"), name="value_dur_neutral")
    cmp = nfi.compare(base, idea, window=("2015", "2024"))
    print(cmp)                    # side by side
    print(cmp.explain())          # what changed, by how much, whether it is noise, where it came from
    sw = nfi.sweep(base, {"portfolio.n": [50, 100, 200]})
    sw.selection()                # best variant, deflated by every trial the session ran
    nfi.promote(cmp[1], "CANDIDATE")

Every run goes through the session, so the trial count behind any statistic is the real one.
``vocabulary()`` lists the building blocks in one page (what an agent puts in its tool prompt).
"""

from __future__ import annotations

from nautilus_fi.kernel.data import as_source
from nautilus_fi.kernel.session import Session

_DEFAULT: Session | None = None


def session(root=None, source=None, name: str | None = None, hypothesis=None, start=None, end=None, window=None) -> Session:
    """Start (and make default) a research session. ``root``: where its ledger, records and result
    cache persist (None: a temporary directory); ``source``: its default data source."""
    global _DEFAULT
    _DEFAULT = Session(root, source, name, hypothesis, start, end, window)
    return _DEFAULT


def current_session() -> Session:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Session()
    return _DEFAULT


def _sess(s: Session | None, source) -> Session:
    sess = s or current_session()
    if source is not None and sess.source is None:
        sess.source = as_source(source)
    return sess


def catalog(source=None):
    """What the source holds (searchable: ``catalog().search("oas")``)."""
    src = as_source(source) if source is not None else current_session()._source(None)
    return src.catalog()


def run(strategy, source=None, start=None, end=None, window=None, fidelity: str = "paper", session: Session | None = None, **kw):
    """Run one strategy (Paper by default; ``fidelity="implement"`` for the event engine)."""
    return _sess(session, source).run(strategy, start, end, window, fidelity, source, **kw)


def compare(*strategies, source=None, start=None, end=None, window=None, fidelity: str = "paper", session: Session | None = None):
    """Run strategies on shared data and line them up; the first is the base for ``explain``."""
    return _sess(session, source).compare(*strategies, start=start, end=end, window=window, fidelity=fidelity, source=source)


def sweep(base, grid=None, source=None, start=None, end=None, window=None, metric: str = "sharpe", workers: int = 1, session: Session | None = None):
    """Run variants (``{name: strategy}``, a base plus ``{"path": values}``, or a builder plus
    ``{"kw": values}``); ``workers > 1`` uses processes on a memory-mapped data snapshot."""
    return _sess(session, source).sweep(base, grid, metric, start, end, window, source, workers)


def explain(variant, base):
    """Why result ``variant`` differs from result ``base``."""
    return variant.explain(base)


def promote(result, level: str = "CANDIDATE", implement=None, session: Session | None = None):
    """Submit a result through the official-record gates (raises ``GateFailure`` with every failed check)."""
    return (session or current_session()).promote(result, level, implement)


def evaluate(signal, source=None, universe=None, start=None, end=None, window=None, horizons=(1, 5, 21, 63), schedule: str = "BME", returns: str = "xr", lag: int = 1, quantiles: int = 5, weighting: str = "equal", neutralise_groups: tuple = (), session: Session | None = None, name: str | None = None):
    """Does the signal rank instruments by their forward returns? Rank IC per horizon, IC decay,
    quantile returns, Newey-West t-statistics and bootstrap intervals (``paper.evaluate``), on the
    signal and universe as kernel expressions, forward returns starting after ``lag`` trading days.
    Recorded in the session's ledger as an EVALUATE trial."""
    import numpy as np
    import pandas as pd

    from nautilus_fi.data.panel import Panel
    from nautilus_fi.domain.conventions import Conventions, Timing
    from nautilus_fi.kernel.engine import prepare
    from nautilus_fi.kernel.expr import Field, alive_mask
    from nautilus_fi.kernel.mechanics import NextClose
    from nautilus_fi.kernel.strategy import Strategy
    from nautilus_fi.paper.evaluate import EvaluationSettings, evaluate_signal

    sess = _sess(session, source)
    src = sess._source(source)
    from nautilus_fi.kernel.session import _window

    lo, hi = _window(start, end, window) if (start is not None or end is not None or window is not None) else (sess.start, sess.end)
    sig = Field(signal) if isinstance(signal, str) else signal
    probe = Strategy(name or repr(sig), universe=universe, signal=sig, returns=returns, execution=NextClose(lag), schedule=schedule)
    prep = prepare([probe], src, lo, hi, extra_fields=("index_weight",) if weighting == "benchmark" else ())
    s = probe.resolved()
    ctx = prep.ctx_for(s)
    U = np.asarray(s.universe.evaluate(ctx), dtype=bool) & alive_mask(ctx)
    S = np.asarray(s.signal.evaluate(ctx.within(U, f"U:{s.universe.key}")), dtype=np.float64)
    p = ctx.panel
    t_lo = 0 if lo is None else int(p.dates.searchsorted(lo))
    vals = {k: v[t_lo:] for k, v in p.values.items()}
    vals["__universe"] = U[t_lo:].astype(np.float64)
    panel = Panel(p.dates[t_lo:], p.ids, vals, dict(p.lags))
    frame = pd.DataFrame(S[t_lo:], index=panel.dates, columns=panel.ids)
    st = EvaluationSettings(tuple(horizons), quantiles, str(s.schedule), returns, "__universe", weighting, neutralise_groups=tuple(neutralise_groups))
    rep = evaluate_signal(panel, frame, Conventions(timing=Timing(lag + 1)), st, prep.attributes if len(prep.attributes.columns) else None, name=name or repr(sig))
    sess.ledger.record_evaluation(rep, s.hypothesis or sess.hypothesis if (s.hypothesis or sess.hypothesis) and sess.ledger.hypothesis(s.hypothesis or sess.hypothesis) else None, note=f"session {sess.name}")
    return rep


VOCABULARY = """\
nautilus_fi kernel - building blocks (all Python objects; combine, then nfi.run / nfi.compare / nfi.sweep)

DATA       src = PanelSource(panel) | FrameSource({field: wide_df}) | CallableSource(fetch(field, start, end, ids), fields)
           | FieldStoreSource(root) | SQLSource(loader, mappings) | SyntheticSource(n_bonds=..., start=..., end=...)
           CompositeSource(a, b) routes fields; MappedSource(src, id_map); CachedSource(src, root) memory-maps years of data.
           src.catalog().search("spread") lists fields (entity asset / attribute / series / overlay, availability lag).
SIGNALS    Field("oas_bp"), Attribute("sector"), TimeSeries("vix"); arithmetic + - * / and comparisons.
           time series: .lag(n) .diff(n) .pct_change(n) .rolling(n).mean()/std()/zscore()/min()/max() .ewm(hl).mean() .ffill()
           per date (within the universe): .rank(by=) .zscore(by=) .demean(by=) .winsor(p) .neutralise("oasd", by="sector")
           .bucket([4, 7, 10]) / .bucket(n=5) (categories for by=) .top(n, by=) .scale(); combine(a, b, weights=[...]); where(c, a, b)
           custom: expr.apply(fn) (arrays) / expr.cs(fn) (one date's cross-section); look-ahead is checked automatically.
UNIVERSE   predicates: (Field("rating_score") <= 10) & (Field("oasd") > 1) & ~Attribute("sector").isin(["FIN"]); All()
INDICES    Index("LUACTRUU") (members = universe, weights = benchmark); .exclude(sector="FIN") .only(currency="USD")
           .where(pred) .cap(issuer=0.03) .cap(name=0.02) .reweight("equal"); Blend((a, 0.6), (b, 0.4)); ReturnSeries("idx.tr")
PORTFOLIO  EqualWeight(top=) TopN(n, weighting=ByField("amt_out"), by=, buffer=) Quantile(5, long=5, short=1, by=)
           RankWeighted(power=) BenchmarkWeight() Tilt(0.5, max_active=0.01) Hold(expr) Pairs([(a, b)], exposure="dts")
           .cap(issuer=0.03) .subject_to(*constraints) (nearest feasible) .checked(*constraints) (record only)
OPTIMISER  Optimise(3e-4 * Alpha() - TrackingError(by=["sector"], fields=["oasd", "dts"]) - Costs(), *constraints)
           terms: Alpha(expr=None) Linear(expr) | penalties (negative): Costs() TrackingError() ActiveShare() Risk(PCARisk(250)) CVaR(0.95)
           constraints: Weight() <= 0.02, ActiveWeight("issuer") <= 0.005, ActiveWeight("sector").within(0.02),
           GroupWeight("sector", "FIN") <= 0.3, Exposure("oasd").within(5, 7), ActiveExposure("dts", relative=True).within(0.05),
           Turnover() <= 0.1, Leverage() <= 1.5, Budget() == 1, CashWeight() <= 0.02, Names() <= 150, MinHolding(0.001),
           CVaRLimit(0.95) <= 0.02, LinearMeasure(expr) >= x; .soft(penalty); opt.replace(Turnover() <= 0.05) / .add() / .without(Turnover)
SCHEDULE   "BME" MonthEnd() MonthStart() Weekly(2) "W-WED" Daily() EveryN(5) QuarterEnd() Explicit(dates) Triggered(0.05, "turnover")
EXECUTION  NextClose(lag=1, cost=HalfSpread() * 1.5 + SqrtImpact(10), participation=Participation(0.1)) | SameClose()
           costs: HalfSpread(field=) FixedCost(bp) SqrtImpact(coef_bp) FieldCost(expr) NoCost(); cost * number_or_expr
FINANCING  Cash("SOFR", borrow_spread_bp=25, short_fee_bp=30) | NoCash()
MISSING    Fail() ZeroFill() DropDates(0.5) Refill("tr_alt")
METRICS    Metrics(annualisation=Calendar365()|Trading252(), turnover=HalfSum()|TwoWay()|OneWayBuys(), excess=Compounded()|VendorExcess())
OVERLAY    Hedge(["TY"], fields=["oad"], targets={"oad": 0.0}) (overlay instruments from the source)
STRATEGY   Strategy(name, universe=, signal=, portfolio=, benchmark=, schedule=, execution=, financing=, missing=, metrics=,
           overlay=, capital=, returns="tr", decider=None (escape hatch: Decider.decide(view) -> Target))
           .with_(**changes) .set("execution.lag", 2) .under("MRI_v1") .diff(other) .fingerprint .describe() .validate(src)
RUN        nfi.session(root, source=src); nfi.run(s, window=("2015", "2024")); nfi.compare(base, variant) -> .explain()
           nfi.sweep(base, {"portfolio.n": [50, 100]}, workers=4) -> .table() .selection(); nfi.evaluate(signal, universe=)
           nfi.run(s, fidelity="implement") (event engine); nfi.promote(result, "CANDIDATE")
RESULT     .stats() .summary() .headline() .report() .returns .gross .costs .benchmark_returns .exposures .groups["sector"]
           .contributions .trades .weights .construction .violations .explain(base)
"""


def vocabulary() -> str:
    """The building blocks on one page."""
    return VOCABULARY


__all__ = ["VOCABULARY", "catalog", "compare", "current_session", "evaluate", "explain", "promote", "run", "session", "sweep", "vocabulary"]
