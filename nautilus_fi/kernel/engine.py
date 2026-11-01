"""The vectorised engine: a ``Strategy`` on a ``DataSource``, to a ``Result``.

Two steps, so many strategies can share one load:

1. ``prepare(strategies, source, start, end)`` loads, once, everything they need (fields, attributes,
   date series, index constituents, overlays), with enough history before ``start`` for the
   longest look-back, and opens an expression cache on it.
2. ``simulate(strategy, prepared)`` evaluates the universe, signal and benchmark over the whole
   window with numpy (cached: variants that share them pay once), builds every rebalance's target
   in one vectorised pass when the construction does not depend on the current portfolio, and
   runs one loop over trading days for what is path dependent:

   * the weights held at the previous close earn the period: assets, cash (``Financing``),
     financing of negative cash, borrow on shorts, overlays; a held instrument that stops
     existing is redeemed into cash; a missing return follows ``MissingReturns``;
   * on rebalance dates (``schedule``) the target is computed (vectorised row, construction on the
     cross-section, or the decider), checked against the construction's constraints, and queued
     for the trade date ``execution.lag`` trading days later;
   * trades execute on that close at the cost model's price (``Participation`` caps the daily
     trade and carries the rest), and the new weights earn from the next period.

After the loop, contributions by instrument and by group, exposures, excess returns and the
statistics are computed in bulk. ``run`` is ``prepare`` + ``simulate``.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.domain.calendars import Triggered, WithInterim
from nautilus_fi.domain.conventions import (
    ExitPolicy,
    MissingReturnPolicy,
    MissingReturnsError,
    detect_bad_dates,
    excess_return_series,
)
from nautilus_fi.kernel.data import (
    OVERLAY,
    DataQualityError,
    DataRequest,
    DataSource,
    MissingDataError,
    as_source,
)
from nautilus_fi.kernel.expr import (
    Available,
    Cat,
    EvalContext,
    Expr,
    ExprCache,
    Field,
    alive_mask,
    check_lookahead,
)
from nautilus_fi.kernel.mechanics import turnover_vectors
from nautilus_fi.kernel.portfolio import Batch, KernelCross, _MinHold
from nautilus_fi.kernel.result import Result
from nautilus_fi.kernel.strategy import REPORT_EXPOSURES, Requirements, Strategy

_LOOKAHEAD_VERDICTS: dict[str, dict] = {}


@dataclass
class RunSettings:
    first_decision: bool = True                     # decide on the first day of the window
    record_weights: bool | None = None              # keep daily weights (None: when dates x instruments <= 60M)
    check_lookahead: bool = True                    # off: recorded in the result, which cannot then be promoted
    report_groups: tuple[str, ...] | None = None    # attributes for group analytics (None: categorical attributes with <= 60 groups)
    rating_buckets: tuple[float, ...] = (4.5, 7.5, 10.5)   # rating_score -> AAA-AA / A / BBB / HY
    exposure_fields: tuple[str, ...] = REPORT_EXPOSURES
    return_scale: tuple[float, float] | None = (1e-6, 0.02)   # median |return| per trading day must lie inside (None: no check)
    record_fields: tuple = ()                       # recorded with each rebalance: date series or expressions (a regime score), first value per date
    name: str | None = None


_REPORTING_ONLY = ("record_weights", "report_groups", "rating_buckets", "exposure_fields", "name", "record_fields")


def settings_key(st: RunSettings) -> dict:
    """The settings that can change a result's numbers (part of its cache key and trial identity);
    reporting-only settings are left out."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(st).items() if k not in _REPORTING_ONLY}


@dataclass
class Prepared:
    source: DataSource
    catalog: object
    panel: object
    attributes: pd.DataFrame
    series: pd.DataFrame
    overlays: object | None
    requirements: Requirements
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    cache: ExprCache
    ctx: EvalContext
    load_seconds: float
    info: dict = field(default_factory=dict)
    derived: dict = field(default_factory=dict)

    def ctx_for(self, strategy: Strategy) -> EvalContext:
        r = strategy.returns
        if isinstance(r, Expr):                                         # returns from an expression (a spread proxy, a hedged return)
            name = f"returns:{hashlib.sha1(r.key.encode()).hexdigest()[:12]}"
            hit = self.derived.get(name)
            if hit is None:
                from nautilus_fi.data.panel import Panel

                vals = np.asarray(r.evaluate(self.ctx.full()), dtype=np.float64)
                p = self.ctx.panel
                panel = Panel(p.dates, p.ids, {**p.values, name: vals}, {**p.lags, name: 0})
                hit = self.ctx._clone(panel=panel, token=f"{self.ctx.token}|{name}", return_field=name)
                hit.cache.pin(panel)
                self.derived[name] = hit
            return hit
        if r == self.ctx.return_field:
            return self.ctx
        return self.ctx._clone(return_field=r)

    def covers(self, strategy: Strategy) -> bool:
        req = strategy.requirements(self.catalog)
        r = self.requirements
        return req.fields <= r.fields and req.attributes <= r.attributes and req.series <= r.series and req.indices <= r.indices and req.overlay_fields <= r.overlay_fields and set(req.overlay_ids) <= set(r.overlay_ids) and req.lookback <= r.lookback


def _ts(x):
    return None if x is None else pd.Timestamp(x).normalize()


def prepare(strategies, source, start=None, end=None, cache: ExprCache | None = None, extra_fields: tuple[str, ...] = ()) -> Prepared:
    """Load once what every strategy in ``strategies`` needs over [start - look-back, end]."""
    t0 = time.perf_counter()
    source = as_source(source)
    strategies = [strategies] if isinstance(strategies, Strategy) else list(strategies)
    cat = source.catalog()
    req = Requirements()
    for s in strategies:
        req = req.union(s.requirements(cat))
    req.fields |= {f for f in extra_fields if f in cat}
    start, end = _ts(start), _ts(end)
    from nautilus_fi.kernel.expr import UNBOUNDED

    load_start = None if (start is None or req.lookback >= UNBOUNDED) else start - pd.Timedelta(days=int(np.ceil(req.lookback * 7 / 5)) + 10)
    panel = source.load(DataRequest(tuple(sorted(req.fields)), load_start, end))
    if len(panel.dates) == 0:
        raise ValueError(f"{source.name} returned no dates for {load_start} .. {end}")
    attrs = source.attributes(panel.ids, sorted(req.attributes)) if req.attributes else pd.DataFrame(index=panel.ids)
    notes: list[str] = []
    snapped = int(getattr(source, "snapped_observations", 0) or 0)
    if snapped:
        notes.append(f"{source.name}: {snapped} field observations dated between trading days moved to the next one (returns compounded, levels carried)")
    series = pd.DataFrame(index=panel.dates)
    if req.series:
        raw = source.series(sorted(req.series), load_start, end)
        raw.index = pd.to_datetime(raw.index).normalize()
        returns = set().union(*[_return_series(s) for s in strategies]) if strategies else set()
        cols = {}
        for c in raw.columns:
            cols[c], n = _snap_series(raw[[c]], panel.dates, c in returns)
            if n:
                snapped += n
                notes.append(f"series {c!r}: {n} observations dated between trading days " + ("compounded into the next one" if c in returns else "carried to the next one"))
        series = pd.concat([cols[c] for c in raw.columns], axis=1) if cols else series
    overlays = None
    if req.overlay_ids:
        ov = source.load(DataRequest(tuple(sorted(req.overlay_fields)), load_start, end, tuple(req.overlay_ids), entity=OVERLAY))
        from nautilus_fi.kernel.data import align

        overlays = align([ov], dates=panel.dates, ids=pd.Index(list(req.overlay_ids)))
    cache = cache or ExprCache()
    ctx = EvalContext(panel, attrs, series, source, cat, cache=cache, return_field=strategies[0].returns if (strategies and isinstance(strategies[0].returns, str)) else "tr")
    info = {"source": source.name, "source_version": source.version(), "dates": len(panel.dates), "instruments": len(panel.ids), "fields": sorted(req.fields), "load_start": str(panel.dates[0].date()), "load_end": str(panel.dates[-1].date()), "panel_mb": round(panel.nbytes() / 1e6, 1), "snapped_observations": snapped, "notes": notes, "point_in_time": source.point_in_time() if hasattr(source, "point_in_time") else None}
    return Prepared(source, cat, panel, attrs, series, overlays, req, start, end, cache, ctx, time.perf_counter() - t0, info)


def _return_series(s: Strategy) -> set[str]:
    """Date series a strategy uses as per-period returns (cash, risk-free, published benchmark
    returns): compounded, not carried, when their dates differ from the panel's."""
    from nautilus_fi.kernel.benchmark import ReturnSeries

    s = s.resolved()
    out = {n for n in (s.financing.series, s.metrics.risk_free) if n}

    def walk(b):
        if isinstance(b, ReturnSeries):
            out.add(b.series)
        for c in getattr(b, "_children", ()) or ():
            walk(c)

    if s.benchmark is not None:
        walk(s.benchmark)
    return out


def _snap_series(frame: pd.DataFrame, dates: pd.DatetimeIndex, compound: bool) -> tuple[pd.DataFrame, int]:
    from nautilus_fi.data.panel import _snap

    if len(dates) == 0:
        return frame.reindex(dates), 0
    return _snap(frame, dates, compound)


def run(strategy: Strategy, source, start=None, end=None, settings: RunSettings | None = None) -> Result:
    prep = prepare([strategy], source, start, end)
    return simulate(strategy, prep, start, end, settings)


# ====================================================================================== simulation
def _verdict(expr, ctx: EvalContext) -> dict:
    key = f"{ctx.token}|{expr.key}"
    if key not in _LOOKAHEAD_VERDICTS:
        _LOOKAHEAD_VERDICTS[key] = check_lookahead(expr, ctx)
    return _LOOKAHEAD_VERDICTS[key]


def simulate(strategy: Strategy, prep: Prepared, start=None, end=None, settings: RunSettings | None = None) -> Result:
    t0 = time.perf_counter()
    st = settings or RunSettings()
    s = strategy.resolved()
    ctx = prep.ctx_for(s)
    panel = ctx.panel
    dates, ids = panel.dates, panel.ids
    T, N = len(dates), len(ids)
    start = _ts(start) if start is not None else prep.start
    end = _ts(end) if end is not None else prep.end
    t_lo = 0 if start is None else int(dates.searchsorted(start))
    t_hi = T - 1 if end is None else int(dates.searchsorted(end, side="right")) - 1
    if t_hi - t_lo < 1:
        raise ValueError(f"the window {start} .. {end} needs at least two dates of data")
    timings: dict = {"load_seconds": prep.load_seconds}
    notes: list[str] = list(prep.info.get("notes", ()))

    # ---------------------------------------------------------------- look-ahead (structural; empirical for custom code)
    lookahead = []
    if st.check_lookahead:
        for e in s.expressions():
            la = e.lookahead()
            if la is None or la > 0:
                lookahead.append(_verdict(e, ctx.within(None, "")))
    else:
        notes.append("look-ahead checks switched off (RunSettings(check_lookahead=False)): the result cannot be promoted")

    # ---------------------------------------------------------------- data availability windows
    run_start = dates[t_lo]
    for e in s.expressions():
        guarded = {n.name for n in e.walk() if isinstance(n, Available)}
        for name in sorted(e.names() - guarded):
            info = prep.catalog.get(name) if prep.catalog is not None else None
            if info is not None and info.start and pd.Timestamp(info.start) > run_start:
                raise MissingDataError(f"{name!r} has no history before {info.start} but the run starts {run_start.date()}: start on or after it, or guard the rule with Available({name!r}) (e.g. ~Available({name!r}) | (Field({name!r}) > x))")

    # ---------------------------------------------------------------- realised returns and existence
    ta = time.perf_counter()
    rname = ctx.return_field                                            # the return field (or the panel column an expression became)
    if rname not in panel:
        raise ctx.unknown(rname)
    R = np.array(ctx.realised(rname), dtype=np.float64)           # realised, unlagged; a writable copy (missing-return policies)
    alive = alive_mask(ctx)
    exit_from = _exit_rows(ctx)
    _check_return_scale(R, alive, t_lo, t_hi, dates, rname, st.return_scale)

    # ---------------------------------------------------------------- missing-return policy: bad dates and refills
    mr = s.missing
    missing: dict = {"policy": mr.policy.value, "bad_dates": [], "zero_filled": 0, "refilled": 0, "redeemed": 0, "unexplained_exits": 0}
    refill = np.asarray(ctx.realised(mr.refill), dtype=np.float64) if mr.refill and mr.refill in panel else None
    rows = np.arange(t_lo, t_hi + 1)                                    # the run's periods (DropDates removes rows)
    refilled = None
    bad_rows: list[int] = []
    if mr.policy in (MissingReturnPolicy.DROP_DATE, MissingReturnPolicy.REFILL):
        win = slice(t_lo + 1, t_hi + 1)                                 # the first date has no return period
        bad = detect_bad_dates(pd.DataFrame(R[win], index=dates[win], columns=ids), mr.threshold, pd.DataFrame(alive[win], index=dates[win], columns=ids), mr.zeros_only)
        bad_rows = sorted(int(dates.get_loc(b)) for b in bad)
        missing["bad_dates"] = [str(dates[i].date()) for i in bad_rows]
        if mr.policy == MissingReturnPolicy.REFILL:
            if bad_rows and refill is None:
                raise MissingReturnsError(f"{len(bad_rows)} bad dates under {mr!r} but " + (f"the source has no {mr.refill!r}" if mr.refill else "no refill field is named: Refill('<field>')"))
            if refill is not None:
                fill = alive & ~np.isfinite(R) & np.isfinite(refill)
                fill[: t_lo + 1] = False
                fill[t_hi + 1:] = False
                for i in bad_rows:
                    fill[i] = alive[i] & np.isfinite(refill[i])          # a bad date is replaced, not only its gaps
                empty = [i for i in bad_rows if alive[i].any() and not fill[i].any()]
                if empty:
                    raise MissingReturnsError(f"{len(empty)} bad dates have no {mr.refill!r} value for any live name ({[str(dates[i].date()) for i in empty[:5]]}): their returns would stay as published (zeros); supply the refill or choose another policy")
                partly = {str(dates[i].date()): int((alive[i] & ~fill[i]).sum()) for i in bad_rows if (alive[i] & ~fill[i]).any()}
                if partly:
                    missing["bad_dates_partly_refilled"] = partly                  # names on a bad date the refill does not cover
                R = np.where(fill, refill, R)
                refilled = fill
                missing["refilled_bad_dates"] = int(sum(int(fill[i].sum()) for i in bad_rows))
        else:
            for i in bad_rows:
                R[i] = np.where(alive[i], 0.0, R[i])                     # neutral for benchmark drift; the row is not a period of the run
            rows = np.setdiff1d(rows, np.asarray(bad_rows, dtype=np.int64))
            missing["dropped"] = len(bad_rows)
    pos_of = {int(t): j for j, t in enumerate(rows)}

    # ---------------------------------------------------------------- benchmark (on the policy-resolved returns) and universe
    bench = s.benchmark
    bctx = _resolved_context(ctx, rname, R, alive, f"{mr!r}:{bad_rows}") if (refilled is not None or bad_rows) else ctx
    Wb = bench.weights(bctx) if (bench is not None and bench.has_weights) else None          # the index's holdings (returns, analytics)
    Wk = bench.known_weights(bctx) if Wb is not None else None                               # as a decision may know them
    bret = np.array(bench.returns(bctx), dtype=np.float64) if bench is not None else None
    if bench is not None:
        _benchmark_gaps(bench, bctx, Wb, R, alive, rows, mr, missing, dates, ids)
    U = np.asarray(s.universe.evaluate(ctx), dtype=bool) & alive
    timings["universe_seconds"] = time.perf_counter() - ta

    # ---------------------------------------------------------------- schedule (over the run's dates)
    cal = s.schedule
    if hasattr(cal, "resolve"):                                         # dates read from the source (IndexDates)
        cal = cal.resolve(prep.source)
    trading_days = list(dates.date[rows])
    dec_rows = dates.get_indexer(pd.to_datetime(cal.dates(trading_days, st.first_decision)))
    dec_rows = np.array([t for t in dec_rows if t >= 0 and int(t) in pos_of], dtype=np.int64)
    dec_index = {int(t): k for k, t in enumerate(dec_rows)}
    interim_rows: set[int] = set()
    if isinstance(cal, WithInterim):                                    # between fixed dates: the interim trigger and construction
        fixed = set(dates.get_indexer(pd.to_datetime(sorted(cal.fixed_dates(trading_days, st.first_decision)))))
        interim_rows = {int(t) for t in dec_rows if int(t) not in fixed}
        s_interim = s.with_(portfolio=cal.portfolio) if cal.portfolio is not None else s

    # ---------------------------------------------------------------- signal (decision rows, within the universe)
    ta = time.perf_counter()
    S = None
    ukey = f"U:{s.universe.key}"
    push = getattr(getattr(s.portfolio, "settings", None), "on_infeasible", "") == "push"     # a pushed rebalance retries on later days
    if s.signal is not None:
        sctx = ctx.within(U, ukey) if push else ctx.within(U, ukey).at_rows(dec_rows)
        S = np.asarray(s.signal.evaluate(sctx), dtype=np.float64)
    timings["signal_seconds"] = time.perf_counter() - ta
    if st.check_lookahead and s.decider is not None and len(dec_rows) >= 3:
        lookahead.append(_check_decider(s, ctx, U, Wk, dec_rows))

    # ---------------------------------------------------------------- cash, accrual, overlays
    fin = s.financing
    cash, cash_gap = _period_series(prep.series, fin.series, dates, rows)
    if fin.series and cash is None:
        notes.append(f"cash series {fin.series!r} not in the source: cash earns zero")
        missing["cash_series_absent"] = fin.series
    if cash is None:
        cash, cash_gap = np.zeros(T), np.zeros(T, dtype=bool)
    dt = np.zeros(T)                                                    # calendar years in each period of the run
    dt[rows[1:]] = (dates.values[rows[1:]] - dates.values[rows[:-1]]).astype("timedelta64[D]").astype(float) / 365.0
    ov = prep.overlays if s.overlay is not None else None
    K = len(ov.ids) if ov is not None else 0
    ovR = np.nan_to_num(np.asarray(ov[rname if rname in ov else "tr"], dtype=np.float64)) if K else np.zeros((T, 0))
    ov_fin = np.array([k in s.overlay.financed for k in ov.ids], dtype=bool) if K else np.zeros(0, dtype=bool)

    # ---------------------------------------------------------------- vectorised targets
    cons = s.portfolio
    targets = None
    vector_path = s.decider is None and not cons.path_dependent and not isinstance(cal, (Triggered, WithInterim)) and len(dec_rows) > 0
    if vector_path:
        ta = time.perf_counter()
        B = Batch(None if S is None else np.where(U[dec_rows], S[dec_rows], np.nan), U[dec_rows], None if Wk is None else Wk[dec_rows], None, ctx, dec_rows)
        targets = cons.batch(B)
        timings["targets_seconds"] = time.perf_counter() - ta
    needs_cs = targets is None or s.overlay is not None or bool(cons.constraints())

    # ---------------------------------------------------------------- the loop over the run's periods
    lag = s.execution.lag
    cost_model = s.execution.cost
    part = s.execution.participation
    rctx = ctx._clone(fills={}, audit=[])                               # cost models record what they fill and charge
    short_rate = None
    if fin.short_rate:
        if fin.short_rate not in prep.series.columns:
            raise MissingDataError(f"short_rate series {fin.short_rate!r} is not in the source")
        short_rate = prep.series[fin.short_rate].reindex(dates).to_numpy(dtype=np.float64)
        if np.isnan(short_rate[rows]).any():
            if mr.policy == MissingReturnPolicy.FAIL:
                raise MissingReturnsError(f"short_rate series {fin.short_rate!r} has gaps in the run under {mr!r}")
            missing["short_rate_zero_filled"] = int(np.isnan(short_rate[rows]).sum())
            short_rate = np.nan_to_num(short_rate)
    px_tol = s.execution.price_tolerance_bp
    PX = np.asarray(ctx.field(s.execution.price_field, 0), dtype=np.float64) if (px_tol is not None and s.execution.price_field in panel) else None
    if px_tol is not None and PX is None:
        raise MissingDataError(f"price_tolerance_bp needs the price field {s.execution.price_field!r}")
    record = st.record_weights if st.record_weights is not None else (T * N <= 60_000_000)
    W = np.zeros((T, N), dtype=np.float32) if record else None
    w = np.zeros(N)
    o = np.zeros(K)
    nav = float(s.capital)
    gross, net, costs, navs = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    cash_w, gexp, nexp, names = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T, dtype=np.int64)
    cash_ret, fin_ret, ov_ret = np.zeros(T), np.zeros(T), np.zeros(T)
    pending: dict[int, tuple] = {}                                      # run position -> (target, overlays, decision row, forced exits)
    trades_rows, weight_rows, cons_rows, viol_rows, ov_rows = [], [], [], [], []
    last_decision = None
    borrow, short_fee = fin.borrow_spread_bp / 1e4, fin.short_fee_bp / 1e4
    id_pos = {k: i for i, k in enumerate(ids)}
    counts = {"relaxed_rebalances": 0, "infeasible_rebalances": 0, "aborted_rebalances": 0, "price_tolerance_skips": 0, "pushed_rebalances": 0}
    n_rows = len(rows)
    constant = s.hold == "constant"
    held_w = w
    tl = time.perf_counter()
    for j, t in enumerate(rows):
        t = int(t)
        # 1) earn the period on the weights held at the previous close
        if j > 0:
            rt = R[t]
            held = w != 0
            dead = held & ~alive[t]
            if dead.any():
                unexplained = dead & (t < exit_from) if exit_from is not None else np.zeros(N, dtype=bool)
                if unexplained.any():
                    if mr.policy == MissingReturnPolicy.FAIL:
                        raise MissingReturnsError(f"{int(unexplained.sum())} held instruments have no {rname!r} return from {dates[t].date()} on and no maturity, call or exit date in the source ({list(ids[unexplained][:5])}); add an 'alive' field or exit dates, or choose ZeroFill(), DropDates() or Refill(field) to redeem them at their last price")
                    missing["unexplained_exits"] += int(unexplained.sum())
                missing["redeemed"] += int((dead & ~unexplained).sum())
                w = np.where(dead, 0.0, w)
                held = w != 0
            gaps = held & ~np.isfinite(rt)
            if gaps.any():
                if mr.policy in (MissingReturnPolicy.ZERO, MissingReturnPolicy.DROP_DATE):
                    missing["zero_filled"] += int(gaps.sum())
                elif mr.policy == MissingReturnPolicy.REFILL:
                    raise MissingReturnsError(f"{int(gaps.sum())} missing {rname!r} returns on held instruments on {dates[t].date()} ({list(ids[gaps][:5])}) remain after Refill({mr.refill!r})")
                else:
                    raise MissingReturnsError(f"{int(gaps.sum())} missing {rname!r} returns on held instruments on {dates[t].date()} ({list(ids[gaps][:5])}) under {mr!r}; choose ZeroFill(), DropDates() or Refill(field) explicitly")
            if refilled is not None:
                missing["refilled"] += int((refilled[t] & held).sum())
            rc = np.where(held, np.nan_to_num(rt), 0.0)
            asset = float(w @ rc)
            cw = 1.0 - float(w.sum())
            if cash_gap[t] and abs(cw) > 1e-9:
                if mr.policy == MissingReturnPolicy.FAIL:
                    raise MissingReturnsError(f"cash series {fin.series!r} has no value for the period ending {dates[t].date()} while the portfolio holds {cw:.2%} in cash, under {mr!r}; fill the series or choose ZeroFill() explicitly")
                missing["cash_zero_filled"] = missing.get("cash_zero_filled", 0) + 1
            cp = cw * cash[t] if cw >= 0 else cw * (cash[t] + borrow * dt[t])
            sp = float(-w[w < 0].sum()) * (short_fee + (short_rate[t] if short_rate is not None else 0.0)) * dt[t]
            op = float(o @ (ovR[t] - np.where(ov_fin, cash[t], 0.0))) if K else 0.0
            g = asset + cp - sp + op
            gross[t], cash_ret[t], fin_ret[t], ov_ret[t] = g, cp, -sp, op
            if constant:                                                # constant exposure: the traded weights, less redeemed names
                w = np.where(w != 0, held_w, 0.0)
            else:
                w = w * (1.0 + rc) / (1.0 + g)
                if K:
                    o = o * (1.0 + ovR[t]) / (1.0 + g)
        # 2) decide at the close
        k = dec_index.get(t)
        if k is not None:
            if t in interim_rows:
                out = _decide(t, k, s_interim, ctx, None, S, U, Wk, w, o, nav, True, ov, cons_rows, viol_rows, last_decision, id_pos, st, counts, trigger=cal.interim)
            else:
                out = _decide(t, k, s, ctx, targets, S, U, Wk, w, o, nav, needs_cs, ov, cons_rows, viol_rows, last_decision, id_pos, st, counts, trigger=s.schedule if isinstance(s.schedule, Triggered) else None)
            if out is not None:
                if j + lag < n_rows:
                    pending[j + lag] = (out[0], out[1], t, out[2])
                last_decision = t
            elif counts.pop("_pushed_now", False) and j + 1 < n_rows:   # infeasible today: try again tomorrow
                dec_index.setdefault(int(rows[j + 1]), k)
        # 3) trade on the close
        c = 0.0
        if j in pending:
            tw, to, dec_t, forced = pending.pop(j)
            tw = np.where(alive[t] | (tw == 0), tw, 0.0)
            if PX is not None and t != dec_t:                           # prices that moved beyond tolerance since the decision: not traded
                moved = np.abs(PX[t] / PX[dec_t] - 1.0) * 1e4 > px_tol
                moved &= np.isfinite(PX[t]) & np.isfinite(PX[dec_t]) & (tw != w)
                if moved.any():
                    counts["price_tolerance_skips"] += int(moved.sum())
                    tw = np.where(moved, w, tw)
            dw = tw - w
            fill = 1.0
            if part is not None:
                cap = part.capacity(ctx, t, nav)
                ex = np.clip(dw, -cap, cap)
                short_by = np.abs(dw - ex).sum()
                if short_by > 1e-12:
                    fill = float(np.abs(ex).sum() / max(np.abs(dw).sum(), 1e-18))
                    if j + 1 < n_rows and (j + 1) not in pending:
                        pending[j + 1] = (tw, to, dec_t, forced)
                dw = ex
            if np.any(dw != 0):
                c = cost_model.total(rctx, t, dw, nav)
            if K:
                c += float(np.abs(to - o).sum()) * s.overlay.cost_bp / 1e4
                if s.overlay.cost_per_contract:                         # exchange / tick costs per contract
                    cn = np.array([float(s.overlay.contract_notional.get(k, np.inf)) for k in ov.ids])
                    c += float((np.abs(to - o) * nav / cn).sum() * s.overlay.cost_per_contract / max(nav, 1.0))
            tov = turnover_vectors(w, w + dw)
            exits = forced & (dw < 0) if forced is not None else None
            trades_rows.append({"decision_date": dates[dec_t], "trade_date": dates[t], **tov, "cost": c, "n_buys": int((dw > 1e-12).sum()), "n_sells": int((dw < -1e-12).sum()), "fill_ratio": fill, "initial_build": bool(not np.any(w)), "forced_sells": float(-dw[exits].sum()) if exits is not None else 0.0, "n_forced": int(exits.sum()) if exits is not None else 0})
            w = w + dw
            w[np.abs(w) < 1e-15] = 0.0
            held_w = w.copy()
            if K:
                o = to.copy()
                if np.any(o):
                    ov_rows.append(pd.DataFrame({"date": dates[t], "instrument_id": ov.ids, "notional": o}))
            nz = np.nonzero(w)[0]
            weight_rows.append((t, nz, w[nz].copy()))
        costs[t] = c
        net[t] = (1.0 + gross[t]) * (1.0 - c) - 1.0
        nav *= 1.0 + net[t]
        if not nav > 0:
            raise InsolvencyError(f"{s.name}: NAV reached {nav:,.0f} on {dates[t].date()} (period return {net[t]:.2%}, gross exposure {np.abs(w).sum():.2f}x): the run stops; check the return scale, leverage and financing")
        navs[t] = nav
        cash_w[t] = 1.0 - w.sum()
        gexp[t], nexp[t], names[t] = float(np.abs(w).sum()), float(w.sum()), int((w != 0).sum())
        if record:
            W[t] = w
    timings["loop_seconds"] = time.perf_counter() - tl
    for k_, v_ in rctx.fills.items():
        missing[k_] = missing.get(k_, 0) + int(v_)
    cost_audit = pd.concat(rctx.audit, ignore_index=True) if rctx.audit else None

    # ---------------------------------------------------------------- bulk analytics
    ta = time.perf_counter()
    win = slice(int(rows[0]), int(rows[-1]) + 1) if len(rows) == rows[-1] - rows[0] + 1 else rows     # views unless dates were dropped
    idx = dates[win]
    daily = pd.DataFrame({"cash_weight": cash_w[win], "gross_exposure": gexp[win], "net_exposure": nexp[win], "n_names": names[win], "cash_return": cash_ret[win], "financing_return": fin_ret[win], "overlay_return": ov_ret[win]}, index=idx)
    exposures = pd.DataFrame(index=idx)
    groups: dict = {}
    contributions = pd.Series(dtype=float)
    excess = None
    if record:
        Ww = W[win]
        prevW = np.vstack([np.zeros((1, N), dtype=np.float32), Ww[:-1]])
        Rc = np.nan_to_num(R[win]).astype(np.float32)
        Rc[~alive[win]] = 0.0
        C = prevW * Rc                                      # contribution of each instrument to each period
        tot = C.sum(axis=0)
        nz = np.nonzero(np.abs(tot) > 0)[0]
        contributions = pd.Series(tot[nz].astype(float), index=ids[nz]).sort_values()
        Wbw = Wb[win].astype(np.float32) if Wb is not None else None
        for f in st.exposure_fields:
            if f in panel:
                X = np.nan_to_num(np.asarray(ctx.field(f))[win]).astype(np.float32, copy=False)
                exposures[f] = np.einsum("tn,tn->t", Ww, X, dtype=np.float64)
                if Wbw is not None:
                    exposures[f"active_{f}"] = exposures[f] - np.einsum("tn,tn->t", Wbw, X, dtype=np.float64)
        CB = None
        for g, cat in _report_groups(ctx, st).items():
            G = len(cat.categories)
            if G == 0:
                continue
            gs = _Grouper(cat, win, T, N)
            tables = {"weight": pd.DataFrame(gs.sum(Ww), index=idx, columns=cat.categories), "contribution": pd.DataFrame(gs.sum(C), index=idx, columns=cat.categories)}
            if Wbw is not None:
                if CB is None:
                    CB = np.vstack([np.zeros((1, N), dtype=np.float32), Wbw[:-1]]) * Rc
                tables["benchmark_weight"] = pd.DataFrame(gs.sum(Wbw), index=idx, columns=cat.categories)
                tables["benchmark_contribution"] = pd.DataFrame(gs.sum(CB), index=idx, columns=cat.categories)
            groups[g] = tables
        legs = {}
        for name, fname in (("excess_vendor", s.metrics.excess.vendor), ("treasury_leg", s.metrics.excess.treasury)):
            if fname and fname in panel:
                X = np.nan_to_num(np.asarray(ctx.realised(fname))[win]).astype(np.float32, copy=False)
                leg = np.einsum("tn,tn->t", prevW, X, dtype=np.float64)
                if name == "treasury_leg":
                    leg = leg + (1.0 - prevW.sum(axis=1)) * cash[win]
                leg[0] = 0.0
                legs[name] = leg
                daily[name] = leg
        if legs:
            try:
                excess = excess_return_series(pd.Series(gross[win], index=idx), None if "treasury_leg" not in legs else pd.Series(legs["treasury_leg"], index=idx), s.conventions(), None, None if "excess_vendor" not in legs else pd.Series(legs["excess_vendor"], index=idx))
            except ValueError as e:
                notes.append(f"excess returns unavailable: {e}")
    weights = pd.concat([pd.DataFrame({"date": dates[t], "instrument_id": ids[nz], "weight": vals}) for t, nz, vals in weight_rows], ignore_index=True) if weight_rows else pd.DataFrame(columns=["date", "instrument_id", "weight"])
    timings["analytics_seconds"] = time.perf_counter() - ta
    timings["simulate_seconds"] = time.perf_counter() - t0
    bench_series = None if bret is None else pd.Series(bret[win], index=idx, name="benchmark")
    rf = None
    if s.metrics.risk_free:
        rfv, rf_gap = _period_series(prep.series, s.metrics.risk_free, dates, rows)
        if rfv is None:
            notes.append(f"risk-free series {s.metrics.risk_free!r} not in the source: Sharpe uses raw returns")
        else:
            n_gap = int(rf_gap[rows[1:]].sum())
            if n_gap:
                if mr.policy == MissingReturnPolicy.FAIL:
                    raise MissingReturnsError(f"risk-free series {s.metrics.risk_free!r} has no value on {n_gap} periods of the run under {mr!r}")
                missing["risk_free_zero_filled"] = n_gap
            rf = pd.Series(rfv[win], index=idx)
    if prep.info.get("snapped_observations"):
        missing["snapped_observations"] = int(prep.info["snapped_observations"])
    diag = {"timings": timings, "lookahead": lookahead, "lookahead_checked": bool(st.check_lookahead), "notes": notes, "data": prep.info, "cache": {"entries": len(prep.cache), "hits": prep.cache.hits, "misses": prep.cache.misses}, "vectorised_targets": targets is not None, "n_decisions": len(dec_rows), "universe_avg": float(U[win].sum(axis=1).mean()), "settings": settings_key(st), **counts}
    res = Result(st.name or s.name, strategy.fingerprint, "paper", s, s.conventions(), pd.Series(net[win], index=idx, name="net"), pd.Series(gross[win], index=idx, name="gross"), pd.Series(costs[win], index=idx, name="cost"), pd.Series(navs[win], index=idx, name="nav"), bench_series, daily, pd.DataFrame(trades_rows), weights, exposures, groups, contributions, pd.DataFrame(cons_rows), pd.DataFrame(viol_rows), missing, excess, rf, diag, W[win] if record else None, ids if record else None, None, prep.info.get("source_version"), (str(idx[0].date()), str(idx[-1].date())))
    if ov_rows:
        res.diagnostics["overlays"] = pd.concat(ov_rows, ignore_index=True)
    if cost_audit is not None:
        res.diagnostics["cost_audit"] = cost_audit
    return res


def _check_decider(s: Strategy, ctx: EvalContext, U: np.ndarray, Wk: np.ndarray | None, dec_rows: np.ndarray) -> dict:
    """The empirical look-ahead check for custom code that cannot be analysed: the decider's target
    on two decision dates must not change when every later row is removed from the panel."""
    import copy

    from nautilus_fi.data.panel import Panel
    from nautilus_fi.domain.spec import as_target
    from nautilus_fi.kernel.expr import LookaheadError
    from nautilus_fi.paper.engine import _PanelView

    p, ids = ctx.panel, ctx.panel.ids
    name = type(s.decider).__name__
    report = {"expr": f"decider {name}", "structural_lead": None, "empirical": True, "cuts": []}

    def target(panel, t):
        cand = np.nonzero(U[t] | ((Wk[t] > 0) if Wk is not None else False))[0]
        bmask = (Wk[t] > 0) if Wk is not None else np.zeros(len(ids), dtype=bool)
        view = _PanelView(panel, t, cand, ids[np.nonzero(U[t])[0]], pd.Series(dtype=float), pd.Series(Wk[t][bmask], index=ids[bmask]) if Wk is not None else pd.Series(dtype=float), ctx.attributes.iloc[cand], float(s.capital))
        if s.universe_rule is not None:
            view._universe = s.universe_rule.apply(view, view.universe)
        tg = as_target(copy.deepcopy(s.decider).decide(view), True)
        x = tg.weights if tg.weights is not None else tg.scores
        return None if x is None else pd.Series(x, dtype=float).sort_index()

    for t in sorted({int(dec_rows[int(len(dec_rows) * f)]) for f in (0.4, 0.8)}):
        if t >= ctx.T - 1:
            continue
        sub = Panel(p.dates[: t + 1], ids, {k: v[: t + 1] for k, v in p.values.items()}, dict(p.lags))
        a, b = target(p, t), target(sub, t)
        same = (a is None and b is None) or (a is not None and b is not None and a.index.equals(b.index) and np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True))
        report["cuts"].append({"cut": str(p.dates[t].date()), "rows_changed": 0 if same else 1})
        if not same:
            raise LookaheadError(f"the decider {name} decides differently on {p.dates[t].date()} when later data are removed: it reads the future")
    return report


class InsolvencyError(ValueError):
    """The portfolio's NAV reached zero: the run stops (statistics after it would be meaningless)."""


_EXIT_ATTRIBUTES = ("maturity", "call_date", "exit_date", "redemption_date")


def _exit_rows(ctx: EvalContext) -> np.ndarray | None:
    """Row from which each instrument may stop existing without that being missing data: None when
    the source publishes existence (an ``alive`` field: every stop is an exit); otherwise a week
    before its earliest maturity, call or exit date attribute (``T``: never, the returns just stop)."""
    if "alive" in ctx.panel:
        return None
    out = np.full(ctx.N, ctx.T, dtype=np.int64)
    for c in _EXIT_ATTRIBUTES:
        if c in ctx.attributes.columns:
            d = pd.to_datetime(ctx.attributes[c], errors="coerce").to_numpy(dtype="datetime64[ns]")
            ok = ~np.isnat(d)
            r = np.full(ctx.N, ctx.T, dtype=np.int64)
            r[ok] = ctx.panel.dates.searchsorted(d[ok] - np.timedelta64(7, "D"))
            out = np.minimum(out, r)
    return out


def _check_return_scale(R: np.ndarray, alive: np.ndarray, t_lo: int, t_hi: int, dates: pd.DatetimeIndex, name: str, bounds) -> None:
    """Stop percent-scale (x100) or near-constant return fields before they produce statistics: the
    median absolute return per period must lie within ``bounds`` (per trading day; the upper bound
    widens with the square root of the panel spacing)."""
    if bounds is None:
        return
    step = max(1, int((t_hi - t_lo) * R.shape[1] // 2_000_000))      # a row sample is enough for a median, and light at index scale
    block = R[t_lo + 1: t_hi + 1: step]
    x = np.abs(block[np.isfinite(block) & alive[t_lo + 1: t_hi + 1: step]])
    x = x[x > 0]
    if len(x) < 50:
        return
    med = float(np.median(x))
    span = dates[t_lo: t_hi + 1]
    spacing = float(np.median(np.diff(span.values).astype("timedelta64[D]").astype(float))) if len(span) > 1 else 1.0
    lo, hi = float(bounds[0]), float(bounds[1]) * math.sqrt(max(1.0, spacing * 5.0 / 7.0))
    if not lo < med < hi:
        hint = "percentages (x100)?" if med >= hi else "basis points, a near-constant or mostly-zero field?"
        raise DataQualityError(f"the median absolute {name!r} return is {med:.4g} per period, outside ({lo:g}, {hi:.3g}): {hint} Returns must be simple decimal returns per period (0.0012 for 12bp); rescale the field, or set RunSettings(return_scale=None) to skip the check")


def _resolved_context(ctx: EvalContext, name: str, R: np.ndarray, alive: np.ndarray, tag: str) -> EvalContext:
    """A context whose realised ``name`` is the policy-resolved matrix, so the benchmark earns what
    the portfolio earns (refills, dropped dates). Existence is unchanged."""
    from nautilus_fi.data.panel import Panel

    p = ctx.panel
    panel = Panel(p.dates, p.ids, {**p.values, name: R}, dict(p.lags))
    token = f"{ctx.token}|resolved:{hashlib.sha1(tag.encode()).hexdigest()[:12]}"
    out = ctx._clone(panel=panel, token=token)
    out.cache.pin(panel)
    out.cache.put(f"{token}|alive|{out.return_field}", alive)
    return out


def _benchmark_gaps(bench, ctx: EvalContext, Wb: np.ndarray | None, R: np.ndarray, alive: np.ndarray, rows: np.ndarray, mr, missing: dict, dates, ids) -> None:
    """The missing-return policy for the benchmark: members it holds that exist but have no realised
    return (refills are already in ``R``) fail the run under Fail / Refill and are counted under
    ZeroFill / DropDates; a return series with a gap is treated the same way."""
    t = rows[1:]
    if not len(t):
        return
    G = None
    if Wb is not None:
        G = (Wb[t - 1] > 0) & alive[t] & ~np.isfinite(R[t])
        n = G.sum(axis=1)
    else:
        n = np.asarray(bench.return_gaps(ctx), dtype=bool)[t].astype(np.int64)
    total = int(n.sum())
    if not total:
        return
    first = int(np.argmax(n > 0))
    who = "" if G is None else f": {', '.join(map(str, ids[G[first]][:5]))}"
    if mr.policy in (MissingReturnPolicy.FAIL, MissingReturnPolicy.REFILL):
        raise MissingReturnsError(f"benchmark {bench!r}: {total} missing member returns (first on {dates[t[first]].date()}{who}) under {mr!r}; choose ZeroFill() or DropDates() explicitly, or supply a refill field")
    missing["benchmark_zero_filled"] = total


def _period_series(series: pd.DataFrame, name: str | None, dates: pd.DatetimeIndex, rows: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """A per-period return series (cash, risk-free) on the run's periods: values compounded over
    any dates the run removed, with a flag where the period has no value."""
    if not name or name not in series.columns:
        return None, None
    raw = series[name].reindex(dates).to_numpy(dtype=np.float64)
    gap = ~np.isfinite(raw)
    vals = np.nan_to_num(raw)
    out, flag = vals.copy(), gap.copy()
    if len(rows) > 1 and np.any(np.diff(rows) > 1):
        c = np.concatenate([[0.0], np.cumsum(np.log1p(vals))])
        cg = np.concatenate([[0], np.cumsum(gap)])
        p, t = rows[:-1], rows[1:]
        out[t] = np.expm1(c[t + 1] - c[p + 1])
        flag[t] = (cg[t + 1] - cg[p + 1]) > 0
    return out, flag


def _group_sum(X: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    """Per-row sums of ``X`` within groups (codes -1 ignored)."""
    T = X.shape[0]
    ok = codes >= 0
    flat = (np.arange(T, dtype=np.int64)[:, None] * G + codes)[ok]
    return np.bincount(flat, weights=X[ok].astype(np.float64), minlength=T * G).reshape(T, G)


class _Grouper:
    """Per-row group sums for one grouping, reusing the index work across arrays: a one-hot matrix
    product when groups are static, one bincount index when they change over time."""

    def __init__(self, cat: Cat, win: slice, T: int, N: int) -> None:
        self.G = max(len(cat.categories), 1)
        c = cat.codes
        if c.ndim == 1:
            self.onehot = np.zeros((N, self.G), dtype=np.float32)
            ok = c >= 0
            self.onehot[np.nonzero(ok)[0], c[ok]] = 1.0
        else:
            self.onehot = None
            cw = c[win]
            self.ok = cw >= 0
            self.Tw = cw.shape[0]
            self.flat = (np.arange(self.Tw, dtype=np.int64)[:, None] * self.G + cw)[self.ok]

    def sum(self, X: np.ndarray) -> np.ndarray:
        if self.onehot is not None:
            return (np.asarray(X, dtype=np.float32) @ self.onehot).astype(np.float64)
        return np.bincount(self.flat, weights=np.asarray(X)[self.ok].astype(np.float64), minlength=self.Tw * self.G).reshape(self.Tw, self.G)


def _report_groups(ctx: EvalContext, st: RunSettings) -> dict[str, Cat]:
    out: dict[str, Cat] = {}
    names = st.report_groups
    if names is None:
        names = tuple(c for c in ctx.attributes.columns if not pd.api.types.is_numeric_dtype(ctx.attributes[c]) and 1 < ctx.attributes[c].nunique() <= 60)
        if ctx.catalog is not None:                                     # dated categories (a sector map that changes over time)
            names += tuple(f for f in ctx.panel.fields if f in ctx.catalog and getattr(ctx.catalog[f], "categories", ()) and f not in names)
    for g in names:
        try:
            out[g] = ctx.groups(g)
        except KeyError:
            continue
    if "rating_score" in ctx.panel and "rating" not in out:
        out["rating"] = Field("rating_score").bucket(list(st.rating_buckets)).evaluate(ctx.full())
        out["rating"].categories = ["AAA-AA", "A", "BBB", "HY"][: len(out["rating"].categories)] if len(st.rating_buckets) == 3 else out["rating"].categories
    return out


# ====================================================================================== decisions
def _cross(t: int, s: Strategy, ctx: EvalContext, scores_row, urow: np.ndarray, wbrow: np.ndarray | None, w, nav, extra_names=()) -> KernelCross:
    cand = urow | (w != 0)
    if wbrow is not None:
        cand |= wbrow > 0
    cols = np.nonzero(cand)[0]
    ids = ctx.panel.ids[cols]
    want = set(s.portfolio.names()) | set(extra_names) | (set(s.overlay.fields) if s.overlay is not None else set())
    data = {}
    for n in sorted(want):
        if n in ctx.panel:
            data[n] = np.asarray(ctx.field(n)[t], dtype=np.float64)[cols]
        elif n in ctx.attributes.columns:
            data[n] = ctx.attributes[n].to_numpy()[cols]
    df = pd.DataFrame(data, index=ids)
    scores = None if scores_row is None else pd.Series(scores_row[cols], index=ids)
    cost_bp = pd.Series(s.execution.cost.linear(ctx, t, nav)[cols] * 1e4, index=ids)
    min_w = None
    if any(isinstance(c.measure, _MinHold) and c.lo is None for c in s.portfolio.constraints()):
        price = next((ctx.field(f)[t][cols] for f in ("price_dirty", "price_clean") if f in ctx.panel), None)
        if price is not None and "min_piece" in ctx.attributes.columns:
            min_w = pd.Series(ctx.attributes["min_piece"].to_numpy(dtype=float)[cols] * np.asarray(price, dtype=float) / 100.0 / max(nav, 1.0), index=ids).fillna(0.0)
    return KernelCross(ctx.panel.dates[t], ids, df, pd.Series(w[cols], index=ids), pd.Series(wbrow[cols] if wbrow is not None else np.zeros(len(cols)), index=ids), scores, pd.Series(urow[cols], index=ids), cost_bp, min_w, None, nav, ctx, t, cols, s.exits.value)


def _joint_overlays(cs, s: Strategy, ov, o, t: int) -> None:
    """Hand the optimiser the overlay instruments as decision variables (``Hedge(joint=True)``):
    their exposures per unit notional at ``t``, the current notionals, the cost and the cap."""
    if s.overlay is None or ov is None or not getattr(s.overlay, "joint", False):
        return
    cs.overlay_data = pd.DataFrame({f: np.asarray(ov[f][t], dtype=np.float64) for f in s.overlay.fields if f in ov}, index=ov.ids).fillna(0.0)
    cs.overlay_current = np.asarray(o, dtype=np.float64).copy()
    cs.overlay_cost_bp = s.overlay.cost_bp
    cs.overlay_max = s.overlay.max_gross


def _degenerate(s: Strategy, S, U: np.ndarray, t: int) -> str | None:
    """Why a rebalance keeps the book instead of trading: an empty universe, or a signal the
    construction ranks by with no value for any name in the universe (paperport's abort rule:
    an outage must not read as a decision to sell everything)."""
    u = U[t]
    if not u.any():
        return "empty universe"
    if S is not None and s.portfolio.needs_scores and not np.isfinite(S[t][u]).any():
        return "no signal value for any name in the universe"
    return None


def _apply_exits(policy: ExitPolicy, tw: np.ndarray, w: np.ndarray, leavers: np.ndarray, handled: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Held names outside the universe at a decision. ``SELL`` forces them to zero (their sales are
    reported as forced and sit outside the optimiser's turnover budget); ``EXCLUDE`` lets them only
    shrink; ``HOLD_NOT_BUY`` keeps them, never above their current weight. ``handled``: the
    construction (an optimiser) already applied the policy to them."""
    if policy == ExitPolicy.SELL:
        tw = tw.copy()
        tw[leavers] = 0.0
        return tw, leavers.copy()
    if handled:
        return tw, None
    if policy == ExitPolicy.EXCLUDE:
        same = np.sign(tw) == np.sign(w)
        out = np.where(leavers, np.where(same, np.sign(w) * np.minimum(np.abs(tw), np.abs(w)), 0.0), tw)
        return out, None
    out = tw.copy()
    out[leavers] = w[leavers]
    others = ~leavers & (out != 0)
    if (tw >= 0).all() and others.any():
        room = float(tw.sum()) - float(w[leavers].sum())
        total = float(out[others].sum())
        out[others] *= max(room, 0.0) / total if total > 0 else 0.0
    return out, None


def _decide(t, k, s: Strategy, ctx, targets, S, U, Wb, w, o, nav, needs_cs, ov, cons_rows, viol_rows, last_decision, id_pos, st, counts, trigger=None):
    ids = ctx.panel.ids
    N = len(ids)
    date = ctx.panel.dates[t]
    diag: dict = {}
    cs = None
    overlay_target = o.copy()
    if s.decider is None:
        reason = _degenerate(s, S, U, t)
        if reason is not None:
            cons_rows.append({"date": date, "action": "aborted", "reason": reason})
            counts["aborted_rebalances"] += 1
            return None
    if s.decider is not None:
        from nautilus_fi.domain.spec import as_target
        from nautilus_fi.paper.engine import _PanelView

        cand = np.nonzero(U[t] | (w != 0) | ((Wb[t] > 0) if Wb is not None else False))[0]
        held = w != 0
        bmask = (Wb[t] > 0) if Wb is not None else np.zeros(N, dtype=bool)
        view = _PanelView(ctx.panel, t, cand, ids[np.nonzero(U[t])[0]], pd.Series(w[held], index=ids[held]), pd.Series(Wb[t][bmask], index=ids[bmask]) if Wb is not None else pd.Series(dtype=float), ctx.attributes.iloc[cand], nav)
        if s.universe_rule is not None:
            view._universe = s.universe_rule.apply(view, view.universe)
        try:
            target = as_target(s.decider.decide(view), True)
        except KeyError as e:
            from nautilus_fi.kernel.strategy import StrategyError

            raise StrategyError(f"the decider {type(s.decider).__name__} read {e} which the strategy did not load: declare it with Strategy(..., fields=({e!s},))") from e
        if target.skip:
            cons_rows.append({"date": date, "action": "skip", "note": target.note})
            return None
        if target.weights is not None:
            wt = pd.Series(target.weights, dtype=float)
            tw = np.zeros(N)
            pos = np.array([id_pos.get(str(i), -1) for i in wt.index])
            ok = pos >= 0
            tw[pos[ok]] = wt.to_numpy()[ok]
            diag = {"method": "Target.weights"}
        else:
            row = np.full(N, np.nan)
            sc = pd.Series(target.scores, dtype=float)
            pos = np.array([id_pos.get(str(i), -1) for i in sc.index])
            ok = pos >= 0
            row[pos[ok]] = sc.to_numpy()[ok]
            urow = U[t]
            if s.universe_rule is not None:
                urow = np.zeros(N, dtype=bool)
                upos = np.array([id_pos.get(str(i), -1) for i in view.universe], dtype=np.int64)
                urow[upos[upos >= 0]] = True
            if not urow.any() or (s.portfolio.needs_scores and not np.isfinite(row[urow]).any()):
                cons_rows.append({"date": date, "action": "aborted", "reason": "empty universe" if not urow.any() else "no score for any name in the universe"})
                counts["aborted_rebalances"] += 1
                return None
            cs = _cross(t, s, ctx, row, urow, None if Wb is None else Wb[t], w, nav)
            _joint_overlays(cs, s, ov, o, t)
            wt, diag = s.portfolio.weights(cs)
            tw = _to_vector(wt, id_pos, N)
        if target.overlays is not None and ov is not None:
            overlay_target = pd.Series(target.overlays).reindex(ov.ids).fillna(0.0).to_numpy()
    elif targets is not None:
        tw = np.asarray(targets[k], dtype=np.float64)
        if needs_cs:
            cs = _cross(t, s, ctx, None if S is None else S[t], U[t], None if Wb is None else Wb[t], w, nav)
    else:
        cs = _cross(t, s, ctx, None if S is None else S[t], U[t], None if Wb is None else Wb[t], w, nav)
        _joint_overlays(cs, s, ov, o, t)
        wt, diag = s.portfolio.weights(cs)
        tw = _to_vector(wt, id_pos, N)
    leavers = (w != 0) & ~U[t]
    forced = None
    if leavers.any():
        tw, forced = _apply_exits(s.exits, tw, w, leavers, getattr(s.portfolio, "handles_exits", False) and s.decider is None)
    viol = []
    if cs is not None:
        wt_cs = pd.Series(tw[cs.cols], index=cs.ids)
        joint = diag.pop("overlays", None) if isinstance(diag.get("overlays"), pd.Series) else None
        if joint is not None:
            cs.overlay_target = joint.reindex(cs.overlay_data.index).fillna(0.0).to_numpy()   # the checks count the hedge's exposure
        viol = diag.pop("_violations", None) or s.portfolio.check(wt_cs, cs)
        if s.overlay is not None and ov is not None:
            ov_data = pd.DataFrame({f: np.asarray(ov[f][t], dtype=np.float64) for f in s.overlay.fields if f in ov}, index=ov.ids)
            if joint is not None:
                sized, odiag = joint, {"method": "solved with the book"}
            else:
                sized, odiag = s.overlay.size(wt_cs, cs, ov_data)
            overlay_target = sized.reindex(ov.ids).fillna(0.0).to_numpy()
            if getattr(s.overlay, "lot", None):
                overlay_target = np.round(overlay_target / s.overlay.lot) * s.overlay.lot
            if getattr(s.overlay, "tolerance", 0.0) and np.abs(overlay_target - o).max(initial=0.0) <= s.overlay.tolerance:
                overlay_target = o.copy()                               # within tolerance: the hedge stays
                odiag = {**odiag, "kept_within_tolerance": True}
            diag["overlay"] = odiag
    if trigger is not None and last_decision is not None:
        cal = trigger
        if cal.measure == "score_drop" and S is not None:              # the book's score, then and now
            held = (w != 0) & np.isfinite(S[last_decision]) & np.isfinite(S[t])
            gw = float(np.abs(w[held]).sum())
            ch = float(w[held] @ (S[last_decision][held] - S[t][held]) / gw) if gw > 0 else 0.0
        elif cal.measure == "signal" and cal.field:
            F = ctx.field(cal.field)                    # as known at each decision; ranks over the candidates, as the Paper engine does
            cand = U[t] | (w != 0) | ((Wb[t] > 0) if Wb is not None else False)
            ch = float((pd.Series(F[t][cand]).rank(pct=True) - pd.Series(F[last_decision][cand]).rank(pct=True)).abs().mean())
            ch = ch if np.isfinite(ch) else 0.0
        elif cal.measure == "signal" and S is not None:
            prev, now = S[last_decision], S[t]
            both = np.isfinite(prev) & np.isfinite(now)
            ch = float(np.abs(pd.Series(now[both]).rank(pct=True).to_numpy() - pd.Series(prev[both]).rank(pct=True).to_numpy()).mean()) if both.any() else 0.0
        else:
            ch = turnover_vectors(w, tw)[str(cal.turnover).upper()]
        if ch < cal.threshold:
            cons_rows.append({"date": date, "action": "trigger_skip", "measure": cal.measure, "value": ch, "threshold": cal.threshold})
            return None
    relaxed = [r for r in (diag.get("relaxations") or []) if r.get("constraint") == "all limits" or r.get("slack") is not None]
    hard_relaxed = any(r.get("constraint") == "all limits" for r in relaxed)
    action = "rebalance"
    if diag.get("ok") is False and str(diag.get("message", "")).startswith("push"):
        counts["pushed_rebalances"] += 1
        counts["_pushed_now"] = True
        cons_rows.append({"date": date, "action": "pushed", "reason": diag.get("message")})
        return None
    if diag.get("ok") is False:
        counts["infeasible_rebalances"] += 1
        action = "infeasible_kept"
    elif hard_relaxed:
        counts["relaxed_rebalances"] += 1
        action = "rebalance_relaxed"
    flat = {kk: vv for kk, vv in diag.items() if not isinstance(vv, (dict, list, pd.DataFrame, pd.Series))}
    if diag.get("binding"):
        flat["binding"] = "; ".join(str(b) for b in diag["binding"][:30])
    if relaxed:
        flat["relaxed_constraints"] = "; ".join(f"{r.get('constraint')}" + (f" by {r['slack']:.4g}" if isinstance(r.get("slack"), (int, float)) else "") for r in relaxed[:30])
    if isinstance(diag.get("optimiser"), dict):
        flat.update({f"opt_{kk}": vv for kk, vv in diag["optimiser"].items() if not isinstance(vv, (dict, list))})
    row = {"date": date, "action": action, "n_names": int((tw != 0).sum()), "gross": float(np.abs(tw).sum()), "net": float(tw.sum()), **flat, "n_violations": len(viol)}
    for f in st.record_fields:                                          # per-rebalance regime records
        if isinstance(f, str) and f in ctx.series.columns:
            row[f"record_{f}"] = float(ctx.series[f].iloc[t])
        else:
            v = np.asarray((f if isinstance(f, Expr) else Field(f)).evaluate(ctx.full())[t], dtype=float)
            fin = v[np.isfinite(v)]
            row[f"record_{f if isinstance(f, str) else repr(f)}"] = float(fin[0]) if len(fin) else float("nan")
    if forced is not None and forced.any():
        row["n_forced_exits"] = int(forced.sum())
    cons_rows.append(row)
    for v in viol:
        viol_rows.append({"date": date, **v})
    return tw, overlay_target, forced


@dataclass
class Decision:
    """One live decision: what the strategy would hold after trading at ``as_of``'s close, from
    the current book. ``action`` is the engine's (rebalance, rebalance_relaxed, infeasible_kept,
    aborted, skip, pushed, trigger_skip); a decision that keeps the book has ``weights`` equal to
    the current ones. ``check(weights)`` runs the strategy's constraints on other weights (the
    post-trade book after rounding)."""

    as_of: pd.Timestamp
    action: str
    weights: pd.Series                       # target weight by instrument (non-zero)
    current: pd.Series
    overlays: pd.Series                      # overlay notional (share of NAV) by contract
    scores: pd.Series | None                 # the signal over the universe
    universe: pd.Index
    benchmark: pd.Series                     # benchmark weights as known at the decision
    forced_exits: pd.Index
    log: dict
    violations: list
    scheduled: bool                          # as_of is one of the schedule's decision dates
    analytics: pd.DataFrame                  # requested fields at the decision, by instrument
    _cross: object = None
    _portfolio: object = None

    def check(self, weights: pd.Series | dict) -> list[dict]:
        if self._cross is None:
            return []
        return self._portfolio.check(pd.Series(weights, dtype=float).reindex(self._cross.ids).fillna(0.0), self._cross)


def decide_at(strategy: Strategy, source, as_of, current: pd.Series | dict | None = None, nav: float | None = None, overlays: pd.Series | dict | None = None, last_decision=None, settings: RunSettings | None = None, fields: tuple[str, ...] = (), prepared: Prepared | None = None) -> Decision:
    """The strategy's decision at ``as_of`` (the last data date on or before it) from the
    ``current`` weights: the same universe, signal, construction, exit policy and overlay
    sizing a backtest runs, once. ``last_decision``: the previous decision date, for a
    ``Triggered`` schedule's threshold. ``fields``: values to return in ``analytics`` (coverage
    checks for a proposal)."""
    st = settings or RunSettings()
    s = strategy.resolved()
    as_of = _ts(as_of)
    first = min(as_of, _ts(last_decision)) if last_decision is not None else as_of     # a trigger compares with the previous decision
    prep = prepared or prepare([strategy], source, first, as_of, extra_fields=tuple(fields))
    ctx = prep.ctx_for(s)
    dates, ids = ctx.panel.dates, ctx.panel.ids
    t = int(dates.searchsorted(as_of, side="right")) - 1
    if t < 0:
        raise ValueError(f"no data on or before {as_of.date()}")
    N = len(ids)
    id_pos = {k: i for i, k in enumerate(ids)}
    cur = pd.Series(current if current is not None else {}, dtype=float)
    unknown = [i for i in cur.index if str(i) not in id_pos and cur[i] != 0]
    if unknown:
        raise MissingDataError(f"{len(unknown)} held instruments are not in {prep.source.name} on {dates[t].date()} ({unknown[:5]}): map the book's symbols to the source's ids")
    w = _to_vector(cur, id_pos, N)
    alive = alive_mask(ctx)
    U = np.asarray(s.universe.evaluate(ctx), dtype=bool) & alive
    bench = s.benchmark
    Wk = bench.known_weights(ctx) if (bench is not None and bench.has_weights) else None
    last_row = None
    if last_decision is not None:
        last_row = int(dates.searchsorted(_ts(last_decision), side="right")) - 1
        last_row = last_row if 0 <= last_row < t else None
    S = None
    if s.signal is not None:                                            # the decision row, and the previous decision's for a trigger
        S = np.asarray(s.signal.evaluate(ctx.within(U, f"U:{s.universe.key}").at_rows(np.array([t] if last_row is None else [last_row, t]))), dtype=np.float64)
    ov = prep.overlays if s.overlay is not None else None
    o = np.zeros(len(ov.ids)) if ov is not None else np.zeros(0)
    if ov is not None and overlays is not None:
        o = pd.Series(overlays, dtype=float).reindex(ov.ids).fillna(0.0).to_numpy()
    cal = s.schedule.resolve(prep.source) if hasattr(s.schedule, "resolve") else s.schedule
    ahead = list(pd.bdate_range(dates[t] + pd.Timedelta(days=1), periods=10).date)     # the month or week goes on after a live as_of
    scheduled = dates[t].date() in set(pd.to_datetime(list(cal.dates(list(dates.date[: t + 1]) + ahead, False))).date)
    trig = s.schedule if isinstance(s.schedule, Triggered) else (s.schedule.interim if isinstance(s.schedule, WithInterim) else None)
    cons_rows: list = []
    viol_rows: list = []
    counts = {"relaxed_rebalances": 0, "infeasible_rebalances": 0, "aborted_rebalances": 0, "price_tolerance_skips": 0, "pushed_rebalances": 0}
    nav = float(nav if nav is not None else s.capital)
    out = _decide(t, 0, s, ctx, None, S, U, Wk, w, o, nav, True, ov, cons_rows, viol_rows, last_row, id_pos, st, counts, trigger=trig)
    log = cons_rows[-1] if cons_rows else {}
    if out is None:
        tw, to, forced = w.copy(), o.copy(), None
    else:
        tw, to, forced = out
    nz = np.nonzero(tw)[0]
    cs = None
    if s.decider is None:
        cs = _cross(t, s, ctx, None if S is None else S[t], U[t], None if Wk is None else Wk[t], w, nav)
        _joint_overlays(cs, s, ov, o, t)
    an = {}
    for f in fields:
        if f in ctx.panel:
            an[f] = np.asarray(ctx.field(f)[t], dtype=np.float64)
        elif f in ctx.attributes.columns:
            an[f] = ctx.attributes[f].to_numpy()
    if S is not None:
        an["signal"] = S[t]
    analytics = pd.DataFrame(an, index=ids)
    bw = Wk[t] if Wk is not None else np.zeros(N)
    return Decision(dates[t], str(log.get("action", "rebalance")), pd.Series(tw[nz], index=ids[nz]), pd.Series(w[w != 0], index=ids[w != 0]), pd.Series(to, index=ov.ids) if ov is not None else pd.Series(dtype=float), pd.Series(S[t][U[t]], index=ids[U[t]]) if S is not None else None, ids[U[t]], pd.Series(bw[bw > 0], index=ids[bw > 0]), ids[forced] if forced is not None else ids[:0], log, [v for v in viol_rows], scheduled, analytics, cs, s.portfolio)


def _to_vector(wt: pd.Series, id_pos: dict, N: int) -> np.ndarray:
    tw = np.zeros(N)
    if len(wt):
        pos = np.array([id_pos.get(str(i), -1) for i in wt.index])
        ok = pos >= 0
        tw[pos[ok]] = np.asarray(wt, dtype=float)[ok]
    return tw


__all__ = ["Decision", "InsolvencyError", "Prepared", "RunSettings", "decide_at", "prepare", "run", "settings_key", "simulate"]
