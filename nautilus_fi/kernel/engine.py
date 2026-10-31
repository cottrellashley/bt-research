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

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.domain.calendars import Triggered
from nautilus_fi.domain.conventions import (
    MissingReturnPolicy,
    MissingReturnsError,
    detect_bad_dates,
    excess_return_series,
)
from nautilus_fi.kernel.data import OVERLAY, DataRequest, DataSource, as_source
from nautilus_fi.kernel.expr import Cat, EvalContext, ExprCache, Field, alive_mask, check_lookahead
from nautilus_fi.kernel.mechanics import turnover_vectors
from nautilus_fi.kernel.portfolio import Batch, KernelCross, _MinHold
from nautilus_fi.kernel.result import Result
from nautilus_fi.kernel.strategy import REPORT_EXPOSURES, Requirements, Strategy

_LOOKAHEAD_VERDICTS: dict[str, dict] = {}


@dataclass
class RunSettings:
    first_decision: bool = True                     # decide on the first day of the window
    record_weights: bool | None = None              # keep daily weights (None: when dates x instruments <= 60M)
    check_lookahead: bool = True
    report_groups: tuple[str, ...] | None = None    # attributes for group analytics (None: categorical attributes with <= 60 groups)
    rating_buckets: tuple[float, ...] = (4.5, 7.5, 10.5)   # rating_score -> AAA-AA / A / BBB / HY
    exposure_fields: tuple[str, ...] = REPORT_EXPOSURES
    name: str | None = None


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

    def ctx_for(self, strategy: Strategy) -> EvalContext:
        if strategy.returns == self.ctx.return_field:
            return self.ctx
        return self.ctx._clone(return_field=strategy.returns)

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
    load_start = None if start is None else start - pd.Timedelta(days=int(np.ceil(req.lookback * 7 / 5)) + 10)
    panel = source.load(DataRequest(tuple(sorted(req.fields)), load_start, end))
    if len(panel.dates) == 0:
        raise ValueError(f"{source.name} returned no dates for {load_start} .. {end}")
    attrs = source.attributes(panel.ids, sorted(req.attributes)) if req.attributes else pd.DataFrame(index=panel.ids)
    series = source.series(sorted(req.series), load_start, end).reindex(panel.dates) if req.series else pd.DataFrame(index=panel.dates)
    overlays = None
    if req.overlay_ids:
        ov = source.load(DataRequest(tuple(sorted(req.overlay_fields)), load_start, end, tuple(req.overlay_ids), entity=OVERLAY))
        from nautilus_fi.kernel.data import align

        overlays = align([ov], dates=panel.dates, ids=pd.Index(list(req.overlay_ids)))
    cache = cache or ExprCache()
    ctx = EvalContext(panel, attrs, series, source, cat, cache=cache, return_field=strategies[0].returns if strategies else "tr")
    info = {"source": source.name, "source_version": source.version(), "dates": len(panel.dates), "instruments": len(panel.ids), "fields": sorted(req.fields), "load_start": str(panel.dates[0].date()), "load_end": str(panel.dates[-1].date()), "panel_mb": round(panel.nbytes() / 1e6, 1)}
    return Prepared(source, cat, panel, attrs, series, overlays, req, start, end, cache, ctx, time.perf_counter() - t0, info)


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
    notes: list[str] = []

    # ---------------------------------------------------------------- look-ahead (structural; empirical for custom code)
    lookahead = []
    if st.check_lookahead:
        for e in s.expressions():
            la = e.lookahead()
            if la is None or la > 0:
                lookahead.append(_verdict(e, ctx.within(None, "")))

    # ---------------------------------------------------------------- returns, existence, benchmark, universe
    ta = time.perf_counter()
    if s.returns not in panel:
        raise ctx.unknown(s.returns)
    R = np.array(ctx.field(s.returns), dtype=np.float64)              # writable copy (missing-return policies)
    alive = alive_mask(ctx)
    bench = s.benchmark
    Wb = bench.weights(ctx) if (bench is not None and bench.has_weights) else None
    bret = bench.returns(ctx) if bench is not None else None
    U = np.asarray(s.universe.evaluate(ctx), dtype=bool) & alive
    timings["universe_seconds"] = time.perf_counter() - ta

    # ---------------------------------------------------------------- schedule
    cal = s.schedule
    trading_days = list(dates.date[t_lo:t_hi + 1])
    dec_rows = dates.get_indexer(pd.to_datetime(cal.dates(trading_days, st.first_decision)))
    dec_rows = dec_rows[dec_rows >= 0]
    dec_index = {int(t): k for k, t in enumerate(dec_rows)}

    # ---------------------------------------------------------------- signal (decision rows, within the universe)
    ta = time.perf_counter()
    S = None
    ukey = f"U:{s.universe.key}"
    if s.signal is not None:
        sctx = ctx.within(U, ukey).at_rows(dec_rows)
        S = np.asarray(s.signal.evaluate(sctx), dtype=np.float64)
    timings["signal_seconds"] = time.perf_counter() - ta

    # ---------------------------------------------------------------- missing-return policy: bad dates
    mr = s.missing
    missing = {"policy": mr.policy.value, "bad_dates": [], "zero_filled": 0, "refilled": 0, "redeemed": 0}
    refill = np.asarray(ctx.field(mr.refill), dtype=np.float64) if mr.refill and mr.refill in panel else None
    if mr.policy in (MissingReturnPolicy.DROP_DATE, MissingReturnPolicy.REFILL):
        win = slice(t_lo + 1, t_hi + 1)
        bad = detect_bad_dates(pd.DataFrame(R[win], index=dates[win], columns=ids), mr.threshold, pd.DataFrame(alive[win], index=dates[win], columns=ids))
        missing["bad_dates"] = [str(pd.Timestamp(b).date()) for b in bad]
        for b in bad:
            i = dates.get_loc(b)
            if mr.policy == MissingReturnPolicy.REFILL:
                if refill is None:
                    raise MissingReturnsError(f"{len(bad)} bad dates under Refill but the source has no {mr.refill!r}")
                R[i] = np.where(np.isfinite(refill[i]), refill[i], R[i])
                missing["refilled"] += int(np.isfinite(refill[i]).sum())
            else:
                R[i] = 0.0
        if mr.policy == MissingReturnPolicy.DROP_DATE:
            missing["dropped"] = len(bad)

    # ---------------------------------------------------------------- cash, overlays, horizon
    fin = s.financing
    if fin.series and fin.series in prep.series.columns:
        cash = np.nan_to_num(prep.series[fin.series].reindex(dates).to_numpy(dtype=np.float64))
    else:
        cash = np.zeros(T)
        if fin.series:
            notes.append(f"cash series {fin.series!r} not in the source: cash earns zero")
    dt = np.concatenate([[1 / 252], np.diff(dates.values).astype("timedelta64[D]").astype(float) / 365.0])
    ov = prep.overlays if s.overlay is not None else None
    K = len(ov.ids) if ov is not None else 0
    ovR = np.nan_to_num(np.asarray(ov[s.returns], dtype=np.float64)) if K else np.zeros((T, 0))
    ov_fin = np.array([k in s.overlay.financed for k in ov.ids], dtype=bool) if K else np.zeros(0, dtype=bool)

    # ---------------------------------------------------------------- vectorised targets
    cons = s.portfolio
    targets = None
    vector_path = s.decider is None and not cons.path_dependent and not isinstance(cal, Triggered) and len(dec_rows) > 0
    if vector_path:
        ta = time.perf_counter()
        B = Batch(None if S is None else np.where(U[dec_rows], S[dec_rows], np.nan), U[dec_rows], None if Wb is None else Wb[dec_rows], None, ctx, dec_rows)
        targets = cons.batch(B)
        timings["targets_seconds"] = time.perf_counter() - ta
    needs_cs = targets is None or s.overlay is not None or bool(cons.constraints())

    # ---------------------------------------------------------------- the loop
    lag = s.execution.lag
    cost_model = s.execution.cost
    part = s.execution.participation
    record = st.record_weights if st.record_weights is not None else (T * N <= 60_000_000)
    W = np.zeros((T, N), dtype=np.float32) if record else None
    w = np.zeros(N)
    o = np.zeros(K)
    nav = float(s.capital)
    gross, net, costs, navs = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    cash_w, gexp, nexp, names = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T, dtype=np.int64)
    cash_ret, fin_ret, ov_ret = np.zeros(T), np.zeros(T), np.zeros(T)
    pending: dict[int, tuple] = {}
    trades_rows, weight_rows, cons_rows, viol_rows, ov_rows = [], [], [], [], []
    last_decision = None
    borrow, short_fee = fin.borrow_spread_bp / 1e4, fin.short_fee_bp / 1e4
    id_pos = {k: i for i, k in enumerate(ids)}
    tl = time.perf_counter()
    for t in range(t_lo, t_hi + 1):
        # 1) earn the period on the weights held at the previous close
        if t > t_lo:
            rt = R[t]
            held = w != 0
            dead = held & ~alive[t]
            if dead.any():
                missing["redeemed"] += int(dead.sum())
                w = np.where(dead, 0.0, w)
                held = w != 0
            gaps = held & ~np.isfinite(rt)
            if gaps.any():
                if mr.policy == MissingReturnPolicy.ZERO:
                    missing["zero_filled"] += int(gaps.sum())
                elif mr.policy == MissingReturnPolicy.REFILL and refill is not None and np.isfinite(refill[t][gaps]).all():
                    rt = np.where(gaps, refill[t], rt)
                    missing["refilled"] += int(gaps.sum())
                elif mr.policy == MissingReturnPolicy.DROP_DATE:
                    missing["zero_filled"] += int(gaps.sum())
                else:
                    raise MissingReturnsError(f"{int(gaps.sum())} missing {s.returns!r} returns on held instruments on {dates[t].date()} ({list(ids[gaps][:5])}) under {mr!r}; choose ZeroFill(), DropDates() or Refill(field) explicitly")
            rc = np.where(held, np.nan_to_num(rt), 0.0)
            asset = float(w @ rc)
            cw = 1.0 - float(w.sum())
            cp = cw * cash[t] if cw >= 0 else cw * (cash[t] + borrow * dt[t])
            sp = float(-w[w < 0].sum()) * short_fee * dt[t]
            op = float(o @ (ovR[t] - np.where(ov_fin, cash[t], 0.0))) if K else 0.0
            g = asset + cp - sp + op
            gross[t], cash_ret[t], fin_ret[t], ov_ret[t] = g, cp, -sp, op
            w = w * (1.0 + rc) / (1.0 + g)
            if K:
                o = o * (1.0 + ovR[t]) / (1.0 + g)
        # 2) decide at the close
        k = dec_index.get(t)
        if k is not None:
            out = _decide(t, k, s, ctx, targets, S, U, Wb, w, o, nav, needs_cs, ov, cons_rows, viol_rows, last_decision, id_pos, st)
            if out is not None:
                if t + lag <= t_hi:
                    pending[t + lag] = (out[0], out[1], t)
                last_decision = t
        # 3) trade on the close
        c = 0.0
        if t in pending:
            tw, to, dec_t = pending.pop(t)
            tw = np.where(alive[t] | (tw == 0), tw, 0.0)
            dw = tw - w
            fill = 1.0
            if part is not None:
                cap = part.capacity(ctx, t, nav)
                ex = np.clip(dw, -cap, cap)
                short_by = np.abs(dw - ex).sum()
                if short_by > 1e-12:
                    fill = float(np.abs(ex).sum() / max(np.abs(dw).sum(), 1e-18))
                    if t + 1 <= t_hi and (t + 1) not in pending:
                        pending[t + 1] = (tw, to, dec_t)
                dw = ex
            if np.any(dw != 0):
                c = cost_model.total(ctx, t, dw, nav)
            if K:
                c += float(np.abs(to - o).sum()) * s.overlay.cost_bp / 1e4
            tov = turnover_vectors(w, w + dw)
            trades_rows.append({"decision_date": dates[dec_t], "trade_date": dates[t], **tov, "cost": c, "n_buys": int((dw > 1e-12).sum()), "n_sells": int((dw < -1e-12).sum()), "fill_ratio": fill})
            w = w + dw
            w[np.abs(w) < 1e-15] = 0.0
            if K:
                o = to.copy()
                if np.any(o):
                    ov_rows.append(pd.DataFrame({"date": dates[t], "instrument_id": ov.ids, "notional": o}))
            nz = np.nonzero(w)[0]
            weight_rows.append((t, nz, w[nz].copy()))
        costs[t] = c
        net[t] = (1.0 + gross[t]) * (1.0 - c) - 1.0
        nav *= 1.0 + net[t]
        navs[t] = nav
        cash_w[t] = 1.0 - w.sum()
        gexp[t], nexp[t], names[t] = float(np.abs(w).sum()), float(w.sum()), int((w != 0).sum())
        if record:
            W[t] = w
    timings["loop_seconds"] = time.perf_counter() - tl

    # ---------------------------------------------------------------- bulk analytics
    ta = time.perf_counter()
    win = slice(t_lo, t_hi + 1)
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
                X = np.nan_to_num(np.asarray(ctx.field(fname))[win]).astype(np.float32, copy=False)
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
    if s.metrics.risk_free and s.metrics.risk_free in prep.series.columns:
        rf = prep.series[s.metrics.risk_free].reindex(idx)
    diag = {"timings": timings, "lookahead": lookahead, "notes": notes, "data": prep.info, "cache": {"entries": len(prep.cache), "hits": prep.cache.hits, "misses": prep.cache.misses}, "vectorised_targets": targets is not None, "n_decisions": len(dec_rows), "universe_avg": float(U[win].sum(axis=1).mean())}
    res = Result(st.name or s.name, strategy.fingerprint, "paper", s, s.conventions(), pd.Series(net[win], index=idx, name="net"), pd.Series(gross[win], index=idx, name="gross"), pd.Series(costs[win], index=idx, name="cost"), pd.Series(navs[win], index=idx, name="nav"), bench_series, daily, pd.DataFrame(trades_rows), weights, exposures, groups, contributions, pd.DataFrame(cons_rows), pd.DataFrame(viol_rows), missing, excess, rf, diag, W[win] if record else None, ids if record else None, None, prep.info.get("source_version"), (str(idx[0].date()), str(idx[-1].date())))
    if ov_rows:
        res.diagnostics["overlays"] = pd.concat(ov_rows, ignore_index=True)
    return res


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
    return KernelCross(ctx.panel.dates[t], ids, df, pd.Series(w[cols], index=ids), pd.Series(wbrow[cols] if wbrow is not None else np.zeros(len(cols)), index=ids), scores, pd.Series(urow[cols], index=ids), cost_bp, min_w, None, nav, ctx, t, cols)


def _decide(t, k, s: Strategy, ctx, targets, S, U, Wb, w, o, nav, needs_cs, ov, cons_rows, viol_rows, last_decision, id_pos, st):
    ids = ctx.panel.ids
    N = len(ids)
    date = ctx.panel.dates[t]
    diag: dict = {}
    cs = None
    overlay_target = o.copy()
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
            cs = _cross(t, s, ctx, row, urow, None if Wb is None else Wb[t], w, nav)
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
        wt, diag = s.portfolio.weights(cs)
        tw = _to_vector(wt, id_pos, N)
    viol = []
    if cs is not None:
        wt_cs = pd.Series(tw[cs.cols], index=cs.ids)
        viol = diag.pop("_violations", None) or s.portfolio.check(wt_cs, cs)
        if s.overlay is not None and ov is not None:
            ov_data = pd.DataFrame({f: np.asarray(ov[f][t], dtype=np.float64) for f in s.overlay.fields if f in ov}, index=ov.ids)
            sized, odiag = s.overlay.size(wt_cs, cs, ov_data)
            overlay_target = sized.reindex(ov.ids).fillna(0.0).to_numpy()
            diag["overlay"] = odiag
    if isinstance(s.schedule, Triggered) and last_decision is not None:
        cal = s.schedule
        if cal.measure == "signal" and S is not None:
            prev, now = S[last_decision], S[t]
            both = np.isfinite(prev) & np.isfinite(now)
            ch = float(np.abs(pd.Series(now[both]).rank(pct=True).to_numpy() - pd.Series(prev[both]).rank(pct=True).to_numpy()).mean()) if both.any() else 0.0
        else:
            ch = turnover_vectors(w, tw)["HALF_SUM"]
        if ch < cal.threshold:
            cons_rows.append({"date": date, "action": "trigger_skip", "measure": cal.measure, "value": ch, "threshold": cal.threshold})
            return None
    flat = {kk: vv for kk, vv in diag.items() if not isinstance(vv, (dict, list, pd.DataFrame, pd.Series))}
    if diag.get("binding"):
        flat["binding"] = "; ".join(str(b) for b in diag["binding"][:30])
    if isinstance(diag.get("optimiser"), dict):
        flat.update({f"opt_{kk}": vv for kk, vv in diag["optimiser"].items() if not isinstance(vv, (dict, list))})
    cons_rows.append({"date": date, "action": "rebalance", "n_names": int((tw != 0).sum()), "gross": float(np.abs(tw).sum()), "net": float(tw.sum()), **flat, "n_violations": len(viol)})
    for v in viol:
        viol_rows.append({"date": date, **v})
    return tw, overlay_target


def _to_vector(wt: pd.Series, id_pos: dict, N: int) -> np.ndarray:
    tw = np.zeros(N)
    if len(wt):
        pos = np.array([id_pos.get(str(i), -1) for i in wt.index])
        ok = pos >= 0
        tw[pos[ok]] = np.asarray(wt, dtype=float)[ok]
    return tw


__all__ = ["Prepared", "RunSettings", "prepare", "run", "simulate"]
