"""Implement fidelity: the same ``Strategy`` through the event engine.

``run_implement(strategy, source, start, end)`` evaluates the strategy's universe and signal on the
source's panel exactly as the vectorised engine does (point in time, same caches), then runs it in
the Nautilus event engine: the universe and scores reach a ``SpecStrategy`` through
``KernelDecider``, the kernel construction builds the weights on the event engine's own
cross-section (``KernelConstruction``, which can still evaluate kernel expressions at the date),
and orders go through RFQ execution, fills at bid / ask dirty, minimum pieces, dated settlement,
funding and the ledger. The event-engine environment comes from ``source.environment(...)`` (the
synthetic source builds one; a warehouse source needs contract terms and evaluated prices) or is
passed as ``env``.

The result is a kernel ``Result`` with the same statistics as Paper (computed by the same
``Metrics``), the engine's own result in ``raw``, and implement-only figures (fill rate, spread
cost, delay drift) under ``implement_*`` keys. ``reconcile(paper, implement)`` decomposes the
return gap between the two fidelities.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from nautilus_fi.construct.methods import ConstructionOutput
from nautilus_fi.domain.spec import Decider, ResearchSpec, Target
from nautilus_fi.kernel.engine import Prepared, prepare
from nautilus_fi.kernel.expr import alive_mask
from nautilus_fi.kernel.portfolio import KernelCross
from nautilus_fi.kernel.result import Result
from nautilus_fi.kernel.strategy import Strategy


class _Evaluated:
    """The strategy's universe, signal and benchmark on the prepared panel (all rows)."""

    def __init__(self, s: Strategy, prep: Prepared) -> None:
        self.s, self.prep = s, prep
        self.ctx = prep.ctx_for(s)
        alive = alive_mask(self.ctx)
        self.U = np.asarray(s.universe.evaluate(self.ctx), dtype=bool) & alive
        self.S = None if s.signal is None else np.asarray(s.signal.evaluate(self.ctx.within(self.U, f"U:{s.universe.key}")), dtype=np.float64)
        self.dates = self.ctx.panel.dates
        self.ids = self.ctx.panel.ids
        self.pos = {k: i for i, k in enumerate(self.ids)}

    def row(self, d) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(d).normalize(), side="right")) - 1

    def cols(self, ids) -> np.ndarray:
        return np.array([self.pos.get(str(i), -1) for i in ids], dtype=np.int64)


class KernelDecider(Decider):
    """Universe and scores from the kernel evaluation, restricted to what the event engine can
    trade on the date."""

    name = "KernelDecider"

    def __init__(self, evaluated: _Evaluated) -> None:
        self._e = evaluated
        self.strategy = evaluated.s.fingerprint

    def decide(self, view):
        e = self._e
        t = e.row(view.date)
        if t < 0:
            return Target(skip=True, note="before the panel")
        cand = pd.Index(view.universe)
        cols = e.cols(cand)
        ok = cols >= 0
        ok[ok] &= e.U[t, cols[ok]]
        uni = cand[ok]
        view._universe = uni
        if e.S is None:
            return Target(scores=pd.Series(1.0, index=uni))
        sc = e.S[t, cols[ok]]
        return Target(scores=pd.Series(sc, index=uni).dropna())

    def params(self) -> dict:
        return {"strategy": self.strategy}


class KernelConstruction:
    """The strategy's kernel construction on the event engine's cross-section."""

    def __init__(self, evaluated: _Evaluated) -> None:
        self._e = evaluated
        self.constraints = []

    def build(self, cs, overlay_data=None) -> ConstructionOutput:
        e, s = self._e, self._e.s
        t = e.row(cs.date)
        kc = KernelCross(cs.date, cs.ids, cs.data, cs.current, cs.benchmark, cs.scores, cs.eligible, cs.cost_bp, cs.min_weight, cs.scenarios, cs.nav, e.ctx, t, e.cols(cs.ids))
        weights, diag = s.portfolio.weights(kc)
        violations = diag.pop("_violations", None) or s.portfolio.check(weights, kc)
        overlays = None
        if s.overlay is not None and overlay_data is not None:
            overlays, diag["overlay"] = s.overlay.size(weights, kc, overlay_data)
        diag.setdefault("method", type(s.portfolio).__name__)
        return ConstructionOutput(weights, overlays, diag, violations, bool(diag.get("ok", True)))

    def to_dict(self) -> dict:
        return {"kernel": repr(self._e.s.portfolio), "fingerprint": self._e.s.fingerprint}


def run_implement(strategy: Strategy, source, start=None, end=None, env=None, settings=None, execution=None, prepared: Prepared | None = None, universe_ids=None) -> Result:
    """Run ``strategy`` in the event engine on ``env`` (default: ``source.environment``)."""
    from nautilus_fi.backtest import BacktestSettings, FixedIncomeBacktest
    from nautilus_fi.backtest.spec_strategy import SpecStrategy, SpecStrategyConfig
    from nautilus_fi.execution import NET_SPREAD

    t0 = time.perf_counter()
    s = strategy.resolved()
    prep = prepared or prepare([s], source, start, end)
    ev = _Evaluated(s, prep)
    lo = 0 if prep.start is None else int(ev.dates.searchsorted(prep.start))
    if env is None:
        ids = universe_ids
        if ids is None:
            ever = ev.U[lo:].any(axis=0)
            ids = list(ev.ids[ever])
        env = prep.source.environment(prep.start or ev.dates[0], prep.end or ev.dates[-1], ids)
    spec = ResearchSpec(s.name, KernelDecider(ev), KernelConstruction(ev), s.conventions(), calendar=s.schedule, hypothesis_id=s.hypothesis)
    cfg = SpecStrategyConfig(rebalance=s.schedule, universe_from_benchmark=False)
    bs = settings or BacktestSettings(starting_cash=float(s.capital), conventions=s.conventions())
    fi = FixedIncomeBacktest(env, SpecStrategy(spec, cfg), execution or NET_SPREAD, settings=bs, name=s.name).run()
    return result_from_implement(fi, strategy, ev, time.perf_counter() - t0)


def result_from_implement(fi, strategy: Strategy, ev: _Evaluated, seconds: float = 0.0) -> Result:
    s = strategy.resolved()
    nav = fi.nav["nav"].astype(float)
    nav.index = pd.DatetimeIndex(pd.to_datetime(nav.index)).normalize()
    idx = nav.index
    net = nav.pct_change().fillna(0.0)
    f = fi.fills
    cost = pd.Series(0.0, index=idx)
    trades = pd.DataFrame()
    if f is not None and len(f):
        tr = f[f["protocol"] != "EXPIRATION"].copy()
        tr["date"] = pd.to_datetime(tr["date"]).dt.normalize()
        mv = (tr["face"] * tr["clean_price"] / 100.0).abs()
        tr["mv"] = mv
        g = tr.groupby("date")
        prev = nav.shift(1).reindex(sorted(g.groups)).bfill()
        cost = (g["spread_cost"].sum() / prev).reindex(idx).fillna(0.0)
        buys = tr[tr["side"] == "BUY"].groupby("date")["mv"].sum()
        sells = tr[tr["side"] == "SELL"].groupby("date")["mv"].sum()
        days = sorted(g.groups)
        b = buys.reindex(days).fillna(0.0) / prev
        sl = sells.reindex(days).fillna(0.0) / prev
        trades = pd.DataFrame({"decision_date": days, "trade_date": days, "ONE_WAY_BUYS": b.to_numpy(), "HALF_SUM": (0.5 * (b + sl)).to_numpy(), "TWO_WAY": (b + sl).to_numpy(), "cost": cost.reindex(days).to_numpy(), "n_buys": tr[tr["side"] == "BUY"].groupby("date").size().reindex(days).fillna(0).astype(int).to_numpy(), "n_sells": tr[tr["side"] == "SELL"].groupby("date").size().reindex(days).fillna(0).astype(int).to_numpy()})
    gross = (1.0 + net) / (1.0 - cost) - 1.0
    h = fi.holdings
    W = None
    weights = pd.DataFrame(columns=["date", "instrument_id", "weight"])
    contributions = pd.Series(dtype=float)
    groups: dict = {}
    if h is not None and len(h):
        hh = h.copy()
        hh["date"] = pd.to_datetime(hh["date"]).dt.normalize()
        wide = hh.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum").reindex(idx).fillna(0.0)
        wide = wide.div(nav, axis=0)
        cols = ev.cols(wide.columns)
        rows = np.array([ev.row(d) for d in idx])
        W = np.zeros((len(idx), len(ev.ids)), dtype=np.float32)
        ok = cols >= 0
        W[:, cols[ok]] = wide.to_numpy()[:, ok]
        t_, n_ = np.nonzero(W)
        weights = pd.DataFrame({"date": idx[t_], "instrument_id": ev.ids[n_], "weight": W[t_, n_].astype(float)})
        R = np.nan_to_num(np.asarray(ev.ctx.field(s.returns), dtype=np.float64))[np.clip(rows, 0, None)]
        prevW = np.vstack([np.zeros((1, W.shape[1]), dtype=np.float32), W[:-1]])
        C = prevW * R
        tot = C.sum(axis=0)
        nz = np.nonzero(np.abs(tot) > 0)[0]
        contributions = pd.Series(tot[nz], index=ev.ids[nz]).sort_values()
        from nautilus_fi.kernel.engine import RunSettings, _group_sum, _report_groups

        for gname, cat in _report_groups(ev.ctx, RunSettings()).items():
            codes = cat.broadcast(len(ev.dates), len(ev.ids))[np.clip(rows, 0, None)]
            G = len(cat.categories)
            if G:
                groups[gname] = {"weight": pd.DataFrame(_group_sum(W, codes, G), index=idx, columns=cat.categories), "contribution": pd.DataFrame(_group_sum(C.astype(np.float32), codes, G), index=idx, columns=cat.categories)}
    bench = None
    if s.benchmark is not None:
        br = s.benchmark.returns(ev.ctx)
        bench = pd.Series(br, index=ev.dates).reindex(idx).fillna(0.0)
    daily = pd.DataFrame({"cash_weight": (fi.nav["cash"].to_numpy(dtype=float) / nav.to_numpy()) if "cash" in fi.nav else np.nan, "n_names": (W != 0).sum(axis=1) if W is not None else 0, "gross_exposure": np.abs(W).sum(axis=1) if W is not None else np.nan}, index=idx)
    raw_stats = fi.stats()
    extra = {f"implement_{k}": v for k, v in raw_stats.items() if k in ("fill_rate", "spread_cost_bp", "delay_drift_bp", "avg_delay_days", "n_fills", "status", "outcome", "funding_breach_days", "income_share_of_pnl", "ledger_balanced_all_days")}
    diag = {"timings": {"implement_seconds": seconds}, "extra_stats": extra, "notes": [], "data": ev.prep.info}
    viol = fi.construction[fi.construction.get("stage", pd.Series(dtype=str)) == "spec"] if len(fi.construction) and "stage" in fi.construction else pd.DataFrame()
    return Result(s.name, strategy.fingerprint, "implement", s, s.conventions(), net.rename("net"), gross.rename("gross"), cost.rename("cost"), nav.rename("nav"), bench, daily, trades, weights, pd.DataFrame(index=idx), groups, contributions, pd.DataFrame(), viol, {"policy": s.missing.policy.value, "zero_filled": 0}, None, None, diag, W, ev.ids if W is not None else None, fi, ev.prep.info.get("source_version"), (str(idx[0].date()), str(idx[-1].date())))


def reconcile(paper: Result, implement: Result, source=None, prepared: Prepared | None = None, cash: pd.Series | None = None):
    """Paper vs Implement for the same strategy (``paper.reconcile`` on kernel results): holdings
    gap, cash, modelled vs actual execution costs and a residual. The holdings gap is priced with
    the source's returns, so pass the ``source`` or the session's ``prepared`` data."""
    from nautilus_fi.data.panel import Panel
    from nautilus_fi.kernel.data import DataRequest, as_source, window
    from nautilus_fi.paper.reconcile import reconcile as _reconcile

    if implement.raw is None:
        raise ValueError("the implement result carries no event-engine result")
    s = paper.strategy.resolved()
    rf = s.returns
    lo, hi = paper.returns.index[0], paper.returns.index[-1]
    if prepared is not None:
        panel = window(prepared.panel, lo, hi, fields=[rf])
        if cash is None and s.financing.series and s.financing.series in prepared.series:
            cash = prepared.series[s.financing.series]
    elif source is not None:
        src = as_source(source)
        panel = src.load(DataRequest((rf,), lo, hi, None if paper.ids is None else tuple(paper.ids)))
        if cash is None and s.financing.series and s.financing.series in src.catalog().names("series"):
            cash = src.series([s.financing.series], lo, hi)[s.financing.series]
    else:
        raise ValueError("reconcile needs the source (or the prepared data) to price the holdings gap")
    panel = Panel(panel.dates, panel.ids, {rf: np.asarray(panel.values[rf], dtype=np.float64)}, {})
    pw = SimpleNamespace(weights=paper.daily_weights_long(), returns=paper.returns, gross=paper.gross)
    return _reconcile(pw, implement.raw, panel, cash, return_field=rf)


__all__ = ["KernelConstruction", "KernelDecider", "reconcile", "result_from_implement", "run_implement"]
