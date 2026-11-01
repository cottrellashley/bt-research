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
from nautilus_fi.kernel.strategy import REPORT_EXPOSURES, Strategy


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
        kc = KernelCross(cs.date, cs.ids, cs.data, cs.current, cs.benchmark, cs.scores, cs.eligible, cs.cost_bp, cs.min_weight, cs.scenarios, cs.nav, e.ctx, t, e.cols(cs.ids), s.exits.value)
        joint = s.overlay is not None and getattr(s.overlay, "joint", False) and overlay_data is not None
        if joint:                                           # the hedge solved with the book, as in the Paper engine
            kc.overlay_data = overlay_data[[f for f in s.overlay.fields if f in overlay_data.columns]].astype(float).fillna(0.0)
            kc.overlay_current = np.zeros(len(kc.overlay_data))
            kc.overlay_cost_bp, kc.overlay_max = s.overlay.cost_bp, s.overlay.max_gross
        weights, diag = s.portfolio.weights(kc)
        joint_ov = diag.pop("overlays", None) if joint else None
        if joint_ov is not None:
            kc.overlay_target = joint_ov.reindex(kc.overlay_data.index).fillna(0.0).to_numpy()
        w0 = cs.current.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float)
        leavers = (w0 != 0) & ~cs.eligible.reindex(cs.ids).fillna(False).to_numpy(dtype=bool)
        if leavers.any():                                   # the strategy's exit policy, as in the Paper engine
            from nautilus_fi.kernel.engine import _apply_exits

            tw, _ = _apply_exits(s.exits, weights.reindex(cs.ids).fillna(0.0).to_numpy(dtype=float), w0, leavers, getattr(s.portfolio, "handles_exits", False))
            weights = pd.Series(tw, index=cs.ids)
            weights = weights[weights.abs() > 1e-12]
        violations = diag.pop("_violations", None) or s.portfolio.check(weights, kc)
        overlays = None
        if s.overlay is not None and overlay_data is not None:
            overlays, diag["overlay"] = (joint_ov, {"method": "solved with the book"}) if joint_ov is not None else s.overlay.size(weights, kc, overlay_data)
        diag.setdefault("method", type(s.portfolio).__name__)
        return ConstructionOutput(weights, overlays, diag, violations, bool(diag.get("ok", True)))

    def to_dict(self) -> dict:
        return {"kernel": repr(self._e.s.portfolio), "fingerprint": self._e.s.fingerprint}


BYTES_PER_BOND_DAY = 2048                                   # event-engine objects per bond-day (1-2 KB measured: tools/probes/implement_scale.py)


def _available_memory() -> float | None:
    try:
        for line in open("/proc/meminfo"):                      # noqa: SIM115 - two lines of a pseudo-file
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) * 1024.0
    except OSError:
        pass
    try:
        import os

        return float(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) * 0.5
    except (ValueError, OSError, AttributeError):
        return None


def implement_memory(bond_days: int) -> float:
    """Bytes the event engine needs for ``bond_days`` (instrument x date pairs alive in the run)."""
    return float(bond_days) * BYTES_PER_BOND_DAY


def run_implement(strategy: Strategy, source, start=None, end=None, env=None, settings=None, execution=None, prepared: Prepared | None = None, universe_ids=None, forced_actions=None, memory_limit_gb: float | str | None = "auto") -> Result:
    """Run ``strategy`` in the event engine on ``env`` (default: ``source.environment``).
    ``forced_actions``: a ``portfolio.ForcedActionPolicy`` (downgrades, index removals, maturity
    bands) for the event engine, when ``settings`` is not given. Implement keeps an object per
    bond-day (``BYTES_PER_BOND_DAY``): before building the environment the run's bond-days are
    sized against ``memory_limit_gb`` ("auto": 80% of the memory available now; None: no check)
    and a run that does not fit raises ``MemoryError`` naming the remedies (a sample of names,
    a shorter window, or the Paper engine, Implement being the validation tier)."""
    from nautilus_fi.backtest import BacktestSettings, FixedIncomeBacktest
    from nautilus_fi.backtest.spec_strategy import SpecStrategy, SpecStrategyConfig
    from nautilus_fi.execution import NET_SPREAD

    t0 = time.perf_counter()
    s = strategy.resolved()
    if not isinstance(s.returns, str):                                  # a fidelity choice, not a wrong argument type
        raise ValueError("Implement values positions from prices and terms: a strategy whose returns are an expression runs at Paper fidelity only")  # noqa: TRY004
    if s.hold != "drift":
        raise ValueError("Implement holds positions in face, so weights move with prices between rebalances: hold='constant' runs at Paper fidelity only")
    prep = prepared or prepare([s], source, start, end)
    ev = _Evaluated(s, prep)
    lo = 0 if prep.start is None else int(ev.dates.searchsorted(prep.start))
    if env is None:
        ids = universe_ids
        if ids is None:
            ever = ev.U[lo:].any(axis=0)
            ids = list(ev.ids[ever])
        if memory_limit_gb is not None:
            pos = ev.ids.get_indexer(pd.Index(ids))
            alive = alive_mask(prep.ctx_for(s))[lo:, pos[pos >= 0]]
            need = implement_memory(int(alive.sum()))
            limit = (0.8 * (_available_memory() or float("inf"))) if memory_limit_gb == "auto" else float(memory_limit_gb) * 1e9
            if need > limit:
                raise MemoryError(f"Implement needs about {need / 1e9:.1f} GB for {int(alive.sum()):,} bond-days ({len(ids):,} names), over the {limit / 1e9:.1f} GB limit: run a sample of names (universe_ids=), a shorter window, or the Paper engine (Implement is the validation tier); memory_limit_gb=None skips the check")
        env = prep.source.environment(prep.start or ev.dates[0], prep.end or ev.dates[-1], ids)
    spec = ResearchSpec(s.name, KernelDecider(ev), KernelConstruction(ev), s.conventions(), calendar=s.schedule, hypothesis_id=s.hypothesis)
    cfg = SpecStrategyConfig(rebalance=s.schedule, universe_from_benchmark=False)
    bs = settings or BacktestSettings(starting_cash=float(s.capital), conventions=s.conventions(), forced_actions=forced_actions)
    strat = SpecStrategy(spec, cfg)
    fi = FixedIncomeBacktest(env, strat, execution or NET_SPREAD, settings=bs, name=s.name).run()
    return result_from_implement(fi, strategy, ev, time.perf_counter() - t0, getattr(strat, "spec_log", None))


def result_from_implement(fi, strategy: Strategy, ev: _Evaluated, seconds: float = 0.0, construction_log: list | None = None) -> Result:
    """The event engine's result in the kernel schema: returns, costs and turnover from its fills,
    weights from its holdings, and from those weights the same exposures, group tables,
    contributions, construction log and excess-return legs a Paper result carries."""
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
        R = np.nan_to_num(np.asarray(ev.ctx.realised(s.returns), dtype=np.float64))[np.clip(rows, 0, None)]
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
    Wb = None
    if s.benchmark is not None:
        br = s.benchmark.returns(ev.ctx)
        bench = pd.Series(br, index=ev.dates).reindex(idx)
        if bench.isna().any():
            raise ValueError(f"the Implement run has {int(bench.isna().sum())} dates the panel does not: the benchmark cannot be aligned to them")
        Wb = s.benchmark.weights(ev.ctx) if s.benchmark.has_weights else None
    daily = pd.DataFrame({"cash_weight": (fi.nav["cash"].to_numpy(dtype=float) / nav.to_numpy()) if "cash" in fi.nav else np.nan, "n_names": (W != 0).sum(axis=1) if W is not None else 0, "gross_exposure": np.abs(W).sum(axis=1) if W is not None else np.nan}, index=idx)
    exposures = pd.DataFrame(index=idx)
    excess = None
    if W is not None:
        r_ = np.clip(rows, 0, None)
        Wbr = Wb[r_] if Wb is not None else None
        for f in REPORT_EXPOSURES:
            if f in ev.ctx.panel:
                X = np.nan_to_num(np.asarray(ev.ctx.field(f), dtype=np.float64))[r_]
                exposures[f] = np.einsum("tn,tn->t", W.astype(np.float64), X)
                if Wbr is not None:
                    exposures[f"active_{f}"] = exposures[f] - np.einsum("tn,tn->t", Wbr, X)
        legs = {}
        cash = ev.prep.series[s.financing.series].reindex(idx).fillna(0.0).to_numpy() if (s.financing.series and s.financing.series in ev.prep.series) else np.zeros(len(idx))
        for name, fname in (("excess_vendor", s.metrics.excess.vendor), ("treasury_leg", s.metrics.excess.treasury)):
            if fname and fname in ev.ctx.panel:
                X = np.nan_to_num(np.asarray(ev.ctx.realised(fname), dtype=np.float64))[r_]
                leg = np.einsum("tn,tn->t", prevW.astype(np.float64), X)
                if name == "treasury_leg":
                    leg = leg + (1.0 - prevW.sum(axis=1)) * cash
                leg[0] = 0.0
                legs[name] = leg
                daily[name] = leg
        if legs:
            from nautilus_fi.domain.conventions import excess_return_series

            try:
                excess = excess_return_series(gross, None if "treasury_leg" not in legs else pd.Series(legs["treasury_leg"], index=idx), s.conventions(), None, None if "excess_vendor" not in legs else pd.Series(legs["excess_vendor"], index=idx))
            except ValueError:
                excess = None
    construction = pd.DataFrame(construction_log or [])
    raw_stats = fi.stats()
    extra = {f"implement_{k}": v for k, v in raw_stats.items() if k in ("fill_rate", "spread_cost_bp", "delay_drift_bp", "avg_delay_days", "n_fills", "status", "outcome", "funding_breach_days", "income_share_of_pnl", "ledger_balanced_all_days")}
    diag = {"timings": {"implement_seconds": seconds}, "extra_stats": extra, "notes": [], "data": ev.prep.info}
    viol = fi.construction[fi.construction.get("stage", pd.Series(dtype=str)) == "spec"] if len(fi.construction) and "stage" in fi.construction else pd.DataFrame()
    return Result(s.name, strategy.fingerprint, "implement", s, s.conventions(), net.rename("net"), gross.rename("gross"), cost.rename("cost"), nav.rename("nav"), bench, daily, trades, weights, exposures, groups, contributions, construction, viol, {"policy": s.missing.policy.value, "zero_filled": 0}, excess, None, diag, W, ev.ids if W is not None else None, fi, ev.prep.info.get("source_version"), (str(idx[0].date()), str(idx[-1].date())))


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
