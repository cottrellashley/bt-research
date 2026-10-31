"""Sessions: a research thread that counts every trial, caches every result and shares data.

A PM's "what if" is rarely one run. A ``Session`` is the thread those runs belong to:

* **every run is a trial** in a ledger (``governance.TrialLedger``: one atomic file per trial,
  safe with many agents writing at once), tagged with the session, the strategy's fingerprint,
  its description and its statistics. A repeated run of the same strategy on the same data and
  window is served from the cache and not counted twice;
* **selection statistics see the whole search**: ``deflated_sharpe(result)`` deflates by the
  number of distinct variants this session tried, ``pbo()`` estimates the probability that the
  best of them is overfit;
* **data load once**: runs share one prepared panel and one expression cache, so variants that
  share a universe, benchmark or signal compute it once;
* **sweeps** run variants (``{name: strategy}``, a base plus a grid of paths, or a builder plus
  keyword grids) sequentially on the shared cache, or in worker processes that read one
  memory-mapped snapshot of the data;
* ``promote(result, level)`` submits a result through the official-record gates.

``Session(root=...)`` persists the ledger, records and a result cache under ``root``; without a
root the session lives in a temporary directory.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import itertools
import json
import os
import pickle
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nautilus_fi.governance.ledger import TrialLedger
from nautilus_fi.kernel.data import DataSource, as_source, save_panel
from nautilus_fi.kernel.engine import Prepared, RunSettings, prepare, simulate
from nautilus_fi.kernel.result import Comparison, Result
from nautilus_fi.kernel.strategy import Strategy

_TEMP_DIRS: list[str] = []


def _cleanup() -> None:
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup)


def _window(start, end, window):
    if window is not None:
        start, end = window
    return (None if start is None else pd.Timestamp(str(start)).normalize(), None if end is None else (pd.Timestamp(str(end)) + (pd.offsets.YearEnd(0) if len(str(end)) == 4 else pd.Timedelta(0))).normalize())


@dataclasses.dataclass
class Variant:
    label: str
    params: dict
    strategy: Strategy


def variants_of(base, grid: Mapping | None = None) -> list[Variant]:
    """``{label: strategy}``, ``[strategy, ...]``, ``(base, {"path": [values]})`` or
    ``(builder, {"kw": [values]})`` -> variants."""
    if isinstance(base, Mapping):
        return [Variant(str(k), {}, v.with_(name=str(k)) if v.name == "strategy" else v) for k, v in base.items()]
    if isinstance(base, (list, tuple)) and grid is None:
        return [Variant(s.name if sum(x.name == s.name for x in base) == 1 else f"{s.name}#{i}", {}, s) for i, s in enumerate(base)]
    if grid is None:
        if isinstance(base, Strategy):
            return [Variant(base.name, {}, base)]
        raise TypeError("sweep needs strategies, or a base strategy / builder and a grid")
    keys = list(grid)
    out = []
    for combo in itertools.product(*[list(grid[k]) for k in keys]):
        params = dict(zip(keys, combo))
        label = ",".join(f"{k.split('.')[-1]}={_lbl(v)}" for k, v in params.items())
        if isinstance(base, Strategy):
            s = base
            for k, v in params.items():
                s = s.set(k, v)
            s = s.with_(name=f"{base.name}[{label}]")
        elif callable(base):
            s = base(**params)
            if not isinstance(s, Strategy):
                raise TypeError(f"the builder returned {type(s).__name__}, not a Strategy")
            if s.name == "strategy":
                s = s.with_(name=f"[{label}]")
        else:
            raise TypeError("a grid needs a base Strategy or a builder callable")
        out.append(Variant(label, params, s))
    return out


def _lbl(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    r = repr(v)
    return r if len(r) <= 40 else r[:37] + "..."


@dataclasses.dataclass
class SweepResult:
    variants: list[Variant]
    results: list[Result]
    metric: str
    session: Session
    seconds: float
    mode: str = "sequential"
    notes: list = dataclasses.field(default_factory=list)

    @property
    def comparison(self) -> Comparison:
        return Comparison(list(self.results))

    def table(self, keys=("annual_return", "annual_vol", "sharpe", "information_ratio", "tracking_error", "turnover", "cost_drag", "max_drawdown")) -> pd.DataFrame:
        rows = []
        for v, r in zip(self.variants, self.results):
            s = r.stats()
            rows.append({"variant": v.label, **{f"param.{k}": _lbl(x) for k, x in v.params.items()}, **{k: s.get(k) for k in keys if k in s}})
        df = pd.DataFrame(rows).set_index("variant")
        return df.sort_values(self.metric, ascending=False) if self.metric in df else df

    def best(self) -> Result:
        return self.comparison.best(self.metric)

    def selection(self) -> dict:
        """The best variant's deflated Sharpe ratio given every trial in the session, and PBO
        across the sweep's variants."""
        best = self.best()
        return {"best": best.name, "metric": self.metric, "value": best.stats().get(self.metric), "deflated_sharpe": self.session.deflated_sharpe(best), "pbo": self.session.pbo(results=self.results)}

    def __repr__(self) -> str:
        return f"SweepResult({len(self.results)} variants, {self.seconds:.1f}s, {self.mode})\n{self.table().to_string()}"


class Session:
    def __init__(self, root: str | Path | None = None, source=None, name: str | None = None, hypothesis=None, start=None, end=None, window=None, settings: RunSettings | None = None, cache_bytes: float = 6e9) -> None:
        self.name = name or f"session-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if root is None:
            root = tempfile.mkdtemp(prefix="nfi-session-")
            _TEMP_DIRS.append(root)
            self.persistent = False
        else:
            self.persistent = True
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = TrialLedger(self.root / "ledger")
        self.source = as_source(source) if source is not None else None
        self.start, self.end = _window(start, end, window)
        self.settings = settings or RunSettings()
        self.cache_bytes = cache_bytes                 # bound on evaluated expressions kept in memory
        self.hypothesis = None
        if hypothesis is not None:
            self.hypothesis = hypothesis if isinstance(hypothesis, str) else hypothesis.hypothesis_id
            if not isinstance(hypothesis, str) and self.ledger.hypothesis(self.hypothesis) is None:
                self.ledger.register(hypothesis)
        self._results: dict[str, Result] = {}
        self._recorded: set[str] = {self._trial_key(t["spec_hash"], tuple(t["period"]), t["stage"]) for t in self.ledger.trials_list() if t.get("session") == self.name}
        self._prepared: Prepared | None = None
        self._prep_source: DataSource | None = None
        self._prep_strategies: list[Strategy] = []
        self.log: list[dict] = []

    # ------------------------------------------------------------------ plumbing
    def _source(self, source) -> DataSource:
        src = as_source(source) if source is not None else self.source
        if src is None:
            raise ValueError("no data source: pass source= or create the session with Session(source=...)")
        return src

    def key(self, strategy: Strategy, source: DataSource, start, end, fidelity: str) -> str:
        v = source.version() or f"id{id(source)}"
        return hashlib.sha256(json.dumps([strategy.fingerprint, v, str(start), str(end), fidelity]).encode()).hexdigest()[:20]

    def prepared(self, strategies: list[Strategy], source: DataSource, start, end) -> Prepared:
        """The session's shared data, reloaded only when a strategy needs more."""
        p = self._prepared
        if p is not None and self._prep_source is source and p.start == start and p.end == end and all(p.covers(s) for s in strategies):
            return p
        keep = self._prep_strategies if (p is not None and self._prep_source is source and p.start == start and p.end == end) else []
        allv = [*keep, *strategies]
        from nautilus_fi.kernel.expr import ExprCache

        self._prepared = prepare(allv, source, start, end, cache=ExprCache(self.cache_bytes))
        self._prep_source = source
        self._prep_strategies = allv[-50:]
        return self._prepared

    def _cache_path(self, key: str) -> Path:
        return self.root / "results" / f"{key}.pkl"

    def _cached(self, key: str) -> Result | None:
        if key in self._results:
            return self._results[key]
        p = self._cache_path(key)
        if self.persistent and p.exists():
            try:
                r = pickle.loads(p.read_bytes())
                self._results[key] = r
                return r
            except Exception:                        # noqa: BLE001 - stale or foreign pickle: recompute
                return None
        return None

    def _store(self, key: str, r: Result) -> None:
        self._results[key] = r
        if self.persistent:
            try:
                from nautilus_fi.core.fileio import atomic_write_bytes

                atomic_write_bytes(self._cache_path(key), pickle.dumps(dataclasses.replace(r, daily_weights=None, raw=None)))
            except (pickle.PicklingError, AttributeError, TypeError):    # unpicklable custom code: memory cache only
                self.log.append({"note": f"{r.name}: not cached on disk (unpicklable custom code)"})

    @staticmethod
    def _trial_key(fingerprint: str, period: tuple, stage: str) -> str:
        return f"{fingerprint}|{period[0]}|{period[1]}|{stage}"

    def _record(self, key: str, r: Result, stage: str) -> None:
        """One trial per (strategy, period, stage) per session, however often it is run."""
        rets = r.returns.iloc[1:]
        period = (str(rets.index[0].date()), str(rets.index[-1].date()))
        tk = self._trial_key(r.fingerprint, period, stage)
        if tk in self._recorded:
            return
        s = r.strategy
        hyp = s.hypothesis or self.hypothesis
        if hyp is not None and self.ledger.hypothesis(hyp) is None:
            hyp = None
        summary = {k: v for k, v in r.stats().items() if isinstance(v, (int, float, str, bool)) or v is None}
        self.ledger.record(stage, r.name, r.fingerprint, s.variant(), period, summary, rets, hyp, r.conventions.to_dict(), False, "COMPLETED", "", session=self.name, description=s.describe())
        self._recorded.add(tk)

    # ------------------------------------------------------------------ running
    def run(self, strategy: Strategy, start=None, end=None, window=None, fidelity: str = "paper", source=None, record: bool = True, settings: RunSettings | None = None, **implement_kw) -> Result:
        """Run (or fetch from the cache) one strategy; every new run is a trial in the ledger."""
        src = self._source(source)
        start, end = _window(start, end, window) if (start is not None or end is not None or window is not None) else (self.start, self.end)
        key = self.key(strategy, src, start, end, fidelity)
        hit = self._cached(key)
        t0 = time.perf_counter()
        if hit is None:
            if fidelity == "paper":
                prep = self.prepared([strategy], src, start, end)
                hit = simulate(strategy, prep, start, end, settings or self.settings)
            elif fidelity == "implement":
                from nautilus_fi.kernel.implement import run_implement

                hit = run_implement(strategy, src, start, end, **implement_kw)
            else:
                raise ValueError(f"fidelity must be 'paper' or 'implement', not {fidelity!r}")
            self._store(key, hit)
            cached = False
        else:
            cached = True
        if record:
            self._record(key, hit, "PAPER" if fidelity == "paper" else "IMPLEMENT")
        self.log.append({"name": hit.name, "fingerprint": hit.fingerprint, "fidelity": fidelity, "cached": cached, "seconds": time.perf_counter() - t0, "key": key})
        return hit

    def compare(self, *strategies: Strategy, start=None, end=None, window=None, fidelity: str = "paper", source=None) -> Comparison:
        """Run each strategy (sharing data and caches) and line the results up; the first is the base."""
        src = self._source(source)
        start, end = _window(start, end, window) if (start is not None or end is not None or window is not None) else (self.start, self.end)
        if fidelity == "paper":
            self.prepared(list(strategies), src, start, end)
        results = [self.run(s, start, end, fidelity=fidelity, source=src) for s in strategies]
        n = self.n_trials()
        notes = [f"{n} trials in session {self.name!r} so far: one comparison's t-statistic does not account for the search; use session.deflated_sharpe(result) or sweep(...).selection()"] if n > 5 else []
        return Comparison(results, notes=notes)

    def sweep(self, base, grid: Mapping | None = None, metric: str = "sharpe", start=None, end=None, window=None, source=None, workers: int = 1, keep_weights: bool = False) -> SweepResult:
        """Run variants: ``{label: strategy}``, ``[strategies]``, ``(base, {"path": values})`` or
        ``(builder, {"kw": values})``. ``workers > 1`` runs them in processes reading one memory-
        mapped snapshot of the data (variants with unpicklable custom code run here)."""
        t0 = time.perf_counter()
        src = self._source(source)
        start, end = _window(start, end, window) if (start is not None or end is not None or window is not None) else (self.start, self.end)
        vs = variants_of(base, grid)
        prep = self.prepared([v.strategy for v in vs], src, start, end)
        results: list[Result | None] = [None] * len(vs)
        todo = []
        for i, v in enumerate(vs):
            key = self.key(v.strategy, src, start, end, "paper")
            hit = self._cached(key)
            if hit is not None:
                results[i] = hit
                self._record(key, hit, "PAPER")
            else:
                todo.append((i, key))
        mode, notes = "sequential", []
        if workers > 1 and len(todo) > 1:
            par, seq = [], []
            for i, key in todo:
                try:
                    payload = pickle.dumps(vs[i].strategy)
                    (seq if b"__main__" in payload else par).append((i, key))
                except (pickle.PicklingError, AttributeError, TypeError):
                    seq.append((i, key))
            if seq:
                notes.append(f"{len(seq)} variant(s) with custom code defined in a script or lambda ran in this process (workers cannot import it)")
            if len(par) == 1:
                seq, par = seq + par, []
            if par:
                mode = f"{min(workers, len(par))} processes"
                for (i, key), r in zip(par, _run_parallel([vs[i].strategy for i, _ in par], prep, start, end, workers, self.root, self.settings, keep_weights)):
                    results[i] = r
                    self._store(key, r)
                    self._record(key, r, "PAPER")
            todo = seq
        for i, key in todo:
            r = simulate(vs[i].strategy, prep, start, end, self.settings)
            if not keep_weights:
                r = dataclasses.replace(r, daily_weights=None)
            results[i] = r
            self._store(key, r)
            self._record(key, r, "PAPER")
        return SweepResult(vs, results, metric, self, time.perf_counter() - t0, mode, notes)

    # ------------------------------------------------------------------ selection statistics
    def trials(self) -> pd.DataFrame:
        return self.ledger.trials(session=self.name)

    def n_trials(self) -> int:
        return self.ledger.n_trials(None, ("PAPER", "IMPLEMENT", "VALIDATION"), session=self.name)

    def deflated_sharpe(self, result: Result) -> dict:
        """The result's Sharpe ratio deflated by every distinct trial in this session."""
        r = result.returns.iloc[1:]
        ppy = result.strategy.resolved().metrics.periods_per_year(r.index)
        return self.ledger.deflated_sharpe(None, r, ppy, ("PAPER", "VALIDATION"), session=self.name)

    def pbo(self, n_splits: int = 8, results: list[Result] | None = None) -> dict:
        """Probability of backtest overfitting across the session's trials (or ``results``)."""
        if results is not None:
            from nautilus_fi.research.statistics import probability_of_backtest_overfitting

            m = pd.DataFrame({r.name: r.returns.iloc[1:] for r in results}).dropna()
            if m.shape[1] < 2 or len(m) < n_splits * 4:
                return {"pbo": float("nan"), "n_trials": m.shape[1], "note": "needs at least two variants on common dates"}
            out = probability_of_backtest_overfitting(m.to_numpy(), n_splits=n_splits)
            out["n_trials"] = m.shape[1]
            return out
        return self.ledger.pbo(None, n_splits, ("PAPER", "VALIDATION"), session=self.name)

    # ------------------------------------------------------------------ governance
    def promote(self, result: Result, level: str = "CANDIDATE", implement: Result | None = None, reconciliation: dict | None = None, registry_root: str | Path | None = None):
        """Submit through the official-record gates (``governance.RecordRegistry``); raises
        ``GateFailure`` listing every failed check."""
        from nautilus_fi.governance.records import RecordLevel, RecordRegistry

        reg = RecordRegistry(registry_root or (self.root / "records"))
        lvl = level if isinstance(level, RecordLevel) else RecordLevel(str(level).upper())
        spec = _SpecProxy(result.fingerprint, result.strategy.hypothesis or self.hypothesis)
        raw = implement.raw if implement is not None else None
        if implement is not None and reconciliation is None:
            from nautilus_fi.kernel.implement import reconcile

            reconciliation = reconcile(result, implement, source=self.source, prepared=self._prepared).summary
        data = {"source_version": result.source_version or "", "fingerprint": result.fingerprint}
        return reg.submit(lvl, result, spec, self.ledger, raw, reconciliation, data, result.name)

    def summary(self) -> str:
        t = self.trials()
        return f"session {self.name}: {0 if t.empty else len(t)} trials recorded, {len(self._results)} results cached, root {self.root}{'' if self.persistent else ' (temporary)'}"

    def __repr__(self) -> str:
        return self.summary()


@dataclasses.dataclass
class _SpecProxy:
    hash: str
    hypothesis_id: str | None


# ====================================================================================== parallel sweeps
def _run_parallel(strategies: list[Strategy], prep: Prepared, start, end, workers: int, root: Path, settings: RunSettings, keep_weights: bool) -> list[Result]:
    """Run ``strategies`` in ``workers`` subprocesses (``kernel.worker``) that read one memory-mapped
    snapshot of the prepared data."""
    import subprocess
    import sys

    from nautilus_fi.core.fileio import atomic_write_bytes

    snap = root / "snapshots" / uuid.uuid4().hex[:12]
    try:
        save_panel(prep.panel, snap / "panel")
        if prep.overlays is not None:
            save_panel(prep.overlays, snap / "overlays")
        constituents = {ix: prep.source.constituents(ix) for ix in prep.requirements.indices}
        atomic_write_bytes(snap / "meta.pkl", pickle.dumps({"attributes": prep.attributes, "series": prep.series, "constituents": constituents, "fields": list(prep.catalog), "version": prep.info.get("source_version")}))
        n = min(workers, len(strategies))
        chunks: list[list] = [[] for _ in range(n)]
        for i, s in enumerate(strategies):
            chunks[i % n].append((i, pickle.dumps(s)))
        for k, chunk in enumerate(chunks):
            atomic_write_bytes(snap / f"tasks{k}.pkl", pickle.dumps({"start": start, "end": end, "settings": settings, "keep_weights": keep_weights, "tasks": chunk}))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        procs = [subprocess.Popen([sys.executable, "-m", "nautilus_fi.kernel.worker", str(snap), str(k)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) for k in range(n)]
        out: dict[int, Result] = {}
        for k, p in enumerate(procs):
            _, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"sweep worker {k} failed:\n{err.decode(errors='replace')[-3000:]}")
            out.update(pickle.loads((snap / f"out{k}.pkl").read_bytes()))
    finally:
        shutil.rmtree(snap, ignore_errors=True)
    return [out[i] for i in range(len(strategies))]


__all__ = ["Session", "SweepResult", "Variant", "variants_of"]
