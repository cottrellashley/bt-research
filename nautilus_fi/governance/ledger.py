"""The trial ledger: every research trial, recorded automatically, so selection statistics count the
whole search and not just the winner.

Each Evaluate, Paper and Implement run can write one append-only line (stage, hypothesis, spec hash,
variant parameters, period, headline statistics, conventions, holdout use) plus its return series.
From those the ledger gives the number of trials and the cross-trial variance of Sharpe ratios that
the deflated Sharpe ratio needs, and the aligned variant return matrix that PBO (CSCV) needs. The
holdout guard of ``research.ExperimentRegistry`` applies: once a hypothesis's holdout has been seen,
further runs over it are refused unless marked final.

Several processes (research agents, parallel sweeps) can write one ledger at once: each trial is
its own file written atomically (``trials/<time>-<id>.json``, returns ``returns/<id>.parquet``), and
the hypothesis registry merges under a lock. Trials carry a ``session`` (a research thread), so
selection statistics can be scoped to everything one thread tried.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_fi.research.experiment import ExperimentRecord, ExperimentRegistry, HoldoutViolation, Hypothesis
from nautilus_fi.research.statistics import deflated_sharpe_ratio, probability_of_backtest_overfitting

STAGES = ("EVALUATE", "PAPER", "IMPLEMENT", "OOS", "VALIDATION")
COUNTED_STAGES = ("PAPER", "IMPLEMENT", "VALIDATION")     # what a trial is, for every count (n_trials, DSR, PBO)


class LedgerError(RuntimeError):
    """The ledger cannot be read completely: counts built on it would understate the search."""


@dataclass
class Trial:
    trial_id: str
    hypothesis_id: str | None
    stage: str
    name: str
    spec_hash: str
    variant: dict
    period: tuple[str, str]
    summary: dict
    conventions: dict
    uses_holdout: bool
    status: str
    note: str = ""
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    returns_file: str | None = None
    session: str = ""
    description: str = ""
    data_version: str = ""                     # the source version the trial ran on
    universe: str = ""                         # the universe it selected from (search-scope matching)


def _clean(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class TrialLedger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "returns").mkdir(parents=True, exist_ok=True)
        (self.root / "trials").mkdir(parents=True, exist_ok=True)
        self.registry = ExperimentRegistry(self.root / "hypotheses.json")
        self.path = self.root / "trials.jsonl"                # legacy single-file ledgers are still read

    # ------------------------------------------------------------------ hypotheses
    def register(self, h: Hypothesis) -> Hypothesis:
        """Pre-register a hypothesis: the registry stamps the registration time (a supplied one is
        ignored) and freezes the content by hash; an id can be registered once, atomically."""
        return self.registry.register_new(h)

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        if hypothesis_id not in self.registry.hypotheses:
            self.registry.reload()
        return self.registry.hypotheses.get(hypothesis_id)

    # ------------------------------------------------------------------ recording
    def record(self, stage: str, name: str, spec_hash: str, variant: dict, period: tuple[str, str], summary: dict, returns: pd.Series | None = None, hypothesis_id: str | None = None, conventions: dict | None = None, uses_holdout: bool = False, status: str = "COMPLETED", note: str = "", session: str = "", description: str = "", data_version: str = "", universe: str = "") -> Trial:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; use one of {STAGES}")
        payload = json.dumps([stage, name, spec_hash, variant, period, hypothesis_id, session, data_version or ""], sort_keys=True, default=str)
        tid = hashlib.sha256(payload.encode()).hexdigest()[:16]
        existing = self._find(tid)
        if existing is not None:                                 # idempotent: the same trial recorded again is the same record
            if existing.uses_holdout and existing.hypothesis_id is not None:
                raise HoldoutViolation(f"holdout for {existing.hypothesis_id} already viewed (trial {existing.trial_id}): running it again is a second look")
            return existing
        if hypothesis_id is not None:
            self.registry.reload()
            if hypothesis_id not in self.registry.hypotheses:
                raise KeyError(f"hypothesis {hypothesis_id!r} is not registered: pre-register it before running trials")
            # holdout guard (raises HoldoutViolation)
            self.registry.record(ExperimentRecord(hypothesis_id, spec_hash, {k: _clean(v) for k, v in variant.items()}, period, {}, status, uses_holdout, note))
            uses_holdout = self.registry.records[-1].uses_holdout
        rfile = None
        if returns is not None and len(returns):
            from nautilus_fi.core.fileio import tmp_path

            rfile = f"returns/{tid}.parquet"
            tmp = tmp_path(self.root / rfile)
            pd.DataFrame({"date": pd.to_datetime(returns.index), "ret": returns.to_numpy(dtype=float)}).to_parquet(tmp, index=False)
            tmp.replace(self.root / rfile)
        trial = Trial(tid, hypothesis_id, stage, name, spec_hash, {k: _clean(v) for k, v in variant.items()}, (str(period[0]), str(period[1])), {k: _clean(v) for k, v in summary.items() if isinstance(_clean(v), (int, float, str, bool, type(None)))}, conventions or {}, uses_holdout, status, note, returns_file=rfile, session=session, description=description, data_version=data_version or "", universe=universe or "")
        from nautilus_fi.core.fileio import atomic_write_text

        stamp = trial.created_utc.replace(":", "").replace("-", "").replace("+", "")
        atomic_write_text(self.root / "trials" / f"{stamp}-{tid}.json", json.dumps(asdict(trial), default=str))
        return trial

    def _find(self, tid: str) -> Trial | None:
        folder = self.root / "trials"
        for f in folder.glob(f"*-{tid}.json"):
            d = json.loads(f.read_text())
            d["period"] = tuple(d["period"])
            return Trial(**{k: v for k, v in d.items() if k in Trial.__dataclass_fields__})
        return None

    def rebuild_index(self) -> tuple[pd.DataFrame, list[str]]:
        """Re-read every trial file into one index (``index.parquet``) and list problems (a returns
        file that is missing): the ledger's state rebuilt from its append-only records."""
        rows = self.trials_list()
        problems = [f"{t['trial_id']}: returns file {t['returns_file']} missing" for t in rows if t.get("returns_file") and not (self.root / t["returns_file"]).exists()]
        df = pd.DataFrame([{k: (json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else v) for k, v in t.items()} for t in rows])
        if len(df):
            df.to_parquet(self.root / "index.parquet", index=False)
        return df, problems

    def record_result(self, result, spec=None, stage: str = "PAPER", uses_holdout: bool = False, hypothesis_id: str | None = None, note: str = "") -> Trial:
        """Record a ``PaperResult`` or a ``FixedIncomeResult``."""
        hyp = hypothesis_id or (spec.hypothesis_id if spec is not None else None)
        if hasattr(result, "spec_hash"):                         # PaperResult
            r = result.returns.iloc[1:]
            summary = result.stats()
            return self.record(stage, result.name, result.spec_hash, spec.variant() if spec is not None else {}, (str(r.index[0].date()), str(r.index[-1].date())), summary, r, hyp, result.conventions.to_dict(), uses_holdout, "COMPLETED", note)
        m = result.manifest
        r = result.returns().iloc[1:]
        summary = result.stats()
        sh = (m.spec or {}).get("hash") if m is not None else None
        return self.record(stage, result.name, sh or (m.config_hash if m is not None else ""), spec.variant() if spec is not None else {}, (m.start, m.end) if m is not None else ("", ""), summary, r, hyp, m.conventions if m is not None else {}, uses_holdout, result.status, note)

    def record_evaluation(self, report, hypothesis_id: str | None = None, note: str = "") -> Trial:
        s = report.summary()
        h0 = s.index[0] if len(s) else None
        summary = {f"{k}_h{h}": v for h, row in s.iterrows() for k, v in row.items()}
        period = (str(report.ic.index.min().date()), str(report.ic.index.max().date())) if len(report.ic) else ("", "")
        series = report.ic[h0] if h0 is not None else None
        return self.record("EVALUATE", report.signal, hashlib.sha256(json.dumps({"signal": report.signal, "settings": asdict(report.settings)}, default=str, sort_keys=True).encode()).hexdigest()[:16], {"signal": report.signal}, period, summary, series, hypothesis_id, report.conventions.to_dict(), False, "COMPLETED", note)

    # ------------------------------------------------------------------ queries
    def trials_list(self) -> list[dict]:
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()] if self.path.exists() else []
        folder = self.root / "trials"
        if folder.exists():
            for f in sorted(folder.glob("*.json")):
                if f.name.startswith("."):                       # a file being written right now (atomic_write_text's temporary)
                    continue
                try:
                    rows.append(json.loads(f.read_text()))
                except (json.JSONDecodeError, OSError) as e:
                    raise LedgerError(f"trial file {f.name} cannot be read ({e}): the ledger is incomplete, so trial counts would be understated; restore or remove it explicitly") from e
        rows.sort(key=lambda t: t.get("created_utc", ""))
        return rows

    @staticmethod
    def _match(t: dict, hypothesis_id: str | None, stages, session: str | None) -> bool:
        return (hypothesis_id is None or t["hypothesis_id"] == hypothesis_id) and (stages is None or t["stage"] in stages) and (session is None or t.get("session", "") == session)

    def trials(self, hypothesis_id: str | None = None, stage: str | None = None, session: str | None = None) -> pd.DataFrame:
        rows = [t for t in self.trials_list() if self._match(t, hypothesis_id, None if stage is None else (stage,), session)]
        if not rows:
            return pd.DataFrame()
        flat = [{**{k: v for k, v in t.items() if k not in ("summary", "variant", "conventions")}, **{f"param.{k}": v for k, v in t["variant"].items()}, **{f"result.{k}": v for k, v in t["summary"].items()}} for t in rows]
        return pd.DataFrame(flat)

    @staticmethod
    def _identity(t: dict) -> tuple:
        return (t["spec_hash"], t.get("data_version", ""))

    def counted(self, hypothesis_id: str | None = None, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None, trials: list[dict] | None = None) -> list[dict]:
        """The trials counted for multiple testing, one per distinct (strategy, data version):
        Paper, Validation and Implement runs that did not use the holdout (re-running a strategy, or
        running it at another fidelity, is not a new trial). The latest record of each is kept."""
        groups: dict[tuple, list[dict]] = {}
        for t in (trials if trials is not None else self.trials_list()):
            if self._match(t, hypothesis_id, stages, session) and not t["uses_holdout"]:
                groups.setdefault(self._identity(t), []).append(t)
        out = []
        for ts in groups.values():                            # one record per trial: Paper / Validation returns over an Implement run's
            pool = [t for t in ts if t.get("returns_file")] or ts
            out.append(max(pool, key=lambda t: (t["stage"] != "IMPLEMENT", t.get("created_utc", ""))))
        return out

    def scope(self, universe: str = "", period: tuple[str, str] | None = None, data_version: str = "") -> list[dict]:
        """The search a promoted result was selected from: every counted trial, whatever hypothesis
        it was filed under (or none), on the same universe (or none recorded) whose period overlaps
        ``period`` and which ran on the same data version (or none recorded)."""
        out = []
        for t in self.trials_list():
            if universe and t.get("universe") and t["universe"] != universe:
                continue
            if data_version and t.get("data_version") and t["data_version"] != data_version:
                continue
            if period is not None and t.get("period") and (str(t["period"][1]) < str(period[0]) or str(t["period"][0]) > str(period[1])):
                continue
            out.append(t)
        return self.counted(None, COUNTED_STAGES, None, out)

    def n_trials(self, hypothesis_id: str | None, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None) -> int:
        return len(self.counted(hypothesis_id, stages, session))

    def _returns_of(self, trials: list[dict]) -> dict[str, pd.Series]:
        """Trial returns, one per distinct series: a strategy relabelled (another hypothesis, name or
        session) earns exactly what it earned before and is not a new draw."""
        out, seen = {}, set()
        for t in trials:
            if not t.get("returns_file"):
                continue
            df = pd.read_parquet(self.root / t["returns_file"])
            r = pd.Series(df["ret"].to_numpy(), index=pd.to_datetime(df["date"]))
            sig = hashlib.sha256(pd.util.hash_pandas_object(r, index=True).to_numpy().tobytes()).hexdigest()
            if sig in seen:
                continue
            seen.add(sig)
            out[t["trial_id"]] = r
        return out

    def n_distinct(self, trials: list[dict]) -> int:
        """Trials that are distinct draws: distinct return series, plus trials recorded without returns."""
        return len(self._returns_of(trials)) + sum(1 for t in trials if not t.get("returns_file"))

    def trial_returns(self, hypothesis_id: str | None, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None) -> dict[str, pd.Series]:
        return self._returns_of(self.counted(hypothesis_id, stages, session))

    def returns_matrix(self, hypothesis_id: str | None, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None, trials: list[dict] | None = None) -> pd.DataFrame:
        """Trial returns aligned on their common dates (what PBO needs)."""
        cols = self._returns_of(trials) if trials is not None else self.trial_returns(hypothesis_id, stages, session)
        return pd.DataFrame(cols).dropna() if cols else pd.DataFrame()

    def sharpe_variance(self, hypothesis_id: str | None, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None, trials: list[dict] | None = None) -> float | None:
        """Cross-trial variance of per-period Sharpe ratios, each on its own dates (what the deflated
        Sharpe ratio needs; walk-forward validation windows do not overlap)."""
        series = self._returns_of(trials) if trials is not None else self.trial_returns(hypothesis_id, stages, session)
        sr = [float(r.mean() / r.std(ddof=1)) for r in series.values() if len(r) > 2 and r.std(ddof=1) > 0]
        if len(sr) < 2:
            return None
        return float(np.var(sr, ddof=1))

    def deflated_sharpe(self, hypothesis_id: str | None, returns: pd.Series, periods_per_year: float = 252.0, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None, trials: list[dict] | None = None) -> dict:
        """The deflated Sharpe ratio of ``returns`` given the trials (``trials``: an explicit set,
        e.g. ``scope(...)``; else the hypothesis's counted trials)."""
        r = pd.Series(returns).dropna()
        group = trials if trials is not None else self.counted(hypothesis_id, stages, session)
        n = max(self.n_distinct(group), 1)
        var = self.sharpe_variance(None, trials=group)
        sr = float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year)) if r.std(ddof=1) else float("nan")
        out = deflated_sharpe_ratio(sr, n, len(r), float(r.skew()), float(r.kurt() + 3.0), var, round(periods_per_year))
        out["ledger_trials"] = n
        return out

    def pbo(self, hypothesis_id: str | None, n_splits: int = 8, stages: tuple[str, ...] = COUNTED_STAGES, session: str | None = None, trials: list[dict] | None = None) -> dict:
        m = self.returns_matrix(hypothesis_id, stages, session, trials)
        if m.shape[1] < 2 or len(m) < n_splits * 4:
            return {"pbo": float("nan"), "n_trials": m.shape[1], "note": "needs at least two trials on common dates"}
        out = probability_of_backtest_overfitting(m.to_numpy(), n_splits=n_splits)
        out["n_trials"] = m.shape[1]
        return out

    def holdout_viewed(self, hypothesis_id: str) -> bool:
        self.registry.reload()
        return hypothesis_id in self.registry.holdout_viewed


__all__ = ["COUNTED_STAGES", "STAGES", "LedgerError", "Trial", "TrialLedger"]
