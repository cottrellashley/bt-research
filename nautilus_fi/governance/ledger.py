"""The trial ledger: every research trial, recorded automatically, so selection statistics count the
whole search and not just the winner.

Each Evaluate, Paper and Implement run can write one append-only line (stage, hypothesis, spec hash,
variant parameters, period, headline statistics, conventions, holdout use) plus its return series.
From those the ledger gives the number of trials and the cross-trial variance of Sharpe ratios that
the deflated Sharpe ratio needs, and the aligned variant return matrix that PBO (CSCV) needs. The
holdout guard of ``research.ExperimentRegistry`` applies: once a hypothesis's holdout has been seen,
further runs over it are refused unless marked final.
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

from nautilus_fi.research.experiment import ExperimentRecord, ExperimentRegistry, Hypothesis
from nautilus_fi.research.statistics import deflated_sharpe_ratio, probability_of_backtest_overfitting

STAGES = ("EVALUATE", "PAPER", "IMPLEMENT", "OOS", "VALIDATION")


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
        self.registry = ExperimentRegistry(self.root / "hypotheses.json")
        self.path = self.root / "trials.jsonl"

    # ------------------------------------------------------------------ hypotheses
    def register(self, h: Hypothesis) -> Hypothesis:
        """Pre-register a hypothesis (its timestamp must precede every trial that cites it)."""
        if h.hypothesis_id in self.registry.hypotheses:
            raise ValueError(f"hypothesis {h.hypothesis_id} already registered; register a new id for a changed hypothesis")
        return self.registry.register(h)

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return self.registry.hypotheses.get(hypothesis_id)

    # ------------------------------------------------------------------ recording
    def record(self, stage: str, name: str, spec_hash: str, variant: dict, period: tuple[str, str], summary: dict, returns: pd.Series | None = None, hypothesis_id: str | None = None, conventions: dict | None = None, uses_holdout: bool = False, status: str = "COMPLETED", note: str = "") -> Trial:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; use one of {STAGES}")
        if hypothesis_id is not None:
            if hypothesis_id not in self.registry.hypotheses:
                raise KeyError(f"hypothesis {hypothesis_id!r} is not registered: pre-register it before running trials")
            # holdout guard (raises HoldoutViolation)
            self.registry.record(ExperimentRecord(hypothesis_id, spec_hash, {k: _clean(v) for k, v in variant.items()}, period, {}, status, uses_holdout, note))
            uses_holdout = self.registry.records[-1].uses_holdout
        payload = json.dumps([stage, name, spec_hash, variant, period, hypothesis_id, len(self.trials_list())], sort_keys=True, default=str)
        tid = hashlib.sha256(payload.encode()).hexdigest()[:16]
        rfile = None
        if returns is not None and len(returns):
            rfile = f"returns/{tid}.parquet"
            pd.DataFrame({"date": pd.to_datetime(returns.index), "ret": returns.to_numpy(dtype=float)}).to_parquet(self.root / rfile, index=False)
        trial = Trial(tid, hypothesis_id, stage, name, spec_hash, {k: _clean(v) for k, v in variant.items()}, (str(period[0]), str(period[1])), {k: _clean(v) for k, v in summary.items() if isinstance(_clean(v), (int, float, str, bool, type(None)))}, conventions or {}, uses_holdout, status, note, returns_file=rfile)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(trial), default=str) + "\n")
        return trial

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
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def trials(self, hypothesis_id: str | None = None, stage: str | None = None) -> pd.DataFrame:
        rows = [t for t in self.trials_list() if (hypothesis_id is None or t["hypothesis_id"] == hypothesis_id) and (stage is None or t["stage"] == stage)]
        if not rows:
            return pd.DataFrame()
        flat = [{**{k: v for k, v in t.items() if k not in ("summary", "variant", "conventions")}, **{f"param.{k}": v for k, v in t["variant"].items()}, **{f"result.{k}": v for k, v in t["summary"].items()}} for t in rows]
        return pd.DataFrame(flat)

    def n_trials(self, hypothesis_id: str, stages: tuple[str, ...] = ("PAPER", "IMPLEMENT", "VALIDATION")) -> int:
        return sum(1 for t in self.trials_list() if t["hypothesis_id"] == hypothesis_id and t["stage"] in stages and not t["uses_holdout"])

    def trial_returns(self, hypothesis_id: str, stages: tuple[str, ...] = ("PAPER", "VALIDATION")) -> dict[str, pd.Series]:
        out = {}
        for t in self.trials_list():
            if t["hypothesis_id"] != hypothesis_id or t["stage"] not in stages or t["uses_holdout"] or not t.get("returns_file"):
                continue
            df = pd.read_parquet(self.root / t["returns_file"])
            out[t["trial_id"]] = pd.Series(df["ret"].to_numpy(), index=pd.to_datetime(df["date"]))
        return out

    def returns_matrix(self, hypothesis_id: str, stages: tuple[str, ...] = ("PAPER", "VALIDATION")) -> pd.DataFrame:
        """Trial returns aligned on their common dates (what PBO needs)."""
        cols = self.trial_returns(hypothesis_id, stages)
        return pd.DataFrame(cols).dropna() if cols else pd.DataFrame()

    def sharpe_variance(self, hypothesis_id: str, stages: tuple[str, ...] = ("PAPER", "VALIDATION")) -> float | None:
        """Cross-trial variance of per-period Sharpe ratios, each on its own dates (what the deflated
        Sharpe ratio needs; walk-forward validation windows do not overlap)."""
        series = self.trial_returns(hypothesis_id, stages)
        sr = [float(r.mean() / r.std(ddof=1)) for r in series.values() if len(r) > 2 and r.std(ddof=1) > 0]
        if len(sr) < 2:
            return None
        return float(np.var(sr, ddof=1))

    def deflated_sharpe(self, hypothesis_id: str, returns: pd.Series, periods_per_year: float = 252.0, stages: tuple[str, ...] = ("PAPER", "VALIDATION")) -> dict:
        r = pd.Series(returns).dropna()
        n = max(self.n_trials(hypothesis_id, stages), 1)
        var = self.sharpe_variance(hypothesis_id, stages)
        sr = float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year)) if r.std(ddof=1) else float("nan")
        out = deflated_sharpe_ratio(sr, n, len(r), float(r.skew()), float(r.kurt() + 3.0), var, round(periods_per_year))
        out["ledger_trials"] = n
        return out

    def pbo(self, hypothesis_id: str, n_splits: int = 8, stages: tuple[str, ...] = ("PAPER", "VALIDATION")) -> dict:
        m = self.returns_matrix(hypothesis_id, stages)
        if m.shape[1] < 2 or len(m) < n_splits * 4:
            return {"pbo": float("nan"), "n_trials": m.shape[1], "note": "needs at least two trials on common dates"}
        out = probability_of_backtest_overfitting(m.to_numpy(), n_splits=n_splits)
        out["n_trials"] = m.shape[1]
        return out

    def holdout_viewed(self, hypothesis_id: str) -> bool:
        return hypothesis_id in self.registry.holdout_viewed


__all__ = ["STAGES", "Trial", "TrialLedger"]
