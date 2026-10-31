"""Hypothesis registration and experiment lineage: every material variant is preserved, including
failures, so a final strategy is seen as one member of the search. The holdout is guarded: once
viewed, later decisions cannot be relabelled out-of-sample."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class HoldoutViolation(RuntimeError):
    pass


@dataclass
class Hypothesis:
    hypothesis_id: str
    rationale: str
    universe: str
    decision_clock: str
    signal: str
    benchmark: str
    capital: float
    constraints: str
    holding_period: str
    costs: str
    failure_criteria: str
    development_period: tuple[str, str]
    validation_period: tuple[str, str]
    holdout_period: tuple[str, str]
    registered_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ExperimentRecord:
    hypothesis_id: str
    config_hash: str
    variant: dict
    period: tuple[str, str]
    summary: dict
    status: str                               # COMPLETED | FAILED | ...
    uses_holdout: bool = False
    note: str = ""
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ExperimentRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.hypotheses: dict[str, Hypothesis] = {}
        self.records: list[ExperimentRecord] = []
        self.holdout_viewed: set[str] = set()
        if self.path and self.path.exists():
            self._load()

    def register(self, h: Hypothesis) -> Hypothesis:
        self.hypotheses[h.hypothesis_id] = h
        self._save()
        return h

    def record(self, rec: ExperimentRecord) -> ExperimentRecord:
        h = self.hypotheses.get(rec.hypothesis_id)
        if h is not None:
            start, end = rec.period
            if rec.uses_holdout or (end > h.holdout_period[0] and start < h.holdout_period[1]):
                rec.uses_holdout = True
                if rec.hypothesis_id in self.holdout_viewed and rec.note != "final":
                    raise HoldoutViolation(f"holdout for {rec.hypothesis_id} already viewed: further runs are development evidence")
                self.holdout_viewed.add(rec.hypothesis_id)
        self.records.append(rec)
        self._save()
        return rec

    def n_trials(self, hypothesis_id: str) -> int:
        return sum(1 for r in self.records if r.hypothesis_id == hypothesis_id and not r.uses_holdout)

    def variants_frame(self, hypothesis_id: str | None = None) -> pd.DataFrame:
        rows = [{"hypothesis": r.hypothesis_id, "config_hash": r.config_hash, "period": f"{r.period[0]}..{r.period[1]}", "status": r.status, "uses_holdout": r.uses_holdout, **{f"param.{k}": v for k, v in r.variant.items()}, **{f"result.{k}": v for k, v in r.summary.items()}} for r in self.records if hypothesis_id is None or r.hypothesis_id == hypothesis_id]
        return pd.DataFrame(rows)

    def reload(self) -> None:
        """Re-read the file (another process may have registered or recorded since)."""
        if self.path and self.path.exists():
            self.hypotheses, self.records, self.holdout_viewed = {}, [], set()
            self._load()

    def _save(self) -> None:
        """Merge with what is on disk, then replace the file atomically, under a lock: concurrent
        writers never lose each other's hypotheses, records or holdout flags."""
        if not self.path:
            return
        from nautilus_fi.core.fileio import atomic_write_text, file_lock

        with file_lock(self.path.with_name(self.path.name + ".lock")):
            if self.path.exists():
                disk = ExperimentRegistry(None)
                disk.path = self.path
                disk._load()
                for k, v in disk.hypotheses.items():
                    self.hypotheses.setdefault(k, v)
                seen = {_record_key(r) for r in self.records}
                self.records = [r for r in disk.records if _record_key(r) not in seen] + self.records
                self.records.sort(key=lambda r: r.created_utc)
                self.holdout_viewed |= disk.holdout_viewed
            payload = {"hypotheses": {k: asdict(v) for k, v in self.hypotheses.items()}, "records": [asdict(r) for r in self.records], "holdout_viewed": sorted(self.holdout_viewed)}
            atomic_write_text(self.path, json.dumps(payload, indent=2, default=str))

    def _load(self) -> None:
        payload = json.loads(self.path.read_text())
        for k, v in payload.get("hypotheses", {}).items():
            v["development_period"], v["validation_period"], v["holdout_period"] = tuple(v["development_period"]), tuple(v["validation_period"]), tuple(v["holdout_period"])
            self.hypotheses[k] = Hypothesis(**v)
        for r in payload.get("records", []):
            r["period"] = tuple(r["period"])
            self.records.append(ExperimentRecord(**r))
        self.holdout_viewed = set(payload.get("holdout_viewed", []))


def _record_key(r: ExperimentRecord) -> tuple:
    return (r.hypothesis_id, r.config_hash, r.created_utc, str(r.period), r.note)


__all__ = ["ExperimentRecord", "ExperimentRegistry", "HoldoutViolation", "Hypothesis"]
