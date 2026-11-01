"""Hypothesis registration and experiment lineage: every material variant is preserved, including
failures, so a final strategy is seen as one member of the search. The holdout is guarded: once
viewed, later decisions cannot be relabelled out-of-sample."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
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
    content_hash: str = ""                    # set at registration: the hypothesis is frozen by it


def hypothesis_hash(h: Hypothesis) -> str:
    """The content of a hypothesis (everything but its registration stamp and hash)."""
    body = {k: v for k, v in asdict(h).items() if k not in ("registered_utc", "content_hash")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


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
        """Store ``h`` as given (in-memory registries, tests). ``register_new`` is the guarded path."""
        self.hypotheses[h.hypothesis_id] = h
        self._save()
        return h

    def register_new(self, h: Hypothesis) -> Hypothesis:
        """Register a hypothesis once: the registry stamps the registration time (a supplied one is
        ignored, so it cannot be backdated) and freezes the content by hash; the existence check and
        the write are one locked step, so concurrent registrations of an id cannot overwrite each other."""
        h = replace(h, registered_utc=datetime.now(UTC).isoformat(), content_hash="")
        h = replace(h, content_hash=hypothesis_hash(h))
        if not self.path:
            if h.hypothesis_id in self.hypotheses:
                raise ValueError(f"hypothesis {h.hypothesis_id} already registered; register a new id for a changed hypothesis")
            self.hypotheses[h.hypothesis_id] = h
            return h
        from nautilus_fi.core.fileio import file_lock

        with file_lock(self.path.with_name(self.path.name + ".lock")):
            self.reload()
            if h.hypothesis_id in self.hypotheses:
                raise ValueError(f"hypothesis {h.hypothesis_id} already registered; register a new id for a changed hypothesis")
            self.hypotheses[h.hypothesis_id] = h
            self._write()
        return h

    def frozen(self, hypothesis_id: str) -> bool:
        """The hypothesis still has the content it was registered with."""
        h = self.hypotheses.get(hypothesis_id)
        return h is not None and bool(h.content_hash) and h.content_hash == hypothesis_hash(h)

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
        from nautilus_fi.core.fileio import file_lock

        with file_lock(self.path.with_name(self.path.name + ".lock")):
            if self.path.exists():
                disk = ExperimentRegistry(None)
                disk.path = self.path
                disk._load()
                for k, v in disk.hypotheses.items():           # a registered hypothesis on disk is never replaced
                    self.hypotheses[k] = v
                seen = {_record_key(r) for r in self.records}
                self.records = [r for r in disk.records if _record_key(r) not in seen] + self.records
                self.records.sort(key=lambda r: r.created_utc)
                self.holdout_viewed |= disk.holdout_viewed
            self._write()

    def _write(self) -> None:
        from nautilus_fi.core.fileio import atomic_write_text

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


__all__ = ["ExperimentRecord", "ExperimentRegistry", "HoldoutViolation", "Hypothesis", "hypothesis_hash"]
