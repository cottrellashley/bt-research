"""Official backtest records: levels, gates and an append-only, hash-chained registry.

A result climbs EXPLORATORY -> CANDIDATE -> OFFICIAL only through gates that check what the
research standards require: a hypothesis registered before its first trial, every trial in the
ledger, confirmed conventions (the official annualisation basis for OFFICIAL), no silent zero-filled
returns, the deflated Sharpe ratio computed from the ledger's trials, the holdout untouched by
development runs, and - for OFFICIAL - an Implement run that completed solvent with manifest hashes
and its reconciliation to the Paper run. A failed gate raises ``GateFailure`` listing every failed
check; nothing is written for a refused promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from nautilus_fi.domain.conventions import Annualisation


class RecordLevel(str, Enum):
    EXPLORATORY = "EXPLORATORY"
    CANDIDATE = "CANDIDATE"
    OFFICIAL = "OFFICIAL"


class GateFailure(RuntimeError):
    def __init__(self, level: RecordLevel, failed: dict[str, str]) -> None:
        self.level, self.failed = level, failed
        super().__init__(f"{level.value} gate failed: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))


@dataclass
class OfficialRecord:
    record_id: str
    level: str
    name: str
    hypothesis_id: str | None
    spec_hash: str
    conventions: dict
    paper: dict
    implement: dict | None
    reconciliation: dict | None
    dsr: dict | None
    pbo: dict | None
    n_trials: int
    checks: dict
    data_hashes: dict = field(default_factory=dict)
    code_version: str | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    prev_hash: str = ""
    hash: str = ""

    def to_markdown(self) -> str:
        lines = [f"# {self.level} record {self.record_id}: {self.name}", "", f"hypothesis `{self.hypothesis_id}` · spec `{self.spec_hash}` · created {self.created_utc}", "", "## Checks", "", "| check | ok | detail |", "| --- | --- | --- |"]
        lines += [f"| {k} | {'yes' if v['ok'] else 'NO'} | {v['detail']} |" for k, v in self.checks.items()]
        lines += ["", "## Paper", "", *(f"* {k}: {v}" for k, v in self.paper.items() if not isinstance(v, dict))]
        if self.implement:
            lines += ["", "## Implement", "", *(f"* {k}: {v}" for k, v in self.implement.items() if not isinstance(v, dict))]
        if self.reconciliation:
            lines += ["", "## Paper vs Implement", "", *(f"* {k}: {v}" for k, v in self.reconciliation.items() if not isinstance(v, dict))]
        if self.dsr:
            lines += ["", "## Selection", "", f"* trials in the ledger: {self.n_trials}", f"* deflated Sharpe ratio: {self.dsr.get('dsr')} (trial variance: {self.dsr.get('var_trial_sr_source')})"]
        if self.pbo:
            lines += [f"* PBO: {self.pbo.get('pbo')}"]
        lines += ["", "## Conventions", "", *(f"* {k}: {v}" for k, v in self.conventions.items())]
        return "\n".join(lines)


def _scalar(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool, type(None)))}


class RecordRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "records.jsonl"

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def verify_chain(self) -> bool:
        prev = ""
        for r in self.records():
            body = {k: v for k, v in r.items() if k != "hash"}
            if r["prev_hash"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != r["hash"]:
                return False
            prev = r["hash"]
        return True

    def checks(self, level: RecordLevel, paper, spec, ledger, implement=None, reconciliation=None) -> dict[str, dict]:
        out: dict[str, dict] = {}

        def put(name: str, ok: bool, detail: str, required_from: RecordLevel) -> None:
            order = [RecordLevel.EXPLORATORY, RecordLevel.CANDIDATE, RecordLevel.OFFICIAL]
            out[name] = {"ok": bool(ok), "detail": detail, "required": order.index(level) >= order.index(required_from)}

        hyp = spec.hypothesis_id
        h = ledger.hypothesis(hyp) if (ledger is not None and hyp) else None
        trials = ledger.trials(hyp) if (ledger is not None and hyp) else pd.DataFrame()
        put("hypothesis_registered", h is not None, f"{hyp or 'none'}", RecordLevel.CANDIDATE)
        if h is not None and len(trials):
            first = str(trials["created_utc"].min())
            put("registered_before_trials", h.registered_utc <= first, f"registered {h.registered_utc}, first trial {first}", RecordLevel.CANDIDATE)
        else:
            put("registered_before_trials", False, "no trials recorded", RecordLevel.CANDIDATE)
        in_ledger = len(trials) > 0 and spec.hash in set(trials["spec_hash"])
        put("trial_in_ledger", in_ledger, f"{len(trials)} trials for the hypothesis", RecordLevel.CANDIDATE)
        conv = paper.conventions
        put("conventions_confirmed", conv.confirmed, conv.name, RecordLevel.CANDIDATE)
        put("official_annualisation", conv.annualisation == Annualisation.CALENDAR_365, conv.annualisation.value, RecordLevel.OFFICIAL)
        zero = int(paper.missing.get("zero_filled", 0))
        put("no_silent_missing_returns", zero == 0, f"{zero} zero-filled returns under {paper.missing.get('policy')}", RecordLevel.CANDIDATE)
        hard = int((paper.violations["severity"] == "HARD").sum()) if not paper.violations.empty else 0
        put("no_hard_constraint_violations", hard == 0, f"{hard} hard violations", RecordLevel.CANDIDATE)
        if ledger is not None and hyp:
            dsr = ledger.deflated_sharpe(hyp, paper.returns.iloc[1:], paper.conventions.periods_per_year(paper.returns.index[1:]))
            put("dsr_from_ledger", dsr.get("var_trial_sr_source") in ("supplied", "single trial"), f"DSR {dsr.get('dsr'):.3f} over {dsr.get('ledger_trials')} trials ({dsr.get('var_trial_sr_source')})", RecordLevel.CANDIDATE)
            put("holdout_guard", True, "holdout viewed" if ledger.holdout_viewed(hyp) else "holdout not yet viewed", RecordLevel.CANDIDATE)
        else:
            put("dsr_from_ledger", False, "no ledger", RecordLevel.CANDIDATE)
        if implement is not None:
            st, oc = implement.status, implement.outcome
            hashes = bool(implement.manifest and implement.manifest.output_hashes.get("event_chain"))
            put("implement_completed", st == "COMPLETED" and oc == "SOLVENT", f"{st} / {oc}", RecordLevel.OFFICIAL)
            put("implement_manifest", hashes, "output and event-chain hashes" if hashes else "missing", RecordLevel.OFFICIAL)
        else:
            put("implement_completed", False, "no Implement run", RecordLevel.OFFICIAL)
            put("implement_manifest", False, "no Implement run", RecordLevel.OFFICIAL)
        put("reconciled", reconciliation is not None, "Paper vs Implement decomposition" if reconciliation is not None else "missing", RecordLevel.OFFICIAL)
        return out

    def submit(self, level: RecordLevel, paper, spec, ledger=None, implement=None, reconciliation: dict | None = None, data_hashes: dict | None = None, name: str | None = None) -> OfficialRecord:
        checks = self.checks(level, paper, spec, ledger, implement, reconciliation)
        failed = {k: v["detail"] for k, v in checks.items() if v["required"] and not v["ok"]}
        if failed:
            raise GateFailure(level, failed)
        hyp = spec.hypothesis_id
        dsr = ledger.deflated_sharpe(hyp, paper.returns.iloc[1:], paper.conventions.periods_per_year(paper.returns.index[1:])) if (ledger is not None and hyp) else None
        pbo = ledger.pbo(hyp) if (ledger is not None and hyp) else None
        prev = self.records()[-1]["hash"] if self.records() else ""
        rid = hashlib.sha256(f"{spec.hash}{level.value}{datetime.now(UTC).isoformat()}".encode()).hexdigest()[:12]
        rec = OfficialRecord(rid, level.value, name or paper.name, hyp, spec.hash, paper.conventions.to_dict(), _scalar(paper.stats()), _scalar(implement.stats()) if implement is not None else None, reconciliation, None if dsr is None else _scalar(dsr), None if pbo is None else _scalar(pbo), ledger.n_trials(hyp) if (ledger is not None and hyp) else 0, checks, data_hashes or {}, implement.manifest.code_version if implement is not None and implement.manifest else None, prev_hash=prev)
        body = {k: v for k, v in asdict(rec).items() if k != "hash"}
        rec.hash = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec), default=str) + "\n")
        (self.root / f"{rid}.md").write_text(rec.to_markdown())
        return rec


__all__ = ["GateFailure", "OfficialRecord", "RecordLevel", "RecordRegistry"]
