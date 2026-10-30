"""Linked run outputs: manifest, performance, implementation, holdings and liquidity, risk and
attribution, evidence quality, robustness; drill-down explanations; machine- and human-readable
exports. Unknown and unsupported are reported explicitly, never as zero."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nautilus_fi.core.events import EventLog, EventType
from nautilus_fi.core.ledger import Ledger
from nautilus_fi.core.manifest import ExperimentManifest


@dataclass
class ReportBundle:
    manifest: ExperimentManifest
    performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    implementation: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)
    liquidity: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk: pd.DataFrame = field(default_factory=pd.DataFrame)
    attribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness: pd.DataFrame = field(default_factory=pd.DataFrame)
    support: pd.DataFrame = field(default_factory=pd.DataFrame)
    invariants: pd.DataFrame = field(default_factory=pd.DataFrame)
    headline: dict = field(default_factory=dict)
    unknowns: list[str] = field(default_factory=list)

    def export(self, folder: str | Path) -> Path:
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(self.manifest.to_json())
        for name in ("performance", "implementation", "holdings", "liquidity", "risk", "attribution", "evidence", "robustness", "support", "invariants"):
            df = getattr(self, name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_parquet(out / f"{name}.parquet", index=True)
        (out / "report.md").write_text(self.to_markdown())
        return out

    def to_markdown(self) -> str:
        m = self.manifest
        lines = [f"# Run {m.run_id[:8]}: {m.hypothesis or 'unnamed hypothesis'}", "", f"**Run mode** {m.run_mode.value} · **Deployability** {m.deployability.value} · **Status** {m.computational_status.value} / {m.economic_outcome.value}", f"**Universe** {m.universe} · **Dates** {m.start} → {m.end} · **Capital** {m.capital:,.0f} {m.base_currency} · **Benchmark** {m.benchmark.name if m.benchmark else 'none'}", ""]
        if self.support is not None and not self.support.empty:
            worst = self.support.sort_values("status").iloc[-1] if "status" in self.support else None
            lines += ["## Support status", "", self.support.to_markdown(index=False), ""]
            _ = worst
        if self.headline:
            lines += ["## Headline", ""] + [f"- **{k}**: {v}" for k, v in self.headline.items()] + [""]
        if self.unknowns:
            lines += ["## Unknown / unsupported", ""] + [f"- {u}" for u in self.unknowns] + [""]
        lines += ["## Most consequential assumptions", ""]
        used = [a for a in m.assumptions][:15]
        lines += [f"- {a.get('name')} = {a.get('value')} {a.get('unit', '')} ({a.get('level')}, {a.get('source', '')})" for a in used] or ["- none recorded"]
        lines.append("")
        for title, df in (("Performance", self.performance), ("Implementation", self.implementation), ("Evidence quality", self.evidence), ("Risk", self.risk), ("Attribution", self.attribution), ("Robustness", self.robustness), ("Invariants", self.invariants)):
            if isinstance(df, pd.DataFrame) and not df.empty:
                lines += [f"## {title}", "", df.head(40).to_markdown(), ""]
        lines += ["## Data versions", ""] + [f"- {d.name}: {d.rows} rows, hash {d.content_hash}, knowledge time {d.knowledge_time}, policy {d.availability_policy}" for d in m.data_versions]
        lines += ["", "## Reproducibility", "", f"- config hash {m.config_hash}", f"- code {m.code_version}", f"- environment {json.dumps(m.environment)}", f"- seeds {m.seeds}", f"- output hashes {m.output_hashes}"]
        return "\n".join(lines)


def explain_order(log: EventLog, order_id: str) -> pd.DataFrame:
    """Why an order did or did not fill: its full event chain."""
    rows = [e.to_record() for e in log if e.order_id == order_id]
    return pd.DataFrame(rows)


def explain_payment(log: EventLog, ledger: Ledger, event_id: str) -> dict:
    """Why a payment belongs to the portfolio: lineage events and the ledger postings it produced."""
    lineage = log.lineage(event_id)
    entries = [e for e in ledger.entries if e.event_id == event_id]
    return {"lineage": pd.DataFrame([e.to_record() for e in lineage]), "postings": pd.DataFrame([{"entry": e.entry_id, "account": p.account.value, "key": p.key, "debit": p.debit, "credit": p.credit, "memo": p.memo or e.memo} for e in entries for p in e.postings])}


def explain_eligibility(reasons: dict[str, list[str]], symbol: str) -> list[str]:
    return reasons.get(symbol, ["no eligibility record"])


__all__ = ["EventType", "ReportBundle", "explain_eligibility", "explain_order", "explain_payment"]
