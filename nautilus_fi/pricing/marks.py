"""Marks, evidence and valuation policy (blueprint section 14).

Four values are kept apart: the signal input, the independent valuation estimate, the portfolio
accounting mark and the execution reference. A declared ``MarkPolicy`` selects marks by freshness,
size, side, venue, outliers and relevance; competing candidates are retained, never overwritten.
Evidence is classified on separate axes rather than compressed into one score, and lineage lets a
strategy that is scored by the model it trades be detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd


class EvidenceType(str, Enum):
    TRANSACTION = "TRANSACTION"
    FIRM_QUOTE = "FIRM_QUOTE"
    INDICATIVE_QUOTE = "INDICATIVE_QUOTE"
    COMPOSITE = "COMPOSITE"
    EVALUATED = "EVALUATED"
    MODEL = "MODEL"
    STALE = "STALE"

    @property
    def rank(self) -> int:
        return list(EvidenceType).index(self)


@dataclass(frozen=True)
class MarkCandidate:
    clean: float
    evidence: EvidenceType
    source: str
    ts_ns: int
    bid: float | None = None
    ask: float | None = None
    side: str = "MID"                        # BID | MID | ASK | TRADE
    size: float | None = None
    quality: float | None = None
    age_days: int = 0
    lineage: tuple[str, ...] = ()            # data sources / models this value depends on
    note: str = ""


@dataclass
class MarkPolicy:
    purpose: str = "RESEARCH_NAV"            # RESEARCH_NAV | EXIT_VALUE | BENCHMARK_REPLICATION | ACCOUNTING
    side: str = "MID"                        # MID | BID | ASK | POSITION_AWARE (bid for longs, ask for shorts)
    max_age_days: dict[str, int] = field(default_factory=lambda: {"TRANSACTION": 1, "FIRM_QUOTE": 0, "INDICATIVE_QUOTE": 1, "COMPOSITE": 1, "EVALUATED": 1, "MODEL": 5, "STALE": 10})
    min_transaction_size: float = 250_000.0
    outlier_pct: float = 0.03                # discard candidates more than 3% from the robust centre
    preference: tuple[str, ...] = ("FIRM_QUOTE", "TRANSACTION", "COMPOSITE", "EVALUATED", "INDICATIVE_QUOTE", "MODEL", "STALE")
    min_quality: float = 0.0
    stale_handling: str = "MODEL_ROLL"       # CARRY_FLAG | MODEL_ROLL | SUSPEND
    distressed_small_trade_pct: float = 0.10 # a small print more than this far from other evidence does not dominate


@dataclass
class MarkSelection:
    chosen: MarkCandidate | None
    retained: list[MarkCandidate]
    excluded: list[tuple[MarkCandidate, str]]
    rationale: str
    dispersion_pct: float | None


class MarkSelector:
    def __init__(self, policy: MarkPolicy | None = None) -> None:
        self.policy = policy or MarkPolicy()

    def select(self, candidates: list[MarkCandidate], position_sign: int = 1) -> MarkSelection:
        p = self.policy
        excluded: list[tuple[MarkCandidate, str]] = []
        pool: list[MarkCandidate] = []
        for c in candidates:
            if c.age_days > p.max_age_days.get(c.evidence.value, 1):
                excluded.append((c, f"stale ({c.age_days}d > {p.max_age_days.get(c.evidence.value)})"))
                continue
            if c.quality is not None and c.quality < p.min_quality:
                excluded.append((c, "quality below minimum"))
                continue
            pool.append(c)
        if not pool:
            return MarkSelection(None, [], excluded, "no admissible evidence", None)
        values = np.array([c.clean for c in pool])
        centre = float(np.median(values))
        dispersion = float((values.max() - values.min()) / centre) if centre else None
        kept: list[MarkCandidate] = []
        for c in pool:
            dev = abs(c.clean - centre) / centre if centre else 0.0
            if len(pool) >= 3 and dev > p.outlier_pct:
                excluded.append((c, f"outlier {dev:.1%} from centre"))
                continue
            if c.evidence == EvidenceType.TRANSACTION and c.size is not None and c.size < p.min_transaction_size and len(pool) > 1 and dev > p.distressed_small_trade_pct:
                excluded.append((c, "small distressed print cannot dominate"))
                continue
            kept.append(c)
        if not kept:
            kept = pool
        order = {e: i for i, e in enumerate(p.preference)}
        kept.sort(key=lambda c: (order.get(c.evidence.value, 99), c.age_days, -(c.quality or 0.0)))
        chosen = kept[0]
        side = p.side
        if side == "POSITION_AWARE":
            side = "BID" if position_sign >= 0 else "ASK"
        value = chosen.clean
        if side == "BID" and chosen.bid is not None:
            value = chosen.bid
        elif side == "ASK" and chosen.ask is not None:
            value = chosen.ask
        chosen_sided = MarkCandidate(value, chosen.evidence, chosen.source, chosen.ts_ns, chosen.bid, chosen.ask, side, chosen.size, chosen.quality, chosen.age_days, chosen.lineage, chosen.note)
        return MarkSelection(chosen_sided, kept, excluded, f"policy {p.purpose}: {chosen.evidence.value} from {chosen.source} ({len(kept)} retained, {len(excluded)} excluded)", dispersion)


@dataclass
class FourValues:
    signal_input: MarkCandidate | None = None
    valuation_estimate: MarkCandidate | None = None
    accounting_mark: MarkCandidate | None = None
    execution_reference: MarkCandidate | None = None

    def shared_lineage(self) -> set[str]:
        sets = [set(v.lineage) for v in (self.signal_input, self.accounting_mark) if v is not None]
        return set.intersection(*sets) if len(sets) == 2 else set()

    def circularity_flags(self) -> list[str]:
        flags = []
        shared = self.shared_lineage()
        if shared:
            flags.append(f"signal and accounting mark share inputs: {sorted(shared)}")
        if self.signal_input is not None and self.accounting_mark is not None and self.signal_input.evidence == EvidenceType.MODEL and self.accounting_mark.evidence == EvidenceType.MODEL:
            flags.append("both signal and mark are model values: apparent convergence may be manufactured")
        return flags


@dataclass
class EvidenceClassification:
    """Separate fields, no composite score (blueprint section 14)."""

    data_completeness: str = "UNKNOWN"       # HIGH | MEDIUM | LOW | NONE | UNKNOWN
    temporal_authenticity: str = "UNKNOWN"  # publication-time stamped vs assumed
    valuation_evidence: str = "UNKNOWN"     # TRANSACTION / QUOTE / EVALUATED / MODEL
    execution_calibration: str = "UNKNOWN"  # calibrated on own orders / transactions / assumed
    product_validation: str = "UNKNOWN"     # VALIDATED / APPROXIMATION / EXPERIMENTAL
    notes: tuple[str, ...] = ()

    @staticmethod
    def from_mark(c: MarkCandidate | None, timestamps_real: bool, exec_evidence: str, product_status: str) -> EvidenceClassification:
        if c is None:
            return EvidenceClassification("NONE", "UNKNOWN", "NONE", exec_evidence, product_status)
        ve = {EvidenceType.TRANSACTION: "TRANSACTION", EvidenceType.FIRM_QUOTE: "QUOTE", EvidenceType.INDICATIVE_QUOTE: "QUOTE", EvidenceType.COMPOSITE: "COMPOSITE", EvidenceType.EVALUATED: "EVALUATED", EvidenceType.MODEL: "MODEL", EvidenceType.STALE: "STALE"}[c.evidence]
        completeness = "HIGH" if c.quality is None or c.quality >= 0.7 else "MEDIUM" if c.quality >= 0.4 else "LOW"
        return EvidenceClassification(completeness, "PUBLISHED_TIME" if timestamps_real else "ASSUMED_TIME", ve, exec_evidence, product_status)


def attribute_by_evidence(rows: pd.DataFrame, value_cols: tuple[str, ...] = ("market_value", "pnl", "turnover", "risk")) -> pd.DataFrame:
    """Share of NAV, P&L, turnover and risk by valuation evidence class (``rows`` has an ``evidence`` column)."""
    if rows.empty or "evidence" not in rows:
        return pd.DataFrame()
    out = {}
    for col in value_cols:
        if col in rows:
            total = rows[col].abs().sum()
            out[f"{col}_share"] = rows.groupby("evidence")[col].apply(lambda s: s.abs().sum() / total if total else 0.0)
    return pd.DataFrame(out).fillna(0.0)


__all__ = ["EvidenceClassification", "EvidenceType", "FourValues", "MarkCandidate", "MarkPolicy", "MarkSelection", "MarkSelector", "attribute_by_evidence"]
_ = date
