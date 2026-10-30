"""Correction events and restatement bridges (blueprint sections 9 and 11).

A late servicing correction creates a restated economic view and a current adjustment; both stay
traceable to the source record and the affected ownership interval. Old postings are never edited:
the ledger reverses the original entries and re-posts, and the bridge records the difference between
the as-known and the as-corrected views.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nautilus_fi.core.events import Event, EventLog, EventSource, EventType
from nautilus_fi.core.ledger import JournalEntry, Ledger
from nautilus_fi.core.time import TimePoint, to_timestamp


@dataclass
class CorrectionBridge:
    original_event_id: str
    correction_event_id: str
    instrument: str | None
    field: str
    as_known: object
    as_corrected: object
    reversed_entries: list[str] = field(default_factory=list)
    reposted_entries: list[str] = field(default_factory=list)
    nav_effect: float = 0.0
    affected_from_ns: int | None = None
    affected_to_ns: int | None = None


class RestatementBridge:
    """Applies corrections: emits a CORRECTION event, reverses the original postings, re-posts."""

    def __init__(self, event_log: EventLog, ledger: Ledger) -> None:
        self.event_log = event_log
        self.ledger = ledger
        self.bridges: list[CorrectionBridge] = []

    def correct(self, original: Event, field_name: str, corrected_value, now_ns: int, entries_to_reverse: list[JournalEntry], repost: list[JournalEntry] | None = None, note: str = "") -> CorrectionBridge:
        corr = Event(EventType.CORRECTION, TimePoint(original.time.economic_ns, now_ns), EventSource.ENVIRONMENT, original.instrument, original.position_id, original.order_id,
                     payload={"field": field_name, "as_known": original.payload.get(field_name), "as_corrected": corrected_value, "note": note}, caused_by=(original.id,), corrects=original.id)
        corr = self.event_log.append(corr)
        nav_before = self.ledger.nav().amount
        reversed_ids = [self.ledger.reverse(e.entry_id, now_ns, corr.id, f"correction of {original.id}").entry_id for e in entries_to_reverse]
        reposted_ids = [self.ledger.post(JournalEntry(corr.id, now_ns, e.postings, e.memo)).entry_id for e in (repost or [])]
        bridge = CorrectionBridge(original.id, corr.id, original.instrument, field_name, original.payload.get(field_name), corrected_value, reversed_ids, reposted_ids, self.ledger.nav().amount - nav_before, original.time.economic_ns, now_ns)
        self.bridges.append(bridge)
        return bridge

    def to_frame(self) -> pd.DataFrame:
        rows = [{"original_event": b.original_event_id, "correction_event": b.correction_event_id, "instrument": b.instrument, "field": b.field, "as_known": b.as_known, "as_corrected": b.as_corrected, "nav_effect": b.nav_effect, "affected_from": to_timestamp(b.affected_from_ns) if b.affected_from_ns else None, "corrected_at": to_timestamp(b.affected_to_ns) if b.affected_to_ns else None} for b in self.bridges]
        return pd.DataFrame(rows)


__all__ = ["CorrectionBridge", "RestatementBridge"]
