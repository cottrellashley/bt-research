"""Append-only event log (blueprint section 4): what happened and why.

The event log records lifecycle, trading, settlement, funding and accounting events with
deterministic identifiers, causal links and a hash chain, so a pinned replay can be verified and a
reviewer can follow one position from eligibility to reported return. The ledger records the
financial effect separately; neither substitutes for the other.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from nautilus_fi.core.time import TimePoint, to_timestamp


class EventType(str, Enum):
    # issuance / primary
    ISSUANCE = "ISSUANCE"
    AUCTION_ALLOCATION = "AUCTION_ALLOCATION"
    # orders and trades
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_APPROVED = "ORDER_APPROVED"
    ORDER_ROUTED = "ORDER_ROUTED"
    QUOTE_REQUESTED = "QUOTE_REQUESTED"
    QUOTE_RECEIVED = "QUOTE_RECEIVED"
    QUOTE_DECLINED = "QUOTE_DECLINED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELED = "ORDER_CANCELED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    ORDER_REJECTED = "ORDER_REJECTED"
    TRADE = "TRADE"
    SETTLEMENT = "SETTLEMENT"
    SETTLEMENT_FAIL = "SETTLEMENT_FAIL"
    # lifecycle
    FIXING = "FIXING"
    RATE_RESET = "RATE_RESET"
    ACCRUAL = "ACCRUAL"
    COUPON_ENTITLEMENT = "COUPON_ENTITLEMENT"
    COUPON_PAYMENT = "COUPON_PAYMENT"
    COUPON_DEFERRAL = "COUPON_DEFERRAL"
    PIK_CAPITALISATION = "PIK_CAPITALISATION"
    PRINCIPAL_REDUCTION = "PRINCIPAL_REDUCTION"
    PRINCIPAL_INCREASE = "PRINCIPAL_INCREASE"
    FACTOR_UPDATE = "FACTOR_UPDATE"
    INDEX_RATIO_UPDATE = "INDEX_RATIO_UPDATE"
    DRAWDOWN = "DRAWDOWN"
    REPAYMENT = "REPAYMENT"
    EXERCISE_NOTICE = "EXERCISE_NOTICE"
    EXERCISE = "EXERCISE"
    PARTIAL_CALL = "PARTIAL_CALL"
    TENDER_OFFER = "TENDER_OFFER"
    TENDER_ELECTION = "TENDER_ELECTION"
    TENDER_SETTLEMENT = "TENDER_SETTLEMENT"
    PUT_NOTICE = "PUT_NOTICE"
    AMENDMENT = "AMENDMENT"
    MISSED_PAYMENT = "MISSED_PAYMENT"
    GRACE_PERIOD_END = "GRACE_PERIOD_END"
    DEFAULT = "DEFAULT"
    ACCELERATION = "ACCELERATION"
    CLAIM_REGISTERED = "CLAIM_REGISTERED"
    RECOVERY_DISTRIBUTION = "RECOVERY_DISTRIBUTION"
    RESTRUCTURING = "RESTRUCTURING"
    EXCHANGE = "EXCHANGE"
    WRITE_DOWN = "WRITE_DOWN"
    CONVERSION = "CONVERSION"
    MATURITY = "MATURITY"
    REDEMPTION = "REDEMPTION"
    TRIGGER_TEST = "TRIGGER_TEST"
    CREDIT_EVENT = "CREDIT_EVENT"
    DELIVERY = "DELIVERY"
    # funding / collateral
    MARGIN_CALL = "MARGIN_CALL"
    COLLATERAL_TRANSFER = "COLLATERAL_TRANSFER"
    VARIATION_MARGIN = "VARIATION_MARGIN"
    FUNDING = "FUNDING"
    REPO_OPEN = "REPO_OPEN"
    REPO_ROLL = "REPO_ROLL"
    REPO_CLOSE = "REPO_CLOSE"
    BORROW_OPEN = "BORROW_OPEN"
    BORROW_RECALL = "BORROW_RECALL"
    BORROW_CLOSE = "BORROW_CLOSE"
    MANUFACTURED_PAYMENT = "MANUFACTURED_PAYMENT"
    FORCED_ACTION = "FORCED_ACTION"
    FUNDING_BREACH = "FUNDING_BREACH"
    # marks / data / control
    MARK = "MARK"
    FX_FIXING = "FX_FIXING"
    CORRECTION = "CORRECTION"
    CAPITAL_FLOW = "CAPITAL_FLOW"
    FEE = "FEE"
    TAX = "TAX"
    DECISION = "DECISION"
    CONSTRAINT_RELAXED = "CONSTRAINT_RELAXED"
    RUN_STATUS = "RUN_STATUS"


class EventSource(str, Enum):
    ENVIRONMENT = "ENVIRONMENT"      # realized historical events (observed)
    STRATEGY = "STRATEGY"            # decisions (modelled)
    EXECUTION = "EXECUTION"          # fills, rejects (modelled)
    LIFECYCLE = "LIFECYCLE"          # contractual consequences (observed or modelled)
    FUNDING = "FUNDING"
    ACCOUNTING = "ACCOUNTING"
    SYSTEM = "SYSTEM"


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


@dataclass(frozen=True)
class Event:
    type: EventType
    time: TimePoint
    source: EventSource
    instrument: str | None = None
    position_id: str | None = None
    order_id: str | None = None
    payload: dict = field(default_factory=dict)
    caused_by: tuple[str, ...] = ()
    observed: bool = True              # True = realized history; False = modelled (our decisions / fills)
    seq: int = 0
    corrects: str | None = None        # id of the event this one corrects

    @property
    def id(self) -> str:
        key = f"{self.type.value}|{self.time.economic_ns}|{self.time.available_ns}|{self.source.value}|{self.instrument}|{self.position_id}|{self.order_id}|{_canonical(self.payload)}|{self.seq}|{self.corrects}"
        return hashlib.sha256(key.encode()).hexdigest()[:20]

    def to_record(self) -> dict:
        return {
            "id": self.id, "type": self.type.value, "source": self.source.value, "observed": self.observed,
            "economic_ts": to_timestamp(self.time.economic_ns), "available_ts": to_timestamp(self.time.available_ns),
            "publication_ts": to_timestamp(self.time.publication_ns) if self.time.publication_ns is not None else None,
            "instrument": self.instrument, "position_id": self.position_id, "order_id": self.order_id,
            "caused_by": list(self.caused_by), "corrects": self.corrects, "seq": self.seq, "payload": _canonical(self.payload),
        }


class EventLog:
    """Append-only, hash-chained event log with deterministic sequence numbers."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._by_id: dict[str, Event] = {}
        self._chain = hashlib.sha256()
        self._seq = 0

    def append(self, event: Event) -> Event:
        if event.seq == 0 and self._seq > 0 or event.seq == 0:
            event = Event(event.type, event.time, event.source, event.instrument, event.position_id, event.order_id, event.payload, event.caused_by, event.observed, self._seq + 1, event.corrects)
        if event.id in self._by_id:
            raise ValueError(f"Duplicate event id {event.id}: duplicate delivery must not be appended")
        for dep in event.caused_by:
            if dep not in self._by_id:
                raise ValueError(f"Event {event.id} cites unknown cause {dep}")
        self._events.append(event)
        self._by_id[event.id] = event
        self._chain.update(event.id.encode())
        self._seq = event.seq
        return event

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def get(self, event_id: str) -> Event | None:
        return self._by_id.get(event_id)

    @property
    def chain_hash(self) -> str:
        return self._chain.hexdigest()

    def events_for(self, instrument: str | None = None, types: Iterable[EventType] | None = None, position_id: str | None = None) -> list[Event]:
        types_set = set(types) if types else None
        return [e for e in self._events if (instrument is None or e.instrument == instrument) and (types_set is None or e.type in types_set) and (position_id is None or e.position_id == position_id)]

    def between(self, start_ns: int, end_ns: int) -> list[Event]:
        return [e for e in self._events if start_ns <= e.time.economic_ns <= end_ns]

    def lineage(self, event_id: str) -> list[Event]:
        """All ancestors of an event (why it happened), oldest first."""
        out: list[Event] = []
        seen: set[str] = set()
        stack = [event_id]
        while stack:
            eid = stack.pop()
            ev = self._by_id.get(eid)
            if ev is None or eid in seen:
                continue
            seen.add(eid)
            out.append(ev)
            stack.extend(ev.caused_by)
        return sorted(out, key=lambda e: e.seq)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_record() for e in self._events])

    def to_parquet(self, path: str) -> None:
        self.to_frame().to_parquet(path, index=False)


__all__ = ["Event", "EventLog", "EventSource", "EventType"]
