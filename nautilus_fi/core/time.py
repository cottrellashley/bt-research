"""Time model (blueprint sections 1 and 9): economic time, publication, availability, ingestion.

``TimePoint`` is attached to observations and events. Derived features inherit the latest required
availability among their inputs plus the modelled computation and operational delays. A daily
timestamp must never masquerade as a known intraday time: callers pass explicit times of day.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import pandas as pd

NS = 1_000_000_000
NS_PER_DAY = 86_400 * NS


def to_ns(value) -> int:
    """UTC nanoseconds from a Timestamp / datetime / date / ISO string / int."""
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return int(pd.Timestamp(datetime.combine(value, time(0, 0), tzinfo=UTC)).value)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value)


def to_timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), tz="UTC")


def date_of(ns: int) -> date:
    return to_timestamp(ns).date()


def at(d: date, t: time) -> int:
    return int(pd.Timestamp(datetime.combine(d, t, tzinfo=UTC)).value)


@dataclass(frozen=True)
class TimePoint:
    economic_ns: int
    available_ns: int
    publication_ns: int | None = None
    ingestion_ns: int | None = None

    def __post_init__(self) -> None:
        if self.available_ns < self.economic_ns:
            raise ValueError("available_ns cannot precede economic_ns")
        if self.publication_ns is not None and self.publication_ns < self.economic_ns:
            raise ValueError("publication_ns cannot precede economic_ns")
        if self.publication_ns is not None and self.available_ns < self.publication_ns:
            raise ValueError("available_ns cannot precede publication_ns")

    @staticmethod
    def known_at(economic_ns: int, available_ns: int | None = None) -> TimePoint:
        return TimePoint(economic_ns, available_ns if available_ns is not None else economic_ns)

    @staticmethod
    def derived(inputs: Iterable[TimePoint], computation_delay_ns: int = 0, operational_delay_ns: int = 0, economic_ns: int | None = None) -> TimePoint:
        """Availability of a derived feature = latest input availability + delays."""
        inputs = list(inputs)
        if not inputs:
            raise ValueError("derived() needs at least one input")
        avail = max(i.available_ns for i in inputs) + computation_delay_ns + operational_delay_ns
        econ = economic_ns if economic_ns is not None else max(i.economic_ns for i in inputs)
        return TimePoint(min(econ, avail), avail)

    def visible_at(self, now_ns: int) -> bool:
        return self.available_ns <= now_ns


@dataclass(frozen=True)
class DataVintage:
    """Which version of the database the run may see (ingestion cutoff); None = latest."""

    ingestion_cutoff_ns: int | None = None
    label: str = "latest"

    def admits(self, ingestion_ns: int | None) -> bool:
        if self.ingestion_cutoff_ns is None or ingestion_ns is None:
            return True
        return ingestion_ns <= self.ingestion_cutoff_ns


__all__ = ["NS", "NS_PER_DAY", "DataVintage", "TimePoint", "at", "date_of", "to_ns", "to_timestamp"]
