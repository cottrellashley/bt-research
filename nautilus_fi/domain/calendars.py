"""Rebalance calendars shared by the panel engine and the event engine.

A calendar answers one question for a trading day: is a rebalance decision due? Both engines supply
the neighbouring trading days, so month ends, week days and index dates are decided from the
holiday calendar (public in advance), never from data. Strings parse to calendars:

* ``"D"`` daily, ``"5"`` every five trading days
* ``"W"`` first trading day of each ISO week (legacy), ``"W-WED"`` the last trading day on or before
  Wednesday in each week
* ``"M"`` / ``"BMS"`` first trading day of the month (legacy ``M``), ``"BME"`` last trading day of the month
* ``"Q"`` / ``"BQS"`` first trading day of the quarter, ``"BQE"`` last trading day of the quarter
* ``IndexCalendar(name, dates)`` / ``Explicit(dates)`` vendor or index rebalance dates aligned to
  trading days
* ``Triggered(threshold, measure)`` evaluated every base day; the engine rebalances only when the
  proposed change (turnover, or signal change) exceeds the threshold
"""

from __future__ import annotations

import dataclasses
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

WEEKDAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


class RebalanceCalendar:
    name: str = "calendar"
    triggered: bool = False

    def is_due(self, d: date, last: date | None, prev_td: date | None, next_td: date | None, elapsed: int | None = None) -> bool:
        """``last``: previous decision date; ``prev_td`` / ``next_td``: neighbouring trading days;
        ``elapsed``: trading days since ``last``."""
        raise NotImplementedError

    def dates(self, trading_days: Sequence[date], first_decision: bool = False) -> list[date]:
        """Decision dates among ``trading_days`` (sorted); ``first_decision`` adds the first day."""
        out: list[date] = []
        last_i: int | None = None
        n = len(trading_days)
        for i, d in enumerate(trading_days):
            prev_td = trading_days[i - 1] if i > 0 else None
            next_td = trading_days[i + 1] if i + 1 < n else None
            last = trading_days[last_i] if last_i is not None else None
            if (first_decision and i == 0) or self.is_due(d, last, prev_td, next_td, None if last_i is None else i - last_i):
                out.append(d)
                last_i = i
        return out

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Daily(RebalanceCalendar):
    name: str = "D"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return True


@dataclass(frozen=True)
class EveryN(RebalanceCalendar):
    n: int = 5
    name: str = "EVERY_N"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        if last is None:
            return True
        if elapsed is None:
            raise ValueError("EveryN needs the trading days elapsed since the last decision")
        return elapsed >= self.n


@dataclass(frozen=True)
class WeekStart(RebalanceCalendar):
    """First trading day of each ISO week (the legacy ``"W"`` rule)."""

    name: str = "W"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        if last is not None:
            return d.isocalendar()[:2] != last.isocalendar()[:2]
        return prev_td is None or d.isocalendar()[:2] != prev_td.isocalendar()[:2]


@dataclass(frozen=True)
class Weekly(RebalanceCalendar):
    """The last trading day on or before ``weekday`` in each week (``W-WED``: Wednesday, or Tuesday
    when Wednesday is a holiday)."""

    weekday: int = 2
    name: str = "W-WED"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        if d.weekday() > self.weekday:
            return False
        if next_td is None:
            return d.weekday() == self.weekday
        same_week = next_td.isocalendar()[:2] == d.isocalendar()[:2]
        return not same_week or next_td.weekday() > self.weekday


@dataclass(frozen=True)
class MonthStart(RebalanceCalendar):
    """First trading day of the month (``BMS``; the legacy ``"M"``)."""

    name: str = "BMS"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        if last is not None:
            return (d.year, d.month) != (last.year, last.month)
        return prev_td is None or prev_td.month != d.month


@dataclass(frozen=True)
class MonthEnd(RebalanceCalendar):
    """Last trading day of the month (``BME``)."""

    name: str = "BME"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return next_td is None or next_td.month != d.month


@dataclass(frozen=True)
class QuarterStart(RebalanceCalendar):
    name: str = "BQS"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        q = (d.year, (d.month - 1) // 3)
        if last is not None:
            return q != (last.year, (last.month - 1) // 3)
        return prev_td is None or (prev_td.year, (prev_td.month - 1) // 3) != q


@dataclass(frozen=True)
class QuarterEnd(RebalanceCalendar):
    name: str = "BQE"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return next_td is None or (next_td.year, (next_td.month - 1) // 3) != (d.year, (d.month - 1) // 3)


@dataclass(frozen=True)
class Explicit(RebalanceCalendar):
    """Explicit target dates aligned to trading days: a target on a holiday moves to the previous
    trading day (``align="on_or_before"``) or the next one (``"on_or_after"``)."""

    targets: tuple[date, ...] = ()
    align: str = "on_or_before"
    name: str = "EXPLICIT"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        ts = self.targets
        if not ts:
            return False
        if self.align == "on_or_before":
            # due if some target t satisfies d <= t < next_td
            i = bisect_left(ts, d)
            return i < len(ts) and (next_td is None or ts[i] < next_td)
        # on_or_after: due if some target t satisfies prev_td < t <= d
        i = bisect_right(ts, d) - 1
        return i >= 0 and (prev_td is None or ts[i] > prev_td)


def IndexCalendar(name: str, dates: Iterable[date], align: str = "on_or_before") -> Explicit:
    return Explicit(tuple(sorted(set(dates))), align, f"INDEX:{name}")


@dataclass(frozen=True)
class Triggered(RebalanceCalendar):
    """Evaluated on every ``base`` date; the engine computes the proposed target and rebalances only
    when ``measure`` exceeds ``threshold``: ``"turnover"`` (one-way, of NAV) or ``"signal"`` (mean
    absolute change in cross-sectional percentile rank of ``field`` since the last rebalance)."""

    threshold: float = 0.05
    measure: str = "turnover"
    field: str | None = None
    base: RebalanceCalendar = dataclasses.field(default_factory=Daily)
    name: str = "TRIGGERED"
    triggered: bool = True

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return self.base.is_due(d, last, prev_td, next_td, elapsed)


_FIXED = {"D": Daily, "W": WeekStart, "M": MonthStart, "BMS": MonthStart, "MS": MonthStart, "BME": MonthEnd, "ME": MonthEnd, "Q": QuarterStart, "BQS": QuarterStart, "QS": QuarterStart, "BQE": QuarterEnd, "QE": QuarterEnd}


def parse_calendar(spec: str | int | RebalanceCalendar) -> RebalanceCalendar:
    if isinstance(spec, RebalanceCalendar):
        return spec
    if isinstance(spec, int) or str(spec).isdigit():
        return EveryN(int(spec), name=str(int(spec)))
    s = str(spec).upper()
    if s in _FIXED:
        return _FIXED[s]()
    if s.startswith("W-") and s[2:] in WEEKDAYS:
        return Weekly(WEEKDAYS[s[2:]], name=s)
    raise ValueError(f"unknown rebalance calendar {spec!r}; use D, N, W, W-<DAY>, M/BMS, BME, Q/BQS, BQE or a RebalanceCalendar")


__all__ = [
    "Daily",
    "EveryN",
    "Explicit",
    "IndexCalendar",
    "MonthEnd",
    "MonthStart",
    "QuarterEnd",
    "QuarterStart",
    "RebalanceCalendar",
    "Triggered",
    "WeekStart",
    "Weekly",
    "parse_calendar",
]
