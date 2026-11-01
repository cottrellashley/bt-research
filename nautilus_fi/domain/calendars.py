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
    when Wednesday is a holiday); ``every=2``: every other such week (``2W-WED``, biweekly)."""

    weekday: int = 2
    name: str = "W-WED"
    every: int = 1

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        if d.weekday() > self.weekday:
            return False
        if next_td is None:
            due = d.weekday() == self.weekday
        else:
            same_week = next_td.isocalendar()[:2] == d.isocalendar()[:2]
            due = not same_week or next_td.weekday() > self.weekday
        if due and self.every > 1 and last is not None:
            due = (d.toordinal() - d.weekday()) - (last.toordinal() - last.weekday()) >= 7 * self.every
        return due


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
class YearStart(RebalanceCalendar):
    """First trading day of the year (``BYS``)."""

    name: str = "BYS"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        if last is not None:
            return d.year != last.year
        return prev_td is None or prev_td.year != d.year


@dataclass(frozen=True)
class YearEnd(RebalanceCalendar):
    """Last trading day of the year (``BYE``; pandas ``Y`` / ``A``)."""

    name: str = "BYE"

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return next_td is None or next_td.year != d.year


@dataclass(frozen=True)
class NthBusinessDay(RebalanceCalendar):
    """The ``n``-th trading day of each month (``n=2``: the second; ``n=-1``: the last), on the
    trading days given (holidays already out)."""

    n: int = 2
    name: str = "NTH_BD"

    def dates(self, trading_days: Sequence[date], first_decision: bool = False) -> list[date]:
        by_month: dict[tuple, list[date]] = {}
        for d in trading_days:
            by_month.setdefault((d.year, d.month), []).append(d)
        out = []
        for days in by_month.values():
            k = self.n - 1 if self.n > 0 else self.n
            if -len(days) <= k < len(days):
                out.append(days[k])
        if first_decision and trading_days and (not out or out[0] != trading_days[0]):
            out = [trading_days[0], *out]
        return sorted(set(out))

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        raise NotImplementedError("NthBusinessDay decides from the month's trading days: use dates()")


@dataclass(frozen=True)
class AnchoredWeekly(RebalanceCalendar):
    """Anchor dates (the benchmark's month ends by default, or ``anchors``) plus rebalances
    ``offsets`` calendar days after each (7 / 14 / 21), rolled forward over holidays; an offset date
    with fewer than ``min_gap_days`` before the next anchor is skipped (Erin Beta+'s weekly rule)."""

    offsets: tuple[int, ...] = (7, 14, 21)
    min_gap_days: int = 7
    anchors: tuple[date, ...] = ()
    name: str = "ANCHORED_WEEKLY"

    def dates(self, trading_days: Sequence[date], first_decision: bool = False) -> list[date]:
        days = list(trading_days)
        if not days:
            return []
        anchors = sorted(set(self.anchors)) if self.anchors else MonthEnd().dates(days)
        snapped = [days[min(bisect_left(days, a), len(days) - 1)] for a in anchors]
        out = {a for a in snapped if days[0] <= a <= days[-1]}
        for i, a in enumerate(snapped):
            nxt = snapped[i + 1] if i + 1 < len(snapped) else None
            for off in self.offsets:
                target = date.fromordinal(a.toordinal() + off)
                j = bisect_left(days, target)                 # holidays roll forward
                if j >= len(days):
                    continue
                d = days[j]
                if nxt is not None and (d >= nxt or (nxt - d).days < self.min_gap_days):
                    continue
                out.add(d)
        if first_decision:
            out.add(days[0])
        return sorted(out)

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        raise NotImplementedError("AnchoredWeekly decides from the anchors: use dates()")


@dataclass(frozen=True)
class IndexDates(RebalanceCalendar):
    """The reconstitution dates of an index, read from the source's constituents when the run is
    prepared (``align`` as for ``Explicit``: providers publish reconstitutions ahead)."""

    index: str = ""
    align: str = "on_or_before"
    name: str = "INDEX_DATES"

    def resolve(self, source) -> Explicit:
        import pandas as pd

        c = source.constituents(self.index)
        ds = sorted({pd.Timestamp(x).date() for x in pd.to_datetime(c["date"])})
        return Explicit(tuple(ds), self.align, f"INDEX:{self.index}")

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        raise NotImplementedError(f"IndexDates({self.index!r}) is resolved from the source when the run is prepared")


@dataclass(frozen=True)
class Explicit(RebalanceCalendar):
    """Explicit target dates aligned to trading days: a target on a holiday moves to the next
    trading day (``align="on_or_after"``, the default: an instruction cannot act before it exists)
    or the previous one (``"on_or_before"``, for dates published ahead, like index reconstitutions)."""

    targets: tuple[date, ...] = ()
    align: str = "on_or_after"
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
    when ``measure`` reaches ``threshold``: ``"turnover"`` (of NAV, under ``turnover``: half the sum
    of absolute weight changes by default) or ``"signal"`` (mean absolute change in cross-sectional
    percentile rank since the last rebalance, of ``field`` when given, else the strategy's signal)."""

    threshold: float = 0.05
    measure: str = "turnover"                    # turnover | signal | score_drop (portfolio-weighted score fallen by threshold since the last rebalance)
    field: str | None = None
    turnover: str = "HALF_SUM"                   # ONE_WAY_BUYS | HALF_SUM | TWO_WAY
    base: RebalanceCalendar = dataclasses.field(default_factory=Daily)
    name: str = "TRIGGERED"
    triggered: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.base, RebalanceCalendar):                # "BME", "WS" ...: the named calendar
            object.__setattr__(self, "base", parse_calendar(self.base))

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return self.base.is_due(d, last, prev_td, next_td, elapsed)

    def dates(self, trading_days: Sequence[date], first_decision: bool = False) -> list[date]:
        return self.base.dates(trading_days, first_decision)


@dataclass(frozen=True)
class WithInterim(RebalanceCalendar):
    """A fixed calendar plus interim rebalances between its dates: on every ``interim`` base day
    the engine checks the ``interim`` trigger and rebalances with ``portfolio`` (default: the
    strategy's) when it fires. ``fixed`` dates always rebalance with the strategy's construction."""

    fixed: RebalanceCalendar = dataclasses.field(default_factory=MonthEnd)
    interim: Triggered = dataclasses.field(default_factory=lambda: Triggered(0.1, "score_drop"))
    portfolio: object = None
    name: str = "WITH_INTERIM"

    def __post_init__(self) -> None:
        if not isinstance(self.fixed, RebalanceCalendar):
            object.__setattr__(self, "fixed", parse_calendar(self.fixed))

    def dates(self, trading_days: Sequence[date], first_decision: bool = False) -> list[date]:
        return sorted(set(self.fixed.dates(trading_days, first_decision)) | set(self.interim.dates(trading_days, False)))

    def fixed_dates(self, trading_days: Sequence[date], first_decision: bool = False) -> set[date]:
        return set(self.fixed.dates(trading_days, first_decision))

    def is_due(self, d, last, prev_td, next_td, elapsed=None) -> bool:
        return self.fixed.is_due(d, last, prev_td, next_td, elapsed) or self.interim.is_due(d, last, prev_td, next_td, elapsed)


_FIXED = {"D": Daily, "W": WeekStart, "WS": WeekStart, "M": MonthStart, "BMS": MonthStart, "MS": MonthStart, "BME": MonthEnd, "ME": MonthEnd, "Q": QuarterStart, "BQS": QuarterStart, "QS": QuarterStart, "BQE": QuarterEnd, "QE": QuarterEnd,
          "Y": YearEnd, "A": YearEnd, "YE": YearEnd, "BYE": YearEnd, "BA": YearEnd, "YS": YearStart, "BYS": YearStart, "AS": YearStart}
_LEGACY = {"M": "BMS", "W": "WS", "Q": "BQS"}           # period starts here, period ends in pandas: say which


def parse_calendar(spec: str | int | RebalanceCalendar) -> RebalanceCalendar:
    if isinstance(spec, RebalanceCalendar):
        return spec
    if isinstance(spec, int) or str(spec).isdigit():
        return EveryN(int(spec), name=str(int(spec)))
    s = str(spec).upper()
    if s in _LEGACY:
        import warnings

        warnings.warn(f"rebalance calendar {spec!r} means the first trading day of the period here and the period end in pandas: write {_LEGACY[s]!r} (or the end: {_LEGACY[s].replace('S', 'E') if s != 'W' else 'W-FRI'!r})", FutureWarning, stacklevel=2)
    if s in _FIXED:
        return _FIXED[s]()
    if s.startswith("W-") and s[2:] in WEEKDAYS:
        return Weekly(WEEKDAYS[s[2:]], name=s)
    if s.startswith("2W-") and s[3:] in WEEKDAYS:
        return Weekly(WEEKDAYS[s[3:]], name=s, every=2)
    raise ValueError(f"unknown rebalance calendar {spec!r}; use D, N, WS, W-<DAY>, 2W-<DAY>, BMS, BME, BQS, BQE, BYS, BYE or a RebalanceCalendar")


__all__ = [
    "AnchoredWeekly",
    "Daily",
    "EveryN",
    "Explicit",
    "IndexCalendar",
    "IndexDates",
    "MonthEnd",
    "MonthStart",
    "NthBusinessDay",
    "QuarterEnd",
    "QuarterStart",
    "RebalanceCalendar",
    "Triggered",
    "WeekStart",
    "Weekly",
    "WithInterim",
    "YearEnd",
    "YearStart",
    "parse_calendar",
]
