"""Daily timeline of a fixed income backtest.

Each business day has four moments (venue wall-clock, expressed in UTC labels by default):

* ``window_time``       the execution window: indicative quotes arrive, resting orders can fill
* ``mark_time``         evaluated prices / marks are struck (``ts_event``)
* ``mark_publish_time`` marks become knowable (``ts_init``): strategies and the portfolio see them
* ``decision_time``     strategies run; orders submitted now rest until the next window (t+1)

That ordering is what enforces the "signal from day-t close cannot trade at day-t close" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import pandas as pd

NS_PER_DAY = 86_400_000_000_000


@dataclass(frozen=True)
class ExecutionWindowSchedule:
    window_time: time = time(10, 0)
    mark_time: time = time(16, 0)
    mark_publish_time: time = time(16, 30)
    decision_time: time = time(16, 45)
    snapshot_time: time = time(16, 35)

    @staticmethod
    def to_ns(d: date, t: time) -> int:
        return int(pd.Timestamp(datetime.combine(d, t, tzinfo=UTC)).value)

    def window_ns(self, d: date) -> int:
        return self.to_ns(d, self.window_time)

    def mark_ns(self, d: date) -> int:
        return self.to_ns(d, self.mark_time)

    def publish_ns(self, d: date) -> int:
        return self.to_ns(d, self.mark_publish_time)

    def decision_ns(self, d: date) -> int:
        return self.to_ns(d, self.decision_time)

    def snapshot_ns(self, d: date) -> int:
        return self.to_ns(d, self.snapshot_time)

    @staticmethod
    def date_of(ns: int) -> date:
        return pd.Timestamp(ns, tz="UTC").date()

    @staticmethod
    def time_of(ns: int) -> time:
        return pd.Timestamp(ns, tz="UTC").time()

    def is_window(self, ns: int) -> bool:
        return self.time_of(ns) == self.window_time

    def is_publish(self, ns: int) -> bool:
        return self.time_of(ns) == self.mark_publish_time


__all__ = ["NS_PER_DAY", "ExecutionWindowSchedule"]
