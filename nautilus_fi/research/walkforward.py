"""Chronological development / validation / test splits with purging and embargo for labels that
mature after their observation date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from nautilus_fi.instruments.schedule import add_months


@dataclass(frozen=True)
class Split:
    dev_start: date
    dev_end: date
    val_end: date
    test_end: date

    @property
    def test_start(self) -> date:
        return self.val_end


def walk_forward_splits(start: date, end: date, dev_months: int = 36, val_months: int = 12, test_months: int = 12, step_months: int = 12, expanding: bool = True) -> list[Split]:
    splits: list[Split] = []
    dev_start = start
    dev_end = add_months(start, dev_months)
    while True:
        val_end = add_months(dev_end, val_months)
        test_end = add_months(val_end, test_months)
        if test_end > end:
            break
        splits.append(Split(dev_start, dev_end, val_end, test_end))
        dev_end = add_months(dev_end, step_months)
        if not expanding:
            dev_start = add_months(dev_start, step_months)
    return splits


def purged_train_indices(dates: pd.DatetimeIndex, test_start: pd.Timestamp, test_end: pd.Timestamp, label_horizon_days: int, embargo_days: int = 0, unit: str = "trading", trading_days: pd.DatetimeIndex | None = None) -> np.ndarray:
    """Training observations whose labels (observation + horizon) overlap the test window are purged;
    an embargo removes observations just after the test window.

    ``unit="trading"`` (default) counts horizon and embargo in trading days on ``trading_days``
    (weekdays spanning the sample when not given; pass the holiday calendar's days for exactness);
    ``unit="calendar"`` counts calendar days."""
    dates = pd.DatetimeIndex(dates)
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    if unit == "calendar":
        label_end = dates + pd.Timedelta(days=label_horizon_days)
        embargo_end = test_end + pd.Timedelta(days=embargo_days)
    elif unit == "trading":
        td = pd.DatetimeIndex(trading_days) if trading_days is not None else pd.bdate_range(min(dates.min(), test_start), max(dates.max(), test_end) + pd.Timedelta(days=7 * (label_horizon_days + embargo_days) // 5 + 14))
        td = td.sort_values()
        pos = np.clip(td.searchsorted(dates), 0, len(td) - 1)
        label_end = td[np.clip(pos + label_horizon_days, 0, len(td) - 1)]
        end_pos = min(int(td.searchsorted(test_end, side="right")) - 1 + embargo_days, len(td) - 1)
        embargo_end = td[max(end_pos, 0)]
    else:
        raise ValueError("unit must be 'trading' or 'calendar'")
    overlaps = (label_end >= test_start) & (dates <= test_end)
    embargo = (dates > test_end) & (dates <= embargo_end)
    keep = ~(overlaps | embargo)
    return np.where(keep)[0]


__all__ = ["Split", "purged_train_indices", "walk_forward_splits"]
