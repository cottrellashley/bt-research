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


def purged_train_indices(dates: pd.DatetimeIndex, test_start: pd.Timestamp, test_end: pd.Timestamp, label_horizon_days: int, embargo_days: int = 0) -> np.ndarray:
    """Training observations whose labels (observation + horizon) overlap the test window are purged;
    an embargo removes observations just after the test window."""
    dates = pd.DatetimeIndex(dates)
    label_end = dates + pd.Timedelta(days=label_horizon_days)
    overlaps = (label_end >= test_start) & (dates <= test_end)
    embargo = (dates > test_end) & (dates <= test_end + pd.Timedelta(days=embargo_days))
    keep = ~(overlaps | embargo)
    return np.where(keep)[0]


__all__ = ["Split", "purged_train_indices", "walk_forward_splits"]
