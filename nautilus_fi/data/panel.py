"""Dense panels: one (dates x entities) matrix per field.

The panel engine, signal evaluation, benchmarks and the optimiser all work on cross-sections and
time series of the same fields, so they share one columnar representation instead of per-bond-day
objects: a 9,000-bond x 10-year daily field is a 2,520 x 9,000 float32 matrix (91 MB).

Point-in-time access goes through ``asof_row``: a decision on date T may use row ``T - lag`` where
``lag`` is the field's availability lag in trading days (``domain.fields``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from nautilus_fi.domain.fields import field_def


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    ids: pd.Index
    values: dict[str, np.ndarray] = field(default_factory=dict)       # field -> (T, N)
    lags: dict[str, int] = field(default_factory=dict)                 # field -> availability lag (trading days)

    def __post_init__(self) -> None:
        self.dates = pd.DatetimeIndex(self.dates)
        self.ids = pd.Index(self.ids)
        self._pos = {d: i for i, d in enumerate(self.dates.date)}
        self._id_pos = {k: i for i, k in enumerate(self.ids)}
        for k, v in self.values.items():
            if v.shape != (len(self.dates), len(self.ids)):
                raise ValueError(f"field {k}: shape {v.shape} != ({len(self.dates)}, {len(self.ids)})")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_long(cls, df: pd.DataFrame, fields: Iterable[str] | None = None, date_col: str = "date", id_col: str = "instrument_id", dates: pd.DatetimeIndex | None = None, ids: Iterable[str] | None = None, dtype=np.float64) -> Panel:
        """Long frame (one row per entity-date, one column per field) -> panel."""
        d = pd.to_datetime(df[date_col]).dt.normalize()
        fields = list(fields) if fields is not None else [c for c in df.columns if c not in (date_col, id_col, "publish_ts", "knowledge_time", "source")]
        all_dates = pd.DatetimeIndex(sorted(d.unique())) if dates is None else pd.DatetimeIndex(dates)
        all_ids = pd.Index(sorted(df[id_col].astype(str).unique())) if ids is None else pd.Index(list(ids))
        ri = all_dates.get_indexer(d)
        ci = all_ids.get_indexer(df[id_col].astype(str))
        ok = (ri >= 0) & (ci >= 0)
        values = {}
        for f in fields:
            m = np.full((len(all_dates), len(all_ids)), np.nan, dtype=dtype)
            m[ri[ok], ci[ok]] = pd.to_numeric(df[f], errors="coerce").to_numpy()[ok]
            values[f] = m
        return cls(all_dates, all_ids, values, {f: field_def(f).availability_lag_days for f in fields})

    @classmethod
    def from_wide(cls, frames: dict[str, pd.DataFrame], dtype=np.float64) -> Panel:
        """``{field: DataFrame(index=dates, columns=ids)}`` -> panel on the union of dates and ids."""
        dates = pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(f.index)) for f in frames.values()])))
        ids = pd.Index(sorted(set().union(*[set(map(str, f.columns)) for f in frames.values()])))
        vals = {}
        for k, f in frames.items():
            g = f.copy()
            g.index = pd.to_datetime(g.index)
            g.columns = g.columns.map(str)
            vals[k] = g.reindex(index=dates, columns=ids).to_numpy(dtype=dtype)
        return cls(dates, ids, vals, {k: field_def(k).availability_lag_days for k in frames})

    # ------------------------------------------------------------------ access
    @property
    def fields(self) -> list[str]:
        return list(self.values)

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.values[name], index=self.dates, columns=self.ids)

    def date_pos(self, d: date | pd.Timestamp) -> int:
        """Position of ``d``; a date between panel dates maps to the previous one (-1 before the start)."""
        d = pd.Timestamp(d).date()
        if d in self._pos:
            return self._pos[d]
        return int(self.dates.searchsorted(pd.Timestamp(d), side="right")) - 1

    def id_pos(self, ids: Iterable[str]) -> np.ndarray:
        return np.array([self._id_pos.get(k, -1) for k in ids], dtype=np.int64)

    def asof_row(self, name: str, i: int) -> np.ndarray:
        """Row of ``name`` usable by a decision at date position ``i`` (shifted by the field's lag)."""
        j = i - self.lags.get(name, 0)
        if j < 0:
            return np.full(len(self.ids), np.nan)
        return self.values[name][j]

    def cross_section(self, i: int, names: Iterable[str], cols: np.ndarray | None = None) -> pd.DataFrame:
        names = list(names)
        idx = self.ids if cols is None else self.ids[cols]
        data = {n: (self.asof_row(n, i) if cols is None else self.asof_row(n, i)[cols]) for n in names if n in self.values}
        return pd.DataFrame(data, index=idx)

    def history(self, name: str, i: int, n: int, cols: np.ndarray | None = None) -> pd.DataFrame:
        """The ``n`` rows of ``name`` known at position ``i`` (lag-shifted), oldest first."""
        j = i - self.lags.get(name, 0)
        lo = max(0, j - n + 1)
        block = self.values[name][lo:j + 1] if j >= 0 else self.values[name][:0]
        if cols is not None:
            block = block[:, cols]
        return pd.DataFrame(block, index=self.dates[lo:j + 1] if j >= 0 else self.dates[:0], columns=self.ids if cols is None else self.ids[cols])

    # ------------------------------------------------------------------ transforms
    def with_field(self, name: str, matrix: np.ndarray | pd.DataFrame, lag: int | None = None) -> Panel:
        m = matrix.reindex(index=self.dates, columns=self.ids).to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
        if m.shape != (len(self.dates), len(self.ids)):
            raise ValueError(f"field {name}: shape {m.shape} does not match the panel")
        self.values[name] = m
        self.lags[name] = field_def(name).availability_lag_days if lag is None else lag
        return self

    def subset(self, ids: Iterable[str] | None = None, start=None, end=None) -> Panel:
        rows = np.ones(len(self.dates), dtype=bool)
        if start is not None:
            rows &= self.dates >= pd.Timestamp(start)
        if end is not None:
            rows &= self.dates <= pd.Timestamp(end)
        cols = np.arange(len(self.ids)) if ids is None else self.id_pos(ids)
        cols = cols[cols >= 0]
        return Panel(self.dates[rows], self.ids[cols], {k: v[rows][:, cols] for k, v in self.values.items()}, dict(self.lags))

    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in self.values.values()))

    def to_long(self, names: Iterable[str] | None = None, dropna: bool = True) -> pd.DataFrame:
        names = list(names or self.values)
        t, n = len(self.dates), len(self.ids)
        out = pd.DataFrame({"date": np.repeat(self.dates.values, n), "instrument_id": np.tile(self.ids.values, t)})
        for k in names:
            out[k] = self.values[k].reshape(-1)
        return out.dropna(subset=names, how="all").reset_index(drop=True) if dropna else out


__all__ = ["Panel"]
