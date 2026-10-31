"""File loaders: vendor extracts (CSV, Parquet, Excel) into canonical frames with the same mappings
as the SQL loaders - Bloomberg index constituents exports (BQL / PORT), custodian positions,
one-off vendor files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nautilus_fi.data.loaders.base import DatasetMapping, canonicalise


def read_any(path: str | Path, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p, **kwargs)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, **kwargs)
    return pd.read_csv(p, **kwargs)


def load_file(path: str | Path, mapping: DatasetMapping, start=None, end=None, **read_kwargs) -> pd.DataFrame:
    df = canonicalise(read_any(path, **read_kwargs), mapping)
    if mapping.date_column in df and (start is not None or end is not None):
        d = pd.to_datetime(df[mapping.date_column])
        keep = pd.Series(True, index=df.index)
        if start is not None:
            keep &= d >= pd.Timestamp(start)
        if end is not None:
            keep &= d <= pd.Timestamp(end)
        df = df[keep]
    return df.reset_index(drop=True)


BLOOMBERG_INDEX_MEMBERS = DatasetMapping("benchmark_constituents", "file", {"date": "Date", "instrument_id": "ID_CUSIP", "weight": "Weight"}, scale={"weight": 0.01}, description="Bloomberg index members export (weights in percent)")


def load_bloomberg_index_members(path: str | Path, mapping: DatasetMapping = BLOOMBERG_INDEX_MEMBERS, **kwargs) -> pd.DataFrame:
    df = load_file(path, mapping, **kwargs)
    df["weight"] = df.groupby("date")["weight"].transform(lambda w: w / w.sum())
    return df


__all__ = ["BLOOMBERG_INDEX_MEMBERS", "load_bloomberg_index_members", "load_file", "read_any"]
