"""A bitemporal field store: every vendor pull of every field, never overwritten, read as of a
knowledge time.

Layout (Parquet, partitioned so a ten-year daily history is read by year, not loaded whole)::

    <root>/<field>/year=<YYYY>/kt=<knowledge time>.parquet      columns: entity, date, value, source

``read(field, start, end, as_of)`` returns, for each (entity, date), the value from the latest
snapshot known at ``as_of`` (``mode="latest"``) or the first one ever published
(``mode="first"``: what a live process would have seen before any revision). ``panel`` assembles
several fields into a ``Panel``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nautilus_fi.data.panel import Panel


def _kt(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts) if ts is not None else pd.Timestamp.now(tz="UTC")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _kt_from_stem(stem: str) -> pd.Timestamp:
    s = stem[len("kt="):]
    date_part, _, time_part = s.partition("T")
    return pd.Timestamp(f"{date_part}T{time_part.replace('-', ':')}")


class FieldStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write
    def write(self, field: str, df: pd.DataFrame, knowledge_time=None, source: str = "", entity_col: str = "instrument_id", date_col: str = "date", value_col: str | None = None, tags: dict | None = None) -> list[Path]:
        """Store one pull of ``field`` (columns entity, date, value). One file per year touched;
        an existing snapshot is never overwritten."""
        kt = _kt(knowledge_time)
        value_col = value_col or (field if field in df.columns else "value")
        frame = pd.DataFrame({"entity": df[entity_col].astype(str).to_numpy(), "date": pd.to_datetime(df[date_col]).dt.normalize().to_numpy(), "value": pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)})
        frame["source"] = source
        paths = []
        stamp = kt.isoformat().replace(":", "-")
        for year, g in frame.groupby(frame["date"].dt.year):
            folder = self.root / field / f"year={int(year)}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"kt={stamp}.parquet"
            if path.exists():
                raise FileExistsError(f"snapshot already exists and must not be overwritten: {path}")
            table = pa.Table.from_pandas(g.reset_index(drop=True), preserve_index=False)
            meta = dict(table.schema.metadata or {})
            meta[b"nautilus_fi.knowledge_time"] = kt.isoformat().encode()
            meta[b"nautilus_fi.tags"] = json.dumps(tags or {}).encode()
            pq.write_table(table.replace_schema_metadata(meta), path)
            paths.append(path)
        return paths

    def write_wide(self, df: pd.DataFrame, fields: Iterable[str], knowledge_time=None, source: str = "", entity_col: str = "instrument_id", date_col: str = "date") -> dict[str, list[Path]]:
        return {f: self.write(f, df[[entity_col, date_col, f]].dropna(subset=[f]), knowledge_time, source, entity_col, date_col, value_col=f) for f in fields}

    # ------------------------------------------------------------------ read
    def fields(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def snapshots(self, field: str, start=None, end=None) -> list[tuple[pd.Timestamp, Path]]:
        folder = self.root / field
        if not folder.exists():
            return []
        y0 = pd.Timestamp(start).year if start is not None else None
        y1 = pd.Timestamp(end).year if end is not None else None
        out = []
        for ydir in sorted(folder.glob("year=*")):
            y = int(ydir.name.split("=", 1)[1])
            if (y0 is not None and y < y0) or (y1 is not None and y > y1):
                continue
            out += [(_kt_from_stem(p.stem), p) for p in sorted(ydir.glob("kt=*.parquet"))]
        return out

    def read(self, field: str, start=None, end=None, as_of=None, mode: str = "latest", entities: Iterable[str] | None = None) -> pd.DataFrame:
        snaps = self.snapshots(field, start, end)
        if as_of is not None:
            cut = _kt(as_of)
            snaps = [(kt, p) for kt, p in snaps if kt <= cut]
        if not snaps:
            return pd.DataFrame(columns=["entity", "date", "value", "source", "knowledge_time"])
        want = set(map(str, entities)) if entities is not None else None
        frames = []
        for kt, p in snaps:
            f = pq.read_table(p).to_pandas()
            if start is not None:
                f = f[f["date"] >= pd.Timestamp(start)]
            if end is not None:
                f = f[f["date"] <= pd.Timestamp(end)]
            if want is not None:
                f = f[f["entity"].isin(want)]
            if len(f):
                f = f.assign(knowledge_time=kt)
                frames.append(f)
        if not frames:
            return pd.DataFrame(columns=["entity", "date", "value", "source", "knowledge_time"])
        m = pd.concat(frames, ignore_index=True).sort_values("knowledge_time", kind="stable")
        keep = "last" if mode == "latest" else "first"
        return m.drop_duplicates(subset=["entity", "date"], keep=keep).sort_values(["date", "entity"]).reset_index(drop=True)

    def panel(self, fields: Iterable[str], start=None, end=None, as_of=None, mode: str = "latest", entities: Iterable[str] | None = None, dates: pd.DatetimeIndex | None = None, dtype=np.float64) -> Panel:
        fields = list(fields)
        longs = {f: self.read(f, start, end, as_of, mode, entities) for f in fields}
        all_dates = dates if dates is not None else pd.DatetimeIndex(sorted(set().union(*[set(df["date"]) for df in longs.values()]))) if longs else pd.DatetimeIndex([])
        all_ids = pd.Index(sorted(set(map(str, entities)))) if entities is not None else pd.Index(sorted(set().union(*[set(df["entity"]) for df in longs.values()])))
        vals = {}
        for f, df in longs.items():
            m = np.full((len(all_dates), len(all_ids)), np.nan, dtype=dtype)
            if len(df):
                ri = pd.DatetimeIndex(all_dates).get_indexer(pd.to_datetime(df["date"]))
                ci = all_ids.get_indexer(df["entity"])
                ok = (ri >= 0) & (ci >= 0)
                m[ri[ok], ci[ok]] = df["value"].to_numpy()[ok]
            vals[f] = m
        return Panel(pd.DatetimeIndex(all_dates), all_ids, vals)

    def describe(self) -> pd.DataFrame:
        rows = []
        for f in self.fields():
            for kt, p in self.snapshots(f):
                rows.append({"field": f, "year": p.parent.name.split("=", 1)[1], "knowledge_time": kt, "rows": pq.read_metadata(p).num_rows, "path": str(p)})
        return pd.DataFrame(rows)


__all__ = ["FieldStore"]
