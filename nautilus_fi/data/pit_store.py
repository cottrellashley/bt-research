"""A minimal point-in-time (bitemporal) Parquet store.

Rules from the reference architecture:

* store every vendor pull with a retrieval (knowledge) timestamp and **never overwrite**;
* read "as of" a knowledge time so revised history cannot silently change results;
* keep matured, called and defaulted instruments.

Layout::

    <root>/<dataset>/kt=<knowledge_time ISO, ':' -> '-'>.parquet

``read(dataset, as_of)`` returns the latest snapshot known at ``as_of``; ``read_asof(dataset, as_of,
keys)`` merges all snapshots known at ``as_of`` keeping each key's most recent revision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class PointInTimeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------------------------
    def write(self, dataset: str, df: pd.DataFrame, knowledge_time: pd.Timestamp | None = None, tags: dict | None = None) -> Path:
        kt = pd.Timestamp(knowledge_time) if knowledge_time is not None else pd.Timestamp.now(tz="UTC")
        kt = kt.tz_localize("UTC") if kt.tzinfo is None else kt.tz_convert("UTC")
        folder = self.root / dataset
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"kt={kt.isoformat().replace(':', '-')}.parquet"
        if path.exists():
            raise FileExistsError(f"Snapshot already exists and must not be overwritten: {path}")
        table = pa.Table.from_pandas(df, preserve_index=False)
        meta = dict(table.schema.metadata or {})
        meta[b"nautilus_fi.knowledge_time"] = kt.isoformat().encode()
        meta[b"nautilus_fi.tags"] = json.dumps(tags or {}).encode()
        pq.write_table(table.replace_schema_metadata(meta), path)
        return path

    # -- reading -----------------------------------------------------------------------------
    def snapshots(self, dataset: str) -> list[tuple[pd.Timestamp, Path]]:
        folder = self.root / dataset
        if not folder.exists():
            return []
        out = []
        for p in sorted(folder.glob("kt=*.parquet")):
            stamp = p.stem[len("kt="):]
            # restore ':' in the time part (date part has '-' separators that must stay)
            date_part, _, time_part = stamp.partition("T")
            kt = pd.Timestamp(f"{date_part}T{time_part.replace('-', ':')}")
            out.append((kt, p))
        return out

    def _known(self, dataset: str, as_of: pd.Timestamp | None) -> list[tuple[pd.Timestamp, Path]]:
        snaps = self.snapshots(dataset)
        if as_of is None:
            return snaps
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
        return [(kt, p) for kt, p in snaps if kt <= as_of]

    def read(self, dataset: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Latest snapshot whose knowledge time is <= ``as_of`` (all history if ``as_of`` is None)."""
        known = self._known(dataset, as_of)
        if not known:
            raise FileNotFoundError(f"No snapshot of {dataset!r} known at {as_of}")
        return pq.read_table(known[-1][1]).to_pandas()

    def read_asof(self, dataset: str, as_of: pd.Timestamp | None, keys: list[str]) -> pd.DataFrame:
        """Merge all snapshots known at ``as_of``; for each key keep the most recently published row."""
        known = self._known(dataset, as_of)
        if not known:
            raise FileNotFoundError(f"No snapshot of {dataset!r} known at {as_of}")
        frames = []
        for kt, p in known:
            f = pq.read_table(p).to_pandas()
            f["_knowledge_time"] = kt
            frames.append(f)
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("_knowledge_time").drop_duplicates(subset=keys, keep="last")
        return merged.reset_index(drop=True)

    def read_first_known(self, dataset: str, keys: list[str]) -> pd.DataFrame:
        """Each key's first published value (what was knowable at the time, before any revision)."""
        known = self._known(dataset, None)
        if not known:
            raise FileNotFoundError(f"No snapshot of {dataset!r}")
        frames = []
        for kt, p in known:
            f = pq.read_table(p).to_pandas()
            f["_knowledge_time"] = kt
            frames.append(f)
        merged = pd.concat(frames, ignore_index=True)
        return merged.sort_values("_knowledge_time").drop_duplicates(subset=keys, keep="first").reset_index(drop=True)

    def describe(self) -> pd.DataFrame:
        rows = []
        for ds in sorted(p.name for p in self.root.iterdir() if p.is_dir()):
            for kt, p in self.snapshots(ds):
                meta = pq.read_metadata(p)
                rows.append({"dataset": ds, "knowledge_time": kt, "rows": meta.num_rows, "path": str(p)})
        return pd.DataFrame(rows)


__all__ = ["PointInTimeStore"]
