"""Loaders: vendor tables in, canonical frames out, every pull stamped with its knowledge time.

A ``DatasetMapping`` says where a canonical dataset (``evaluated_prices``, ``analytics``,
``benchmark_constituents``, ``reference``, ``ratings`` ...; see ``docs/data_contracts.md``) lives in
a source system: the query (``:start`` / ``:end`` placeholders), the source column for each
canonical column, unit scaling, and the source's knowledge-time column if it has one. ``SQLLoader``
runs mappings against any DB-API connection (Snowflake, Postgres, SQLite for tests), in chunks;
``load_into_stores`` writes each pull to the point-in-time stores (never overwriting) and
``builder_from_frames`` hands canonical frames to ``MarketEnvironmentBuilder``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nautilus_fi.data import schemas

CANONICAL = {
    "reference": schemas.REFERENCE, "evaluated_prices": schemas.EVALUATED_PRICES, "composite_quotes": schemas.COMPOSITE_QUOTES, "trades": schemas.TRADES, "curves": schemas.CURVES,
    "fixings": schemas.FIXINGS, "factors": schemas.FACTORS, "ratings": schemas.RATINGS, "corporate_actions": schemas.CORPORATE_ACTIONS, "stress": schemas.STRESS, "benchmark": schemas.BENCHMARK,
    "index_ratios": schemas.INDEX_RATIOS, "analytics": schemas.ANALYTICS, "fields": schemas.FIELDS, "benchmark_constituents": schemas.BENCHMARK_CONSTITUENTS, "supplied_cashflows": schemas.SUPPLIED_CASHFLOWS,
}


@dataclass
class DatasetMapping:
    name: str                                         # canonical dataset name (key of CANONICAL)
    query: str                                        # SQL with :start / :end, or a bare table name
    columns: dict[str, str]                           # canonical column -> source column
    date_column: str = "date"                         # canonical column the date filter applies to
    knowledge_time_column: str | None = None          # source column with the publication / load time
    scale: dict[str, float] = field(default_factory=dict)    # canonical column -> multiplier (e.g. percent -> fraction 0.01)
    constants: dict[str, object] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    description: str = ""

    def sql(self) -> str:
        q = self.query.strip()
        if re.fullmatch(r"[\w.\"]+", q):
            src = self.columns.get(self.date_column, self.date_column)
            return f"SELECT * FROM {q} WHERE {src} >= :start AND {src} <= :end"
        return q

    def to_dict(self) -> dict:
        return {"name": self.name, "query": self.query, "columns": self.columns, "date_column": self.date_column, "knowledge_time_column": self.knowledge_time_column, "scale": self.scale, "constants": self.constants, "params": self.params, "description": self.description}

    @staticmethod
    def from_dict(d: dict) -> DatasetMapping:
        return DatasetMapping(**d)


def load_catalog(path: str | Path) -> dict[str, DatasetMapping]:
    """A JSON catalogue of mappings (``{"datasets": [mapping, ...], "variables": {...}}``); ``{var}``
    placeholders in queries are filled from ``variables`` (database, schema ...)."""
    raw = json.loads(Path(path).read_text())
    var = raw.get("variables", {})
    out = {}
    for d in raw["datasets"]:
        d = dict(d)
        d["query"] = d["query"].format(**var)
        m = DatasetMapping.from_dict(d)
        out[m.name] = m
    return out


def canonicalise(df: pd.DataFrame, mapping: DatasetMapping) -> pd.DataFrame:
    """Source frame -> canonical columns (renamed, scaled, constants added), schema-validated."""
    src_to_can = {v: k for k, v in mapping.columns.items()}
    keep = [c for c in df.columns if c in src_to_can]
    out = df[keep].rename(columns=src_to_can).copy()
    for c, k in mapping.scale.items():
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce") * k
    for c, v in mapping.constants.items():
        out[c] = v
    if mapping.knowledge_time_column and mapping.knowledge_time_column in df.columns:
        out["publish_ts"] = pd.to_datetime(df[mapping.knowledge_time_column], utc=True).to_numpy()
    schema = CANONICAL.get(mapping.name)
    return schema.validate(out) if schema is not None else out


class SQLLoader:
    """Runs mappings on a DB-API 2.0 connection. ``connect`` returns a connection (called lazily and
    once); ``paramstyle`` is the driver's (qmark for SQLite, pyformat for the Snowflake default)."""

    def __init__(self, connect: Callable[[], object], paramstyle: str = "qmark", chunk_rows: int = 250_000) -> None:
        self._connect = connect
        self._conn = None
        self.paramstyle = paramstyle
        self.chunk_rows = chunk_rows
        self.pulls: list[dict] = []

    @property
    def conn(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _bind(self, sql: str, params: dict) -> tuple[str, object]:
        names = re.findall(r"(?<!:):(\w+)", sql)              # ':name' parameters, not '::type' casts
        if self.paramstyle == "named":
            return sql, params
        if self.paramstyle == "pyformat":
            return re.sub(r"(?<!:):(\w+)", r"%(\1)s", sql), params
        return re.sub(r"(?<!:):(\w+)", "?", sql), [params[n] for n in names]

    def fetch_raw(self, mapping: DatasetMapping, start=None, end=None) -> pd.DataFrame:
        params = {**mapping.params, "start": str(pd.Timestamp(start).date()) if start is not None else "1900-01-01", "end": str(pd.Timestamp(end).date()) if end is not None else "2999-12-31"}
        used = set(re.findall(r"(?<!:):(\w+)", mapping.sql()))
        sql, bound = self._bind(mapping.sql(), {k: v for k, v in params.items() if k in used})
        cur = self.conn.cursor()
        cur.execute(sql, bound)
        cols = [c[0] for c in cur.description]
        frames = []
        if hasattr(cur, "fetch_pandas_batches"):
            frames = list(cur.fetch_pandas_batches())
        else:
            while True:
                rows = cur.fetchmany(self.chunk_rows)
                if not rows:
                    break
                frames.append(pd.DataFrame(rows, columns=cols))
        cur.close()
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        self.pulls.append({"dataset": mapping.name, "rows": len(df), "start": params["start"], "end": params["end"], "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat()})
        return df

    def fetch(self, mapping: DatasetMapping, start=None, end=None) -> pd.DataFrame:
        return canonicalise(self.fetch_raw(mapping, start, end), mapping)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def load_into_stores(loader, mappings: Iterable[DatasetMapping], start=None, end=None, pit_store=None, field_store=None, knowledge_time=None, field_datasets: tuple[str, ...] = ("analytics", "fields")) -> dict[str, pd.DataFrame]:
    """Fetch each mapping; write frames to the ``PointInTimeStore`` and per-field panels to the
    ``FieldStore`` (analytics and generic fields), stamped with ``knowledge_time`` (default: now)."""
    out = {}
    kt = pd.Timestamp(knowledge_time) if knowledge_time is not None else pd.Timestamp.now(tz="UTC")
    for m in mappings:
        df = loader.fetch(m, start, end)
        out[m.name] = df
        if pit_store is not None:
            pit_store.write(m.name, df, kt, tags={"start": str(start), "end": str(end), "query": m.query[:200]})
        if field_store is not None and m.name in field_datasets and len(df):
            cols = [c for c in df.columns if c not in ("date", "instrument_id", "publish_ts")]
            field_store.write_wide(df, cols, kt, source=m.name)
    return out


BUILDER_METHODS = {"evaluated_prices": "with_evaluated_prices", "composite_quotes": "with_composite_quotes", "trades": "with_trades", "curves": "with_curves", "fixings": "with_fixings", "factors": "with_factors", "index_ratios": "with_index_ratios", "ratings": "with_ratings", "corporate_actions": "with_corporate_actions", "stress": "with_stress", "benchmark": "with_benchmark", "analytics": "with_analytics", "fields": "with_fields", "benchmark_constituents": "with_benchmark_constituents", "supplied_cashflows": "with_supplied_cashflows"}


def builder_from_frames(frames: dict[str, pd.DataFrame], start, end, **builder_kwargs):
    """Canonical frames -> a ``MarketEnvironmentBuilder`` (``reference`` is required)."""
    from nautilus_fi.backtest.environment import MarketEnvironmentBuilder

    if "reference" not in frames:
        raise ValueError("builder_from_frames needs the reference dataset")
    b = MarketEnvironmentBuilder(frames["reference"], start, end, **builder_kwargs)
    for name, df in frames.items():
        meth = BUILDER_METHODS.get(name)
        if meth is not None and df is not None and len(df):
            getattr(b, meth)(df)
    return b


__all__ = ["BUILDER_METHODS", "CANONICAL", "DatasetMapping", "SQLLoader", "builder_from_frames", "canonicalise", "load_catalog", "load_into_stores"]
