"""The data seam: one small protocol between the research kernel and wherever data lives.

A ``DataSource`` answers two questions: what is there (``catalog()``) and give me these fields for
these dates (``load(DataRequest) -> Panel``). Everything else is optional:

* static attributes per instrument (issuer, sector, currency): ``attributes(ids, names)``;
* date series (cash rates, index returns, a VIX level): ``series(names, start, end)``;
* index constituents and weights: ``constituents(index, start, end)``;
* overlay instruments (futures, CDX, swaps) as a second panel: ``load(..., entity="overlay")``;
* ``version()``: a string that changes when the data change, so results can be cached and
  reproduced (``None``: unknown, results are cached for the session only).

A company data platform plugs in by implementing ``catalog`` and ``load``, or with no class at all
through ``CallableSource(fetch, fields)``, where ``fetch(field, start, end, ids)`` returns a long
frame (date, instrument_id, value) or a wide one (dates x ids). An agent's data-access tool is
exactly such a function.

Adapters:

* ``PanelSource`` / ``FrameSource``: in-memory panels or pandas frames;
* ``SyntheticSource``: the synthetic credit universe (``synthetic.panels``) at any size;
* ``FieldStoreSource``: the bitemporal field store, read as of a knowledge time (latest or first
  publication);
* ``SQLSource``: dataset mappings on any DB-API connection (``data.loaders``; Snowflake included);
* ``CompositeSource``: route each field to the first source that has it;
* ``MappedSource``: identifier mapping (CUSIP -> ISIN, vendor id -> internal id);
* ``CachedSource``: memory and memory-mapped disk cache keyed by source version and request. Years
  of history load once, later windows are slices, and parallel workers share the same pages.

Availability: ``FieldInfo.lag`` is the number of trading days after its date a value may be used
(vendor analytics published the next morning: 1). The kernel applies it when it reads a field, so
every expression sees only what was known at the decision time.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nautilus_fi.core.fileio import atomic_write_text, file_lock
from nautilus_fi.data.panel import DataQualityError, Panel
from nautilus_fi.domain.fields import field_def

ASSET, ATTRIBUTE, SERIES, OVERLAY = "asset", "attribute", "series", "overlay"
ENTITIES = (ASSET, ATTRIBUTE, SERIES, OVERLAY)


class UnknownFieldError(KeyError):
    """A field, series, attribute or index the source does not have (with close matches)."""

    def __init__(self, name: str, kind: str = "field", suggestions: Iterable[str] = (), source: str = "") -> None:
        self.name, self.kind, self.suggestions = name, kind, list(suggestions)
        hint = f"; did you mean {', '.join(repr(s) for s in self.suggestions)}?" if self.suggestions else ""
        where = f" in {source}" if source else ""
        super().__init__(f"unknown {kind} {name!r}{where}{hint} (catalog().search(text) lists what exists)")

    def __str__(self) -> str:
        return self.args[0]


class MissingDataError(ValueError):
    """Data a strategy component needs is absent: the message says which and what to do."""




# ====================================================================================== catalogue
@dataclass(frozen=True)
class FieldInfo:
    name: str
    description: str = ""
    unit: str = ""
    kind: str = "numeric"                  # numeric | category | flag
    entity: str = ASSET                    # asset (dates x ids) | attribute (per id) | series (per date) | overlay
    lag: int = 0                           # trading days after its date a value may be used
    frequency: str = "D"
    source: str = ""
    start: str | None = None
    end: str | None = None
    tags: tuple[str, ...] = ()
    label: str = "end"                     # end: a value covers the period ending on its date | start: the period starting on it (START_DATE labelling)
    form: str = "simple"                   # a return field's form: simple | log | price (levels: returns are their changes) | cumulative (additive, fraction)
    categories: tuple = ()                 # labels of a coded category field (code k is categories[k]): dated sector / rating / issuer maps

    def __post_init__(self) -> None:
        if self.entity not in ENTITIES:
            raise ValueError(f"FieldInfo.entity must be one of {ENTITIES}, not {self.entity!r}")
        if self.lag < 0:
            raise ValueError(f"field {self.name}: a negative availability lag would read the future")
        if self.label not in ("end", "start"):
            raise ValueError(f"field {self.name}: label must be 'end' or 'start', not {self.label!r}")
        if self.form not in ("simple", "log", "price", "cumulative"):
            raise ValueError(f"field {self.name}: form must be simple, log, price or cumulative, not {self.form!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def field_info(name: str, entity: str = ASSET, lag: int | None = None, source: str = "", **kw) -> FieldInfo:
    """``FieldInfo`` from the field registry's definition (unit, description, lag) where it has one."""
    fd = field_def(name)
    return FieldInfo(name, kw.pop("description", fd.description), kw.pop("unit", fd.unit), kw.pop("kind", fd.kind if fd.kind in ("numeric", "category", "flag") else "numeric"), entity, fd.availability_lag_days if lag is None else int(lag), kw.pop("frequency", fd.frequency), source, **kw)


def fields_from_dictionary(df: pd.DataFrame, source: str = "") -> list[FieldInfo]:
    """``FieldInfo`` objects from a data dictionary table (one row per field; columns ``name`` and
    any of ``description``, ``unit``, ``kind``, ``entity``, ``lag``, ``frequency``, ``start``,
    ``end``, ``tags``): how a company platform's own metadata becomes a kernel catalogue."""
    out = []
    for rec in df.to_dict("records"):
        kw = {k: rec[k] for k in ("description", "unit", "kind", "entity", "frequency", "start", "end", "label") if k in rec and pd.notna(rec[k])}
        if "lag" in rec and pd.notna(rec["lag"]):
            kw["lag"] = int(rec["lag"])
        if "tags" in rec and isinstance(rec["tags"], str):
            kw["tags"] = tuple(t.strip() for t in rec["tags"].split(",") if t.strip())
        out.append(FieldInfo(str(rec["name"]), source=source, **{k: (str(v) if k in ("start", "end") else v) for k, v in kw.items()}))
    return out


class Catalog:
    """What a source holds: fields by entity, index names, and whether it can read as of a
    knowledge time. Searchable, so an agent can find ``oas`` without knowing the vendor's name."""

    def __init__(self, fields: Iterable[FieldInfo] = (), indices: Iterable[str] = (), name: str = "", supports_as_of: bool = False) -> None:
        self._f: dict[str, FieldInfo] = {}
        for f in fields:
            self._f.setdefault(f.name, f)
        self.indices = tuple(dict.fromkeys(indices))
        self.name = name
        self.supports_as_of = supports_as_of

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._f

    def __getitem__(self, name: str) -> FieldInfo:
        try:
            return self._f[name]
        except KeyError:
            raise UnknownFieldError(name, "field", self.suggest(name), self.name) from None

    def __iter__(self):
        return iter(self._f.values())

    def __len__(self) -> int:
        return len(self._f)

    def get(self, name: str, default: FieldInfo | None = None) -> FieldInfo | None:
        return self._f.get(name, default)

    def names(self, entity: str | None = None) -> list[str]:
        return [k for k, f in self._f.items() if entity is None or f.entity == entity]

    def suggest(self, name: str, n: int = 3) -> list[str]:
        names = list(self._f) + list(self.indices)
        close = difflib.get_close_matches(name, names, n=n, cutoff=0.5)
        sub = [k for k in names if name.lower() in k.lower() and k not in close]
        return (close + sub)[:n]

    def search(self, text: str) -> list[FieldInfo]:
        """Fields whose name, description or tags contain every word of ``text`` (case-insensitive)."""
        words = [w for w in re.split(r"\s+", text.lower()) if w]
        out = []
        for f in self._f.values():
            hay = " ".join([f.name, f.description, f.unit, " ".join(f.tags), f.entity]).lower()
            if all(w in hay for w in words):
                out.append(f)
        return sorted(out, key=lambda f: (not f.name.lower().startswith(words[0]) if words else False, f.name))

    def merge(self, other: Catalog) -> Catalog:
        return Catalog([*self, *other], [*self.indices, *other.indices], self.name or other.name, self.supports_as_of and other.supports_as_of)

    def frame(self) -> pd.DataFrame:
        rows = [f.to_dict() for f in self._f.values()]
        return pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame(columns=[*FieldInfo.__dataclass_fields__][1:])

    def describe(self, entity: str | None = None) -> str:
        lines = [f"catalog {self.name or ''} ({len(self)} fields{', indices: ' + ', '.join(self.indices) if self.indices else ''}{', as-of reads' if self.supports_as_of else ''})", "", "| field | entity | unit | lag | description |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| {f.name} | {f.entity} | {f.unit} | {f.lag} | {f.description} |" for f in self._f.values() if entity is None or f.entity == entity]
        return "\n".join(lines)

    def __repr__(self) -> str:
        by = {e: len(self.names(e)) for e in ENTITIES if self.names(e)}
        return f"Catalog({self.name!r}, {by}, indices={list(self.indices)})"


# ====================================================================================== requests
def _ts(x) -> pd.Timestamp | None:
    if x is None:
        return None
    t = pd.Timestamp(x)
    return t.tz_localize(None).normalize() if t.tzinfo is not None else t.normalize()


@dataclass(frozen=True)
class DataRequest:
    fields: tuple[str, ...]
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    ids: tuple[str, ...] | None = None
    as_of: pd.Timestamp | None = None             # knowledge-time cut (bitemporal sources)
    entity: str = ASSET

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(dict.fromkeys(self.fields)))
        object.__setattr__(self, "start", _ts(self.start))
        object.__setattr__(self, "end", _ts(self.end))
        if self.ids is not None:
            object.__setattr__(self, "ids", tuple(map(str, self.ids)))
        if self.as_of is not None:
            t = pd.Timestamp(self.as_of)
            object.__setattr__(self, "as_of", t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC"))

    def with_(self, **changes) -> DataRequest:
        return replace(self, **changes)

    @property
    def ids_key(self) -> str:
        return "all" if self.ids is None else hashlib.sha256("\x1f".join(sorted(self.ids)).encode()).hexdigest()[:16]

    def key(self) -> str:
        payload = {"fields": list(self.fields), "start": str(self.start), "end": str(self.end), "ids": self.ids_key, "as_of": str(self.as_of), "entity": self.entity}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def window(panel: Panel, start=None, end=None, ids: Iterable[str] | None = None, fields: Iterable[str] | None = None) -> Panel:
    """Rows of ``panel`` in [start, end] (views, no copies when ``ids`` is None)."""
    lo = 0 if start is None else int(panel.dates.searchsorted(_ts(start)))
    hi = len(panel.dates) if end is None else int(panel.dates.searchsorted(_ts(end), side="right"))
    names = list(fields) if fields is not None else panel.fields
    if ids is None:
        vals = {k: panel.values[k][lo:hi] for k in names}
        out_ids = panel.ids
    else:
        cols = panel.id_pos(list(ids))
        keep = cols >= 0
        cols = cols[keep]
        out_ids = panel.ids[cols]
        vals = {k: panel.values[k][lo:hi][:, cols] for k in names}
    return Panel(panel.dates[lo:hi], out_ids, vals, {k: panel.lags.get(k, 0) for k in names})


def align(panels: list[Panel], dates: pd.DatetimeIndex | None = None, ids: pd.Index | None = None) -> Panel:
    """Several panels on the union of their dates and ids (no copy for a single panel)."""
    panels = [p for p in panels if p is not None]
    if not panels:
        return Panel(pd.DatetimeIndex([]), pd.Index([], dtype=object), {})
    if len(panels) == 1 and dates is None and ids is None:
        return panels[0]
    all_dates = dates if dates is not None else panels[0].dates
    all_ids = ids if ids is not None else panels[0].ids
    if dates is None:
        for p in panels[1:]:
            if not p.dates.equals(all_dates):
                all_dates = all_dates.union(p.dates)
    if ids is None:
        for p in panels[1:]:
            if not p.ids.equals(all_ids):
                all_ids = all_ids.union(p.ids)
    vals, lags = {}, {}
    for p in panels:
        same = p.dates.equals(all_dates) and p.ids.equals(all_ids)
        ri = None if same else all_dates.get_indexer(p.dates)
        ci = None if same else all_ids.get_indexer(p.ids)
        for k, v in p.values.items():
            if same:
                m = v
            else:
                m = np.full((len(all_dates), len(all_ids)), np.nan, dtype=np.float64 if v.dtype.kind != "f" else v.dtype)
                ok_r, ok_c = ri >= 0, ci >= 0
                m[np.ix_(ri[ok_r], ci[ok_c])] = v[np.ix_(ok_r, ok_c)]
            if k in vals:                                     # a field from two panels: they must agree where both have it
                a, b = np.asarray(vals[k], dtype=np.float64), np.asarray(m, dtype=np.float64)
                both = np.isfinite(a) & np.isfinite(b)
                clash = both & ~np.isclose(a, b, rtol=1e-9, atol=1e-12)
                if clash.any():
                    r, c = np.argwhere(clash)[0]
                    raise DataQualityError(f"field {k!r} comes from more than one panel and they disagree on {int(clash.sum())} values (first {all_dates[r]:%Y-%m-%d} {all_ids[c]}: {a[r, c]:.6g} vs {b[r, c]:.6g}); keep one source per field")
                vals[k] = np.where(np.isfinite(a), a, b)
                continue
            vals[k] = m
            lags[k] = p.lags.get(k, 0)
    return Panel(pd.DatetimeIndex(all_dates), pd.Index(all_ids), vals, lags)


def array_fingerprint(a: np.ndarray, budget: int = 1 << 20) -> str:
    """Sampled content fingerprint of a large array (shape, dtype, strided sample, NaN-aware sum)."""
    a = np.asarray(a)
    flat = a.reshape(-1)
    step = max(1, flat.size // budget)
    h = hashlib.sha1(f"{a.shape}{a.dtype}".encode())
    sample = np.ascontiguousarray(flat[::step])
    h.update(sample.tobytes())
    if a.dtype.kind == "f":
        h.update(repr(float(np.nansum(flat, dtype=np.float64))).encode())
    return h.hexdigest()[:16]


def frame_fingerprint(df: pd.DataFrame | pd.Series | None) -> str:
    if df is None:
        return "none"
    try:
        return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[:16]
    except TypeError:
        return hashlib.sha1(df.to_csv().encode()).hexdigest()[:16]


# ====================================================================================== protocol
class DataSource:
    """Base class. Implement ``catalog`` and ``load``; the rest has defaults."""

    name: str = "source"

    def catalog(self) -> Catalog:
        raise NotImplementedError

    def load(self, request: DataRequest) -> Panel:
        raise NotImplementedError

    def attributes(self, ids: Iterable[str] | None = None, names: Iterable[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(index=pd.Index(list(ids) if ids is not None else [], dtype=object))

    def series(self, names: Iterable[str], start=None, end=None) -> pd.DataFrame:
        names = list(names)
        if names:
            raise UnknownFieldError(names[0], "series", self.catalog().suggest(names[0]), self.name)
        return pd.DataFrame()

    def constituents(self, index: str, start=None, end=None) -> pd.DataFrame:
        raise UnknownFieldError(index, "index", self.catalog().suggest(index), self.name)

    def environment(self, start, end, ids: Iterable[str] | None = None):
        """An event-engine ``MarketEnvironment`` for Implement fidelity (optional capability)."""
        raise NotImplementedError(f"{self.name} cannot build an event-engine environment: pass env= to implement(), or use a source with contract terms and evaluated prices")

    def version(self) -> str | None:
        return None

    def point_in_time(self) -> bool | None:
        """Whether every value is as it was first known (no later revisions): True when the source
        can show it (knowledge-time reads of first publications, generated data), False when it
        serves revised vintages, None when it cannot say. Official records need True."""
        return getattr(self, "_point_in_time", None)

    # ------------------------------------------------------------------ conveniences
    def panel(self, fields: Iterable[str], start=None, end=None, ids: Iterable[str] | None = None, as_of=None, entity: str = ASSET) -> Panel:
        return self.load(DataRequest(tuple(fields), start, end, None if ids is None else tuple(ids), as_of, entity))

    def check(self, request: DataRequest) -> None:
        """Raise ``UnknownFieldError`` for any requested field the catalogue does not list."""
        cat = self.catalog()
        for f in request.fields:
            info = cat.get(f)
            if info is None or (info.entity != request.entity and not (request.entity == ASSET and info.entity == ASSET)):
                raise UnknownFieldError(f, "field" if request.entity == ASSET else f"{request.entity} field", cat.suggest(f), self.name)

    def describe(self) -> str:
        return self.catalog().describe()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _series_frame(series, names, start, end, source_name: str, cat: Catalog | None = None) -> pd.DataFrame:
    out = {}
    for n in names:
        if n not in series:
            raise UnknownFieldError(n, "series", (cat.suggest(n) if cat is not None else difflib.get_close_matches(n, list(series), n=3)), source_name)
        s = pd.Series(series[n], dtype=float)
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
        out[n] = s
    df = pd.DataFrame(out)
    if start is not None:
        df = df[df.index >= _ts(start)]
    if end is not None:
        df = df[df.index <= _ts(end)]
    return df


# ====================================================================================== in memory
class PanelSource(DataSource):
    """A ``Panel`` (dates x ids per field) plus optional attributes, date series, index
    constituents and an overlay panel, all in memory. Knowledge-time reads are not supported
    (``as_of`` is ignored with a warning); availability lags come from ``fields`` or the panel."""

    def __init__(self, panel: Panel, attributes: pd.DataFrame | None = None, series: pd.DataFrame | Mapping[str, pd.Series] | None = None, constituents: Mapping[str, pd.DataFrame] | None = None, overlays: Panel | None = None, fields: Iterable[FieldInfo] | None = None, name: str = "panel", version: str | None = None, environment: Callable | None = None, point_in_time: bool | None = None) -> None:
        self._point_in_time = point_in_time
        self._panel = panel
        self._attrs = attributes if attributes is not None else pd.DataFrame(index=panel.ids)
        self._attrs.index = self._attrs.index.astype(str)
        if isinstance(series, pd.DataFrame):
            series = {c: series[c] for c in series.columns}
        self._series = dict(series or {})
        self._constituents = dict(constituents or {})
        self._overlays = overlays
        self._user_fields = {f.name: f for f in (fields or [])}
        self.name = name
        self._version = version
        self._env = environment
        self._cat: Catalog | None = None

    @property
    def raw_panel(self) -> Panel:
        return self._panel

    def catalog(self) -> Catalog:
        if self._cat is None:
            lo, hi = (str(self._panel.dates[0].date()), str(self._panel.dates[-1].date())) if len(self._panel.dates) else (None, None)
            fs = [self._user_fields.get(k) or field_info(k, ASSET, self._panel.lags.get(k), self.name, start=lo, end=hi) for k in self._panel.fields]
            for c in self._attrs.columns:
                kind = "numeric" if pd.api.types.is_numeric_dtype(self._attrs[c]) else "category"
                fs.append(self._user_fields.get(c) or FieldInfo(str(c), field_def(str(c)).description, "", kind, ATTRIBUTE, 0, "static", self.name))
            for k in self._series:
                fs.append(self._user_fields.get(k) or FieldInfo(k, "", "", "numeric", SERIES, 0, "D", self.name))
            if self._overlays is not None:
                fs += [FieldInfo(k, f"overlay instruments: {', '.join(map(str, self._overlays.ids))}", "", "numeric", OVERLAY, 0, "D", self.name) for k in self._overlays.fields if k not in self._panel.fields]
            self._cat = Catalog(fs, self._constituents, self.name)
        return self._cat

    def load(self, request: DataRequest) -> Panel:
        if request.as_of is not None:
            warnings.warn(f"{self.name}: in-memory panels have no knowledge times; as_of={request.as_of} ignored", stacklevel=2)
        base = self._overlays if request.entity == OVERLAY else self._panel
        if base is None:
            raise UnknownFieldError(request.fields[0] if request.fields else "", f"{request.entity} field", (), self.name)
        missing = [f for f in request.fields if f not in base]
        if missing:
            raise UnknownFieldError(missing[0], "field", self.catalog().suggest(missing[0]), self.name)
        p = window(base, request.start, request.end, request.ids, request.fields)
        cat = self.catalog()
        p.lags = {f: (cat[f].lag if f in cat and cat[f].entity in (ASSET, OVERLAY) else base.lags.get(f, 0)) for f in request.fields}
        return p

    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        a = self._attrs
        if names is not None:
            missing = [n for n in names if n not in a.columns]
            if missing:
                raise UnknownFieldError(missing[0], "attribute", self.catalog().suggest(missing[0]), self.name)
            a = a[list(names)]
        return a.reindex(pd.Index(list(map(str, ids)))) if ids is not None else a

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        return _series_frame(self._series, list(names), start, end, self.name, self.catalog())

    def constituents(self, index: str, start=None, end=None) -> pd.DataFrame:
        if index not in self._constituents:
            raise UnknownFieldError(index, "index", difflib.get_close_matches(index, list(self._constituents), n=3), self.name)
        c = self._constituents[index]
        d = pd.to_datetime(c["date"])
        keep = np.ones(len(c), dtype=bool)
        if end is not None:
            keep &= (d <= _ts(end)).to_numpy()
        return c[keep]

    def environment(self, start, end, ids=None):
        if self._env is None:
            return super().environment(start, end, ids)
        return self._env(start, end, ids)

    def version(self) -> str | None:
        if self._version is None:
            p = self._panel
            h = hashlib.sha1()
            h.update(f"{self.name}|{len(p.dates)}|{len(p.ids)}|{p.dates[0] if len(p.dates) else ''}|{p.dates[-1] if len(p.dates) else ''}".encode())
            h.update(hashlib.sha1("\x1f".join(map(str, p.ids)).encode()).digest())
            for k in sorted(p.fields):
                h.update(k.encode())
                h.update(array_fingerprint(p.values[k]).encode())
            h.update(frame_fingerprint(self._attrs).encode())
            for k in sorted(self._series):
                h.update(k.encode() + frame_fingerprint(pd.Series(self._series[k])).encode())
            for k in sorted(self._constituents):
                h.update(k.encode() + frame_fingerprint(self._constituents[k]).encode())
            if self._overlays is not None:
                for k in sorted(self._overlays.fields):
                    h.update(array_fingerprint(self._overlays.values[k]).encode())
            self._version = "mem:" + h.hexdigest()[:16]
        return self._version


class FrameSource(PanelSource):
    """pandas in, source out: ``fields`` as ``{name: wide frame (dates x ids)}`` or one long frame
    (date, instrument_id, one column per field)."""

    def __init__(self, fields: Mapping[str, pd.DataFrame] | pd.DataFrame, attributes: pd.DataFrame | None = None, series=None, constituents=None, overlays: Mapping[str, pd.DataFrame] | None = None, info: Iterable[FieldInfo] | None = None, name: str = "frames", lags: Mapping[str, int] | None = None, date_col: str = "date", id_col: str = "instrument_id", version: str | None = None, calendar=None, point_in_time: bool | None = None, categories: Mapping[str, pd.DataFrame] | None = None) -> None:
        """``categories``: dated category fields as wide frames of labels (a sector or rating map that
        changes over time), coded into the panel with their labels in the catalogue."""
        info = list(info or [])
        if categories:
            if isinstance(fields, pd.DataFrame):
                raise TypeError("categories go with wide fields ({name: frame}); add the coded column to a long frame instead")
            fields = dict(fields)
            for k, frame in categories.items():
                labels = sorted({str(v) for v in pd.unique(frame.to_numpy().ravel()) if not pd.isna(v)})
                code = {lab: float(i) for i, lab in enumerate(labels)}
                fields[k] = frame.apply(lambda col, code=code: col.map(lambda x, code=code: np.nan if pd.isna(x) else code.get(str(x), np.nan)))
                info.append(FieldInfo(k, f"dated category ({len(labels)} labels)", "", "category", categories=tuple(labels), source=name))
        if isinstance(fields, pd.DataFrame):
            panel = Panel.from_long(fields, date_col=date_col, id_col=id_col)
        else:
            panel = Panel.from_wide(dict(fields), calendar=calendar)
        self.snapped_observations = int(getattr(panel, "snapped", 0))
        for k, v in (lags or {}).items():
            panel.lags[k] = int(v)
        ov = Panel.from_wide(dict(overlays)) if overlays else None
        super().__init__(panel, attributes, series, constituents, ov, info, name, version, point_in_time=point_in_time)


class SyntheticSource(PanelSource):
    """The synthetic credit universe (``synthetic.panels``) as a source: bond panels, attributes,
    the cash series, index-level series (``index.tr``, ``index.xr`` ...), the index constituents as
    ``SYNTH_IG`` and the TY / CDX_IG overlay panel. For plumbing, tests and scale only."""

    INDEX = "SYNTH_IG"

    def __init__(self, config=None, name: str = "synthetic", **config_kwargs) -> None:
        from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel

        cfg = config or SyntheticPanelConfig(**config_kwargs)
        self.config = cfg
        self.data = generate_synthetic_panel(cfg)
        d = self.data
        series = {"cash": d.cash, **{f"index.{c}": d.index[c] for c in d.index.columns}, **{f"treasury.{c}": d.treasury[c] for c in d.treasury.columns}}
        info = [field_info("signal.value", description="synthetic value signal: idiosyncratic spread plus noise (genuine but noisy skill)"), field_info("alive", description="1 while the bond exists (issued and not matured or defaulted)", kind="flag"),
                FieldInfo("cash", "overnight cash return per period", "fraction", "numeric", SERIES, 0, "D", name)]
        super().__init__(d.panel, d.attributes, series, {self.INDEX: d.membership}, d.overlays, info, name, "synthetic:" + hashlib.sha1(repr(cfg).encode()).hexdigest()[:16], self._environment, point_in_time=True)     # generated once: never revised

    def _environment(self, start, end, ids=None):
        from nautilus_fi.data.loaders.base import builder_from_frames
        from nautilus_fi.synthetic.panels import to_market_inputs

        syms = list(ids) if ids is not None else list(self.data.ids)
        frames = to_market_inputs(self.data, syms)
        return builder_from_frames(frames, pd.Timestamp(start).date(), pd.Timestamp(end).date()).build()


# ====================================================================================== callables
class CallableSource(DataSource):
    """The thinnest integration: ``fetch(field, start, end, ids)`` returns a long frame with
    columns (date, instrument_id, value | <field>) or a wide frame (index dates, columns ids).

    ``fields``: names, ``FieldInfo`` objects or ``{name: description}``. ``attributes``: a frame or
    ``f(ids, names) -> frame``; ``series``: a dict of series or ``f(name, start, end) -> series``;
    ``constituents``: a dict of frames or ``f(index, start, end) -> frame``."""

    def __init__(self, fetch: Callable, fields, attributes=None, series=None, constituents=None, name: str = "callable", version: str | Callable[[], str] | None = None, indices: Iterable[str] = (), overlay_fetch: Callable | None = None, point_in_time: bool | None = None) -> None:
        self._point_in_time = point_in_time
        self._fetch = fetch
        if isinstance(fields, Mapping):
            fields = [field_info(k, description=v) if isinstance(v, str) else v for k, v in fields.items()]
        self._fields = [f if isinstance(f, FieldInfo) else field_info(str(f), source=name) for f in fields]
        self._attrs = attributes
        self._series = series
        self._constituents = constituents
        self._overlay_fetch = overlay_fetch
        self.name = name
        self._version = version
        self._indices = tuple(indices) or (tuple(constituents) if isinstance(constituents, Mapping) else ())

    def catalog(self) -> Catalog:
        fs = list(self._fields)
        names = {f.name for f in fs}
        if isinstance(self._attrs, pd.DataFrame):
            fs += [FieldInfo(str(c), "", "", "numeric" if pd.api.types.is_numeric_dtype(self._attrs[c]) else "category", ATTRIBUTE, source=self.name) for c in self._attrs.columns if c not in names]
        if isinstance(self._series, Mapping):
            fs += [FieldInfo(k, "", "", "numeric", SERIES, source=self.name) for k in self._series if k not in names]
        return Catalog(fs, self._indices, self.name)

    def load(self, request: DataRequest) -> Panel:
        cat = self.catalog()
        fetch = self._overlay_fetch if request.entity == OVERLAY else self._fetch
        if fetch is None:
            raise UnknownFieldError(request.fields[0], f"{request.entity} field", (), self.name)
        panels = []
        for f in request.fields:
            if f not in cat:
                raise UnknownFieldError(f, "field", cat.suggest(f), self.name)
            df = fetch(f, request.start, request.end, None if request.ids is None else list(request.ids))
            panels.append(_frame_to_panel(df, f))
        p = align(panels)
        if request.ids is not None:
            p = window(p, ids=[i for i in request.ids if i in set(p.ids)])
        p = window(p, request.start, request.end)
        p.lags = {f: cat[f].lag for f in request.fields}
        return p

    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        a = self._attrs(None if ids is None else list(ids), None if names is None else list(names)) if callable(self._attrs) else self._attrs
        if a is None:
            return super().attributes(ids, names)
        a = a.copy()
        a.index = a.index.astype(str)
        if names is not None:
            a = a[[n for n in names if n in a.columns]]
        return a.reindex(pd.Index(list(map(str, ids)))) if ids is not None else a

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        names = list(names)
        if callable(self._series):
            return _series_frame({n: self._series(n, start, end) for n in names}, names, start, end, self.name)
        return _series_frame(self._series or {}, names, start, end, self.name, self.catalog())

    def constituents(self, index: str, start=None, end=None) -> pd.DataFrame:
        if callable(self._constituents):
            return self._constituents(index, start, end)
        if isinstance(self._constituents, Mapping) and index in self._constituents:
            return self._constituents[index]
        return super().constituents(index, start, end)

    def version(self) -> str | None:
        return self._version() if callable(self._version) else self._version


def _frame_to_panel(df: pd.DataFrame, name: str) -> Panel:
    """A fetched frame (long or wide) -> one-field panel."""
    if df is None or len(df) == 0:
        return Panel(pd.DatetimeIndex([]), pd.Index([], dtype=object), {name: np.zeros((0, 0))})
    cols = set(df.columns)
    if "date" in cols and ("instrument_id" in cols or "entity" in cols):
        idc = "instrument_id" if "instrument_id" in cols else "entity"
        vc = name if name in cols else "value"
        if vc not in cols:
            raise ValueError(f"fetched frame for {name!r} needs a {name!r} or 'value' column; got {sorted(cols)}")
        return Panel.from_long(df.rename(columns={vc: name}), [name], id_col=idc)
    return Panel.from_wide({name: df})


# ====================================================================================== stores
class FieldStoreSource(DataSource):
    """The bitemporal ``FieldStore`` as a source. ``mode="latest"``: the latest snapshot known at
    ``as_of``; ``"first"``: the first value ever published (what a live process saw before
    revisions). Attributes, series and constituents can be supplied alongside."""

    def __init__(self, store, attributes: pd.DataFrame | None = None, series=None, constituents=None, lags: Mapping[str, int] | None = None, mode: str = "latest", name: str = "field_store") -> None:
        from nautilus_fi.data.field_store import FieldStore

        self.store = store if isinstance(store, FieldStore) else FieldStore(store)
        self._attrs = attributes
        self._series = dict(series or {})
        self._constituents = dict(constituents or {})
        self._lags = dict(lags or {})
        self.mode = mode
        self.name = name

    def catalog(self) -> Catalog:
        fs = [field_info(f, ASSET, self._lags.get(f), self.name) for f in self.store.fields()]
        if self._attrs is not None:
            fs += [FieldInfo(str(c), "", "", "numeric" if pd.api.types.is_numeric_dtype(self._attrs[c]) else "category", ATTRIBUTE, source=self.name) for c in self._attrs.columns]
        fs += [FieldInfo(k, "", "", "numeric", SERIES, source=self.name) for k in self._series]
        return Catalog(fs, self._constituents, self.name, supports_as_of=True)

    def load(self, request: DataRequest) -> Panel:
        self.check(request)
        p = self.store.panel(request.fields, request.start, request.end, request.as_of, self.mode, request.ids)
        cat = self.catalog()
        p.lags = {f: cat[f].lag for f in request.fields}
        return p

    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        if self._attrs is None:
            return super().attributes(ids, names)
        a = self._attrs if names is None else self._attrs[list(names)]
        return a.reindex(pd.Index(list(map(str, ids)))) if ids is not None else a

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        return _series_frame(self._series, list(names), start, end, self.name, self.catalog())

    def constituents(self, index, start=None, end=None) -> pd.DataFrame:
        if index not in self._constituents:
            return super().constituents(index, start, end)
        return self._constituents[index]

    def version(self) -> str | None:
        snaps = sorted(str(p) for f in self.store.fields() for _, p in self.store.snapshots(f))
        return "fieldstore:" + hashlib.sha1(("\n".join(snaps) + self.mode).encode()).hexdigest()[:16]

    def point_in_time(self) -> bool | None:
        return self.mode == "first"                           # first publications; "latest" serves revisions


class SQLSource(DataSource):
    """Dataset mappings (``data.loaders.DatasetMapping``) on a DB-API connection as a source. The
    value columns of each mapping are fields (``analytics`` -> ``oas_bp``, ``oad``, ``oasd`` ...); a
    request fetches each mapping it needs once for the window. ``attributes``: the ``reference``
    mapping or a frame; ``series``: ``{name: (mapping, column)}`` for date-level tables."""

    KEYS = ("date", "instrument_id", "publish_ts")

    def __init__(self, loader, mappings, attributes=None, series: Mapping[str, tuple] | None = None, constituents: Mapping[str, object] | None = None, lags: Mapping[str, int] | None = None, name: str = "sql", version: str | None = None, point_in_time: bool | None = None) -> None:
        self._point_in_time = point_in_time
        self.loader = loader
        self.mappings = {m.name: m for m in mappings} if not isinstance(mappings, Mapping) else dict(mappings)
        self._attrs = attributes
        self._series = dict(series or {})
        self._constituents = dict(constituents or {})
        self._lags = dict(lags or {})
        self.name = name
        self._version = version
        self._memo: dict[tuple, pd.DataFrame] = {}
        self._field_map = {c: m.name for m in self.mappings.values() if m.name not in ("reference",) for c in m.columns if c not in self.KEYS}

    @classmethod
    def snowflake(cls, mappings, attributes=None, series=None, constituents=None, lags=None, name: str = "snowflake", **connect_kwargs) -> SQLSource:
        """Credentials from the environment (``SNOWFLAKE_*``), never from code; see ``data.loaders.snowflake``."""
        from nautilus_fi.data.loaders.snowflake import snowflake_loader

        return cls(snowflake_loader(**connect_kwargs), mappings, attributes, series, constituents, lags, name)

    def catalog(self) -> Catalog:
        fs = [field_info(c, ASSET, self._lags.get(c), self.name, description=f"{self.mappings[m].description or m} column {c}".strip()) for c, m in self._field_map.items()]
        fs += [FieldInfo(k, f"{v[0]}.{v[1]}", "", "numeric", SERIES, source=self.name) for k, v in self._series.items()]
        ref = self.mappings.get("reference")
        if ref is not None:
            fs += [FieldInfo(c, "reference data", "", "category", ATTRIBUTE, source=self.name) for c in ref.columns if c not in self.KEYS and c != "symbol"]
        return Catalog(fs, self._constituents, self.name)

    def _fetch(self, mapping: str, start, end) -> pd.DataFrame:
        key = (mapping, str(start), str(end))
        if key not in self._memo:
            self._memo[key] = self.loader.fetch(self.mappings[mapping], start, end)
        return self._memo[key]

    def load(self, request: DataRequest) -> Panel:
        cat = self.catalog()
        by_map: dict[str, list[str]] = {}
        for f in request.fields:
            if f not in self._field_map:
                raise UnknownFieldError(f, "field", cat.suggest(f), self.name)
            by_map.setdefault(self._field_map[f], []).append(f)
        panels = []
        for m, fs in by_map.items():
            df = self._fetch(m, request.start, request.end)
            if request.ids is not None:
                df = df[df["instrument_id"].astype(str).isin(set(request.ids))]
            panels.append(Panel.from_long(df, fs))
        p = align(panels)
        p.lags = {f: cat[f].lag for f in request.fields}
        return p

    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        if isinstance(self._attrs, pd.DataFrame):
            a = self._attrs
        elif "reference" in self.mappings:
            a = self._fetch("reference", None, None).set_index("symbol")
        else:
            return super().attributes(ids, names)
        a.index = a.index.astype(str)
        if names is not None:
            a = a[[n for n in names if n in a.columns]]
        return a.reindex(pd.Index(list(map(str, ids)))) if ids is not None else a

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        out = {}
        for n in names:
            if n not in self._series:
                raise UnknownFieldError(n, "series", self.catalog().suggest(n), self.name)
            m, col = self._series[n][:2]
            df = self._fetch(m, start, end)
            out[n] = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=pd.to_datetime(df["date"]).dt.normalize())
        return pd.DataFrame(out)

    def constituents(self, index, start=None, end=None) -> pd.DataFrame:
        spec = self._constituents.get(index)
        if spec is None:
            return super().constituents(index, start, end)
        if isinstance(spec, pd.DataFrame):
            return spec
        df = self._fetch(spec, start, end)
        return df[df["benchmark"] == index] if "benchmark" in df.columns else df

    def version(self) -> str | None:
        return self._version


# ====================================================================================== composition
class CompositeSource(DataSource):
    """Several sources as one: each field, series, attribute and index comes from the first source
    whose catalogue lists it; panels are aligned on the union of dates and ids."""

    def __init__(self, *sources: DataSource, name: str = "composite") -> None:
        if not sources:
            raise ValueError("CompositeSource needs at least one source")
        self.sources = sources
        self.name = name

    def catalog(self) -> Catalog:
        cat = self.sources[0].catalog()
        for s in self.sources[1:]:
            cat = cat.merge(s.catalog())
        cat.name = self.name
        return cat

    def _owner(self, name: str, entity: str | None = None) -> DataSource | None:
        for s in self.sources:
            f = s.catalog().get(name)
            if f is not None and (entity is None or f.entity == entity):
                return s
        return None

    def load(self, request: DataRequest) -> Panel:
        groups: dict[int, list[str]] = {}
        for f in request.fields:
            s = self._owner(f, OVERLAY if request.entity == OVERLAY else None)
            if s is None:
                raise UnknownFieldError(f, "field", self.catalog().suggest(f), self.name)
            groups.setdefault(self.sources.index(s), []).append(f)
        return align([self.sources[k].load(request.with_(fields=tuple(fs))) for k, fs in groups.items()])

    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        frames = []
        for s in self.sources:
            want = [n for n in (names or s.catalog().names(ATTRIBUTE)) if n in s.catalog().names(ATTRIBUTE)]
            if want:
                frames.append(s.attributes(ids, want))
        if not frames:
            return super().attributes(ids, names)
        out = frames[0]
        for f in frames[1:]:
            out = out.join(f[[c for c in f.columns if c not in out.columns]], how="outer")
        return out.reindex(pd.Index(list(map(str, ids)))) if ids is not None else out

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        cols = []
        for n in names:
            s = self._owner(n, SERIES)
            if s is None:
                raise UnknownFieldError(n, "series", self.catalog().suggest(n), self.name)
            cols.append(s.series([n], start, end))
        return pd.concat(cols, axis=1) if cols else pd.DataFrame()

    def constituents(self, index, start=None, end=None) -> pd.DataFrame:
        for s in self.sources:
            if index in s.catalog().indices:
                return s.constituents(index, start, end)
        return super().constituents(index, start, end)

    def environment(self, start, end, ids=None):
        for s in self.sources:
            try:
                return s.environment(start, end, ids)
            except NotImplementedError:
                continue
        return super().environment(start, end, ids)

    def version(self) -> str | None:
        vs = [s.version() for s in self.sources]
        return None if any(v is None for v in vs) else "composite:" + hashlib.sha1("|".join(vs).encode()).hexdigest()[:16]

    def point_in_time(self) -> bool | None:
        vs = [s.point_in_time() for s in self.sources]
        return False if any(v is False for v in vs) else (True if all(v is True for v in vs) else None)


class MappedSource(DataSource):
    """Identifier mapping: ``mapping`` maps the kernel's ids to the inner source's ids (a dict or a
    Series indexed by kernel id). Unmapped ids are dropped and reported in ``unmapped``."""

    def __init__(self, inner: DataSource, mapping: Mapping[str, str] | pd.Series, name: str | None = None) -> None:
        self.inner = inner
        m = pd.Series(mapping, dtype=object) if not isinstance(mapping, pd.Series) else mapping.astype(object)
        m.index = m.index.astype(str)
        self.to_inner = m.astype(str)
        self.to_outer = pd.Series(self.to_inner.index, index=self.to_inner.to_numpy())
        if self.to_outer.index.duplicated().any():
            raise ValueError("identifier mapping is not one-to-one")
        self.name = name or f"mapped:{inner.name}"
        self.unmapped: set[str] = set()

    def catalog(self) -> Catalog:
        return self.inner.catalog()

    def _out(self, ids: Iterable[str]) -> list[str]:
        return [self.to_outer.get(i, None) for i in ids]

    def load(self, request: DataRequest) -> Panel:
        inner_ids = None
        if request.ids is not None:
            have = [i for i in request.ids if i in self.to_inner.index]
            self.unmapped |= set(request.ids) - set(have)
            inner_ids = tuple(self.to_inner[have])
        p = self.inner.load(request.with_(ids=inner_ids))
        outer = self._out(p.ids)
        keep = np.array([o is not None for o in outer], dtype=bool)
        self.unmapped |= {str(i) for i, k in zip(p.ids, keep) if not k}
        return Panel(p.dates, pd.Index([o for o in outer if o is not None]), {k: v[:, keep] for k, v in p.values.items()}, dict(p.lags))

    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        inner_ids = None if ids is None else [self.to_inner.get(i) for i in ids if i in self.to_inner.index]
        a = self.inner.attributes(inner_ids, names)
        a.index = pd.Index([self.to_outer.get(i, i) for i in a.index.astype(str)])
        return a.reindex(pd.Index(list(map(str, ids)))) if ids is not None else a

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        return self.inner.series(names, start, end)

    def constituents(self, index, start=None, end=None) -> pd.DataFrame:
        c = self.inner.constituents(index, start, end).copy()
        c["instrument_id"] = c["instrument_id"].astype(str).map(self.to_outer)
        return c.dropna(subset=["instrument_id"])

    def version(self) -> str | None:
        v = self.inner.version()
        return None if v is None else f"{v}+map:{frame_fingerprint(self.to_inner)}"

    def point_in_time(self) -> bool | None:
        return self.inner.point_in_time()


# ====================================================================================== caching
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) + "-" + hashlib.sha1(name.encode()).hexdigest()[:6]


def save_panel(panel: Panel, root: str | Path) -> Path:
    """Write a panel as one ``.npy`` per field (memory-mappable) plus dates, ids and lags."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for k, v in panel.values.items():
        fn = _safe(k) + ".npy"
        tmp = root / f".{fn}.{uuid.uuid4().hex}.tmp.npy"
        np.save(tmp, np.ascontiguousarray(v))
        os.replace(tmp, root / fn)
        files[k] = fn
    np.save(root / "dates.npy", panel.dates.values.astype("datetime64[ns]").astype(np.int64))
    atomic_write_text(root / "meta.json", json.dumps({"ids": [str(i) for i in panel.ids], "files": files, "lags": {k: int(panel.lags.get(k, 0)) for k in panel.values}}))
    return root


def load_panel(root: str | Path, mmap: bool = True, fields: Iterable[str] | None = None) -> Panel:
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text())
    dates = pd.DatetimeIndex(np.load(root / "dates.npy").astype("datetime64[ns]"))
    names = list(fields) if fields is not None else list(meta["files"])
    vals = {k: np.load(root / meta["files"][k], mmap_mode="r" if mmap else None) for k in names}
    return Panel(dates, pd.Index(meta["ids"]), vals, {k: meta["lags"].get(k, 0) for k in names})


@dataclass
class _Block:
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    panel: Panel                                 # one field

    def covers(self, start, end) -> bool:
        lo_ok = self.start is None or (start is not None and start >= self.start)
        hi_ok = self.end is None or (end is not None and end <= self.end)
        return lo_ok and hi_ok


class CachedSource(DataSource):
    """Cache in front of any source. Each (field, ids, as-of) keeps one block covering the widest
    window requested so far; narrower requests are slices (views). With ``root`` the blocks are
    ``.npy`` files read memory-mapped, shared by every process that opens the cache.

    ``version()`` is the inner source's version or, when the inner source cannot say (a live
    warehouse), a snapshot id pinned on first use: results stay reproducible against the cached
    data until ``refresh()``."""

    def __init__(self, inner: DataSource, root: str | Path | None = None, memory_bytes: float = 8e9, name: str | None = None) -> None:
        self.inner = inner
        self.root = Path(root).expanduser() if root is not None else None
        self.memory_bytes = memory_bytes
        self.name = name or f"cached:{inner.name}"
        self._blocks: OrderedDict[tuple, _Block] = OrderedDict()
        self._cat: Catalog | None = None
        self._attr_cache: dict = {}
        self._series_cache: dict = {}
        self._const_cache: dict = {}
        self.stats = {"hits": 0, "misses": 0, "disk_hits": 0}

    # ------------------------------------------------------------------ metadata
    def catalog(self) -> Catalog:
        if self._cat is None:
            self._cat = self.inner.catalog()
        return self._cat

    def point_in_time(self) -> bool | None:
        return self.inner.point_in_time()

    def version(self) -> str | None:
        v = self.inner.version()
        if v is not None:
            return v
        if self.root is None:
            return None
        pin = self.root / "snapshot.json"
        with file_lock(self.root / ".lock"):
            if not pin.exists():
                atomic_write_text(pin, json.dumps({"snapshot": uuid.uuid4().hex[:16], "created_utc": datetime.now(UTC).isoformat(), "source": self.inner.name}))
        return "snapshot:" + json.loads(pin.read_text())["snapshot"]

    def refresh(self) -> None:
        """Forget cached data (and the pinned snapshot): the next request reads the inner source."""
        self._blocks.clear()
        self._attr_cache.clear()
        self._series_cache.clear()
        self._const_cache.clear()
        self._cat = None
        if self.root is not None and (self.root / "snapshot.json").exists():
            (self.root / "snapshot.json").unlink()

    def _dir(self, key: tuple) -> Path | None:
        if self.root is None:
            return None
        v = self.version() or "unversioned"
        field, entity, ids_key, as_of = key
        return self.root / _safe(v) / entity / _safe(field) / f"{ids_key}-{hashlib.sha1(str(as_of).encode()).hexdigest()[:8]}"

    # ------------------------------------------------------------------ blocks
    def _get_block(self, key: tuple) -> _Block | None:
        blk = self._blocks.get(key)
        if blk is not None:
            self._blocks.move_to_end(key)
            return blk
        d = self._dir(key)
        if d is not None and (d / "current.json").exists():
            cur = json.loads((d / "current.json").read_text())
            p = load_panel(d / cur["dir"], mmap=True)
            blk = _Block(_ts(cur["start"]) if cur["start"] else None, _ts(cur["end"]) if cur["end"] else None, p)
            self._blocks[key] = blk
            self.stats["disk_hits"] += 1
            return blk
        return None

    def _put_block(self, key: tuple, blk: _Block) -> None:
        d = self._dir(key)
        if d is not None:
            sub = uuid.uuid4().hex[:12]
            save_panel(blk.panel, d / sub)
            with file_lock(d / ".lock"):
                atomic_write_text(d / "current.json", json.dumps({"dir": sub, "start": None if blk.start is None else str(blk.start.date()), "end": None if blk.end is None else str(blk.end.date())}))
            blk = _Block(blk.start, blk.end, load_panel(d / sub, mmap=True))
        self._blocks[key] = blk
        self._blocks.move_to_end(key)
        total = sum(b.panel.nbytes() for b in self._blocks.values() if not any(isinstance(v, np.memmap) for v in b.panel.values.values()))
        while total > self.memory_bytes and len(self._blocks) > 1:
            _, old = self._blocks.popitem(last=False)
            total -= old.panel.nbytes()

    def load(self, request: DataRequest) -> Panel:
        keys = {f: (f, request.entity, request.ids_key, str(request.as_of)) for f in request.fields}
        have, need = {}, []
        for f, k in keys.items():
            blk = self._get_block(k)
            if blk is not None and blk.covers(request.start, request.end):
                have[f] = blk
                self.stats["hits"] += 1
            else:
                need.append((f, blk))
                self.stats["misses"] += 1
        if need:
            starts = [request.start] + [b.start for _, b in need if b is not None]
            ends = [request.end] + [b.end for _, b in need if b is not None]
            lo = None if any(s is None for s in starts) else min(starts)
            hi = None if any(e is None for e in ends) else max(ends)
            fresh = self.inner.load(request.with_(fields=tuple(f for f, _ in need), start=lo, end=hi))
            for f, _ in need:
                blk = _Block(lo, hi, Panel(fresh.dates, fresh.ids, {f: fresh.values[f]}, {f: fresh.lags.get(f, 0)}))
                self._put_block(keys[f], blk)
                have[f] = self._blocks[keys[f]]
        parts = [window(have[f].panel, request.start, request.end) for f in request.fields]
        return align(parts) if len(parts) > 1 else (parts[0] if parts else Panel(pd.DatetimeIndex([]), pd.Index([], dtype=object), {}))

    # ------------------------------------------------------------------ pass-through with memo
    def attributes(self, ids=None, names=None) -> pd.DataFrame:
        key = (None if names is None else tuple(names))
        if key not in self._attr_cache:
            self._attr_cache[key] = self.inner.attributes(None, names)
        a = self._attr_cache[key]
        return a.reindex(pd.Index(list(map(str, ids)))) if ids is not None else a

    def series(self, names, start=None, end=None) -> pd.DataFrame:
        names = list(names)
        miss = [n for n in names if n not in self._series_cache]
        if miss:
            df = self.inner.series(miss)
            for n in miss:
                self._series_cache[n] = df[n]
        df = pd.DataFrame({n: self._series_cache[n] for n in names})
        if start is not None:
            df = df[df.index >= _ts(start)]
        if end is not None:
            df = df[df.index <= _ts(end)]
        return df

    def constituents(self, index, start=None, end=None) -> pd.DataFrame:
        if index not in self._const_cache:
            self._const_cache[index] = self.inner.constituents(index)
        c = self._const_cache[index]
        if end is not None:
            c = c[pd.to_datetime(c["date"]) <= _ts(end)]
        return c

    def environment(self, start, end, ids=None):
        return self.inner.environment(start, end, ids)


def as_source(obj) -> DataSource:
    """Accept a source, a ``Panel``, a ``SyntheticPanelData`` or ``{field: wide frame}``."""
    if isinstance(obj, DataSource):
        return obj
    if isinstance(obj, Panel):
        return PanelSource(obj)
    if type(obj).__name__ == "SyntheticPanelData":
        d = obj
        series = {"cash": d.cash, **{f"index.{c}": d.index[c] for c in d.index.columns}}
        return PanelSource(d.panel, d.attributes, series, {SyntheticSource.INDEX: d.membership}, d.overlays, name="synthetic")
    if isinstance(obj, Mapping):
        return FrameSource(obj)
    raise TypeError(f"cannot use {type(obj).__name__} as a data source; wrap it in a DataSource")


__all__ = [
    "ASSET",
    "ATTRIBUTE",
    "OVERLAY",
    "SERIES",
    "CachedSource",
    "CallableSource",
    "Catalog",
    "CompositeSource",
    "DataQualityError",
    "DataRequest",
    "DataSource",
    "FieldInfo",
    "FieldStoreSource",
    "FrameSource",
    "MappedSource",
    "MissingDataError",
    "PanelSource",
    "SQLSource",
    "SyntheticSource",
    "UnknownFieldError",
    "align",
    "array_fingerprint",
    "as_source",
    "atomic_write_text",
    "field_info",
    "fields_from_dictionary",
    "file_lock",
    "frame_fingerprint",
    "load_panel",
    "save_panel",
    "window",
]
