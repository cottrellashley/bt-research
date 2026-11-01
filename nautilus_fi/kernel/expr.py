"""Expressions: fields, signals and universe rules as Python objects, evaluated over all dates at once.

``Field("oas_bp").zscore(by="sector").neutralise("oasd")`` is a small graph, not a computation.
The engine evaluates it once over the whole (dates x instruments) panel with numpy - no per-date
pandas - caches every node by its structural key, and hands each rebalance its row. The same
objects describe the universe (``(Field("rating_score") <= 10) & (Field("oasd") > 1)``) and the
inputs of construction methods and optimiser terms.

Point in time by construction: ``Field`` applies the source's availability lag when it reads, so
row ``t`` of every expression holds only what was known at the close of ``t``. Time-series
transforms look back (``lag``, ``diff``, ``rolling``, ``ewm``); a negative lag (``lead``) reads the
future and is refused for signals (``LookaheadError``). Custom Python (``apply``, ``cs``) cannot be
analysed structurally, so ``check_lookahead`` re-evaluates the expression on truncated panels and
compares: a value that changes when later data are removed used later data.

Cross-sectional transforms (``rank``, ``zscore``, ``demean``, ``winsor``, ``neutralise``, ``bucket``,
``top``) work per date within the strategy's universe (and within groups given by ``by=``: an
attribute such as ``"sector"``, a list of them, or a category expression such as
``Field("rating_score").bucket([4, 7, 10])``).

Comparison operators build predicates (``Field("x") > 1``, ``Attribute("sector") == "FIN"``); use
``&``, ``|`` and ``~`` to combine them. Expressions are immutable and hash by structure; compare two
with ``a.same_as(b)``.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nautilus_fi.data.panel import Panel
from nautilus_fi.kernel.data import MissingDataError, UnknownFieldError


class LookaheadError(ValueError):
    """An expression reads data that were not known at the decision time."""


# ====================================================================================== helpers
def callable_fingerprint(fn: Callable) -> dict:
    """Identity of user code for hashing: qualified name, source (or bytecode) digest, closure and
    default values. Editing the function changes the fingerprint, so caches never go stale."""
    name = getattr(fn, "__qualname__", None) or type(fn).__name__
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        src = (code.co_code.hex() + repr(code.co_consts)) if code is not None else repr(fn)
    h = hashlib.sha256(src.encode())
    closure = getattr(fn, "__closure__", None) or ()
    for cell in closure:
        try:
            h.update(_value_digest(cell.cell_contents).encode())
        except ValueError:
            h.update(b"<empty>")
    for d in getattr(fn, "__defaults__", None) or ():
        h.update(_value_digest(d).encode())
    return {"callable": name, "module": getattr(fn, "__module__", ""), "sha": h.hexdigest()[:16]}


def _value_digest(v) -> str:
    if isinstance(v, np.ndarray):
        return hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()[:12]
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return hashlib.sha1(pd.util.hash_pandas_object(v, index=True).to_numpy().tobytes()).hexdigest()[:12]
    if isinstance(v, Expr):
        return v.key
    if callable(v):
        return callable_fingerprint(v)["sha"]
    return repr(v)[:200]


def _fmt(v) -> str:
    if isinstance(v, Expr):
        return repr(v)
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return repr(v)


def shift_rows(v: np.ndarray, n: int, fill=np.nan) -> np.ndarray:
    if n == 0:
        return v
    out = np.full(v.shape, fill, dtype=v.dtype if v.dtype.kind in "fb" else np.float64)
    if n > 0:
        out[n:] = v[:-n]
    else:
        out[:n] = v[-n:]
    return out


def row_rank(x: np.ndarray, method: str = "average") -> np.ndarray:
    """Ranks 1..n per row (NaN stays NaN): ``average`` for ties, or ``first`` (ordinal)."""
    T, N = x.shape
    nan = np.isnan(x)
    xs = np.where(nan, np.inf, x)
    order = np.argsort(xs, axis=1, kind="stable")
    if method == "first":
        r = np.empty((T, N))
        np.put_along_axis(r, order, np.broadcast_to(np.arange(1, N + 1, dtype=float), (T, N)), axis=1)
    else:
        sx = np.take_along_axis(xs, order, axis=1)
        pos = np.broadcast_to(np.arange(N), (T, N))
        new = np.ones((T, N), dtype=bool)
        new[:, 1:] = sx[:, 1:] != sx[:, :-1]
        start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
        last = np.ones((T, N), dtype=bool)
        last[:, :-1] = new[:, 1:]
        end = np.minimum.accumulate(np.where(last, pos, N - 1)[:, ::-1], axis=1)[:, ::-1]
        r = np.empty((T, N))
        np.put_along_axis(r, order, (start + end) / 2.0 + 1.0, axis=1)
    r[nan] = np.nan
    return r


def group_stats(x: np.ndarray, valid: np.ndarray, codes: np.ndarray, G: int):
    """Per (row, group) count, mean and sum of squared deviations of ``x`` over ``valid`` cells;
    ``codes`` (N,) or (T, N), -1 = no group. Returns (cnt, mean, ssd, valid, flat)."""
    T, N = x.shape
    c = np.broadcast_to(codes, (T, N))
    valid = valid & (c >= 0)
    flat = (np.arange(T, dtype=np.int64)[:, None] * G + c)[valid]
    xv = x[valid]
    cnt = np.bincount(flat, minlength=T * G).reshape(T, G)
    s1 = np.bincount(flat, weights=xv, minlength=T * G).reshape(T, G)
    mean = s1 / np.maximum(cnt, 1)
    dev = xv - mean.reshape(-1)[flat]
    ssd = np.bincount(flat, weights=dev * dev, minlength=T * G).reshape(T, G)
    return cnt, mean, ssd, valid, flat


@dataclass
class Cat:
    """A categorical value: integer codes ((N,) or (T, N); -1 = missing) and their labels."""

    codes: np.ndarray
    categories: list

    def code_of(self, value) -> int:
        try:
            return self.categories.index(value)
        except ValueError:
            s = str(value)
            for i, c in enumerate(self.categories):
                if str(c) == s:
                    return i
            return -2                                   # matches nothing (not even missing)

    def broadcast(self, T: int, N: int) -> np.ndarray:
        return np.broadcast_to(self.codes, (T, N))


def factorize(values: Sequence) -> Cat:
    s = pd.Series(list(values), dtype=object)
    codes, uniques = pd.factorize(s, sort=True, use_na_sentinel=True)
    return Cat(codes.astype(np.int32), list(uniques))


def combine_codes(cats: list[Cat], T: int, N: int) -> Cat:
    if len(cats) == 1:
        return cats[0]
    codes = np.zeros((T, N), dtype=np.int64)
    missing = np.zeros((T, N), dtype=bool)
    for c in cats:
        cc = c.broadcast(T, N)
        missing |= cc < 0
        codes = codes * max(len(c.categories), 1) + np.maximum(cc, 0)
    codes = np.where(missing, -1, codes)
    uniq, inv = np.unique(codes[~missing], return_inverse=True)
    out = np.full((T, N), -1, dtype=np.int32)
    out[~missing] = inv
    labels = []
    for u in uniq:
        parts = []
        for c in reversed(cats):
            k = max(len(c.categories), 1)
            parts.append(str(c.categories[int(u % k)]) if c.categories else "")
            u //= k
        labels.append("|".join(reversed(parts)))
    static = all(c.codes.ndim == 1 for c in cats)
    return Cat(out[0] if static else out, labels)


# ====================================================================================== context
class ExprCache:
    """Evaluated nodes by key, bounded by bytes (least recently used dropped first)."""

    def __init__(self, max_bytes: float = 6e9) -> None:
        self.max_bytes = max_bytes
        self._d: OrderedDict[str, object] = OrderedDict()
        self._bytes = 0
        self._pins: list = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _size(v) -> int:
        if isinstance(v, np.ndarray):
            return 0 if v.base is not None and v.strides and 0 in v.strides else v.nbytes
        if isinstance(v, Cat):
            return v.codes.nbytes
        return 0

    def get(self, key: str):
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
            self.hits += 1
        else:
            self.misses += 1
        return v

    def put(self, key: str, value) -> None:
        if isinstance(value, np.ndarray) and value.flags.writeable and value.base is None:
            value.setflags(write=False)
        self._d[key] = value
        self._bytes += self._size(value)
        while self._bytes > self.max_bytes and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self._bytes -= self._size(old)

    def pin(self, obj) -> None:
        """Keep ``obj`` alive for the cache's lifetime (panel ids are part of the keys)."""
        self._pins.append(obj)

    def __len__(self) -> int:
        return len(self._d)


class EvalContext:
    """What expressions evaluate against: a panel (point-in-time rows), static attributes, date
    series, the source (for index constituents), an optional universe mask for cross-sectional
    transforms, optional rows of interest, and a shared cache."""

    def __init__(self, panel: Panel, attributes: pd.DataFrame | None = None, series: pd.DataFrame | None = None, source=None, catalog=None, universe: np.ndarray | None = None, universe_key: str = "", rows: np.ndarray | None = None, cache: ExprCache | None = None, return_field: str = "tr", lags: dict | None = None) -> None:
        self.panel = panel
        self.T, self.N = len(panel.dates), len(panel.ids)
        self.attributes = attributes.reindex(panel.ids) if attributes is not None else pd.DataFrame(index=panel.ids)
        self.series = series.reindex(panel.dates) if series is not None else pd.DataFrame(index=panel.dates)
        self.source = source
        self.catalog = catalog
        self.universe = universe
        self.universe_key = universe_key
        self.rows = rows
        self.cache = cache if cache is not None else ExprCache()
        self.cache.pin(panel)
        self.return_field = return_field
        self.lags = dict(lags or {})
        self.token = f"p{id(panel)}:{self.T}x{self.N}"
        self.notes: list[str] = []

    # ------------------------------------------------------------------ derived contexts
    def _clone(self, **kw) -> EvalContext:
        c = object.__new__(EvalContext)
        c.__dict__.update(self.__dict__)
        c.__dict__.update(kw)
        return c

    def within(self, universe: np.ndarray | None, key: str) -> EvalContext:
        return self._clone(universe=universe, universe_key=key)

    def full(self) -> EvalContext:
        return self if self.rows is None else self._clone(rows=None)

    def at_rows(self, rows: np.ndarray | None) -> EvalContext:
        return self._clone(rows=None if rows is None else np.asarray(rows, dtype=np.int64))

    @property
    def rows_key(self) -> str:
        return "" if self.rows is None else hashlib.sha1(np.asarray(self.rows, dtype=np.int64).tobytes()).hexdigest()[:10]

    # ------------------------------------------------------------------ data
    def label_of(self, name: str) -> str:
        """``"start"`` for a field whose values are stamped with the start of the period they cover."""
        if self.catalog is not None and name in self.catalog:
            return getattr(self.catalog[name], "label", "end")
        return "end"

    def lag_of(self, name: str) -> int:
        """Rows after its stamp a value may inform a decision: the availability lag, plus one for a
        start-labelled value (it covers the period after its stamp, known only at that period's end)."""
        extra = 1 if self.label_of(name) == "start" else 0
        if name in self.lags:
            return int(self.lags[name]) + extra
        if self.catalog is not None and name in self.catalog:
            return int(self.catalog[name].lag) + extra
        return int(self.panel.lags.get(name, 0)) + extra

    def has(self, name: str) -> bool:
        return name in self.panel or name in self.attributes.columns or name in self.series.columns

    def unknown(self, name: str, kind: str = "field") -> UnknownFieldError:
        names = list(self.panel.fields) + list(self.attributes.columns) + list(self.series.columns)
        sugg = self.catalog.suggest(name) if self.catalog is not None else []
        if not sugg:
            import difflib

            sugg = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        return UnknownFieldError(name, kind, sugg, getattr(self.source, "name", ""))

    def field(self, name: str, lag: int | None = None) -> np.ndarray:
        if name not in self.panel:
            raise self.unknown(name)
        n = self.lag_of(name) if lag is None else int(lag)
        key = f"{self.token}|field:{name}|lag:{n}"          # one entry per effective lag (realised and known reads share it at lag 0)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        raw = self.panel[name]
        v = raw if raw.dtype == np.float64 else raw.astype(np.float64)
        out = shift_rows(np.asarray(v), n) if n else np.asarray(v)
        self.cache.put(key, out)
        return out

    def form_of(self, name: str) -> str:
        if self.catalog is not None and name in self.catalog:
            return getattr(self.catalog[name], "form", "simple")
        return "simple"

    def realised(self, name: str) -> np.ndarray:
        """``name`` as realised, without the availability lag: row ``t`` is the value for the period
        ending on date ``t`` (a start-labelled field moves one row later), as a simple return per
        period whatever form it was declared in (log returns, price levels or additive cumulative
        returns are converted). Returns, the legs of excess returns and refills are outcomes, read
        this way; the lag governs only what a decision may see (``field``)."""
        v = self.field(name, 1 if self.label_of(name) == "start" else 0)
        form = self.form_of(name)
        if form == "simple":
            return v
        key = f"{self.token}|realised:{name}|{form}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if form == "log":
            out = np.expm1(v)
        elif form == "price":
            out = v / shift_rows(v, 1) - 1.0
        else:
            out = v - shift_rows(v, 1)
        self.cache.put(key, out)
        return out

    def attribute(self, name: str) -> pd.Series:
        if name not in self.attributes.columns:
            raise self.unknown(name, "attribute")
        return self.attributes[name]

    def series_values(self, name: str, lag: int | None = None) -> np.ndarray:
        if name not in self.series.columns:
            raise self.unknown(name, "series")
        v = self.series[name].to_numpy(dtype=np.float64)
        n = self.lag_of(name) if lag is None else int(lag)
        return shift_rows(v[:, None], n) if n else v[:, None]

    def groups(self, by) -> Cat:
        """Group codes for ``by``: attribute name, category field, list of either, or a category
        expression."""
        if by is None:
            return Cat(np.zeros(self.N, dtype=np.int32), ["ALL"])
        if isinstance(by, (list, tuple)):
            return combine_codes([self.groups(b) for b in by], self.T, self.N)
        if isinstance(by, Expr):
            v = by.evaluate(self.full())
            if isinstance(v, Cat):
                return v
            return _numeric_to_cat(v)
        key = f"{self.token}|groups:{by}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if by in self.attributes.columns:
            out = factorize(self.attributes[by].tolist())
        elif by in self.panel:
            out = _numeric_to_cat(self.field(by))                            # dated categories: point in time, as known
            labels = getattr(self.catalog[by], "categories", ()) if (self.catalog is not None and by in self.catalog) else ()
            if labels:
                out.categories = [labels[int(c)] if 0 <= int(c) < len(labels) else str(c) for c in out.categories]
        else:
            raise self.unknown(by, "group attribute")
        self.cache.put(key, out)
        return out

    def mask(self) -> np.ndarray | None:
        return self.universe


def _numeric_to_cat(v: np.ndarray) -> Cat:
    fin = np.isfinite(v)
    uniq = np.unique(v[fin]) if fin.any() else np.zeros(0)
    if len(uniq) > 512:
        raise ValueError(f"grouping by a numeric field with {len(uniq)} distinct values; bucket it first (.bucket([...]) or .bucket(n=5))")
    codes = np.full(v.shape, -1, dtype=np.int32)
    codes[fin] = np.searchsorted(uniq, v[fin]).astype(np.int32)
    return Cat(codes, [float(u) for u in uniq])


# ====================================================================================== base
_STATIC = -(10**9)                         # look-ahead of something that does not vary in time


class Expr:
    """Base of every expression. Subclasses implement ``_eval(ctx)`` and set ``_children``."""

    kind = "numeric"                      # numeric | bool | category
    cross_sectional = False               # depends on the universe mask
    row_local = False                     # custom per-row Python: honours ctx.rows
    _children: tuple = ()

    # ------------------------------------------------------------------ identity
    def _params(self) -> list:
        return []

    def _label(self) -> str:
        return type(self).__name__

    @property
    def key(self) -> str:
        k = self.__dict__.get("_key")
        if k is None:
            k = self._make_key()
            self.__dict__["_key"] = k
        return k

    def _make_key(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        args = ", ".join(_fmt(p) for p in self._params())
        return f"{self._label()}({args})"

    def __hash__(self) -> int:
        return hash(self.key)

    def same_as(self, other) -> bool:
        return isinstance(other, Expr) and other.key == self.key

    def __bool__(self) -> bool:
        raise TypeError("an expression has no truth value: combine predicates with & | ~ and compare expressions with .same_as()")

    # ------------------------------------------------------------------ analysis
    def children(self) -> tuple:
        return self._children

    def walk(self):
        yield self
        for c in self._children:
            yield from c.walk()

    def names(self) -> set[str]:
        """Every data name referenced (fields, attributes, series)."""
        out: set[str] = set()
        for e in self.walk():
            out |= e._own_names()
        return out

    def _own_names(self) -> set[str]:
        return set()

    def indices(self) -> set[str]:
        out: set[str] = set()
        for e in self.walk():
            out |= e._own_indices()
        return out

    def _own_indices(self) -> set[str]:
        return set()

    def lookback(self) -> int:
        """Rows of history needed before the first row with a value."""
        return self._own_lookback() + max((c.lookback() for c in self._children), default=0)

    def _own_lookback(self) -> int:
        return 0

    def lookahead(self) -> int | None:
        """Rows of future data used (> 0 is look-ahead); None when it cannot be analysed."""
        sub = [c.lookahead() for c in self._children]
        if any(s is None for s in sub):
            return None
        own = self._own_lead()
        if own is None:
            return None
        if self._shifts():
            return own + max(sub, default=_STATIC)
        return max([*sub, own])

    def _own_lead(self) -> int | None:
        return 0 if not self._children else _STATIC

    def _shifts(self) -> bool:
        return False

    @property
    def opaque(self) -> bool:
        return any(isinstance(e, (Apply, CrossSectional)) and not getattr(e, "causal", False) for e in self.walk())

    def callables(self) -> list[dict]:
        out = []
        for e in self.walk():
            fn = getattr(e, "fn", None)
            if fn is not None:
                out.append(callable_fingerprint(fn))
        return out

    def _sensitive(self) -> tuple[bool, bool]:
        s = self.__dict__.get("_sens")
        if s is None:
            u = any(e.cross_sectional for e in self.walk())
            r = any(e.row_local for e in self.walk())
            s = (u, r)
            self.__dict__["_sens"] = s
        return s

    # ------------------------------------------------------------------ evaluation
    def _eval(self, ctx: EvalContext):
        raise NotImplementedError

    def evaluate(self, ctx: EvalContext):
        u, r = self._sensitive()
        key = f"{ctx.token}|{self.key}" + (f"|u:{ctx.universe_key}" if u else "") + (f"|r:{ctx.rows_key}" if r else "")
        hit = ctx.cache.get(key)
        if hit is not None:
            return hit
        out = self._eval(ctx)
        if isinstance(out, np.ndarray) and out.shape != (ctx.T, ctx.N):
            out = np.broadcast_to(out, (ctx.T, ctx.N))
        ctx.cache.put(key, out)
        return out

    def frame(self, ctx: EvalContext) -> pd.DataFrame:
        v = self.evaluate(ctx)
        if isinstance(v, Cat):
            labels = np.array([*v.categories, None], dtype=object)
            return pd.DataFrame(labels[np.where(v.broadcast(ctx.T, ctx.N) < 0, len(v.categories), v.broadcast(ctx.T, ctx.N))], index=ctx.panel.dates, columns=ctx.panel.ids)
        return pd.DataFrame(np.asarray(v), index=ctx.panel.dates, columns=ctx.panel.ids)

    # ------------------------------------------------------------------ arithmetic
    def __add__(self, o):
        return BinOp("+", self, o)

    def __radd__(self, o):
        return BinOp("+", o, self)

    def __sub__(self, o):
        return BinOp("-", self, o)

    def __rsub__(self, o):
        return BinOp("-", o, self)

    def __mul__(self, o):
        return BinOp("*", self, o)

    def __rmul__(self, o):
        return BinOp("*", o, self)

    def __truediv__(self, o):
        return BinOp("/", self, o)

    def __rtruediv__(self, o):
        return BinOp("/", o, self)

    def __pow__(self, o):
        return BinOp("**", self, o)

    def __neg__(self):
        return Unary("neg", self)

    def __abs__(self):
        return Unary("abs", self)

    # ------------------------------------------------------------------ predicates
    def __lt__(self, o):
        return Compare("<", self, o)

    def __le__(self, o):
        return Compare("<=", self, o)

    def __gt__(self, o):
        return Compare(">", self, o)

    def __ge__(self, o):
        return Compare(">=", self, o)

    def __eq__(self, o):                                 # type: ignore[override]
        return Compare("==", self, o)

    def __ne__(self, o):                                 # type: ignore[override]
        return Compare("!=", self, o)

    def __and__(self, o):
        return Logical("&", self, o)

    def __rand__(self, o):
        return Logical("&", o, self)

    def __or__(self, o):
        return Logical("|", self, o)

    def __ror__(self, o):
        return Logical("|", o, self)

    def __invert__(self):
        return Not(self)

    def isin(self, values) -> Expr:
        return IsIn(self, tuple(values))

    def between(self, lower, upper) -> Expr:
        return (self >= lower) & (self <= upper)

    def notna(self) -> Expr:
        return Unary("notna", self)

    def isna(self) -> Expr:
        return Unary("isna", self)

    # ------------------------------------------------------------------ elementwise
    def abs(self) -> Expr:
        return Unary("abs", self)

    def log(self) -> Expr:
        return Unary("log", self)

    def log1p(self) -> Expr:
        return Unary("log1p", self)

    def exp(self) -> Expr:
        return Unary("exp", self)

    def sqrt(self) -> Expr:
        return Unary("sqrt", self)

    def sign(self) -> Expr:
        return Unary("sign", self)

    def clip(self, lower=None, upper=None) -> Expr:
        return Clip(self, lower, upper)

    def fillna(self, value) -> Expr:
        return FillNA(self, value)

    def where(self, cond, other=np.nan) -> Expr:
        return Where(cond, self, other)

    def named(self, name: str) -> Expr:
        return Named(self, name)

    # ------------------------------------------------------------------ time series (look back only)
    def lag(self, n: int = 1) -> Expr:
        return Shift(self, int(n))

    def lead(self, n: int = 1) -> Expr:
        """The value ``n`` rows later: look-ahead, only for labels (forward returns), never signals."""
        return Shift(self, -int(n))

    def diff(self, n: int = 1) -> Expr:
        return Diff(self, int(n))

    def pct_change(self, n: int = 1) -> Expr:
        return PctChange(self, int(n))

    def rolling(self, window: int, min_periods: int | None = None) -> _RollingBuilder:
        return _RollingBuilder(self, int(window), min_periods)

    def ewm(self, halflife: float, min_periods: int = 1) -> _EwmBuilder:
        return _EwmBuilder(self, float(halflife), int(min_periods))

    def ffill(self, limit: int | None = None, days: int | None = None) -> Expr:
        """Carry the last value forward: at most ``limit`` rows, and/or at most ``days`` calendar days
        old (MRI v2 carries inputs 5 days, Erin 7); unbounded otherwise (all history is loaded)."""
        return FFill(self, limit, days)

    def cumsum(self) -> Expr:
        return CumSum(self)

    # ------------------------------------------------------------------ cross-sectional (per date, within the universe)
    def rank(self, by=None, ascending: bool = True) -> Expr:
        return Rank(self, _by(by), ascending)

    def zscore(self, by=None, min_group: int = 3) -> Expr:
        return ZScore(self, _by(by), int(min_group))

    def demean(self, by=None) -> Expr:
        return Demean(self, _by(by))

    def cs_quantile(self, q: float) -> Expr:
        """The cross-sectional ``q`` quantile per date within the universe, broadcast."""
        return CSQuantile(self, q)

    def winsor(self, lower: float = 0.01, upper: float | None = None) -> Expr:
        return Winsor(self, float(lower), float(1.0 - lower if upper is None else upper))

    def neutralise(self, *exposures, by=None) -> Expr:
        return Neutralise(self, tuple(e if isinstance(e, Expr) else Field(e) for e in exposures), _by(by))

    neutralize = neutralise

    def bucket(self, edges: Sequence[float] | None = None, n: int | None = None, by=None) -> Expr:
        return Bucket(self, None if edges is None else tuple(float(e) for e in edges), n, _by(by))

    def top(self, n: int, by=None) -> Expr:
        return Top(self, int(n), _by(by), False)

    def bottom(self, n: int, by=None) -> Expr:
        return Top(self, int(n), _by(by), True)

    def scale(self, total: float = 1.0) -> Expr:
        return Scale(self, float(total))

    # ------------------------------------------------------------------ custom Python
    def apply(self, fn: Callable, name: str | None = None, lookback: int = 0, causal: bool = False) -> Expr:
        return Apply(fn, (self,), name, lookback, causal)

    def cs(self, fn: Callable, name: str | None = None) -> Expr:
        return CrossSectional(fn, (self,), name)


def _by(by):
    if by is None or isinstance(by, Expr):
        return by
    if isinstance(by, (list, tuple)):
        return tuple(b if isinstance(b, Expr) else str(b) for b in by)
    return str(by)


def lift(v) -> Expr:
    if isinstance(v, Expr):
        return v
    if isinstance(v, (int, float, np.floating, np.integer, bool)):
        return Const(float(v))
    if isinstance(v, str):
        return Const(v)
    raise TypeError(f"cannot use {type(v).__name__} in an expression")


# ====================================================================================== leaves
class Field(Expr):
    """A panel field, attribute or date series by name, as known at each date (availability lag
    applied). ``lag=`` overrides the source's lag (it may only be larger: reading later than
    published is allowed, earlier is look-ahead)."""

    def __init__(self, name: str, lag: int | None = None) -> None:
        self.name = str(name)
        self.lag_days = None if lag is None else int(lag)

    def _params(self):
        return [self.name] + ([] if self.lag_days is None else [self.lag_days])

    def _own_names(self):
        return {self.name}

    def _own_lookback(self) -> int:
        return self.lag_days or 0

    def _eval(self, ctx):
        if self.name in ctx.panel:
            min_lag = ctx.lag_of(self.name)
            if self.lag_days is not None and self.lag_days < min_lag:
                raise LookaheadError(f"Field({self.name!r}, lag={self.lag_days}) reads it {min_lag - self.lag_days} day(s) before it is published (availability lag {min_lag})")
            return ctx.field(self.name, self.lag_days)
        if self.name in ctx.attributes.columns:
            col = ctx.attributes[self.name]
            if pd.api.types.is_numeric_dtype(col):
                return np.broadcast_to(col.to_numpy(dtype=np.float64), (ctx.T, ctx.N))
            return ctx.groups(self.name)
        if self.name in ctx.series.columns:
            return np.broadcast_to(ctx.series_values(self.name, self.lag_days), (ctx.T, ctx.N))
        raise ctx.unknown(self.name)


class Attribute(Field):
    """A static per-instrument attribute (issuer, sector, currency ...)."""

    def _eval(self, ctx):
        if self.name not in ctx.attributes.columns:
            raise ctx.unknown(self.name, "attribute")
        return super()._eval(ctx)


class TimeSeries(Field):
    """A date series (cash rate, index return, VIX) broadcast across instruments."""

    def _eval(self, ctx):
        if self.name not in ctx.series.columns:
            raise ctx.unknown(self.name, "series")
        return np.broadcast_to(ctx.series_values(self.name, self.lag_days), (ctx.T, ctx.N))


class Const(Expr):
    def __init__(self, value) -> None:
        self.value = value

    def _params(self):
        return [self.value]

    def _own_lead(self):
        return _STATIC

    def _eval(self, ctx):
        if isinstance(self.value, str):
            raise TypeError(f"a string constant {self.value!r} only makes sense in a comparison with a category")
        return np.full((ctx.T, ctx.N), float(self.value))

    def __repr__(self) -> str:
        return _fmt(self.value)


class All(Expr):
    """Every instrument that exists on the date (the ``alive`` field when the source has one,
    else from the first to the last finite return)."""

    kind = "bool"

    def _own_names(self):
        return set()

    def _eval(self, ctx):
        return alive_mask(ctx)

    def __repr__(self) -> str:
        return "All()"


class Available(Expr):
    """True where ``name`` has a published history: within its catalogue window (``FieldInfo.start``
    to ``end``) and, for a panel field, from its first value on. Guard a rule on a field that starts
    later than the run (ESG, analyst views): ``~Available("esg") | (Field("esg") > 3)``; reading such
    a field unguarded before its start is refused."""

    kind = "bool"

    def __init__(self, name: str) -> None:
        self.name = str(name)

    def _params(self):
        return [self.name]

    def _own_names(self):
        return {self.name}

    def _eval(self, ctx):
        T, N = ctx.T, ctx.N
        ok = np.ones((T, N), dtype=bool)
        info = ctx.catalog.get(self.name) if ctx.catalog is not None else None
        dates = ctx.panel.dates
        if info is not None and info.start:
            ok &= np.asarray(dates >= pd.Timestamp(info.start))[:, None]
        if info is not None and info.end:
            ok &= np.asarray(dates <= pd.Timestamp(info.end))[:, None]
        if self.name in ctx.panel:
            seen = np.maximum.accumulate(np.isfinite(ctx.field(self.name)), axis=0)
            ok &= seen
        return ok

    def __repr__(self) -> str:
        return f"Available({self.name!r})"


def alive_mask(ctx: EvalContext) -> np.ndarray:
    key = f"{ctx.token}|alive|{ctx.return_field}"
    hit = ctx.cache.get(key)
    if hit is not None:
        return hit
    if "alive" in ctx.panel:
        out = np.nan_to_num(np.asarray(ctx.panel["alive"], dtype=np.float64)) > 0
    elif ctx.return_field in ctx.panel:
        fin = np.isfinite(np.asarray(ctx.realised(ctx.return_field), dtype=np.float64))
        T = ctx.T
        first = np.where(fin.any(axis=0), fin.argmax(axis=0), T)
        last = np.where(fin.any(axis=0), T - 1 - fin[::-1].argmax(axis=0), -1)
        tt = np.arange(T)[:, None]
        out = (tt >= first[None, :] - 1) & (tt <= last[None, :])
    else:
        raise MissingDataError(f"cannot tell which instruments exist: the source has neither an 'alive' field nor the return field {ctx.return_field!r}")
    ctx.cache.put(key, out)
    return out


# ====================================================================================== elementwise
_BIN = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "**": np.power}


class BinOp(Expr):
    def __init__(self, op: str, a, b) -> None:
        self.op, self.a, self.b = op, lift(a), lift(b)
        self._children = (self.a, self.b)

    def __repr__(self) -> str:
        return f"({self.a!r} {self.op} {self.b!r})"

    def _eval(self, ctx):
        a, b = self.a.evaluate(ctx), self.b.evaluate(ctx)
        if isinstance(a, Cat) or isinstance(b, Cat):
            raise TypeError(f"arithmetic on a category in {self!r}")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = _BIN[self.op](np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        if self.op == "/":
            out = np.where(np.isfinite(out), out, np.nan)
        return out


_UN = {"neg": np.negative, "abs": np.abs, "log": np.log, "log1p": np.log1p, "exp": np.exp, "sqrt": np.sqrt, "sign": np.sign}


class Unary(Expr):
    def __init__(self, op: str, a) -> None:
        self.op, self.a = op, lift(a)
        self._children = (self.a,)
        if op in ("notna", "isna"):
            self.kind = "bool"

    def __repr__(self) -> str:
        return f"-{self.a!r}" if self.op == "neg" else f"{self.a!r}.{self.op}()"

    def _eval(self, ctx):
        a = self.a.evaluate(ctx)
        if self.op in ("notna", "isna"):
            ok = (a.broadcast(ctx.T, ctx.N) >= 0) if isinstance(a, Cat) else (np.asarray(a, dtype=bool) if self.a.kind == "bool" else np.isfinite(a))
            return ok if self.op == "notna" else ~ok
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = _UN[self.op](np.asarray(a, dtype=np.float64))
        return np.where(np.isfinite(out), out, np.nan) if self.op in ("log", "sqrt", "log1p") else out


class Clip(Expr):
    def __init__(self, a, lower, upper) -> None:
        self.a, self.lower, self.upper = lift(a), lower, upper
        self._children = (self.a,) + tuple(x for x in (lower, upper) if isinstance(x, Expr))

    def _params(self):
        return [self.a, self.lower, self.upper]

    def _label(self):
        return "clip"

    def _eval(self, ctx):
        a = self.a.evaluate(ctx)
        lo = self.lower.evaluate(ctx) if isinstance(self.lower, Expr) else (-np.inf if self.lower is None else self.lower)
        hi = self.upper.evaluate(ctx) if isinstance(self.upper, Expr) else (np.inf if self.upper is None else self.upper)
        return np.clip(a, lo, hi)


class FillNA(Expr):
    def __init__(self, a, value) -> None:
        self.a, self.value = lift(a), value
        self._children = (self.a,) + ((value,) if isinstance(value, Expr) else ())

    def _params(self):
        return [self.a, self.value]

    def _label(self):
        return "fillna"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        v = self.value.evaluate(ctx) if isinstance(self.value, Expr) else float(self.value)
        return np.where(np.isnan(a), v, a)


class Where(Expr):
    def __init__(self, cond, a, b=np.nan) -> None:
        self.cond, self.a, self.b = lift(cond), lift(a), lift(b) if not (isinstance(b, float) and math.isnan(b)) else Const(np.nan)
        self._children = (self.cond, self.a, self.b)

    def _params(self):
        return [self.cond, self.a, self.b]

    def _label(self):
        return "where"

    def _eval(self, ctx):
        c = np.asarray(self.cond.evaluate(ctx), dtype=bool)
        return np.where(c, self.a.evaluate(ctx), self.b.evaluate(ctx))


class Named(Expr):
    """A label for reports and the trial ledger; evaluation is the wrapped expression's."""

    def __init__(self, a: Expr, name: str) -> None:
        self.a, self.name = a, name
        self._children = (a,)
        self.kind = a.kind

    def __repr__(self) -> str:
        return f"{self.a!r}.named({self.name!r})"

    def _eval(self, ctx):
        return self.a.evaluate(ctx)


# ====================================================================================== predicates
_CMP = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal, "!=": np.not_equal}


class Compare(Expr):
    kind = "bool"

    def __init__(self, op: str, a, b) -> None:
        self.op, self.a, self.b = op, lift(a), lift(b)
        self._children = (self.a, self.b)

    def __repr__(self) -> str:
        return f"({self.a!r} {self.op} {self.b!r})"

    def _eval(self, ctx):
        a = self.a.evaluate(ctx)
        if isinstance(a, Cat):
            if self.op not in ("==", "!="):
                raise TypeError(f"{self!r}: categories only support == and != (or .isin())")
            if not isinstance(self.b, Const):
                raise TypeError(f"{self!r}: compare a category with a constant")
            code = a.code_of(self.b.value)
            eq = a.broadcast(ctx.T, ctx.N) == code
            return eq if self.op == "==" else ~eq & (a.broadcast(ctx.T, ctx.N) >= 0)
        b = self.b.evaluate(ctx)
        with np.errstate(invalid="ignore"):
            out = _CMP[self.op](np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        return out & np.isfinite(a) & np.isfinite(b) if self.op != "!=" else out & np.isfinite(a)


class IsIn(Expr):
    kind = "bool"

    def __init__(self, a, values: tuple) -> None:
        self.a, self.values = lift(a), values
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"{self.a!r}.isin({list(self.values)!r})"

    def _eval(self, ctx):
        a = self.a.evaluate(ctx)
        if isinstance(a, Cat):
            codes = [a.code_of(v) for v in self.values]
            return np.isin(a.broadcast(ctx.T, ctx.N), codes)
        return np.isin(np.asarray(a, dtype=np.float64), np.asarray(self.values, dtype=np.float64))


class Logical(Expr):
    kind = "bool"

    def __init__(self, op: str, a, b) -> None:
        self.op = op
        self.a, self.b = lift(a), lift(b)
        self._children = (self.a, self.b)

    def __repr__(self) -> str:
        return f"({self.a!r} {self.op} {self.b!r})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=bool)
        b = np.asarray(self.b.evaluate(ctx), dtype=bool)
        return (a & b) if self.op == "&" else (a | b)


class Not(Expr):
    kind = "bool"

    def __init__(self, a) -> None:
        self.a = lift(a)
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"~{self.a!r}"

    def _eval(self, ctx):
        return ~np.asarray(self.a.evaluate(ctx), dtype=bool)


# ====================================================================================== time series
class Shift(Expr):
    def __init__(self, a, n: int) -> None:
        self.a, self.n = lift(a), int(n)
        self._children = (self.a,)
        self.kind = self.a.kind

    def __repr__(self) -> str:
        return f"{self.a!r}.lag({self.n})" if self.n >= 0 else f"{self.a!r}.lead({-self.n})"

    def _own_lookback(self):
        return max(self.n, 0)

    def _own_lead(self):
        return -self.n

    def _shifts(self):
        return True

    def _eval(self, ctx):
        a = self.a.evaluate(ctx.full())
        if isinstance(a, Cat):
            codes = shift_rows(a.broadcast(ctx.T, ctx.N).astype(np.float64), self.n)
            return Cat(np.where(np.isnan(codes), -1, codes).astype(np.int32), a.categories)
        if self.a.kind == "bool":
            return shift_rows(np.asarray(a, dtype=bool), self.n, False)
        return shift_rows(np.asarray(a, dtype=np.float64), self.n)


class Diff(Expr):
    def __init__(self, a, n: int = 1) -> None:
        self.a, self.n = lift(a), int(n)
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"{self.a!r}.diff({self.n})"

    def _own_lookback(self):
        return self.n

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx.full()), dtype=np.float64)
        return a - shift_rows(a, self.n)


class PctChange(Diff):
    def __repr__(self) -> str:
        return f"{self.a!r}.pct_change({self.n})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx.full()), dtype=np.float64)
        prev = shift_rows(a, self.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / prev - 1.0
        return np.where(np.isfinite(out), out, np.nan)


def rolling_sum_count(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fin = np.isfinite(a)
    x = np.where(fin, a, 0.0)
    c = np.cumsum(fin, axis=0, dtype=np.float64)
    s = np.cumsum(x, axis=0)
    q = np.cumsum(x * x, axis=0)
    def win(m):
        out = m.copy()
        out[w:] = m[w:] - m[:-w]
        return out
    return win(s), win(q), win(c)


class Rolling(Expr):
    HOW = ("mean", "sum", "std", "var", "min", "max", "median", "count", "zscore")

    def __init__(self, a, window: int, how: str, min_periods: int | None = None) -> None:
        if how not in self.HOW:
            raise ValueError(f"rolling {how!r}: use one of {self.HOW}")
        self.a, self.window, self.how = lift(a), int(window), how
        self.min_periods = self.window if min_periods is None else int(min_periods)
        self._children = (self.a,)

    def __repr__(self) -> str:
        mp = "" if self.min_periods == self.window else f", min_periods={self.min_periods}"
        return f"{self.a!r}.rolling({self.window}{mp}).{self.how}()"

    def _own_lookback(self):
        return self.window - 1

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx.full()), dtype=np.float64)
        w, mp = self.window, self.min_periods
        if self.how in ("mean", "sum", "std", "var", "count", "zscore"):
            s, q, c = rolling_sum_count(a, w)
            ok = c >= max(mp, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = s / c
                if self.how == "mean":
                    out = mean
                elif self.how == "sum":
                    out = s
                elif self.how == "count":
                    return c
                else:
                    var = np.maximum(q / c - mean * mean, 0.0) * c / np.maximum(c - 1.0, 1.0)
                    if self.how == "var":
                        out = var
                    elif self.how == "std":
                        out = np.sqrt(var)
                    else:
                        out = (a - mean) / np.sqrt(var)
                        ok &= c >= 2
            return np.where(ok & np.isfinite(out), out, np.nan)
        r = pd.DataFrame(a).rolling(w, min_periods=max(mp, 1))
        return getattr(r, self.how)().to_numpy()


class _RollingBuilder:
    def __init__(self, a: Expr, window: int, min_periods: int | None) -> None:
        self.a, self.window, self.min_periods = a, window, min_periods

    def _r(self, how: str) -> Expr:
        return Rolling(self.a, self.window, how, self.min_periods)

    def mean(self):
        return self._r("mean")

    def sum(self):
        return self._r("sum")

    def std(self):
        return self._r("std")

    def var(self):
        return self._r("var")

    def min(self):
        return self._r("min")

    def max(self):
        return self._r("max")

    def median(self):
        return self._r("median")

    def count(self):
        return self._r("count")

    def zscore(self):
        """(x - rolling mean) / rolling std: a time-series z-score per instrument."""
        return self._r("zscore")


class EWM(Expr):
    def __init__(self, a, halflife: float, how: str = "mean", min_periods: int = 1) -> None:
        self.a, self.halflife, self.how, self.min_periods = lift(a), float(halflife), how, int(min_periods)
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"{self.a!r}.ewm({self.halflife:g}).{self.how}()"

    def _own_lookback(self):
        return math.ceil(4 * self.halflife)

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx.full()), dtype=np.float64)
        e = pd.DataFrame(a).ewm(halflife=self.halflife, min_periods=self.min_periods, ignore_na=True)
        return getattr(e, self.how)().to_numpy()


class _EwmBuilder:
    def __init__(self, a: Expr, halflife: float, min_periods: int) -> None:
        self.a, self.halflife, self.min_periods = a, halflife, min_periods

    def mean(self):
        return EWM(self.a, self.halflife, "mean", self.min_periods)

    def std(self):
        return EWM(self.a, self.halflife, "std", self.min_periods)


UNBOUNDED = 10 ** 9                  # look-back of a node that depends on all history (load from the first observation)


class FFill(Expr):
    def __init__(self, a, limit: int | None = None, days: int | None = None) -> None:
        self.a, self.limit, self.days = lift(a), limit, (None if days is None else int(days))
        self._children = (self.a,)

    def _params(self):
        return [self.limit] + ([] if self.days is None else [self.days])

    def __repr__(self) -> str:
        args = ([] if self.limit is None else [str(self.limit)]) + ([] if self.days is None else [f"days={self.days}"])
        return f"{self.a!r}.ffill({', '.join(args)})"

    def _own_lookback(self):
        if self.limit is None and self.days is None:
            return UNBOUNDED                                   # an unbounded carry needs every earlier value
        rows = [int(self.limit)] if self.limit is not None else []
        if self.days is not None:
            rows.append(int(np.ceil(self.days * 5 / 7)) + 1)
        return min(rows)

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx.full()), dtype=np.float64)
        out = pd.DataFrame(a).ffill(limit=self.limit).to_numpy()
        if self.days is not None:                              # calendar-day staleness, not rows
            T = a.shape[0]
            idx = np.where(np.isfinite(a), np.arange(T)[:, None], -1)
            last = np.maximum.accumulate(idx, axis=0)
            d = ctx.panel.dates.values.astype("datetime64[D]").astype(np.int64)
            age = d[:, None] - np.where(last >= 0, d[np.clip(last, 0, None)], np.iinfo(np.int64).min // 2)
            out = np.where(age <= self.days, out, np.nan)
        return out


class CumSum(Expr):
    def __init__(self, a) -> None:
        self.a = lift(a)
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"{self.a!r}.cumsum()"

    def _own_lookback(self):
        return UNBOUNDED

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx.full()), dtype=np.float64)
        return np.where(np.isfinite(a), np.cumsum(np.nan_to_num(a), axis=0), np.nan)


# ====================================================================================== cross-sectional
class _CS(Expr):
    cross_sectional = True

    def _valid(self, ctx, a: np.ndarray) -> np.ndarray:
        ok = np.isfinite(a)
        m = ctx.mask()
        return ok & m if m is not None else ok

    def _by_repr(self) -> str:
        by = getattr(self, "by", None)
        if by is None:
            return ""
        return f"by={_fmt(list(by)) if isinstance(by, tuple) else _fmt(by)}"


class Rank(_CS):
    """Percentile rank in (0, 1] per date (and group), ties averaged; ``ascending=False``: the
    largest value ranks 1/n."""

    def __init__(self, a, by=None, ascending: bool = True) -> None:
        self.a, self.by, self.ascending = lift(a), by, ascending
        self._children = (self.a,) + ((by,) if isinstance(by, Expr) else ())

    def __repr__(self) -> str:
        parts = [p for p in (self._by_repr(), "" if self.ascending else "ascending=False") if p]
        return f"{self.a!r}.rank({', '.join(parts)})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        x = a if self.ascending else -a
        valid = self._valid(ctx, a)
        return grouped_rank(np.where(valid, x, np.nan), ctx.groups(self.by), ctx.T, ctx.N)


class CSQuantile(_CS):
    """The cross-sectional ``q`` quantile per date within the universe (``@universe.oas.p90``),
    broadcast to every instrument: ``Field("oas_bp") < Field("oas_bp").cs_quantile(0.9)``."""

    def __init__(self, a, q: float) -> None:
        self.a, self.q = lift(a), float(q)
        self._children = (self.a,)

    def _params(self):
        return [self.q]

    def __repr__(self) -> str:
        return f"{self.a!r}.cs_quantile({self.q:g})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        v = np.where(self._valid(ctx, a), a, np.nan)
        with np.errstate(all="ignore"):
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                qv = np.nanquantile(v, self.q, axis=1)
        return np.broadcast_to(qv[:, None], (ctx.T, ctx.N))


def grouped_rank(x: np.ndarray, g: Cat, T: int, N: int, method: str = "average") -> np.ndarray:
    """Percentile ranks of ``x`` per row within groups (NaN = excluded)."""
    G = max(len(g.categories), 1)
    codes = g.broadcast(T, N)
    valid = np.isfinite(x) & (codes >= 0)
    r1 = row_rank(np.where(valid, x, np.nan), method)
    if G == 1:
        n = valid.sum(axis=1, keepdims=True).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(valid, r1 / n, np.nan)
    key = np.where(valid, codes.astype(np.float64) * (N + 1) + r1, np.nan)
    r2 = row_rank(key, method)
    flat = (np.arange(T, dtype=np.int64)[:, None] * G + codes)[valid]
    cnt = np.bincount(flat, minlength=T * G).reshape(T, G).astype(np.float64)
    before = np.cumsum(cnt, axis=1) - cnt
    out = np.full((T, N), np.nan)
    out[valid] = (r2[valid] - before.reshape(-1)[flat]) / cnt.reshape(-1)[flat]
    return out


class ZScore(_CS):
    """(x - mean) / sd per date within the universe (and group: groups smaller than ``min_group``
    use the date's overall moments)."""

    def __init__(self, a, by=None, min_group: int = 3) -> None:
        self.a, self.by, self.min_group = lift(a), by, int(min_group)
        self._children = (self.a,) + ((by,) if isinstance(by, Expr) else ())

    def __repr__(self) -> str:
        return f"{self.a!r}.zscore({self._by_repr()})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        valid = self._valid(ctx, a)
        xm = np.where(valid, a, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(xm, axis=1, keepdims=True) if valid.any() else np.zeros((ctx.T, 1))
            sd = np.nanstd(xm, axis=1, keepdims=True)
            overall = (xm - mu) / np.where(sd > 0, sd, np.nan)
        if self.by is None:
            return np.where(valid, overall, np.nan)
        g = ctx.groups(self.by)
        G = max(len(g.categories), 1)
        cnt, mean, ssd, v2, flat = group_stats(a, valid, g.codes, G)
        sdg = np.sqrt(ssd / np.maximum(cnt, 1))
        out = overall.copy()
        big = cnt.reshape(-1)[flat] >= self.min_group
        sdf = sdg.reshape(-1)[flat]
        zz = np.where(sdf > 0, (a[v2] - mean.reshape(-1)[flat]) / np.where(sdf > 0, sdf, 1.0), 0.0)
        out[v2] = np.where(big, zz, overall[v2])
        return np.where(valid, out, np.nan)


class Demean(_CS):
    def __init__(self, a, by=None) -> None:
        self.a, self.by = lift(a), by
        self._children = (self.a,) + ((by,) if isinstance(by, Expr) else ())

    def __repr__(self) -> str:
        return f"{self.a!r}.demean({self._by_repr()})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        valid = self._valid(ctx, a)
        g = ctx.groups(self.by)
        _cnt, mean, _, v2, flat = group_stats(a, valid, g.codes, max(len(g.categories), 1))
        out = np.full((ctx.T, ctx.N), np.nan)
        out[v2] = a[v2] - mean.reshape(-1)[flat]
        return out


class Winsor(_CS):
    def __init__(self, a, lower: float, upper: float) -> None:
        self.a, self.lower, self.upper = lift(a), lower, upper
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"{self.a!r}.winsor({self.lower:g}, {self.upper:g})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        valid = self._valid(ctx, a)
        xm = np.where(valid, a, np.nan)
        out = np.full((ctx.T, ctx.N), np.nan)
        rows = np.nonzero(valid.sum(axis=1) >= 3)[0]
        if len(rows):
            q = np.nanquantile(xm[rows], [self.lower, self.upper], axis=1)
            out[rows] = np.clip(xm[rows], q[0][:, None], q[1][:, None])
        return np.where(valid, out, np.nan) if len(rows) else xm


class Neutralise(_CS):
    """Residual of a per-date regression on ``exposures`` (and group dummies, by demeaning within
    ``by`` groups): the part of the signal that duration, DTS or sector do not explain."""

    def __init__(self, a, exposures: tuple, by=None, ridge: float = 1e-10) -> None:
        self.a, self.exposures, self.by, self.ridge = lift(a), exposures, by, ridge
        self._children = (self.a, *exposures) + ((by,) if isinstance(by, Expr) else ())

    def __repr__(self) -> str:
        args = [repr(e.name) if isinstance(e, Field) and type(e) is Field and e.lag_days is None else repr(e) for e in self.exposures]
        if self.by is not None:
            args.append(self._by_repr())
        return f"{self.a!r}.neutralise({', '.join(args)})"

    def _eval(self, ctx):
        y = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        X = [np.asarray(e.evaluate(ctx), dtype=np.float64) for e in self.exposures]
        valid = self._valid(ctx, y)
        for x in X:
            valid &= np.isfinite(x)
        g = ctx.groups(self.by)
        G = max(len(g.categories), 1)

        def dm(v):
            _, mean, _, v2, flat = group_stats(v, valid, g.codes, G)
            out = np.zeros((ctx.T, ctx.N))
            out[v2] = v[v2] - mean.reshape(-1)[flat]
            return out

        yd = dm(y)
        if not X:
            return np.where(valid, yd, np.nan)
        Xd = [dm(x) for x in X]                               # k arrays T x N (zero outside valid)
        k = len(Xd)
        resid = np.full((ctx.T, ctx.N), np.nan)
        step = max(1, int(2e7 // max(ctx.N * k, 1)))           # rows per chunk: bounded memory
        for lo in range(0, ctx.T, step):
            hi = min(ctx.T, lo + step)
            Xc = np.stack([x[lo:hi] for x in Xd], axis=2)
            yc = yd[lo:hi]
            XtX = np.einsum("tnk,tnl->tkl", Xc, Xc) + self.ridge * np.eye(k)[None]
            Xty = np.einsum("tnk,tn->tk", Xc, yc)
            beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
            resid[lo:hi] = yc - np.einsum("tnk,tk->tn", Xc, beta)
        return np.where(valid, resid, np.nan)


class Bucket(_CS):
    """A category from a numeric value: fixed ``edges`` (``[4, 7, 10]`` -> four buckets) or ``n``
    per-date quantile buckets (within ``by`` groups)."""

    kind = "category"

    def __init__(self, a, edges: tuple | None = None, n: int | None = None, by=None) -> None:
        if (edges is None) == (n is None):
            raise ValueError("bucket: give edges or n, not both")
        self.a, self.edges, self.n, self.by = lift(a), edges, n, by
        self._children = (self.a,) + ((by,) if isinstance(by, Expr) else ())
        self.cross_sectional = n is not None

    def __repr__(self) -> str:
        return f"{self.a!r}.bucket({_fmt(list(self.edges)) if self.edges is not None else f'n={self.n}'}{', ' + self._by_repr() if self.by is not None else ''})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        fin = np.isfinite(a)
        if self.edges is not None:
            codes = np.where(fin, np.searchsorted(np.asarray(self.edges), a, side="right"), -1).astype(np.int32)
            e = list(self.edges)
            labels = [f"<{e[0]:g}"] + [f"{e[i]:g}-{e[i + 1]:g}" for i in range(len(e) - 1)] + [f">={e[-1]:g}"]
            return Cat(codes, labels)
        valid = self._valid(ctx, a)
        pct = grouped_rank(np.where(valid, a, np.nan), ctx.groups(self.by), ctx.T, ctx.N, "first")
        codes = np.where(valid, np.clip(np.ceil(pct * self.n) - 1, 0, self.n - 1), -1).astype(np.int32)
        return Cat(codes, [f"Q{i + 1}" for i in range(self.n)])


class Top(_CS):
    """True for the ``n`` largest (``bottom``: smallest) values per date (and group)."""

    kind = "bool"

    def __init__(self, a, n: int, by=None, bottom: bool = False) -> None:
        self.a, self.n, self.by, self.bottom_ = lift(a), int(n), by, bottom
        self._children = (self.a,) + ((by,) if isinstance(by, Expr) else ())

    def __repr__(self) -> str:
        return f"{self.a!r}.{'bottom' if self.bottom_ else 'top'}({self.n}{', ' + self._by_repr() if self.by is not None else ''})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        valid = self._valid(ctx, a)
        x = np.where(valid, a if self.bottom_ else -a, np.nan)
        g = ctx.groups(self.by)
        G = max(len(g.categories), 1)
        if G == 1:
            r = row_rank(x, "first")
            return np.nan_to_num(r, nan=np.inf) <= self.n
        pct = grouped_rank(x, g, ctx.T, ctx.N, "first")
        codes = g.broadcast(ctx.T, ctx.N)
        flat = (np.arange(ctx.T, dtype=np.int64)[:, None] * G + codes)[valid & (codes >= 0)]
        cnt = np.bincount(flat, minlength=ctx.T * G).reshape(ctx.T, G)
        out = np.zeros((ctx.T, ctx.N), dtype=bool)
        m = valid & (codes >= 0)
        out[m] = np.round(pct[m] * cnt.reshape(-1)[flat]) <= self.n
        return out


class Scale(_CS):
    def __init__(self, a, total: float = 1.0) -> None:
        self.a, self.total = lift(a), total
        self._children = (self.a,)

    def __repr__(self) -> str:
        return f"{self.a!r}.scale({self.total:g})"

    def _eval(self, ctx):
        a = np.asarray(self.a.evaluate(ctx), dtype=np.float64)
        valid = self._valid(ctx, a)
        x = np.where(valid, a, 0.0)
        s = np.abs(x).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(valid & (s > 0), x / s * self.total, np.nan)


class Combine(_CS):
    """Weighted blend of signals, each z-scored per date first (``standardise=True``); a missing
    component counts as zero where any other is present."""

    def __init__(self, exprs: tuple, weights: tuple | None = None, standardise: bool = True) -> None:
        self.exprs = tuple(lift(e) for e in exprs)
        self.weights = tuple(float(w) for w in (weights or [1.0] * len(self.exprs)))
        if len(self.weights) != len(self.exprs):
            raise ValueError("combine: one weight per signal")
        self.standardise = standardise
        self._children = self.exprs

    def __repr__(self) -> str:
        return f"combine({', '.join(repr(e) for e in self.exprs)}, weights={_fmt(list(self.weights))}{'' if self.standardise else ', standardise=False'})"

    def _eval(self, ctx):
        total = np.zeros((ctx.T, ctx.N))
        any_ = np.zeros((ctx.T, ctx.N), dtype=bool)
        for e, w in zip(self.exprs, self.weights):
            v = (ZScore(e) if self.standardise else e).evaluate(ctx)
            v = np.asarray(v, dtype=np.float64)
            fin = np.isfinite(v)
            total += np.where(fin, v, 0.0) * w
            any_ |= fin
        return np.where(any_, total, np.nan)


def combine(*exprs, weights: Sequence[float] | None = None, standardise: bool = True) -> Expr:
    return Combine(tuple(exprs), None if weights is None else tuple(weights), standardise)


def where(cond, a, b=np.nan) -> Expr:
    return Where(cond, a, b)


# ====================================================================================== custom Python
class Apply(Expr):
    """``fn(*arrays) -> array``: custom vectorised code on (dates x instruments) arrays. It cannot
    be analysed, so ``check_lookahead`` tests it on truncated panels; declare ``causal=True``
    only when the function provably uses rows <= t for row t."""

    def __init__(self, fn: Callable, inputs: tuple, name: str | None = None, lookback: int = 0, causal: bool = False) -> None:
        self.fn, self.inputs, self.name, self.lookback_rows, self.causal = fn, tuple(lift(i) for i in inputs), name or getattr(fn, "__name__", "fn"), int(lookback), causal
        self._children = self.inputs

    def __repr__(self) -> str:
        return f"apply({self.name}, {', '.join(repr(i) for i in self.inputs)})"

    def _make_key(self) -> str:
        return f"{self!r}#{callable_fingerprint(self.fn)['sha']}"

    def _own_lookback(self):
        return self.lookback_rows

    def _own_lead(self):
        return 0 if self.causal else None

    def _eval(self, ctx):
        arrays = [np.asarray(i.evaluate(ctx.full()), dtype=np.float64) for i in self.inputs]
        out = self.fn(*arrays)
        if isinstance(out, pd.DataFrame):
            out = out.to_numpy(dtype=np.float64)
        out = np.asarray(out, dtype=np.float64)
        if out.shape != (ctx.T, ctx.N):
            raise ValueError(f"{self!r} returned shape {out.shape}, expected {(ctx.T, ctx.N)}")
        return out


class CrossSectional(Expr):
    """``fn(cross_section) -> Series``: custom per-date code. ``cross_section`` is a Series (one
    input) or a DataFrame (several, columns named by input) over the instruments in the universe;
    the result is aligned by instrument id. Evaluated only on the rows the engine needs (decision
    dates) unless a time-series transform sits above it."""

    cross_sectional = True
    row_local = True

    def __init__(self, fn: Callable, inputs: tuple, name: str | None = None) -> None:
        self.fn, self.inputs, self.name = fn, tuple(lift(i) for i in inputs), name or getattr(fn, "__name__", "fn")
        self._children = self.inputs

    def __repr__(self) -> str:
        return f"cs({self.name}, {', '.join(repr(i) for i in self.inputs)})"

    def _make_key(self) -> str:
        return f"{self!r}#{callable_fingerprint(self.fn)['sha']}"

    def _eval(self, ctx):
        arrays = [i.evaluate(ctx) for i in self.inputs]
        ids = ctx.panel.ids
        m = ctx.mask()
        rows = ctx.rows if ctx.rows is not None else np.arange(ctx.T)
        out = np.full((ctx.T, ctx.N), np.nan)
        cols = [i.name if isinstance(i, Field) else f"x{k}" for k, i in enumerate(self.inputs)]
        for t in rows:
            ok = np.ones(ctx.N, dtype=bool) if m is None else m[t].copy()
            vals = []
            for a in arrays:
                if isinstance(a, Cat):
                    labels = np.array([*a.categories, None], dtype=object)
                    cc = a.broadcast(ctx.T, ctx.N)[t]
                    vals.append(labels[np.where(cc < 0, len(a.categories), cc)])
                else:
                    vals.append(np.asarray(a[t], dtype=np.float64))
            ok &= np.any([np.isfinite(v) for v in vals if v.dtype.kind == "f"] or [np.ones(ctx.N, dtype=bool)], axis=0)
            if not ok.any():
                continue
            if len(vals) == 1:
                x = pd.Series(vals[0][ok], index=ids[ok], name=cols[0])
            else:
                x = pd.DataFrame({c: v[ok] for c, v in zip(cols, vals)}, index=ids[ok])
            res = self.fn(x)
            res = pd.Series(res, dtype=float) if not isinstance(res, pd.Series) else res.astype(float)
            if not res.index.equals(x.index):
                res = res.reindex(x.index)
            out[t, ok] = res.to_numpy()
        return out


def apply(fn: Callable, *inputs, name: str | None = None, lookback: int = 0, causal: bool = False) -> Expr:
    return Apply(fn, tuple(inputs), name, lookback, causal)


def cross_sectional(fn: Callable, *inputs, name: str | None = None) -> Expr:
    return CrossSectional(fn, tuple(inputs), name)


# ====================================================================================== look-ahead
def check_lookahead(expr: Expr, ctx: EvalContext, cuts: int = 3, rtol: float = 1e-9, atol: float = 1e-12, allow: bool = False) -> dict:
    """Structural check (``lead() > 0``), then an empirical one for code that cannot be analysed:
    evaluate on panels truncated at ``cuts`` dates and require every value up to the cut to be
    unchanged. Raises ``LookaheadError`` unless ``allow``; returns what was checked."""
    lead = expr.lookahead()
    report = {"expr": repr(expr), "structural_lead": lead, "empirical": False, "cuts": []}
    if lead is not None and lead > 0:
        msg = f"{expr!r} reads {lead} row(s) into the future (lead/negative lag); signals must use data known at the decision"
        if not allow:
            raise LookaheadError(msg)
        report["violation"] = msg
        return report
    if lead is not None:
        return report
    report["empirical"] = True
    full = expr.evaluate(ctx)
    T = ctx.T
    for c in sorted({int(T * f) for f in np.linspace(0.35, 0.85, cuts)}):
        if c < 2 or c >= T - 1:
            continue
        p = ctx.panel
        sub = Panel(p.dates[: c + 1], p.ids, {k: v[: c + 1] for k, v in p.values.items()}, dict(p.lags))
        cctx = EvalContext(sub, ctx.attributes, ctx.series.iloc[: c + 1], ctx.source, ctx.catalog, None if ctx.universe is None else ctx.universe[: c + 1], ctx.universe_key + f"|cut{c}", None, ExprCache(), ctx.return_field, ctx.lags)
        part = expr.evaluate(cctx)
        a = full.broadcast(T, ctx.N)[: c + 1] if isinstance(full, Cat) else np.asarray(full)[: c + 1]
        b = part.broadcast(c + 1, ctx.N) if isinstance(part, Cat) else np.asarray(part)
        if a.dtype.kind == "f":
            same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        else:
            same = a == b
        bad = np.nonzero(~same.all(axis=1))[0]
        report["cuts"].append({"cut": str(p.dates[c].date()), "rows_changed": len(bad)})
        if len(bad):
            msg = f"{expr!r} changes on {p.dates[bad[0]].date()} when data after {p.dates[c].date()} are removed: it uses later data"
            if not allow:
                raise LookaheadError(msg)
            report["violation"] = msg
            return report
    return report


__all__ = [
    "All",
    "Apply",
    "Attribute",
    "Available",
    "CSQuantile",
    "Cat",
    "Combine",
    "Const",
    "CrossSectional",
    "EvalContext",
    "Expr",
    "ExprCache",
    "Field",
    "LookaheadError",
    "TimeSeries",
    "alive_mask",
    "apply",
    "callable_fingerprint",
    "check_lookahead",
    "combine",
    "cross_sectional",
    "grouped_rank",
    "lift",
    "row_rank",
    "where",
]
