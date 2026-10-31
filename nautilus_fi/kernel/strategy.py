"""The strategy: one immutable Python object that says what to hold and how it is traded.

::

    s = Strategy("credit_value",
                 universe=Index("LUACTRUU") & (Field("rating_score") <= 10) & (Field("oasd") > 1),
                 signal=Field("oas_bp").zscore(by="sector").neutralise("oasd"),
                 portfolio=Optimise(3e-4 * Alpha() - TrackingError(by=["sector"]) - Costs(),
                                    ActiveWeight("issuer") <= 0.005, Turnover() <= 0.10),
                 benchmark=Index("LUACTRUU"),
                 schedule=MonthEnd(),
                 execution=NextClose(lag=1, cost=HalfSpread() * 1.5 + SqrtImpact(10)),
                 financing=Cash("SOFR"))

    tighter = s.with_(portfolio=s.portfolio.replace(Turnover() <= 0.05), name="turnover_5")
    print(tighter.diff(s))            # what changed, component by component
    tighter.fingerprint               # content hash, including the source of custom functions

Every component is a Python object, and what the engine does is decided by those objects:
nothing is read from configuration files. ``with_`` returns a changed copy; ``set("execution.lag",
2)`` changes a nested value by path (what sweeps use). ``describe()`` prints the strategy as code
and prose for a PM; ``requirements(catalog)`` lists the data it needs; ``validate(source)`` finds
missing data and look-ahead before a long run; ``conventions()`` is the ``Conventions`` record the
manifests, the ledger and the official gates read; ``under("MRI_v1")`` re-runs it under a legacy
framework's conventions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import re
from enum import Enum

import numpy as np
import pandas as pd

from nautilus_fi.domain.calendars import RebalanceCalendar, parse_calendar
from nautilus_fi.domain.conventions import PROFILES, WeightDrift
from nautilus_fi.kernel.benchmark import Benchmark, Index, as_benchmark
from nautilus_fi.kernel.data import (
    ASSET,
    ATTRIBUTE,
    OVERLAY,
    SERIES,
    Catalog,
    UnknownFieldError,
    frame_fingerprint,
)
from nautilus_fi.kernel.expr import All, Expr, Field, LookaheadError, callable_fingerprint, lift
from nautilus_fi.kernel.mechanics import Cash as _Cash
from nautilus_fi.kernel.mechanics import (
    Execution,
    Financing,
    Metrics,
    MissingReturns,
    NextClose,
    conventions_of,
    mechanics_from_conventions,
)
from nautilus_fi.kernel.mechanics import Fail as _Fail
from nautilus_fi.kernel.portfolio import (
    BenchmarkWeight,
    Construction,
    EqualWeight,
    Hedge,
    Quantile,
    as_construction,
)

KERNEL_VERSION = "1"
OPTIONAL = ("alive", "xr", "tsy_r", "oad", "oasd", "dts", "oas_bp", "sector", "rating_score", "issuer", "half_spread_bp", "price_dirty", "price_clean", "min_piece", "cash")
REPORT_EXPOSURES = ("oad", "oasd", "dts", "oas_bp")


class StrategyError(ValueError):
    """The strategy cannot run on this source (missing data, look-ahead, inconsistent parts)."""


# ====================================================================================== canonical specs
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def spec_of(v, depth: int = 0):
    """A JSON-able, deterministic description of any strategy component (for hashing and diffs)."""
    if depth > 12:
        return "<deep>"
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (np.integer, np.floating)):
        return spec_of(v.item(), depth)
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, Expr):
        code = v.callables()
        return {"expr": repr(v), **({"code": code} if code else {})}
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return {"frame": frame_fingerprint(v), "shape": list(v.shape)}
    if isinstance(v, np.ndarray):
        return {"array": hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()[:16], "shape": list(v.shape)}
    if isinstance(v, (list, tuple)):
        return [spec_of(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): spec_of(x, depth + 1) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (set, frozenset)):
        return sorted(spec_of(x, depth + 1) for x in v)
    if isinstance(v, RebalanceCalendar):
        d = {"calendar": str(v)}
        if dataclasses.is_dataclass(v):
            d.update({f.name: spec_of(getattr(v, f.name), depth + 1) for f in dataclasses.fields(v) if f.name != "name"})
        return d
    if _is_decider(v):
        try:
            code = hashlib.sha256(inspect.getsource(type(v)).encode()).hexdigest()[:16]
        except (OSError, TypeError):
            code = type(v).__qualname__
        return {"decider": type(v).__qualname__, "code": code, "params": spec_of(v.params(), depth + 1)}
    if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(v):
        return callable_fingerprint(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {"type": type(v).__qualname__, **{f.name: spec_of(getattr(v, f.name), depth + 1) for f in dataclasses.fields(v) if not f.name.startswith("_")}}
    if hasattr(v, "__dict__") and not isinstance(v, type):
        items = {k: x for k, x in vars(v).items() if not k.startswith("_")}
        out = {"type": type(v).__qualname__, **{k: spec_of(x, depth + 1) for k, x in sorted(items.items())}}
        if callable(v) and type(v).__call__ is not object.__call__ and not isinstance(v, (Construction,)):
            try:
                out["code"] = hashlib.sha256(inspect.getsource(type(v)).encode()).hexdigest()[:16]
            except (OSError, TypeError):
                pass
        return out
    if callable(v):
        return callable_fingerprint(v)
    return {"type": type(v).__qualname__, "repr": _ADDR.sub("", repr(v))[:300]}


def _is_decider(v) -> bool:
    from nautilus_fi.domain.spec import Decider

    return isinstance(v, Decider)


def _leaf_diffs(a, b, path: str, out: list) -> None:
    if a == b:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("type") != b.get("type") and ("type" in a or "type" in b):
            out.append((path, a, b))
            return
        for k in sorted(set(a) | set(b), key=str):
            _leaf_diffs(a.get(k), b.get(k), f"{path}.{k}" if path else str(k), out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _leaf_diffs(x, y, f"{path}[{i}]", out)
        return
    out.append((path, a, b))


@dataclasses.dataclass
class Change:
    component: str
    before: str
    after: str
    details: list = dataclasses.field(default_factory=list)       # (path, before, after) leaves

    def __str__(self) -> str:
        return f"{self.component}: {self.before}  ->  {self.after}"


@dataclasses.dataclass
class Diff:
    changes: list[Change]

    def __bool__(self) -> bool:
        return bool(self.changes)

    def __len__(self) -> int:
        return len(self.changes)

    def __iter__(self):
        return iter(self.changes)

    def components(self) -> list[str]:
        return [c.component for c in self.changes]

    def __str__(self) -> str:
        if not self.changes:
            return "no differences"
        return "\n".join(f"* {c}" for c in self.changes)

    __repr__ = __str__


# ====================================================================================== requirements
@dataclasses.dataclass
class Requirements:
    fields: set = dataclasses.field(default_factory=set)
    attributes: set = dataclasses.field(default_factory=set)
    series: set = dataclasses.field(default_factory=set)
    indices: set = dataclasses.field(default_factory=set)
    overlay_fields: set = dataclasses.field(default_factory=set)
    overlay_ids: tuple = ()
    lookback: int = 0
    missing_optional: set = dataclasses.field(default_factory=set)

    def union(self, other: Requirements) -> Requirements:
        return Requirements(self.fields | other.fields, self.attributes | other.attributes, self.series | other.series, self.indices | other.indices, self.overlay_fields | other.overlay_fields, tuple(dict.fromkeys(self.overlay_ids + other.overlay_ids)), max(self.lookback, other.lookback), self.missing_optional & other.missing_optional)

    def __str__(self) -> str:
        return (f"fields {sorted(self.fields)}; attributes {sorted(self.attributes)}; series {sorted(self.series)}; indices {sorted(self.indices)}"
                f"{'; overlays ' + str(list(self.overlay_ids)) + ' ' + str(sorted(self.overlay_fields)) if self.overlay_ids else ''}; look-back {self.lookback} rows")


def _max_window(obj, depth: int = 0) -> int:
    if depth > 8 or obj is None:
        return 0
    best = 0
    if isinstance(obj, (list, tuple)):
        for x in obj:
            best = max(best, _max_window(x, depth + 1))
        return best
    if hasattr(obj, "__dict__") and not isinstance(obj, (Expr, type, pd.DataFrame, pd.Series)):
        for k, v in vars(obj).items():
            if k == "window" and isinstance(v, int):
                best = max(best, v)
            elif not k.startswith("_") or k in ("_children",):
                best = max(best, _max_window(v, depth + 1))
    return best


# ====================================================================================== the strategy
@dataclasses.dataclass(frozen=True, eq=False)
class Strategy:
    name: str = "strategy"
    universe: Expr | None = None                       # None: the benchmark's members, else every live instrument
    signal: Expr | None = None
    portfolio: Construction | None = None              # None: Quantile(5) with a signal, BenchmarkWeight with a benchmark, else EqualWeight
    benchmark: Benchmark | None = None
    schedule: RebalanceCalendar | str = "BME"
    execution: Execution = dataclasses.field(default_factory=NextClose)
    financing: Financing = dataclasses.field(default_factory=_Cash)
    missing: MissingReturns = dataclasses.field(default_factory=_Fail)
    metrics: Metrics = dataclasses.field(default_factory=Metrics)
    overlay: Hedge | None = None
    capital: float = 100e6
    returns: str = "tr"
    decider: object = None                             # escape hatch: a domain.spec.Decider (decide(view) -> Target)
    universe_rule: object = None                       # legacy domain.spec.UniverseRule (decider path)
    fields: tuple = ()                                 # extra data a decider reads (expressions declare their own)
    lookback: int = 0                                  # extra history rows a decider reads (view.history)
    hypothesis: str | None = None
    profile: str = "kernel"
    notes: str = ""

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if isinstance(self.signal, str):
            set_(self, "signal", Field(self.signal))
        if self.universe is not None and not isinstance(self.universe, Expr):
            set_(self, "universe", lift(self.universe))
        set_(self, "benchmark", as_benchmark(self.benchmark))
        if self.portfolio is not None:
            set_(self, "portfolio", as_construction(self.portfolio))
        set_(self, "schedule", parse_calendar(self.schedule))
        set_(self, "fields", tuple(self.fields))
        if self.signal is not None and self.signal.kind != "numeric":
            raise StrategyError(f"the signal must be numeric, not a {self.signal.kind}: {self.signal!r}")
        if self.universe is not None and self.universe.kind != "bool":
            raise StrategyError(f"the universe must be a predicate (comparisons, Index(...), & | ~), not {self.universe!r}")

    # ------------------------------------------------------------------ variants
    def with_(self, **changes) -> Strategy:
        """A changed copy: ``s.with_(signal=..., name=...)``."""
        unknown = set(changes) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError(f"Strategy has no field(s) {sorted(unknown)}; fields: {[f.name for f in dataclasses.fields(self)]}")
        return dataclasses.replace(self, **changes)

    def set(self, path: str, value) -> Strategy:
        """A copy with a nested value changed by path: ``s.set("execution.lag", 2)``,
        ``s.set("portfolio.n", 30)``, ``s.set("financing.borrow_spread_bp", 50)``."""
        head, _, rest = path.partition(".")
        if not rest:
            return self.with_(**{head: value})
        return self.with_(**{head: _replace_path(getattr(self, head), rest, value)})

    def under(self, profile: str) -> Strategy:
        """This strategy under a legacy framework's conventions (timing, missing returns, excess
        returns, turnover, annualisation, benchmark drift); costs, universe and signal unchanged."""
        conv = PROFILES.get(profile)
        if conv is None:
            raise KeyError(f"unknown conventions profile {profile!r}; known: {sorted(PROFILES)}")
        m = mechanics_from_conventions(conv)
        bench = self.benchmark
        if isinstance(bench, Index):
            bench = Index(bench.name, m["benchmark_drift"], bench.weight_field)
        return self.with_(execution=dataclasses.replace(self.execution, lag=m["execution_lag"]), missing=m["missing"], metrics=m["metrics"], benchmark=bench, profile=profile, name=f"{self.name}[{profile}]")

    @classmethod
    def from_spec(cls, spec, name: str | None = None) -> Strategy:
        """A legacy ``ResearchSpec`` (decider + ConstructionSpec + conventions) as a kernel strategy."""
        from nautilus_fi.kernel.mechanics import MissingReturns as MR

        conv = spec.conventions
        m = mechanics_from_conventions(conv)
        return cls(name or spec.name, portfolio=spec.construction, schedule=spec.rebalance, execution=NextClose(conv.timing.trade_offset_days), missing=MR(conv.missing_returns, conv.bad_date_threshold), metrics=m["metrics"], decider=spec.decider, universe_rule=spec.universe, hypothesis=spec.hypothesis_id, profile=conv.name)

    # ------------------------------------------------------------------ resolution
    def resolved(self) -> Strategy:
        """Defaults filled in: the universe (benchmark members or every live instrument) and the
        portfolio (``Quantile(5)`` with a signal, ``BenchmarkWeight()`` with a benchmark,
        ``EqualWeight()`` otherwise)."""
        uni = self.universe
        if uni is None:
            uni = self.benchmark if (self.benchmark is not None and self.benchmark.has_weights) else All()
        port = self.portfolio
        if port is None:
            if self.decider is not None:
                port = EqualWeight()
            elif self.signal is not None:
                port = Quantile(5)
            elif self.benchmark is not None and self.benchmark.has_weights:
                port = BenchmarkWeight()
            else:
                port = EqualWeight()
        if uni is self.universe and port is self.portfolio:
            return self
        return dataclasses.replace(self, universe=uni, portfolio=port)

    # ------------------------------------------------------------------ identity
    def spec(self) -> dict:
        s = self.resolved()
        return {f.name: spec_of(getattr(s, f.name)) for f in dataclasses.fields(s) if f.name not in ("name", "notes")}

    @property
    def fingerprint(self) -> str:
        fp = self.__dict__.get("_fp")
        if fp is None:
            from nautilus_fi.version import __version__

            payload = json.dumps({"kernel": KERNEL_VERSION, "package": __version__, "spec": self.spec()}, sort_keys=True, default=str)
            fp = hashlib.sha256(payload.encode()).hexdigest()[:16]
            object.__setattr__(self, "_fp", fp)
        return fp

    @property
    def hash(self) -> str:                             # ResearchSpec-compatible
        return self.fingerprint

    @property
    def hypothesis_id(self) -> str | None:             # ResearchSpec-compatible
        return self.hypothesis

    def __eq__(self, other) -> bool:
        return isinstance(other, Strategy) and other.fingerprint == self.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def diff(self, other: Strategy) -> Diff:
        """What differs between ``other`` (before) and this strategy (after), by component."""
        a, b = other.spec(), self.spec()
        ra, rb = other.resolved(), self.resolved()
        changes = []
        for f in dataclasses.fields(self):
            if f.name in ("name", "notes"):
                continue
            if a.get(f.name) != b.get(f.name):
                leaves: list = []
                _leaf_diffs(a.get(f.name), b.get(f.name), f.name, leaves)
                changes.append(Change(f.name, _short(getattr(ra, f.name)), _short(getattr(rb, f.name)), leaves))
        return Diff(changes)

    def variant(self) -> dict:
        """Ledger parameters of this trial: its components as code."""
        s = self.resolved()
        return {"name": self.name, "fingerprint": self.fingerprint, "universe": _short(s.universe), "signal": _short(s.signal), "portfolio": _short(s.portfolio), "benchmark": _short(s.benchmark), "schedule": str(s.schedule), "execution": _short(s.execution), "financing": _short(s.financing), "missing": _short(s.missing)}

    def describe(self) -> str:
        s = self.resolved()
        rows = [("universe", s.universe), ("signal", s.signal), ("portfolio", s.portfolio), ("benchmark", s.benchmark), ("schedule", f"{s.schedule} ({_calendar_prose(s.schedule)})"), ("execution", f"{s.execution!r}: {s.execution.describe()}"), ("financing", s.financing), ("missing", s.missing), ("metrics", f"{s.metrics!r}: annualisation {s.metrics.annualisation.value}, turnover {s.metrics.turnover.value}, excess {s.metrics.excess!r}"), ("overlay", s.overlay), ("capital", f"{s.capital:,.0f}"), ("returns", s.returns)]
        if s.decider is not None:
            rows.insert(1, ("decider", f"{type(s.decider).__name__}({', '.join(f'{k}={v!r}' for k, v in s.decider.params().items())})"))
        lines = [f"Strategy {self.name!r}  [fingerprint {self.fingerprint}{', hypothesis ' + self.hypothesis if self.hypothesis else ''}{', profile ' + self.profile if self.profile != 'kernel' else ''}]"]
        lines += [f"  {k:<10} {v if isinstance(v, str) else repr(v)}" for k, v in rows if v is not None]
        if self.notes:
            lines.append(f"  notes      {self.notes}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Strategy({self.name!r}, fingerprint={self.fingerprint})"

    # ------------------------------------------------------------------ data
    def expressions(self) -> list[Expr]:
        s = self.resolved()
        out = [e for e in (s.universe, s.signal, s.benchmark) if e is not None]
        out += _exprs_in(s.portfolio)
        out += _exprs_in(s.execution.cost)
        return out

    def requirements(self, catalog: Catalog | None = None) -> Requirements:
        """Fields, attributes, series and indices this strategy needs, classified by the catalogue
        (unknown names raise with suggestions); the look-back in rows before the first decision."""
        s = self.resolved()
        names: set[str] = set()
        indices: set[str] = set()
        lookback = 0
        for e in self.expressions():
            names |= e.names()
            indices |= e.indices()
            lookback = max(lookback, e.lookback())
        names |= s.portfolio.names() | s.execution.names() | s.missing.names() | set(s.fields)
        names |= legacy_names(s.decider, s.universe_rule, getattr(s.portfolio, "spec", None))
        lookback = max(lookback, int(s.lookback))
        names.add(s.returns)
        if s.financing.series:
            names.add(s.financing.series)
        if s.metrics.risk_free:
            names.add(s.metrics.risk_free)
        optional = set(OPTIONAL) | s.metrics.names() | set(REPORT_EXPOSURES) | s.execution.optional_names()
        if s.overlay is not None:
            names |= s.overlay.names()
        lookback = max(lookback, _max_window(s.portfolio))
        req = Requirements(indices=indices, lookback=lookback)
        if s.overlay is not None:
            req.overlay_ids = tuple(s.overlay.instruments)
            req.overlay_fields = {s.returns, *s.overlay.fields}
        if catalog is None:
            req.fields = names
            return req
        maxlag = 0
        for n in sorted(names | optional):
            info = catalog.get(n)
            if info is None:
                if n in names and n not in optional and n != s.financing.series:
                    raise UnknownFieldError(n, "field", catalog.suggest(n), catalog.name)
                req.missing_optional.add(n)
                continue
            if info.entity == ASSET:
                req.fields.add(n)
                if n in names:
                    maxlag = max(maxlag, info.lag)
            elif info.entity == ATTRIBUTE:
                req.attributes.add(n)
            elif info.entity == SERIES:
                req.series.add(n)
            elif info.entity == OVERLAY:
                req.overlay_fields.add(n)
        for ix in indices:
            if ix not in catalog.indices:
                raise UnknownFieldError(ix, "index", catalog.suggest(ix), catalog.name)
        req.lookback = lookback + maxlag
        return req

    def validate(self, source=None, strict: bool = False) -> list[str]:
        """Problems found before running: missing data, look-ahead, parts that do not fit."""
        s = self.resolved()
        issues: list[str] = []
        if source is not None:
            try:
                req = self.requirements(source.catalog())
                if s.financing.series and s.financing.series in req.missing_optional:
                    issues.append(f"financing series {s.financing.series!r} not in the source: cash earns zero (use NoCash() to say so, or name the series)")
            except UnknownFieldError as e:
                issues.append(str(e))
        for e in self.expressions():
            la = e.lookahead()
            if la is not None and la > 0:
                issues.append(f"look-ahead: {e!r} reads {la} row(s) into the future")
            elif la is None:
                issues.append(f"note: {e!r} contains custom code; the engine checks it on truncated panels before using it")
        if s.portfolio.needs_scores and s.signal is None and s.decider is None:
            issues.append(f"{s.portfolio!r} ranks by a signal but the strategy has none")
        if s.benchmark is not None and not s.benchmark.has_weights and isinstance(s.portfolio, BenchmarkWeight):
            issues.append(f"{s.portfolio!r} needs benchmark constituents; {s.benchmark!r} is a return series")
        if strict and any(not i.startswith("note:") for i in issues):
            raise StrategyError("; ".join(i for i in issues if not i.startswith("note:")))
        return issues

    def check_lookahead(self) -> None:
        for e in self.expressions():
            la = e.lookahead()
            if la is not None and la > 0:
                raise LookaheadError(f"{e!r} reads {la} row(s) into the future")

    def conventions(self):
        s = self.resolved()
        drift = s.benchmark.drift if isinstance(s.benchmark, Index) else WeightDrift.DRIFT
        prof = PROFILES.get(self.profile)
        return conventions_of(s.execution, s.missing, s.metrics, s.schedule, drift, s.financing, self.profile, prof.confirmed if prof is not None else True)


def legacy_names(decider=None, rule=None, construction=None) -> set[str]:
    """The data a legacy ``Decider`` / ``UniverseRule`` / ``ConstructionSpec`` reads, where it can
    be read off its parameters (``FieldSignal`` fields, rule fields, constraint attributes ...).
    A decider can also declare ``fields``."""
    out: set[str] = set()
    if decider is not None:
        for attr in ("field_name", "zscore_by"):
            v = getattr(decider, attr, None)
            if isinstance(v, str):
                out.add(v)
        for attr in ("neutralise_exposures", "neutralise_groups", "fields"):
            out |= {x for x in (getattr(decider, attr, None) or ()) if isinstance(x, str)}
    if rule is not None:
        out |= {f for f, _, _ in rule.field_rules} | set(rule.require) | set(rule.include) | set(rule.exclude)
    if construction is not None and hasattr(construction, "constraints"):
        for c in construction.constraints:
            out |= {x for x in (getattr(c, "attribute", None), getattr(c, "group_by", None)) if x}
        m = getattr(construction, "method", None)
        for attr in ("by", "weighting", "exposure"):
            v = getattr(m, attr, None)
            if isinstance(v, str) and v not in ("equal", "benchmark", "notional"):
                out.add(v)
        obj = getattr(m, "objective", None)
        if obj is not None and getattr(obj, "tracking", None) is not None:
            out |= set(obj.tracking.fields) | set(obj.tracking.by)
        if getattr(construction, "issuer_cap", None) is not None:
            out.add(construction.issuer_col)
        if getattr(construction, "overlay", None) is not None:
            out |= set(construction.overlay.fields)
    return out


def _short(v) -> str:
    if v is None:
        return "None"
    r = repr(v)
    return r if len(r) <= 400 else r[:397] + "..."


def _calendar_prose(cal) -> str:
    return {"D": "every trading day", "BME": "last trading day of the month", "BMS": "first trading day of the month", "W": "first trading day of the week", "BQE": "last trading day of the quarter", "BQS": "first trading day of the quarter"}.get(str(cal), type(cal).__name__)


def _exprs_in(obj, depth: int = 0) -> list[Expr]:
    if obj is None or depth > 8:
        return []
    if isinstance(obj, Expr):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [e for x in obj for e in _exprs_in(x, depth + 1)]
    if hasattr(obj, "__dict__") and not isinstance(obj, (type, pd.DataFrame, pd.Series)):
        return [e for k, v in vars(obj).items() if not k.startswith("_") or k == "constraints_" for e in _exprs_in(v, depth + 1)]
    return []


def _replace_path(obj, path: str, value):
    head, _, rest = path.partition(".")
    if rest:
        value = _replace_path(getattr(obj, head), rest, value)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{head: value})
    if hasattr(obj, "with_"):
        return obj.with_(**{head: value})
    raise TypeError(f"cannot set {head!r} on {type(obj).__name__}")


__all__ = ["KERNEL_VERSION", "Change", "Diff", "Requirements", "Strategy", "StrategyError", "legacy_names", "spec_of"]
