"""The research spec and the decision API shared by every engine.

A strategy that should run in both the panel engine (Paper) and the event engine (Implement) is a
``Decider``: ``decide(view) -> Target``. The ``DecisionView`` is the point-in-time cross-section
the engine hands it (fields as known at the decision time, holdings, benchmark weights, the
eligible universe); the ``Target`` carries scores for a construction method, final weights, or
overlay notionals. ``ResearchSpec`` bundles decider, construction, universe rule, calendar and
conventions, and hashes them into the manifest and the trial ledger.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from nautilus_fi.domain.calendars import RebalanceCalendar, parse_calendar
from nautilus_fi.domain.conventions import Conventions


class DecisionView(ABC):
    """What a decider may see on a decision date. ``ids`` = eligible universe + holdings + benchmark."""

    date: pd.Timestamp
    nav: float

    @property
    @abstractmethod
    def universe(self) -> pd.Index: ...

    @property
    @abstractmethod
    def holdings(self) -> pd.Series: ...

    @property
    @abstractmethod
    def benchmark(self) -> pd.Series: ...

    @property
    @abstractmethod
    def attributes(self) -> pd.DataFrame: ...

    @abstractmethod
    def field(self, name: str) -> pd.Series: ...

    @abstractmethod
    def has(self, name: str) -> bool: ...

    def history(self, name: str, n: int) -> pd.DataFrame:
        raise NotImplementedError(f"{type(self).__name__} does not keep field history")

    @property
    def ids(self) -> pd.Index:
        return self.universe.union(self.holdings.index).union(self.benchmark.index)

    def fields(self, names: list[str]) -> pd.DataFrame:
        return pd.DataFrame({n: self.field(n) for n in names if self.has(n)}).reindex(self.ids)


@dataclass
class Target:
    scores: pd.Series | None = None            # handed to the spec's construction method
    weights: pd.Series | None = None           # final weights (construction skipped)
    overlays: pd.Series | None = None          # overlay notionals as fractions of NAV
    skip: bool = False                         # keep the current portfolio
    note: str = ""


class Decider:
    """Base class: override ``decide``. ``params()`` (public attributes by default) is what the
    trial ledger records as the variant."""

    name: str = "decider"

    def decide(self, view: DecisionView) -> Target | pd.Series | None:
        raise NotImplementedError

    def params(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and isinstance(v, (int, float, str, bool, tuple, list, type(None)))}

    def fit(self, train: Any) -> Decider:
        """Walk-forward hook: return a fitted copy (default: no fitting)."""
        return self


@dataclass
class FieldSignal(Decider):
    """Scores = a field (sign-adjusted), optionally standardised within a group and neutralised
    against groups / numeric exposures: the common cross-sectional signal pipeline."""

    field_name: str = "signal.value"
    sign: float = 1.0
    zscore_by: str | None = None
    neutralise_groups: tuple[str, ...] = ()
    neutralise_exposures: tuple[str, ...] = ()
    winsor: float | None = 0.01
    name: str = "FieldSignal"

    def decide(self, view: DecisionView) -> Target:
        from nautilus_fi.construct.signals import neutralise, winsorise, zscore

        x = self.sign * view.field(self.field_name).reindex(view.universe)
        if self.winsor:
            x = winsorise(x, self.winsor, 1 - self.winsor)
        if self.neutralise_groups or self.neutralise_exposures:
            attrs = view.attributes.reindex(x.index)
            groups = attrs[list(self.neutralise_groups)] if self.neutralise_groups else None
            exps = view.fields(list(self.neutralise_exposures)).reindex(x.index) if self.neutralise_exposures else None
            x = neutralise(x, groups, exps)
        if self.zscore_by:
            x = zscore(x, view.attributes.reindex(x.index)[self.zscore_by])
        return Target(scores=x.dropna())


@dataclass
class UniverseRule:
    """Filters on the eligible universe: ``field_rules`` ``[(field, op, value)]`` on point-in-time
    fields, ``include`` / ``exclude`` on static attributes, ``require`` fields that must be present."""

    field_rules: list = field(default_factory=list)
    include: dict = field(default_factory=dict)
    exclude: dict = field(default_factory=dict)
    require: tuple[str, ...] = ()

    _OPS: ClassVar[dict] = {"<=": np.less_equal, "<": np.less, ">=": np.greater_equal, ">": np.greater, "==": np.equal, "!=": np.not_equal}

    def apply(self, view: DecisionView, ids: pd.Index) -> pd.Index:
        keep = pd.Series(True, index=ids)
        attrs = view.attributes.reindex(ids)
        for col, vals in self.include.items():
            keep &= attrs[col].isin(vals)
        for col, vals in self.exclude.items():
            keep &= ~attrs[col].isin(vals)
        for f, op, v in self.field_rules:
            x = view.field(f).reindex(ids).astype(float)
            keep &= pd.Series(self._OPS[op](x.to_numpy(), v), index=ids) & x.notna()
        for f in self.require:
            keep &= view.field(f).reindex(ids).notna()
        return ids[keep.to_numpy()]

    def to_dict(self) -> dict:
        return {"field_rules": [list(r) for r in self.field_rules], "include": self.include, "exclude": self.exclude, "require": list(self.require)}


@dataclass
class ResearchSpec:
    name: str
    decider: Decider
    construction: Any = None                   # construct.ConstructionSpec (None: Target.weights are final)
    conventions: Conventions = field(default_factory=Conventions)
    calendar: str | RebalanceCalendar | None = None
    universe: UniverseRule | None = None
    hypothesis_id: str | None = None
    params: dict = field(default_factory=dict)

    @property
    def rebalance(self) -> RebalanceCalendar:
        return parse_calendar(self.calendar or self.conventions.rebalance)

    def to_dict(self) -> dict:
        return {"name": self.name, "decider": {"class": type(self.decider).__name__, **self.decider.params()}, "construction": self.construction.to_dict() if self.construction is not None else None, "conventions": self.conventions.to_dict(), "calendar": str(self.rebalance), "universe": self.universe.to_dict() if self.universe is not None else None, "hypothesis_id": self.hypothesis_id, "params": self.params}

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def variant(self) -> dict:
        """Parameters that distinguish this trial from its siblings in the ledger."""
        return {**self.decider.params(), **self.params}


def as_target(out: Target | pd.Series | None, has_construction: bool) -> Target:
    if out is None:
        return Target(skip=True)
    if isinstance(out, Target):
        return out
    s = pd.Series(out, dtype=float)
    return Target(scores=s) if has_construction else Target(weights=s)


__all__ = ["Decider", "DecisionView", "FieldSignal", "ResearchSpec", "Target", "UniverseRule", "as_target"]
