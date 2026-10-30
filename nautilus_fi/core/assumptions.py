"""Governed assumptions (blueprint section 13): every assumption has a name, unit, scope, effective
dates, source, uncertainty range, dependencies and owner, with explicit precedence between system
defaults, product defaults, experiment overrides and instrument overrides. The resolved set is
exported with every run; an unseen default can be more consequential than the chosen model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum

import pandas as pd


class AssumptionLevel(IntEnum):
    SYSTEM_DEFAULT = 0
    PRODUCT_DEFAULT = 1
    EXPERIMENT_OVERRIDE = 2
    INSTRUMENT_OVERRIDE = 3


@dataclass(frozen=True)
class Assumption:
    name: str
    value: object
    unit: str
    level: AssumptionLevel
    scope: str = "*"                      # "*", "product:CORPORATE", "instrument:XS123", "currency:EUR"
    effective_from: date | None = None
    effective_to: date | None = None
    source: str = ""
    rationale: str = ""
    low: object | None = None             # uncertainty range
    high: object | None = None
    depends_on: tuple[str, ...] = ()
    owner: str = ""

    def applies(self, scope_tags: set[str], as_of: date | None) -> bool:
        if self.scope != "*" and self.scope not in scope_tags:
            return False
        if as_of is not None:
            if self.effective_from and as_of < self.effective_from:
                return False
            if self.effective_to and as_of > self.effective_to:
                return False
        return True


class AssumptionSet:
    def __init__(self) -> None:
        self._items: list[Assumption] = []
        self._used: dict[str, Assumption] = {}

    def add(self, a: Assumption) -> Assumption:
        self._items.append(a)
        return a

    def default(self, name: str, value, unit: str, source: str = "", rationale: str = "", low=None, high=None, owner: str = "") -> Assumption:
        return self.add(Assumption(name, value, unit, AssumptionLevel.SYSTEM_DEFAULT, "*", None, None, source, rationale, low, high, (), owner))

    def product_default(self, product: str, name: str, value, unit: str, **kw) -> Assumption:
        return self.add(Assumption(name, value, unit, AssumptionLevel.PRODUCT_DEFAULT, f"product:{product}", **kw))

    def override(self, name: str, value, unit: str, scope: str = "*", **kw) -> Assumption:
        return self.add(Assumption(name, value, unit, AssumptionLevel.EXPERIMENT_OVERRIDE, scope, **kw))

    def instrument_override(self, instrument: str, name: str, value, unit: str, **kw) -> Assumption:
        return self.add(Assumption(name, value, unit, AssumptionLevel.INSTRUMENT_OVERRIDE, f"instrument:{instrument}", **kw))

    def resolve(self, name: str, scope_tags: set[str] | None = None, as_of: date | None = None, default=None) -> Assumption | None:
        tags = scope_tags or set()
        candidates = [a for a in self._items if a.name == name and a.applies(tags, as_of)]
        if not candidates:
            if default is not None:
                return Assumption(name, default, "", AssumptionLevel.SYSTEM_DEFAULT, "*", rationale="implicit default (undeclared!)")
            return None
        best = max(candidates, key=lambda a: (int(a.level), 0 if a.scope == "*" else 1))
        self._used[f"{name}|{best.scope}"] = best
        return best

    def value(self, name: str, scope_tags: set[str] | None = None, as_of: date | None = None, default=None):
        a = self.resolve(name, scope_tags, as_of, default)
        return a.value if a is not None else default

    def export_resolved(self) -> pd.DataFrame:
        rows = [{"name": a.name, "value": a.value, "unit": a.unit, "level": a.level.name, "scope": a.scope, "effective_from": a.effective_from, "effective_to": a.effective_to, "source": a.source, "rationale": a.rationale, "low": a.low, "high": a.high, "depends_on": ",".join(a.depends_on), "owner": a.owner} for a in self._items]
        return pd.DataFrame(rows)

    def used(self) -> pd.DataFrame:
        rows = [{"name": a.name, "value": a.value, "unit": a.unit, "level": a.level.name, "scope": a.scope, "source": a.source} for a in self._used.values()]
        return pd.DataFrame(rows)

    def inventory(self, impact: dict[str, float] | None = None) -> pd.DataFrame:
        """Assumption inventory ranked by declared economic impact (section 21)."""
        df = self.export_resolved()
        if df.empty:
            return df
        df["impact"] = df["name"].map(impact or {}).fillna(0.0)
        df["has_range"] = df["low"].notna() & df["high"].notna()
        return df.sort_values(["impact", "name"], ascending=[False, True]).reset_index(drop=True)


__all__ = ["Assumption", "AssumptionLevel", "AssumptionSet"]
_ = field
