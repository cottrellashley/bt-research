"""Effective-dated contract terms (blueprint section 9): an effective-time query asks which terms
applied on date t; an information-time query asks which version the actor could know. Amendments
are stored as versions with announcement and effective dates and never overwrite the base record."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

from nautilus_fi.instruments.bond import BondSpec


@dataclass(frozen=True)
class Amendment:
    effective_date: date
    announce_date: date
    changes: dict
    source: str = ""
    note: str = ""


@dataclass
class TermsHistory:
    base: BondSpec
    amendments: list[Amendment] = field(default_factory=list)

    def amend(self, effective_date: date, announce_date: date | None = None, source: str = "", note: str = "", **changes) -> Amendment:
        a = Amendment(effective_date, announce_date or effective_date, dict(changes), source, note)
        self.amendments.append(a)
        return a

    def _apply(self, items: list[Amendment]) -> BondSpec:
        spec = self.base
        for a in sorted(items, key=lambda x: x.effective_date):
            changes = {k: v for k, v in a.changes.items() if hasattr(spec, k)}
            spec = replace(spec, **changes)
            if "schedule" in spec.__dict__:
                spec.__dict__.pop("schedule", None)  # drop cached schedule after a term change
        return spec

    def as_of(self, effective: date) -> BondSpec:
        """Terms in force on ``effective`` (effective-time query)."""
        return self._apply([a for a in self.amendments if a.effective_date <= effective])

    def known_as_of(self, info_date: date, effective: date | None = None) -> BondSpec:
        """Terms the actor could know on ``info_date`` for the effective date (information-time query)."""
        eff = effective or info_date
        return self._apply([a for a in self.amendments if a.announce_date <= info_date and a.effective_date <= eff])

    def pending(self, info_date: date) -> list[Amendment]:
        return [a for a in self.amendments if a.announce_date <= info_date < a.effective_date]


__all__ = ["Amendment", "TermsHistory"]
