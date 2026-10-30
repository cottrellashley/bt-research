"""Identifier history: CUSIP/ISIN changes through time (exchange offers, 144A -> registered swaps)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class IdentifierChange:
    old: str
    new: str
    effective: date
    reason: str = ""


@dataclass
class IdentifierHistory:
    changes: list[IdentifierChange] = field(default_factory=list)

    def add(self, old: str, new: str, effective: date, reason: str = "") -> None:
        self.changes.append(IdentifierChange(old, new, effective, reason))

    def resolve(self, identifier: str, as_of: date) -> str:
        """Follow identifier changes effective on or before ``as_of``."""
        current = identifier
        changed = True
        while changed:
            changed = False
            for c in sorted(self.changes, key=lambda c: c.effective):
                if c.old == current and c.effective <= as_of:
                    current = c.new
                    changed = True
        return current

    def canonical(self, identifier: str) -> str:
        """Resolve to the final identifier regardless of date (for joining history onto one symbol)."""
        return self.resolve(identifier, date.max)


__all__ = ["IdentifierChange", "IdentifierHistory"]
