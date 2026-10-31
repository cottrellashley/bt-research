"""Point-in-time universe construction.

The universe for a date is rebuilt from issuance / maturity / corporate-action data and point-in-time
ratings, never from today's index membership or screens. Dead bonds (matured, called, defaulted) stay
in the environment with their final cashflows and recovery prices.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.enums import AssetType


@dataclass
class UniverseFilter:
    asset_types: set[AssetType] | None = None
    currencies: set[str] | None = None
    min_rating_score: int | None = None       # best rating allowed (1 = AAA)
    max_rating_score: int | None = None       # worst rating allowed (10 = BBB-)
    min_years_to_maturity: float | None = None
    max_years_to_maturity: float | None = None
    min_amount_outstanding: float | None = None
    min_age_days: int = 0                      # exclude brand-new issues until seasoned
    require_mark: bool = True
    exclude_defaulted: bool = True
    exclude_called: bool = True
    custom: Callable[[BondSpec, date], bool] | None = None


@dataclass
class PointInTimeUniverse:
    env: object
    membership: dict[date, set[str]] | None = None   # optional PIT index membership snapshots
    _membership_dates: list[date] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.membership:
            self._membership_dates = sorted(self.membership)

    def _in_index(self, symbol: str, d: date) -> bool:
        if not self.membership:
            return True
        i = bisect_right(self._membership_dates, d)
        if i == 0:
            return False
        return symbol in self.membership[self._membership_dates[i - 1]]

    @staticmethod
    def from_weights(env, panel, field_name: str = "benchmark.weight") -> PointInTimeUniverse:
        """Membership snapshots from a daily benchmark weight field (one snapshot per change)."""
        W = panel[field_name] > 0
        memb: dict[date, set[str]] = {}
        prev = None
        for i, d in enumerate(panel.dates.date):
            row = W[i]
            if prev is None or (row != prev).any():
                memb[d] = set(panel.ids[row])
                prev = row
        u = PointInTimeUniverse(env, memb)
        return u

    def members(self, d: date, filt: UniverseFilter | None = None) -> list[BondSpec]:
        filt = filt or UniverseFilter()
        out = []
        for symbol, bond in self.env.bonds.items():
            if not bond.is_alive(d) or (d - bond.issue_date).days < filt.min_age_days:
                continue
            if not self._in_index(symbol, d):
                continue
            if filt.exclude_defaulted and self.env.is_defaulted(symbol, d):
                continue
            if filt.exclude_called and self.env.is_terminated(symbol, d):
                continue
            if filt.asset_types and bond.asset_type not in filt.asset_types:
                continue
            if filt.currencies and bond.currency not in filt.currencies:
                continue
            score = self.env.rating_score(symbol, d)
            if filt.min_rating_score is not None and (score is None or score < filt.min_rating_score):
                continue
            if filt.max_rating_score is not None and (score is None or score > filt.max_rating_score):
                continue
            yrs = bond.remaining_years(d)
            if filt.min_years_to_maturity is not None and yrs is not None and yrs < filt.min_years_to_maturity:
                continue
            if filt.max_years_to_maturity is not None and yrs is not None and yrs > filt.max_years_to_maturity:
                continue
            if filt.min_amount_outstanding is not None and (bond.amount_outstanding or 0) < filt.min_amount_outstanding:
                continue
            if filt.require_mark and self.env.mark(symbol, d) is None:
                continue
            if filt.custom is not None and not filt.custom(bond, d):
                continue
            out.append(bond)
        return out

    def symbols(self, d: date, filt: UniverseFilter | None = None) -> list[str]:
        return [b.symbol for b in self.members(d, filt)]

    @staticmethod
    def from_symbols(env, symbols: Iterable[str]) -> PointInTimeUniverse:
        u = PointInTimeUniverse(env)
        keep = set(symbols)
        u.membership = {date.min: keep}
        u._membership_dates = [date.min]
        return u


__all__ = ["PointInTimeUniverse", "UniverseFilter"]
