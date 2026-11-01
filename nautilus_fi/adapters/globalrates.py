"""globalrates: government bond and rates research across countries (gap analysis §5.9; read
from a blurry photo).

Canonical conventions (no legacy profile). The components:

* ``constant_maturity``: constant-maturity returns from yields;
* ``carry_rolldown``: carry and roll-down at a tenor and horizons;
* ``fx_hedged``: returns hedged into a base currency with rolled forwards;
* ``country_calendar``: the holiday calendar of a market (US bond market, TARGET for the euro
  area, the UK).
"""

from __future__ import annotations

from nautilus_fi.adapters.parity import ParityCase, Tolerance

PROFILE = None
REQUIREMENTS = ("constant-maturity returns", "carry", "roll-down", "FX-hedged returns", "per-country calendars")
_COUNTRY = {"US": "US_BOND", "USA": "US_BOND", "GB": "UK", "UK": "UK", "EU": "TARGET", "EUR": "TARGET", **{c: "TARGET" for c in ("DE", "FR", "IT", "ES", "NL", "BE", "AT", "FI", "IE", "PT", "GR")}}


def constant_maturity(yields, freq: int = 2):
    from nautilus_fi.kernel.returns import constant_maturity_returns

    return constant_maturity_returns(yields, freq)


def carry_rolldown(yields, tenor: float, horizons_months=(1, 3)):
    from nautilus_fi.kernel.returns import carry_rolldown as _cr

    return _cr(yields, tenor, horizons_months)


def fx_hedged(local, spot, forward):
    from nautilus_fi.kernel.returns import fx_hedged_returns

    return fx_hedged_returns(local, spot, forward)


def country_calendar(country: str):
    """The market's holiday calendar: ``US_BOND``, ``TARGET`` (euro-area government bonds) or
    ``UK``; another market raises (register it with ``instruments.calendar.register_calendar``)."""
    from nautilus_fi.instruments.calendar import get_calendar

    key = country.upper()
    name = _COUNTRY.get(key, key)
    return get_calendar(name)


PARITY = [ParityCase("globalrates", "globalrates", "not named in the gap analysis: the framework's stored country returns", (Tolerance("returns", bp_per_year=5.0, min_corr=0.999),))]

__all__ = ["PARITY", "PROFILE", "REQUIREMENTS", "carry_rolldown", "constant_maturity", "country_calendar", "fx_hedged"]
