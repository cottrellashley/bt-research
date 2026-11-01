"""Business day calendars.

A deliberately small implementation: weekends plus an explicit holiday set, with generators for the
US bond market (SIFMA-style), TARGET (euro-area settlement) and UK (England and Wales bank holiday)
calendars. Supply your own holiday lists for other markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from nautilus_fi.instruments.enums import BusinessDayConvention


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month // 12), (month % 12) + 1, 1) - timedelta(days=1)
    offset = (d.weekday() - weekday) % 7
    return d - timedelta(days=offset)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def us_bond_market_holidays(year: int) -> set[date]:
    """Approximate SIFMA US bond market full-closure holidays for a year."""
    hol = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),            # MLK
        _nth_weekday(year, 2, 0, 3),            # Presidents' Day
        _easter(year) - timedelta(days=2),      # Good Friday
        _last_weekday(year, 5, 0),              # Memorial Day
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),            # Labor Day
        _nth_weekday(year, 10, 0, 2),           # Columbus Day
        _observed(date(year, 11, 11)),          # Veterans Day
        _nth_weekday(year, 11, 3, 4),           # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))   # Juneteenth
    return hol


def target_holidays(year: int) -> set[date]:
    """TARGET2 closing days (euro-area payments and government bond settlement): New Year,
    Good Friday, Easter Monday, 1 May, Christmas Day and 26 December."""
    e = _easter(year)
    return {date(year, 1, 1), e - timedelta(days=2), e + timedelta(days=1), date(year, 5, 1), date(year, 12, 25), date(year, 12, 26)}


_UK_MOVED = {1995: {"early_may": date(1995, 5, 8)}, 2002: {"spring": date(2002, 6, 4)}, 2012: {"spring": date(2012, 6, 4)}, 2020: {"early_may": date(2020, 5, 8)}, 2022: {"spring": date(2022, 6, 2)}}
_UK_EXTRA = {date(1999, 12, 31), date(2002, 6, 3), date(2011, 4, 29), date(2012, 6, 5), date(2022, 6, 3), date(2022, 9, 19), date(2023, 5, 8)}


def uk_bank_holidays(year: int) -> set[date]:
    """England and Wales bank holidays, with weekend substitutes and the one-off days since 1995
    (jubilees, royal weddings and funerals, the coronation)."""
    e = _easter(year)
    moved = _UK_MOVED.get(year, {})
    ny = date(year, 1, 1)
    ny = ny + timedelta(days=(7 - ny.weekday()) % 7) if ny.weekday() >= 5 else ny
    xmas, boxing = date(year, 12, 25), date(year, 12, 26)
    if xmas.weekday() == 5:                                       # Saturday: Monday and Tuesday
        xmas, boxing = date(year, 12, 27), date(year, 12, 28)
    elif xmas.weekday() == 6:                                     # Sunday: Boxing Day Monday, Christmas Tuesday
        xmas = date(year, 12, 27)
    elif boxing.weekday() == 5:                                   # Christmas Friday: Boxing Day Monday
        boxing = date(year, 12, 28)
    hol = {ny, e - timedelta(days=2), e + timedelta(days=1), moved.get("early_may", _nth_weekday(year, 5, 0, 1)), moved.get("spring", _last_weekday(year, 5, 0)), _last_weekday(year, 8, 0), xmas, boxing}
    return hol | {d for d in _UK_EXTRA if d.year == year}


@dataclass
class BondCalendar:
    """Weekend + holiday calendar with business day adjustment and arithmetic."""

    name: str = "WEEKENDS"
    holidays: set[date] = field(default_factory=set)

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def adjust(self, d: date, convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING) -> date:
        if convention == BusinessDayConvention.UNADJUSTED or self.is_business_day(d):
            return d
        if convention == BusinessDayConvention.PRECEDING:
            return self.previous_business_day(d)
        following = self.next_business_day(d)
        if convention == BusinessDayConvention.MODIFIED_FOLLOWING and following.month != d.month:
            return self.previous_business_day(d)
        return following

    def next_business_day(self, d: date) -> date:
        d = d + timedelta(days=1)
        while not self.is_business_day(d):
            d += timedelta(days=1)
        return d

    def previous_business_day(self, d: date) -> date:
        d = d - timedelta(days=1)
        while not self.is_business_day(d):
            d -= timedelta(days=1)
        return d

    def add_business_days(self, d: date, n: int) -> date:
        step = 1 if n >= 0 else -1
        remaining = abs(n)
        while remaining > 0:
            d += timedelta(days=step)
            if self.is_business_day(d):
                remaining -= 1
        return d

    def business_days_between(self, start: date, end: date) -> int:
        """Number of business days in (start, end]."""
        if end < start:
            return -self.business_days_between(end, start)
        n = 0
        d = start
        while d < end:
            d += timedelta(days=1)
            if self.is_business_day(d):
                n += 1
        return n

    def business_days(self, start: date, end: date) -> list[date]:
        """All business days in [start, end]."""
        out = []
        d = start
        while d <= end:
            if self.is_business_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out


def us_bond_market_calendar(start_year: int = 1990, end_year: int = 2060) -> BondCalendar:
    hol: set[date] = set()
    for y in range(start_year, end_year + 1):
        hol |= us_bond_market_holidays(y)
    return BondCalendar(name="US_BOND", holidays=hol)


def target_calendar(start_year: int = 1999, end_year: int = 2060) -> BondCalendar:
    return BondCalendar(name="TARGET", holidays=set().union(*[target_holidays(y) for y in range(start_year, end_year + 1)]))


def uk_calendar(start_year: int = 1990, end_year: int = 2060) -> BondCalendar:
    return BondCalendar(name="UK", holidays=set().union(*[uk_bank_holidays(y) for y in range(start_year, end_year + 1)]))


_CALENDARS: dict[str, BondCalendar] = {}


def get_calendar(name: str | None) -> BondCalendar:
    """Registry lookup: ``US_BOND`` (SIFMA-style), ``TARGET``, ``UK``, ``WEEKENDS`` or a
    user-registered calendar."""
    key = (name or "WEEKENDS").upper()
    if key not in _CALENDARS:
        if key == "US_BOND":
            _CALENDARS[key] = us_bond_market_calendar()
        elif key == "TARGET":
            _CALENDARS[key] = target_calendar()
        elif key == "UK":
            _CALENDARS[key] = uk_calendar()
        elif key == "WEEKENDS":
            _CALENDARS[key] = BondCalendar()
        else:
            raise KeyError(f"Unknown calendar {name!r}; register it with register_calendar()")
    return _CALENDARS[key]


def register_calendar(calendar: BondCalendar) -> None:
    _CALENDARS[calendar.name.upper()] = calendar
