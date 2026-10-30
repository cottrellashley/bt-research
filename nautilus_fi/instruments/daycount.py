"""Day count conventions."""

from __future__ import annotations

from datetime import date

from nautilus_fi.instruments.enums import DayCount, Frequency


def _is_leap(y: int) -> bool:
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _days_in_year(y: int) -> int:
    return 366 if _is_leap(y) else 365


def _thirty_360_us(start: date, end: date) -> int:
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def _thirty_e_360(start: date, end: date) -> int:
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def year_fraction(
    start: date,
    end: date,
    convention: DayCount,
    ref_start: date | None = None,
    ref_end: date | None = None,
    frequency: Frequency | int | None = None,
) -> float:
    """Accrual year fraction between ``start`` and ``end``.

    For ``ACT/ACT ICMA`` the reference period (``ref_start``, ``ref_end``) and coupon ``frequency``
    are required: the fraction is ``days / (freq * days_in_reference_period)``.
    """
    if end <= start:
        return 0.0
    if convention == DayCount.THIRTY_360:
        return _thirty_360_us(start, end) / 360.0
    if convention == DayCount.THIRTY_E_360:
        return _thirty_e_360(start, end) / 360.0
    if convention == DayCount.ACT_360:
        return (end - start).days / 360.0
    if convention == DayCount.ACT_365F:
        return (end - start).days / 365.0
    if convention == DayCount.ACT_ACT_ISDA:
        if start.year == end.year:
            return (end - start).days / _days_in_year(start.year)
        total = (date(start.year + 1, 1, 1) - start).days / _days_in_year(start.year)
        total += float(end.year - start.year - 1)
        total += (end - date(end.year, 1, 1)).days / _days_in_year(end.year)
        return total
    if convention == DayCount.ACT_ACT_ICMA:
        if ref_start is None or ref_end is None or not frequency:
            raise ValueError("ACT/ACT ICMA requires the reference period and coupon frequency")
        period_days = (ref_end - ref_start).days
        if period_days <= 0:
            raise ValueError("Invalid ACT/ACT ICMA reference period")
        return (end - start).days / (int(frequency) * period_days)
    raise ValueError(f"Unsupported day count {convention}")
