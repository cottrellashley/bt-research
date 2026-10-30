"""Inflation-linked conventions (blueprint section 6): index lag and interpolation, known versus
projected CPI, indexed principal and contractual floors, per local convention rather than a global
'coupon plus inflation' shortcut."""

from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class IndexationConvention:
    name: str
    lag_months: int = 3
    daily_interpolation: bool = True           # Canadian method (US TIPS, UK post-2005, EUR)
    floor_at_maturity: bool = True             # principal floor at par (US TIPS, EUR OATei/BTPei); UK: no floor
    floor_coupons: bool = False
    ratio_decimals: int = 5
    publication_lag_days: int = 14             # CPI for month m is published ~ this many days into month m+1

    def publication_date(self, year: int, month: int) -> date:
        y, m = (year + 1, 1) if month == 12 else (year, month + 1)
        return date(y, m, min(self.publication_lag_days, _cal.monthrange(y, m)[1]))


US_TIPS = IndexationConvention("US_TIPS", 3, True, True, False, 5, 14)
UK_GILT_3M = IndexationConvention("UK_GILT_3M", 3, True, False, False, 5, 20)
UK_GILT_8M = IndexationConvention("UK_GILT_8M", 8, False, False, False, 5, 20)
EUR_OATEI = IndexationConvention("EUR_OATEI", 3, True, True, False, 5, 17)
CA_RRB = IndexationConvention("CA_RRB", 3, True, False, False, 5, 20)

CONVENTIONS = {c.name: c for c in (US_TIPS, UK_GILT_3M, UK_GILT_8M, EUR_OATEI, CA_RRB)}


def _shift_month(year: int, month: int, k: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + k
    return idx // 12, idx % 12 + 1


def reference_cpi(cpi: dict[tuple[int, int], float], d: date, conv: IndexationConvention, known_as_of: date | None = None, assumed_inflation: float = 0.02) -> tuple[float, str]:
    """Reference CPI for settlement/payment date ``d`` under the convention.

    Canadian method: linear interpolation between the CPI of month ``m - lag`` and ``m - lag + 1``
    using the day of month. Returns ``(value, status)`` where status is KNOWN or PROJECTED (a needed
    CPI print was not published by ``known_as_of``; it is projected at ``assumed_inflation``).
    """
    y0, m0 = _shift_month(d.year, d.month, -conv.lag_months)
    y1, m1 = _shift_month(y0, m0, 1)

    def value(y: int, m: int) -> tuple[float, bool]:
        known = (y, m) in cpi and (known_as_of is None or conv.publication_date(y, m) <= known_as_of)
        if known:
            return cpi[(y, m)], True
        # project from the latest known month
        known_months = sorted(k for k in cpi if known_as_of is None or conv.publication_date(*k) <= known_as_of)
        if not known_months:
            raise LookupError("no CPI history")
        ky, km = known_months[-1]
        months_ahead = (y * 12 + m) - (ky * 12 + km)
        return cpi[(ky, km)] * (1.0 + assumed_inflation) ** (months_ahead / 12.0), False

    v0, k0 = value(y0, m0)
    if not conv.daily_interpolation:
        return v0, "KNOWN" if k0 else "PROJECTED"
    v1, k1 = value(y1, m1)
    days_in_month = _cal.monthrange(d.year, d.month)[1]
    ref = v0 + (d.day - 1) / days_in_month * (v1 - v0)
    return ref, "KNOWN" if (k0 and k1) else "PROJECTED"


def index_ratio(cpi: dict[tuple[int, int], float], d: date, base_cpi: float, conv: IndexationConvention, known_as_of: date | None = None, assumed_inflation: float = 0.02) -> tuple[float, str]:
    ref, status = reference_cpi(cpi, d, conv, known_as_of, assumed_inflation)
    return round(ref / base_cpi, conv.ratio_decimals), status


def base_cpi_for(cpi: dict[tuple[int, int], float], dated_date: date, conv: IndexationConvention) -> float:
    ref, _ = reference_cpi(cpi, dated_date, conv)
    return ref


def redemption_per_100(ratio_at_maturity: float, conv: IndexationConvention) -> float:
    """Indexed principal at maturity, floored at par where the contract says so."""
    return 100.0 * (max(ratio_at_maturity, 1.0) if conv.floor_at_maturity else ratio_at_maturity)


def coupon_per_100(coupon_rate: float, frequency: int, ratio_at_payment: float, conv: IndexationConvention) -> float:
    base = max(ratio_at_payment, 1.0) if conv.floor_coupons else ratio_at_payment
    return coupon_rate / frequency * 100.0 * base


def real_price_to_nominal(real_clean: float, ratio: float) -> float:
    return real_clean * ratio


__all__ = ["CA_RRB", "CONVENTIONS", "EUR_OATEI", "UK_GILT_3M", "UK_GILT_8M", "US_TIPS", "IndexationConvention", "base_cpi_for", "coupon_per_100", "index_ratio", "real_price_to_nominal", "redemption_per_100", "reference_cpi"]
_ = timedelta
