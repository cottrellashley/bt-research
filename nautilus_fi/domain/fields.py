"""Field definitions: every input a strategy, a construction method or a report can use is a named
field with units, frequency, availability and fill rules, wherever it comes from (vendor analytics,
computed analytics, index weights, ratings, model scores).

Built-in definitions cover the fixed income vocabulary; anything else (``signal.*``, ``model.*``)
is registered on the fly. The ``availability_lag_days`` of a field is how many trading days after
its date a value may be used: vendor analytics published the next morning have lag 1, so a decision
on day T sees the analytics dated T-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnalyticsSource(str, Enum):
    VENDOR = "VENDOR"                          # vendor analytics only (missing stays missing)
    COMPUTED = "COMPUTED"                      # computed from terms and marks only (today's behaviour)
    VENDOR_ELSE_COMPUTED = "VENDOR_ELSE_COMPUTED"
    COMPARE = "COMPARE"                        # compute both, use vendor, report the differences


@dataclass(frozen=True)
class FieldDef:
    name: str
    unit: str = ""
    kind: str = "numeric"                      # numeric | category | flag
    frequency: str = "D"
    availability_lag_days: int = 0
    fill: str = "none"                         # none | ffill
    ffill_limit: int | None = None
    description: str = ""


_BUILTIN = [
    FieldDef("tr", "fraction", description="total return over the period ending on the date"),
    FieldDef("xr", "fraction", description="excess return over duration-matched Treasuries (vendor definition)"),
    FieldDef("tsy_r", "fraction", description="return of the duration / key-rate matched Treasury portfolio"),
    FieldDef("price_clean", "per 100", fill="ffill", ffill_limit=5),
    FieldDef("price_dirty", "per 100", fill="ffill", ffill_limit=5),
    FieldDef("accrued", "per 100"),
    FieldDef("bid", "per 100"),
    FieldDef("ask", "per 100"),
    FieldDef("half_spread_bp", "bp", description="half bid/ask spread in bp of price"),
    FieldDef("oas_bp", "bp", description="option-adjusted spread (z-spread for bullets)"),
    FieldDef("oad", "years", description="option-adjusted (modified) duration"),
    FieldDef("oasd", "years", description="option-adjusted spread duration"),
    FieldDef("dts", "years x bp", description="duration times spread (oasd x oas)"),
    FieldDef("ytm", "fraction"),
    FieldDef("ytw", "fraction"),
    FieldDef("convexity", ""),
    FieldDef("krd_0.5", "years"), FieldDef("krd_1", "years"), FieldDef("krd_2", "years"), FieldDef("krd_3", "years"), FieldDef("krd_5", "years"),
    FieldDef("krd_7", "years"), FieldDef("krd_10", "years"), FieldDef("krd_20", "years"), FieldDef("krd_30", "years"),
    FieldDef("amt_out", "face", fill="ffill"),
    FieldDef("rating_score", "score", kind="category", fill="ffill", description="1 = AAA ... 22 = D"),
    FieldDef("index_weight", "fraction", description="benchmark weight at the date"),
    FieldDef("member", "flag", kind="flag", description="benchmark membership flag"),
    FieldDef("years_to_maturity", "years"),
    FieldDef("dv01", "per 100 per bp"),
]
FIELD_REGISTRY: dict[str, FieldDef] = {f.name: f for f in _BUILTIN}

# canonical names for the analytics carried on BondMark / BondAnalytics
MARK_FIELD_ALIASES = {"z_spread_bp": "oas_bp", "mod_duration": "oad", "spread_duration": "oasd", "ytm": "ytm", "clean_mid": "price_clean"}


def field_def(name: str) -> FieldDef:
    """Definition of ``name``; unknown names (signals, model outputs) get a numeric default."""
    return FIELD_REGISTRY.get(name) or FieldDef(name)


def register_field(fd: FieldDef) -> FieldDef:
    FIELD_REGISTRY[fd.name] = fd
    return fd


__all__ = ["FIELD_REGISTRY", "MARK_FIELD_ALIASES", "AnalyticsSource", "FieldDef", "field_def", "register_field"]
