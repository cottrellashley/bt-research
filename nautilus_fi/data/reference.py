"""Reference data (bond terms) <-> DataFrame conversion."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from nautilus_fi.data.schemas import REFERENCE
from nautilus_fi.instruments.bond import AmortizationTerms, BondSpec, CallOption, FloatingTerms, Identifiers
from nautilus_fi.instruments.enums import (
    AssetType,
    BusinessDayConvention,
    CouponType,
    DayCount,
    Frequency,
    PrincipalType,
    Seniority,
)


def _to_date(v) -> date | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if isinstance(v, date) and not isinstance(v, pd.Timestamp):
        return v
    return pd.Timestamp(v).date()


def _schedule(v) -> list[CallOption]:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" or (isinstance(v, (list, tuple)) and not v):
        return []
    if isinstance(v, str):
        v = json.loads(v)
    out = []
    for item in v:
        if isinstance(item, dict):
            out.append(CallOption(_to_date(item["date"]), float(item.get("price", 100.0))))
        else:
            d, px = item
            out.append(CallOption(_to_date(d), float(px)))
    return out


def _enum(cls, v, default):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return default
    if isinstance(v, cls):
        return v
    try:
        return cls(v)
    except ValueError:
        return cls[str(v).upper()]


def _frequency(v) -> Frequency:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return Frequency.SEMI_ANNUAL
    if isinstance(v, Frequency):
        return v
    if isinstance(v, str) and not v.isdigit():
        return Frequency[v.upper()]
    return Frequency(int(v))


def bond_from_row(row: pd.Series | dict) -> BondSpec:
    r = dict(row)
    g = r.get
    coupon_type = _enum(CouponType, g("coupon_type"), CouponType.FIXED)
    principal_type = _enum(PrincipalType, g("principal_type"), PrincipalType.BULLET)
    floating = None
    if coupon_type == CouponType.FLOATING:
        floating = FloatingTerms(
            index=g("float_index") or "SOFR",
            spread_bp=float(g("float_spread_bp") or 0.0),
            fixing_lag_days=int(g("float_reset_lag_days") or 2),
        )
    amort = None
    if principal_type in (PrincipalType.AMORTIZING, PrincipalType.SINKING):
        sink = g("sinking_schedule")
        sink_list = []
        if isinstance(sink, str) and sink:
            sink_list = [(_to_date(d), float(f)) for d, f in json.loads(sink)]
        elif isinstance(sink, (list, tuple)):
            sink_list = [(_to_date(d), float(f)) for d, f in sink]
        term = g("original_term_months")
        amort = AmortizationTerms(
            sinking_schedule=sink_list,
            original_term_months=int(term) if term not in (None, "") and not pd.isna(term) else None,
            projected_cpr=float(g("projected_cpr")) if g("projected_cpr") not in (None, "") and not pd.isna(g("projected_cpr")) else None,
        )
    lag = g("settlement_lag_days")
    terms_source = str(g("terms_source") or "TERMS").upper()
    if terms_source not in ("TERMS", "VENDOR"):
        raise ValueError(f"{r['symbol']}: terms_source must be TERMS or VENDOR")
    issue = _to_date(r.get("issue_date"))
    if issue is None:
        if terms_source != "VENDOR":
            raise ValueError(f"{r['symbol']}: issue_date is required unless terms_source is VENDOR")
        issue = date(1900, 1, 1)                          # terms-free: existence comes from vendor data
    return BondSpec(
        terms_source=terms_source,
        symbol=str(r["symbol"]),
        issuer=str(r["issuer"]),
        issue_date=issue,
        maturity=_to_date(g("maturity")),
        coupon_rate=float(g("coupon_rate") or 0.0),
        coupon_type=coupon_type,
        frequency=_frequency(g("frequency")),
        day_count=_enum(DayCount, g("day_count"), DayCount.THIRTY_360),
        currency=str(g("currency") or "USD"),
        asset_type=_enum(AssetType, g("asset_type"), AssetType.CORPORATE),
        principal_type=principal_type,
        name=str(g("name") or ""),
        sector=str(g("sector") or ""),
        country=str(g("country") or "US"),
        seniority=_enum(Seniority, g("seniority"), Seniority.SENIOR_UNSECURED),
        rating_initial=(None if g("rating_initial") in (None, "") or (isinstance(g("rating_initial"), float) and pd.isna(g("rating_initial"))) else str(g("rating_initial"))),
        dated_date=_to_date(g("dated_date")),
        first_coupon_date=_to_date(g("first_coupon_date")),
        business_day_convention=_enum(BusinessDayConvention, g("business_day_convention"), BusinessDayConvention.FOLLOWING),
        calendar_name=str(g("calendar_name") or "US_BOND"),
        settlement_lag_days=int(lag) if lag not in (None, "") and not pd.isna(lag) else None,
        min_piece=float(g("min_piece") or 2_000.0),
        increment=float(g("increment") or 1_000.0),
        amount_outstanding=float(g("amount_outstanding")) if g("amount_outstanding") not in (None, "") and not pd.isna(g("amount_outstanding")) else None,
        is_144a=bool(g("is_144a") or False),
        is_benchmark=bool(g("is_benchmark") or False),
        floating=floating,
        amortization=amort,
        call_schedule=_schedule(g("call_schedule")),
        put_schedule=_schedule(g("put_schedule")),
        identifiers=Identifiers(cusip=g("cusip"), isin=g("isin"), figi=g("figi"), bbg_id=g("bbg_id"), ticker=g("ticker")),
    )


def reference_from_dataframe(df: pd.DataFrame) -> dict[str, BondSpec]:
    REFERENCE.validate(df)
    return {str(row["symbol"]): bond_from_row(row) for _, row in df.iterrows()}


def reference_to_dataframe(bonds: dict[str, BondSpec] | list[BondSpec]) -> pd.DataFrame:
    items = bonds.values() if isinstance(bonds, dict) else bonds
    rows = []
    for b in items:
        rows.append({
            "symbol": b.symbol, "issuer": b.issuer, "name": b.name, "issue_date": b.issue_date, "maturity": b.maturity,
            "coupon_rate": b.coupon_rate, "coupon_type": b.coupon_type.value, "frequency": int(b.frequency),
            "day_count": b.day_count.value, "currency": b.currency, "asset_type": b.asset_type.value,
            "principal_type": b.principal_type.value, "sector": b.sector, "country": b.country,
            "seniority": b.seniority.value, "rating_initial": b.rating_initial, "min_piece": b.min_piece,
            "increment": b.increment, "amount_outstanding": b.amount_outstanding, "is_144a": b.is_144a,
            "is_benchmark": b.is_benchmark,
            "float_index": b.floating.index if b.floating else None,
            "float_spread_bp": b.floating.spread_bp if b.floating else None,
            "call_schedule": json.dumps([[c.date.isoformat(), c.price] for c in b.call_schedule]) if b.call_schedule else None,
            "original_term_months": b.amortization.original_term_months if b.amortization else None,
            "projected_cpr": b.amortization.projected_cpr if b.amortization else None,
            "settlement_lag_days": b.settlement_lag_days, "calendar_name": b.calendar_name,
            "cusip": b.identifiers.cusip, "isin": b.identifiers.isin,
        })
    return pd.DataFrame(rows)


__all__ = ["bond_from_row", "reference_from_dataframe", "reference_to_dataframe"]
