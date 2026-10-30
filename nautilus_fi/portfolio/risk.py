"""Product-matched risk: rates and curve (DV01, key rates), credit spread (CS01, DTS), default
(jump to default), inflation, FX, convexity, prepayment / extension, liquidity (days to liquidate),
funding and counterparty concentration, with full revaluation for large shocks. Standalone and
netted measures are both kept."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RiskReport:
    nav: float
    dv01: float
    cs01: float
    dts: float
    convexity_pnl_100bp: float
    key_rate_dv01: dict[str, float]
    fx_delta: dict[str, float]
    jump_to_default: float
    inflation_dv01: float
    prepayment_extension: float
    days_to_liquidate_p90: float
    funding_need: float
    counterparty_exposure: dict[str, float]
    standalone_total: float
    netted_total: float
    by_instrument: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_series(self) -> pd.Series:
        return pd.Series({"nav": self.nav, "dv01": self.dv01, "cs01": self.cs01, "dts": self.dts, "convexity_pnl_100bp": self.convexity_pnl_100bp, "jump_to_default": self.jump_to_default, "inflation_dv01": self.inflation_dv01, "prepayment_extension": self.prepayment_extension, "days_to_liquidate_p90": self.days_to_liquidate_p90, "funding_need": self.funding_need, "standalone_total": self.standalone_total, "netted_total": self.netted_total, **{f"krd_{k}": v for k, v in self.key_rate_dv01.items()}, **{f"fx_{k}": v for k, v in self.fx_delta.items()}})


def compute_risk(rows: pd.DataFrame, nav: float, liquidity_daily_capacity: dict[str, float] | None = None, netting_sets: dict[str, str] | None = None) -> RiskReport:
    """``rows`` columns: symbol, market_value, duration, spread_duration, z_spread_bp, convexity, currency,
    rating_bucket (for JTD), recovery, is_linker, real_duration, prepay_sensitivity (optional), krd_<tenor> (optional)."""
    if rows is None or rows.empty:
        return RiskReport(nav, 0, 0, 0, 0, {}, {}, 0, 0, 0, 0, 0, {}, 0, 0)
    r = rows.copy().fillna({"duration": 0.0, "spread_duration": 0.0, "z_spread_bp": 0.0, "convexity": 0.0, "recovery": 0.4})
    mv = r["market_value"].astype(float)
    r["dv01"] = mv * r["duration"] * 1e-4
    r["cs01"] = mv * r["spread_duration"] * 1e-4
    r["dts_contrib"] = (mv / nav) * r["spread_duration"] * r["z_spread_bp"] if nav else 0.0
    krd = {c[4:]: float((mv * r[c].fillna(0.0) * 1e-4).sum()) for c in r.columns if c.startswith("krd_")}
    fx = {ccy: float(g["market_value"].sum()) for ccy, g in r.groupby("currency")} if "currency" in r else {}
    jtd = float((mv * (1.0 - r["recovery"])).where(r.get("rating_bucket", pd.Series("", index=r.index)).isin(["BB", "B", "CCC-D", "NR"]), 0.0).sum()) if "rating_bucket" in r else 0.0
    infl = float((mv * r.get("real_duration", pd.Series(0.0, index=r.index)).fillna(0.0) * 1e-4).sum())
    prepay = float((mv * r.get("prepay_sensitivity", pd.Series(0.0, index=r.index)).fillna(0.0)).sum())
    convexity = float((0.5 * mv * r["convexity"] * 0.01 ** 2).sum())
    days = []
    for _, row in r.iterrows():
        cap = (liquidity_daily_capacity or {}).get(row["symbol"], float("inf"))
        days.append(abs(row["market_value"]) / cap if cap and cap != float("inf") else 0.0)
    r["days_to_liquidate"] = days
    dtl = float(pd.Series(days).quantile(0.9)) if days else 0.0
    funding = float(-mv[mv < 0].sum()) if (mv < 0).any() else 0.0
    cp: dict[str, float] = {}
    if netting_sets:
        for sym, ns in netting_sets.items():
            cp[ns] = cp.get(ns, 0.0) + float(r.loc[r["symbol"] == sym, "market_value"].sum())
    standalone = float(r["dv01"].abs().sum() + r["cs01"].abs().sum())
    netted = float(abs(r["dv01"].sum()) + abs(r["cs01"].sum()))
    return RiskReport(nav, float(r["dv01"].sum()), float(r["cs01"].sum()), float(r["dts_contrib"].sum()), convexity, krd, fx, jtd, infl, prepay, dtl, funding, {k: abs(v) for k, v in cp.items()}, standalone, netted, r)


def full_revaluation(positions: list[dict], reprice: Callable[[dict, dict], float], shocks: list[dict]) -> pd.DataFrame:
    """P&L under each shock by repricing every position (``reprice(position, shock) -> value``)."""
    rows = []
    for shock in shocks:
        base = sum(reprice(p, {}) for p in positions)
        shocked = sum(reprice(p, shock) for p in positions)
        rows.append({"shock": shock.get("name", str(shock)), "pnl": shocked - base, "pnl_pct": (shocked - base) / base if base else float("nan")})
    return pd.DataFrame(rows)


__all__ = ["RiskReport", "compute_risk", "full_revaluation"]
