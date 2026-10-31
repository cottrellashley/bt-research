"""Paper vs Implement: the same spec in both engines, and where the return goes between them.

``paper_data_from_environment`` puts the panel engine on exactly the prices the event engine marks
with (total returns from the environment's marks, accrued and cashflows; the environment's half
spreads as costs; its benchmark). ``reconcile`` then decomposes the daily return difference:

* ``holdings_gap``   (paper weights - implemented weights) x asset returns: decision-to-fill delay,
                     partial fills, expiries, funding holds, rounding to minimum pieces;
* ``cash``           cash-weight difference x the cash return (settlement drag, buffers);
* ``paper_costs``    the paper engine's modelled costs (gross -> net);
* ``execution``      the event engine's spread cost on its fills;
* ``residual``       everything else (pricing and accrual conventions, income timing, financing).

Cumulative figures are sums of daily contributions (arithmetic attribution, no geometric linking).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.data.panel import Panel
from nautilus_fi.paper.engine import PaperData, PaperResult


def paper_data_from_environment(env, cash: pd.Series | None = None, benchmark: bool = True) -> PaperData:
    panel = env.total_return_panel()
    attrs = pd.DataFrame({"issuer": [b.issuer for b in env.bonds.values()], "sector": [b.sector for b in env.bonds.values()], "currency": [b.currency for b in env.bonds.values()], "asset_type": [b.asset_type.value for b in env.bonds.values()], "rating_initial": [b.rating_score_initial for b in env.bonds.values()]}, index=list(env.bonds))
    bench = None
    if benchmark and env.benchmark_obj is not None:
        bench = env.benchmark_obj
    elif benchmark and env.benchmark is not None:
        from nautilus_fi.benchmarks.benchmark import ReturnSeriesBenchmark

        bench = ReturnSeriesBenchmark("benchmark", env.benchmark)
    if env.field_panel is not None:
        for f in env.field_panel.fields:
            if f not in panel:
                panel.with_field(f, env.field_panel.subset(ids=list(panel.ids))[f], env.field_panel.lags.get(f, 0))
    return PaperData(panel, attrs, cash, bench, excess_field=None, treasury_field=None)


def implemented_weights(result) -> pd.DataFrame:
    """Close-of-day weights of the event engine's holdings (dates x symbols)."""
    h = result.holdings
    if h is None or h.empty:
        return pd.DataFrame()
    nav = result.nav["nav"]
    w = h.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum").fillna(0.0)
    return w.div(nav.reindex(w.index), axis=0)


@dataclass
class Reconciliation:
    daily: pd.DataFrame
    summary: dict
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.summary)


def reconcile(paper: PaperResult, implement, panel: Panel, cash: pd.Series | None = None, return_field: str = "tr") -> Reconciliation:
    if paper.weights.empty:
        raise ValueError("the paper result has no weights; run it with PaperSettings(record_daily_weights=True)")
    idx = paper.returns.index.intersection(implement.nav.index)
    wp = paper.weights.pivot_table(index="date", columns="instrument_id", values="weight", aggfunc="sum")
    wp = wp.reindex(paper.returns.index).ffill().fillna(0.0)
    wi = implemented_weights(implement).reindex(idx).ffill().fillna(0.0)
    cols = wp.columns.union(wi.columns)
    wp, wi = wp.reindex(columns=cols).fillna(0.0).reindex(idx), wi.reindex(columns=cols).fillna(0.0)
    R = panel.frame(return_field).reindex(index=idx, columns=cols).fillna(0.0)
    c = (cash.reindex(idx).fillna(0.0) if cash is not None else pd.Series(0.0, index=idx))
    gap = ((wp.shift(1) - wi.shift(1)) * R).sum(axis=1)
    cash_eff = ((1 - wp.shift(1).sum(axis=1)) - (1 - wi.shift(1).sum(axis=1))) * c
    r_i = implement.nav["nav"].reindex(idx).pct_change().fillna(0.0)
    r_pg = paper.gross.reindex(idx).fillna(0.0)
    r_pn = paper.returns.reindex(idx).fillna(0.0)
    nav_prev = implement.nav["nav"].reindex(idx).shift(1)
    ex = pd.Series(0.0, index=idx)
    f = implement.fills
    if f is not None and not f.empty:
        t = f[f["protocol"] != "EXPIRATION"].copy()
        t["date"] = pd.to_datetime(t["date"])
        ex = (t.groupby("date")["spread_cost"].sum().reindex(idx).fillna(0.0) / nav_prev).fillna(0.0)
    paper_costs = r_pg - r_pn
    diff = r_pg - r_i
    resid = diff - gap - cash_eff - ex
    daily = pd.DataFrame({"paper_gross": r_pg, "paper_net": r_pn, "implemented": r_i, "difference": diff, "holdings_gap": gap.fillna(0.0), "cash": cash_eff.fillna(0.0), "paper_costs": paper_costs, "execution": ex, "residual": resid.fillna(0.0)})
    d = daily.iloc[1:]
    summary = {"paper_gross_total": float((1 + d["paper_gross"]).prod() - 1), "paper_net_total": float((1 + d["paper_net"]).prod() - 1), "implemented_total": float((1 + d["implemented"]).prod() - 1), "shortfall_vs_gross": float(d["difference"].sum()), "holdings_gap": float(d["holdings_gap"].sum()), "cash": float(d["cash"].sum()), "execution": float(d["execution"].sum()), "paper_costs": float(d["paper_costs"].sum()), "residual": float(d["residual"].sum()), "tracking_difference_vol": float(d["difference"].std() * np.sqrt(252)), "daily_correlation": float(d["paper_gross"].corr(d["implemented"])), "weight_gap_mean_abs": float((wp - wi).abs().sum(axis=1).mean())}
    return Reconciliation(daily, summary)


__all__ = ["Reconciliation", "implemented_weights", "paper_data_from_environment", "reconcile"]
