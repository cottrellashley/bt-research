"""Return calculation with capital and timing rules (blueprint section 17): NAV from assets less
liabilities, external flows separated from investment flows, time-weighted and money-weighted
returns, explicit capital denominators for leveraged books."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReturnSeries:
    twr_daily: pd.Series
    twr_cumulative: pd.Series
    mwr_annual: float | None
    denominator: str
    notes: list[str]


def time_weighted_returns(nav: pd.Series, external_flows: pd.Series | None = None, flow_timing: str = "START") -> pd.Series:
    """Daily TWR chain-linked around external flows. ``flow_timing`` START assumes flows arrive before
    the day's return (modified Dietz style for daily data)."""
    nav = nav.astype(float)
    flows = (external_flows.reindex(nav.index).fillna(0.0) if external_flows is not None else pd.Series(0.0, index=nav.index))
    prev = nav.shift(1)
    if flow_timing == "START":
        denom = prev + flows
        ret = (nav - prev - flows) / denom
    else:
        ret = (nav - flows - prev) / prev
    return ret.fillna(0.0).replace([np.inf, -np.inf], 0.0)


def money_weighted_return(dates: pd.DatetimeIndex, cashflows: np.ndarray, terminal_value: float) -> float | None:
    """Annualised IRR of investor cashflows (negative = contribution) with the terminal value as the last inflow."""
    if len(dates) == 0:
        return None
    t = np.array([(d - dates[0]).days / 365.25 for d in dates] + [(dates[-1] - dates[0]).days / 365.25 + 1e-9])
    cf = np.append(np.asarray(cashflows, dtype=float), terminal_value)

    def npv(r: float) -> float:
        return float(np.sum(cf / (1.0 + r) ** t))

    lo, hi = -0.99, 10.0
    while npv(lo) * npv(hi) > 0 and hi < 1e9:
        hi *= 10.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def capital_denominator(nav: pd.Series, gross_exposure: pd.Series | None = None, committed_capital: float | None = None, method: str = "NAV") -> pd.Series:
    """Denominator for returns of leveraged / derivative books: NAV, committed capital or gross exposure."""
    if method == "COMMITTED" and committed_capital:
        return pd.Series(committed_capital, index=nav.index)
    if method == "GROSS" and gross_exposure is not None:
        return gross_exposure.reindex(nav.index).replace(0.0, np.nan).ffill()
    return nav


def compute_returns(nav: pd.Series, external_flows: pd.Series | None = None, denominator: str = "NAV", committed_capital: float | None = None, gross_exposure: pd.Series | None = None) -> ReturnSeries:
    notes = []
    if (nav <= 0).any():
        notes.append("NAV at or below zero on some dates: returns on those dates are undefined and set to 0")
    twr = time_weighted_returns(nav, external_flows)
    if denominator != "NAV":
        denom = capital_denominator(nav, gross_exposure, committed_capital, denominator)
        pnl = nav.diff().fillna(0.0) - (external_flows.reindex(nav.index).fillna(0.0) if external_flows is not None else 0.0)
        twr = (pnl / denom.shift(1)).fillna(0.0)
        notes.append(f"returns on {denominator} denominator")
    cum = (1.0 + twr).cumprod() - 1.0
    flows = external_flows.reindex(nav.index).fillna(0.0) if external_flows is not None else pd.Series(0.0, index=nav.index)
    cfs = -flows.values.copy()
    cfs[0] = -float(nav.iloc[0])
    mwr = money_weighted_return(pd.DatetimeIndex(nav.index), cfs, float(nav.iloc[-1]))
    return ReturnSeries(twr, cum, mwr, denominator, notes)


__all__ = ["ReturnSeries", "capital_denominator", "compute_returns", "money_weighted_return", "time_weighted_returns"]
