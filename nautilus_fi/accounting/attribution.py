"""P&L attribution that reconciles to the ledger (blueprint section 18).

Daily decomposition per instrument in a declared order: carry (accrual and roll-down), rates
(curve move at constant spread), spread, optionality / behaviour (model residual), FX, defaults and
recoveries (events), execution (fill versus reference), financing and fees (ledger), model changes
(mark policy or engine switches) and an explicit residual. The residual is published, not renamed
alpha."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

ORDER = ("carry", "rates", "spread", "optionality", "fx", "defaults_recoveries", "execution", "financing_fees", "model_change", "residual")


@dataclass
class AttributionInputs:
    """Per instrument per day. Prices are clean per 100 current face; ``face`` is current face."""

    symbol: str
    d: date
    face: float
    clean_prev: float
    clean_now: float
    accrued_prev: float
    accrued_now: float
    coupon_cash: float                 # coupon cash received today (per 100)
    duration_prev: float | None
    spread_duration_prev: float | None
    dy_bp: float | None                # move in the risk-free yield at the bond's horizon
    ds_bp: float | None                # move in the bond's own spread
    fx_prev: float = 1.0
    fx_now: float = 1.0
    execution_cost: float = 0.0        # currency, fills vs reference today
    default_recovery_pnl: float = 0.0  # currency, from events
    model_change_pnl: float = 0.0      # currency, mark source / engine switch
    carry_days: int = 1
    ytm_prev: float | None = None


def attribute_instrument(x: AttributionInputs) -> dict[str, float]:
    """Sequential decomposition in local currency then translated; returns components plus the total.

    Carry is the income at constant yield when ``ytm_prev`` is given (accrual and coupons plus the
    roll-down / pull to par, ``ytm_prev x dirty_prev x days / 365.25``), else accrual and coupons.
    Rates and spread are duration times the moves; the price change they and the roll-down leave is
    optionality / behaviour, and without moves it is residual."""
    scale = x.face / 100.0
    total_local = ((x.clean_now + x.accrued_now + x.coupon_cash) - (x.clean_prev + x.accrued_prev)) * scale
    income = (x.accrued_now - x.accrued_prev + x.coupon_cash) * scale
    rolldown = 0.0
    if x.ytm_prev is not None and x.carry_days:
        rolldown = x.ytm_prev * (x.clean_prev + x.accrued_prev) * x.carry_days / 365.25 * scale - income
    carry = income + rolldown
    rates = -(x.duration_prev or 0.0) * (x.dy_bp or 0.0) / 1e4 * x.clean_prev * scale
    spread = -(x.spread_duration_prev or 0.0) * (x.ds_bp or 0.0) / 1e4 * x.clean_prev * scale
    price_total = (x.clean_now - x.clean_prev) * scale
    optionality = price_total - rolldown - rates - spread if (x.dy_bp is not None or x.ds_bp is not None) else 0.0
    local = {"carry": carry, "rates": rates, "spread": spread, "optionality": optionality}
    translated = {k: v * x.fx_now for k, v in local.items()}
    fx_pnl = (x.fx_now - x.fx_prev) * (x.clean_prev + x.accrued_prev) * scale
    translated["fx"] = fx_pnl
    translated["defaults_recoveries"] = x.default_recovery_pnl
    translated["execution"] = -abs(x.execution_cost)
    translated["financing_fees"] = 0.0
    translated["model_change"] = x.model_change_pnl
    total = total_local * x.fx_now + fx_pnl + x.default_recovery_pnl - abs(x.execution_cost) + x.model_change_pnl
    translated["residual"] = total - sum(translated[k] for k in ORDER if k != "residual")
    translated["total"] = total
    return translated


@dataclass
class AttributionReport:
    rows: pd.DataFrame
    ledger_pnl: float
    attributed_total: float
    residual_vs_ledger: float
    order: tuple[str, ...] = ORDER
    notes: list[str] = field(default_factory=list)

    def by_component(self) -> pd.Series:
        return self.rows[list(self.order) + ["total"]].sum()

    def by_instrument(self) -> pd.DataFrame:
        return self.rows.groupby("symbol")[list(self.order) + ["total"]].sum()


def attribute_portfolio(inputs: list[AttributionInputs], ledger_pnl: float, financing_fees: float = 0.0, tolerance: float = 1e-6) -> AttributionReport:
    """Attribute every instrument, add portfolio-level financing/fees from the ledger, and reconcile."""
    recs = []
    for x in inputs:
        comp = attribute_instrument(x)
        recs.append({"symbol": x.symbol, "date": x.d, **comp})
    df = pd.DataFrame(recs, columns=["symbol", "date", *ORDER, "total"])
    if financing_fees:
        df.loc[len(df)] = {"symbol": "PORTFOLIO", "date": inputs[0].d if inputs else None, **{k: 0.0 for k in ORDER}, "financing_fees": financing_fees, "total": financing_fees}
    attributed = float(df["total"].sum()) if len(df) else 0.0
    residual = ledger_pnl - attributed
    notes = []
    if abs(residual) > tolerance:
        notes.append(f"unexplained {residual:,.2f} between ledger P&L and attributed total (kept as residual, not alpha)")
        if len(df):
            df.loc[len(df)] = {"symbol": "UNEXPLAINED", "date": inputs[0].d if inputs else None, **{k: 0.0 for k in ORDER}, "residual": residual, "total": residual}
    return AttributionReport(df, ledger_pnl, attributed, residual, ORDER, notes)


def attribution_series(daily_reports: list[AttributionReport]) -> pd.DataFrame:
    rows = []
    for rep in daily_reports:
        s = rep.by_component()
        s["ledger_pnl"] = rep.ledger_pnl
        s["residual_vs_ledger"] = rep.residual_vs_ledger
        rows.append(s)
    return pd.DataFrame(rows)


__all__ = ["ORDER", "AttributionInputs", "AttributionReport", "attribute_instrument", "attribute_portfolio", "attribution_series"]
