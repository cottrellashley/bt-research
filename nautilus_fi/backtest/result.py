"""Backtest results: performance, implementation, evidence quality, reconciliation, invariants and
the linked report bundle (blueprint section 22)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.core.manifest import ExperimentManifest
from nautilus_fi.domain.conventions import (
    Conventions,
    MissingReturnPolicy,
    MissingReturnsError,
    TurnoverDefinition,
    apply_missing_policy,
)
from nautilus_fi.execution.assumptions import ExecutionAssumptions
from nautilus_fi.reports.bundle import ReportBundle
from nautilus_fi.validation.invariants import InvariantSuite


@dataclass
class FixedIncomeResult:
    name: str
    assumptions: ExecutionAssumptions
    nav: pd.DataFrame
    fills: pd.DataFrame
    cashflows: pd.DataFrame
    holdings: pd.DataFrame
    decisions: pd.DataFrame
    account_report: pd.DataFrame | None = None
    fills_report: pd.DataFrame | None = None
    positions_report: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    marks: pd.DataFrame | None = None
    manifest: ExperimentManifest | None = None
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    entries: pd.DataFrame = field(default_factory=pd.DataFrame)
    reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    support: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    funding_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    funding_rejects: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_log: object = None
    ledger: object = None
    fx_end: dict[str, float] = field(default_factory=dict)      # spot rates (base per unit) on the last day, for translated checks
    conventions: Conventions = field(default_factory=Conventions)
    construction: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per constraint check (pre / post) per rebalance
    triggers: pd.DataFrame = field(default_factory=pd.DataFrame)       # triggered-calendar evaluations
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ series
    def returns(self) -> pd.Series:
        if self.nav.empty:
            return pd.Series(dtype=float)
        return self.nav["nav"].pct_change().fillna(0.0)

    def benchmark_returns(self, policy: MissingReturnPolicy | None = None) -> pd.Series | None:
        """Benchmark returns on the NAV dates under the conventions' missing-return policy (never a
        silent zero: FAIL raises, ZERO is recorded as a choice)."""
        if self.benchmark is None or self.nav.empty:
            return None
        b = self.benchmark.copy()
        b.index = pd.to_datetime(b.index)
        b = b[~b.index.duplicated(keep="last")].reindex(self.nav.index)
        b.iloc[0] = 0.0 if pd.isna(b.iloc[0]) else b.iloc[0]          # first NAV date has no return period
        out, rep = apply_missing_policy(b, policy or self.conventions.missing_returns)
        self.extra["benchmark_missing"] = rep.to_dict()
        return out.reindex(self.nav.index).fillna(0.0) if rep.policy == MissingReturnPolicy.DROP_DATE.value else out

    @property
    def status(self) -> str:
        return self.manifest.computational_status.value if self.manifest else "UNKNOWN"

    @property
    def outcome(self) -> str:
        return self.manifest.economic_outcome.value if self.manifest else "UNKNOWN"

    # ------------------------------------------------------------------ statistics
    def stats(self, cash_rate: float = 0.0, periods_per_year: int | None = None) -> dict:
        """Headline statistics on one stated basis (``self.conventions``). ``periods_per_year`` overrides
        the convention's annualisation for volatility / Sharpe only when given explicitly."""
        conv = self.conventions
        out: dict = {"name": self.name, "assumptions": self.assumptions.name, "status": self.status, "outcome": self.outcome, "conventions": conv.name, "annualisation": conv.annualisation.value}
        if self.nav.empty:
            return out
        nav = self.nav["nav"]
        rets = self.returns()
        r = rets.iloc[1:] if len(rets) > 1 else rets
        ppy = float(periods_per_year) if periods_per_year else conv.periods_per_year(r.index)
        years = conv.years(r.index) if len(r) else float("nan")
        total = nav.iloc[-1] / nav.iloc[0] - 1.0
        out["start_nav"], out["end_nav"], out["total_return"] = float(nav.iloc[0]), float(nav.iloc[-1]), float(total)
        out["cagr"] = float((1 + total) ** (1 / years) - 1) if years and years > 0 and 1 + total > 0 else float("nan")
        out["ann_vol"] = float(r.std() * np.sqrt(ppy)) if len(r) > 1 else float("nan")
        excess = r - cash_rate / ppy
        out["sharpe"] = float(excess.mean() / excess.std() * np.sqrt(ppy)) if len(r) > 1 and excess.std() > 0 else float("nan")
        out["periods_per_year"] = ppy
        dd = nav / nav.cummax() - 1.0
        out["max_drawdown"] = float(dd.min())
        out["avg_duration"] = float(self.nav["duration"].mean())
        out["avg_dts"] = float(self.nav["dts"].mean())
        out["avg_model_marked_share"] = float(self.nav["model_marked_share"].mean()) if "model_marked_share" in self.nav else float("nan")
        for col in [c for c in self.nav.columns if c.startswith("evidence_")]:
            out[f"avg_{col}"] = float(self.nav[col].mean())
        out["ledger_balanced_all_days"] = bool(self.nav["ledger_balanced"].all()) if "ledger_balanced" in self.nav else None
        out["funding_breach_days"] = int(self.nav["funding_breach"].sum()) if "funding_breach" in self.nav else 0
        out["max_repo_balance"] = float(self.nav["repo_balance"].max()) if "repo_balance" in self.nav else 0.0
        out["funding_actions"] = len(self.funding_events) if self.funding_events is not None else 0
        out["avg_unsettled_cash"] = float(self.nav["unsettled_cash"].abs().mean()) if "unsettled_cash" in self.nav else float("nan")
        if not self.reconciliation.empty:
            out["max_cash_reconciliation_diff"] = float(self.reconciliation["difference"].abs().max())
        cf = self.cashflows
        if not cf.empty:
            out["income_total"] = float(cf[cf["type"].isin(["COUPON", "CASH_INTEREST"])]["amount"].sum())
            out["financing_total"] = float(cf[cf["type"] == "FINANCING"]["amount"].sum())
        pnl = float(nav.iloc[-1] - nav.iloc[0])
        out["pnl"] = pnl
        out["income_share_of_pnl"] = float(out.get("income_total", 0.0) / pnl) if pnl else float("nan")
        f = self.fills
        if not f.empty:
            traded = f[f["protocol"] != "EXPIRATION"]
            mv = (traded["face"] * traded["clean_price"] / 100.0).abs()
            buys, sells = float(mv[traded["side"] == "BUY"].sum()), float(mv[traded["side"] == "SELL"].sum())
            traded_mv = buys + sells
            per_year = nav.mean() * years if years and years > 0 else float("nan")
            out["traded_mv"] = traded_mv
            out["n_fills"] = len(traded)
            out["turnover_annual_two_way"] = float(traded_mv / per_year)
            out["turnover_annual_half_sum"] = float(0.5 * traded_mv / per_year)
            out["turnover_annual_one_way_buys"] = float(buys / per_year)
            out["turnover_annual"] = out[{TurnoverDefinition.TWO_WAY: "turnover_annual_two_way", TurnoverDefinition.HALF_SUM: "turnover_annual_half_sum", TurnoverDefinition.ONE_WAY_BUYS: "turnover_annual_one_way_buys"}[conv.turnover]]
            out["turnover_definition"] = conv.turnover.value
            out["spread_cost_total"] = float(traded["spread_cost"].sum())
            out["delay_drift_total"] = float(traded["delay_cost"].sum())
            out["spread_cost_bp"] = float(traded["spread_cost"].sum() / traded_mv * 1e4) if traded_mv else float("nan")
            out["delay_drift_bp"] = float(traded["delay_cost"].sum() / traded_mv * 1e4) if traded_mv else float("nan")
            out["avg_delay_days"] = float(traded["delay_days"].mean())
            out["cost_drag_bp_nav"] = float(traded["spread_cost"].sum() / per_year * 1e4)   # execution cost only; delay drift is attribution, not a second charge
            if "reference_source" in traded:
                out["reference_sources"] = ",".join(sorted({str(x).split("(")[0] for x in traded["reference_source"].dropna().unique()}))
        fr = self.fills_report
        if fr is not None and len(fr) and "quantity" in fr:
            own = fr[~fr.index.astype(str).str.startswith("EXPIRATION")] if fr.index.name == "client_order_id" else fr
            req = pd.to_numeric(own["quantity"], errors="coerce").sum()
            filled = pd.to_numeric(own["filled_qty"], errors="coerce").sum()
            out["fill_rate"] = float(filled / req) if req else float("nan")
            out["n_orders"] = len(own)
        d = self.decisions
        if not d.empty:
            out["decision_fill_share"] = float((d["offered_face"] > 0).mean())
            out["funding_holds"] = int((d["state"] == "FUNDING_HOLD").sum()) if "state" in d else 0
        if not self.construction.empty:
            post = self.construction[self.construction["stage"] == "post"] if "stage" in self.construction else self.construction
            out["constraint_breaches_post"] = int((post["severity"] == "HARD").sum()) if "severity" in post else 0
        try:
            b = self.benchmark_returns()
        except MissingReturnsError as e:
            out["benchmark_error"] = str(e)
            b = None
        if b is not None:
            b = b.iloc[1:] if len(b) > 1 else b
            active = r - b.reindex(r.index)
            out["benchmark_total_return"] = float((1 + b).prod() - 1)
            out["excess_return_annual"] = float(active.mean() * ppy)
            out["tracking_error"] = float(active.std() * np.sqrt(ppy))
            out["information_ratio"] = float(out["excess_return_annual"] / out["tracking_error"]) if out["tracking_error"] > 0 else float("nan")
        return out

    def summary(self) -> pd.Series:
        return pd.Series(self.stats())

    def holdings_weights(self) -> pd.DataFrame:
        """Close-of-day weights (market value / NAV) by symbol from the holdings history."""
        if self.holdings is None or self.holdings.empty:
            return pd.DataFrame()
        w = self.holdings.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum").fillna(0.0)
        return w.div(self.nav["nav"].reindex(w.index), axis=0)

    def excess_returns(self, treasury: pd.DataFrame | None = None, key_durations: dict[str, float] | None = None, cash: pd.Series | None = None, vendor_xr: pd.DataFrame | None = None, freq: str | None = None) -> pd.Series:
        """Excess return over Treasuries under the conventions.

        ``vendor_xr`` (dates x symbols, the vendor's bond excess returns): holdings-weighted, the
        exact figure when the vendor's definition is wanted. Otherwise a duration-matched leg:
        ``treasury`` holds key-tenor Treasury returns (columns ``r_<tenor>``), split between the two
        tenors bracketing the portfolio's duration (``key_durations``: duration per column, default
        the tenor), the rest earning ``cash``."""
        from nautilus_fi.domain.conventions import excess_return_series

        r = self.returns().iloc[1:]
        if vendor_xr is not None:
            w = self.holdings_weights().shift(1).reindex(r.index).fillna(0.0)
            x = vendor_xr.copy()
            x.index = pd.to_datetime(x.index)
            x = x.reindex(index=r.index, columns=w.columns).fillna(0.0)
            ven = (w * x).sum(axis=1)
            return excess_return_series(r, None, self.conventions.with_(prefer_vendor_excess=True), freq, ven)
        if treasury is None:
            raise ValueError("excess_returns needs Treasury key-tenor returns or vendor bond excess returns")
        t = treasury.copy()
        t.index = pd.to_datetime(t.index)
        cols = [c for c in t.columns if c.startswith("r_")]
        durs = np.array([float((key_durations or {}).get(c, float(c[2:]))) for c in cols])
        order = np.argsort(durs)
        cols, durs = [cols[i] for i in order], durs[order]
        D = self.nav["duration"].shift(1).reindex(r.index).fillna(0.0).to_numpy()
        T = t[cols].reindex(r.index).fillna(0.0).to_numpy()
        c = cash.reindex(r.index).fillna(0.0).to_numpy() if cash is not None else np.zeros(len(r))
        leg = np.zeros(len(r))
        for i, d in enumerate(D):
            if d <= durs[0]:
                wlo = d / durs[0] if durs[0] else 0.0
                leg[i] = wlo * T[i, 0] + (1 - wlo) * c[i]
            elif d >= durs[-1]:
                leg[i] = d / durs[-1] * T[i, -1] + (1 - d / durs[-1]) * c[i]
            else:
                j = int(np.searchsorted(durs, d))
                whi = (d - durs[j - 1]) / (durs[j] - durs[j - 1])
                leg[i] = (1 - whi) * T[i, j - 1] + whi * T[i, j]
        return excess_return_series(r, pd.Series(leg, index=r.index), self.conventions, freq)

    # ------------------------------------------------------------------ evidence
    def evidence_table(self) -> pd.DataFrame:
        """Share of market value and turnover by valuation evidence class (blueprint section 14)."""
        if self.holdings.empty:
            return pd.DataFrame()
        h = self.holdings.copy()
        mv = h.groupby("evidence")["market_value"].sum()
        mv_share = (mv / mv.sum()).rename("market_value_share") if mv.sum() else mv.rename("market_value_share")
        out = pd.DataFrame(mv_share)
        col = "mark_evidence" if "mark_evidence" in self.fills else "evidence"
        if not self.fills.empty and col in self.fills:
            t = self.fills[(self.fills["protocol"] != "EXPIRATION") & (self.fills[col] != "")]
            tv = (t["face"] * t["clean_price"] / 100.0).abs().groupby(t[col]).sum()
            out["turnover_share"] = (tv / tv.sum()).reindex(out.index).fillna(0.0) if tv.sum() else 0.0
        return out

    def execution_evidence(self) -> pd.Series:
        """Share of traded value by the execution evidence level the fill simulation rested on
        (daily evaluations / transactions / timestamped quotes / full RFQ history)."""
        if self.fills.empty or "evidence" not in self.fills:
            return pd.Series(dtype=float)
        t = self.fills[(self.fills["protocol"] != "EXPIRATION") & (self.fills["evidence"] != "")]
        tv = (t["face"] * t["clean_price"] / 100.0).abs().groupby(t["evidence"]).sum()
        return tv / tv.sum() if tv.sum() else tv

    def invariants(self) -> pd.DataFrame:
        suite = InvariantSuite()
        base = self.manifest.base_currency if self.manifest else "USD"
        fx = (lambda c: 1.0 if c == base else self.fx_end.get(c, 1.0))
        if self.ledger is not None:
            suite.conservation(self.ledger, fx)
        if self.event_log is not None and self.ledger is not None:
            suite.payments_reconcile(self.event_log, self.ledger, base, fx)
        if self.fills_report is not None and len(self.fills_report):
            fr = self.fills_report.reset_index() if self.fills_report.index.name == "client_order_id" else self.fills_report
            suite.unfilled_orders_do_not_change_ownership(fr, {}, {})
        return suite.to_frame()

    def attribution(self) -> pd.DataFrame:
        if self.nav.empty:
            return pd.DataFrame()
        nav = self.nav["nav"]
        pnl = nav.diff().fillna(0.0)
        cf = self.cashflows.copy()
        income = pd.Series(0.0, index=nav.index)
        if not cf.empty:
            cf["date"] = pd.to_datetime(cf["date"])
            daily = cf[cf["type"].isin(["COUPON", "CASH_INTEREST", "FINANCING"])].groupby("date")["amount"].sum()
            income = income.add(daily.reindex(nav.index).fillna(0.0), fill_value=0.0)
        exec_cost = pd.Series(0.0, index=nav.index)
        if not self.fills.empty:
            fc = self.fills.copy()
            fc["date"] = pd.to_datetime(fc["date"])
            exec_cost = exec_cost.add(-fc.groupby("date")["spread_cost"].sum().reindex(nav.index).fillna(0.0), fill_value=0.0)
        return pd.DataFrame({"pnl": pnl, "income": income, "execution": exec_cost, "price_and_other": pnl - income - exec_cost})

    def bundle(self) -> ReportBundle:
        m = self.manifest or ExperimentManifest(start="", end="")
        stats = self.stats()
        headline = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in stats.items() if k in ("total_return", "cagr", "sharpe", "max_drawdown", "fill_rate", "spread_cost_bp", "avg_model_marked_share", "status", "outcome")}
        unknowns = []
        if not self.support.empty:
            for _, row in self.support.iterrows():
                if row["status"] != "VALIDATED":
                    unknowns.append(f"{row['product_family']}/{row['feature']}: {row['status']} {row.get('approximation', '')}".strip())
        unknowns.append("execution calibration: assumed (RFQ panel and liquidity gate not calibrated on own order history)")
        perf = pd.DataFrame([stats]).T.rename(columns={0: self.name})
        impl = self.decisions.groupby(["state", "reason"]).size().rename("count").reset_index() if not self.decisions.empty else pd.DataFrame()
        return ReportBundle(m, performance=perf, implementation=impl, holdings=self.holdings.tail(200), liquidity=self.funding_events, risk=pd.DataFrame(), attribution=self.attribution().sum().to_frame("total") if not self.nav.empty else pd.DataFrame(), evidence=self.evidence_table(), robustness=pd.DataFrame(), support=self.support, invariants=self.invariants(), headline=headline, unknowns=unknowns)


def compare(results: dict[str, FixedIncomeResult] | list[FixedIncomeResult], keys: list[str] | None = None) -> pd.DataFrame:
    items = results.items() if isinstance(results, dict) else [(r.name, r) for r in results]
    rows = {name: r.stats() for name, r in items}
    df = pd.DataFrame(rows)
    keys = keys or ["status", "outcome", "total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "income_total", "turnover_annual", "fill_rate", "avg_delay_days", "spread_cost_bp", "delay_drift_bp", "cost_drag_bp_nav", "avg_duration", "avg_dts", "avg_model_marked_share", "funding_breach_days", "max_cash_reconciliation_diff", "ledger_balanced_all_days", "benchmark_total_return", "excess_return_annual", "information_ratio"]
    return df.reindex([k for k in keys if k in df.index])


__all__ = ["FixedIncomeResult", "compare"]
