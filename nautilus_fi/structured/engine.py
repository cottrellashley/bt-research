"""Deal cashflow engines: the internal waterfall engine and the adapter contract for external
engines (Intex and others). Every run pins engine version, deal snapshot and assumptions; if a vendor
cannot reproduce a historical model version the output snapshot is retained and disclosed."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import pandas as pd

from nautilus_fi.instruments.schedule import add_months
from nautilus_fi.structured.collateral import CollateralPool, PoolAssumptions, PoolCashflow, RemittanceReport
from nautilus_fi.structured.waterfall import Deal, DealState, WaterfallResult, run_waterfall_period


@dataclass
class TrancheCashflow:
    tranche: str
    period_end: date
    interest: float
    principal: float
    deferred: float
    factor_end: float
    source: str                     # REALIZED | PROJECTED


@dataclass
class EngineSnapshot:
    engine: str
    version: str
    deal_id: str
    as_of: date
    assumptions_hash: str
    deal_snapshot_id: str
    reproducible: bool = True
    note: str = ""


class DealCashflowEngine(Protocol):
    def project(self, deal: Deal, state: DealState, pool: CollateralPool, start: date, periods: int, assumptions: PoolAssumptions, fixings: dict[str, float], remittances: list[RemittanceReport] | None = None) -> tuple[list[WaterfallResult], list[TrancheCashflow], EngineSnapshot]: ...


class InternalWaterfallEngine:
    name = "nautilus_fi.internal_waterfall"
    version = "0.1"

    def project(self, deal: Deal, state: DealState, pool: CollateralPool, start: date, periods: int, assumptions: PoolAssumptions, fixings: dict[str, float], remittances: list[RemittanceReport] | None = None) -> tuple[list[WaterfallResult], list[TrancheCashflow], EngineSnapshot]:
        """Run ``periods`` collection periods from ``start``. Realized remittances (observed) are used
        for periods they cover; later periods are projected under ``assumptions``. A period is never both."""
        by_period = {r.period_end: r for r in (remittances or [])}
        months = 12 // deal.payment_frequency
        results: list[WaterfallResult] = []
        flows: list[TrancheCashflow] = []
        d = start
        for _ in range(periods):
            d = add_months(d, months)
            if d in by_period:
                pcf: PoolCashflow = pool.apply_remittance(by_period[d])
            else:
                pcf = pool.project_period(d, assumptions, months)
            if pcf.source == "PROJECTED" and deal.reinvestment_end is not None and d <= deal.reinvestment_end:
                pass  # reinvestment handled by the waterfall (principal retained in the deal)
            res = run_waterfall_period(deal, state, d, pool.balance, pcf.interest, pcf.total_principal, pcf.losses, fixings)
            if res.reinvested > 0:
                pool.reinvest(res.reinvested, (assumptions.reinvestment_spread_bp or 0.0) / 1e4)
            results.append(res)
            for t in deal.tranches:
                flows.append(TrancheCashflow(t.name, d, res.interest_paid[t.name], res.principal_paid[t.name] + (res.equity_cash if t.kind == "EQUITY" else 0.0), res.deferred[t.name], state.factor(t.name, deal), pcf.source))
        snap = EngineSnapshot(self.name, self.version, deal.deal_id, start, hashlib.sha1(json.dumps(assumptions.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:12], hashlib.sha1(json.dumps({t.name: state.balances[t.name] for t in deal.tranches}, sort_keys=True).encode()).hexdigest()[:12])
        return results, flows, snap


class ExternalEngineAdapter(Protocol):
    """Contract for vendor engines (e.g. Intex): run a deal as of a date under assumptions and return
    tranche cashflows with a provenance snapshot. Implementations must pin engine/model versions."""

    engine_name: str

    def run(self, deal_id: str, as_of: date, assumptions: dict, scenario: dict | None = None) -> tuple[list[TrancheCashflow], EngineSnapshot]: ...


def tranche_flows_frame(flows: list[TrancheCashflow]) -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in flows])


def results_frame(results: list[WaterfallResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"period_end": r.period_end, "interest_in": r.available_interest, "principal_in": r.available_principal, "fees": r.fees, "interest_paid": sum(r.interest_paid.values()), "principal_paid": sum(r.principal_paid.values()), "reinvested": r.reinvested, "equity_cash": r.equity_cash, "diverted": r.diverted_interest, "conservation_error": r.conservation_error, "tests_failed": [k for k, v in r.tests.items() if not (v["oc_pass"] and v["ic_pass"])]})
    return pd.DataFrame(rows)


__all__ = ["DealCashflowEngine", "EngineSnapshot", "ExternalEngineAdapter", "InternalWaterfallEngine", "TrancheCashflow", "results_frame", "tranche_flows_frame"]
_ = field
