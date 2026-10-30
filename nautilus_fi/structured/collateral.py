"""Collateral pool cashflows: realized from remittance reports (replay environment) or projected
from prepayment / default / severity assumptions (valuation). Never mix the two for one period."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class PoolAssumptions:
    cpr: float = 0.08                 # annual voluntary prepayment
    cdr: float = 0.02                 # annual default rate
    severity: float = 0.35            # loss given default
    recovery_lag_months: int = 12
    reinvestment_spread_bp: float | None = None   # for CLOs: spread on reinvested collateral
    measure: str = "REAL_WORLD"       # label the probability model (blueprint section 3)


@dataclass
class RemittanceReport:
    """What the trustee/servicer reported for a collection period (observed)."""

    period_end: date
    publication_date: date
    beginning_balance: float
    interest_collected: float
    scheduled_principal: float
    prepayments: float
    defaults: float
    losses: float
    recoveries: float
    ending_balance: float
    wac: float


@dataclass
class PoolCashflow:
    period_end: date
    beginning_balance: float
    interest: float
    scheduled_principal: float
    prepayments: float
    defaults: float
    losses: float
    recoveries: float
    ending_balance: float
    source: str                        # REALIZED | PROJECTED

    @property
    def total_principal(self) -> float:
        return self.scheduled_principal + self.prepayments + self.recoveries

    @property
    def available_cash(self) -> float:
        return self.interest + self.total_principal


@dataclass
class CollateralPool:
    balance: float
    wac: float
    wam_months: int
    original_balance: float | None = None
    amortizing: bool = True
    pending_recoveries: list[tuple[date, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.original_balance is None:
            self.original_balance = self.balance

    def project_period(self, period_end: date, a: PoolAssumptions, months: int = 1) -> PoolCashflow:
        """One collection period under assumptions (projected)."""
        beg = self.balance
        smm = 1.0 - (1.0 - a.cpr) ** (months / 12.0)
        mdr = 1.0 - (1.0 - a.cdr) ** (months / 12.0)
        interest = beg * self.wac * months / 12.0
        defaults = beg * mdr
        performing = beg - defaults
        if self.amortizing and self.wam_months > 0:
            r = self.wac / 12.0
            n = self.wam_months
            sched = performing * (1.0 / ((1 + r) ** n - 1) * r) * months if r > 0 else performing / n * months
            sched = min(sched, performing)
        else:
            sched = 0.0
        prepay = (performing - sched) * smm
        losses = defaults * a.severity
        recov_now = sum(v for d, v in self.pending_recoveries if d <= period_end)
        self.pending_recoveries = [(d, v) for d, v in self.pending_recoveries if d > period_end]
        from nautilus_fi.instruments.schedule import add_months

        self.pending_recoveries.append((add_months(period_end, a.recovery_lag_months), defaults - losses))
        end = performing - sched - prepay
        self.balance = max(end, 0.0)
        self.wam_months = max(self.wam_months - months, 0)
        return PoolCashflow(period_end, beg, interest, sched, prepay, defaults, losses, recov_now, self.balance, "PROJECTED")

    def apply_remittance(self, r: RemittanceReport) -> PoolCashflow:
        """Realized collections from a remittance report (observed)."""
        self.balance = r.ending_balance
        self.wac = r.wac
        self.wam_months = max(self.wam_months - 1, 0)
        return PoolCashflow(r.period_end, r.beginning_balance, r.interest_collected, r.scheduled_principal, r.prepayments, r.defaults, r.losses, r.recoveries, r.ending_balance, "REALIZED")

    def reinvest(self, amount: float, spread_over_wac: float = 0.0) -> None:
        """CLO reinvestment: principal collections buy new collateral (balance stays up)."""
        if amount <= 0:
            return
        new_wac = (self.balance * self.wac + amount * (self.wac + spread_over_wac)) / (self.balance + amount) if (self.balance + amount) else self.wac
        self.balance += amount
        self.wac = new_wac


__all__ = ["CollateralPool", "PoolAssumptions", "PoolCashflow", "RemittanceReport"]
