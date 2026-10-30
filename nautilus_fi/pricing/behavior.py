"""Behaviour models (blueprint section 13): prepayment, issuer exercise, default and recovery timing,
and borrower draws are separate from pricing engines so alternatives can be compared and the
probability measure labelled."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol


class PrepaymentModel(Protocol):
    model_id: str
    measure: str

    def cpr(self, age_months: int, wac: float, market_rate: float, burnout: float, month: int) -> float: ...


@dataclass
class PSAModel:
    """PSA ramp: CPR rises 0.2% per month to 6% at month 30, scaled by ``speed`` (100 = standard)."""

    speed: float = 100.0
    model_id: str = "psa"
    measure: str = "REAL_WORLD"

    def cpr(self, age_months: int, wac: float = 0.0, market_rate: float = 0.0, burnout: float = 0.0, month: int = 1) -> float:
        return min(age_months, 30) / 30.0 * 0.06 * self.speed / 100.0


@dataclass
class RefinanceIncentiveModel:
    """Turnover + refinancing incentive (S-curve) with seasoning ramp, burnout and seasonality."""

    turnover_cpr: float = 0.06
    max_refi_cpr: float = 0.50
    incentive_midpoint: float = 0.0075     # 75 bp in-the-money
    incentive_slope: float = 600.0
    seasoning_months: int = 30
    burnout_decay: float = 0.6             # multiplies refi response by (1 - burnout)^decay
    seasonality: dict[int, float] = field(default_factory=lambda: {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.05, 5: 1.1, 6: 1.15, 7: 1.15, 8: 1.1, 9: 1.0, 10: 0.95, 11: 0.9, 12: 0.85})
    model_id: str = "refi-incentive-v1"
    measure: str = "REAL_WORLD"

    def cpr(self, age_months: int, wac: float, market_rate: float, burnout: float = 0.0, month: int = 6) -> float:
        import math

        ramp = min(age_months, self.seasoning_months) / self.seasoning_months
        incentive = wac - market_rate
        refi = self.max_refi_cpr / (1.0 + math.exp(-self.incentive_slope * (incentive - self.incentive_midpoint)))
        refi *= max(0.0, 1.0 - burnout) ** self.burnout_decay
        return min(0.95, (self.turnover_cpr * ramp + refi) * self.seasonality.get(month, 1.0))


def cpr_to_smm(cpr: float) -> float:
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


@dataclass
class IssuerCallPolicy:
    """Issuer exercises a fixed-price call when refinancing is cheaper by more than a threshold after
    notice; make-whole calls are only exercised when the make-whole price is below market value."""

    threshold_bp: float = 25.0
    notice_days: int = 30
    model_id: str = "issuer-call-economic-v1"

    def exercise(self, coupon: float, refinancing_rate: float, call_price: float, market_clean: float | None, make_whole: bool = False) -> bool:
        if make_whole:
            return market_clean is not None and call_price < market_clean
        return refinancing_rate + self.threshold_bp / 1e4 < coupon and (market_clean is None or market_clean >= call_price - 0.5)


@dataclass
class DefaultModel:
    """Real-world annual default probabilities by rating bucket and recovery by seniority; not a
    substitute for risk-neutral credit calibration (blueprint section 3)."""

    annual_pd: dict[str, float] = field(default_factory=lambda: {"AAA-AA": 0.0003, "A": 0.0007, "BBB": 0.002, "BB": 0.009, "B": 0.035, "CCC-D": 0.25, "NR": 0.02})
    recovery: dict[str, float] = field(default_factory=lambda: {"SENIOR_SECURED": 0.6, "SENIOR_UNSECURED": 0.4, "SUBORDINATED": 0.25, "JUNIOR": 0.15, "PREFERRED": 0.1, "UNKNOWN": 0.4})
    recovery_lag_months: int = 12
    model_id: str = "rating-pd-v1"
    measure: str = "REAL_WORLD"

    def default_probability(self, bucket: str, horizon_years: float) -> float:
        pd_annual = self.annual_pd.get(bucket, 0.02)
        return 1.0 - (1.0 - pd_annual) ** horizon_years

    def expected_recovery(self, seniority: str) -> float:
        return self.recovery.get(seniority, 0.4)

    def recovery_date(self, default_date: date) -> date:
        from nautilus_fi.instruments.schedule import add_months

        return add_months(default_date, self.recovery_lag_months)


@dataclass
class DrawBehavior:
    """Revolver utilisation path: base utilisation rising under stress."""

    base_utilisation: float = 0.35
    stress_sensitivity: float = 0.5
    model_id: str = "revolver-draw-v1"

    def utilisation(self, stress: float) -> float:
        return min(1.0, self.base_utilisation + self.stress_sensitivity * max(stress, 0.0))


__all__ = ["DefaultModel", "DrawBehavior", "IssuerCallPolicy", "PSAModel", "PrepaymentModel", "RefinanceIncentiveModel", "cpr_to_smm"]
