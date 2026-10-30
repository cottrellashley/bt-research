"""Perpetuals and hybrid capital (blueprint section 6): coupon deferral or cancellation, discretionary
calls with coupon resets, write-down and conversion triggers. No redemption is assumed on the first
call date; an exercise policy decides."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class DeferralType(str, Enum):
    NONE = "NONE"
    CUMULATIVE = "CUMULATIVE"
    NON_CUMULATIVE = "NON_CUMULATIVE"


@dataclass
class HybridTerms:
    first_call: date | None
    call_frequency_years: float = 5.0
    reset_index: str = "UST5Y"
    reset_spread_bp: float = 300.0
    deferral: DeferralType = DeferralType.NON_CUMULATIVE
    trigger_metric: str | None = "CET1"
    trigger_level: float | None = 0.05125
    trigger_action: str = "WRITE_DOWN"             # WRITE_DOWN | CONVERSION
    write_down_fraction: float = 1.0
    conversion_ratio: float | None = None
    perpetual: bool = True
    issuer_call_economics_threshold_bp: float = 25.0


@dataclass
class IssuerCallPolicy:
    """Issuer calls when refinancing is cheaper than the reset coupon by more than a threshold, or
    when regulatory value is lost. Never assumes redemption at the first call date."""

    threshold_bp: float = 25.0
    regulatory_value_lost: bool = False

    def calls(self, reset_coupon: float, issuer_new_issue_rate: float) -> bool:
        if self.regulatory_value_lost:
            return True
        return issuer_new_issue_rate + self.threshold_bp / 1e4 < reset_coupon


def reset_coupon(terms: HybridTerms, benchmark_rate: float) -> float:
    return benchmark_rate + terms.reset_spread_bp / 1e4


def expected_redemption(terms: HybridTerms, as_of: date, benchmark_rate: float, issuer_new_issue_rate: float, policy: IssuerCallPolicy | None = None) -> date | None:
    """Model-dependent expected redemption date (None = treat as perpetual)."""
    policy = policy or IssuerCallPolicy(terms.issuer_call_economics_threshold_bp)
    if terms.first_call is None or terms.first_call <= as_of:
        return None
    return terms.first_call if policy.calls(reset_coupon(terms, benchmark_rate), issuer_new_issue_rate) else None


def trigger_breached(terms: HybridTerms, metric_value: float | None) -> bool:
    return terms.trigger_metric is not None and terms.trigger_level is not None and metric_value is not None and metric_value < terms.trigger_level


def deferral_consequence(terms: HybridTerms, coupon_per_100: float) -> tuple[float, float]:
    """(arrears added, coupon lost) for a deferred coupon."""
    if terms.deferral == DeferralType.CUMULATIVE:
        return coupon_per_100, 0.0
    if terms.deferral == DeferralType.NON_CUMULATIVE:
        return 0.0, coupon_per_100
    return 0.0, 0.0


__all__ = ["DeferralType", "HybridTerms", "IssuerCallPolicy", "deferral_consequence", "expected_redemption", "reset_coupon", "trigger_breached"]
