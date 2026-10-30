"""Distressed and restructured claims (blueprint section 6): default state, contractual claims,
disputed interest, exchange terms and recovery distributions. Old debt becomes cash plus new debt
and equity with correct units and basis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.instruments.analytics import accrued_interest
from nautilus_fi.instruments.bond import BondSpec


@dataclass
class Claim:
    instrument: str
    principal_per_100: float
    prepetition_interest_per_100: float
    disputed_per_100: float = 0.0
    post_petition_interest_accrues: bool = False
    seniority: str = "SENIOR_UNSECURED"
    recovery_estimate_pct: float | None = None      # % of allowed claim
    expected_distribution: date | None = None

    @property
    def allowed_per_100(self) -> float:
        return self.principal_per_100 + self.prepetition_interest_per_100 - self.disputed_per_100


def claim_at_default(spec: BondSpec, default_date: date, factor: float = 1.0, disputed_per_100: float = 0.0) -> Claim:
    accrued = accrued_interest(spec, default_date) * factor
    return Claim(spec.symbol, 100.0 * factor, accrued, disputed_per_100, False, spec.seniority.value)


def claim_value_per_100(claim: Claim, as_of: date, discount_rate: float = 0.15) -> float:
    """Recovery estimate discounted to the expected distribution date at a distressed rate."""
    if claim.recovery_estimate_pct is None:
        raise ValueError("recovery estimate required")
    value = claim.allowed_per_100 * claim.recovery_estimate_pct
    if claim.expected_distribution and claim.expected_distribution > as_of:
        t = (claim.expected_distribution - as_of).days / 365.25
        value /= (1.0 + discount_rate) ** t
    return value


@dataclass
class ExchangeLeg:
    instrument: str
    kind: str                # CASH | DEBT | EQUITY | WARRANT
    ratio: float             # units of new per 100 face of old (cash: currency per 100)
    value_per_unit: float | None = None


@dataclass
class ExchangeTerms:
    old_instrument: str
    legs: list[ExchangeLeg] = field(default_factory=list)
    election_deadline: date | None = None
    consent_fee_per_100: float = 0.0
    minimum_participation: float | None = None

    def value_per_100(self, marks: dict[str, float]) -> float:
        total = self.consent_fee_per_100
        for leg in self.legs:
            if leg.kind == "CASH":
                total += leg.ratio
            else:
                px = marks.get(leg.instrument, leg.value_per_unit)
                if px is None:
                    raise LookupError(f"no value for {leg.instrument}")
                total += leg.ratio * px
        return total

    def new_positions(self, old_face: float) -> list[tuple[str, float, str]]:
        return [(leg.instrument, old_face / 100.0 * leg.ratio, leg.kind) for leg in self.legs if leg.kind != "CASH"]

    def cash(self, old_face: float) -> float:
        return old_face / 100.0 * (sum(leg.ratio for leg in self.legs if leg.kind == "CASH") + self.consent_fee_per_100)


__all__ = ["Claim", "ExchangeLeg", "ExchangeTerms", "claim_at_default", "claim_value_per_100"]
