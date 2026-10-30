"""Tolerances chosen for the actual calculation: exact for identifiers and eligibility, currency
precision for finalized cash, justified numerical tolerances for pricing and simulation."""

from __future__ import annotations

from dataclasses import dataclass

from nautilus_fi.core.quantities import currency_precision


@dataclass(frozen=True)
class Tolerance:
    name: str
    absolute: float = 0.0
    relative: float = 0.0
    unit: str = ""
    rationale: str = ""


TOLERANCES: dict[str, Tolerance] = {
    "identifier": Tolerance("identifier", 0.0, 0.0, "exact", "identifiers and event eligibility compare exactly"),
    "cash_posting": Tolerance("cash_posting", 0.005, 0.0, "currency", "finalized cash postings at currency precision (half a cent)"),
    "nav_conservation": Tolerance("nav_conservation", 1e-4, 0.0, "currency", "assets - liabilities = equity, floating point only"),
    "accrued_interest": Tolerance("accrued_interest", 1e-9, 0.0, "per 100", "deterministic day-count arithmetic"),
    "bond_price_analytic": Tolerance("bond_price_analytic", 1e-6, 0.0, "per 100", "closed-form discounting"),
    "curve_repricing_bp": Tolerance("curve_repricing_bp", 0.5, 0.0, "bp", "calibration instruments reprice within half a basis point"),
    "lattice_price": Tolerance("lattice_price", 0.25, 0.0, "per 100", "trinomial tree discretisation vs external engine"),
    "oas_bp": Tolerance("oas_bp", 5.0, 0.0, "bp", "model-dependent spread; cross-engine disagreement is reported, not hidden"),
    "monte_carlo": Tolerance("monte_carlo", 0.0, 0.01, "relative", "sampling error at the declared path count"),
    "waterfall_conservation": Tolerance("waterfall_conservation", 1e-6, 0.0, "currency", "deal cash in equals cash out"),
    "replay_hash": Tolerance("replay_hash", 0.0, 0.0, "exact", "pinned replay reproduces event chain and outputs exactly"),
}


def within(name: str, a: float, b: float, currency: str | None = None) -> bool:
    t = TOLERANCES[name]
    if t.unit == "exact":
        return a == b
    tol = t.absolute
    if name == "cash_posting" and currency:
        tol = 0.51 * 10 ** (-currency_precision(currency))
    if t.relative:
        tol = max(tol, t.relative * max(abs(a), abs(b)))
    return abs(a - b) <= tol


__all__ = ["TOLERANCES", "Tolerance", "within"]
