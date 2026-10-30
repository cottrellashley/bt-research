"""Pricing engines (blueprint sections 8 and 13): a pricing engine maps a contract and state to a
value under a framework and returns values with diagnostics and applicability flags. Unavailable
outputs are reported as unavailable, never as zero."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

import numpy as np

from nautilus_fi.instruments.analytics import accrued_interest, price_from_spread
from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.cashflows import CashflowType, MarketInputs, project_cashflows
from nautilus_fi.instruments.curves import ZeroCurve, _bisect
from nautilus_fi.instruments.enums import CouponType


@dataclass
class PricingResult:
    model_id: str
    clean: float | None
    dirty: float | None
    currency: str
    diagnostics: dict = field(default_factory=dict)
    applicable: bool = True
    flags: tuple[str, ...] = ()
    sensitivities: dict[str, float] = field(default_factory=dict)
    unavailable: tuple[str, ...] = ()


class PricingEngine(Protocol):
    model_id: str
    version: str

    def supports(self, spec: BondSpec) -> bool: ...

    def price(self, spec: BondSpec, settle: date, curve: ZeroCurve, spread_bp: float = 0.0, inputs: MarketInputs | None = None, **params) -> PricingResult: ...


class DiscountingEngine:
    """Deterministic cashflows discounted off a curve plus spread. Optionality is *not* priced:
    callable or putable contracts get an explicit flag so the result is never mistaken for an OAS value."""

    model_id = "discounting"
    version = "1.0"

    def supports(self, spec: BondSpec) -> bool:
        return True

    def price(self, spec: BondSpec, settle: date, curve: ZeroCurve, spread_bp: float = 0.0, inputs: MarketInputs | None = None, **params) -> PricingResult:
        flags = []
        if spec.call_schedule or spec.put_schedule:
            flags.append("embedded options ignored (use HullWhiteTreeEngine for OAS)")
        if spec.coupon_type == CouponType.FLOATING and (inputs is None or not (inputs.fixings or inputs.daily_fixings or inputs.flat_forward_rate is not None)):
            flags.append("floating coupons projected at 0% index: supply fixings or a flat forward")
        clean = price_from_spread(spec, settle, curve, spread_bp, inputs)
        acc = accrued_interest(spec, settle, inputs)
        return PricingResult(self.model_id, clean, clean + acc, spec.currency, {"spread_bp": spread_bp, "curve": curve.name}, True, tuple(flags), unavailable=("oas", "effective_duration") if flags else ())


class HullWhiteTreeEngine:
    """One-factor Hull-White trinomial tree for callable / putable bonds: OAS, option-adjusted
    duration and convexity by full revaluation. Experimental; cross-check against QuantLib."""

    model_id = "hw1f-trinomial"
    version = "0.1"

    def __init__(self, a: float = 0.03, sigma: float = 0.01, steps_per_year: int = 24) -> None:
        self.a, self.sigma, self.steps_per_year = a, sigma, steps_per_year

    def supports(self, spec: BondSpec) -> bool:
        return bool(spec.call_schedule or spec.put_schedule) and spec.maturity is not None and spec.coupon_type in (CouponType.FIXED, CouponType.STEP, CouponType.ZERO)

    # ---- tree construction (Hull & White 1994 two-stage procedure) --------------------
    def _build(self, curve: ZeroCurve, T: float, spread: float = 0.0):
        n = max(int(np.ceil(T * self.steps_per_year)), 2)
        dt = T / n
        a, sigma = self.a, self.sigma
        dr = sigma * np.sqrt(3.0 * dt)
        jmax = int(np.ceil(0.184 / (a * dt))) if a > 0 else n
        js = np.arange(-jmax, jmax + 1)
        # branching probabilities per node (standard, with modified branching at the edges)
        pu, pm, pd = np.zeros(len(js)), np.zeros(len(js)), np.zeros(len(js))
        M = -a * dt
        for idx, j in enumerate(js):
            if j == jmax:
                pu[idx] = 7 / 6 + (j * j * M * M + 3 * j * M) / 2
                pm[idx] = -1 / 3 - j * j * M * M - 2 * j * M
                pd[idx] = 1 / 6 + (j * j * M * M + j * M) / 2
            elif j == -jmax:
                pu[idx] = 1 / 6 + (j * j * M * M - j * M) / 2
                pm[idx] = -1 / 3 - j * j * M * M + 2 * j * M
                pd[idx] = 7 / 6 + (j * j * M * M - 3 * j * M) / 2
            else:
                pu[idx] = 1 / 6 + (j * j * M * M + j * M) / 2
                pm[idx] = 2 / 3 - j * j * M * M
                pd[idx] = 1 / 6 + (j * j * M * M - j * M) / 2
        # Arrow-Debreu prices and alpha shifts to fit discount factors
        alpha = np.zeros(n + 1)
        Q = np.zeros((n + 1, len(js)))
        Q[0, jmax] = 1.0
        dfs = np.array([curve.df((i + 1) * dt) * np.exp(-spread * (i + 1) * dt) for i in range(n + 1)])
        for i in range(n + 1):
            # alpha_i such that sum_j Q[i,j] exp(-(alpha_i + j dr) dt) = P(0, (i+1) dt)
            s = np.sum(Q[i] * np.exp(-js * dr * dt))
            alpha[i] = (np.log(s) - np.log(dfs[i])) / dt
            if i < n:
                for idx, j in enumerate(js):
                    if Q[i, idx] == 0.0:
                        continue
                    disc = np.exp(-(alpha[i] + j * dr) * dt)
                    if j == jmax:
                        targets = (idx, idx - 1, idx - 2)
                    elif j == -jmax:
                        targets = (idx + 2, idx + 1, idx)
                    else:
                        targets = (idx + 1, idx, idx - 1)
                    for tgt, p in zip(targets, (pu[idx], pm[idx], pd[idx])):
                        Q[i + 1, tgt] += Q[i, idx] * p * disc
        return n, dt, js, dr, alpha, pu, pm, pd, jmax

    def _price_tree(self, spec: BondSpec, settle: date, curve: ZeroCurve, spread: float, inputs: MarketInputs | None) -> tuple[float, float]:
        flows = project_cashflows(spec, settle, inputs)
        T = max((spec.maturity - settle).days / 365.25, 1 / 365.25)
        n, dt, js, dr, alpha, pu, pm, pd, jmax = self._build(curve, T, spread)
        cf_at = np.zeros(n + 1)
        for f in flows:
            k = min(int(round((f.date - settle).days / 365.25 / dt)), n)
            cf_at[k] += f.amount
        calls = {min(int(round((c.date - settle).days / 365.25 / dt)), n): c.price for c in spec.call_schedule if c.date > settle}
        puts = {min(int(round((p.date - settle).days / 365.25 / dt)), n): p.price for p in spec.put_schedule if p.date > settle}
        V = np.zeros(len(js)) + cf_at[n]                 # terminal: final coupon + redemption
        V_nocall = V.copy()
        for i in range(n - 1, -1, -1):
            disc = np.exp(-(alpha[i] + js * dr) * dt)
            def roll(Vn):
                out = np.zeros(len(js))
                for idx, j in enumerate(js):
                    if j == jmax:
                        targets = (idx, idx - 1, idx - 2)
                    elif j == -jmax:
                        targets = (idx + 2, idx + 1, idx)
                    else:
                        targets = (idx + 1, idx, idx - 1)
                    out[idx] = disc[idx] * (pu[idx] * Vn[targets[0]] + pm[idx] * Vn[targets[1]] + pd[idx] * Vn[targets[2]])
                return out
            V = roll(V) + cf_at[i]
            V_nocall = roll(V_nocall) + cf_at[i]
            accrued_i = accrued_interest(spec, settle) if i == 0 else 0.0
            if i in calls and i > 0:
                V = np.minimum(V, calls[i] + cf_at[i] - 0.0)      # issuer calls when continuation exceeds call price (plus current coupon)
            if i in puts and i > 0:
                V = np.maximum(V, puts[i] + cf_at[i])
            _ = accrued_i
        mid = jmax
        return float(V[mid]), float(V_nocall[mid])

    def price(self, spec: BondSpec, settle: date, curve: ZeroCurve, spread_bp: float = 0.0, inputs: MarketInputs | None = None, **params) -> PricingResult:
        if not self.supports(spec):
            return PricingResult(self.model_id, None, None, spec.currency, applicable=False, flags=("not a fixed-rate callable/putable bond",), unavailable=("clean", "dirty", "oas"))
        dirty, dirty_nocall = self._price_tree(spec, settle, curve, spread_bp / 1e4, inputs)
        acc = accrued_interest(spec, settle, inputs)
        return PricingResult(self.model_id, dirty - acc, dirty, spec.currency, {"a": self.a, "sigma": self.sigma, "steps_per_year": self.steps_per_year, "option_value": dirty_nocall - dirty}, True, ("experimental lattice; validate against QuantLib",))

    def oas(self, spec: BondSpec, settle: date, curve: ZeroCurve, clean_price: float, inputs: MarketInputs | None = None) -> dict:
        acc = accrued_interest(spec, settle, inputs)
        target = clean_price + acc

        def f(s: float) -> float:
            return self._price_tree(spec, settle, curve, s, inputs)[0] - target

        oas = _bisect(f, -0.05, 0.30, tol=1e-7)
        up = self._price_tree(spec, settle, curve.shifted(10.0), oas, inputs)[0]
        dn = self._price_tree(spec, settle, curve.shifted(-10.0), oas, inputs)[0]
        eff_dur = (dn - up) / (2 * target * 0.001)
        eff_cvx = (up + dn - 2 * target) / (target * 0.001 ** 2)
        return {"oas_bp": oas * 1e4, "effective_duration": eff_dur, "effective_convexity": eff_cvx, "model_id": self.model_id, "steps_per_year": self.steps_per_year}


class StructuredTrancheEngine:
    """Price a tranche from projected deal cashflows: discount margin / spread over the projection curve."""

    model_id = "tranche-dm"
    version = "0.1"

    def price_from_flows(self, flows: list[tuple[date, float]], settle: date, curve: ZeroCurve, spread_bp: float, current_face: float) -> dict:
        pv = 0.0
        wal_num = 0.0
        prin = 0.0
        for d, amt in flows:
            t = (d - settle).days / 365.25
            if t <= 0:
                continue
            pv += amt * curve.df(t) * np.exp(-spread_bp / 1e4 * t)
        return {"dirty_value": pv, "price_pct_current": pv / current_face * 100.0 if current_face else float("nan"), "spread_bp": spread_bp, "wal": wal_num / prin if prin else None}

    def spread_from_price(self, flows: list[tuple[date, float]], settle: date, curve: ZeroCurve, dirty_value: float, current_face: float) -> float:
        return _bisect(lambda s: self.price_from_flows(flows, settle, curve, s, current_face)["dirty_value"] - dirty_value, -200.0, 5000.0, tol=1e-6)


class EngineRegistry:
    def __init__(self) -> None:
        self.engines: list = [HullWhiteTreeEngine(), DiscountingEngine()]

    def choose(self, spec: BondSpec):
        for e in self.engines:
            if e.supports(spec):
                return e
        return DiscountingEngine()


__all__ = ["CashflowType", "DiscountingEngine", "EngineRegistry", "HullWhiteTreeEngine", "PricingEngine", "PricingResult", "StructuredTrancheEngine"]
