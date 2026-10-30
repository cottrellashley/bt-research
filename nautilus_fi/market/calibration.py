"""Calibration recipes: instruments, quotes, weights, exclusions, interpolation, solver, residuals and
status, using only the decision's permitted information set. Residuals are reported by tenor; a
failed or unstable calibration yields an explicit status instead of yesterday's curve."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd

from nautilus_fi.instruments.curves import NelsonSiegelSvensson, ZeroCurve, bootstrap_par_curve
from nautilus_fi.instruments.derivatives import hazard_from_spread


class CalibrationStatus(str, Enum):
    OK = "OK"
    PARTIAL = "PARTIAL"          # some instruments excluded / residuals above tolerance
    STALE = "STALE"              # carried forward explicitly with age
    FAILED = "FAILED"


@dataclass
class CalibrationRecipe:
    curve_name: str
    as_of: date
    method: str
    instruments: list[dict] = field(default_factory=list)          # tenor, quote, type, source, weight, included, source_time
    interpolation: str = "linear-zero"
    extrapolation: str = "flat"
    solver: str = "bisection"
    objective: str = "exact repricing"
    tolerance_bp: float = 0.5
    constraints: str = "none"
    calendar_version: str = "US_BOND"
    starting_point: str = ""
    parameters: dict = field(default_factory=dict)
    residuals_bp: dict[float, float] = field(default_factory=dict)
    max_residual_bp: float = 0.0
    negative_forwards: bool = False
    status: CalibrationStatus = CalibrationStatus.FAILED
    message: str = ""
    info_cutoff: str = ""
    lineage: tuple[str, ...] = ()
    elapsed_ms: float = 0.0
    age_days: int = 0

    def to_record(self) -> dict:
        return {"curve": self.curve_name, "as_of": self.as_of, "method": self.method, "status": self.status.value, "n_instruments": sum(1 for i in self.instruments if i.get("included", True)), "n_excluded": sum(1 for i in self.instruments if not i.get("included", True)), "max_residual_bp": self.max_residual_bp, "negative_forwards": self.negative_forwards, "message": self.message, "info_cutoff": self.info_cutoff, "elapsed_ms": self.elapsed_ms, "age_days": self.age_days}


class CurveCalibrator:
    def __init__(self, allow_negative_rates: bool = True, residual_tolerance_bp: float = 0.5) -> None:
        self.allow_negative_rates = allow_negative_rates
        self.tolerance = residual_tolerance_bp

    def _check_forwards(self, curve: ZeroCurve) -> bool:
        t = curve.tenors
        fwds = [curve.forward(t[i], t[i + 1]) for i in range(len(t) - 1)]
        return any(f < 0 for f in fwds)

    def bootstrap_par(self, name: str, as_of: date, quotes: list[dict], freq: int = 2, info_cutoff: str = "", lineage: tuple[str, ...] = ()) -> tuple[ZeroCurve | None, CalibrationRecipe]:
        """``quotes``: [{tenor, quote (decimal par yield), source, weight, source_time}]"""
        t0 = time.perf_counter()
        recipe = CalibrationRecipe(name, as_of, "par-bootstrap", [dict(q, included=q.get("included", True)) for q in quotes], info_cutoff=info_cutoff, lineage=lineage, calendar_version="n/a")
        included = [q for q in recipe.instruments if q["included"] and q.get("quote") is not None and np.isfinite(q["quote"])]
        if len(included) < 2:
            recipe.status, recipe.message = CalibrationStatus.FAILED, f"only {len(included)} usable quotes"
            recipe.elapsed_ms = (time.perf_counter() - t0) * 1e3
            return None, recipe
        try:
            tenors = [float(q["tenor"]) for q in included]
            yields = [float(q["quote"]) for q in included]
            curve = bootstrap_par_curve(as_of, tenors, yields, freq=freq, name=name)
            res = {T: (curve.par_yield(T, freq) - y) * 1e4 for T, y in zip(tenors, yields)}
            recipe.residuals_bp = res
            recipe.max_residual_bp = max(abs(v) for v in res.values())
            recipe.negative_forwards = self._check_forwards(curve)
            if recipe.negative_forwards and not self.allow_negative_rates:
                recipe.status, recipe.message = CalibrationStatus.FAILED, "negative forwards not allowed by policy"
                return None, recipe
            recipe.status = CalibrationStatus.OK if recipe.max_residual_bp <= self.tolerance else CalibrationStatus.PARTIAL
            recipe.message = "repriced within tolerance" if recipe.status == CalibrationStatus.OK else "residuals above tolerance"
            return curve, recipe
        except Exception as e:  # noqa: BLE001 - calibration failure is a result, not a crash
            recipe.status, recipe.message = CalibrationStatus.FAILED, f"{type(e).__name__}: {e}"
            return None, recipe
        finally:
            recipe.elapsed_ms = (time.perf_counter() - t0) * 1e3

    def fit_nss(self, name: str, as_of: date, quotes: list[dict], info_cutoff: str = "", lineage: tuple[str, ...] = (), max_residual_bp: float = 15.0) -> tuple[ZeroCurve | None, CalibrationRecipe]:
        t0 = time.perf_counter()
        recipe = CalibrationRecipe(name, as_of, "nelson-siegel-svensson", [dict(q, included=q.get("included", True)) for q in quotes], interpolation="parametric", solver="grid+OLS", objective="weighted least squares", info_cutoff=info_cutoff, lineage=lineage)
        included = [q for q in recipe.instruments if q["included"]]
        if len(included) < 5:
            recipe.status, recipe.message = CalibrationStatus.FAILED, "NSS needs at least 5 quotes"
            recipe.elapsed_ms = (time.perf_counter() - t0) * 1e3
            return None, recipe
        try:
            tenors = np.array([float(q["tenor"]) for q in included])
            yields = np.array([float(q["quote"]) for q in included])
            weights = np.array([float(q.get("weight", 1.0)) for q in included])
            nss = NelsonSiegelSvensson.fit(tenors, yields, weights)
            fitted = np.asarray(nss.yield_(tenors))
            recipe.parameters = {"beta0": nss.beta0, "beta1": nss.beta1, "beta2": nss.beta2, "beta3": nss.beta3, "tau1": nss.tau1, "tau2": nss.tau2}
            recipe.residuals_bp = {float(t): float(r) * 1e4 for t, r in zip(tenors, fitted - yields)}
            recipe.max_residual_bp = float(np.max(np.abs(fitted - yields)) * 1e4)
            curve = nss.to_zero_curve(as_of, name=name)
            recipe.negative_forwards = self._check_forwards(curve)
            if recipe.max_residual_bp > max_residual_bp:
                recipe.status, recipe.message = CalibrationStatus.FAILED, f"max residual {recipe.max_residual_bp:.1f}bp exceeds {max_residual_bp}bp"
                return None, recipe
            recipe.status = CalibrationStatus.OK if recipe.max_residual_bp <= 2.0 else CalibrationStatus.PARTIAL
            recipe.message = "fit ok"
            return curve, recipe
        except Exception as e:  # noqa: BLE001
            recipe.status, recipe.message = CalibrationStatus.FAILED, f"{type(e).__name__}: {e}"
            return None, recipe
        finally:
            recipe.elapsed_ms = (time.perf_counter() - t0) * 1e3

    def credit_from_cds(self, entity: str, as_of: date, spreads: list[dict], discount: ZeroCurve, recovery: float = 0.4) -> tuple[dict[float, float], CalibrationRecipe]:
        """Term structure of flat hazards per CDS tenor (piecewise constant proxy)."""
        t0 = time.perf_counter()
        recipe = CalibrationRecipe(f"CDS:{entity}", as_of, "cds-hazard-bootstrap", [dict(q, included=True) for q in spreads], solver="bisection")
        hazards: dict[float, float] = {}
        try:
            for q in spreads:
                hazards[float(q["tenor"])] = hazard_from_spread(float(q["quote"]), recovery, discount, float(q["tenor"]))
            recipe.parameters = {"recovery": recovery, "hazards": hazards}
            recipe.status, recipe.message = CalibrationStatus.OK, "ok"
        except Exception as e:  # noqa: BLE001
            recipe.status, recipe.message = CalibrationStatus.FAILED, f"{type(e).__name__}: {e}"
        recipe.elapsed_ms = (time.perf_counter() - t0) * 1e3
        return hazards, recipe

    @staticmethod
    def carry_forward(previous: ZeroCurve, as_of: date, max_age_days: int = 3) -> tuple[ZeroCurve | None, CalibrationRecipe]:
        """Explicit stale carry: allowed only within ``max_age_days`` and always flagged STALE."""
        age = (as_of - previous.as_of).days
        recipe = CalibrationRecipe(previous.name, as_of, "carry-forward", status=CalibrationStatus.STALE, message=f"carried from {previous.as_of}", age_days=age)
        if age > max_age_days:
            recipe.status, recipe.message = CalibrationStatus.FAILED, f"stale beyond {max_age_days} days ({age})"
            return None, recipe
        return previous.with_as_of(as_of), recipe


class CalibrationLog:
    def __init__(self) -> None:
        self.recipes: list[CalibrationRecipe] = []

    def add(self, recipe: CalibrationRecipe) -> CalibrationRecipe:
        self.recipes.append(recipe)
        return recipe

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_record() for r in self.recipes])

    def failures(self) -> list[CalibrationRecipe]:
        return [r for r in self.recipes if r.status == CalibrationStatus.FAILED]


__all__ = ["CalibrationLog", "CalibrationRecipe", "CalibrationStatus", "CurveCalibrator"]
