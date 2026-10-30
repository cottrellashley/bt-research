"""Discount / zero curves, par bootstrapping, Nelson-Siegel-Svensson fits and spread curves.

Everything here is pure numpy so it can run inside strategies and the offline pricing pipeline.
Curves must be built from instruments observable on ``as_of`` only (never fit on later data).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        # expand bracket a few times before giving up
        for _ in range(20):
            lo, hi = lo - (hi - lo), hi + (hi - lo)
            flo, fhi = f(lo), f(hi)
            if flo * fhi <= 0:
                break
        else:
            raise ValueError("root not bracketed")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < tol or (hi - lo) < tol:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


@dataclass
class ZeroCurve:
    """Continuously-compounded zero curve with linear interpolation in zero rates."""

    as_of: date
    tenors: np.ndarray
    zero_rates: np.ndarray
    name: str = "GOVT"

    def __post_init__(self) -> None:
        self.tenors = np.asarray(self.tenors, dtype=float)
        self.zero_rates = np.asarray(self.zero_rates, dtype=float)
        if self.tenors.ndim != 1 or self.tenors.shape != self.zero_rates.shape:
            raise ValueError("tenors and zero_rates must be 1-D arrays of equal length")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        if np.any(self.tenors <= 0):
            raise ValueError("tenors must be positive")

    # -- queries -----------------------------------------------------------------------------
    def zero(self, t: float | np.ndarray) -> float | np.ndarray:
        t = np.asarray(t, dtype=float)
        out = np.interp(t, self.tenors, self.zero_rates)  # flat extrapolation
        return float(out) if out.ndim == 0 else out

    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        out = np.exp(-np.asarray(self.zero(t_arr)) * np.maximum(t_arr, 0.0))
        return float(out) if out.ndim == 0 else out

    def forward(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 must exceed t1")
        return (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)

    def par_yield(self, t: float, freq: int = 2) -> float:
        """Par coupon (annual decimal) for a bond maturing in ``t`` years paying ``freq`` times a year."""
        if t <= 0:
            return float(self.zero_rates[0])
        if freq <= 0 or t < 1.0 / freq:
            d = self.df(t)
            return (1.0 / d - 1.0) / t
        n = int(round(t * freq))
        times = np.arange(1, n + 1) / freq
        dfs = np.asarray(self.df(times))
        return float(freq * (1.0 - dfs[-1]) / dfs.sum())

    # -- transformations -----------------------------------------------------------------
    def shifted(self, bp: float) -> ZeroCurve:
        return ZeroCurve(self.as_of, self.tenors.copy(), self.zero_rates + bp / 1e4, self.name)

    def bumped(self, tenor: float, bp: float) -> ZeroCurve:
        """Key-rate bump at ``tenor``: the pillar is inserted (interpolated) if absent, then shifted.

        With linear interpolation in zero rates the bump is triangular between the neighbouring
        pillars, so key-rate durations over a set of tenors partition the parallel-shift duration.
        """
        tenors = self.tenors
        rates = self.zero_rates
        if not np.any(np.isclose(tenors, tenor)):
            z = float(np.interp(tenor, tenors, rates))
            tenors = np.sort(np.append(tenors, tenor))
            rates = np.interp(tenors, self.tenors, self.zero_rates)
            rates[np.isclose(tenors, tenor)] = z
        idx = int(np.argmin(np.abs(tenors - tenor)))
        rates = rates.copy()
        rates[idx] += bp / 1e4
        return ZeroCurve(self.as_of, tenors.copy(), rates, self.name)

    def with_pillars(self, tenors: Sequence[float]) -> ZeroCurve:
        """Insert pillars (interpolated, so the curve is unchanged) e.g. a common key-rate grid."""
        all_t = np.unique(np.concatenate([self.tenors, np.asarray(tenors, dtype=float)]))
        return ZeroCurve(self.as_of, all_t, np.interp(all_t, self.tenors, self.zero_rates), self.name)

    def with_as_of(self, as_of: date) -> ZeroCurve:
        return ZeroCurve(as_of, self.tenors.copy(), self.zero_rates.copy(), self.name)

    # -- constructors --------------------------------------------------------------------
    @classmethod
    def flat(cls, as_of: date, rate: float, name: str = "FLAT") -> ZeroCurve:
        return cls(as_of, np.array([0.25, 1.0, 5.0, 10.0, 30.0]), np.full(5, float(rate)), name)

    @classmethod
    def from_par_yields(
        cls, as_of: date, tenors: Sequence[float], par_yields: Sequence[float], freq: int = 2, name: str = "GOVT"
    ) -> ZeroCurve:
        return bootstrap_par_curve(as_of, tenors, par_yields, freq=freq, name=name)

    # -- interop ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"as_of": self.as_of.isoformat(), "name": self.name, "tenors": self.tenors.tolist(), "zero_rates": self.zero_rates.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> ZeroCurve:
        return cls(date.fromisoformat(d["as_of"]), np.array(d["tenors"]), np.array(d["zero_rates"]), d.get("name", "GOVT"))

    def to_nautilus(self, ts_event: int, ts_init: int):
        """Nautilus ``YieldCurve`` custom data (cacheable through ``cache.add_yield_curve``)."""
        from nautilus_trader.model import YieldCurve

        return YieldCurve(self.name, self.tenors.tolist(), self.zero_rates.tolist(), ts_event, ts_init)


def bootstrap_par_curve(
    as_of: date, tenors: Sequence[float], par_yields: Sequence[float], freq: int = 2, name: str = "GOVT"
) -> ZeroCurve:
    """Bootstrap continuously compounded zero rates from par yields (decimals).

    Pillars shorter than one coupon period are treated as money-market (simple) yields. Longer pillars
    are solved sequentially so each par bond reprices to 100, interpolating linearly in zero rates
    between pillars.
    """
    tenors_arr = np.asarray(tenors, dtype=float)
    yields_arr = np.asarray(par_yields, dtype=float)
    order = np.argsort(tenors_arr)
    tenors_arr, yields_arr = tenors_arr[order], yields_arr[order]
    known_t: list[float] = []
    known_z: list[float] = []
    for T, c in zip(tenors_arr, yields_arr):
        if T < 1.0 / freq - 1e-9:
            d = 1.0 / (1.0 + c * T)
            known_t.append(float(T))
            known_z.append(float(-np.log(d) / T))
            continue
        n = int(round(T * freq))
        times = np.arange(1, n + 1) / freq
        times[-1] = T

        def price_minus_par(z: float) -> float:
            tt = np.array(known_t + [T])
            zz = np.array(known_z + [z])
            zeros = np.interp(times, tt, zz)
            dfs = np.exp(-zeros * times)
            return float((c / freq) * dfs.sum() + dfs[-1] - 1.0)

        z_star = _bisect(price_minus_par, -0.2, 1.0)
        known_t.append(float(T))
        known_z.append(float(z_star))
    return ZeroCurve(as_of, np.array(known_t), np.array(known_z), name)


@dataclass
class NelsonSiegelSvensson:
    """NSS parametric yield curve (rates as decimals, tenors in years)."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float = 0.0
    tau1: float = 1.5
    tau2: float = 10.0

    @staticmethod
    def _loadings(t: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
        t = np.maximum(np.asarray(t, dtype=float), 1e-6)
        x1, x2 = t / tau1, t / tau2
        l1 = (1 - np.exp(-x1)) / x1
        l2 = l1 - np.exp(-x1)
        l3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
        return np.column_stack([np.ones_like(t), l1, l2, l3])

    def yield_(self, t: float | np.ndarray) -> float | np.ndarray:
        L = self._loadings(np.atleast_1d(t), self.tau1, self.tau2)
        y = L @ np.array([self.beta0, self.beta1, self.beta2, self.beta3])
        return float(y[0]) if np.isscalar(t) else y

    @classmethod
    def fit(
        cls,
        tenors: Sequence[float],
        yields: Sequence[float],
        weights: Sequence[float] | None = None,
        tau1_grid: Sequence[float] = (0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0),
        tau2_grid: Sequence[float] = (3.0, 5.0, 8.0, 12.0, 18.0, 25.0),
    ) -> NelsonSiegelSvensson:
        t = np.asarray(tenors, dtype=float)
        y = np.asarray(yields, dtype=float)
        w = np.ones_like(y) if weights is None else np.sqrt(np.asarray(weights, dtype=float))
        best = None
        for tau1 in tau1_grid:
            for tau2 in tau2_grid:
                if tau2 <= tau1:
                    continue
                L = cls._loadings(t, tau1, tau2)
                beta, *_ = np.linalg.lstsq(L * w[:, None], y * w, rcond=None)
                sse = float(np.sum(((L @ beta - y) * w) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, beta, tau1, tau2)
        assert best is not None
        _, beta, tau1, tau2 = best
        return cls(float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3]), tau1, tau2)

    def to_zero_curve(self, as_of: date, tenors: Sequence[float] = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30), name: str = "GOVT") -> ZeroCurve:
        t = np.asarray(tenors, dtype=float)
        return ZeroCurve(as_of, t, np.asarray(self.yield_(t)), name)


@dataclass
class SpreadCurve:
    """Spread (bp) versus duration for one bucket (issuer, or sector x rating), piecewise linear."""

    knots: np.ndarray
    spreads_bp: np.ndarray
    n_obs: int = 0
    key: str = ""

    def __post_init__(self) -> None:
        self.knots = np.asarray(self.knots, dtype=float)
        self.spreads_bp = np.asarray(self.spreads_bp, dtype=float)

    def spread(self, duration: float) -> float:
        return float(np.interp(duration, self.knots, self.spreads_bp))

    @classmethod
    def fit(
        cls,
        durations: Sequence[float],
        spreads_bp: Sequence[float],
        weights: Sequence[float] | None = None,
        knots: Sequence[float] = (0.5, 2.0, 4.0, 7.0, 10.0, 20.0),
        prior: SpreadCurve | None = None,
        shrinkage: float = 2.0,
        key: str = "",
    ) -> SpreadCurve:
        """Weighted least squares on a hat-function basis with ridge shrinkage toward ``prior``.

        With few observations the fit collapses to the prior (e.g. the sector-rating curve when an
        issuer has too few liquid bonds), which is the matrix-pricing fallback described in the
        reference architecture.
        """
        d = np.asarray(durations, dtype=float)
        s = np.asarray(spreads_bp, dtype=float)
        k = np.asarray(knots, dtype=float)
        w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
        A = np.zeros((len(d), len(k)))
        for j in range(len(k)):
            basis = np.zeros_like(k)
            basis[j] = 1.0
            A[:, j] = np.interp(d, k, basis)
        prior_vals = prior.spreads_bp if prior is not None and len(prior.knots) == len(k) else np.full(len(k), float(np.average(s, weights=w)) if len(s) else 0.0)
        if prior is not None and len(prior.knots) != len(k):
            prior_vals = np.array([prior.spread(x) for x in k])
        lam = shrinkage
        lhs = A.T @ (A * w[:, None]) + lam * np.eye(len(k))
        rhs = A.T @ (s * w) + lam * prior_vals
        x = np.linalg.solve(lhs, rhs)
        return cls(k, x, n_obs=len(d), key=key)


__all__ = ["NelsonSiegelSvensson", "SpreadCurve", "ZeroCurve", "bootstrap_par_curve"]
_ = field
