"""assetiq: factor-space credit allocation (gap analysis §5.7; read from a blurry photo, so less
certain than the other sections).

Canonical conventions (no legacy profile). The components:

* ``to_factor`` / ``from_factor``: positions in factor space (x = w x DTS) and back;
* ``objective``: MaxSharpe and MaxIR (frontier searches over a base optimiser);
* ``tev_band``: a TEV budget and floor: the frontier searched between them only;
* ``hierarchy``: parent / child exposure limits (a sector and its sub-sectors);
* ``tilt_bands``: signal-to-tilt bands;
* ``kde_copula_scenarios``: KDE marginals joined by a Gaussian copula, as scenarios for ``CVaR``;
* ``risk_report``: ex-ante risk (TEV, contributions, marginal TEV, budget use).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance

PROFILE = None
REQUIREMENTS = ("factor-space positions", "MaxSharpe / MaxIR", "TEV budget and floor", "parent/child hierarchy", "signal-to-tilt bands", "KDE + Gaussian copula", "ex-ante risk")


def to_factor(weights: pd.Series, dts: pd.Series) -> pd.Series:
    """Positions in DTS space: x = w x DTS (a name's contribution to spread risk)."""
    return weights * dts.reindex(weights.index)


def from_factor(x: pd.Series, dts: pd.Series) -> pd.Series:
    """Weights from DTS-space positions (names without DTS get none)."""
    d = dts.reindex(x.index)
    return (x / d.where(d.abs() > 1e-12)).fillna(0.0)


def objective(kind: str, base, model, levels=None, periods_per_year: float = 252.0):
    """The frontier search over ``base`` (an ``Optimise``): ``"sharpe"`` (``MaxSharpe``) or
    ``"ir"`` (``MaxIR``: active risk) across the risk ``levels``."""
    from nautilus_fi.kernel.portfolio import MaxIR, MaxSharpe

    kw = {"periods_per_year": periods_per_year} | ({"levels": tuple(levels)} if levels is not None else {})
    if kind.lower() in ("sharpe", "maxsharpe"):
        return MaxSharpe(base, model, **kw)
    if kind.lower() in ("ir", "maxir"):
        return MaxIR(base, model, **kw)
    raise ValueError(f"unknown objective {kind!r}: sharpe or ir")


def tev_band(base, model, floor: float, budget: float, points: int = 6, kind: str = "ir", periods_per_year: float = 252.0):
    """A TEV budget and floor: the frontier search (``MaxIR`` / ``MaxSharpe``) over active-risk
    levels from ``floor`` to ``budget`` only, so the chosen portfolio spends at least the floor
    and at most the budget. A floor above the risk of the most active portfolio the constraints
    allow cannot bind: check the result's TEV against it."""
    if not 0 < floor <= budget:
        raise ValueError(f"the floor {floor} must lie in (0, budget {budget}]")
    levels = tuple(float(x) for x in np.linspace(floor, budget, max(int(points), 2))) if budget > floor else (float(budget),)
    return objective(kind, base, model, levels, periods_per_year)


def hierarchy(parent: str, child: str, parent_band: float, child_band: float) -> list:
    """Active weight limits at two levels: each ``parent`` group within +/-``parent_band`` and
    each ``child`` group within +/-``child_band`` of the benchmark."""
    from nautilus_fi.kernel.portfolio import ActiveWeight

    return [ActiveWeight(parent).within(parent_band), ActiveWeight(child).within(child_band)]


def tilt_bands(score: pd.Series | pd.DataFrame, bands: list[tuple[float, float, float]]):
    """A signal mapped to tilts by bands ``(lo, hi, tilt)``: lo <= s < hi -> tilt; outside every
    band, no tilt."""
    s = score.astype(float)
    out = s * 0.0
    for lo, hi, tilt in bands:
        out = out.mask((s >= lo) & (s < hi), float(tilt))
    return out.where(s.notna())


def kde_copula_scenarios(returns: pd.DataFrame, n: int = 2000, seed: int = 0, bandwidth: str | float = "scott") -> pd.DataFrame:
    """Scenarios from KDE marginals joined by a Gaussian copula: the dependence is the normal
    copula of the rank correlations, each marginal its own kernel density."""
    from scipy.stats import gaussian_kde, norm

    X = returns.dropna(how="any")
    if len(X) < 10:
        raise ValueError("at least 10 complete rows are needed")
    u = (X.rank() - 0.5) / len(X)
    z = norm.ppf(u.to_numpy())
    C = np.corrcoef(z, rowvar=False) if X.shape[1] > 1 else np.ones((1, 1))
    C = C + 1e-9 * np.eye(len(C))
    rng = np.random.default_rng(seed)
    g = rng.multivariate_normal(np.zeros(len(C)), C, size=n)
    U = norm.cdf(g)
    out = np.empty_like(U)
    for j, c in enumerate(X.columns):
        kde = gaussian_kde(X[c].to_numpy(), bw_method=bandwidth)
        draws = np.sort(kde.resample(max(4 * n, 4000), seed=seed + j).ravel())
        out[:, j] = np.quantile(draws, U[:, j])
    return pd.DataFrame(out, columns=X.columns)


class KDECopulaScenarios:
    """A ``CVaR`` scenario provider: KDE + Gaussian copula over the last ``window`` rows of a
    return field, as known at the decision."""

    def __init__(self, window: int = 500, field: str = "xr", n: int = 2000, seed: int = 0) -> None:
        self.window, self.field, self.n, self.seed = int(window), field, int(n), int(seed)

    def names(self):
        return {self.field}

    def __call__(self, cs) -> pd.DataFrame:
        ctx = getattr(cs, "ctx", None)
        if ctx is None:
            raise ValueError("KDECopulaScenarios needs a kernel cross-section")
        X = np.asarray(ctx.field(self.field), dtype=float)[max(0, cs.t - self.window + 1): cs.t + 1][:, cs.cols]
        hist = pd.DataFrame(np.nan_to_num(X), columns=cs.ids)
        return kde_copula_scenarios(hist, self.n, self.seed)

    def __repr__(self) -> str:
        return f"KDECopulaScenarios({self.window}, {self.field!r}, n={self.n})"


def risk_report(weights: pd.Series, benchmark: pd.Series | None, model, periods_per_year: float = 252.0, groups: pd.Series | None = None, limit: float | None = None, carry: pd.Series | None = None) -> dict:
    """Ex-ante risk: TEV, contributions by name and group, marginal TEV, budget use."""
    from nautilus_fi.kernel.analytics import ex_ante_risk

    return ex_ante_risk(weights, benchmark, model, periods_per_year, groups, limit, carry)


PARITY = [ParityCase("assetiq allocation", "assetiq", "not named in the gap analysis: the framework's stored allocations", (Tolerance("weights", abs=1e-4),))]

__all__ = ["PARITY", "PROFILE", "REQUIREMENTS", "KDECopulaScenarios", "from_factor", "hierarchy", "kde_copula_scenarios", "objective", "risk_report", "tev_band", "tilt_bands", "to_factor"]
