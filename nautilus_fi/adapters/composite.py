"""New composite score / xrv: bond-level signal research (gap analysis §5.6).

Canonical conventions (no legacy profile). The components:

* ``forward_excess``: bond-level forward excess returns with exit, missing and left-index flags;
* ``investible``: the investible OAS cap as a universe filter;
* ``quintile_spread``: Q5 - Q1 as mean, median and winsorised mean with monotonicity flags,
  optionally equal-issuer weighted;
* ``nw_tstats``: Newey-West t-statistics at 13 and 26 lags;
* ``bootstrap``: a moving-block bootstrap interval;
* ``acceptance``: pre-registered acceptance (declared subperiods, DEV / holdout, contamination
  by earlier trials, the prospective shadow verdict).
"""

from __future__ import annotations

import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance
from nautilus_fi.kernel.expr import Field

PROFILE = None
REQUIREMENTS = ("forward excess returns", "investible OAS cap", "Q5-Q1", "equal-issuer weighting", "Newey-West 13/26", "moving-block bootstrap", "pre-registered acceptance")


def forward_excess(tr: pd.DataFrame, ir: pd.DataFrame | None = None, horizons=(21,), lag: int = 1, alive: pd.DataFrame | None = None, membership: pd.DataFrame | None = None, dates=None) -> pd.DataFrame:
    """Per bond, date and horizon: forward total, rate and excess returns starting ``lag``
    periods on, with ``exit`` / ``missing`` / ``left_index`` flags."""
    from nautilus_fi.kernel.returns import forward_returns

    return forward_returns(tr, horizons, ir, lag, alive, membership, dates)


def investible(oas_cap_bp: float, field: str = "oas_bp"):
    """Names with OAS at most ``oas_cap_bp`` (distressed names out of the investible set)."""
    return Field(field) <= float(oas_cap_bp)


def quintile_spread(scores: pd.DataFrame, fwd: pd.DataFrame, issuers: pd.Series | None = None, n_quantiles: int = 5, winsor: float = 0.05) -> dict:
    """Top minus bottom quantile of forward returns (mean, median, winsorised mean), the quantile
    means and monotonicity flags; ``issuers``: equal-issuer weighting."""
    from nautilus_fi.kernel.analytics import signal_extras

    out = signal_extras(scores, fwd, n_quantiles=n_quantiles, issuers=issuers, winsor=winsor)
    return {k: v for k, v in out.items() if k.startswith(("spread", "quantile", "monoton", "top_minus"))} or out


def nw_tstats(x: pd.Series, lags=(13, 26)) -> dict:
    """Newey-West t-statistics of the mean of ``x`` at each lag."""
    from nautilus_fi.research.statistics import newey_west_tstat

    return {int(lag): float(newey_west_tstat(x.dropna(), lags=int(lag))) for lag in lags}


def bootstrap(x: pd.Series, stat=None, block_len: int = 13, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """A moving-block bootstrap interval (point, low, high) for ``stat`` (the mean by default)."""
    from nautilus_fi.research.statistics import block_bootstrap_ci

    return block_bootstrap_ci(x.dropna(), stat=stat or (lambda r, *a, **k: float(pd.Series(r).mean())), block_len=block_len, n_boot=n_boot, alpha=alpha, seed=seed)


def acceptance(result, ledger=None, hypothesis=None, subperiods=None, shadow_start=None, **kw) -> dict:
    """The pre-registered acceptance rules: subperiod statistics, contamination and the shadow
    verdict (``validation.preregistered``)."""
    from nautilus_fi.kernel.validation import preregistered

    return preregistered(result, ledger, hypothesis, subperiods, shadow_start, **kw)


PARITY = [ParityCase("composite score study", "composite", "not named in the gap analysis: the study's stored quintile tables", (Tolerance("table:spread", abs=1e-6),))]

__all__ = ["PARITY", "PROFILE", "REQUIREMENTS", "acceptance", "bootstrap", "forward_excess", "investible", "nw_tstats", "quintile_spread"]
