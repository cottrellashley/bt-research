"""credit-pairs: relative-value pairs in spread units (gap analysis §5.9; read from a blurry
photo).

Canonical conventions (no legacy profile). The components:

* ``pair_pnl``: pair P&L in spread units with a rolling, lagged beta;
* ``imm_dates``: the IMM roll dates (20 March / June / September / December);
* ``zscore_rules``: entry / exit / stop / maximum-hold rules on the pair's z-score;
* ``dv01_neutral``: the second leg's notional that matches the first's BPV;
* ``etf_shares``: an ETF leg in whole shares (or lots);
* ``adf``: the augmented Dickey-Fuller test with MacKinnon p-values;
* ``bh_fdr``: Benjamini-Hochberg false-discovery control across many pairs.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance

PROFILE = None
REQUIREMENTS = ("spread-unit pairs with rolling beta", "IMM roll", "z-score rules", "DV01-neutral CDS sizing", "ETF sizing", "ADF", "BH-FDR")


def pair_pnl(a: pd.Series, b: pd.Series, beta: float | str = "rolling", window: int = 60, lag: int = 1) -> pd.DataFrame:
    from nautilus_fi.kernel.returns import pair_spread_pnl

    return pair_spread_pnl(a, b, beta, window, lag)


def imm_dates(start, end) -> list[date]:
    """IMM dates (the 20th of March, June, September and December) in [start, end]."""
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    out = [date(y, m, 20) for y in range(s.year, e.year + 1) for m in (3, 6, 9, 12)]
    return [d for d in out if s <= d <= e]


def zscore_rules(z, entry: float = 2.0, exit: float = 0.5, stop: float | None = 4.0, max_hold: int | None = None, size: float = 0.1, sign: float = -1.0):
    from nautilus_fi.kernel.portfolio import ZScoreRule

    return ZScoreRule(z, entry, exit, stop, max_hold, size, sign=sign)


def dv01_neutral(notional_a: float, bpv_a: float, bpv_b: float) -> float:
    """The notional of leg B (opposite sign) whose BPV offsets leg A's: ``-N_a x BPV_a / BPV_b``
    (BPV per unit notional)."""
    if not bpv_b:
        raise ValueError("leg B has zero BPV")
    return -float(notional_a) * float(bpv_a) / float(bpv_b)


def etf_shares(notional: float, price: float, lot: int = 1) -> int:
    """Whole shares (in lots) nearest ``notional / price``."""
    if price <= 0:
        raise ValueError("the ETF price must be positive")
    return int(np.round(notional / price / lot) * lot)


# MacKinnon (1994) response surfaces for the constant-only ADF regression, one series
_TAU_MAX, _TAU_MIN, _TAU_STAR = 2.74, -18.83, -1.61
_SMALL = (2.1659, 1.4412, 0.038269)
_LARGE = (1.7339, 0.93202, -0.12745, -0.010368)
CRITICAL = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def _mackinnon_p(tau: float) -> float:
    from scipy.stats import norm

    if tau > _TAU_MAX:
        return 1.0
    if tau < _TAU_MIN:
        return 0.0
    c = _SMALL if tau <= _TAU_STAR else _LARGE
    return float(norm.cdf(sum(ci * tau ** i for i, ci in enumerate(c))))


def adf(x: pd.Series | np.ndarray, lags: int | None = None) -> dict:
    """Augmented Dickey-Fuller test (constant, no trend): the t-statistic on the lagged level,
    its MacKinnon p-value and the critical values. ``lags`` of the difference (default:
    ``int(12 * (n / 100) ** 0.25)``, Schwert)."""
    y = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(y)
    if n < 20:
        raise ValueError("ADF needs at least 20 observations")
    p = int(12 * (n / 100) ** 0.25) if lags is None else int(lags)
    dy = np.diff(y)
    rows = len(dy) - p
    X = [np.ones(rows), y[p:-1]]
    for k in range(1, p + 1):
        X.append(dy[p - k:len(dy) - k])
    X = np.column_stack(X)
    Y = dy[p:]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    s2 = resid @ resid / (rows - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    tau = float(beta[1] / se)
    return {"stat": tau, "pvalue": _mackinnon_p(tau), "lags": p, "n": rows, "critical": dict(CRITICAL)}


def bh_fdr(pvalues: pd.Series | np.ndarray, q: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg at false-discovery rate ``q``: the adjusted p-values and which
    hypotheses are rejected."""
    p = pd.Series(pvalues, dtype=float)
    m = int(p.notna().sum())
    order = p.sort_values()
    ranks = np.arange(1, len(order) + 1)
    adj = (order * m / ranks)[::-1].cummin()[::-1].clip(upper=1.0)
    thresh = q * ranks / m
    passed = order.to_numpy() <= thresh
    k = int(np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    reject = pd.Series(False, index=order.index)
    reject.iloc[:k] = True
    return pd.DataFrame({"pvalue": p, "adjusted": adj.reindex(p.index), "reject": reject.reindex(p.index)})


PARITY = [ParityCase("credit pairs", "credit_pairs", "not named in the gap analysis: the framework's stored pair P&L", (Tolerance("returns", bp_per_year=5.0, min_corr=0.999),))]

__all__ = ["CRITICAL", "PARITY", "PROFILE", "REQUIREMENTS", "adf", "bh_fdr", "dv01_neutral", "etf_shares", "imm_dates", "pair_pnl", "zscore_rules"]
