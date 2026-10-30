"""Dependence-aware uncertainty and selection-bias diagnostics: block bootstrap, Newey-West Sharpe
standard errors, the deflated Sharpe ratio (Bailey & Lopez de Prado) and the probability of
backtest overfitting (CSCV). None is a certificate; each states its assumptions."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series | np.ndarray, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float) - rf / periods_per_year
    if r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def newey_west_sharpe_se(returns: pd.Series | np.ndarray, periods_per_year: int = 252, lags: int | None = None) -> float:
    """Standard error of the annualised Sharpe ratio with autocorrelation-robust variance of the mean
    (stale marks and overlapping holdings inflate naive confidence)."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 10:
        return float("nan")
    lags = lags if lags is not None else int(math.floor(4 * (n / 100) ** (2 / 9)))
    rc = r - r.mean()
    gamma0 = float(np.dot(rc, rc) / n)
    lr_var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        gk = float(np.dot(rc[k:], rc[:-k]) / n)
        lr_var += 2 * w * gk
    sd = r.std(ddof=1)
    se_mean = math.sqrt(max(lr_var, 1e-18) / n)
    return float(se_mean / sd * math.sqrt(periods_per_year))


def block_bootstrap_ci(returns: pd.Series | np.ndarray, stat=sharpe_ratio, block_len: int = 20, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Circular block bootstrap confidence interval for a statistic of a dependent return series."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    vals = []
    n_blocks = int(math.ceil(n / block_len))
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(block_len) + s) % n for s in starts])[:n]
        vals.append(stat(r[idx]))
    vals = np.array(vals)
    return float(stat(r)), float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam approximation (sufficient here; avoids a hard scipy dependency)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe_ratio(observed_sr: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0, var_trial_sr: float | None = None, periods_per_year: int = 252) -> dict:
    """Bailey & Lopez de Prado deflated Sharpe: probability that the observed (per-period) Sharpe
    exceeds the expected maximum Sharpe from ``n_trials`` independent trials. Inputs are per-period
    Sharpe ratios; ``observed_sr`` annualised is converted. States assumptions explicitly."""
    sr = observed_sr / math.sqrt(periods_per_year)
    euler = 0.5772156649
    v = var_trial_sr if var_trial_sr is not None else (sr ** 2) * 0.5 / max(n_trials, 1)
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(v) * ((1 - euler) * _norm_ppf(1 - 1.0 / n_trials) + euler * _norm_ppf(1 - 1.0 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * math.sqrt(max(n_obs - 1, 1)) / denom
    return {"dsr": _norm_cdf(z), "expected_max_sr_annual": sr0 * math.sqrt(periods_per_year), "observed_sr_annual": observed_sr, "n_trials": n_trials, "assumptions": "trials treated as independent; correlated variants overstate n_trials; non-normality via skew/kurt"}


def min_track_record_length(observed_sr: float, benchmark_sr: float = 0.0, skew: float = 0.0, kurt: float = 3.0, confidence: float = 0.95, periods_per_year: int = 252) -> float:
    sr, sr_b = observed_sr / math.sqrt(periods_per_year), benchmark_sr / math.sqrt(periods_per_year)
    z = _norm_ppf(confidence)
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) * (z / max(sr - sr_b, 1e-9)) ** 2


def probability_of_backtest_overfitting(perf: pd.DataFrame | np.ndarray, n_splits: int = 8, metric=sharpe_ratio) -> dict:
    """CSCV (Bailey, Borwein, Lopez de Prado, Zhu): ``perf`` is T x N per-period returns of N variants.
    Returns PBO = probability the in-sample best variant ranks below median out of sample."""
    M = np.asarray(perf, dtype=float)
    T, N = M.shape
    if n_splits % 2:
        n_splits += 1
    bounds = np.linspace(0, T, n_splits + 1, dtype=int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_splits)]
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        train_idx = np.concatenate([groups[i] for i in combo])
        test_idx = np.concatenate([groups[i] for i in range(n_splits) if i not in combo])
        is_perf = np.array([metric(M[train_idx, j]) for j in range(N)])
        oos_perf = np.array([metric(M[test_idx, j]) for j in range(N)])
        best = int(np.nanargmax(is_perf))
        rank = (np.sum(oos_perf < oos_perf[best]) + 0.5) / (N + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_combinations": len(logits), "mean_logit": float(np.mean(logits)), "assumptions": "variants evaluated on identical dates; a common stress episode is not independent evidence"}


__all__ = ["block_bootstrap_ci", "deflated_sharpe_ratio", "min_track_record_length", "newey_west_sharpe_se", "probability_of_backtest_overfitting", "sharpe_ratio"]
