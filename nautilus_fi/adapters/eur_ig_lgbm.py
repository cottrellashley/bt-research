"""EUR IG LightGBM: index timing from a calibrated classifier (gap analysis §5.5).

Conventions: the ``EUR_IG`` profile (trades T+1, first earns T+2). The components:

* ``symmetric_tilt``: exposure ``1 +/- k x s``, clipped and rounded to a step;
* ``oof_calibrate``: out-of-fold (Platt) calibration of the classifier's scores, centred so the
  training mean is 0.5, and ``centred_score`` mapping a probability to -1..+1;
* ``seed_ensemble``: the average of a model fitted with several seeds (10 by default);
* ``fit_blend``: blend weights fitted on training data only (non-negative, summing to one);
* ``timing_metrics``: flips per year, time overweight / neutral / underweight, and the regime
  and whipsaw breakdowns.

The model itself (LightGBM) is the framework's; these are the parts around it the backtest needs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance

PROFILE = "EUR_IG"
REQUIREMENTS = ("symmetric tilt", "OOF calibration", "seed ensemble", "training-only blend", "timing metrics")


def symmetric_tilt(score: pd.Series | np.ndarray | float, k: float = 0.5, clip: float = 0.5, step: float = 0.1):
    """Exposure ``1 + k x s`` for a score in -1..+1, clipped to ``1 +/- clip`` and rounded to the
    nearest ``step`` (symmetric: the same size over- and underweight)."""
    s = np.asarray(score, dtype=float)
    x = np.clip(1.0 + k * s, 1.0 - clip, 1.0 + clip)
    if step:
        x = np.round(np.round(x / step) * step, 10)
    return pd.Series(x, index=score.index) if isinstance(score, pd.Series) else (float(x) if np.ndim(x) == 0 else x)


def _platt(z: np.ndarray, y: np.ndarray, iters: int = 100) -> tuple[float, float]:
    a, b = 1.0, 0.0
    for _ in range(iters):                                               # Newton on the logistic log-likelihood
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        w = p * (1 - p) + 1e-12
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def oof_calibrate(scores: pd.Series, labels: pd.Series, folds: int = 5, centre: bool = True) -> pd.Series:
    """Out-of-fold probabilities: each contiguous fold is calibrated (Platt: logistic on the raw
    score) on the other folds only; ``centre`` shifts each fold's output so that the fitted
    folds' mean prediction is 0.5."""
    df = pd.concat([scores.rename("s"), labels.rename("y")], axis=1).dropna()
    z, y = df["s"].to_numpy(float), df["y"].to_numpy(float)
    out = np.full(len(df), np.nan)
    edges = np.linspace(0, len(df), folds + 1).astype(int)
    for i in range(folds):
        test = np.zeros(len(df), dtype=bool)
        test[edges[i]:edges[i + 1]] = True
        a, b = _platt(z[~test], y[~test])
        p = 1.0 / (1.0 + np.exp(-(a * z[test] + b)))
        if centre:
            fit = 1.0 / (1.0 + np.exp(-(a * z[~test] + b)))
            p = np.clip(p - fit.mean() + 0.5, 0.0, 1.0)
        out[test] = p
    return pd.Series(out, index=df.index, name="p")


def centred_score(p: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """A probability as a score in -1..+1 (0.5 -> 0)."""
    return 2.0 * (p - 0.5)


def seed_ensemble(fit_predict: Callable[[int], np.ndarray | pd.Series], seeds=range(10)):
    """The mean of ``fit_predict(seed)`` over ``seeds``."""
    preds = [fit_predict(int(s)) for s in seeds]
    if isinstance(preds[0], pd.Series):
        return pd.concat(preds, axis=1).mean(axis=1)
    return np.mean(np.vstack([np.asarray(p, dtype=float) for p in preds]), axis=0)


def fit_blend(predictions: pd.DataFrame, target: pd.Series, train_end) -> pd.Series:
    """Blend weights (non-negative, summing to one) from the rows up to ``train_end`` only;
    rows after it are never read, so the blend cannot see the test period."""
    from scipy.optimize import nnls

    end = pd.Timestamp(train_end)
    X = predictions.loc[predictions.index <= end]
    y = target.reindex(X.index)
    ok = X.notna().all(axis=1) & y.notna()
    if not ok.any():
        raise ValueError(f"no training rows on or before {end.date()}")
    w, _ = nnls(X[ok].to_numpy(float), y[ok].to_numpy(float))
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    return pd.Series(w, index=predictions.columns, name="weight")


def timing_metrics(result, neutral: float | None = 1.0, band: float = 0.05, regimes: pd.Series | None = None, whipsaw_days: int = 10) -> dict:
    """Flips per year, time overweight / neutral / underweight (outside ``band`` of ``neutral``),
    and the breakdowns by regime, drawdown episode and whipsaw."""
    from nautilus_fi.kernel.analytics import breakdowns, timing_stats

    return {**timing_stats(result, neutral=neutral, band=band), **breakdowns(result, regimes=regimes, whipsaw_days=whipsaw_days)}


PARITY = [ParityCase("EUR IG timing", "eur_ig_lgbm", "not named in the gap analysis: the framework's stored backtest", (Tolerance("returns", bp_per_year=5.0, min_corr=0.999),))]

__all__ = ["PARITY", "PROFILE", "REQUIREMENTS", "centred_score", "fit_blend", "oof_calibrate", "seed_ensemble", "symmetric_tilt", "timing_metrics"]
