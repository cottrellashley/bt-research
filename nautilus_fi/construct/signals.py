"""Cross-sectional signal transforms: winsorise, standardise within groups, rank, neutralise
against groups and numeric exposures, blend."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorise(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    x = pd.Series(x, dtype=float)
    if x.notna().sum() < 3:
        return x
    lo, hi = x.quantile(lower), x.quantile(upper)
    return x.clip(lo, hi)


def zscore(x: pd.Series, by: pd.Series | None = None, min_group: int = 3) -> pd.Series:
    """Standardise; within ``by`` groups when given (groups smaller than ``min_group`` use the
    overall moments)."""
    x = pd.Series(x, dtype=float)
    overall = (x - x.mean()) / (x.std(ddof=0) or 1.0)
    if by is None:
        return overall
    g = pd.Series(by).reindex(x.index)
    mean = x.groupby(g).transform("mean")
    sd = x.groupby(g).transform(lambda v: v.std(ddof=0))
    size = x.groupby(g).transform("count")
    z = (x - mean) / sd.replace(0.0, np.nan)
    return z.where(size >= min_group, overall).fillna(overall)


def rank_pct(x: pd.Series, by: pd.Series | None = None) -> pd.Series:
    x = pd.Series(x, dtype=float)
    if by is None:
        return x.rank(pct=True)
    return x.groupby(pd.Series(by).reindex(x.index)).rank(pct=True)


def neutralise(x: pd.Series, groups: pd.DataFrame | pd.Series | None = None, exposures: pd.DataFrame | None = None, weights: pd.Series | None = None) -> pd.Series:
    """Residual of ``x`` after regressing on group dummies and numeric exposures (per date
    cross-section): the part of a signal not explained by sector, rating, duration or DTS."""
    x = pd.Series(x, dtype=float)
    ok = x.notna()
    parts = [pd.Series(1.0, index=x.index, name="const")]
    if groups is not None:
        gdf = groups.to_frame() if isinstance(groups, pd.Series) else groups
        for col in gdf.columns:
            d = pd.get_dummies(gdf[col].astype(str), prefix=str(col), drop_first=True, dtype=float)
            parts.append(d)
    if exposures is not None:
        parts.append(exposures.astype(float))
    X = pd.concat(parts, axis=1).reindex(x.index)
    ok &= X.notna().all(axis=1)
    if ok.sum() <= X.shape[1]:
        return x - x.mean()
    Xm, ym = X[ok].to_numpy(), x[ok].to_numpy()
    if weights is not None:
        sw = np.sqrt(pd.Series(weights).reindex(x.index)[ok].fillna(0.0).to_numpy())
        Xm, ym = Xm * sw[:, None], ym * sw
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    resid = pd.Series(np.nan, index=x.index)
    resid[ok] = x[ok].to_numpy() - X[ok].to_numpy() @ beta
    return resid


def blend(signals: dict[str, pd.Series], weights: dict[str, float] | None = None, standardise: bool = True) -> pd.Series:
    weights = weights or {k: 1.0 for k in signals}
    total = None
    for k, s in signals.items():
        v = zscore(s) if standardise else pd.Series(s, dtype=float)
        v = v.fillna(0.0) * weights.get(k, 0.0)
        total = v if total is None else total.add(v, fill_value=0.0)
    return total if total is not None else pd.Series(dtype=float)


__all__ = ["blend", "neutralise", "rank_pct", "winsorise", "zscore"]
