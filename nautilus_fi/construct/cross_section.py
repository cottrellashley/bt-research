"""The cross-section a construction method sees on a rebalance date (identical in every engine)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossSection:
    date: pd.Timestamp
    ids: pd.Index                                  # candidate universe: eligible names, holdings and benchmark members
    data: pd.DataFrame                             # per-id fields and attributes (oasd, dts, krd_*, oas_bp, issuer, sector, rating_score ...)
    current: pd.Series                             # current weights (fractions of NAV)
    benchmark: pd.Series                           # benchmark weights (0 for non-members)
    scores: pd.Series | None = None                # alpha / signal: higher is better
    eligible: pd.Series | None = None              # True: may be bought; False: may only be held or sold
    cost_bp: pd.Series | None = None               # one-way cost of trading one unit of value, bp
    min_weight: pd.Series | None = None            # minimum holding weight (minimum piece converted at NAV)
    scenarios: pd.DataFrame | None = None          # scenario x id returns for CVaR
    nav: float = 1.0

    def __post_init__(self) -> None:
        self.ids = pd.Index(self.ids)
        self.data = self.data.reindex(self.ids)
        self.current = pd.Series(self.current, dtype=float).reindex(self.ids).fillna(0.0)
        self.benchmark = pd.Series(self.benchmark, dtype=float).reindex(self.ids).fillna(0.0)
        if self.scores is not None:
            self.scores = pd.Series(self.scores, dtype=float).reindex(self.ids)
        self.eligible = pd.Series(True, index=self.ids) if self.eligible is None else pd.Series(self.eligible).reindex(self.ids).fillna(False).astype(bool)
        if self.cost_bp is not None:
            self.cost_bp = pd.Series(self.cost_bp, dtype=float).reindex(self.ids)
        if self.min_weight is not None:
            self.min_weight = pd.Series(self.min_weight, dtype=float).reindex(self.ids).fillna(0.0)

    @staticmethod
    def build(date, universe, data: pd.DataFrame, current: pd.Series | None = None, benchmark: pd.Series | None = None, scores: pd.Series | None = None, cost_bp: pd.Series | None = None, min_face: pd.Series | float | None = None, price_dirty: pd.Series | None = None, nav: float = 1.0, scenarios: pd.DataFrame | None = None) -> CrossSection:
        """Candidate ids = eligible universe + current holdings + benchmark members; minimum holding
        weights from a minimum face (``min_face`` x dirty price / 100 / NAV) when given."""
        uni = pd.Index(universe).astype(str)
        cur = pd.Series(current if current is not None else {}, dtype=float)
        cur = cur[cur.abs() > 1e-12]
        bench = pd.Series(benchmark if benchmark is not None else {}, dtype=float)
        bench = bench[bench > 0]
        ids = uni.union(cur.index.astype(str)).union(bench.index.astype(str))
        eligible = pd.Series(ids.isin(uni), index=ids)
        mw = None
        if min_face is not None and price_dirty is not None and nav:
            mf = pd.Series(min_face, index=ids) if np.isscalar(min_face) else pd.Series(min_face).reindex(ids)
            mw = (mf * pd.Series(price_dirty).reindex(ids) / 100.0 / nav).fillna(0.0)
        return CrossSection(pd.Timestamp(date), ids, data, cur, bench, scores, eligible, cost_bp, mw, scenarios, nav)


__all__ = ["CrossSection"]
