"""Cross-sectional signal evaluation (Evaluate): does a signal rank bonds by their forward returns?

On each evaluation date the signal (a field, or a decider's scores) is compared with forward
returns that start after the conventions' trade offset, so the evaluation never uses a return the
decision could not have earned:

* rank IC (Spearman) per date and horizon, IC decay across horizons, hit rate, IC IR;
* quantile portfolio returns (equal or benchmark weighted), top-minus-bottom spread, monotonicity;
* Newey-West t-statistics (lags cover overlapping horizons) and circular block-bootstrap intervals;
* the same after neutralising the signal against groups (sector, rating) and exposures (OASD, DTS);
* top-quantile turnover between evaluation dates.

Every evaluation can be recorded in the trial ledger against a pre-registered hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.construct.signals import neutralise
from nautilus_fi.data.panel import Panel
from nautilus_fi.domain.calendars import parse_calendar
from nautilus_fi.domain.conventions import Conventions
from nautilus_fi.research.statistics import block_bootstrap_ci, newey_west_tstat


@dataclass
class EvaluationSettings:
    horizons: tuple[int, ...] = (1, 5, 21, 63)
    n_quantiles: int = 5
    calendar: str = "BME"
    return_field: str = "xr"
    universe_field: str | None = "member"
    weighting: str = "equal"                       # equal | benchmark (uses index_weight)
    weight_field: str = "index_weight"
    neutralise_groups: tuple[str, ...] = ()
    neutralise_exposures: tuple[str, ...] = ()
    min_names: int = 20
    n_boot: int = 1000
    seed: int = 0


@dataclass
class EvaluationReport:
    signal: str
    conventions: Conventions
    settings: EvaluationSettings
    ic: pd.DataFrame                                # date x horizon
    ic_neutral: pd.DataFrame | None
    quantiles: pd.DataFrame                         # (date, horizon) x quantile mean forward return
    spread: pd.DataFrame                            # date x horizon: top minus bottom
    coverage: pd.Series
    top_turnover: pd.Series
    extra: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        per_year = {"BME": 12, "BMS": 12, "M": 12, "W": 52, "D": 252, "BQE": 4, "Q": 4}.get(self.settings.calendar, 12)
        for h in self.ic.columns:
            ic = self.ic[h].dropna()
            sp = self.spread[h].dropna()
            # evaluation dates are ~21 trading days apart for monthly calendars: overlap lags
            step = 252 / per_year
            lags = max(int(np.ceil(h / step)) - 1, 0)
            qm = self.quantiles.xs(h, level="horizon").mean() if len(self.quantiles) else pd.Series(dtype=float)
            mono = float(pd.Series(qm.to_numpy()).corr(pd.Series(np.arange(len(qm))), method="spearman")) if len(qm) > 2 else float("nan")
            ci = block_bootstrap_ci(sp.to_numpy(), stat=np.mean, block_len=max(lags + 1, 3), n_boot=self.settings.n_boot, seed=self.settings.seed) if len(sp) > 10 else (float("nan"),) * 3
            row = {"horizon": h, "n_dates": len(ic), "mean_ic": float(ic.mean()), "ic_sd": float(ic.std()), "ic_ir": float(ic.mean() / ic.std()) if ic.std() else float("nan"), "ic_tstat_nw": newey_west_tstat(ic, lags), "hit_rate": float((ic > 0).mean()),
                   "spread_mean": float(sp.mean()), "spread_tstat_nw": newey_west_tstat(sp, lags), "spread_ci_low": ci[1], "spread_ci_high": ci[2], "spread_annualised": float(sp.mean() * per_year), "monotonicity": mono, "nw_lags": lags}
            if self.ic_neutral is not None:
                icn = self.ic_neutral[h].dropna()
                row.update({"mean_ic_neutral": float(icn.mean()), "ic_tstat_nw_neutral": newey_west_tstat(icn, lags)})
            rows.append(row)
        return pd.DataFrame(rows).set_index("horizon")

    def decay(self) -> pd.Series:
        return self.ic.mean().rename("mean_ic")

    def quantile_means(self) -> pd.DataFrame:
        return self.quantiles.groupby(level="horizon").mean()


def _forward_growth(r: np.ndarray) -> np.ndarray:
    """Cumulative log growth with missing returns as zero (a matured bond's money sits in cash)."""
    lg = np.log1p(np.nan_to_num(r, nan=0.0))
    return np.vstack([np.zeros(r.shape[1]), np.cumsum(lg, axis=0)])


def evaluate_signal(panel: Panel, signal: str | pd.DataFrame, conventions: Conventions | None = None, settings: EvaluationSettings | None = None, attributes: pd.DataFrame | None = None, ledger=None, hypothesis_id: str | None = None, name: str | None = None) -> EvaluationReport:
    conventions = conventions or Conventions()
    st = settings or EvaluationSettings()
    dates, ids = panel.dates, panel.ids
    T = len(dates)
    sig = panel.frame(signal).to_numpy() if isinstance(signal, str) else signal.reindex(index=dates, columns=ids).to_numpy()
    sig_lag = panel.lags.get(signal, 0) if isinstance(signal, str) else 0
    R = panel[st.return_field].astype(np.float64)
    G = _forward_growth(R)                        # G[k] = growth through row k-1
    alive = panel["alive"] > 0 if "alive" in panel else np.isfinite(R)
    off = conventions.timing.trade_offset_days
    eval_idx = np.nonzero(np.isin(dates.date, parse_calendar(st.calendar).dates(list(dates.date), first_decision=False)))[0]
    uni = panel[st.universe_field] if st.universe_field and st.universe_field in panel else None
    W = panel[st.weight_field] if st.weighting == "benchmark" and st.weight_field in panel else None
    attrs = attributes.reindex(ids) if attributes is not None else None
    ic, icn, sprd, cover, topt = {}, {}, {}, {}, {}
    qrows = []
    prev_top: set | None = None
    for i in eval_idx:
        j = i - sig_lag
        if j < 0:
            continue
        s = sig[j]
        mask = np.isfinite(s) & alive[i]
        if uni is not None:
            mask &= np.nan_to_num(uni[i]) > 0
        if mask.sum() < st.min_names:
            continue
        d = dates[i]
        cover[d] = int(mask.sum())
        s_m = pd.Series(s[mask], index=ids[mask])
        s_n = None
        if st.neutralise_groups or st.neutralise_exposures:
            grp = attrs.loc[s_m.index, list(st.neutralise_groups)] if (attrs is not None and st.neutralise_groups) else None
            ex = pd.DataFrame({f: panel.asof_row(f, i)[mask] for f in st.neutralise_exposures}, index=s_m.index) if st.neutralise_exposures else None
            s_n = neutralise(s_m, grp, ex)
        ranks = s_m.rank(pct=True, method="first")
        q = np.ceil(ranks * st.n_quantiles).clip(1, st.n_quantiles).astype(int)
        top = set(q.index[q == st.n_quantiles])
        if prev_top is not None and top:
            topt[d] = 1.0 - len(top & prev_top) / len(top)
        prev_top = top
        ic[d], icn[d], sprd[d] = {}, {}, {}
        for h in st.horizons:
            a, b = i + off, i + off + h                      # forward window: rows a+1 .. b
            if b >= T:
                continue
            fwd = pd.Series(np.expm1(G[b + 1][mask] - G[a + 1][mask]), index=s_m.index)
            ic[d][h] = float(s_m.corr(fwd, method="spearman"))
            if s_n is not None:
                icn[d][h] = float(s_n.corr(fwd, method="spearman"))
            if W is not None:
                wv = pd.Series(W[i][mask], index=s_m.index)
                qm = (fwd * wv).groupby(q).sum() / wv.groupby(q).sum()
            else:
                qm = fwd.groupby(q).mean()
            qrows.append({"date": d, "horizon": h, **{f"Q{k}": float(v) for k, v in qm.items()}})
            sprd[d][h] = float(qm.get(st.n_quantiles, np.nan) - qm.get(1, np.nan))
    ic_df = pd.DataFrame.from_dict(ic, orient="index").sort_index()
    sp_df = pd.DataFrame.from_dict(sprd, orient="index").sort_index()
    qdf = pd.DataFrame(qrows).set_index(["date", "horizon"]).sort_index() if qrows else pd.DataFrame()
    rep = EvaluationReport(name or (signal if isinstance(signal, str) else "signal"), conventions, st, ic_df, pd.DataFrame.from_dict(icn, orient="index").sort_index() if (st.neutralise_groups or st.neutralise_exposures) else None, qdf, sp_df, pd.Series(cover, name="n_names"), pd.Series(topt, name="top_turnover"))
    if ledger is not None:
        ledger.record_evaluation(rep, hypothesis_id)
    return rep


__all__ = ["EvaluationReport", "EvaluationSettings", "evaluate_signal"]
