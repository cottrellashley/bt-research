"""Walk-forward research with purging and embargo, every trial in the ledger.

For each split (``research.walk_forward_splits``): each candidate parameter set is fitted on the
development window (observations whose labels reach into the validation window purged, an embargo
after it) and scored on the validation window; the best candidate is refitted on development plus
validation (purged against the test window) and run out of sample on the test window. Out-of-sample
segments are stitched into one return series. Every candidate run is a ``VALIDATION`` trial and
every test run an ``OOS`` trial, so the deflated Sharpe ratio and PBO see the whole search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from nautilus_fi.domain.spec import ResearchSpec
from nautilus_fi.paper.engine import PaperData, PaperEngine, PaperResult, PaperSettings
from nautilus_fi.research.walkforward import Split, purged_train_indices


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    folds: pd.DataFrame
    test_results: list[PaperResult] = field(default_factory=list)

    def stats(self, conventions) -> dict:
        from nautilus_fi.domain.conventions import performance_stats

        return performance_stats(self.oos_returns, conventions)


def _score(res: PaperResult, metric: str) -> float:
    s = res.stats()
    v = s.get(metric, s.get("sharpe"))
    return float(v) if v is not None and np.isfinite(v) else -np.inf


def walk_forward(make_spec: Callable[[dict], ResearchSpec], candidates: list[dict], data: PaperData, splits: list[Split], label_horizon_days: int = 21, embargo_days: int = 5, settings: PaperSettings | None = None, ledger=None, hypothesis_id: str | None = None, metric: str = "sharpe") -> WalkForwardResult:
    """``make_spec(params)`` builds the spec for a candidate; its decider's ``fit(train)`` receives a
    dict with the panel and the purged training row positions."""
    base = settings or PaperSettings()
    dates = data.panel.dates
    rows, oos, results = [], [], []
    lag = (base.conventions or make_spec(candidates[0]).conventions).timing.trade_offset_days if candidates else 0
    for k, sp in enumerate(splits):
        dev = np.nonzero((dates >= pd.Timestamp(sp.dev_start)) & (dates < pd.Timestamp(sp.dev_end)))[0]
        val_lo, val_hi = pd.Timestamp(sp.dev_end), pd.Timestamp(sp.val_end) - pd.Timedelta(days=1)
        test_lo, test_hi = pd.Timestamp(sp.test_start), pd.Timestamp(sp.test_end) - pd.Timedelta(days=1)
        keep_val = purged_train_indices(dates[dev], val_lo, val_hi, label_horizon_days, embargo_days, unit="trading", trading_days=dates, execution_lag_days=lag)
        train_val = dev[keep_val]
        best, best_score, best_params = None, -np.inf, None
        for params in candidates:
            spec = make_spec(params)
            spec = replace(spec, decider=spec.decider.fit({"panel": data.panel.subset(end=val_lo - pd.Timedelta(days=1)), "rows": train_val, "params": params}), hypothesis_id=hypothesis_id or spec.hypothesis_id)     # fit sees nothing from the window it is scored on
            res = PaperEngine(spec, data, replace(base, start=val_lo, end=val_hi, name=f"{spec.name}[val{k}]")).run(ledger=ledger, stage="VALIDATION") if ledger is not None else PaperEngine(spec, data, replace(base, start=val_lo, end=val_hi)).run()
            sc = _score(res, metric)
            if sc > best_score:
                best, best_score, best_params = spec, sc, params
        if best is None:
            continue
        devval = np.nonzero((dates >= pd.Timestamp(sp.dev_start)) & (dates <= val_hi))[0]
        keep_test = purged_train_indices(dates[devval], test_lo, test_hi, label_horizon_days, embargo_days, unit="trading", trading_days=dates, execution_lag_days=lag)
        spec = make_spec(best_params)
        spec = replace(spec, decider=spec.decider.fit({"panel": data.panel.subset(end=test_lo - pd.Timedelta(days=1)), "rows": devval[keep_test], "params": best_params}), hypothesis_id=hypothesis_id or spec.hypothesis_id)
        tset = replace(base, start=test_lo, end=test_hi, name=f"{spec.name}[test{k}]")
        test = PaperEngine(spec, data, tset).run(ledger=ledger, stage="OOS") if ledger is not None else PaperEngine(spec, data, tset).run()
        results.append(test)
        oos.append(test.returns.iloc[1:])
        rows.append({"split": k, "dev_start": sp.dev_start, "dev_end": sp.dev_end, "val_end": sp.val_end, "test_end": sp.test_end, "selected": best_params, "validation_score": best_score, "test_score": _score(test, metric), "train_rows_val": len(train_val), "purged_val": len(dev) - len(train_val), "train_rows_test": len(keep_test), "purged_test": len(devval) - len(keep_test)})
    stitched = pd.concat(oos).sort_index() if oos else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return WalkForwardResult(stitched.rename("oos"), pd.DataFrame(rows), results)


__all__ = ["WalkForwardResult", "walk_forward"]
