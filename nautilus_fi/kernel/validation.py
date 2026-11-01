"""Research and validation on the kernel (section 4.11 of the gap analysis).

* ``walk_forward``: purged, embargoed walk-forward over a parameter grid; each candidate is built
  for its split with the training window it may fit on (``builder(params, train=(start, end))``),
  selected on the validation window and run once on the test window; validation and test runs are
  ledger trials (VALIDATION / OOS).
* ``event_splits``: folds that keep every date of an event together, with an embargo after each
  test fold.
* ``select_gated``: shortlist on the primary score within a tolerance of the best, then the best
  P&L above a floor, else ``SelectionFailure``.
* ``placebos``: the strategy with a shuffled, a noise and an anti signal (and repeats), against the
  real one: the share of placebos at least as good is the empirical p-value.
* ``preregistered``: statistics on pre-declared sub-periods, contamination (trials that saw the
  evaluation window before registration) and prospective shadow verdicts (interim / final, share of
  positive blocks).
* ``label_periods``: IS / VAL / OOS labels for a result's dates and statistics per label.
* ``eligible`` / ``leaderboard``: sweep gates (failed or relaxed rebalances, unconstrained
  fallbacks, data gaps, turnover) and a ranking of the results that pass.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nautilus_fi.kernel.expr import Expr, lift


class SelectionFailure(RuntimeError):
    """No candidate passes the selection rule."""


# ====================================================================================== placebo signals
class Shuffled(Expr):
    """The signal's values permuted across instruments on each date (a placebo that keeps the
    cross-sectional distribution and destroys the information)."""

    def __init__(self, a, seed: int = 0) -> None:
        self.a, self.seed = lift(a), int(seed)
        self._children = (self.a,)

    def _params(self):
        return [self.seed]

    def __repr__(self) -> str:
        return f"Shuffled({self.a!r}, seed={self.seed})"

    def _eval(self, ctx):
        v = np.array(self.a.evaluate(ctx), dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        for t in range(v.shape[0]):
            ok = np.nonzero(np.isfinite(v[t]))[0]
            v[t, ok] = v[t, rng.permutation(ok)]
        return v


class Noise(Expr):
    """A standard normal signal per instrument and date (seeded)."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def _params(self):
        return [self.seed]

    def _own_names(self):
        return set()

    def __repr__(self) -> str:
        return f"Noise(seed={self.seed})"

    def _eval(self, ctx):
        return np.random.default_rng(self.seed).normal(size=(ctx.T, ctx.N))


def placebos(session, strategy, n: int = 5, metric: str = "sharpe", kinds=("shuffle", "noise", "anti")) -> dict:
    """Placebo runs (not counted as trials): ``n`` shuffled and ``n`` noise signals and the
    anti-signal, against the real strategy. Returns the table and the empirical p-value (share of
    placebos with ``metric`` at least the real one's)."""
    real = session.run(strategy, record=False).stats().get(metric)
    rows = []
    for k in range(n):
        if "shuffle" in kinds:
            rows.append(("shuffle", k, session.run(strategy.with_(signal=Shuffled(strategy.signal, k), name=f"{strategy.name}[shuffle {k}]"), record=False).stats().get(metric)))
        if "noise" in kinds:
            rows.append(("noise", k, session.run(strategy.with_(signal=Noise(k), name=f"{strategy.name}[noise {k}]"), record=False).stats().get(metric)))
    if "anti" in kinds:
        rows.append(("anti", 0, session.run(strategy.with_(signal=-strategy.signal, name=f"{strategy.name}[anti]"), record=False).stats().get(metric)))
    table = pd.DataFrame(rows, columns=["kind", "seed", metric])
    p = float((table[metric] >= real).mean()) if len(table) else float("nan")
    return {"real": real, "table": table, "p_value": p, "metric": metric}


# ====================================================================================== walk-forward
def walk_forward(session, builder, grid: list[dict], splits, label_horizon: int = 21, embargo: int = 5, metric: str = "sharpe", select=None) -> dict:
    """Purged, embargoed walk-forward on the kernel. ``builder(params, train=(start, end))`` returns
    the strategy for a candidate, fitted (if it fits anything) only on data in ``train``, which ends
    ``execution lag + label_horizon`` trading days before the validation window (purge). Candidates
    run on the validation window (VALIDATION trials); the best by ``metric`` (or ``select(results)``)
    runs on the test window, which starts ``embargo`` days after validation (OOS trial). Returns
    the stitched out-of-sample returns and a table per split."""
    rows, oos = [], []
    for k, sp in enumerate(splits):
        dev_lo, dev_hi = pd.Timestamp(sp.dev_start), pd.Timestamp(sp.dev_end)
        val_lo, val_hi = dev_hi, pd.Timestamp(sp.val_end) - pd.Timedelta(days=1)
        test_lo, test_hi = pd.Timestamp(sp.test_start) + pd.tseries.offsets.BDay(embargo), pd.Timestamp(sp.test_end) - pd.Timedelta(days=1)
        results = []
        for params in grid:
            probe = builder(params, train=(dev_lo, dev_hi))
            lag = int(getattr(probe.execution, "lag", 1))
            train = (dev_lo, val_lo - pd.tseries.offsets.BDay(lag + label_horizon))
            st = builder(params, train=train)
            r = session.run(st.with_(name=f"{st.name}[val {k}]"), start=val_lo, end=val_hi, stage="VALIDATION")
            results.append((params, st, r))
        if select is not None:
            params, st, r = select([x for x in results])
        else:
            params, st, r = max(results, key=lambda x: _score(x[2], metric))
        test_train = (dev_lo, test_lo - pd.tseries.offsets.BDay(int(getattr(st.execution, "lag", 1)) + label_horizon))
        final = builder(params, train=test_train)
        t = session.run(final.with_(name=f"{final.name}[test {k}]"), start=test_lo, end=test_hi, stage="OOS")
        oos.append(t.returns.iloc[1:])
        rows.append({"split": k, "selected": params, "validation_score": _score(r, metric), "test_score": _score(t, metric), "train_end": str(train[1].date()), "validation": f"{val_lo.date()}..{val_hi.date()}", "test": f"{test_lo.date()}..{test_hi.date()}"})
    stitched = pd.concat(oos).sort_index() if oos else pd.Series(dtype=float)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return {"oos_returns": stitched.rename("oos"), "folds": pd.DataFrame(rows)}


def _score(result, metric: str) -> float:
    v = result.stats().get(metric)
    return float(v) if v is not None and np.isfinite(v) else -np.inf


def event_splits(events: pd.Series, n_splits: int = 5, embargo: int = 0) -> list[dict]:
    """Folds that keep each event's dates together (``events``: an event id per date, NaN for
    none): returns ``[{"test": dates, "train": dates}]`` with the ``embargo`` periods after each
    test fold removed from its training set, and all dates of a test event out of training."""
    ev = events.dropna()
    ids = pd.unique(ev)
    folds = np.array_split(np.asarray(ids, dtype=object), n_splits)
    dates = events.index
    out = []
    for f in folds:
        test_mask = events.isin(list(f)).to_numpy()
        test_dates = dates[test_mask]
        if not len(test_dates):
            continue
        last = dates.get_loc(test_dates[-1])
        emb = dates[last + 1: last + 1 + embargo]
        train = dates[~test_mask & ~dates.isin(emb)]
        out.append({"test": test_dates, "train": train, "events": list(f)})
    return out


def select_gated(results, primary: str = "information_ratio", tolerance: float = 0.1, pnl: str = "annual_return", floor: float = 0.0):
    """P&L-gated selection: the candidates whose ``primary`` score is within ``tolerance``
    (relative) of the best, then the one with the best ``pnl`` above ``floor``; raises
    ``SelectionFailure`` when none clears the floor. ``results``: results, or
    ``(params, strategy, result)`` tuples (as ``walk_forward`` passes)."""
    items = [x if isinstance(x, tuple) else (None, None, x) for x in results]
    scores = [_score(x[2], primary) for x in items]
    best = max(scores)
    short = [x for x, s in zip(items, scores, strict=True) if s >= best - tolerance * abs(best)]
    ok = [x for x in short if _score(x[2], pnl) > floor]
    if not ok:
        raise SelectionFailure(f"no candidate within {tolerance:.0%} of the best {primary} has {pnl} above {floor:g}")
    pick = max(ok, key=lambda x: _score(x[2], pnl))
    return pick if isinstance(results[0], tuple) else pick[2]


# ====================================================================================== pre-registration and labels
def label_periods(result, is_end, val_end=None) -> dict:
    """IS / VAL / OOS labels per date (in sample up to ``is_end``, validation to ``val_end``, out
    of sample after) and statistics per label."""
    idx = result.returns.index
    lab = pd.Series("OOS", index=idx)
    lab[idx <= pd.Timestamp(is_end)] = "IS"
    if val_end is not None:
        lab[(idx > pd.Timestamp(is_end)) & (idx <= pd.Timestamp(val_end))] = "VAL"
    rows = {}
    for k, seg in result.returns.iloc[1:].groupby(lab.iloc[1:]):
        sd = seg.std(ddof=1)
        rows[k] = {"periods": len(seg), "total_return": float((1 + seg).prod() - 1), "sharpe": float(seg.mean() / sd * math.sqrt(result.stats().get("periods_per_year") or 252)) if sd else float("nan")}
    return {"labels": lab, "stats": pd.DataFrame(rows).T}


def preregistered(result, ledger=None, hypothesis=None, subperiods: list[tuple] | None = None, shadow_start=None, block_months: int = 3, min_positive_share: float = 0.6, interim_blocks: int = 2) -> dict:
    """The pre-registered evaluation: statistics on the declared ``subperiods``; contamination
    (trials in the ``ledger`` created before the hypothesis was registered whose windows overlap
    its validation or holdout periods); and the prospective shadow verdict on returns from
    ``shadow_start``: ``interim`` after ``interim_blocks`` blocks of ``block_months``, ``final``
    when all blocks are in, PASS when at least ``min_positive_share`` of blocks are positive."""
    out: dict = {}
    r = result.returns.iloc[1:]
    if subperiods:
        rows = []
        for lo, hi in subperiods:
            seg = r[(r.index >= pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))]
            rows.append({"start": lo, "end": hi, "total_return": float((1 + seg).prod() - 1) if len(seg) else float("nan"), "positive": bool(len(seg) and (1 + seg).prod() > 1)})
        out["subperiods"] = pd.DataFrame(rows)
    if ledger is not None and hypothesis is not None:
        h = ledger.hypothesis(hypothesis) if isinstance(hypothesis, str) else hypothesis
        windows = [h.validation_period, h.holdout_period]
        early = [t for t in ledger.trials_list() if t.get("created_utc", "") < h.registered_utc and any(str(t["period"][1]) >= str(w[0]) and str(t["period"][0]) <= str(w[1]) for w in windows)]
        out["contaminated"] = bool(early)
        out["contaminating_trials"] = len(early)
    if shadow_start is not None:
        shadow = r[r.index >= pd.Timestamp(shadow_start)]
        blocks = (1 + shadow).groupby(pd.Grouper(freq=f"{block_months}ME")).prod() - 1
        blocks = blocks[blocks.index <= shadow.index[-1] + pd.offsets.MonthEnd(0)] if len(shadow) else blocks
        share = float((blocks > 0).mean()) if len(blocks) else float("nan")
        stage = "final" if len(blocks) >= 2 * interim_blocks else ("interim" if len(blocks) >= interim_blocks else "pending")
        out["shadow"] = {"blocks": blocks, "positive_share": share, "stage": stage, "verdict": "PENDING" if stage == "pending" else ("PASS" if share >= min_positive_share else "FAIL")}
    return out


# ====================================================================================== sweep gates
def eligible(result, max_failed: int = 0, allow_relaxed: bool = False, max_gap_days: int | None = None, max_turnover: float | None = None) -> tuple[bool, list[str]]:
    """Whether a sweep variant may be ranked: no failed (infeasible, aborted, pushed) rebalances
    beyond ``max_failed``, no relaxed (unconstrained) fallbacks unless allowed, no gap between
    rebalances longer than ``max_gap_days`` calendar days, turnover at most ``max_turnover``."""
    d = result.diagnostics
    reasons = []
    failed = int(d.get("infeasible_rebalances", 0)) + int(d.get("aborted_rebalances", 0)) + int(d.get("pushed_rebalances", 0))
    if failed > max_failed:
        reasons.append(f"{failed} failed rebalances")
    if not allow_relaxed and int(d.get("relaxed_rebalances", 0)):
        reasons.append(f"{d['relaxed_rebalances']} relaxed rebalances")
    if max_gap_days is not None and len(result.trades) > 1:
        gaps = pd.to_datetime(result.trades["trade_date"]).diff().dt.days.max()
        if gaps > max_gap_days:
            reasons.append(f"a {int(gaps)}-day gap between rebalances")
    if max_turnover is not None and (result.stats().get("turnover") or 0.0) > max_turnover:
        reasons.append(f"turnover {result.stats()['turnover']:.2f} above {max_turnover:g}")
    return (not reasons), reasons


def leaderboard(results, metric: str = "information_ratio", **gates) -> pd.DataFrame:
    """Results ranked by ``metric``, with the ones that fail the eligibility gates listed last and
    their reasons."""
    rows = []
    for r in results:
        ok, why = eligible(r, **gates)
        rows.append({"name": r.name, metric: r.stats().get(metric), "eligible": ok, "reasons": "; ".join(why)})
    df = pd.DataFrame(rows)
    return df.sort_values(["eligible", metric], ascending=[False, False]).reset_index(drop=True)


__all__ = ["Noise", "SelectionFailure", "Shuffled", "eligible", "event_splits", "label_periods", "leaderboard", "placebos", "preregistered", "select_gated", "walk_forward"]
