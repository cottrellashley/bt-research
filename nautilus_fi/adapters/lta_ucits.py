"""lta-ucits (THFA / TABC): UCITS index-tracking funds (gap analysis §5.8; items marked [?] were
read from a photo).

Canonical conventions (no legacy profile). The components:

* ``two_stage``: maximise, then minimise a tie-break at no loss (TE or active share);
* ``min_trade``: a MILP minimum trade size;
* ``issuer_limit``: each issuer at most min(4%, 3 x its index weight) [?];
* ``banks_at_index``: banking issuers at or below their index weight;
* ``esg_screen``: the three-state ESG / MSCI screen (eligible, grandfathered, excluded);
* ``grandfathered``: names held but never bought;
* ``flow_construction``: the flow optimiser for creations (buy only) and redemptions (sell only);
* ``turnover_budget``: the index's own turnover plus an allowance;
* ``common_start``: start where every input the strategy reads has history;
* ``paired_t``: paired-difference t-statistics (Newey-West) between two return series;
* ``tabc_hedge``: legacy TABC's Bund-futures hedge under a TE cap of 0.85% [?].
"""

from __future__ import annotations

import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance
from nautilus_fi.kernel.expr import Field

PROFILE = None
REQUIREMENTS = ("two-stage tie-break", "MILP min trade", "issuer min(4%, 3x)", "banks <= index", "three-state ESG", "grandfathered", "flow optimiser", "index turnover + allowance", "common start", "paired t", "TABC Bund hedge")
ESG_STATES = ("ELIGIBLE", "GRANDFATHERED", "EXCLUDED")


def two_stage(objective, *constraints, tie_break=None, tie_tolerance: float = 1e-6, **kw):
    """Stage one maximises ``objective``; stage two minimises ``tie_break`` (e.g.
    ``-ActiveShare()``) without giving up more than ``tie_tolerance`` of it."""
    from nautilus_fi.kernel.portfolio import ActiveShare, Optimise

    return Optimise(objective, *constraints, tie_break=tie_break if tie_break is not None else -ActiveShare(), tie_tolerance=tie_tolerance, **kw)


def min_trade(min_weight: float):
    from nautilus_fi.kernel.portfolio import MinTrade

    return MinTrade(min_weight)


def issuer_limit(absolute: float = 0.04, relative: float = 3.0, group: str = "issuer", floor: float | None = None):
    """Each issuer at most min(``absolute``, ``relative`` x its index weight) [?]."""
    from nautilus_fi.kernel.portfolio import GroupWeight

    return GroupWeight(group).cap(absolute, relative, floor)


def banks_at_index(is_bank, group: str = "issuer"):
    """Banking issuers (``is_bank``: a predicate, e.g. ``Field("sector") == "BANKING"``) at or
    below their index weight."""
    from nautilus_fi.kernel.portfolio import ActiveWeight

    return ActiveWeight(group).where(is_bank) <= 0.0


def esg_screen(state: str = "esg_state"):
    """The three-state screen on attribute ``state``: EXCLUDED names leave the universe (held ones
    are sold); GRANDFATHERED names stay but are never bought. Returns (universe predicate,
    constraint)."""
    from nautilus_fi.kernel.portfolio import NoBuy

    return Field(state) != "EXCLUDED", NoBuy(Field(state) == "GRANDFATHERED")


def grandfathered(names):
    """Names (a list or a predicate) held but never bought."""
    from nautilus_fi.kernel.portfolio import NoBuy

    return NoBuy(names)


def flow_construction(base, flow: float):
    """The flow optimiser: on a creation (``flow > 0``) the optimiser only buys, on a redemption
    only sells, so the flow is invested or raised without churning the rest of the book."""
    from nautilus_fi.kernel.expr import All
    from nautilus_fi.kernel.portfolio import NoBuy, NoSell

    if flow > 0:
        return base.add(NoSell(All()))
    if flow < 0:
        return base.add(NoBuy(All()))
    return base


def turnover_budget(benchmark, allowance: float = 0.0, window: int = 21):
    """Turnover up to the index's own turnover over ``window`` days plus ``allowance``."""
    from nautilus_fi.kernel.portfolio import IndexTurnover

    return IndexTurnover(benchmark, window, allowance)


def common_start(strategy, source) -> pd.Timestamp | None:
    """The first date every field the strategy reads has history (the latest catalog start)."""
    from nautilus_fi.kernel.data import as_source

    src = as_source(source)
    cat = src.catalog()
    req = strategy.requirements(cat)
    starts = [pd.Timestamp(cat.get(n).start) for n in sorted(req.fields) if cat.get(n) is not None and cat.get(n).start]
    return max(starts) if starts else None


def paired_t(a: pd.Series, b: pd.Series, lags: int | None = None) -> dict:
    """Paired-difference statistics of two return series on common dates: mean difference per
    period, its Newey-West t-statistic and the share of periods ``a`` beats ``b``."""
    from nautilus_fi.research.statistics import newey_west_tstat

    x, y = a.align(b, join="inner")
    d = (x - y).dropna()
    return {"n": len(d), "mean_difference": float(d.mean()) if len(d) else float("nan"), "t_stat": float(newey_west_tstat(d, lags=lags)) if len(d) > 2 else float("nan"), "share_positive": float((d > 0).mean()) if len(d) else float("nan")}


def tabc_hedge(instrument: str = "RX", te_cap: float = 0.0085, model=None, field: str = "oad"):
    """Legacy TABC: duration hedged with Bund futures, active risk at most ``te_cap`` a year [?].
    Returns (overlay, constraint or None without a risk model)."""
    from nautilus_fi.kernel.portfolio import TEV, Hedge

    return Hedge((instrument,), (field,)), (TEV(model) <= te_cap) if model is not None else None


PARITY = [ParityCase("THFA", "lta_ucits", "not named in the gap analysis: the fund's stored backtest", (Tolerance("returns", bp_per_year=10.0, min_corr=0.999), Tolerance("turnover", abs=0.02)))]

__all__ = ["ESG_STATES", "PARITY", "PROFILE", "REQUIREMENTS", "banks_at_index", "common_start", "esg_screen", "flow_construction", "grandfathered", "issuer_limit", "min_trade", "paired_t", "tabc_hedge", "turnover_budget", "two_stage"]
