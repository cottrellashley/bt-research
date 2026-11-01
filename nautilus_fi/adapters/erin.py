"""Erin Beta+ (``fi-backtest-experimentation-erin``): the framework's behaviour as named
components over the kernel, and its parity cases (gap analysis §5.1, ERIN-01 to ERIN-23).

Conventions come from the ``erin_beta_plus`` profile (``Strategy.under("erin_beta_plus",
refill=...)``): T-1 inputs earn the T return, published daily benchmark weights, one-way-buys
turnover, and bad dates (>= 80% exactly-zero returns) refilled. The components:

* ``stats_ru_benchmark``: the ``_STATS`` / ``RU`` universe pair, weights switching at month end
  (ERIN-01, ERIN-02);
* ``start_date_field`` and ``guard_duplicate_dates``: START_DATE-labelled returns (a month-end
  return belongs to the next month) and the duplicate-date guard (ERIN-04);
* ``orphan_refill``: bad dates refilled from the next orphan date within 4 days (ERIN-05);
* ``composite_score_2``: Composite Score 2.0 on 0-100, with its null rules and standalone mode
  (ERIN-07);
* ``mri_ratio`` and ``mri_dts_band``: the MRI score mapped to a DTS (or OAS) target ratio, stale
  inputs a hard error, manual overrides, applied as a band that changes by date (ERIN-08);
* ``parse_filter`` / ``filters`` / ``filter_audit`` / ``analyst_scores``: the filter DSL (SELL,
  EXCLUDE, TICKER_UNDERWEIGHT; ``@bench.X`` and ``@universe.X.pN``), its per-rebalance audit and
  the analyst-recommendation input (ERIN-09);
* ``lp_construction``: the LP form: a soft penalty on summed group deviations, IRL1 sparsity and
  one-way-buys turnover with a recorded tier ladder (ERIN-10 to ERIN-13);
* ``schedule`` and ``event_cap``: the anchored weekly schedule and per-event caps (ERIN-14);
* ``costs``: buy-only OASD x spread (3bp IG / 10bp HY) above a 1bp threshold, audited (ERIN-15);
* ``credit_futures``, ``treasury_waterfall`` and ``live_proposal``: the live workflow (ERIN-19 to
  ERIN-22);
* ``apply_tuning`` and ``eligible``: tuning profiles and eligibility gates (ERIN-23).

Nothing here is confirmed: the parity cases (``PARITY``) name the canonical v2 runs whose stored
outputs must be reproduced first. Values marked "to confirm" are readings of the review, not of
the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.adapters.parity import ParityCase, Tolerance
from nautilus_fi.kernel.data import DataQualityError, FieldInfo, MissingDataError
from nautilus_fi.kernel.expr import Expr, Field, lift
from nautilus_fi.kernel.expr import where as _where

PROFILE = "erin_beta_plus"
REQUIREMENTS = tuple(f"ERIN-{i:02d}" for i in (1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23))


# ------------------------------------------------------------------------------------------ universes and returns
def stats_ru_benchmark(stats: str, ru: str):
    """The ``_STATS`` (next month's composition, published ahead: what decisions see) and ``RU``
    (what the index holds this month: returns and analytics) pair as one benchmark; the switch
    happens at month end, when the ``_STATS`` file becomes the returns universe."""
    from nautilus_fi.kernel.benchmark import Index, PairedIndex

    return PairedIndex(Index(stats, drift="STATIC"), Index(ru, drift="STATIC"))


def start_date_field(name: str = "tr", description: str = "total return labelled by period start (Erin START_DATE)", **kw) -> FieldInfo:
    """A return field labelled by the start of its period: a month-end return belongs to the next
    month, so it realises one period later than an end-labelled one."""
    return FieldInfo(name, description, "fraction", label="start", **kw)


def guard_duplicate_dates(frame: pd.DataFrame, date_col: str = "date", keys: tuple[str, ...] = ("instrument_id",)) -> pd.DataFrame:
    """Raise on duplicated (date, key) rows instead of letting a join double them."""
    cols = [date_col, *keys]
    dup = frame.duplicated(cols, keep=False)
    if dup.any():
        ex = frame.loc[dup, cols].drop_duplicates().head(5).to_dict("records")
        raise DataQualityError(f"{int(dup.sum())} rows share a {'/'.join(cols)} (e.g. {ex}): the file has duplicate dates")
    return frame


def orphan_refill(returns: pd.DataFrame, calendar, threshold: float = 0.8, max_days: int = 4, alive: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erin's bad-date refill. ``returns``: dates x instruments, including orphan dates (dates
    with data that are not trading days: a vendor's late file). A trading day is bad when at least
    ``threshold`` of the live names returned exactly zero; it is refilled from the first orphan
    date after it, within ``max_days`` calendar days, that is not itself bad (each orphan used
    once). Returns the refill frame on the trading days (values on refilled days, NaN elsewhere:
    name it as ``Refill(field)`` / ``under(..., refill=field)``) and a report (bad_date,
    orphan_date, zero_share, resolved)."""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize()
    r = returns.copy()
    r.index = pd.DatetimeIndex(pd.to_datetime(r.index)).normalize()
    on = r.reindex(cal)
    live = (alive.reindex(index=cal, columns=r.columns).fillna(False).astype(bool) if alive is not None else on.notna())
    zero_share = ((on == 0.0) & live).sum(axis=1) / live.sum(axis=1).replace(0, np.nan)
    orphans = r.index.difference(cal).sort_values()
    if alive is not None:
        o_live = alive.reindex(index=orphans, columns=r.columns).fillna(False).astype(bool)
    else:
        o_live = r.loc[orphans].notna()
    o_zero = ((r.loc[orphans] == 0.0) & o_live).sum(axis=1) / o_live.sum(axis=1).replace(0, np.nan)
    refill = pd.DataFrame(np.nan, index=cal, columns=r.columns)
    used: set = set()
    rows = []
    for d in zero_share.index[zero_share.fillna(0.0) >= threshold]:
        cands = [o for o in orphans if d < o <= d + pd.Timedelta(days=max_days) and o not in used and not (o_zero.get(o, 1.0) >= threshold)]
        if cands:
            o = cands[0]
            used.add(o)
            refill.loc[d] = r.loc[o].to_numpy()
        rows.append({"bad_date": d, "orphan_date": cands[0] if cands else pd.NaT, "zero_share": float(zero_share[d]), "resolved": bool(cands)})
    return refill, pd.DataFrame(rows, columns=["bad_date", "orphan_date", "zero_share", "resolved"])


# ------------------------------------------------------------------------------------------ Composite Score 2.0
def composite_score_2(components: dict[str, pd.DataFrame], weights: dict[str, float] | None = None, carry_days: int = 7, fallback: float = 19.0, no_row: float = 50.0, min_nulls: int = 2, standalone: str | None = None, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """Composite Score 2.0 (0-100) from component scores (dates x instruments, each 0-100).
    A name-date with ``min_nulls`` or more null components takes its last valid score within
    ``carry_days`` calendar days, otherwise ``fallback`` (19); a name with no row on a date scores
    ``no_row`` (50); with fewer nulls the available components are averaged at their weights,
    renormalised. ``standalone``: one component as the score, under the same rules. ``rows``:
    which name-dates have a row (default: any component present)."""
    names = [standalone] if standalone else list(components)
    if standalone and standalone not in components:
        raise KeyError(f"standalone component {standalone!r} not in {list(components)}")
    idx = pd.DatetimeIndex(sorted(set().union(*[set(pd.to_datetime(f.index)) for f in components.values()])))
    cols = pd.Index(sorted(set().union(*[set(f.columns) for f in components.values()])))
    every = {c: f.set_axis(pd.to_datetime(f.index), axis=0).reindex(index=idx, columns=cols).astype(float) for c, f in components.items()}
    frames = {c: every[c] for c in names}
    for c, f in frames.items():
        v = f.to_numpy()
        if np.nanmax(np.abs(v), initial=0.0) > 100.0 + 1e-9 or np.nanmin(v, initial=0.0) < -1e-9:
            raise DataQualityError(f"component {c!r} is outside 0-100 (range {np.nanmin(v):.3g} .. {np.nanmax(v):.3g}): Composite Score 2.0 components are on a 0-100 scale")
    w = {c: float((weights or {}).get(c, 1.0)) for c in names}
    num = sum(frames[c].fillna(0.0) * w[c] for c in names)
    den = sum(frames[c].notna() * w[c] for c in names)
    nulls = sum(frames[c].isna().astype(int) for c in names)
    present = rows.reindex(index=idx, columns=cols).fillna(False).astype(bool) if rows is not None else sum(f.notna().astype(int) for f in every.values()) > 0
    need = min(min_nulls, len(names))
    score = (num / den.replace(0, np.nan)).where(nulls < need)
    out = score.copy()
    valid = score.notna()
    for col in cols:
        s = score[col]
        last_val = s.ffill()
        last_date = pd.Series(np.where(valid[col], idx, pd.NaT), index=idx).ffill()
        age = (pd.Series(idx, index=idx) - pd.to_datetime(last_date)).dt.days
        carried = last_val.where(age <= carry_days)
        gap = present[col] & s.isna()
        out.loc[gap, col] = carried[gap].fillna(fallback)
    out = out.where(present, no_row)
    return out.clip(0.0, 100.0)


# ------------------------------------------------------------------------------------------ MRI mapping
class StaleInputError(MissingDataError):
    pass


def mri_ratio(score: pd.Series, dates, knots: list[tuple[float, float]], max_stale_days: int = 7, overrides: dict | pd.Series | None = None) -> pd.Series:
    """The MRI score as a target ratio (portfolio DTS / benchmark DTS, or OAS) on ``dates``: the
    last score at or before each date, mapped piecewise linearly through ``knots`` (score, ratio)
    and clamped at the end knots. A score older than ``max_stale_days`` calendar days is a hard
    error. ``overrides``: date -> ratio from that date on (None ends the override)."""
    s = pd.Series(score, dtype=float).dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    xs, ys = zip(*sorted(knots), strict=True)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    pos = s.index.searchsorted(d, side="right") - 1
    out = np.full(len(d), np.nan)
    for i, (day, p) in enumerate(zip(d, pos, strict=True)):
        if p < 0:
            raise StaleInputError(f"no MRI score on or before {day.date()}")
        age = (day - s.index[p]).days
        if age > max_stale_days:
            raise StaleInputError(f"the MRI score on {day.date()} is {age} days old (from {s.index[p].date()}), over {max_stale_days}: refresh the input")
        out[i] = float(np.interp(s.iloc[p], xs, ys))
    res = pd.Series(out, index=d, name="mri_ratio")
    if overrides is not None:
        ov = pd.Series(overrides, dtype=object)
        ov.index = pd.to_datetime(ov.index)
        active = None
        for day in d:
            hits = ov[ov.index <= day]
            active = hits.iloc[-1] if len(hits) else None
            if active is not None and not (isinstance(active, float) and np.isnan(active)):
                res[day] = float(active)
    return res


def mri_dts_band(series: str = "erin.mri_ratio", band: float = 0.05, exposure: str = "dts") -> list:
    """Constraints holding the portfolio's ``exposure`` at the dated target ratio of the
    benchmark's, within ``band`` (ratio points): bounds that change by date, read from the
    ``series`` the source carries (``mri_ratio`` output)."""
    from nautilus_fi.kernel.portfolio import ActiveExposure

    m = ActiveExposure(exposure, relative=True)
    return [m >= Field(series) - 1.0 - band, m <= Field(series) - 1.0 + band]


# ------------------------------------------------------------------------------------------ filter DSL
ACTIONS = ("SELL", "EXCLUDE", "TICKER_UNDERWEIGHT")
_TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)|(?P<str>'[^']*'|\"[^\"]*\")|(?P<ref>@(?:bench|universe)\.[A-Za-z_][\w.]*)|(?P<op><=|>=|==|!=|<|>|\(|\))|(?P<word>[A-Za-z_][\w.]*))")


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if m is None or m.end() == pos:
            raise ValueError(f"cannot read the filter at {text[pos:pos + 20]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def parse_filter(text: str, benchmark=None) -> Expr:
    """A filter expression as a kernel predicate: comparisons of fields, numbers, quoted strings
    and references (``@bench.X``: the benchmark's weighted average of X on the date;
    ``@universe.X.pN``: the N-th percentile of X across the universe), joined by AND / OR / NOT
    and parentheses."""
    from nautilus_fi.kernel.benchmark import BenchmarkStat

    toks = _tokens(text)
    i = 0

    def peek():
        return toks[i] if i < len(toks) else (None, None)

    def take():
        nonlocal i
        t = peek()
        i += 1
        return t

    def operand():
        kind, val = take()
        if kind == "num":
            return float(val)
        if kind == "str":
            return val[1:-1]
        if kind == "ref":
            scope, rest = val[1:].split(".", 1)
            if scope == "bench":
                if benchmark is None:
                    raise ValueError(f"{val} needs the benchmark: parse_filter(text, benchmark)")
                return BenchmarkStat(rest, benchmark)
            m = re.fullmatch(r"(.+)\.p(\d{1,3})", rest)
            if m is None:
                raise ValueError(f"{val}: a universe reference names a percentile, e.g. @universe.oas.p90")
            return Field(m.group(1)).cs_quantile(int(m.group(2)) / 100.0)
        if kind == "op" and val == "(":
            e = disj()
            if take() != ("op", ")"):
                raise ValueError(f"unbalanced parentheses in {text!r}")
            return e
        if kind == "word" and val.upper() not in ("AND", "OR", "NOT"):
            return Field(val)
        raise ValueError(f"unexpected {val!r} in {text!r}")

    def comparison():
        left = operand()
        kind, val = peek()
        if kind == "op" and val in ("<", "<=", ">", ">=", "==", "!="):
            take()
            right = operand()
            lhs = lift(left) if not isinstance(left, Expr) else left
            return {"<": lhs.__lt__, "<=": lhs.__le__, ">": lhs.__gt__, ">=": lhs.__ge__, "==": lhs.__eq__, "!=": lhs.__ne__}[val](right)
        if isinstance(left, Expr) and left.kind == "bool":
            return left
        raise ValueError(f"{text!r}: a filter term must compare two values")

    def negation():
        kind, val = peek()
        if kind == "word" and val.upper() == "NOT":
            take()
            return ~negation()
        return comparison()

    def conj():
        e = negation()
        while peek()[0] == "word" and peek()[1].upper() == "AND":
            take()
            e = e & negation()
        return e

    def disj():
        e = conj()
        while peek()[0] == "word" and peek()[1].upper() == "OR":
            take()
            e = e | conj()
        return e

    out = disj()
    if i != len(toks):
        raise ValueError(f"unexpected {toks[i][1]!r} in {text!r}")
    return out


@dataclass(frozen=True)
class FilterRule:
    action: str
    text: str
    name: str = ""


@dataclass
class FilterSet:
    """Rules as kernel parts: ``universe`` (SELL rules removed: held names are sold), and
    ``constraints`` (EXCLUDE: no buys, held names may shrink; TICKER_UNDERWEIGHT: the issuer at or
    below its index weight)."""

    rules: list[FilterRule]
    exprs: dict[str, Expr]
    universe: Expr | None
    constraints: list = field(default_factory=list)

    def apply(self, strategy):
        """``strategy`` with the rules: SELL in its universe, the constraints added to an
        ``Optimise`` portfolio (other constructions cannot take them: it raises)."""
        from nautilus_fi.kernel.portfolio import Optimise
        from nautilus_fi.kernel.strategy import StrategyError

        s = strategy
        if self.universe is not None:
            base = s.resolved().universe
            s = s.with_(universe=base & self.universe)
        if self.constraints:
            port = s.resolved().portfolio
            if not isinstance(port, Optimise):
                raise StrategyError(f"EXCLUDE / TICKER_UNDERWEIGHT rules are optimiser constraints; the portfolio is {port!r}")
            s = s.with_(portfolio=port.add(*self.constraints))
        return s


def filters(rules: list[str | tuple | FilterRule], benchmark=None, issuer: str = "issuer") -> FilterSet:
    """``rules``: ``"SELL: rating_score > 10.5"`` strings, (action, text) pairs or ``FilterRule``s."""
    from nautilus_fi.kernel.portfolio import ActiveWeight, NoBuy

    parsed: list[FilterRule] = []
    for k, r in enumerate(rules):
        if isinstance(r, FilterRule):
            fr = r
        elif isinstance(r, tuple):
            fr = FilterRule(str(r[0]).upper(), r[1], r[2] if len(r) > 2 else "")
        else:
            act, _, txt = str(r).partition(":")
            fr = FilterRule(act.strip().upper(), txt.strip())
        if fr.action not in ACTIONS:
            raise ValueError(f"unknown filter action {fr.action!r}; known: {ACTIONS}")
        parsed.append(FilterRule(fr.action, fr.text, fr.name or f"{fr.action.lower()}_{k + 1}"))
    exprs = {r.name: parse_filter(r.text, benchmark) for r in parsed}
    sells = [exprs[r.name] for r in parsed if r.action == "SELL"]
    universe = None
    for e in sells:
        universe = ~e if universe is None else universe & ~e
    cons = []
    for r in parsed:
        if r.action == "EXCLUDE":
            cons.append(NoBuy(exprs[r.name]))
        elif r.action == "TICKER_UNDERWEIGHT":
            cons.append(ActiveWeight(issuer).where(exprs[r.name]) <= 0.0)
    return FilterSet(parsed, exprs, universe, cons)


def filter_audit(fs: FilterSet, source, start=None, end=None, schedule="BME", universe=None) -> pd.DataFrame:
    """Which names each rule hits on each decision date: date, rule, action, n_hit, names."""
    from nautilus_fi.domain.calendars import parse_calendar
    from nautilus_fi.kernel.engine import prepare
    from nautilus_fi.kernel.expr import alive_mask
    from nautilus_fi.kernel.strategy import Strategy

    any_rule = None
    for e in fs.exprs.values():
        any_rule = e if any_rule is None else any_rule | e
    probe = Strategy("filter_audit", universe=any_rule if universe is None else universe & any_rule)      # loads what the rules read
    prep = prepare([probe], source, start, end)
    ctx = prep.ctx
    U = alive_mask(ctx) if universe is None else np.asarray(universe.evaluate(ctx), dtype=bool) & alive_mask(ctx)
    dates = ctx.panel.dates
    lo = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start)))
    days = list(dates.date[lo:])
    rows_ = dates.get_indexer(pd.to_datetime(parse_calendar(schedule).dates(days, True)))
    rows = []
    for r in fs.rules:
        hit = np.asarray(fs.exprs[r.name].evaluate(ctx.within(U, "audit")), dtype=bool)
        for t in rows_:
            if t < 0:
                continue
            h = hit[t] & U[t]
            rows.append({"date": dates[t], "rule": r.name, "action": r.action, "text": r.text, "n_hit": int(h.sum()), "names": ",".join(map(str, ctx.panel.ids[h][:20]))})
    return pd.DataFrame(rows, columns=["date", "rule", "action", "text", "n_hit", "names"])


DECODE = {"STRONG_BUY": 2, "STRONG BUY": 2, "BUY": 1, "OUTPERFORM": 1, "OVERWEIGHT": 1, "ADD": 1, "HOLD": 0, "NEUTRAL": 0, "MARKETWEIGHT": 0, "MARKET WEIGHT": 0, "UNDERWEIGHT": -1, "UNDERPERFORM": -1, "REDUCE": -1, "SELL": -1, "STRONG_SELL": -2, "STRONG SELL": -2}


def analyst_scores(recs: pd.DataFrame, dates, ids=None, max_age_days: int = 14, decode: dict | None = None, date_col: str = "date", id_col: str = "instrument_id", value_col: str = "recommendation") -> pd.DataFrame:
    """Analyst recommendations as a score from -2 to +2 (dates x instruments): the latest
    recommendation at or before each date, if at most ``max_age_days`` old; text decoded by
    ``decode`` (default ``DECODE``), numbers taken as they are. Unknown labels raise."""
    m = {k.upper(): v for k, v in (decode or DECODE).items()}
    r = recs[[date_col, id_col, value_col]].copy()
    r[date_col] = pd.to_datetime(r[date_col])

    def dec(x):
        if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x):
            if not -2 <= x <= 2:
                raise DataQualityError(f"recommendation {x} is outside -2..+2")
            return float(x)
        key = str(x).strip().upper()
        if key not in m:
            raise DataQualityError(f"unknown recommendation {x!r}; decode it: analyst_scores(..., decode={{...}})")
        return float(m[key])

    r["score"] = r[value_col].map(dec)
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    ids = pd.Index(ids) if ids is not None else pd.Index(sorted(r[id_col].astype(str).unique()))
    out = pd.DataFrame(np.nan, index=d, columns=ids)
    for i, g in r.sort_values(date_col).groupby(r[id_col].astype(str)):
        if i not in out.columns:
            continue
        g = g.drop_duplicates(date_col, keep="last").set_index(date_col)["score"]
        pos = g.index.searchsorted(d, side="right") - 1
        ok = pos >= 0
        vals = np.full(len(d), np.nan)
        vals[ok] = g.to_numpy()[pos[ok]]
        age = np.full(len(d), np.inf)
        age[ok] = (d[ok] - g.index[pos[ok]]).days
        out[i] = np.where(age <= max_age_days, vals, np.nan)
    return out


# ------------------------------------------------------------------------------------------ LP form, schedule, costs
def lp_construction(alpha=None, groups: tuple[str, ...] = ("sector",), group_penalty: float = 50.0, sparsity: tuple[int, int] | None = (40, 60), buy_turnover: float | None = None, tiers: tuple[float, ...] = (1.5, 2.0), constraints: tuple = ()):
    """Erin's LP: maximise ``alpha`` net of costs with a soft penalty (``group_penalty`` per unit)
    on the summed absolute active weight of each group, IRL1 sparsity to ``sparsity`` names (None:
    off) and a one-way-buys turnover limit relaxed through ``tiers`` (recorded per rebalance)."""
    from nautilus_fi.construct.optimiser import OptimiserSettings
    from nautilus_fi.kernel.portfolio import (
        ActiveWeight,
        Alpha,
        Constraint,
        Costs,
        Optimise,
        Sparse,
        Turnover,
    )

    cons = [Constraint(ActiveWeight(g), 0.0, 0.0, hard=False, penalty=group_penalty) for g in groups]
    if buy_turnover is not None:
        cons.append(Turnover("ONE_WAY_BUYS") <= buy_turnover)
    opt = Optimise((alpha if alpha is not None else Alpha()) - Costs(), *cons, *constraints, settings=OptimiserSettings(on_infeasible="relax", relax_tiers=tuple(tiers)))
    return Sparse(opt, names=sparsity) if sparsity is not None else opt


def schedule(offsets: tuple[int, ...] = (7, 14, 21), min_gap_days: int = 7, anchors=()):
    """The anchored weekly schedule (to confirm: offsets from the month end)."""
    from nautilus_fi.domain.calendars import AnchoredWeekly

    return AnchoredWeekly(tuple(offsets), min_gap_days, tuple(anchors))


def event_cap(max_buys: float):
    """A per-event cap: one-way buys of at most ``max_buys`` of NAV at each rebalance."""
    from nautilus_fi.kernel.portfolio import Turnover

    return Turnover("ONE_WAY_BUYS") <= max_buys


def costs(ig_bp: float = 3.0, hy_bp: float = 10.0, threshold_bp: float = 1.0, hy_above: float = 10.5, rating: str = "rating_score", duration: str = "oasd"):
    """Buys only, at spread duration x (``ig_bp`` for IG, ``hy_bp`` for HY: ``rating`` above
    ``hy_above``), nothing below ``threshold_bp`` of NAV; each charge is audited."""
    from nautilus_fi.kernel.mechanics import BuyOnly, DurationSpreadCost

    return BuyOnly(DurationSpreadCost(_where(Field(rating) > hy_above, hy_bp, ig_bp), duration), threshold_bp)


# ------------------------------------------------------------------------------------------ live workflow
def credit_futures(instruments=("CDX_IG",), field: str = "dts", lot: float | None = None, **kw):
    """Credit futures sized to close the portfolio's DTS gap to the benchmark."""
    from nautilus_fi.kernel.portfolio import Hedge

    return Hedge(tuple(instruments), (field,), active=True, lot=lot, **kw)


def treasury_waterfall(buckets=(("krd_30", "US"), ("krd_10", "TY"), ("krd_5", "FV"), ("krd_2", "TU")), targets: dict | None = None, **kw):
    """Treasury futures by key rate, longest bucket first (to confirm: the contracts per bucket)."""
    from nautilus_fi.kernel.portfolio import KRDWaterfall

    return KRDWaterfall(tuple(buckets), targets, **kw)


def live_proposal(strategy, source, as_of, holdings_face, prices, bonds, cash, path=None, **kw):
    """The live proposal (ask for buys, bid for sells; buys down, trims to nearest, full sales
    exact), checks with severities, futures lines when ``futures`` are given, and the headerless
    QOE file at ``path`` (only without blockers)."""
    from nautilus_fi.propose.proposal import propose_from_strategy
    from nautilus_fi.propose.writers import QOE_TSV, write_orders

    p = propose_from_strategy(strategy, source, as_of, holdings_face, prices, bonds, cash, **kw)
    if path is not None:
        write_orders(p, path, QOE_TSV)
    return p


# ------------------------------------------------------------------------------------------ tuning and gates
TUNING_PROFILES: dict[str, dict] = {}                     # name -> {path: value}: load from configs/canonical/v2 (to confirm)


def apply_tuning(strategy, profile: dict | str):
    """``strategy`` with a tuning profile's settings (``{"portfolio.n": 40, "execution.lag": 1}``)."""
    prof = TUNING_PROFILES[profile] if isinstance(profile, str) else profile
    for path, value in prof.items():
        strategy = strategy.set(path, value)
    return strategy


def eligible(result, max_failed: int = 0, allow_relaxed: bool = False, max_turnover: float | None = None):
    """Erin's eligibility gate for a tuning run: no failed rebalance, no relaxed limit (unless
    allowed) and turnover within ``max_turnover``."""
    from nautilus_fi.kernel.validation import eligible as _eligible

    return _eligible(result, max_failed=max_failed, allow_relaxed=allow_relaxed, max_turnover=max_turnover)


# ------------------------------------------------------------------------------------------ parity
_TOL = (Tolerance("returns", bp_per_year=10.0, min_corr=0.999), Tolerance("turnover", abs=0.02))      # proposed: agree under §9.2
PARITY = [ParityCase(code, "erin", "configs/canonical/v2/canonical_runs.yaml", _TOL, notes="canonical v2 run; JTJSHY read from a photo [?]" if code == "JTJSHY" else "canonical v2 run") for code in ("147HYC", "147IGC", "147SHY", "JTJSHY")]


__all__ = ["ACTIONS", "DECODE", "PARITY", "PROFILE", "REQUIREMENTS", "TUNING_PROFILES", "FilterRule", "FilterSet", "StaleInputError", "analyst_scores", "apply_tuning", "composite_score_2", "costs", "credit_futures", "eligible", "event_cap", "filter_audit", "filters", "guard_duplicate_dates", "live_proposal", "lp_construction", "mri_dts_band", "mri_ratio", "orphan_refill", "parse_filter", "schedule", "start_date_field", "stats_ru_benchmark", "treasury_waterfall"]
