"""One result schema for every engine, and the comparisons a PM asks for.

``Result`` holds the same series and tables whether it came from the vectorised engine (Paper)
or the event engine (Implement): net and gross returns, costs, NAV, benchmark returns, daily
exposures, rebalances with turnover under every definition, weights, contributions by
instrument and by group (sector, rating ...), the construction log and constraint violations.
``stats()`` returns one key set (``mechanics.STAT_KEYS``) computed by one function.

``compare(a, b, ...)`` lines results up; ``explain(variant, base)`` answers "if I change my
strategy this way, what happens and why":

* what changed (the strategy diff, component by component);
* the headline differences (return, Sharpe, tracking error, IR, turnover, costs, drawdown);
* whether the return difference is distinguishable from noise (Newey-West t-statistic and a
  block-bootstrap interval of the annualised difference);
* where it came from, per year: trading costs, holdings by group (and the names that mattered
  most), cash and financing;
* how the portfolios differ on average (exposures, group weights) and by calendar year.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from nautilus_fi.kernel.mechanics import STAT_KEYS

HEADLINE = ("annual_return", "annual_vol", "sharpe", "max_drawdown", "tracking_error", "information_ratio", "active_annual_return", "turnover", "cost_drag", "avg_names", "excess_annual_return")


@dataclasses.dataclass
class Result:
    name: str
    fingerprint: str
    fidelity: str                                   # paper | implement
    strategy: object
    conventions: object
    returns: pd.Series                              # net per period; the first row is the start (0)
    gross: pd.Series
    costs: pd.Series
    nav: pd.Series
    benchmark_returns: pd.Series | None
    daily: pd.DataFrame                             # cash weight, gross / net exposure, names, excess legs
    trades: pd.DataFrame                            # one row per executed rebalance
    weights: pd.DataFrame                           # long: date, instrument_id, weight (after each rebalance)
    exposures: pd.DataFrame                         # portfolio and active exposures per day
    groups: dict                                    # attribute -> {"weight", "contribution", "benchmark_weight", "benchmark_contribution"} (dates x groups)
    contributions: pd.Series                        # arithmetic return contribution per instrument over the window
    construction: pd.DataFrame
    violations: pd.DataFrame
    missing: dict
    excess_returns: pd.Series | None = None
    risk_free: pd.Series | None = None
    diagnostics: dict = dataclasses.field(default_factory=dict)
    daily_weights: np.ndarray | None = None         # dates x instruments (float32) when kept
    ids: pd.Index | None = None
    raw: object = None                              # the engine's own result (FixedIncomeResult for Implement)
    source_version: str | None = None
    window: tuple = (None, None)
    _stats: dict | None = dataclasses.field(default=None, repr=False)

    # ------------------------------------------------------------------ compatibility
    @property
    def spec_hash(self) -> str:
        return self.fingerprint

    @property
    def active(self) -> pd.Series | None:
        return None if self.benchmark_returns is None else (self.returns - self.benchmark_returns).rename("active")

    def excess(self, freq: str | None = None) -> pd.Series:
        if self.excess_returns is None:
            raise ValueError("no excess returns: the source has neither vendor excess returns nor Treasury legs")
        if freq is None:
            return self.excess_returns
        return self.excess_returns.groupby(pd.Grouper(freq=freq)).apply(lambda g: float((1 + g).prod() - 1))

    # ------------------------------------------------------------------ statistics
    def stats(self) -> dict:
        if self._stats is None:
            m = self.strategy.resolved().metrics
            r = self.returns.iloc[1:]
            b = None if self.benchmark_returns is None else self.benchmark_returns.reindex(r.index).fillna(0.0)
            s = m.stats(r, self.gross.iloc[1:], self.costs.iloc[1:], b, None if self.risk_free is None else self.risk_free.reindex(r.index), self.trades, self.daily, None if self.excess_returns is None else self.excess_returns.iloc[1:])
            s = {"name": self.name, "fingerprint": self.fingerprint, "fidelity": self.fidelity, "start": str(r.index[0].date()) if len(r) else None, "end": str(r.index[-1].date()) if len(r) else None, "conventions": getattr(self.conventions, "name", None), **s}
            s["hard_violations"] = int((self.violations["severity"] == "HARD").sum()) if len(self.violations) and "severity" in self.violations else 0
            s["missing_zero_filled"] = int(self.missing.get("zero_filled", 0))
            for k, v in self.diagnostics.get("extra_stats", {}).items():
                s[k] = v
            self._stats = s
        return dict(self._stats)

    def summary(self) -> pd.Series:
        s = self.stats()
        return pd.Series({k: s[k] for k in ("name", "fidelity", "start", "end", *STAT_KEYS, "hard_violations") if k in s})

    def headline(self) -> str:
        s = self.stats()
        parts = [f"{self.name}: annual {_pct(s.get('annual_return'))}", f"vol {_pct(s.get('annual_vol'))}", f"Sharpe {_num(s.get('sharpe'))}", f"max DD {_pct(s.get('max_drawdown'))}"]
        if "tracking_error" in s:
            parts += [f"active {_pct(s.get('active_annual_return'))}", f"TE {_pct(s.get('tracking_error'))}", f"IR {_num(s.get('information_ratio'))}"]
        parts += [f"turnover {_num(s.get('turnover'))}x", f"costs {_pct(s.get('cost_drag'))}/yr", f"{_num(s.get('avg_names'), 0)} names"]
        return ", ".join(parts)

    def report(self) -> str:
        """Markdown for a PM: the strategy, headline statistics, groups and the largest contributors."""
        s = self.stats()
        lines = [f"# {self.name} ({self.fidelity})", "", "```", self.strategy.describe(), "```", "", f"{s.get('start')} .. {s.get('end')} ({_num(s.get('years'), 1)} years, {s.get('annualisation')})", "", "| statistic | value |", "| --- | --- |"]
        for k in HEADLINE:
            if k in s:
                v = s[k]
                lines.append(f"| {k} | {_pct(v) if k not in ('sharpe', 'information_ratio', 'turnover', 'avg_names') else _num(v)} |")
        for g, tables in self.groups.items():
            w = tables["weight"].iloc[1:].mean()
            c = tables["contribution"].iloc[1:].sum() / max(s.get("years") or 1, 1e-9)
            lines += ["", f"## By {g}", "", "| group | avg weight | contribution / yr |", "| --- | --- | --- |"]
            lines += [f"| {k} | {_pct(w.get(k))} | {_pct(c.get(k))} |" for k in w.sort_values(ascending=False).index[:15]]
        top = self.contributions.sort_values()
        if len(top):
            lines += ["", "## Largest contributors", "", ", ".join(f"{k} {_pct(v)}" for k, v in top.iloc[::-1].head(5).items()), "", "## Largest detractors", "", ", ".join(f"{k} {_pct(v)}" for k, v in top.head(5).items())]
        if len(self.violations):
            lines += ["", f"{len(self.violations)} constraint violations recorded ({s.get('hard_violations')} hard)."]
        return "\n".join(lines)

    def weights_on(self, date) -> pd.Series:
        if self.daily_weights is not None and self.ids is not None:
            i = self.returns.index.get_indexer([pd.Timestamp(date)], method="pad")[0]
            row = self.daily_weights[i]
            nz = np.nonzero(row)[0]
            return pd.Series(row[nz].astype(float), index=self.ids[nz])
        w = self.weights[self.weights["date"] <= pd.Timestamp(date)]
        if w.empty:
            return pd.Series(dtype=float)
        last = w[w["date"] == w["date"].max()]
        return last.set_index("instrument_id")["weight"]

    def daily_weights_long(self) -> pd.DataFrame:
        """Daily close weights as a long frame (what ``paper.reconcile`` needs)."""
        if self.daily_weights is None:
            raise ValueError("daily weights were not kept: run with RunSettings(record_weights=True)")
        W = self.daily_weights
        t, n = np.nonzero(W)
        return pd.DataFrame({"date": self.returns.index[t], "instrument_id": self.ids[n], "weight": W[t, n].astype(float)})

    # ------------------------------------------------------------------ comparison
    def compare(self, *others: Result) -> Comparison:
        return Comparison([self, *others])

    def explain(self, base: Result) -> Explanation:
        """Why this result differs from ``base``."""
        return explain(self, base)

    def __repr__(self) -> str:
        return f"Result({self.headline()})"


# ====================================================================================== formatting
def _pct(v, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:+.{d}%}" if v < 0 else f"{v:.{d}%}"


def _num(v, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:,.{d}f}"


def _dpct(v, d: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:+.{d}%}"


# ====================================================================================== comparisons
class Comparison:
    """Results side by side; ``explain(i)`` explains result ``i`` against the base (index 0)."""

    def __init__(self, results: list[Result], base: int = 0, notes: list[str] | None = None) -> None:
        if not results:
            raise ValueError("nothing to compare")
        self.notes = list(notes or [])
        names = [r.name for r in results]
        if len(set(names)) < len(names):
            for k, r in enumerate(results):
                if names.count(r.name) > 1:
                    results[k] = dataclasses.replace(r, name=f"{r.name}#{k}", _stats=None)
        self.results = results
        self.base = base

    def __getitem__(self, k) -> Result:
        if isinstance(k, str):
            return next(r for r in self.results if r.name == k)
        return self.results[k]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def table(self, keys=HEADLINE) -> pd.DataFrame:
        rows = {r.name: {k: r.stats().get(k) for k in keys} for r in self.results}
        return pd.DataFrame(rows).dropna(how="all")

    def differences(self, keys=HEADLINE) -> pd.DataFrame:
        t = self.table(keys)
        base = t.iloc[:, self.base]
        return t.sub(base, axis=0).drop(columns=t.columns[self.base])

    def returns(self, net: bool = True) -> pd.DataFrame:
        return pd.DataFrame({r.name: (r.returns if net else r.gross) for r in self.results})

    def best(self, metric: str = "sharpe") -> Result:
        vals = [(r.stats().get(metric, float("nan")), k) for k, r in enumerate(self.results)]
        vals = [(v, k) for v, k in vals if v is not None and math.isfinite(v)]
        if not vals:
            raise ValueError(f"no result has a finite {metric}")
        return self.results[max(vals)[1]]

    def explain(self, i: int | str = 1) -> Explanation:
        r = self[i]
        e = explain(r, self.results[self.base])
        e.notes += self.notes
        return e

    def __repr__(self) -> str:
        t = self.table()
        fmt = t.copy().astype(object)
        for k in t.index:
            fmt.loc[k] = [(_num(v) if k in ("sharpe", "information_ratio", "turnover", "avg_names") else _pct(v)) for v in t.loc[k]]
        return fmt.to_string()

    def _repr_markdown_(self) -> str:
        return self.table().to_markdown() if hasattr(self.table(), "to_markdown") else repr(self)


@dataclasses.dataclass
class Explanation:
    variant: str
    base: str
    changes: object                                  # strategy Diff
    headline: pd.DataFrame                           # statistic x (variant, base, difference)
    significance: dict
    decomposition: dict                              # per year, arithmetic: costs, holdings, cash_financing, residual
    by_group: dict                                   # attribute -> Series of holdings contribution differences per year
    names: pd.Series                                 # largest per-instrument contribution differences per year
    exposures: pd.Series                             # average exposure differences
    group_weights: dict                              # attribute -> average weight differences
    by_year: pd.DataFrame
    notes: list = dataclasses.field(default_factory=list)
    binding: dict = dataclasses.field(default_factory=dict)      # result -> {constraint: share of rebalances where it bound}

    def text(self) -> str:
        h = self.headline
        def row(k):
            if k not in h.index:
                return None
            v, b, d = h.loc[k, "variant"], h.loc[k, "base"], h.loc[k, "difference"]
            if k in ("sharpe", "information_ratio", "turnover", "avg_names"):
                return f"{k.replace('_', ' ')} {_num(v)} vs {_num(b)} ({'+' if d >= 0 else ''}{_num(d)})"
            return f"{k.replace('_', ' ')} {_pct(v)} vs {_pct(b)} ({_dpct(d)})"
        lines = [f"**{self.variant}** against **{self.base}**", "", "What changed:", str(self.changes) if self.changes else "* nothing in the strategy (different data, window or engine)", ""]
        lines += ["| | " + self.variant + " | " + self.base + " | change |", "| --- | ---: | ---: | ---: |"]
        for k in HEADLINE:
            if k in h.index and (np.isfinite(h.loc[k, "variant"]) or np.isfinite(h.loc[k, "base"])):
                v, b, d = h.loc[k, "variant"], h.loc[k, "base"], h.loc[k, "difference"]
                num = k in ("sharpe", "information_ratio", "turnover", "avg_names")
                f = (lambda x: _num(x)) if num else (lambda x: _pct(x))
                dd = (("+" if d >= 0 else "") + _num(d)) if num and np.isfinite(d) else (_dpct(d) if np.isfinite(d) else "n/a")
                lines.append(f"| {k.replace('_', ' ')} | {f(v)} | {f(b)} | {dd} |")
        lines.append("")
        sig = self.significance
        if sig.get("n", 0) > 10:
            verdict = "distinguishable from noise" if abs(sig.get("t_stat", 0)) >= 2 else "not distinguishable from noise"
            lines.append(f"The return difference of {_dpct(sig['annual_difference'])} per year is {verdict} (Newey-West t = {_num(sig.get('t_stat'))}, 95% interval {_dpct(sig.get('ci_low'))} to {_dpct(sig.get('ci_high'))} per year, {sig['n']} periods).")
        d = self.decomposition
        lines += ["", "Where the difference came from (per year, arithmetic):", f"* trading costs: {_dpct(d.get('costs'))}", f"* holdings: {_dpct(d.get('holdings'))}"]
        for g, s in self.by_group.items():
            if len(s):
                top = s.reindex(s.abs().sort_values(ascending=False).index[:4])
                lines.append(f"  * by {g}: " + ", ".join(f"{k} {_dpct(v)}" for k, v in top.items()))
        if len(self.names):
            lines.append("  * names: " + ", ".join(f"{k} {_dpct(v)}" for k, v in self.names.head(6).items()))
        lines.append(f"* cash, financing and overlays: {_dpct(d.get('cash_financing'))}")
        if abs(d.get("residual", 0.0)) > 1e-5:
            lines.append(f"* compounding and other: {_dpct(d.get('residual'))}")
        ex = [f"{k} {'+' if v >= 0 else ''}{v:.3g}" for k, v in self.exposures.items() if np.isfinite(v) and abs(v) > 1e-3]
        if ex:
            lines += ["", "Average exposure differences: " + ", ".join(ex)]
        for g, s in self.group_weights.items():
            big = s[s.abs() > 0.005].sort_values(key=abs, ascending=False).head(5)
            if len(big):
                lines.append(f"Average {g} weight differences: " + ", ".join(f"{k} {_dpct(v, 1)}" for k, v in big.items()))
        for who, b in self.binding.items():
            if b:
                lines.append(f"Binding constraints in {who} (share of rebalances): " + ", ".join(f"{k} {v:.0%}" for k, v in list(b.items())[:6]))
        if len(self.by_year) > 1:
            lines += ["", "By year: " + ", ".join(f"{y} {_dpct(r['difference'])}" for y, r in self.by_year.iterrows())]
        if self.notes:
            lines += ["", *[f"Note: {n}" for n in self.notes]]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text()

    __repr__ = __str__


def _nw_tstat(x: pd.Series, lags: int) -> float:
    from nautilus_fi.research.statistics import newey_west_tstat

    try:
        return float(newey_west_tstat(x, lags))
    except Exception:                                # noqa: BLE001 - degenerate series
        return float("nan")


def explain(a: Result, b: Result) -> Explanation:
    """Why ``a`` (the variant) differs from ``b`` (the base)."""
    from nautilus_fi.research.statistics import block_bootstrap_ci

    notes = []
    idx = a.returns.index.intersection(b.returns.index)
    if len(idx) < len(a.returns.index) or len(idx) < len(b.returns.index):
        notes.append(f"compared on the {len(idx)} common dates only")
    idx = idx[1:]
    m = a.strategy.resolved().metrics
    years = m.years(idx) if len(idx) else float("nan")
    ppy = m.periods_per_year(idx) if len(idx) > 1 else 252.0
    sa, sb = a.stats(), b.stats()
    keys = [k for k in HEADLINE if sa.get(k) is not None or sb.get(k) is not None]
    head = pd.DataFrame({"variant": [sa.get(k) for k in keys], "base": [sb.get(k) for k in keys]}, index=keys).astype(float)
    head["difference"] = head["variant"] - head["base"]
    d = (a.returns.reindex(idx) - b.returns.reindex(idx)).fillna(0.0)
    sig = {"n": len(d)}
    if len(d) > 10:
        lags = int(min(max(4 * (len(d) / 100) ** (2 / 9), 1), 20))
        try:
            _, lo, hi = block_bootstrap_ci(d.to_numpy(), stat=np.mean, block_len=max(lags, 5), n_boot=500, seed=0)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            lo, hi = (float("nan"), float("nan"))
        sig.update({"annual_difference": float(d.mean() * ppy), "t_stat": _nw_tstat(d, lags), "ci_low": lo * ppy, "ci_high": hi * ppy, "nw_lags": lags})
    yrs = years if years and math.isfinite(years) and years > 0 else 1.0
    gross_d = float((a.gross.reindex(idx) - b.gross.reindex(idx)).sum()) / yrs
    cost_d = -float((a.costs.reindex(idx).fillna(0) - b.costs.reindex(idx).fillna(0)).sum()) / yrs
    net_d = float(d.sum()) / yrs
    asset_a = a.contributions.sum() if len(a.contributions) else 0.0
    asset_b = b.contributions.sum() if len(b.contributions) else 0.0
    holdings_d = float(asset_a - asset_b) / yrs
    by_group = {}
    group_w = {}
    for g in set(a.groups) & set(b.groups):
        ca = a.groups[g]["contribution"].reindex(idx).fillna(0.0).sum()
        cb = b.groups[g]["contribution"].reindex(idx).fillna(0.0).sum()
        by_group[g] = (ca.sub(cb, fill_value=0.0) / yrs).sort_values()
        wa = a.groups[g]["weight"].reindex(idx).mean()
        wb = b.groups[g]["weight"].reindex(idx).mean()
        group_w[g] = wa.sub(wb, fill_value=0.0)
    names = a.contributions.sub(b.contributions, fill_value=0.0) / yrs
    names = names.reindex(names.abs().sort_values(ascending=False).index[:10])
    ex_cols = [c for c in a.exposures.columns if c in b.exposures.columns]
    exposures = (a.exposures[ex_cols].reindex(idx).mean() - b.exposures[ex_cols].reindex(idx).mean()) if ex_cols else pd.Series(dtype=float)
    same = [c for c in exposures.index if c.startswith("active_") and c[7:] in exposures.index and abs(exposures[c] - exposures[c[7:]]) < 1e-9]
    exposures = exposures.drop(index=same)                 # same benchmark: active and absolute differences coincide
    ya = (1 + a.returns.reindex(idx)).groupby(idx.year).prod() - 1
    yb = (1 + b.returns.reindex(idx)).groupby(idx.year).prod() - 1
    by_year = pd.DataFrame({"variant": ya, "base": yb, "difference": ya - yb})
    decomposition = {"net": net_d, "gross": gross_d, "costs": cost_d, "holdings": holdings_d, "cash_financing": gross_d - holdings_d, "residual": net_d - gross_d - cost_d}
    if len(d) and float(d.abs().max()) < 1e-12:
        notes.append("the returns are identical: the change did not alter a single trade (a limit that never binds, or a component the construction ignores)")
    binding = {r.name: _binding_shares(r) for r in (a, b)}
    if a.fidelity != b.fidelity:
        notes.append(f"fidelities differ ({a.fidelity} vs {b.fidelity}): part of the gap is execution realism, see paper.reconcile")
    try:
        changes = a.strategy.diff(b.strategy)
    except Exception as e:                           # noqa: BLE001 - strategies from different code
        changes = f"(diff unavailable: {e})"
    return Explanation(a.name, b.name, changes, head, sig, decomposition, by_group, names, exposures, group_w, by_year, notes, binding)


def _binding_shares(r: Result) -> dict:
    c = r.construction
    if c is None or c.empty or "binding" not in c:
        return {}
    reb = c[c.get("action", "rebalance") == "rebalance"] if "action" in c else c
    counts: dict = {}
    for row in reb["binding"].dropna():
        for label in {":".join(x.split(":")[:2]) if x.count(":") > 1 else x for x in str(row).split("; ") if x}:
            counts[label] = counts.get(label, 0) + 1
    n = max(len(reb), 1)
    return dict(sorted(((k, v / n) for k, v in counts.items()), key=lambda kv: -kv[1]))


def compare(*results: Result) -> Comparison:
    return Comparison(list(results))


__all__ = ["HEADLINE", "Comparison", "Explanation", "Result", "compare", "explain"]
