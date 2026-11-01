"""Parity harness: prove an adapter against its legacy framework's own outputs.

A ``ParityCase`` names a legacy run (where its golden output lives, the window, how to build the
bt-research strategy) and the tolerances it must meet. ``compare`` sets a result (or a series, or
a table) against the golden output and returns a ``ParityReport``: one row per metric with ours,
theirs, the difference, the tolerance and whether it holds. ``profile_confirmation`` turns a set
of passing reports into the evidence a reviewer commits with the change that sets
``confirmed=True`` on the profile in ``domain/conventions.py``. Nothing here flips the flag: a
profile is confirmed by a reviewed change, not by a file a run leaves behind.

Tolerances are proposals until they are agreed (gap analysis §9.2).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Tolerance:
    """``metric``: ``returns`` (the per-period return series: annualised mean difference within
    ``bp_per_year`` and correlation at least ``min_corr``), a statistic of ``Result.stats()``
    (``annual_return``, ``sharpe``, ``volatility``, ``max_drawdown``, ``turnover`` ...: absolute
    difference within ``abs``), ``weights`` (largest weight difference on common dates within
    ``abs``) or ``table:<column>`` (a golden table's column, largest difference within ``abs``)."""

    metric: str
    abs: float | None = None
    bp_per_year: float | None = None
    min_corr: float | None = None


@dataclass(frozen=True)
class ParityCase:
    name: str
    framework: str
    golden: str                                         # where the golden output is, in the legacy repository
    tolerances: tuple[Tolerance, ...]
    window: tuple[str, str] | None = None
    build: Callable | None = None                       # build(**kw) -> Strategy
    notes: str = ""


@dataclass
class ParityReport:
    case: ParityCase
    rows: pd.DataFrame                                  # metric, ours, theirs, difference, tolerance, ok
    golden_sha256: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(len(self.rows)) and bool(self.rows["ok"].all())

    def failures(self) -> pd.DataFrame:
        return self.rows[~self.rows["ok"]]

    def evidence(self) -> dict:
        return {"case": self.case.name, "framework": self.case.framework, "golden": self.case.golden, "golden_sha256": self.golden_sha256, "window": self.case.window, "passed": self.passed, "metrics": json.loads(self.rows.to_json(orient="records"))}

    def to_markdown(self) -> str:
        lines = [f"# Parity: {self.case.name} ({self.case.framework})", "", f"golden `{self.case.golden}` · sha256 `{self.golden_sha256}` · {'PASSED' if self.passed else 'FAILED'}", "", "| metric | ours | theirs | difference | tolerance | ok |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {r.metric} | {_f(r.ours)} | {_f(r.theirs)} | {_f(r.difference)} | {r.tolerance} | {'yes' if r.ok else 'NO'} |" for r in self.rows.itertuples()]
        return "\n".join(lines)


def _f(x) -> str:
    return f"{x:.6g}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else str(x)


class ParityFailure(AssertionError):
    pass


def load_golden(path: str | Path, kind: str = "returns", date_col: str = "date", value_col: str | None = None) -> tuple[pd.Series | pd.DataFrame, str]:
    """A golden output and its content hash. ``kind="returns"``: a CSV / Parquet of per-period
    returns (``date`` plus one value column, or ``value_col``) -> Series; ``"table"``: the frame
    as stored; ``"stats"``: a JSON object of statistics."""
    p = Path(path)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if kind == "stats":
        return json.loads(p.read_text()), sha
    df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)
    if kind == "table":
        return df, sha
    df[date_col] = pd.to_datetime(df[date_col])
    col = value_col or next(c for c in df.columns if c != date_col)
    return df.set_index(date_col)[col].astype(float).sort_index(), sha


def _ppy(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 252.0
    days = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    return 12.0 if days > 20 else (52.0 if days > 5 else 252.0)


def compare(ours, golden, case: ParityCase, golden_sha256: str | None = None) -> ParityReport:
    """``ours``: a kernel ``Result``, a return Series or a table; ``golden``: a return Series, a
    dict of statistics, a weights frame (date, instrument_id, weight) or a table, as the case's
    tolerances need."""
    from nautilus_fi.kernel.result import Result

    rows = []
    for tol in case.tolerances:
        m = tol.metric
        if m == "returns":
            r = ours.returns if isinstance(ours, Result) else pd.Series(ours, dtype=float)
            g = golden if isinstance(golden, pd.Series) else pd.Series(golden.get("returns"), dtype=float)
            a, b = r.align(g, join="inner")
            ok_ = a.notna() & b.notna()
            a, b = a[ok_], b[ok_]
            if len(a) < 3:
                rows.append({"metric": "returns: common periods", "ours": len(a), "theirs": len(g), "difference": np.nan, "tolerance": ">= 3", "ok": False})
                continue
            diff_bp = float((a - b).mean() * _ppy(a.index) * 1e4)
            corr = float(a.corr(b))
            rows.append({"metric": "returns: annualised difference (bp)", "ours": float(a.mean() * _ppy(a.index) * 1e4), "theirs": float(b.mean() * _ppy(b.index) * 1e4), "difference": diff_bp, "tolerance": f"|d| <= {tol.bp_per_year}", "ok": tol.bp_per_year is None or abs(diff_bp) <= tol.bp_per_year})
            if tol.min_corr is not None:
                rows.append({"metric": "returns: correlation", "ours": corr, "theirs": 1.0, "difference": 1.0 - corr, "tolerance": f">= {tol.min_corr}", "ok": corr >= tol.min_corr})
        elif m == "weights":
            w = ours.weights if isinstance(ours, Result) else ours
            a = w.pivot_table(index="date", columns="instrument_id", values="weight", aggfunc="sum")
            b = golden.pivot_table(index="date", columns="instrument_id", values="weight", aggfunc="sum")
            dates = a.index.intersection(b.index)
            cols = a.columns.union(b.columns)
            d = (a.reindex(index=dates, columns=cols).fillna(0.0) - b.reindex(index=dates, columns=cols).fillna(0.0)).abs()
            worst = float(d.to_numpy().max()) if d.size else np.nan
            rows.append({"metric": "weights: largest difference", "ours": len(dates), "theirs": len(b.index), "difference": worst, "tolerance": f"<= {tol.abs}", "ok": bool(len(dates)) and np.isfinite(worst) and worst <= (tol.abs or 0.0)})
        elif m.startswith("table:"):
            col = m.split(":", 1)[1]
            a, b = ours[col], golden[col]
            a, b = pd.Series(a.to_numpy(dtype=float), index=ours.index), pd.Series(b.to_numpy(dtype=float), index=golden.index)
            a, b = a.align(b, join="inner")
            worst = float((a - b).abs().max()) if len(a) else np.nan
            rows.append({"metric": f"table {col}: largest difference", "ours": float(a.abs().mean()) if len(a) else np.nan, "theirs": float(b.abs().mean()) if len(b) else np.nan, "difference": worst, "tolerance": f"<= {tol.abs}", "ok": bool(len(a)) and np.isfinite(worst) and worst <= (tol.abs or 0.0)})
        else:
            st = ours.stats() if isinstance(ours, Result) else dict(ours)
            mine, theirs = st.get(m), (golden.get(m) if isinstance(golden, dict) else None)
            if mine is None or theirs is None:
                rows.append({"metric": m, "ours": mine, "theirs": theirs, "difference": np.nan, "tolerance": f"<= {tol.abs}", "ok": False})
                continue
            d = float(mine) - float(theirs)
            rows.append({"metric": m, "ours": float(mine), "theirs": float(theirs), "difference": d, "tolerance": f"<= {tol.abs}", "ok": abs(d) <= (tol.abs or 0.0)})
    return ParityReport(case, pd.DataFrame(rows, columns=["metric", "ours", "theirs", "difference", "tolerance", "ok"]), golden_sha256)


def run_parity(case: ParityCase, source, golden, golden_sha256: str | None = None, session=None, **build_kw) -> ParityReport:
    """Build the case's strategy, run it over the case's window (in ``session`` when given: a
    parity run is a robustness run, not a counted trial) and compare it with ``golden``."""
    from nautilus_fi.kernel.engine import run

    if case.build is None:
        raise ValueError(f"{case.name}: no strategy builder; compare a result you ran with compare()")
    strategy = case.build(**build_kw)
    start, end = case.window or (None, None)
    res = session.run(strategy, start=start, end=end, record=False) if session is not None else run(strategy, source, start, end)
    return compare(res, golden, case, golden_sha256)


def profile_confirmation(profile: str, reports: list[ParityReport], required: list[str] | None = None) -> dict:
    """Evidence that ``profile`` reproduces its framework: every report passed and every
    ``required`` case is among them. Raises ``ParityFailure`` naming what failed or is missing.
    The returned evidence (cases, golden hashes, tolerances, metrics) goes with the reviewed change
    that sets ``confirmed=True`` in ``domain/conventions.py``; this function does not set it."""
    from nautilus_fi.domain.conventions import PROFILES

    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    names = {r.case.name for r in reports}
    missing = sorted(set(required or ()) - names)
    failed = [r.case.name for r in reports if not r.passed]
    if missing or failed or not reports:
        raise ParityFailure(f"{profile}: " + "; ".join(x for x in (f"failed: {failed}" if failed else "", f"missing: {missing}" if missing else "", "no parity reports" if not reports else "") if x))
    unhashed = [r.case.name for r in reports if not r.golden_sha256]
    if unhashed:
        raise ParityFailure(f"{profile}: golden outputs without a content hash {unhashed}: load them with load_golden()")
    return {"profile": profile, "confirmed_by": "parity", "cases": [r.evidence() for r in reports], "conventions": PROFILES[profile].to_dict()}


__all__ = ["ParityCase", "ParityFailure", "ParityReport", "Tolerance", "compare", "load_golden", "profile_confirmation", "run_parity"]
