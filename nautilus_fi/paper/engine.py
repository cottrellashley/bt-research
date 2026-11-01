"""The panel engine (Paper): vectorised portfolio simulation on return panels.

One loop over trading days, numpy over the cross-section. On each day:

1. the weights held at the previous close earn the period's returns (assets, cash, financing on
   leverage, borrow on shorts, overlays outside the capital budget); a held asset that stops existing
   is redeemed into cash, a missing return on a held asset follows the conventions' policy;
2. on decision dates (the rebalance calendar) the decider sees the point-in-time ``DecisionView``
   and the construction spec turns its target into weights;
3. trades decided ``timing.trade_offset_days`` earlier execute at today's close, paying costs
   in bp of traded value (from a cost field or a default), and become the weights that earn the
   next period.

The same ``ResearchSpec`` runs in the event engine (``backtest.spec_strategy``); results reconcile
through ``paper.reconcile``. Terms are never needed: assets are whatever has a return series.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.benchmarks.benchmark import Benchmark, BenchmarkWeightsUnavailable
from nautilus_fi.construct.cross_section import CrossSection
from nautilus_fi.data.panel import Panel
from nautilus_fi.domain.calendars import parse_calendar
from nautilus_fi.domain.conventions import (
    Conventions,
    MissingReturnPolicy,
    MissingReturnsError,
    detect_bad_dates,
    turnover_all,
)
from nautilus_fi.domain.spec import DecisionView, ResearchSpec, as_target


@dataclass
class PaperData:
    panel: Panel
    attributes: pd.DataFrame | None = None
    cash: pd.Series | None = None                   # cash return per period on the panel dates
    benchmark: Benchmark | None = None
    overlays: Panel | None = None                   # overlay instruments: return per unit notional + exposures
    overlay_financed: tuple[str, ...] = ()          # overlay returns that are total returns of financed positions
    return_field: str = "tr"
    excess_field: str | None = "xr"
    treasury_field: str | None = "tsy_r"
    cost_field: str | None = "half_spread_bp"
    alive_field: str | None = "alive"
    universe_field: str | None = None               # eligible = field > 0 at the decision date
    refill_field: str | None = None                 # source for MissingReturnPolicy.REFILL
    overlay_cost_bp: float = 0.5


@dataclass
class PaperSettings:
    conventions: Conventions | None = None          # None: the spec's conventions
    calendar: object = None                         # None: the spec's calendar
    start: object = None
    end: object = None
    first_decision: bool = True
    initial_nav: float = 100e6
    cost_multiplier: float = 1.0
    default_cost_bp: float = 0.0
    financing_spread_bp: float = 0.0                # annual, paid on negative cash (leverage)
    borrow_fee_bp: float = 30.0                     # annual, on short positions
    min_face: float | None = None                   # minimum holding face: becomes min_weight in the cross-section
    price_field: str = "price_clean"
    fields: list[str] | None = None                 # fields given to the cross-section (None: all panel fields)
    exposure_fields: tuple[str, ...] = ("oad", "oasd", "dts", "oas_bp")
    record_daily_weights: bool = False
    universe_from_benchmark: bool = True
    name: str | None = None


class _PanelView(DecisionView):
    def __init__(self, panel: Panel, i: int, cand: np.ndarray, universe: pd.Index, holdings: pd.Series, bench: pd.Series, attributes: pd.DataFrame, nav: float) -> None:
        self._p, self._i, self._cand = panel, i, cand
        self._universe, self._holdings, self._bench, self._attrs = universe, holdings, bench, attributes
        self.date = panel.dates[i]
        self.nav = nav
        self._ids = panel.ids[cand]

    @property
    def universe(self) -> pd.Index:
        return self._universe

    @property
    def holdings(self) -> pd.Series:
        return self._holdings

    @property
    def benchmark(self) -> pd.Series:
        return self._bench

    @property
    def attributes(self) -> pd.DataFrame:
        return self._attrs

    @property
    def ids(self) -> pd.Index:
        return self._ids

    def has(self, name: str) -> bool:
        return name in self._p

    def field(self, name: str) -> pd.Series:
        return pd.Series(self._p.asof_row(name, self._i)[self._cand], index=self._ids, name=name)

    def history(self, name: str, n: int) -> pd.DataFrame:
        return self._p.history(name, self._i, n, self._cand)


@dataclass
class PaperResult:
    name: str
    spec_hash: str
    conventions: Conventions
    returns: pd.Series                              # net of costs
    gross: pd.Series
    costs: pd.Series
    nav: pd.Series
    daily: pd.DataFrame                             # cash weight, gross / net exposure, names, excess components
    trades: pd.DataFrame                            # one row per executed rebalance: turnover under every definition, cost
    weights: pd.DataFrame                           # long: date, instrument_id, weight (after each rebalance; daily if recorded)
    overlays: pd.DataFrame
    construction: pd.DataFrame
    violations: pd.DataFrame
    exposures: pd.DataFrame                         # portfolio (and active) exposures per day
    benchmark_returns: pd.Series | None
    missing: dict
    spec: dict
    timings: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def active(self) -> pd.Series | None:
        return None if self.benchmark_returns is None else (self.returns - self.benchmark_returns).rename("active")

    def excess(self, freq: str | None = None) -> pd.Series:
        """Portfolio excess return over duration-matched Treasuries under the conventions (vendor
        excess returns when available and preferred)."""
        from nautilus_fi.domain.conventions import excess_return_series

        d = self.daily
        vendor = d["excess_vendor"] if "excess_vendor" in d and d["excess_vendor"].notna().any() else None
        tsy = d["treasury_leg"] if "treasury_leg" in d and d["treasury_leg"].notna().any() else None
        return excess_return_series(self.gross, tsy, self.conventions, freq, vendor)

    def stats(self) -> dict:
        from nautilus_fi.domain.conventions import performance_stats

        r = self.returns.iloc[1:]
        out = {"name": self.name, "spec_hash": self.spec_hash, **performance_stats(r, self.conventions, benchmark=None if self.benchmark_returns is None else self.benchmark_returns.reindex(r.index))}
        g = performance_stats(self.gross.iloc[1:], self.conventions)
        out["gross_annual_return"] = g.get("annual_return")
        years = out.get("years") or float("nan")
        if not self.trades.empty:
            ongoing = self.trades[~self.trades["initial_build"].fillna(False).astype(bool)] if "initial_build" in self.trades else self.trades
            for k in ("ONE_WAY_BUYS", "HALF_SUM", "TWO_WAY"):                    # the build from cash is not turnover
                out[f"turnover_annual_{k.lower()}"] = float(ongoing[k].sum() / years) if years else float("nan")
            out["turnover_annual"] = out[f"turnover_annual_{self.conventions.turnover.value.lower()}"]
            out["cost_drag_annual"] = float(self.costs.sum() / years) if years else float("nan")
            out["n_rebalances"] = len(self.trades)
        out["avg_names"] = float(self.daily["n_names"].mean())
        out["avg_cash_weight"] = float(self.daily["cash_weight"].mean())
        out["max_gross_exposure"] = float(self.daily["gross_exposure"].max())
        out["hard_violations"] = int((self.violations["severity"] == "HARD").sum()) if not self.violations.empty else 0
        d = self.daily
        vendor = d["excess_vendor"].iloc[1:] if "excess_vendor" in d and d["excess_vendor"].notna().any() else None
        tsy = d["treasury_leg"].iloc[1:] if "treasury_leg" in d and d["treasury_leg"].notna().any() else None
        if vendor is not None or tsy is not None:
            from nautilus_fi.domain.conventions import annualised_excess

            try:
                out["excess_annual_return"] = annualised_excess(self.gross.iloc[1:], tsy, self.conventions.excess_return, years, vendor, self.conventions.prefer_vendor_excess)
            except ValueError:
                pass
        out["missing_returns"] = self.missing
        return out

    def summary(self) -> pd.Series:
        return pd.Series({k: v for k, v in self.stats().items() if not isinstance(v, dict)})


class PaperEngine:
    def __init__(self, spec: ResearchSpec, data: PaperData, settings: PaperSettings | None = None) -> None:
        self.spec = spec
        self.data = data
        self.settings = settings or PaperSettings()

    def run(self, ledger=None, stage: str = "PAPER", uses_holdout: bool = False) -> PaperResult:
        t0 = time.perf_counter()
        spec, data, st = self.spec, self.data, self.settings
        conv = st.conventions or spec.conventions
        cal = parse_calendar(st.calendar) if st.calendar is not None else spec.rebalance
        panel = data.panel                          # the full panel: decisions may look back before ``start``
        dates, ids = panel.dates, panel.ids
        T, N = len(dates), len(ids)
        t_lo = 0 if st.start is None else max(int(dates.searchsorted(pd.Timestamp(st.start))), 0)
        t_hi = T - 1 if st.end is None else min(int(dates.searchsorted(pd.Timestamp(st.end), side="right")) - 1, T - 1)
        if t_hi - t_lo < 1:
            raise ValueError("the run window needs at least two panel dates")
        rf = data.return_field
        R = panel[rf].astype(np.float64)
        if data.alive_field and data.alive_field in panel:
            alive = panel[data.alive_field] > 0
        else:                                       # no existence field: an asset exists from its first to its last finite return
            fin_r = np.isfinite(R)
            first = np.where(fin_r.any(axis=0), fin_r.argmax(axis=0), T)
            last = np.where(fin_r.any(axis=0), T - 1 - fin_r[::-1].argmax(axis=0), -1)
            tt = np.arange(T)[:, None]
            alive = (tt >= first[None, :] - 1) & (tt <= last[None, :])
        cash_raw = data.cash.reindex(dates).to_numpy(dtype=np.float64) if data.cash is not None else np.zeros(T)
        xr = panel[data.excess_field].astype(np.float64) if data.excess_field and data.excess_field in panel else None
        tsy = panel[data.treasury_field].astype(np.float64) if data.treasury_field and data.treasury_field in panel else None
        refill = panel[data.refill_field].astype(np.float64) if data.refill_field and data.refill_field in panel else None
        attrs = (data.attributes.reindex(ids) if data.attributes is not None else pd.DataFrame(index=ids))
        bench_w = None
        bench_ret = None
        if data.benchmark is not None:
            bench_ret = data.benchmark.returns(panel, rf) if hasattr(data.benchmark, "returns") else None
            try:
                bench_w = data.benchmark.weight_matrix(panel, rf)
            except BenchmarkWeightsUnavailable:
                bench_w = None
        ov = data.overlays if data.overlays is not None else None
        if ov is not None and not ov.dates.equals(dates):
            raise ValueError("overlay panel dates must match the asset panel dates")
        K = len(ov.ids) if ov is not None else 0
        ovR = np.nan_to_num(ov[rf].astype(np.float64)) if ov is not None else np.zeros((T, 0))
        fin = np.array([k in data.overlay_financed for k in (ov.ids if ov is not None else [])], dtype=bool)

        # missing-return policy: bad dates first (DROP_DATE removes them, REFILL replaces them)
        missing = {"policy": conv.missing_returns.value, "bad_dates": [], "zero_filled": 0, "refilled": 0, "redeemed": 0}
        if conv.missing_returns in (MissingReturnPolicy.DROP_DATE, MissingReturnPolicy.REFILL):
            win = slice(t_lo + 1, t_hi + 1)                  # the first date has no return period
            bad = detect_bad_dates(pd.DataFrame(R[win], index=dates[win], columns=ids), conv.bad_date_threshold, pd.DataFrame(alive[win], index=dates[win], columns=ids), conv.bad_date_zeros_only)
            missing["bad_dates"] = [str(pd.Timestamp(b).date()) for b in bad]
            if bad and conv.missing_returns == MissingReturnPolicy.REFILL:
                if refill is None:
                    raise MissingReturnsError(f"{len(bad)} bad dates under REFILL but no refill_field")
                for b in bad:
                    i = dates.get_loc(b)
                    R[i] = np.where(np.isfinite(refill[i]), refill[i], R[i])
                    missing["refilled"] += int(np.isfinite(refill[i]).sum())
            if bad and conv.missing_returns == MissingReturnPolicy.DROP_DATE:
                drop = {dates.get_loc(b) for b in bad}
                R = R.copy()
                for i in drop:
                    R[i] = np.where(alive[i], 0.0, R[i])
                missing["dropped"] = len(drop)
        refilled = None
        if conv.missing_returns == MissingReturnPolicy.REFILL and refill is not None:     # every gap a refill covers, held or not
            fill = alive & ~np.isfinite(R) & np.isfinite(refill)
            fill[: t_lo + 1] = False
            fill[t_hi + 1:] = False
            R = np.where(fill, refill, R)
            refilled = fill
        if bench_w is not None and (refilled is not None or missing.get("dropped")):     # the benchmark earns what the portfolio earns
            br = np.zeros(T)
            br[1:] = (bench_w[:-1] * np.nan_to_num(R[1:])).sum(axis=1)
            bench_ret = pd.Series(br, index=dates)
        rows = np.arange(t_lo, t_hi + 1)                                     # the run's periods: DROP_DATE removes bad dates
        if missing.get("dropped"):
            rows = np.setdiff1d(rows, np.asarray(sorted(dates.get_loc(b) for b in pd.to_datetime(missing["bad_dates"])), dtype=np.int64))
        from nautilus_fi.kernel.engine import _period_series

        cash, cash_gap = _period_series(pd.DataFrame({"cash": cash_raw}, index=dates), "cash", dates, rows)
        dt = np.zeros(T)
        dt[rows[1:]] = (dates.values[rows[1:]] - dates.values[rows[:-1]]).astype("timedelta64[D]").astype(float) / 365.0
        if bench_w is not None:
            tb = rows[1:]
            gb = (bench_w[tb - 1] > 0) & alive[tb] & ~np.isfinite(R[tb])
            if gb.any():
                if conv.missing_returns in (MissingReturnPolicy.FAIL, MissingReturnPolicy.REFILL):
                    raise MissingReturnsError(f"benchmark: {int(gb.sum())} missing member returns (first on {dates[tb[np.argmax(gb.any(axis=1))]].date()}) under {conv.missing_returns.value}; choose ZERO or DROP_DATE explicitly")
                missing["benchmark_zero_filled"] = int(gb.sum())

        trading_days = list(dates.date[rows])
        decisions = set(dates.get_indexer(pd.to_datetime(cal.dates(trading_days, st.first_decision))))
        lag = conv.timing.trade_offset_days
        pending: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        w = np.zeros(N)
        o = np.zeros(K)
        nav = st.initial_nav
        gross, net, costs = np.zeros(T), np.zeros(T), np.zeros(T)
        navs = np.zeros(T)
        cash_w = np.zeros(T)
        gexp, nexp, names = np.zeros(T), np.zeros(T), np.zeros(T, dtype=int)
        ex_vendor = np.full(T, np.nan) if xr is not None else None
        tsy_leg = np.full(T, np.nan) if tsy is not None else None
        if ex_vendor is not None:
            ex_vendor[t_lo] = 0.0
        if tsy_leg is not None:
            tsy_leg[t_lo] = 0.0
        exp_fields = [f for f in st.exposure_fields if f in panel]
        exposures = {f: np.zeros(T) for f in exp_fields}
        act_exposures = {f: np.zeros(T) for f in exp_fields} if bench_w is not None else {}
        trades_rows, weight_rows, ov_rows, cons_rows, viol_rows = [], [], [], [], []
        last_decision = None
        fin_spread = st.financing_spread_bp / 1e4
        borrow = st.borrow_fee_bp / 1e4
        cost_m = panel[data.cost_field] if data.cost_field and data.cost_field in panel else None
        fields_for_cs = [f for f in (st.fields or panel.fields) if f in panel]
        n_rows = len(rows)
        for j, t in enumerate(rows):
            t = int(t)
            # 1) earn the period
            if j > 0:
                rt = R[t]
                held = w != 0
                dead = held & ~alive[t]
                if dead.any():
                    missing["redeemed"] += int(dead.sum())
                    w = np.where(dead, 0.0, w)
                    held = w != 0
                gaps = held & ~np.isfinite(rt)
                if refilled is not None:
                    missing["refilled"] += int((refilled[t] & held).sum())
                if gaps.any():
                    if conv.missing_returns in (MissingReturnPolicy.ZERO, MissingReturnPolicy.DROP_DATE):
                        missing["zero_filled"] += int(gaps.sum())
                    else:
                        bad_ids = list(ids[gaps][:5])
                        raise MissingReturnsError(f"{int(gaps.sum())} missing returns on held assets on {dates[t].date()} ({bad_ids}) under {conv.missing_returns.value}; choose ZERO, DROP_DATE or REFILL explicitly")
                rc = np.where(held, np.nan_to_num(rt), 0.0)
                asset = float(w @ rc)
                cw = 1.0 - float(w.sum())
                if cash_gap[t] and abs(cw) > 1e-9 and data.cash is not None:
                    if conv.missing_returns == MissingReturnPolicy.FAIL:
                        raise MissingReturnsError(f"cash has no value for the period ending {dates[t].date()} while the portfolio holds {cw:.2%} in cash, under FAIL")
                    missing["cash_zero_filled"] = missing.get("cash_zero_filled", 0) + 1
                cash_part = cw * cash[t] if cw >= 0 else cw * (cash[t] + fin_spread * dt[t])
                short = float(-w[w < 0].sum()) * borrow * dt[t]
                ovr = ovR[t] - np.where(fin, cash[t], 0.0) if K else np.zeros(0)
                ov_part = float(o @ ovr) if K else 0.0
                g = asset + cash_part - short + ov_part
                gross[t] = g
                if ex_vendor is not None:
                    ex_vendor[t] = float(w @ np.where(held, np.nan_to_num(xr[t]), 0.0)) + (ov_part if K else 0.0)
                if tsy_leg is not None:
                    tsy_leg[t] = float(w @ np.where(held, np.nan_to_num(tsy[t]), 0.0)) + cw * cash[t]
                w = w * (1.0 + rc) / (1.0 + g)
                if K:
                    o = o * (1.0 + ovR[t]) / (1.0 + g)
            # 2) decide at today's close
            if t in decisions:
                tgt = self._decide(panel, t, w, o, alive, bench_w, attrs, nav, fields_for_cs, cost_m, cal, last_decision, st, data, ov, cons_rows, viol_rows)
                if tgt is not None:
                    if j + lag < n_rows:
                        pending[j + lag] = (tgt[0], tgt[1], t)
                    last_decision = t
            # 3) execute trades due at today's close
            c = 0.0
            if j in pending:
                tw, to, dec_t = pending.pop(j)
                tw = np.where(alive[t] | (tw == 0), tw, 0.0)                 # cannot buy what no longer exists
                dw = tw - w
                cbp = np.nan_to_num(cost_m[t].astype(np.float64), nan=st.default_cost_bp) if cost_m is not None else np.full(N, st.default_cost_bp)
                c = float(np.abs(dw) @ (cbp / 1e4)) * st.cost_multiplier
                if K:
                    c += float(np.abs(to - o).sum()) * data.overlay_cost_bp / 1e4 * st.cost_multiplier
                tov = turnover_all(pd.Series(w, index=ids)[w != 0], pd.Series(tw, index=ids)[tw != 0])
                trades_rows.append({"decision_date": dates[dec_t], "trade_date": dates[t], **tov, "cost": c, "n_buys": int((dw > 1e-12).sum()), "n_sells": int((dw < -1e-12).sum()), "initial_build": bool(not np.any(w))})
                w, o = tw, (to if K else o)
                nz = np.nonzero(w)[0]
                weight_rows.append(pd.DataFrame({"date": dates[t], "instrument_id": ids[nz], "weight": w[nz]}))
                if K and np.any(o):
                    ov_rows.append(pd.DataFrame({"date": dates[t], "instrument_id": ov.ids, "notional": o}))
            elif st.record_daily_weights:
                nz = np.nonzero(w)[0]
                weight_rows.append(pd.DataFrame({"date": dates[t], "instrument_id": ids[nz], "weight": w[nz]}))
            costs[t] = c
            net[t] = (1.0 + gross[t]) * (1.0 - c) - 1.0
            nav *= 1.0 + net[t]
            navs[t] = nav
            cash_w[t] = 1.0 - w.sum()
            gexp[t], nexp[t], names[t] = float(np.abs(w).sum()), float(w.sum()), int((w != 0).sum())
            for f in exp_fields:
                x = np.nan_to_num(panel[f][t].astype(np.float64))
                exposures[f][t] = float(w @ x)
                if act_exposures:
                    act_exposures[f][t] = exposures[f][t] - float(bench_w[t] @ x)
        win = rows
        idx = dates[win]
        gross, net, costs, navs, cash_w, gexp, nexp, names = (a[win] for a in (gross, net, costs, navs, cash_w, gexp, nexp, names))
        ex_vendor = ex_vendor[win] if ex_vendor is not None else None
        tsy_leg = tsy_leg[win] if tsy_leg is not None else None
        exposures = {f: v[win] for f, v in exposures.items()}
        act_exposures = {f: v[win] for f, v in act_exposures.items()}
        daily = pd.DataFrame({"cash_weight": cash_w, "gross_exposure": gexp, "net_exposure": nexp, "n_names": names}, index=idx)
        if ex_vendor is not None:
            daily["excess_vendor"] = ex_vendor
        if tsy_leg is not None:
            daily["treasury_leg"] = tsy_leg
        expo = pd.DataFrame({**{f: v for f, v in exposures.items()}, **{f"active_{f}": v for f, v in act_exposures.items()}}, index=idx)
        name = st.name or spec.name
        res = PaperResult(name, spec.hash, conv, pd.Series(net, index=idx, name="net"), pd.Series(gross, index=idx, name="gross"), pd.Series(costs, index=idx, name="cost"), pd.Series(navs, index=idx, name="nav"), daily, pd.DataFrame(trades_rows), pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(columns=["date", "instrument_id", "weight"]), pd.concat(ov_rows, ignore_index=True) if ov_rows else pd.DataFrame(), pd.DataFrame(cons_rows), pd.DataFrame(viol_rows), expo, None if bench_ret is None else bench_ret.reindex(idx), missing, spec.to_dict())
        res.timings["run_seconds"] = time.perf_counter() - t0
        if ledger is not None:
            ledger.record_result(res, spec, stage=stage, uses_holdout=uses_holdout)
        return res

    # ------------------------------------------------------------------ decisions
    def _decide(self, panel: Panel, t: int, w, o, alive, bench_w, attrs, nav, fields_for_cs, cost_m, cal, last_decision, st: PaperSettings, data: PaperData, ov, cons_rows, viol_rows):
        spec = self.spec
        ids = panel.ids
        if data.universe_field and data.universe_field in panel:
            elig = np.nan_to_num(panel.asof_row(data.universe_field, t)) > 0
        elif bench_w is not None and st.universe_from_benchmark:
            elig = bench_w[t] > 0
        else:
            elig = alive[t].copy()
        elig &= alive[t]
        held = w != 0
        bmask = bench_w[t] > 0 if bench_w is not None else np.zeros(len(ids), dtype=bool)
        cand = np.nonzero(elig | held | bmask)[0]
        holdings = pd.Series(w[held], index=ids[held])
        bench = pd.Series(bench_w[t][bmask], index=ids[bmask]) if bench_w is not None else pd.Series(dtype=float)
        universe = ids[np.nonzero(elig)[0]]
        view = _PanelView(panel, t, cand, universe, holdings, bench, attrs.iloc[cand], nav)
        if spec.universe is not None:
            universe = spec.universe.apply(view, universe)
            view._universe = universe
        target = as_target(spec.decider.decide(view), spec.construction is not None)
        if target.skip:
            cons_rows.append({"date": panel.dates[t], "action": "skip", "note": target.note})
            return None
        date = panel.dates[t]
        overlay_target = np.asarray(o, dtype=float).copy()
        if target.weights is not None:
            wt = pd.Series(target.weights, dtype=float)
            diag = {"method": "Target.weights"}
            viol = []
        else:
            data_cs = view.fields(fields_for_cs).join(view.attributes, how="left", rsuffix="_attr")
            price = view.field(st.price_field) if view.has(st.price_field) else None
            cost = view.field(data.cost_field) if (data.cost_field and view.has(data.cost_field)) else None
            cs = CrossSection.build(date, universe, data_cs, holdings, bench, target.scores, cost, st.min_face, price, nav)
            ov_data = ov.cross_section(t, ov.fields) if ov is not None else None
            out = spec.construction.build(cs, ov_data)
            wt = out.weights
            diag = out.diagnostics
            viol = out.violations
            if out.overlays is not None and ov is not None:
                overlay_target = out.overlays.reindex(ov.ids).fillna(0.0).to_numpy()
        if target.overlays is not None and ov is not None:
            overlay_target = pd.Series(target.overlays).reindex(ov.ids).fillna(0.0).to_numpy()
        if cal.triggered and last_decision is not None:
            from nautilus_fi.domain.conventions import TurnoverDefinition, turnover

            if cal.measure == "signal" and cal.field:
                now = view.field(cal.field).rank(pct=True)
                prev = pd.Series(panel.asof_row(cal.field, last_decision), index=ids).reindex(now.index).rank(pct=True)
                change = float((now - prev).abs().mean())
            else:
                change = turnover(holdings, wt, TurnoverDefinition.HALF_SUM)
            if change < cal.threshold:
                cons_rows.append({"date": date, "action": "trigger_skip", "measure": cal.measure, "value": change, "threshold": cal.threshold})
                return None
        tw = np.zeros(len(ids))
        pos = panel.id_pos(wt.index.astype(str))
        ok = pos >= 0
        tw[pos[ok]] = wt.to_numpy()[ok]
        cons_rows.append({"date": date, "action": "rebalance", **{k: v for k, v in diag.items() if not isinstance(v, (dict, list))}, **({f"opt_{k}": v for k, v in diag["optimiser"].items()} if isinstance(diag.get("optimiser"), dict) else {})})
        for v in viol:
            viol_rows.append({"date": date, **v})
        return tw, overlay_target


__all__ = ["PaperData", "PaperEngine", "PaperResult", "PaperSettings"]
