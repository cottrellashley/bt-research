"""Metrics and reporting on kernel results (section 4.10 of the gap analysis).

Each function takes a ``Result`` (and where needed the session's source or prepared data) and
returns a table or a dict:

* ``relative_drawdown``: drawdown of the active path, additive or geometric;
* ``capture``: up / down capture, Jensen alpha, beta, correlation, skew, appraisal ratio;
* ``windows``: statistics over YTD, 1Y, 3Y, 5Y, since inception, out of sample and trailing windows,
  each anchored at the prior close;
* ``period_table``: calendar-year returns with a cumulative row, and returns by earn month;
* ``timing_stats``: flips per year, time overweight / neutral / underweight, rolling Sharpe;
* ``breakdowns``: statistics by regime, by drawdown episode, and whipsaws;
* ``hit_rates``: on active returns and from trades (names bought that beat the benchmark);
* ``window_stats``: statistics per holding window between rebalances;
* ``ex_ante_risk``: TEV, contributions, marginal TEV, diversification ratio, budget use, active carry;
* ``carry_spread_split`` / ``factor_contributions``: active return by leg and by factor;
* ``attribution``: each period's return split into rates, carry, spread, FX, residual, cash,
  financing, overlays and costs, adding up exactly to the net return, by group too;
* ``trade_blotter``: every trade, mandatory (forced exits, rules) or discretionary, with its cost,
  and P&L per holding period;
* ``data_coverage``: return coverage of held weight and weight-sum checks;
* ``write_json`` / ``write_sql``: results with as-of stamps, staleness flags and warnings;
* ``signal_extras``: IC uplift over a baseline, top-minus-bottom as mean, median and winsorised
  mean, monotonicity flags, equal-issuer weighting, neutralised outcomes, control matching and
  DTS-normalised returns.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _ppy(result) -> float:
    return float(result.stats().get("periods_per_year") or 252.0)


def _years(idx: pd.DatetimeIndex) -> float:
    return max((idx[-1] - idx[0]).days / 365.0, 1e-9) if len(idx) > 1 else float("nan")


# ====================================================================================== relative performance
def relative_drawdown(result, kind: str = "geometric") -> pd.DataFrame:
    """Drawdown of the active path: ``"additive"`` on the summed active return, ``"geometric"`` on
    the ratio of the portfolio's growth to the benchmark's. Columns: path, peak, drawdown."""
    if result.benchmark_returns is None:
        raise ValueError("relative drawdown needs a benchmark")
    r, b = result.returns.iloc[1:], result.benchmark_returns.reindex(result.returns.index).iloc[1:]
    path = (r - b).cumsum() if kind == "additive" else (1 + r).cumprod() / (1 + b).cumprod()
    peak = path.cummax()
    dd = path - peak if kind == "additive" else path / peak - 1.0
    return pd.DataFrame({"path": path, "peak": peak, "drawdown": dd})


def capture(result, risk_free: pd.Series | None = None) -> dict:
    """Up and down capture (mean return in benchmark up / down periods over the benchmark's),
    Jensen alpha (annualised, against the benchmark, net of ``risk_free``), beta, correlation,
    skew, excess kurtosis and the appraisal ratio (alpha over residual volatility)."""
    r = result.returns.iloc[1:]
    b = result.benchmark_returns.reindex(r.index) if result.benchmark_returns is not None else None
    if b is None:
        raise ValueError("capture needs a benchmark")
    rf = (risk_free if risk_free is not None else result.risk_free)
    rf = rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0
    up, down = b > 0, b < 0
    out = {"up_capture": float(r[up].mean() / b[up].mean()) if up.any() else float("nan"), "down_capture": float(r[down].mean() / b[down].mean()) if down.any() else float("nan")}
    ex, eb = r - rf, b - rf
    var = float(eb.var(ddof=1))
    beta = float(ex.cov(eb) / var) if var else float("nan")
    ppy = _ppy(result)
    resid = ex - beta * eb
    alpha_p = float(resid.mean())
    out.update({"beta": beta, "jensen_alpha": alpha_p * ppy, "correlation": float(r.corr(b)), "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()), "appraisal_ratio": float(alpha_p / resid.std(ddof=1) * math.sqrt(ppy)) if resid.std(ddof=1) else float("nan")})
    return out


def _stats_window(result, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    r = result.returns
    idx = r.index
    i0 = idx.searchsorted(lo) - 1                              # anchored at the prior close
    i0 = max(i0, 0)
    seg = r.iloc[i0 + 1: idx.searchsorted(hi, side="right")]
    if seg.empty:
        return {}
    out = {"start": str(idx[i0].date()), "end": str(seg.index[-1].date()), "total_return": float((1 + seg).prod() - 1)}
    yrs = _years(pd.DatetimeIndex([idx[i0], seg.index[-1]]))
    out["annual_return"] = float((1 + out["total_return"]) ** (1 / yrs) - 1) if yrs >= 1 else out["total_return"]
    sd = float(seg.std(ddof=1)) if len(seg) > 1 else float("nan")
    ppy = _ppy(result)
    out["annual_vol"] = sd * math.sqrt(ppy) if math.isfinite(sd) else float("nan")
    out["sharpe"] = float(seg.mean() / sd * math.sqrt(ppy)) if sd else float("nan")
    if result.benchmark_returns is not None:
        bseg = result.benchmark_returns.reindex(seg.index)
        out["active_return"] = float((1 + seg).prod() - (1 + bseg).prod())
        a = seg - bseg
        te = float(a.std(ddof=1) * math.sqrt(ppy)) if len(a) > 1 else float("nan")
        out["tracking_error"] = te
        out["information_ratio"] = float(a.mean() * ppy / te) if te else float("nan")
    return out


def windows(result, presets=("YTD", "1Y", "3Y", "5Y", "SI"), oos_start=None, trailing: tuple = ()) -> pd.DataFrame:
    """Statistics per window, each starting at the prior close: ``YTD``, ``1Y`` / ``3Y`` / ``5Y``
    (ending at the last date), ``SI`` (since inception), ``OOS`` (from ``oos_start``) and trailing
    windows of ``n`` periods (``trailing=(63, 126)``)."""
    idx = result.returns.index
    last = idx[-1]
    rows = {}
    for p in presets:
        if p == "YTD":
            lo = pd.Timestamp(last.year, 1, 1)
        elif p == "SI":
            lo = idx[1] if len(idx) > 1 else idx[0]
        elif p.endswith("Y"):
            lo = last - pd.DateOffset(years=int(p[:-1]))
            if lo < idx[0]:
                continue
        else:
            raise ValueError(f"unknown window {p!r}")
        rows[p] = _stats_window(result, lo, last)
    if oos_start is not None:
        rows["OOS"] = _stats_window(result, pd.Timestamp(oos_start), last)
    for n in trailing:
        if len(idx) > n:
            rows[f"last_{n}"] = _stats_window(result, idx[-n], last)
    return pd.DataFrame(rows).T


def period_table(result, freq: str = "YE") -> pd.DataFrame:
    """Returns per calendar period (``"YE"`` years, ``"ME"`` months) for the portfolio, the
    benchmark and their difference, with a cumulative row; periods are those the returns were
    earned in (a return dated in January belongs to January)."""
    r = result.returns.iloc[1:]
    g = (1 + r).groupby(pd.Grouper(freq=freq)).prod() - 1
    out = pd.DataFrame({"portfolio": g})
    if result.benchmark_returns is not None:
        b = result.benchmark_returns.reindex(r.index)
        out["benchmark"] = (1 + b).groupby(pd.Grouper(freq=freq)).prod() - 1
        out["active"] = out["portfolio"] - out["benchmark"]
    out.index = [str(i.year) if freq.startswith("Y") else f"{i:%Y-%m}" for i in out.index]
    cum = {"portfolio": float((1 + r).prod() - 1)}
    if "benchmark" in out:
        cum["benchmark"] = float((1 + result.benchmark_returns.reindex(r.index)).prod() - 1)
        cum["active"] = cum["portfolio"] - cum["benchmark"]
    out.loc["cumulative"] = pd.Series(cum)
    return out


# ====================================================================================== timing, regimes, hit rates
def timing_stats(result, neutral: float | None = None, band: float = 0.05, window: int = 252) -> dict:
    """For timing strategies (one exposure through time): flips per year (crossings of the neutral
    exposure), the share of time overweight / neutral / underweight (outside ``band`` of neutral:
    the benchmark's exposure of 1, or ``neutral``), and the rolling ``window``-period Sharpe."""
    e = result.daily["net_exposure"].iloc[1:]
    n0 = 1.0 if neutral is None else float(neutral)
    state = np.where(e > n0 + band, 1, np.where(e < n0 - band, -1, 0))
    s = pd.Series(state, index=e.index)
    nz = s[s != 0]
    flips = int((nz != nz.shift(1)).sum() - 1) if len(nz) else 0
    yrs = _years(e.index)
    r = result.returns.iloc[1:]
    roll = r.rolling(window).mean() / r.rolling(window).std(ddof=1) * math.sqrt(_ppy(result))
    return {"flips_per_year": flips / yrs if math.isfinite(yrs) else float("nan"), "time_overweight": float((s == 1).mean()), "time_neutral": float((s == 0).mean()), "time_underweight": float((s == -1).mean()), "rolling_sharpe": roll}


def breakdowns(result, regimes: pd.Series | None = None, drawdown_threshold: float = -0.05, whipsaw_days: int = 10) -> dict:
    """Statistics by regime label (a date series), by drawdown episode (portfolio drawdowns
    deeper than ``drawdown_threshold``: start, trough, recovery, depth, length) and whipsaws
    (exposure changes reversed within ``whipsaw_days`` periods)."""
    r = result.returns.iloc[1:]
    out: dict = {}
    if regimes is not None:
        lab = regimes.reindex(r.index).ffill()
        rows = {}
        for k, seg in r.groupby(lab):
            sd = seg.std(ddof=1)
            rows[k] = {"periods": len(seg), "mean": float(seg.mean()), "annual_return": float(seg.mean() * _ppy(result)), "sharpe": float(seg.mean() / sd * math.sqrt(_ppy(result))) if sd else float("nan")}
            if result.benchmark_returns is not None:
                rows[k]["active_mean"] = float((seg - result.benchmark_returns.reindex(seg.index)).mean())
        out["by_regime"] = pd.DataFrame(rows).T
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    eps, cur = [], None
    for d, v in dd.items():
        if v < 0 and cur is None:
            cur = {"start": d, "trough": d, "depth": v}
        elif cur is not None:
            if v < cur["depth"]:
                cur["trough"], cur["depth"] = d, v
            if v >= 0:
                cur["recovery"] = d
                if cur["depth"] <= drawdown_threshold:
                    eps.append(cur)
                cur = None
    if cur is not None and cur["depth"] <= drawdown_threshold:
        cur["recovery"] = None
        eps.append(cur)
    out["drawdown_episodes"] = pd.DataFrame(eps)
    e = result.daily["net_exposure"].iloc[1:]
    ch = e.diff().fillna(0.0)
    moves = ch[ch.abs() > 1e-9]
    whips = 0
    for i in range(1, len(moves)):
        a, b = moves.iloc[i - 1], moves.iloc[i]
        if np.sign(a) != np.sign(b) and e.index.get_loc(moves.index[i]) - e.index.get_loc(moves.index[i - 1]) <= whipsaw_days:
            whips += 1
    out["whipsaws"] = whips
    return out


def hit_rates(result, source=None, horizon: int | None = None) -> dict:
    """Hit rate on active returns (share of periods the portfolio beats the benchmark) and, with
    the ``source``, from trades: of the names bought at each rebalance, the share whose return to
    the next rebalance (or ``horizon`` periods) beat the benchmark's."""
    r = result.returns.iloc[1:]
    out = {"periods_positive": float((r > 0).mean())}
    if result.benchmark_returns is not None:
        b = result.benchmark_returns.reindex(r.index)
        out["active_hit_rate"] = float((r > b).mean())
    if source is None or result.daily_weights is None or result.ids is None or not len(result.trades):
        return out
    from nautilus_fi.kernel.data import DataRequest, as_source

    dates = result.returns.index
    rf = result.strategy.returns if isinstance(result.strategy.returns, str) else "tr"
    p = as_source(source).load(DataRequest((rf,), dates[0], dates[-1], tuple(result.ids)))
    R = pd.DataFrame(p[rf], index=p.dates, columns=p.ids).reindex(index=dates, columns=result.ids).to_numpy(dtype=float)
    W = result.daily_weights
    rows = dates.get_indexer(pd.to_datetime(result.trades["trade_date"]))
    rows = rows[rows >= 0]
    hits = tot = 0
    for k, t in enumerate(rows):
        t2 = rows[k + 1] if k + 1 < len(rows) else (min(t + horizon, len(dates) - 1) if horizon else len(dates) - 1)
        prev = W[t - 1] if t > 0 else np.zeros(W.shape[1])
        bought = (W[t] > prev + 1e-5) & (W[t] > 0)
        if not bought.any() or t2 <= t:
            continue
        g = np.nanprod(1 + R[t + 1: t2 + 1], axis=0) - 1
        gb = float(np.prod(1 + result.benchmark_returns.iloc[t + 1: t2 + 1]) - 1) if result.benchmark_returns is not None else 0.0
        hits += int((g[bought] > gb).sum())
        tot += int(bought.sum())
    out["trade_hit_rate"] = hits / tot if tot else float("nan")
    return out


def window_stats(result) -> pd.DataFrame:
    """Statistics per holding window (from each rebalance to the next): return, benchmark and
    active return, turnover and cost at the window's start."""
    r = result.returns
    if not len(result.trades):
        return pd.DataFrame()
    starts = list(pd.to_datetime(result.trades["trade_date"]))
    rows = []
    for k, d in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else r.index[-1]
        seg = r[(r.index > d) & (r.index <= end)]
        if seg.empty:
            continue
        row = {"start": d, "end": end, "periods": len(seg), "return": float((1 + seg).prod() - 1), "turnover": float(result.trades.iloc[k].get("HALF_SUM", np.nan)), "cost": float(result.trades.iloc[k].get("cost", np.nan))}
        if result.benchmark_returns is not None:
            bseg = result.benchmark_returns.reindex(seg.index)
            row["benchmark"] = float((1 + bseg).prod() - 1)
            row["active"] = row["return"] - row["benchmark"]
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================================== risk and attribution
def ex_ante_risk(weights: pd.Series, benchmark: pd.Series | None, model, periods_per_year: float = 252.0, groups: pd.Series | None = None, limit: float | None = None, carry: pd.Series | None = None) -> dict:
    """Ex-ante tracking error (active, or volatility without a benchmark) from a ``RiskModel``:
    TEV, each name's (and group's) contribution to it, marginal TEV, the diversification ratio,
    budget utilisation against ``limit`` and the active carry (``carry`` per name, e.g. OAS)."""
    ids = model.loadings.index
    w = weights.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    wb = benchmark.reindex(ids).fillna(0.0).to_numpy(dtype=float) if benchmark is not None else np.zeros(len(ids))
    a = w - wb
    L = model.loadings.to_numpy(dtype=float)
    F = model.factor_cov.to_numpy(dtype=float)
    spec = model.specific_var.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    Sa = L @ (F @ (L.T @ a)) + spec * a
    var = float(a @ Sa)
    tev = math.sqrt(max(var, 0.0) * periods_per_year)
    mctr = Sa * periods_per_year / tev if tev else np.zeros_like(a)
    contrib = a * mctr
    stand = np.sqrt(np.maximum((np.einsum("ik,kl,il->i", L, F, L) + spec) * periods_per_year, 0.0))
    out = {"tev": tev, "contribution": pd.Series(contrib, index=ids), "marginal": pd.Series(mctr, index=ids), "diversification_ratio": float(np.abs(a) @ stand / tev) if tev else float("nan")}
    if groups is not None:
        out["contribution_by_group"] = out["contribution"].groupby(groups.reindex(ids)).sum()
    if limit:
        out["budget_utilisation"] = tev / float(limit)
    if carry is not None:
        out["active_carry"] = float(a @ carry.reindex(ids).fillna(0.0).to_numpy(dtype=float))
    return out


def carry_spread_split(result, source, oas: str = "oas_bp", duration: str = "oasd") -> pd.DataFrame:
    """Active return per period split into carry (active weight x OAS x dt) and spread (active
    weight x -OASD x dOAS) legs, the rest residual (rates, convexity, selection)."""
    from nautilus_fi.kernel.data import DataRequest, as_source

    if result.daily_weights is None:
        raise ValueError("needs daily weights (RunSettings(record_weights=True))")
    src = as_source(source)
    idx = result.returns.index
    p = src.load(DataRequest((oas, duration), idx[0], idx[-1], tuple(result.ids)))
    S = pd.DataFrame(p[oas], index=p.dates, columns=p.ids).reindex(index=idx, columns=result.ids).to_numpy() / 1e4
    D = pd.DataFrame(p[duration], index=p.dates, columns=p.ids).reindex(index=idx, columns=result.ids).to_numpy()
    W = result.daily_weights.astype(float)
    prev = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    wb = None
    if result.strategy.benchmark is not None and result.strategy.benchmark.has_weights:
        from nautilus_fi.kernel.engine import prepare

        prep = prepare([result.strategy], src, idx[0], idx[-1])
        wb = pd.DataFrame(result.strategy.benchmark.weights(prep.ctx), index=prep.ctx.panel.dates, columns=prep.ctx.panel.ids).reindex(index=idx, columns=result.ids).fillna(0.0).to_numpy()
    a = prev - (np.vstack([np.zeros((1, W.shape[1])), wb[:-1]]) if wb is not None else 0.0)
    dt = np.concatenate([[0.0], np.diff(idx.values).astype("timedelta64[D]").astype(float) / 365.0])
    s0 = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]])
    d0 = np.vstack([np.full((1, D.shape[1]), np.nan), D[:-1]])
    carry = np.nansum(a * s0 * dt[:, None], axis=1)
    spread = np.nansum(-a * d0 * np.vstack([np.zeros((1, S.shape[1])), np.diff(S, axis=0)]), axis=1)
    active = (result.gross - (result.benchmark_returns if result.benchmark_returns is not None else 0.0)).to_numpy()
    return pd.DataFrame({"carry": carry, "spread": spread, "residual": active - carry - spread, "active": active}, index=idx)


LEGS = ("rates", "carry", "spread", "fx", "residual", "cash", "financing", "overlay", "costs")


def attribution(result, source, by: str | None = None, rates: str | None = "tsy_r", oas: str = "oas_bp", duration: str = "oasd", fx: str | None = None, active: bool = False) -> dict:
    """Each period's return by leg: ``rates`` (the weights times each bond's duration-matched
    Treasury return), ``carry`` (OAS x dt), ``spread`` (-OASD x dOAS), ``fx`` (a per-name currency
    return field, when given), ``residual`` (selection, convexity, what the legs leave), and the
    portfolio's ``cash``, ``financing``, ``overlay`` and ``costs`` (net minus gross). The legs add up
    exactly to the net return in every period (``total``); ``groups``: the bond legs by a static
    attribute (``by``), which add up to the portfolio's (the residual to the rounding of stored
    weights); ``active``: the legs of the portfolio minus its benchmark weights, the residual
    closing to the active return. Spreads are read as known at the previous close."""
    from nautilus_fi.kernel.engine import prepare
    from nautilus_fi.kernel.expr import alive_mask

    if result.daily_weights is None:
        raise ValueError("attribution needs daily weights (RunSettings(record_weights=True))")
    s = result.strategy.resolved()
    idx = result.returns.index
    prep = prepare([s], source, idx[0], idx[-1], extra_fields=tuple(f for f in (rates, oas, duration, fx) if f))
    ctx = prep.ctx_for(s)
    rows = ctx.panel.dates.get_indexer(idx)
    cols = ctx.panel.ids.get_indexer(result.ids)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("the result's dates or instruments are not in the source")

    def field(name, realised=True):
        if not name or name not in ctx.panel:
            return None
        X = np.asarray(ctx.realised(name) if realised else ctx.panel[name], dtype=float)
        return X[np.ix_(rows, cols)]

    alive = alive_mask(ctx)[np.ix_(rows, cols)]
    R = np.where(alive, np.nan_to_num(field(ctx.return_field)), 0.0)
    W = result.daily_weights.astype(float)
    prev = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    if active:
        if s.benchmark is None or not s.benchmark.has_weights:
            raise ValueError("active attribution needs a benchmark with weights")
        Wb = np.asarray(s.benchmark.weights(ctx), dtype=float)[np.ix_(rows, cols)]
        prev = prev - np.vstack([np.zeros((1, Wb.shape[1])), Wb[:-1]])
    S = field(oas, realised=False)
    D = field(duration, realised=False)
    dt = np.concatenate([[0.0], np.diff(idx.values).astype("timedelta64[D]").astype(float) / 365.0])
    zero = np.zeros_like(R)
    legs = {"asset": prev * R}
    legs["rates"] = prev * np.where(alive, np.nan_to_num(field(rates)), 0.0) if field(rates) is not None else zero
    if S is not None and D is not None:
        s0 = np.vstack([np.full((1, S.shape[1]), np.nan), S[:-1]]) / 1e4
        d0 = np.vstack([np.full((1, D.shape[1]), np.nan), D[:-1]])
        ds = np.vstack([np.zeros((1, S.shape[1])), np.diff(S, axis=0)]) / 1e4
        legs["carry"] = np.where(alive, np.nan_to_num(prev * s0 * dt[:, None]), 0.0)
        legs["spread"] = np.where(alive, np.nan_to_num(-prev * d0 * ds), 0.0)
    else:
        legs["carry"] = legs["spread"] = zero
    legs["fx"] = prev * np.where(alive, np.nan_to_num(field(fx)), 0.0) if fx and field(fx) is not None else zero
    legs["residual"] = legs["asset"] - legs["rates"] - legs["carry"] - legs["spread"] - legs["fx"]
    total = pd.DataFrame({k: legs[k].sum(axis=1) for k in ("rates", "carry", "spread", "fx")}, index=idx)
    daily = result.daily
    if active:
        bench = result.benchmark_returns.reindex(idx).fillna(0.0) if result.benchmark_returns is not None else 0.0
        target = result.gross - bench
        total["residual"] = target - total.sum(axis=1)
        total["active"] = target
    else:
        for k, col in (("cash", "cash_return"), ("financing", "financing_return"), ("overlay", "overlay_return")):
            total[k] = daily[col].reindex(idx).fillna(0.0) if col in daily else 0.0
        bonds = result.gross - total[["cash", "financing", "overlay"]].sum(axis=1)
        total["residual"] = bonds - total[["rates", "carry", "spread", "fx"]].sum(axis=1)     # exact: the bond return the engine earned
        total["costs"] = result.returns - result.gross
        total["net"] = result.returns
    total = total[[c for c in (*LEGS, "active", "net") if c in total.columns]]
    out = {"total": total, "summary": total.iloc[1:].sum().to_frame("sum")}
    if by is not None:
        if by not in ctx.attributes.columns:
            raise KeyError(f"{by!r} is not a static attribute of the source")
        g = ctx.attributes[by].reindex(result.ids).astype(str).to_numpy()
        frames = []
        for grp in np.unique(g):
            m = g == grp
            frames.append(pd.DataFrame({"date": idx, "group": grp, **{k: legs[k][:, m].sum(axis=1) for k in ("rates", "carry", "spread", "fx", "residual")}, "total": legs["asset"][:, m].sum(axis=1)}))
        out["groups"] = pd.concat(frames, ignore_index=True)
    return out


def factor_contributions(result, source, fields=("oasd", "dts"), return_field: str | None = None) -> pd.DataFrame:
    """Factor returns estimated cross-sectionally per date (OLS of realised returns on the fields,
    Fama-MacBeth) times the portfolio's active exposures at the previous close, summed per
    holding window: what each factor contributed between rebalances."""
    from nautilus_fi.kernel.data import DataRequest, as_source

    if result.daily_weights is None:
        raise ValueError("needs daily weights")
    src = as_source(source)
    idx = result.returns.index
    rf = return_field or (result.strategy.returns if isinstance(result.strategy.returns, str) else "tr")
    p = src.load(DataRequest(tuple(dict.fromkeys((rf, *fields))), idx[0], idx[-1], tuple(result.ids)))
    get = lambda f: pd.DataFrame(p[f], index=p.dates, columns=p.ids).reindex(index=idx, columns=result.ids).to_numpy(dtype=float)
    R = get(rf)
    X = np.stack([get(f) for f in fields], axis=2)
    T = len(idx)
    fret = np.full((T, len(fields)), np.nan)
    for t in range(1, T):
        x = X[t - 1]
        y = R[t]
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if ok.sum() > len(fields) + 2:
            A = np.column_stack([np.ones(ok.sum()), x[ok]])
            coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            fret[t] = coef[1:]
    W = result.daily_weights.astype(float)
    prev = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    expo = np.einsum("tn,tnk->tk", prev, np.nan_to_num(np.vstack([np.full((1,) + X.shape[1:], np.nan), X[:-1]])))
    contrib = pd.DataFrame(np.nan_to_num(expo * fret), index=idx, columns=list(fields))
    if len(result.trades):
        starts = pd.to_datetime(result.trades["trade_date"])
        win = np.searchsorted(starts.values, idx.values, side="right")
        return contrib.groupby(win).sum().set_index(pd.Index([str(starts.iloc[k - 1].date()) if k > 0 else "before" for k in sorted(set(win))], name="window"))
    return contrib


# ====================================================================================== trades and data quality
def trade_blotter(result) -> pd.DataFrame:
    """Every trade: date, instrument, weight before and after, the trade, its side, whether it was
    mandatory (a forced exit or an aborted / rule-driven change) or discretionary, and the
    rebalance's cost shared by traded size. From daily weights."""
    if result.daily_weights is None or result.ids is None or not len(result.trades):
        return pd.DataFrame(columns=["trade_date", "instrument_id", "before", "after", "trade", "side", "mandatory", "cost"])
    W = result.daily_weights.astype(float)
    dates = result.returns.index
    rows = []
    for _, tr in result.trades.iterrows():
        t = dates.get_loc(pd.Timestamp(tr["trade_date"]))
        before = W[t - 1] if t > 0 else np.zeros(W.shape[1])
        after = W[t]
        d = after - before
        nz = np.nonzero(np.abs(d) > 1e-5)[0]                        # smaller changes are the day's drift, not trades
        tot = float(np.abs(d[nz]).sum()) or 1.0
        for i in nz:
            forced = after[i] == 0 and tr.get("n_forced", 0) > 0
            rows.append({"trade_date": dates[t], "instrument_id": result.ids[i], "before": float(before[i]), "after": float(after[i]), "trade": float(d[i]), "side": "BUY" if d[i] > 0 else "SELL", "mandatory": bool(forced), "cost": float(tr.get("cost", 0.0)) * abs(d[i]) / tot})
    return pd.DataFrame(rows)


def data_coverage(result) -> pd.DataFrame:
    """Per date: the share of held weight with a finite realised return (coverage), the sum of
    weights plus cash (should be 1), and the number of names."""
    d = result.daily
    out = pd.DataFrame({"weight_plus_cash": d["net_exposure"] + d["cash_weight"], "names": d["n_names"]})
    fills = sum(int(result.missing.get(k, 0) or 0) for k in ("zero_filled", "refilled", "unexplained_exits"))
    out["return_coverage"] = 1.0 - (fills / max(int(d["n_names"].sum()), 1))
    return out


# ====================================================================================== writers
def _payload(result, as_of=None, stale_after_days: int | None = None) -> dict:
    s = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in result.stats().items() if isinstance(v, (int, float, str, bool)) or v is None}
    last = result.returns.index[-1]
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(datetime.now(UTC).date())
    stale = bool(stale_after_days is not None and (as_of - last).days > stale_after_days)
    warnings = [n for n in result.diagnostics.get("notes", [])]
    if stale:
        warnings.append(f"data end {last.date()} is more than {stale_after_days} days before {as_of.date()}")
    if s.get("data_fills"):
        warnings.append(f"{s['data_fills']} filled or altered data points")
    return {"name": result.name, "fingerprint": result.fingerprint, "as_of": str(as_of.date()), "data_end": str(last.date()), "stale": stale, "written_utc": datetime.now(UTC).isoformat(), "source_version": result.source_version, "conventions": result.conventions.to_dict(), "stats": s, "warnings": warnings}


def write_json(result, path, as_of=None, stale_after_days: int | None = 3, series: bool = True) -> Path:
    """The result's statistics and (``series``) returns as JSON, stamped with the as-of date, the
    data's end, a staleness flag and the run's warnings."""
    out = _payload(result, as_of, stale_after_days)
    if series:
        out["returns"] = {str(k.date()): float(v) for k, v in result.returns.items()}
        if result.benchmark_returns is not None:
            out["benchmark_returns"] = {str(k.date()): float(v) for k, v in result.benchmark_returns.items()}
    p = Path(path)
    from nautilus_fi.core.fileio import atomic_write_text

    atomic_write_text(p, json.dumps(out, indent=2, default=str))
    return p


def write_sql(result, connection, table: str = "BACKTEST_RESULTS", as_of=None, stale_after_days: int | None = 3, if_exists: str = "append") -> int:
    """One row per result (statistics as columns, as-of stamps, staleness and warnings) into
    ``table`` on a DB-API / SQLAlchemy ``connection`` (Snowflake through its connector's pandas
    support, SQLite in tests). Returns the rows written."""
    pl = _payload(result, as_of, stale_after_days)
    row = {"name": pl["name"], "fingerprint": pl["fingerprint"], "as_of": pl["as_of"], "data_end": pl["data_end"], "stale": int(pl["stale"]), "written_utc": pl["written_utc"], "source_version": pl["source_version"], "warnings": "; ".join(pl["warnings"]), **{f"stat_{k}": v for k, v in pl["stats"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}
    pd.DataFrame([row]).to_sql(table, connection, if_exists=if_exists, index=False)
    return 1


# ====================================================================================== signal evaluation extras
def signal_extras(scores: pd.DataFrame, fwd: pd.DataFrame, baseline: pd.DataFrame | None = None, n_quantiles: int = 5, issuers: pd.Series | None = None, groups: pd.Series | None = None, dts: pd.DataFrame | None = None, control: pd.DataFrame | None = None, tolerance: dict | None = None, winsor: float = 0.05) -> dict:
    """Extra signal diagnostics on dates x ids frames of scores and forward returns:

    * IC per date and its uplift over a ``baseline`` signal;
    * top-minus-bottom quantile spread as mean, median and winsorised mean, quantile means and
      monotonicity flags (Spearman of the quantile means; strictly increasing);
    * ``issuers``: names averaged within issuer first (equal-issuer weighting);
    * ``groups``: forward returns demeaned within group (neutralised outcomes, not just a
      neutralised signal);
    * ``dts``: forward returns divided by DTS (risk-normalised);
    * ``control``: for each top-quantile name, the nearest bottom-or-middle name on the
      ``control`` fields within ``tolerance`` (e.g. ``{"oas_bp": 15, "oasd": 0.5}``); the matched
      return difference."""
    S, F = scores.align(fwd, join="inner")
    if dts is not None:
        F = F / dts.reindex_like(F)
    if groups is not None:
        g = groups.reindex(F.columns)
        F = F.T.groupby(g).transform(lambda x: x - x.mean()).T
    S_name, F_name = S, F                                          # control matching is per name
    if issuers is not None:
        iss = issuers.reindex(S.columns)
        S = S.T.groupby(iss).mean().T
        F = F.T.groupby(iss).mean().T
    ic = S.rank(axis=1).corrwith(F.rank(axis=1), axis=1)
    out: dict = {"ic": ic, "mean_ic": float(ic.mean())}
    if baseline is not None:
        B, F2 = baseline.align(fwd, join="inner")
        ic_b = B.rank(axis=1).corrwith(F2.rank(axis=1), axis=1)
        out["ic_uplift"] = float((ic - ic_b.reindex(ic.index)).mean())
    q = S.rank(axis=1, pct=True).apply(lambda r: np.ceil(r * n_quantiles).clip(1, n_quantiles), axis=1)
    qm = pd.DataFrame({k: F.where(q == k).mean(axis=1) for k in range(1, n_quantiles + 1)})
    spread = qm[n_quantiles] - qm[1]
    lo, hi = spread.quantile(winsor), spread.quantile(1 - winsor)
    means = qm.mean()
    rho = float(pd.Series(means.to_numpy()).corr(pd.Series(np.arange(len(means))), method="spearman"))
    out.update({"quantile_means": means, "spread_mean": float(spread.mean()), "spread_median": float(spread.median()), "spread_winsorised_mean": float(spread.clip(lo, hi).mean()), "monotonicity": rho, "monotonic_increasing": bool(np.all(np.diff(means.to_numpy()) > 0))})
    if control is not None and tolerance:
        diffs = []
        S, F = S_name, F_name
        q = S.rank(axis=1, pct=True).apply(lambda r: np.ceil(r * n_quantiles).clip(1, n_quantiles), axis=1)
        for d in S.index:
            top = q.loc[d] == n_quantiles
            pool = q.loc[d] < n_quantiles
            for i in top[top].index:
                cand = [j for j in pool[pool].index if all(abs(float(control.at[i, f]) - float(control.at[j, f])) <= tol for f, tol in tolerance.items() if f in control.columns)]
                if cand and np.isfinite(F.at[d, i]):
                    j = min(cand, key=lambda j: sum(abs(float(control.at[i, f]) - float(control.at[j, f])) / max(tol, 1e-12) for f, tol in tolerance.items()))
                    if np.isfinite(F.at[d, j]):
                        diffs.append(float(F.at[d, i] - F.at[d, j]))
        out["control_matched_spread"] = float(np.mean(diffs)) if diffs else float("nan")
        out["control_matches"] = len(diffs)
    return out


__all__ = [
    "LEGS",
    "attribution",
    "breakdowns",
    "capture",
    "carry_spread_split",
    "data_coverage",
    "ex_ante_risk",
    "factor_contributions",
    "hit_rates",
    "period_table",
    "relative_drawdown",
    "signal_extras",
    "timing_stats",
    "trade_blotter",
    "window_stats",
    "windows",
    "write_json",
    "write_sql",
]
