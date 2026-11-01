"""Return construction and return models (section 4 of the gap analysis).

Everything here turns vendor data into per-period simple returns the engine can earn, or models a
return the data does not publish:

* ``total_return_from_prices``: bond total return from clean prices, accrued and factors, with
  coupons inferred from accrued drops (or given) and defaulted accrued written off as a loss;
* ``decumulate_mtd``: Bloomberg month-to-date series to daily total, rate and excess returns that
  satisfy ``tr = ir + xr`` on every date and compound back to the published MTD at any month end;
* ``align_legs``: inner (warn), strict (raise) or outer (carry zero) alignment of return legs;
* ``resample_returns`` / ``common_dates``: compounding to weekly or monthly periods, and the
  intersection of dates across sleeves;
* ``forward_returns``: bond-level forward TR / IR / XR per horizon with exit, missing-interval and
  index-membership flags (signal evaluation);
* ``spread_return`` / ``SpreadReturn``: the OAS carry-and-spread proxy (also as a kernel
  expression, so ``Strategy(returns=SpreadReturn())`` earns it);
* ``duration_convexity_return``: carry - D x ds + C x ds^2 / 2;
* ``constant_maturity_returns`` / ``carry_rolldown``: government constant-maturity returns from a
  yield curve, with carry and 1m / 3m roll-down;
* ``fx_hedged_returns`` / ``hedge_carry``: currency-hedged returns from spot and forward rates;
* ``cds_returns``: CDS / CDX returns from spreads and risky duration, with an IMM roll adjustment and
  the quarterly premium schedule;
* ``continuous_futures`` / ``empirical_dv01``: rolled, back-adjusted futures series and a rolling
  regression DV01;
* ``pair_spread_pnl``: pairs in spread units with a static or rolling lagged OLS hedge ratio.

Functions take and return pandas objects indexed by date (and instrument columns where relevant);
each says its units. Returns are simple fractions per period unless stated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nautilus_fi.kernel.expr import Expr, Field


# ====================================================================================== bonds from prices
def total_return_from_prices(clean: pd.DataFrame, accrued: pd.DataFrame, factor: pd.DataFrame | None = None, coupons: pd.DataFrame | None = None, coupon_dates: pd.DataFrame | None = None, defaulted: pd.DataFrame | None = None, min_drop: float = 0.05) -> pd.DataFrame:
    """Bond total return per period from prices per 100 of current face:

    ``((clean_t + accrued_t) * f_t + coupon_t * f_{t-1} + (f_{t-1} - f_t) * 100) / ((clean_{t-1} + accrued_{t-1}) * f_{t-1}) - 1``

    ``coupons`` (per 100 face) when known; otherwise a coupon is inferred where accrued drops by
    more than ``min_drop`` on a date allowed by ``coupon_dates`` (all dates when None): the accrued
    before the drop plus one day's accrual. A drop on a ``defaulted`` date is not a coupon: the
    accrued is written off, a loss. Principal paid down (factor falls) is returned at par."""
    clean, accrued = clean.astype(float), accrued.reindex_like(clean).astype(float)
    f = (factor.reindex_like(clean).astype(float) if factor is not None else pd.DataFrame(1.0, index=clean.index, columns=clean.columns)).ffill()
    dirty = clean + accrued
    if coupons is None:
        inc = accrued.diff().where(lambda x: x > 0).rolling(10, min_periods=1).median().shift(1).fillna(0.0)
        drop = accrued.diff() < -min_drop
        if coupon_dates is not None:
            drop &= coupon_dates.reindex_like(clean).fillna(False).astype(bool)
        if defaulted is not None:
            drop &= ~defaulted.reindex_like(clean).fillna(False).astype(bool)
        coupons = (accrued.shift(1) + inc - accrued).where(drop, 0.0).clip(lower=0.0)
    else:
        coupons = coupons.reindex_like(clean).fillna(0.0).astype(float)
    f0 = f.shift(1)
    paydown = (f0 - f).clip(lower=0.0) * 100.0
    out = (dirty * f + coupons * f0 + paydown) / (dirty.shift(1) * f0) - 1.0
    out.iloc[0] = np.nan
    return out


# ====================================================================================== month-to-date series
def decumulate_mtd(mtd_tr: pd.Series | pd.DataFrame, mtd_ir: pd.Series | pd.DataFrame | None = None, percent: bool = True) -> dict:
    """Daily returns from month-to-date cumulative series (Bloomberg ``*_MTD``): within a month
    ``r_t = (1 + mtd_t) / (1 + mtd_{t-1}) - 1``, on a month's first date ``r_t = mtd_t``. The rate
    leg is de-cumulated the same way and the excess leg is the difference, so ``tr = ir + xr`` on
    every date and each leg compounds back to its MTD at any month end. ``percent``: inputs in %."""
    def dec(m):
        m = m.astype(float) / (100.0 if percent else 1.0)
        months = pd.Series(m.index.to_period("M"), index=m.index)
        first = months.ne(months.shift(1))
        prev = m.shift(1)
        if isinstance(m, pd.DataFrame):
            r = (1.0 + m) / (1.0 + prev) - 1.0
            r[first.to_numpy()] = m[first.to_numpy()]
        else:
            r = ((1.0 + m) / (1.0 + prev) - 1.0).where(~first, m)
        return r
    tr = dec(mtd_tr)
    out = {"tr": tr}
    if mtd_ir is not None:
        ir = dec(mtd_ir.reindex(mtd_tr.index))
        out["ir"] = ir
        out["xr"] = tr - ir
    return out


def mtd_from_daily(daily: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Month-to-date compounded returns (fraction) from daily ones: the inverse of ``decumulate_mtd``."""
    g = (1.0 + daily.fillna(0.0)).groupby(daily.index.to_period("M"))
    return g.cumprod() - 1.0


# ====================================================================================== alignment and resampling
def align_legs(total: pd.Series | pd.DataFrame, rate: pd.Series | pd.DataFrame, mode: str = "inner") -> tuple:
    """Align a total-return leg and a rate (Treasury) leg: ``"inner"`` keeps common dates and warns
    about dropped ones, ``"strict"`` raises on any mismatch, ``"outer"`` keeps every date and
    carries zero where a leg has none (recorded in the returned report)."""
    a, b = total.index, rate.index
    only_a, only_b = a.difference(b), b.difference(a)
    report = {"mode": mode, "only_total": len(only_a), "only_rate": len(only_b)}
    if mode == "strict":
        if len(only_a) or len(only_b):
            raise ValueError(f"legs disagree on dates: {len(only_a)} only in the total leg (first {only_a[:1].tolist()}), {len(only_b)} only in the rate leg (first {only_b[:1].tolist()})")
        return total, rate, report
    if mode == "inner":
        common = a.intersection(b)
        if len(only_a) or len(only_b):
            warnings.warn(f"align_legs: dropped {len(only_a)} dates only in the total leg and {len(only_b)} only in the rate leg", stacklevel=2)
        return total.reindex(common), rate.reindex(common), report
    if mode == "outer":
        allx = a.union(b)
        return total.reindex(allx).fillna(0.0), rate.reindex(allx).fillna(0.0), report
    raise ValueError("mode must be inner, strict or outer")


def resample_returns(returns: pd.Series | pd.DataFrame, freq: str = "ME") -> pd.Series | pd.DataFrame:
    """Compound per-period returns into ``freq`` periods (``"W-FRI"``, ``"ME"``, ``"QE"``); a period
    with no observation is NaN, not zero."""
    g = (1.0 + returns).groupby(pd.Grouper(freq=freq))
    out = g.prod(min_count=1) - 1.0
    return out


def common_dates(*series) -> pd.DatetimeIndex:
    """The dates every series has (sleeves compared on the same periods)."""
    idx = None
    for s in series:
        ix = pd.DatetimeIndex(s.dropna(how="all").index if isinstance(s, pd.DataFrame) else s.dropna().index)
        idx = ix if idx is None else idx.intersection(ix)
    return idx if idx is not None else pd.DatetimeIndex([])


# ====================================================================================== forward returns (evaluation)
def forward_returns(tr: pd.DataFrame, horizons=(21,), ir: pd.DataFrame | None = None, lag: int = 1, alive: pd.DataFrame | None = None, membership: pd.DataFrame | None = None, dates=None) -> pd.DataFrame:
    """Forward returns per bond and horizon starting ``lag`` periods after each date: compounded
    total (``fwd_tr``), rate (``fwd_ir``) and excess (``fwd_xr = fwd_tr - fwd_ir``) returns over rows
    ``t+lag+1 .. t+lag+h``. ``exit``: the bond stops existing inside the window (the rest earns zero,
    cash); ``missing``: a return is missing while it exists; ``left_index``: membership ends inside
    the window. One row per (date, instrument_id, horizon) for the ``dates`` asked (all by default)."""
    T = len(tr.index)
    R = tr.to_numpy(dtype=float)
    fin = np.isfinite(R)
    ex = alive.reindex_like(tr).fillna(0.0).to_numpy() > 0 if alive is not None else None
    if ex is None:
        first = np.where(fin.any(axis=0), fin.argmax(axis=0), T)
        last = np.where(fin.any(axis=0), T - 1 - fin[::-1].argmax(axis=0), -1)
        tt = np.arange(T)[:, None]
        ex = (tt >= first[None, :]) & (tt <= last[None, :])
    G = np.vstack([np.zeros(R.shape[1]), np.cumsum(np.log1p(np.nan_to_num(R)), axis=0)])
    GI = np.vstack([np.zeros(R.shape[1]), np.cumsum(np.log1p(np.nan_to_num(ir.reindex_like(tr).to_numpy(dtype=float))), axis=0)]) if ir is not None else None
    gap = np.vstack([np.zeros((1, R.shape[1]), dtype=np.int64), np.cumsum(ex & ~fin, axis=0)])
    gone = np.vstack([np.zeros((1, R.shape[1]), dtype=np.int64), np.cumsum(~ex, axis=0)])
    mem = membership.reindex_like(tr).fillna(0.0).to_numpy() > 0 if membership is not None else None
    out_m = np.vstack([np.zeros((1, R.shape[1]), dtype=np.int64), np.cumsum(~mem, axis=0)]) if mem is not None else None
    rows = range(T) if dates is None else tr.index.get_indexer(pd.DatetimeIndex(dates))
    recs = []
    ids = tr.columns
    for i in rows:
        if i < 0:
            continue
        for h in horizons:
            a, b = i + lag, i + lag + h
            if b >= T:
                continue
            on = ex[i] if mem is None else (ex[i] & mem[i])
            if not on.any():
                continue
            ftr = np.expm1(G[b + 1][on] - G[a + 1][on])
            rec = pd.DataFrame({"date": tr.index[i], "instrument_id": ids[on], "horizon": h, "fwd_tr": ftr})
            if GI is not None:
                rec["fwd_ir"] = np.expm1(GI[b + 1][on] - GI[a + 1][on])
                rec["fwd_xr"] = rec["fwd_tr"] - rec["fwd_ir"]
            rec["exit"] = (gone[b + 1][on] - gone[a + 1][on]) > 0
            rec["missing"] = (gap[b + 1][on] - gap[a + 1][on]) > 0
            if out_m is not None:
                rec["left_index"] = (out_m[b + 1][on] - out_m[a + 1][on]) > 0
            recs.append(rec)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["date", "instrument_id", "horizon", "fwd_tr"])


# ====================================================================================== spread and duration models
def spread_return(oas_bp: pd.DataFrame, oasd: pd.DataFrame, on_exit: str = "carry") -> dict:
    """The spread-return proxy ``r ~ OAS_{t-1} dt - OASD_{t-1} dOAS`` (OAS in bp, dt in years from
    the dates), split into ``carry`` and ``spread`` legs and their sum ``total``. ``on_exit``: where
    the spread stops (the bond leaves the universe) the last period earns ``"carry"`` only,
    ``"last"`` (nothing more: NaN) or ``"redeem"`` (zero)."""
    dt = pd.Series(np.diff(oas_bp.index.values).astype("timedelta64[D]").astype(float) / 365.0, index=oas_bp.index[1:]).reindex(oas_bp.index)
    s0 = oas_bp.shift(1) / 1e4
    carry = s0.mul(dt, axis=0)
    ds = oas_bp.diff() / 1e4
    spread = -oasd.shift(1).reindex_like(oas_bp) * ds
    gone = oas_bp.isna() & s0.notna()
    if on_exit == "carry":
        spread = spread.where(~gone, 0.0)
    elif on_exit == "redeem":
        carry, spread = carry.where(~gone, 0.0), spread.where(~gone, 0.0)
    elif on_exit != "last":
        raise ValueError("on_exit must be carry, last or redeem")
    return {"carry": carry, "spread": spread, "total": carry + spread}


class YearFraction(Expr):
    """Calendar years in each row's period (row 0: the next period's length), broadcast."""

    kind = "numeric"

    def _own_names(self):
        return set()

    def _own_lookback(self):
        return 1

    def _eval(self, ctx):
        d = ctx.panel.dates.values.astype("datetime64[D]").astype(np.int64)
        dt = np.diff(d).astype(float) / 365.0
        dt = np.concatenate([[dt[0] if len(dt) else 1 / 252], dt])
        return np.broadcast_to(dt[:, None], (ctx.T, ctx.N))

    def __repr__(self) -> str:
        return "YearFraction()"


def SpreadReturn(oas: str = "oas_bp", duration: str = "oasd") -> Expr:
    """The spread-return proxy as a realised-return expression (it reads only data up to its row):
    ``Strategy(returns=SpreadReturn())`` earns carry plus spread change on OAS and OASD panels."""
    s = Field(oas, lag=0) / 1e4
    return s.lag(1) * YearFraction() - Field(duration, lag=0).lag(1) * s.diff()


def duration_convexity_return(carry: pd.DataFrame | pd.Series, duration, convexity, ds) -> pd.DataFrame | pd.Series:
    """Sleeve return from a duration / convexity model: ``carry - D ds + C ds^2 / 2`` (``ds`` the
    spread or yield change as a fraction; D and C known at the period start)."""
    return carry - duration * ds + 0.5 * convexity * ds ** 2


# ====================================================================================== government curves
def _par_moddur(y: np.ndarray, tenor: float, freq: int = 2) -> np.ndarray:
    """Modified duration of a par bond with coupon = yield."""
    y = np.maximum(np.asarray(y, dtype=float), 1e-9)
    c = y / freq
    n = max(round(tenor * freq), 1)
    mac = (1.0 + c) / c * (1.0 - (1.0 + c) ** -n) / freq
    return mac / (1.0 + c)


def constant_maturity_returns(yields: pd.DataFrame, freq: int = 2) -> pd.DataFrame:
    """Constant-maturity government returns per tenor column (years as column labels, yields as
    fractions): ``y_{t-1} dt - ModDur_{t-1} dy + Conv dy^2 / 2`` for a par bond rebuilt each period."""
    dt = pd.Series(np.diff(yields.index.values).astype("timedelta64[D]").astype(float) / 365.0, index=yields.index[1:]).reindex(yields.index)
    out = {}
    for tenor in yields.columns:
        y = yields[tenor].astype(float)
        md = pd.Series(_par_moddur(y.to_numpy(), float(tenor), freq), index=y.index)
        conv = md ** 2                                                         # par-bond convexity, to first order
        dy = y.diff()
        out[tenor] = y.shift(1) * dt - md.shift(1) * dy + 0.5 * conv.shift(1) * dy ** 2
    return pd.DataFrame(out)


def carry_rolldown(yields: pd.DataFrame, tenor: float, horizons_months=(1, 3)) -> pd.DataFrame:
    """Carry (yield accrual over the horizon) and roll-down ``y(T) - y(T - h)`` (curve interpolated
    linearly in tenor) for ``tenor`` years, per horizon in months, as yield changes and returns
    (roll-down return ~ ModDur x yield fall)."""
    tenors = np.array(sorted(float(c) for c in yields.columns))
    Y = yields[[c for c in sorted(yields.columns, key=float)]].to_numpy(dtype=float)
    out = {}
    yT = np.array([np.interp(tenor, tenors, row) for row in Y])
    md = _par_moddur(yT, tenor)
    for h in horizons_months:
        yTh = np.array([np.interp(max(tenor - h / 12.0, tenors[0]), tenors, row) for row in Y])
        roll = yT - yTh
        out[f"carry_{h}m"] = yT * h / 12.0
        out[f"rolldown_{h}m_yield"] = roll
        out[f"rolldown_{h}m"] = md * roll
    return pd.DataFrame(out, index=yields.index)


# ====================================================================================== currency hedging
def fx_hedged_returns(local: pd.Series | pd.DataFrame, spot: pd.Series, forward: pd.Series, ratio: float = 1.0, roll: str | None = None) -> pd.Series | pd.DataFrame:
    """Hedged returns in the base currency (rates in base per unit of the asset's currency).
    ``roll=None``: a one-period forward rolled every period, ``(1 + r) h_{t-1} / h_t - 1`` with
    ``h = F / S`` (the local return plus the forward premium, no exposure to the spot move).
    ``roll`` a pandas frequency (``"ME"``, ``"BME"``, ``"QE"``): a hedge programme that sells each
    forward at a roll date's close and holds it to the next: the position's value at the roll is
    hedged, its spot move offset and the forward premium earned evenly to the next roll (growth
    after the roll is unhedged until the next one). ``ratio``: the share of the position hedged."""
    if roll is None:
        h = (forward / spot).reindex(local.index)
        k = h.shift(1) / h
        hedged = (1.0 + local).mul(k, axis=0) - 1.0 if isinstance(local, pd.DataFrame) else (1.0 + local) * k - 1.0
        return hedged if ratio == 1.0 else ratio * hedged + (1.0 - ratio) * unhedged_returns(local, spot)
    idx = pd.DatetimeIndex(local.index)
    S = spot.reindex(idx).ffill().to_numpy(dtype=float)
    F = forward.reindex(idx).ffill().to_numpy(dtype=float)
    last = pd.Series(np.arange(len(idx)), index=idx).resample(roll).last().dropna().astype(int).to_numpy()
    roll_rows = sorted(set(last[last < len(idx) - 1].tolist()) | {0})
    L = np.atleast_2d(np.nan_to_num(np.asarray(local, dtype=float)).T).T
    out = np.zeros_like(L)
    ends = [*roll_rows[1:], len(idx) - 1]
    for tau, nxt in zip(roll_rows, ends, strict=True):
        horizon = max((idx[nxt] - idx[tau]).days, 1)
        prem = F[tau] / S[tau] - 1.0
        grow = np.cumprod(1.0 + L[tau + 1:nxt + 1], axis=0)                    # local growth since the roll
        elapsed = (idx[tau + 1:nxt + 1] - idx[tau]).days.to_numpy(dtype=float)[:, None]
        spot_move = (S[tau + 1:nxt + 1] / S[tau] - 1.0)[:, None]
        V = grow * (1.0 + spot_move) + ratio * (prem * elapsed / horizon - spot_move)
        prev = np.vstack([np.ones((1, L.shape[1])), V[:-1]])
        out[tau + 1:nxt + 1] = V / prev - 1.0
    res = pd.DataFrame(out, index=idx, columns=local.columns) if isinstance(local, pd.DataFrame) else pd.Series(out[:, 0], index=idx, name=getattr(local, "name", None))
    res.iloc[0] = np.nan
    return res


def unhedged_returns(local: pd.Series | pd.DataFrame, spot: pd.Series) -> pd.Series | pd.DataFrame:
    """Returns in the base currency without a hedge: ``(1 + r) S_t / S_{t-1} - 1``."""
    g = (spot / spot.shift(1)).reindex(local.index)
    return (1.0 + local).mul(g, axis=0) - 1.0 if isinstance(local, pd.DataFrame) else (1.0 + local) * g - 1.0


def hedge_carry(spot: pd.Series, forward: pd.Series, tenor_years: float) -> pd.Series:
    """The annualised carry of the hedge: ``(F / S - 1) / tenor``."""
    return (forward / spot - 1.0) / float(tenor_years)


# ====================================================================================== credit default swaps
def _imm_dates(start, end) -> pd.DatetimeIndex:
    yrs = range(pd.Timestamp(start).year - 1, pd.Timestamp(end).year + 2)
    ds = [pd.Timestamp(y, m, 20) for y in yrs for m in (3, 6, 9, 12)]
    return pd.DatetimeIndex([d for d in ds if pd.Timestamp(start) <= d <= pd.Timestamp(end)])


def cds_returns(spread_bp: pd.Series | pd.DataFrame, risky_duration, roll_dates=None, coupon_bp: float | None = None, protection: str = "sell") -> dict:
    """CDS / CDX returns per unit notional for the protection seller (``"buy"``: negated):
    ``-RD_{t-1} dS + c dt`` with ``c`` the fixed ``coupon_bp`` when given, else the spread at the
    period start (spreads in bp, ACT/360). On ``roll_dates`` (on-the-run series changes; IMM dates
    20 Mar / Jun / Sep / Dec when None) the spread change is the roll and earns nothing.
    ``premium``: the accrued premium paid on each IMM date (per unit notional)."""
    s = spread_bp.astype(float) / 1e4
    idx = s.index
    dt = pd.Series(np.diff(idx.values).astype("timedelta64[D]").astype(float) / 360.0, index=idx[1:]).reindex(idx)
    rolls = pd.DatetimeIndex(roll_dates) if roll_dates is not None else _imm_dates(idx[0], idx[-1])
    roll = pd.Series(idx.normalize().isin(rolls.normalize()), index=idx)
    keep = (~roll).astype(float)
    ds = s.diff().mul(keep, axis=0) if isinstance(s, pd.DataFrame) else s.diff() * keep
    rd = pd.Series(float(risky_duration), index=idx) if np.isscalar(risky_duration) else risky_duration
    rate = (pd.DataFrame(coupon_bp / 1e4, index=idx, columns=s.columns) if isinstance(s, pd.DataFrame) else pd.Series(coupon_bp / 1e4, index=idx)) if coupon_bp is not None else s.shift(1)
    carry = rate.mul(dt, axis=0) if isinstance(rate, pd.DataFrame) else rate * dt
    mtm = -(rd.shift(1).mul(ds, axis=0) if isinstance(ds, pd.DataFrame) and isinstance(rd, pd.Series) else rd.shift(1) * ds)
    sign = 1.0 if protection == "sell" else -1.0
    accrued = carry.fillna(0.0).cumsum()
    prem_dates = _imm_dates(idx[0], idx[-1])
    at = accrued.reindex(prem_dates, method="ffill")
    premium = at.diff().fillna(at)
    return {"total": sign * (mtm + carry), "carry": sign * carry, "mtm": sign * mtm, "roll_dates": rolls, "premium": sign * premium}


# ====================================================================================== futures
def continuous_futures(prices: pd.DataFrame, first_notice: dict, roll_days: int = 5, adjust: str = "back", calendar=None) -> dict:
    """A continuous futures series from contract prices (columns: contracts in expiry order):
    hold each contract until ``roll_days`` trading days before its first notice date, then the next.
    ``adjust="back"``: differences at each roll are added to history (back-adjusted levels);
    ``"ratio"``: history scaled by the price ratio; ``"none"``: spliced. Returns the series, the
    held contract per date, the roll dates and daily returns of the held contract (no roll jump)."""
    dates = prices.index
    contracts = list(prices.columns)
    fnd = {c: pd.Timestamp(first_notice[c]) for c in contracts}
    held = []
    k = 0
    for d in dates:
        while k < len(contracts) - 1:
            pos = dates.searchsorted(fnd[contracts[k]])
            if dates.get_loc(d) >= pos - roll_days:
                k += 1
            else:
                break
        held.append(contracts[k])
    held = pd.Series(held, index=dates)
    px = pd.Series([prices.at[d, c] for d, c in held.items()], index=dates, dtype=float)
    rolls = held.index[held.ne(held.shift(1)) & held.shift(1).notna()]
    ret = pd.Series(np.nan, index=dates)
    for i in range(1, len(dates)):
        c = held.iloc[i]
        ret.iloc[i] = prices[c].iloc[i] / prices[c].iloc[i - 1] - 1.0           # the held contract's own move: no roll jump
    level = px.copy()
    if adjust in ("back", "ratio"):
        for r in rolls:
            i = dates.get_loc(r)
            new, old = prices[held.iloc[i]].iloc[i - 1], prices[held.iloc[i - 1]].iloc[i - 1]
            if adjust == "back":
                level.iloc[:i] += new - old
            else:
                level.iloc[:i] *= new / old
    elif adjust != "none":
        raise ValueError("adjust must be back, ratio or none")
    return {"level": level, "contract": held, "rolls": rolls, "returns": ret}


def empirical_dv01(futures_returns: pd.Series, yield_changes: pd.Series, window: int = 60, price: pd.Series | None = None) -> pd.Series:
    """Rolling regression of futures returns on yield changes (bp): the slope is the return per bp,
    i.e. DV01 per unit notional (times ``price`` when given, for DV01 per contract price)."""
    y = yield_changes.reindex(futures_returns.index).astype(float)
    x = futures_returns.astype(float)
    cov = x.rolling(window).cov(y)
    var = y.rolling(window).var()
    slope = -(cov / var)
    return slope * (price.reindex(slope.index) if price is not None else 1.0)


# ====================================================================================== pairs in spread units
def pair_spread_pnl(a: pd.Series, b: pd.Series, beta: float | str = "rolling", window: int = 60, lag: int = 1) -> pd.DataFrame:
    """Pairs in spread units: the P&L per unit long position in ``a - beta b`` is
    ``-(da - beta_{t-1} db)`` in spread points (spreads tighten: gain), with ``beta`` a number or a
    rolling OLS of ``da`` on ``db`` over ``window`` changes, lagged ``lag`` periods (known at the
    decision). Columns: beta, spread (``a - beta b``), pnl_per_unit."""
    da, db = a.diff(), b.diff()
    if beta == "rolling":
        bt = (da.rolling(window).cov(db) / db.rolling(window).var()).shift(lag)
    else:
        bt = pd.Series(float(beta), index=a.index)
    pnl = -(da - bt.shift(1) * db)
    return pd.DataFrame({"beta": bt, "spread": a - bt * b, "pnl_per_unit": pnl})


def equal_move_sizing(returns_a: pd.Series, returns_b: pd.Series, window: int = 60) -> pd.Series:
    """The notional of leg b per unit of leg a that makes a typical (rolling standard deviation)
    move equal in both legs: the equal-1%-move sizing used for ETF pair legs, lagged one period."""
    return (returns_a.rolling(window).std() / returns_b.rolling(window).std()).shift(1)


__all__ = [
    "SpreadReturn",
    "YearFraction",
    "align_legs",
    "carry_rolldown",
    "cds_returns",
    "common_dates",
    "constant_maturity_returns",
    "continuous_futures",
    "decumulate_mtd",
    "duration_convexity_return",
    "empirical_dv01",
    "equal_move_sizing",
    "forward_returns",
    "fx_hedged_returns",
    "hedge_carry",
    "mtd_from_daily",
    "pair_spread_pnl",
    "resample_returns",
    "spread_return",
    "total_return_from_prices",
    "unhedged_returns",
]
