"""Synthetic credit universe as panels, at index scale.

Vectorised generator for the panel engine, signal evaluation, benchmarks and the optimiser: a
Treasury key-rate curve, an overnight cash rate, thousands of bullet corporates across issuers,
sectors and ratings (new issues and maturities inside the window, rating migrations, optional
defaults), OAS from a market factor plus mean-reverting idiosyncratic spreads, analytics (OAD, OASD,
DTS, key-rate durations), daily total returns, key-rate-matched Treasury returns and excess returns,
bid/ask spreads, a monthly-reconstituted market-value weighted index with drifting daily weights,
index-level series, a value signal with genuine but noisy skill, and two overlay series (a 10-year
Treasury future and an IG CDX, as excess returns per unit notional).

For plumbing, tests and scale benchmarks only: the numbers mean nothing about real markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.data.panel import Panel
from nautilus_fi.instruments.calendar import us_bond_market_calendar

RATING_BASE_BP = np.array([0, 35, 40, 45, 55, 65, 75, 90, 110, 130, 150, 230, 280, 330, 400, 470, 550, 750, 900, 1100, 1300, 1500, 1800], dtype=float)  # by score 1..22
SECTORS = ("FIN", "IND", "UTL", "ENE", "TMT")
SECTOR_OFFSET_BP = {"FIN": 8.0, "IND": 0.0, "UTL": -8.0, "ENE": 20.0, "TMT": 4.0}


@dataclass
class SyntheticPanelConfig:
    start: str = "2015-01-02"
    end: str = "2016-12-30"
    n_bonds: int = 600
    n_issuers: int | None = None
    seed: int = 3
    sectors: tuple[str, ...] = SECTORS
    key_tenors: tuple[float, ...] = (2.0, 5.0, 10.0, 30.0)
    hy_share: float = 0.15
    index_min_years: float = 1.0
    index_min_amount: float = 300e6
    index_max_rating_score: int = 10
    idio_halflife_days: float = 60.0
    idio_vol: float = 0.012                     # daily vol of the log idiosyncratic spread
    signal_noise: float = 1.0                   # noise / signal ratio of the value signal
    stress_start: str | None = None             # market spread spike starting here
    stress_size: float = 0.6
    n_defaults: int = 0
    missing_rate: float = 0.0                   # random missing total returns (exercises missing-return policies)
    n_bad_dates: int = 0                        # dates where most returns are reported as exactly zero
    dtype: type = np.float32


@dataclass
class SyntheticPanelData:
    config: SyntheticPanelConfig
    panel: Panel
    attributes: pd.DataFrame
    treasury: pd.DataFrame                      # key-rate yields (y_<t>) and returns (r_<t>)
    cash: pd.Series                             # overnight cash return per period
    index: pd.DataFrame                         # index tr / xr / oasd / oas_bp / dts per day
    membership: pd.DataFrame                    # date (reconstitution), instrument_id, weight
    overlays: Panel                             # TY and CDX_IG excess returns per unit notional + exposures
    true_tr: np.ndarray | None = None           # total returns before injected missing / bad dates
    defaults: dict[str, pd.Timestamp] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.panel.dates

    @property
    def ids(self) -> pd.Index:
        return self.panel.ids


def _price(c: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Clean price per 1 of face of a semi-annual bullet with ``t`` years to maturity (accrual ignored)."""
    y = np.maximum(y, 1e-5)
    n = np.maximum(2.0 * t, 1e-6)
    disc = (1.0 + y / 2.0) ** (-n)
    return c / y * (1.0 - disc) + disc


def generate_synthetic_panel(cfg: SyntheticPanelConfig | None = None) -> SyntheticPanelData:
    cfg = cfg or SyntheticPanelConfig()
    rng = np.random.default_rng(cfg.seed)
    cal = us_bond_market_calendar()
    days = cal.business_days(pd.Timestamp(cfg.start).date(), pd.Timestamp(cfg.end).date())
    dates = pd.DatetimeIndex(pd.to_datetime(days))
    T, N = len(dates), cfg.n_bonds
    dt_years = np.diff(dates.values).astype("timedelta64[D]").astype(float) / 365.0
    dt_years = np.concatenate([[1.0 / 252.0], dt_years])
    dtype = cfg.dtype

    # ---------------------------------------------------------------- Treasury key rates and cash
    kt = np.array(cfg.key_tenors, dtype=float)
    base = np.interp(kt, [2, 5, 10, 30], [0.012, 0.017, 0.022, 0.029])
    level = rng.normal(0, 0.00045, T).cumsum()
    slope = rng.normal(0, 0.00018, T).cumsum()
    load = (kt - kt.mean()) / (kt.max() - kt.min())
    y_key = np.maximum(base[None, :] + level[:, None] + slope[:, None] * load[None, :], 0.001)
    cash_rate = np.maximum(y_key[:, 0] - 0.003, 0.0005)
    cash = pd.Series(cash_rate * dt_years, index=dates, name="cash")
    d_key = kt / (1.0 + y_key / 2.0)                                          # zero-coupon modified durations
    r_key = np.vstack([np.zeros(len(kt)), y_key[:-1] * dt_years[1:, None] - d_key[:-1] * np.diff(y_key, axis=0)])
    treasury = pd.DataFrame(np.hstack([y_key, r_key]), index=dates, columns=[f"y_{t:g}" for t in kt] + [f"r_{t:g}" for t in kt])

    # ---------------------------------------------------------------- issuers and bonds
    n_iss = cfg.n_issuers or max(N // 4, 1)
    iss_sector = rng.choice(len(cfg.sectors), n_iss)
    hy = rng.random(n_iss) < cfg.hy_share
    iss_rating = np.where(hy, rng.integers(11, 17, n_iss), rng.integers(2, 11, n_iss))
    iss_offset = rng.normal(0, 15, n_iss)
    issuer = rng.integers(0, n_iss, N)
    tenor = rng.choice([3, 5, 7, 10, 20, 30], N, p=[0.15, 0.25, 0.2, 0.25, 0.05, 0.10]).astype(float)
    start, end = dates[0], dates[-1]
    lo = (start - pd.to_timedelta(tenor * 365.25 - 60, unit="D")).values.astype("datetime64[D]").astype(np.int64)
    hi = np.full(N, (end - pd.Timedelta(days=45)).value // 86_400_000_000_000)
    issue_day = (lo + rng.random(N) * (hi - lo)).astype(np.int64)
    issue = pd.to_datetime(issue_day, unit="D")
    maturity = issue + pd.to_timedelta(np.round(tenor * 365.25), unit="D")
    amt = np.clip(np.round(np.exp(rng.normal(np.log(700e6), 0.6, N)) / 50e6) * 50e6, 150e6, 5e9)
    ids = pd.Index([f"B{i:05d}" for i in range(N)])
    sector = np.array(cfg.sectors)[iss_sector[issuer]]

    date_ns = dates.values.astype("datetime64[D]").astype(np.int64)
    iss_ns = issue.values.astype("datetime64[D]").astype(np.int64)
    mat_ns = maturity.values.astype("datetime64[D]").astype(np.int64)
    alive = (date_ns[:, None] >= iss_ns[None, :]) & (date_ns[:, None] < mat_ns[None, :])
    years_left = ((mat_ns[None, :] - date_ns[:, None]) / 365.25).astype(dtype)

    # ratings: monthly migrations on top of the issuer rating
    month_id = (dates.year * 12 + dates.month).to_numpy()
    month_starts = np.concatenate([[True], month_id[1:] != month_id[:-1]])
    n_months = int(month_starts.sum())
    moves = np.where(rng.random((n_months, N)) < 0.012, rng.choice([-1, 1], (n_months, N), p=[0.4, 0.6]), 0).cumsum(axis=0)
    rating = np.clip(iss_rating[issuer][None, :] + moves[np.cumsum(month_starts) - 1], 1, 21).astype(dtype)

    # ---------------------------------------------------------------- spreads: market factor + idiosyncratic OU
    m = rng.normal(0, 0.011, T)
    phi_m = np.exp(-np.log(2) / 250.0)
    mf = np.zeros(T)
    for t in range(1, T):
        mf[t] = phi_m * mf[t - 1] + m[t]
    if cfg.stress_start:
        s0 = dates.searchsorted(pd.Timestamp(cfg.stress_start))
        k = np.arange(T) - s0
        mf += np.where(k >= 0, cfg.stress_size * np.minimum(k / 20.0, 1.0) * np.exp(-np.maximum(k - 20, 0) / 120.0), 0.0)
    phi = np.exp(-np.log(2) / cfg.idio_halflife_days)
    x = np.zeros((T, N), dtype=dtype)
    x[0] = rng.normal(0, cfg.idio_vol / np.sqrt(1 - phi ** 2), N)
    eps = rng.normal(0, cfg.idio_vol, (T, N)).astype(dtype)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps[t]
    sect_off = np.array([SECTOR_OFFSET_BP.get(s, 0.0) for s in sector])
    base_bp = RATING_BASE_BP[rating.astype(int)] + sect_off[None, :] + iss_offset[issuer][None, :] + 2.5 * np.minimum(years_left, 30)
    oas = (np.maximum(base_bp, 15.0) * np.exp(mf[:, None] + x)).astype(dtype)

    # ---------------------------------------------------------------- yields, prices, analytics, returns
    tsy_at = np.empty((T, N), dtype=dtype)
    for t in range(T):
        tsy_at[t] = np.interp(np.clip(years_left[t], kt[0], kt[-1]), kt, y_key[t])
    y = tsy_at + oas / 1e4
    yi = np.clip(dates.searchsorted(issue), 0, T - 1)
    coupon = np.round((tsy_at[yi, np.arange(N)] + oas[yi, np.arange(N)] / 1e4) * 800) / 800
    coupon = np.maximum(coupon, 0.005)
    tl = np.maximum(years_left, 1e-3)
    px = _price(coupon[None, :], y, tl)
    bump = 1e-4
    dmod = ((_price(coupon[None, :], y - bump, tl) - _price(coupon[None, :], y + bump, tl)) / (2 * bump * px)).astype(dtype)
    price_clean = (100.0 * px).astype(dtype)
    oasd = dmod
    dts = (oasd * oas).astype(dtype)
    # key-rate durations: duration split between the neighbouring key tenors
    krd = {}
    pos = np.clip(np.searchsorted(kt, years_left, side="left"), 1, len(kt) - 1)
    lo_t, hi_t = kt[pos - 1], kt[pos]
    wt_hi = np.clip((years_left - lo_t) / (hi_t - lo_t), 0.0, 1.0)
    wt_hi = np.where(years_left <= kt[0], 0.0, np.where(years_left >= kt[-1], 1.0, wt_hi))
    for j, t_ in enumerate(kt):
        share = np.where(pos - 1 == j, 1.0 - wt_hi, 0.0) + np.where(pos == j, wt_hi, 0.0)
        if j == 0:
            share = np.where(years_left <= kt[0], 1.0, share)
        krd[f"krd_{t_:g}"] = (dmod * share).astype(dtype)
    tr = np.full((T, N), np.nan, dtype=dtype)
    tr[1:] = ((px[1:] - px[:-1] + coupon[None, :] * dt_years[1:, None]) / px[:-1]).astype(dtype)
    # key-rate matched Treasury portfolio: krd_k / D_k in the zero at tenor k, the rest in cash
    leg = np.zeros((T - 1, N))
    held = np.zeros((T - 1, N))
    for j, t_ in enumerate(kt):
        wk = krd[f"krd_{t_:g}"][:-1] / d_key[:-1, j][:, None]
        leg += wk * r_key[1:, j][:, None]
        held += wk
    tsy_r = np.full((T, N), np.nan, dtype=dtype)
    tsy_r[1:] = (leg + (1.0 - held) * cash.to_numpy()[1:, None]).astype(dtype)
    del leg, held, eps

    # defaults: the price drops to a recovery and the bond leaves the universe
    defaults: dict[str, pd.Timestamp] = {}
    if cfg.n_defaults:
        cand = np.where((rating[-1] >= 13) & (mat_ns > date_ns[T // 3]) & (iss_ns < date_ns[T // 3]))[0]
        for b in rng.choice(cand, min(cfg.n_defaults, len(cand)), replace=False):
            t0 = int(rng.integers(T // 3, T - 5))
            if not alive[t0, b] or not alive[t0 - 1, b]:
                continue
            tr[t0, b] = 0.40 / px[t0 - 1, b] - 1.0
            alive[t0 + 1:, b] = False
            defaults[ids[b]] = dates[t0]
    live_prev = np.vstack([np.zeros(N, dtype=bool), alive[:-1]])
    tr = np.where(alive & live_prev, tr, np.nan).astype(dtype)
    tsy_r = np.where(alive & live_prev, tsy_r, np.nan).astype(dtype)
    xr = (tr - tsy_r).astype(dtype)
    true_tr = tr.copy()
    del px
    if cfg.missing_rate > 0:
        hole = (rng.random((T, N)) < cfg.missing_rate) & alive & live_prev
        tr = np.where(hole, np.nan, tr).astype(dtype)
    if cfg.n_bad_dates:
        for t0 in rng.choice(np.arange(5, T - 5), cfg.n_bad_dates, replace=False):
            tr[t0] = np.where(alive[t0] & live_prev[t0], 0.0, np.nan)

    age = ((date_ns[:, None] - iss_ns[None, :]) / 365.25).astype(dtype)
    hs = (np.where(rating <= 10, 7.0, 25.0) * (1.0 + 0.04 * np.minimum(years_left, 20)) * (1.0 + 0.15 * (age > 2)) * (1.0 + 1.5 * np.clip(mf, 0, None))[:, None]).astype(dtype)
    signal = (x + cfg.signal_noise * cfg.idio_vol / np.sqrt(1 - phi ** 2) * rng.normal(0, 1, (T, N))).astype(dtype)

    nanmask = ~alive
    def m_(a):
        return np.where(nanmask, np.nan, a).astype(dtype)

    # ---------------------------------------------------------------- index: monthly reconstitution, drifting weights
    month_end = np.concatenate([month_id[1:] != month_id[:-1], [True]])
    rebal = np.where(month_end)[0]
    index_w = np.zeros((T, N), dtype=np.float64)
    member = np.zeros((T, N), dtype=dtype)
    rows = []
    mv = amt[None, :] * price_clean / 100.0
    w = np.zeros(N)
    next_rebal = {0: True} | {int(r): True for r in rebal}         # the index exists from the first day
    current = np.zeros(N)
    for t in range(T):
        if t > 0:                                   # weights that earned period t, grown by its returns
            grown = current * (1.0 + np.nan_to_num(true_tr[t].astype(np.float64)))
            grown = np.where(alive[t], grown, 0.0)
            s = grown.sum()
            current = grown / s if s > 0 else grown
        if t in next_rebal and t + 1 < T:           # month-end reconstitution, effective for period t + 1
            elig = alive[t] & alive[t + 1] & (years_left[t] >= cfg.index_min_years) & (amt >= cfg.index_min_amount) & (rating[t] <= cfg.index_max_rating_score)
            w = np.where(elig, mv[t], 0.0)
            w = w / w.sum() if w.sum() > 0 else w
            current = w
            nz = np.nonzero(w)[0]
            rows.append(pd.DataFrame({"date": dates[t], "instrument_id": ids[nz], "weight": w[nz]}))
        index_w[t] = current                        # close-of-day weights: they earn period t + 1
        member[t] = current > 0
    membership = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "instrument_id", "weight"])
    prev_w = np.vstack([np.zeros(N), index_w[:-1]])
    clean_tr = np.nan_to_num(true_tr.astype(np.float64))
    index = pd.DataFrame({"tr": (prev_w * clean_tr).sum(axis=1), "xr": (prev_w * np.nan_to_num(xr.astype(np.float64))).sum(axis=1), "oasd": (index_w * np.nan_to_num(oasd)).sum(axis=1), "oas_bp": (index_w * np.nan_to_num(oas)).sum(axis=1), "dts": (index_w * np.nan_to_num(dts)).sum(axis=1), "oad": (index_w * np.nan_to_num(dmod)).sum(axis=1)}, index=dates)
    index.iloc[0, :2] = 0.0

    # ---------------------------------------------------------------- overlays: TY future and IG CDX (excess returns per unit notional)
    j10 = int(np.argmin(np.abs(kt - 10.0)))
    ty_r = r_key[:, j10] - cash.to_numpy()
    cdx_s = np.maximum(index["oas_bp"].to_numpy() - 20.0, 20.0)
    cdx_dur = 4.5
    cdx_r = np.concatenate([[0.0], cdx_s[:-1] / 1e4 * dt_years[1:] - cdx_dur * np.diff(cdx_s) / 1e4])
    ov_ids = pd.Index(["TY", "CDX_IG"])
    overlays = Panel(dates, ov_ids, {"tr": np.column_stack([ty_r, cdx_r]), "oad": np.column_stack([d_key[:, j10], np.zeros(T)]), "oasd": np.column_stack([np.zeros(T), np.full(T, cdx_dur)]), "oas_bp": np.column_stack([np.full(T, np.nan), cdx_s]), "dv01": np.column_stack([d_key[:, j10] * 0.01, np.full(T, cdx_dur * 0.01)])})

    values = {"tr": tr, "xr": m_(xr), "tsy_r": m_(tsy_r), "oas_bp": m_(oas), "oad": m_(dmod), "oasd": m_(oasd), "dts": m_(dts), "price_clean": m_(price_clean), "half_spread_bp": m_(hs), "amt_out": m_(np.broadcast_to(amt, (T, N))), "rating_score": m_(rating), "years_to_maturity": m_(years_left), "index_weight": index_w.astype(dtype), "member": member, "signal.value": m_(signal), "alive": alive.astype(dtype)}
    values.update({k: m_(v) for k, v in krd.items()})
    panel = Panel(dates, ids, values)
    attributes = pd.DataFrame({"issuer": [f"ISS{i:04d}" for i in issuer], "sector": sector, "coupon": coupon, "issue_date": issue.date, "maturity": maturity.date, "currency": "USD", "amount_outstanding": amt, "min_piece": 2_000.0, "increment": 1_000.0, "rating_initial": iss_rating[issuer]}, index=ids)
    return SyntheticPanelData(cfg, panel, attributes, treasury, cash, index, membership, overlays, true_tr if (cfg.missing_rate or cfg.n_bad_dates) else None, defaults)


def to_market_inputs(data: SyntheticPanelData, symbols: list[str] | None = None, analytics: bool = True, fields: tuple[str, ...] = ("signal.value",), publish_ts: bool = False) -> dict[str, pd.DataFrame]:
    """The panel universe as event-engine inputs: terms (semi-annual 30/360 bullets), evaluated
    prices (mid and bid/ask from the half spread), vendor analytics (OAS, OAD, OASD, key rates),
    Treasury par curves, ratings, benchmark constituents and generic fields."""
    from nautilus_fi.instruments.enums import score_to_rating

    p = data.panel
    ids = list(symbols) if symbols is not None else list(p.ids)
    a = data.attributes.loc[ids]
    ref = pd.DataFrame({"symbol": ids, "issuer": a["issuer"].to_numpy(), "issue_date": pd.to_datetime(a["issue_date"]).dt.date, "maturity": pd.to_datetime(a["maturity"]).dt.date, "coupon_rate": a["coupon"].to_numpy(), "frequency": 2, "day_count": "THIRTY_360", "currency": "USD", "asset_type": "CORPORATE", "sector": a["sector"].to_numpy(), "rating_initial": [score_to_rating(int(r)) for r in a["rating_initial"]], "amount_outstanding": a["amount_outstanding"].to_numpy(), "min_piece": a["min_piece"].to_numpy(), "increment": a["increment"].to_numpy()})
    sub = p.subset(ids=ids)
    long = sub.to_long(["price_clean", "half_spread_bp", "oas_bp", "oad", "oasd", *[f for f in sub.fields if f.startswith("krd_")], "rating_score", *[f for f in fields if f in sub]])
    long = long.dropna(subset=["price_clean"])
    hs = long["half_spread_bp"].fillna(10.0) / 1e4
    ev = pd.DataFrame({"date": long["date"], "instrument_id": long["instrument_id"], "mid": long["price_clean"], "bid": long["price_clean"] * (1 - hs), "ask": long["price_clean"] * (1 + hs), "score": 8.0})
    if publish_ts:
        ev["publish_ts"] = pd.to_datetime(ev["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=16, minutes=15)
    out = {"reference": ref, "evaluated_prices": ev}
    if analytics:
        acols = ["date", "instrument_id", "oas_bp", "oad", "oasd", *[c for c in long.columns if c.startswith("krd_")]]
        out["analytics"] = long[acols].rename(columns={c: f"krd_{c[4:]}" for c in acols if c.startswith("krd_")})
    tre = data.treasury
    out["curves"] = pd.concat([pd.DataFrame({"date": tre.index, "curve": "UST", "tenor": float(c[2:]), "rate": tre[c].to_numpy(), "kind": "par"}) for c in tre.columns if c.startswith("y_")], ignore_index=True)
    rs = long[["date", "instrument_id", "rating_score"]].dropna()
    rs = rs[rs.groupby("instrument_id")["rating_score"].diff().fillna(1) != 0]
    out["ratings"] = pd.DataFrame({"date": rs["date"], "instrument_id": rs["instrument_id"], "rating": [score_to_rating(int(x)) for x in rs["rating_score"]]})
    m = data.membership[data.membership["instrument_id"].isin(ids)].copy()
    m["weight"] = m.groupby("date")["weight"].transform(lambda w: w / w.sum())
    out["benchmark_constituents"] = m
    fl = [f for f in fields if f in sub]
    if fl:
        out["fields"] = long[["date", "instrument_id", *fl]].dropna(subset=fl, how="all")
    return out


__all__ = ["SyntheticPanelConfig", "SyntheticPanelData", "generate_synthetic_panel", "to_market_inputs"]
