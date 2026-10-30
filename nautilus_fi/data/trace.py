"""TRACE transaction cleaning and knowledge-time stamping.

Implements the standard Dick-Nielsen style filters on a *standardized* frame so that FINRA academic,
Enhanced TRACE (WRDS) or vendor extracts can be mapped in with ``columns``:

* remove cancellations and corrections (and the messages they refer to);
* remove reversals (as-of trades reversing an earlier report) and their originals;
* de-duplicate inter-dealer trades reported by both dealers;
* drop when-issued, special-price and locked-in agency duplicates when flagged;
* reversal / outlier filter: drop prices deviating more than ``max_abs_return`` from the median of
  neighbouring prints of the same bond (Edwards, Harris and Piwowar style);
* stamp ``available_ts`` = report time (15-minute dissemination rule) and hide sizes above the
  dissemination cap until the Enhanced TRACE release lag has passed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STANDARD_COLUMNS = {
    "instrument_id": "instrument_id",   # CUSIP / ISIN / your symbol
    "exec_ts": "exec_ts",               # execution timestamp
    "report_ts": "report_ts",           # report / dissemination timestamp (optional)
    "price": "price",
    "size": "size",                     # face amount
    "side": "side",                     # B / S / D (customer buy, customer sell, inter-dealer)
    "status": "status",                 # T trade, C cancel, X correction/reversal (optional)
    "msg_id": "msg_id",                 # message sequence number (optional)
    "orig_msg_id": "orig_msg_id",       # referenced message for cancels/corrections (optional)
    "asof": "asof",                     # A = as-of trade, R = reversal (optional)
    "when_issued": "when_issued",       # optional bool
    "special_price": "special_price",   # optional bool
    "contra": "contra",                 # C customer, D dealer (optional)
}


@dataclass
class TraceCleaningConfig:
    report_lag: pd.Timedelta = pd.Timedelta(minutes=15)
    dissemination_cap_ig: float = 5_000_000.0
    dissemination_cap_hy: float = 1_000_000.0
    enhanced_release_lag: pd.Timedelta = pd.Timedelta(days=182)
    max_abs_return: float = 0.10          # 10% jump vs neighbouring median -> drop
    min_size: float = 0.0                 # drop odd lots below this face amount (e.g. 100k)
    drop_interdealer_duplicates: bool = True


def clean_trace(
    raw: pd.DataFrame,
    columns: dict[str, str] | None = None,
    config: TraceCleaningConfig | None = None,
    is_high_yield: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Return a cleaned, standardized trade frame.

    Output columns: instrument_id, exec_ts, available_ts, price, size, public_size, size_revealed_ts, side.
    """
    cfg = config or TraceCleaningConfig()
    cols = {**STANDARD_COLUMNS, **(columns or {})}
    df = pd.DataFrame({std: raw[src] for std, src in cols.items() if src in raw.columns}).copy()
    required = {"instrument_id", "exec_ts", "price", "size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TRACE frame is missing columns {sorted(missing)}")
    df["exec_ts"] = pd.to_datetime(df["exec_ts"], utc=True)
    if "report_ts" in df:
        df["report_ts"] = pd.to_datetime(df["report_ts"], utc=True)
    else:
        df["report_ts"] = df["exec_ts"] + cfg.report_lag
    df["side"] = df["side"].fillna("D").astype(str).str.upper().str[0] if "side" in df else "D"

    # 1) cancellations / corrections: drop the message and the one it references
    if "status" in df:
        status = df["status"].astype(str).str.upper()
        bad = status.isin({"C", "X", "R"})
        drop_ids = set()
        if "orig_msg_id" in df:
            drop_ids |= set(df.loc[bad, "orig_msg_id"].dropna().astype(str))
        keep = ~bad
        if "msg_id" in df and drop_ids:
            keep &= ~df["msg_id"].astype(str).isin(drop_ids)
        df = df[keep]
    # 2) reversals (as-of code R): drop the reversal and the matching original (same bond/price/size/day)
    if "asof" in df:
        asof = df["asof"].astype(str).str.upper()
        reversals = df[asof == "R"]
        if len(reversals):
            key_cols = ["instrument_id", "price", "size"]
            rev_keys = set(map(tuple, reversals[key_cols + ["exec_ts"]].assign(exec_ts=lambda x: x["exec_ts"].dt.date).itertuples(index=False)))
            day_keys = list(map(tuple, df[key_cols].assign(d=df["exec_ts"].dt.date).itertuples(index=False)))
            df = df[[k not in rev_keys for k in day_keys]]
            df = df[df["asof"].astype(str).str.upper() != "R"]
    # 3) flagged trades
    for flag in ("when_issued", "special_price"):
        if flag in df:
            df = df[~df[flag].fillna(False).astype(bool)]
    # 4) inter-dealer duplicates: both dealers report; keep one per (bond, exec_ts, price, size)
    if cfg.drop_interdealer_duplicates and len(df):
        inter = df["side"] == "D"
        dup = df[inter].duplicated(subset=["instrument_id", "exec_ts", "price", "size"], keep="first")
        df = df[~(inter & dup.reindex(df.index, fill_value=False))]
    # 5) size filter
    if cfg.min_size > 0:
        df = df[df["size"] >= cfg.min_size]
    # 6) outlier / bounce-back filter vs neighbouring prints of the same bond
    df = df.sort_values(["instrument_id", "exec_ts"]).reset_index(drop=True)
    if len(df) > 2:
        med = (
            df.groupby("instrument_id")["price"]
            .transform(lambda s: s.rolling(5, center=True, min_periods=2).median())
        )
        dev = (df["price"] / med - 1.0).abs()
        df = df[~(dev > cfg.max_abs_return).fillna(False)]

    # 7) knowledge-time stamping and dissemination caps
    df["available_ts"] = df["report_ts"].where(df["report_ts"] >= df["exec_ts"], df["exec_ts"] + cfg.report_lag)
    if is_high_yield is None:
        cap = pd.Series(cfg.dissemination_cap_ig, index=df.index)
    else:
        hy = df["instrument_id"].map(is_high_yield if isinstance(is_high_yield, dict) else is_high_yield.to_dict()).fillna(False).astype(bool)
        cap = np.where(hy, cfg.dissemination_cap_hy, cfg.dissemination_cap_ig)
        cap = pd.Series(cap, index=df.index)
    df["public_size"] = np.minimum(df["size"].astype(float), cap.astype(float))
    df["size_revealed_ts"] = np.where(df["size"] > cap, df["available_ts"] + cfg.enhanced_release_lag, df["available_ts"])
    df["size_revealed_ts"] = pd.to_datetime(df["size_revealed_ts"], utc=True)
    out_cols = ["instrument_id", "exec_ts", "available_ts", "price", "size", "public_size", "size_revealed_ts", "side"]
    return df[out_cols].reset_index(drop=True)


def daily_liquidity_stats(trades: pd.DataFrame, dates: pd.DatetimeIndex, volume_window: int = 5, count_window: int = 20, cutoff_time=None) -> pd.DataFrame:
    """Rolling trade statistics per bond-day using only prints *available* by the cutoff.

    With ``cutoff_time`` (e.g. the execution window), prints disseminated after the cutoff on day d
    belong to the next trading day, so a 10:00 decision never sees 10:30 volume. Without it the
    statistics are "by close". Returns volume_5d (public sizes), trade_count_20d and days_since_trade.
    """
    if trades.empty:
        return pd.DataFrame(columns=["date", "instrument_id", "volume_5d", "trade_count_20d", "days_since_trade"])
    t = trades.copy()
    avail = pd.to_datetime(t["available_ts"]).dt.tz_convert("UTC")
    t["date"] = avail.dt.normalize().dt.tz_localize(None)
    idx_all = pd.DatetimeIndex(dates).tz_localize(None) if getattr(dates, "tz", None) is not None else pd.DatetimeIndex(dates)
    if cutoff_time is not None:
        late = avail.dt.time > cutoff_time
        if late.any():
            pos = idx_all.searchsorted(t.loc[late, "date"].values, side="right")
            pos = np.minimum(pos, len(idx_all) - 1)
            t.loc[late, "date"] = idx_all[pos]
    daily = t.groupby(["instrument_id", "date"]).agg(volume=("public_size", "sum"), n=("price", "size")).reset_index()
    idx = pd.DatetimeIndex(dates).tz_localize(None) if getattr(dates, "tz", None) is not None else pd.DatetimeIndex(dates)
    frames = []
    for iid, g in daily.groupby("instrument_id"):
        g = g.set_index("date").reindex(idx).fillna({"volume": 0.0, "n": 0})
        vol = g["volume"].rolling(volume_window, min_periods=1).sum()
        cnt = g["n"].rolling(count_window, min_periods=1).sum()
        traded = g["n"] > 0
        last_trade_pos = pd.Series(np.where(traded, np.arange(len(g)), np.nan), index=g.index).ffill()
        since = pd.Series(np.arange(len(g)), index=g.index) - last_trade_pos
        frames.append(pd.DataFrame({"date": g.index, "instrument_id": iid, "volume_5d": vol.values, "trade_count_20d": cnt.values.astype(int), "days_since_trade": since.fillna(10_000).astype(int).values}))
    return pd.concat(frames, ignore_index=True)


__all__ = ["STANDARD_COLUMNS", "TraceCleaningConfig", "clean_trace", "daily_liquidity_stats"]
