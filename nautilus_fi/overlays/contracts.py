"""Derivative and overlay contracts traded on a margin venue, outside the cash capital budget.

Two representations cover the fixed income overlay set on Nautilus' margin accounts:

* ``FinancedContract`` -> Nautilus ``PerpetualContract`` with daily funding: a price series plus the
  annual carry a long position earns, settled daily through ``FundingRateUpdate`` (a positive Nautilus
  funding rate debits longs). Index total-return swaps (long receives the index, pays SOFR + spread),
  CDX (long risk = sold protection: price 100 - upfront, receives the running coupon), synthetic bond
  total-return swaps (how shorts are carried: the short receives financing and pays the coupon, plus a
  borrow fee charged by the ledger) and swap-like carry instruments.
* ``DatedContract`` -> Nautilus ``FuturesContract`` with an expiry: Treasury futures and FX forwards,
  closed at the last price on expiry (``InstrumentClose``); rolling is the strategy's decision.

Every contract supplies its exposures per unit notional per date (OAD, OASD, DV01, DTS, key rates)
for hedge sizing and risk. Timing matches the cash venue: window quotes at the *prior* close (no
same-day fills), marks at the mark time, funding rates at publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd


def _series(x, dates: list[date], name: str) -> pd.Series:
    if isinstance(x, (int, float)):
        return pd.Series(float(x), index=pd.to_datetime(dates), name=name)
    s = pd.Series(x, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.reindex(pd.to_datetime(dates)).ffill().rename(name)


@dataclass
class FinancedContract:
    symbol: str
    prices: pd.Series                         # price per unit (date-indexed)
    carry: pd.Series | float = 0.0            # annual carry a LONG earns, as a fraction of price (e.g. -(SOFR + 50bp) for a TRS long)
    kind: str = "TRS"                         # TRS | CDX | BOND_TRS | CARRY
    currency: str = "USD"
    asset_class: str = "DEBT"
    multiplier: float = 1.0
    half_spread_bp: float = 1.0
    margin_init: float = 0.05
    margin_maint: float = 0.03
    price_precision: int = 4
    exposures: pd.DataFrame | None = None     # date x {oad, oasd, dv01, dts, krd_*} per unit notional
    underlying: str = ""
    borrow_fee_bp: float = 0.0                # annual fee a SHORT pays on its notional (charged by the ledger)


@dataclass
class DatedContract:
    symbol: str
    prices: pd.Series
    expiry: date
    kind: str = "FUTURE"                      # FUTURE | FX_FORWARD
    currency: str = "USD"
    asset_class: str = "DEBT"
    multiplier: float = 1.0
    half_spread_bp: float = 0.5
    margin_init: float = 0.03
    margin_maint: float = 0.02
    price_precision: int = 4
    exposures: pd.DataFrame | None = None
    underlying: str = ""
    activation: date | None = None


@dataclass
class OverlayBook:
    contracts: dict[str, FinancedContract | DatedContract] = field(default_factory=dict)

    def add(self, c: FinancedContract | DatedContract) -> FinancedContract | DatedContract:
        self.contracts[c.symbol] = c
        return c

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.contracts

    def price(self, symbol: str, d: date) -> float | None:
        c = self.contracts.get(symbol)
        if c is None:
            return None
        s = c.prices
        ts = pd.Timestamp(d)
        if ts in s.index:
            v = s.loc[ts]
        else:
            prior = s.loc[:ts]
            if prior.empty:
                return None
            v = prior.iloc[-1]
        return float(v) if np.isfinite(v) else None

    def exposure(self, symbol: str, d: date) -> dict[str, float]:
        c = self.contracts.get(symbol)
        if c is None or c.exposures is None or c.exposures.empty:
            return {}
        e = c.exposures.copy()
        e.index = pd.to_datetime(e.index)
        prior = e.loc[: pd.Timestamp(d)]
        return {} if prior.empty else {k: float(v) for k, v in prior.iloc[-1].items() if np.isfinite(v)}

    def exposures_frame(self, d: date) -> pd.DataFrame:
        return pd.DataFrame({s: self.exposure(s, d) for s in self.contracts}).T

    # ------------------------------------------------------------------ Nautilus
    def instruments(self, venue: str, start: date):
        from nautilus_trader.model import (
            AssetClass,
            Currency,
            FuturesContract,
            InstrumentId,
            PerpetualContract,
            Price,
            Quantity,
            Symbol,
            Venue,
        )

        out = []
        for c in self.contracts.values():
            iid = InstrumentId(Symbol(c.symbol), Venue(venue))
            ccy = Currency.from_str(c.currency)
            ac = getattr(AssetClass, c.asset_class.upper())
            tick = Price(10 ** (-c.price_precision), c.price_precision)
            if isinstance(c, FinancedContract):
                out.append(PerpetualContract(instrument_id=iid, raw_symbol=Symbol(c.symbol), underlying=c.underlying or c.symbol, asset_class=ac, quote_currency=ccy, settlement_currency=ccy, is_inverse=False, price_precision=c.price_precision, size_precision=0, price_increment=tick, size_increment=Quantity.from_int(1), multiplier=Quantity.from_str(str(c.multiplier)), lot_size=Quantity.from_int(1), ts_event=0, ts_init=0, margin_init=Decimal(str(c.margin_init)), margin_maint=Decimal(str(c.margin_maint))))
            else:
                act = pd.Timestamp(c.activation or start, tz="UTC").value
                exp = pd.Timestamp(c.expiry, tz="UTC").value + 21 * 3_600_000_000_000
                out.append(FuturesContract(instrument_id=iid, raw_symbol=Symbol(c.symbol), asset_class=ac, underlying=c.underlying or c.symbol, activation_ns=act, expiration_ns=exp, currency=ccy, price_precision=c.price_precision, price_increment=tick, multiplier=Quantity.from_str(str(c.multiplier)), lot_size=Quantity.from_int(1), ts_event=0, ts_init=0, margin_init=Decimal(str(c.margin_init)), margin_maint=Decimal(str(c.margin_maint))))
        return out

    def nautilus_data(self, venue: str, schedule, dates: list[date], start: date | None = None, end: date | None = None) -> list:
        from nautilus_trader.model import (
            FundingRateUpdate,
            InstrumentClose,
            InstrumentCloseType,
            InstrumentId,
            MarkPriceUpdate,
            Price,
            Quantity,
            QuoteTick,
            Symbol,
            Venue,
        )

        out: list = []
        size = Quantity.from_int(10_000_000)
        for c in self.contracts.values():
            iid = InstrumentId(Symbol(c.symbol), Venue(venue))
            p = c.price_precision
            px = _series(c.prices, dates, "px").to_numpy()
            carry = _series(c.carry, dates, "carry").to_numpy() if isinstance(c, FinancedContract) else None
            hs = c.half_spread_bp / 1e4
            for t, d in enumerate(dates):
                if isinstance(c, DatedContract) and d > c.expiry:
                    break
                if (start is not None and d < start) or (end is not None and d > end):
                    continue
                ref = px[t - 1] if t > 0 else np.nan
                if np.isfinite(ref) and ref > 0:
                    w = schedule.window_ns(d)
                    out.append(QuoteTick(iid, Price(round(ref * (1 - hs), p), p), Price(round(ref * (1 + hs), p), p), size, size, w, w))
                if not (np.isfinite(px[t]) and px[t] > 0):
                    continue
                m = schedule.mark_ns(d)
                out.append(MarkPriceUpdate(iid, Price(round(px[t], p), p), m, m))
                if carry is not None and np.isfinite(carry[t]):
                    days = (dates[t + 1] - d).days if t + 1 < len(dates) else 1
                    rate = Decimal(str(round(-carry[t] * days / 360.0, 12)))
                    pub = schedule.publish_ns(d)
                    out.append(FundingRateUpdate(iid, rate, pub, pub, None, pub + 4 * 3_600_000_000_000))
                if isinstance(c, DatedContract) and d == c.expiry:
                    out.append(InstrumentClose(iid, Price(round(px[t], p), p), InstrumentCloseType.CONTRACT_EXPIRED, m + 1, m + 1))
        return out


def index_trs(symbol: str, levels: pd.Series, financing: pd.Series | float, spread_bp: float = 0.0, exposures: pd.DataFrame | None = None, **kw) -> FinancedContract:
    """A total-return swap on an index level: the long earns the index and pays financing + spread."""
    fin = financing if isinstance(financing, (int, float)) else pd.Series(financing, dtype=float)
    carry = -(fin + spread_bp / 1e4)
    return FinancedContract(symbol, levels, carry, "TRS", exposures=exposures, **kw)


def cdx(symbol: str, price: pd.Series, coupon_bp: float = 100.0, exposures: pd.DataFrame | None = None, **kw) -> FinancedContract:
    """Credit index (long risk = protection sold): price 100 - upfront points, the long receives the
    running coupon on notional (carry = coupon / price)."""
    p = pd.Series(price, dtype=float)
    return FinancedContract(symbol, p, (coupon_bp / 1e4) * 100.0 / p, "CDX", exposures=exposures, **kw)


def bond_trs(symbol: str, clean: pd.Series, coupon_rate: float, financing: pd.Series | float, spread_bp: float = 0.0, borrow_fee_bp: float = 25.0, exposures: pd.DataFrame | None = None, suffix: str = "-TRS", **kw) -> FinancedContract:
    """Synthetic total-return swap on a cash bond (how shorts are carried): the long earns the coupon
    accrual and pays financing; the short receives financing, pays the coupon and a borrow fee."""
    p = pd.Series(clean, dtype=float)
    fin = financing if isinstance(financing, (int, float)) else pd.Series(financing, dtype=float).reindex(p.index).ffill()
    carry = coupon_rate * 100.0 / p - (fin + spread_bp / 1e4)
    return FinancedContract(symbol + suffix, p, carry, "BOND_TRS", exposures=exposures, underlying=symbol, borrow_fee_bp=borrow_fee_bp, **kw)


def treasury_future(symbol: str, prices: pd.Series, expiry: date, oad: pd.Series | float | None = None, multiplier: float = 1_000.0, **kw) -> DatedContract:
    """A bond future quoted in points (multiplier 1,000 = $100k face per contract for TY); ``oad`` is
    the duration per unit of notional (the CTD's, or a vendor's futures duration) used for hedging."""
    ex = None
    if oad is not None:
        idx = pd.Series(prices).index
        ex = pd.DataFrame({"oad": pd.Series(oad, index=idx) if isinstance(oad, (int, float)) else pd.Series(oad, dtype=float)})
        ex["dv01"] = ex["oad"] * 1e-4
    return DatedContract(symbol, prices, expiry, "FUTURE", multiplier=multiplier, exposures=ex, **kw)


def fx_forward(pair: str, forwards: pd.Series, value_date: date, **kw) -> DatedContract:
    """An outright FX forward (``EURUSD`` long = buy EUR forward), cash-settled at the last price."""
    return DatedContract(f"{pair}-{value_date:%Y%m%d}", forwards, value_date, "FX_FORWARD", asset_class="FX", underlying=pair, price_precision=5, **kw)


def swap_contract(swap, curves: dict, fixings: dict | None = None, **kw) -> FinancedContract:
    """An interest rate swap (``instruments.derivatives.InterestRateSwap``) on the margin venue: one
    unit per 100 of notional priced at 100 + NPV per 100 (our side, from the daily discount / projection
    curves), carrying the net coupon accrual (floating fixing vs fixed rate) daily; exposures carry the
    swap's DV01 and its duration equivalent per unit of notional."""
    days = sorted(curves)
    px, carry, dv01 = {}, {}, {}
    for d in days:
        fix = (fixings or {}).get(d)
        v = swap.value(d, curves[d], curves[d], fix)
        price = 100.0 + v["npv"] / swap.notional * 100.0
        flt = fix if fix is not None else curves[d].zero(0.25)
        net = (flt + swap.float_spread_bp / 1e4 - swap.fixed_rate) if swap.pay_fixed else (swap.fixed_rate - flt - swap.float_spread_bp / 1e4)
        ts = pd.Timestamp(d)
        px[ts], carry[ts], dv01[ts] = price, net * 100.0 / price, v["dv01"] / swap.notional
    idx = pd.DatetimeIndex(list(px))
    ex = pd.DataFrame({"dv01": pd.Series(dv01), "oad": pd.Series(dv01) / 1e-4}, index=idx)
    return FinancedContract(swap.symbol, pd.Series(px), pd.Series(carry), "CARRY", currency=swap.currency, exposures=ex, underlying=swap.float_index, **kw)


__all__ = ["DatedContract", "FinancedContract", "OverlayBook", "bond_trs", "cdx", "fx_forward", "index_trs", "swap_contract", "treasury_future"]
