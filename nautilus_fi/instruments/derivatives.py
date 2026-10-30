"""Overlay derivatives (blueprint section 7): Treasury futures (conversion factors, cheapest to
deliver, daily settlement), interest rate swaps (projection and discount curves, resets), credit
default swaps (premium and protection legs, credit events, index factors) and FX forwards.
Valuation functions return values with diagnostics; margin and collateral live in ``funding``."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from nautilus_fi.instruments.calendar import BondCalendar, get_calendar
from nautilus_fi.instruments.curves import ZeroCurve, _bisect
from nautilus_fi.instruments.daycount import year_fraction
from nautilus_fi.instruments.enums import BusinessDayConvention, DayCount, Frequency
from nautilus_fi.instruments.schedule import generate_schedule


def _t(as_of: date, d: date) -> float:
    return max((d - as_of).days, 0) / 365.25


# ------------------------------------------------------------------------------------------ futures
@dataclass
class TreasuryFuture:
    symbol: str
    expiry: date
    delivery_last: date
    ctd_symbol: str
    conversion_factor: float
    contract_size: float = 100_000.0
    tick: float = 1.0 / 32.0
    currency: str = "USD"

    def invoice_price(self, futures_price: float, ctd_accrued_per_100: float) -> float:
        return futures_price * self.conversion_factor + ctd_accrued_per_100

    def ctd_clean_implied(self, futures_price: float) -> float:
        return futures_price * self.conversion_factor

    def gross_basis(self, ctd_clean: float, futures_price: float) -> float:
        return ctd_clean - futures_price * self.conversion_factor

    def contract_dv01(self, ctd_dv01_per_100: float) -> float:
        """Currency DV01 per contract from the CTD's DV01 per 100 face."""
        return ctd_dv01_per_100 / self.conversion_factor * self.contract_size / 100.0

    def variation_margin(self, prev_settle: float, settle: float, contracts: float) -> float:
        """Daily settled P&L (positive = we receive) for a long of ``contracts``."""
        return (settle - prev_settle) * self.contract_size / 100.0 * contracts

    def implied_repo(self, ctd_clean: float, futures_price: float, accrued_now: float, accrued_delivery: float, days: int, coupon_income: float = 0.0) -> float:
        invoice = self.invoice_price(futures_price, accrued_delivery)
        cost = ctd_clean + accrued_now
        return (invoice + coupon_income - cost) / cost * 360.0 / max(days, 1)

    def cheapest_to_deliver(self, basket: list[tuple[str, float, float, float]], futures_price: float, accrued_now: dict[str, float], accrued_delivery: dict[str, float], days: int) -> tuple[str, float]:
        """``basket`` rows: (symbol, clean price, conversion factor, coupon income). Returns the highest implied repo."""
        best = None
        for sym, px, cf, income in basket:
            fut = TreasuryFuture(self.symbol, self.expiry, self.delivery_last, sym, cf, self.contract_size)
            irr = fut.implied_repo(px, futures_price, accrued_now.get(sym, 0.0), accrued_delivery.get(sym, 0.0), days, income)
            if best is None or irr > best[1]:
                best = (sym, irr)
        return best


# ----------------------------------------------------------------------------------------------- IRS
@dataclass
class InterestRateSwap:
    symbol: str
    notional: float
    currency: str
    start: date
    end: date
    fixed_rate: float
    pay_fixed: bool = True
    fixed_frequency: Frequency = Frequency.ANNUAL
    fixed_day_count: DayCount = DayCount.ACT_360
    float_index: str = "SOFR"
    float_spread_bp: float = 0.0
    float_frequency: Frequency = Frequency.ANNUAL
    float_day_count: DayCount = DayCount.ACT_360
    discount_curve: str = "USD-OIS"
    projection_curve: str = "USD-OIS"
    netting_set: str = ""
    calendar_name: str = "US_BOND"

    def _cal(self) -> BondCalendar:
        return get_calendar(self.calendar_name)

    def fixed_periods(self):
        return generate_schedule(self.start, self.end, self.fixed_frequency, calendar=self._cal(), convention=BusinessDayConvention.MODIFIED_FOLLOWING)

    def float_periods(self):
        return generate_schedule(self.start, self.end, self.float_frequency, calendar=self._cal(), convention=BusinessDayConvention.MODIFIED_FOLLOWING)

    def value(self, as_of: date, discount: ZeroCurve, projection: ZeroCurve | None = None, current_fixing: float | None = None) -> dict:
        """NPV from our perspective, PVs of each leg, par rate, DV01 and accrued (currency)."""
        projection = projection or discount
        pv_fixed = annuity = 0.0
        for p in self.fixed_periods():
            if p.payment_date <= as_of:
                continue
            tau = year_fraction(p.accrual_start, p.accrual_end, self.fixed_day_count)
            df = discount.df(_t(as_of, p.payment_date))
            annuity += tau * df
            pv_fixed += self.notional * self.fixed_rate * tau * df
        pv_float = pv_float_no_spread = 0.0
        for p in self.float_periods():
            if p.payment_date <= as_of:
                continue
            tau = year_fraction(p.accrual_start, p.accrual_end, self.float_day_count)
            df = discount.df(_t(as_of, p.payment_date))
            if p.accrual_start <= as_of and current_fixing is not None:
                fwd = current_fixing
            else:
                t1, t2 = _t(as_of, max(p.accrual_start, as_of)), _t(as_of, p.accrual_end)
                fwd = projection.forward(t1, t2) if t2 > t1 else projection.zero(t2)
                # convert continuous forward to a simple rate over the period
                fwd = (np.exp(fwd * max(t2 - t1, 1e-9)) - 1.0) / max(t2 - t1, 1e-9) if t2 > t1 else fwd
            pv_float_no_spread += self.notional * fwd * tau * df
            pv_float += self.notional * (fwd + self.float_spread_bp / 1e4) * tau * df
        npv = (pv_float - pv_fixed) if self.pay_fixed else (pv_fixed - pv_float)
        par = pv_float_no_spread / (self.notional * annuity) if annuity else float("nan")
        bumped = self._bumped_npv(as_of, discount, projection, current_fixing)
        return {"npv": npv, "pv_fixed": pv_fixed, "pv_float": pv_float, "par_rate": par, "annuity": annuity, "dv01": (bumped - npv), "sign": "pay_fixed" if self.pay_fixed else "receive_fixed"}

    def _bumped_npv(self, as_of, discount, projection, current_fixing) -> float:
        d2, p2 = discount.shifted(1.0), (projection or discount).shifted(1.0)
        pv_fixed = pv_float = 0.0
        for p in self.fixed_periods():
            if p.payment_date <= as_of:
                continue
            pv_fixed += self.notional * self.fixed_rate * year_fraction(p.accrual_start, p.accrual_end, self.fixed_day_count) * d2.df(_t(as_of, p.payment_date))
        for p in self.float_periods():
            if p.payment_date <= as_of:
                continue
            tau = year_fraction(p.accrual_start, p.accrual_end, self.float_day_count)
            if p.accrual_start <= as_of and current_fixing is not None:
                fwd = current_fixing
            else:
                t1, t2 = _t(as_of, max(p.accrual_start, as_of)), _t(as_of, p.accrual_end)
                fwd = (np.exp(p2.forward(t1, t2) * (t2 - t1)) - 1.0) / (t2 - t1) if t2 > t1 else p2.zero(t2)
            pv_float += self.notional * (fwd + self.float_spread_bp / 1e4) * tau * d2.df(_t(as_of, p.payment_date))
        return (pv_float - pv_fixed) if self.pay_fixed else (pv_fixed - pv_float)


# ----------------------------------------------------------------------------------------------- CDS
def hazard_from_spread(spread_bp: float, recovery: float, discount: ZeroCurve | None = None, maturity_years: float = 5.0) -> float:
    """Flat hazard rate consistent with a par spread (solved, not the credit-triangle shortcut)."""
    target = spread_bp / 1e4
    if discount is None:
        return target / max(1.0 - recovery, 1e-6)

    def f(h: float) -> float:
        return _par_spread_flat(h, recovery, discount, maturity_years) - target

    return _bisect(f, 1e-6, 5.0, tol=1e-10)


def _legs_flat(h: float, recovery: float, discount: ZeroCurve, maturity_years: float, steps_per_year: int = 4) -> tuple[float, float, float]:
    n = max(int(round(maturity_years * steps_per_year)), 1)
    dt = maturity_years / n
    prot = prem = annuity = 0.0
    s_prev = 1.0
    for i in range(1, n + 1):
        t = i * dt
        s = np.exp(-h * t)
        df = discount.df(t)
        prot += df * (s_prev - s) * (1.0 - recovery)
        annuity += dt * df * s + 0.5 * dt * df * (s_prev - s)   # accrual on default
        s_prev = s
    prem = annuity
    return prot, prem, annuity


def _par_spread_flat(h: float, recovery: float, discount: ZeroCurve, maturity_years: float) -> float:
    prot, _, annuity = _legs_flat(h, recovery, discount, maturity_years)
    return prot / annuity if annuity else 0.0


@dataclass
class CreditDefaultSwap:
    symbol: str
    reference_entity: str
    notional: float
    currency: str
    effective: date
    maturity: date
    running_spread_bp: float = 100.0
    recovery: float = 0.40
    buy_protection: bool = True
    upfront_pct: float = 0.0                     # points upfront paid (received if negative) at inception
    index_series: str | None = None
    constituents: int | None = None
    factor: float = 1.0                          # index notional factor after credit events
    netting_set: str = ""
    terminated: bool = False

    def value(self, as_of: date, discount: ZeroCurve, par_spread_bp: float | None = None, hazard: float | None = None) -> dict:
        """NPV from our perspective given the market par spread (or a hazard rate)."""
        T = _t(as_of, self.maturity)
        if T <= 0 or self.terminated:
            return {"npv": 0.0, "protection_pv": 0.0, "premium_pv": 0.0, "par_spread_bp": par_spread_bp or 0.0, "cs01": 0.0, "accrued_premium": 0.0}
        h = hazard if hazard is not None else hazard_from_spread(par_spread_bp if par_spread_bp is not None else self.running_spread_bp, self.recovery, discount, T)
        prot, prem_annuity, annuity = _legs_flat(h, self.recovery, discount, T)
        notional = self.notional * self.factor
        protection_pv = notional * prot
        premium_pv = notional * self.running_spread_bp / 1e4 * prem_annuity
        npv_buyer = protection_pv - premium_pv
        npv = npv_buyer if self.buy_protection else -npv_buyer
        # CS01: 1bp wider par spread
        h2 = hazard_from_spread((par_spread_bp if par_spread_bp is not None else self.running_spread_bp) + 1.0, self.recovery, discount, T)
        prot2, prem2, _ = _legs_flat(h2, self.recovery, discount, T)
        npv2_buyer = notional * (prot2 - self.running_spread_bp / 1e4 * prem2)
        cs01 = (npv2_buyer - npv_buyer) * (1 if self.buy_protection else -1)
        # accrued premium since last IMM date (20 Mar/Jun/Sep/Dec)
        accrued = notional * self.running_spread_bp / 1e4 * _days_since_imm(as_of) / 360.0
        return {"npv": npv, "protection_pv": protection_pv, "premium_pv": premium_pv, "par_spread_bp": _par_spread_flat(h, self.recovery, discount, T) * 1e4, "cs01": cs01, "accrued_premium": accrued, "hazard": h}

    def credit_event(self, auction_recovery: float, event_date: date, weight: float | None = None) -> dict:
        """Protection payout (buyer receives) and accrued premium owed; index deals reduce the factor."""
        w = weight if weight is not None else (1.0 / self.constituents if self.constituents else 1.0)
        affected = self.notional * self.factor * w if self.index_series else self.notional * self.factor
        payout = affected * (1.0 - auction_recovery)
        accrued = affected * self.running_spread_bp / 1e4 * _days_since_imm(event_date) / 360.0
        if self.index_series:
            self.factor *= 1.0 - w
        else:
            self.terminated = True
        sign = 1.0 if self.buy_protection else -1.0
        return {"payout_to_us": sign * payout, "accrued_premium_we_pay": sign * accrued, "affected_notional": affected, "remaining_factor": self.factor}


def _days_since_imm(d: date) -> int:
    imm_months = (3, 6, 9, 12)
    candidates = [date(d.year, m, 20) for m in imm_months] + [date(d.year - 1, 12, 20)]
    last = max(c for c in candidates if c <= d)
    return (d - last).days


# ------------------------------------------------------------------------------------------ FX forward
@dataclass
class FXForward:
    symbol: str
    pair: str                        # e.g. "EURUSD": base EUR, quote USD
    notional_base: float             # > 0 buy base currency forward
    forward_rate: float
    settlement: date
    netting_set: str = ""

    @property
    def base(self) -> str:
        return self.pair[:3]

    @property
    def quote(self) -> str:
        return self.pair[3:]

    def value(self, as_of: date, market_forward: float, discount_quote: ZeroCurve) -> dict:
        """NPV in the quote currency: notional x (market forward - contract rate) discounted."""
        T = _t(as_of, self.settlement)
        df = discount_quote.df(T) if T > 0 else 1.0
        npv = self.notional_base * (market_forward - self.forward_rate) * df
        return {"npv_quote": npv, "df": df, "T": T}

    def settlement_cashflows(self, spot_at_settlement: float, deliverable: bool = True) -> dict[str, float]:
        if deliverable:
            return {self.base: self.notional_base, self.quote: -self.notional_base * self.forward_rate}
        return {self.quote: self.notional_base * (spot_at_settlement - self.forward_rate)}

    def roll(self, as_of: date, market_forward_close: float, new_settlement: date, new_forward: float, discount_quote: ZeroCurve) -> tuple[float, FXForward]:
        """Close at the market forward and open the next contract; returns (realized close value, new contract)."""
        close_value = self.value(as_of, market_forward_close, discount_quote)["npv_quote"]
        return close_value, FXForward(f"{self.symbol}-R", self.pair, self.notional_base, new_forward, new_settlement, self.netting_set)


__all__ = ["CreditDefaultSwap", "FXForward", "InterestRateSwap", "TreasuryFuture", "hazard_from_spread"]
_ = field
