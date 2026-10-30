"""Money market instruments and fund accounting: discount-yield quotation, money-market and
bond-equivalent yields, CDs with interest at maturity, amortized cost with shadow NAV deviation
and liquidity-fee mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.enums import AssetType, CouponType, DayCount, Frequency


def discount_yield_to_price(dy: float, days: int, basis: int = 360) -> float:
    return 100.0 * (1.0 - dy * days / basis)


def price_to_discount_yield(price: float, days: int, basis: int = 360) -> float:
    return (100.0 - price) / 100.0 * basis / days


def money_market_yield(price: float, days: int, basis: int = 360) -> float:
    return (100.0 / price - 1.0) * basis / days


def bond_equivalent_yield(price: float, days: int, year_days: int = 365) -> float:
    return (100.0 / price - 1.0) * year_days / days


def bill(symbol: str, issue: date, maturity: date, issuer: str = "UST", basis: DayCount = DayCount.ACT_360) -> BondSpec:
    return BondSpec(symbol, issuer, issue, maturity, 0.0, coupon_type=CouponType.ZERO, asset_type=AssetType.BILL, day_count=basis, min_piece=100, increment=100)


def commercial_paper(symbol: str, issuer: str, issue: date, maturity: date) -> BondSpec:
    return BondSpec(symbol, issuer, issue, maturity, 0.0, coupon_type=CouponType.ZERO, asset_type=AssetType.BILL, day_count=DayCount.ACT_360, min_piece=250_000, increment=1_000, sector="CP")


def certificate_of_deposit(symbol: str, issuer: str, issue: date, maturity: date, rate: float) -> BondSpec:
    """Interest-bearing CD: single coupon at maturity on an ACT/360 basis."""
    return BondSpec(symbol, issuer, issue, maturity, rate, coupon_type=CouponType.FIXED, frequency=Frequency.ANNUAL, asset_type=AssetType.BILL, day_count=DayCount.ACT_360, min_piece=250_000, increment=1_000, sector="CD")


@dataclass
class AmortizedCostPolicy:
    enabled: bool = True
    shadow_deviation_limit: float = 0.005        # 50 bp deviation between amortized cost and market NAV
    max_maturity_days: int = 397

    def amortized_price(self, purchase_price: float, purchase_date: date, maturity: date, as_of: date, redemption: float = 100.0) -> float:
        total = max((maturity - purchase_date).days, 1)
        elapsed = min(max((as_of - purchase_date).days, 0), total)
        return purchase_price + (redemption - purchase_price) * elapsed / total

    def shadow_deviation(self, market_nav: float, amortized_nav: float) -> float:
        return market_nav / amortized_nav - 1.0 if amortized_nav else 0.0

    def breaches(self, market_nav: float, amortized_nav: float) -> bool:
        return abs(self.shadow_deviation(market_nav, amortized_nav)) > self.shadow_deviation_limit


def liquidity_fee(net_redemption_pct: float, liquidity_cost_bp: float, threshold_pct: float = 0.05, de_minimis_bp: float = 1.0) -> float:
    """SEC 2a-7 (2023) style mandatory liquidity fee for institutional prime funds: when daily net
    redemptions exceed the threshold the fee equals the estimated liquidity cost unless it is de minimis."""
    if net_redemption_pct <= threshold_pct or liquidity_cost_bp < de_minimis_bp:
        return 0.0
    return liquidity_cost_bp / 1e4


__all__ = ["AmortizedCostPolicy", "bill", "bond_equivalent_yield", "certificate_of_deposit", "commercial_paper", "discount_yield_to_price", "liquidity_fee", "money_market_yield", "price_to_discount_yield"]
