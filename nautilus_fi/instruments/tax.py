"""Optional investor-specific tax module: withholding, municipal exemption, transaction taxes."""

from __future__ import annotations

from dataclasses import dataclass, field

from nautilus_fi.instruments.enums import AssetType


@dataclass
class TaxProfile:
    jurisdiction: str = "US"
    investor_type: str = "TAX_EXEMPT"           # US_INDIVIDUAL | US_CORP | TAX_EXEMPT | FOREIGN
    ordinary_rate: float = 0.0
    capital_gains_rate: float = 0.0
    state_rate: float = 0.0
    muni_exempt_federal: bool = True
    muni_exempt_state: bool = False
    withholding: dict[str, float] = field(default_factory=dict)     # issuer country -> rate on coupons
    treaty_rates: dict[str, float] = field(default_factory=dict)
    transaction_taxes: dict[str, float] = field(default_factory=dict)  # market -> rate on notional

    def coupon_tax_rate(self, asset_type: AssetType, issuer_country: str = "US") -> float:
        if self.investor_type == "TAX_EXEMPT":
            return 0.0
        if asset_type == AssetType.MUNICIPAL and self.jurisdiction == "US":
            fed = 0.0 if self.muni_exempt_federal else self.ordinary_rate
            st = 0.0 if self.muni_exempt_state else self.state_rate
            return fed + st
        return self.ordinary_rate + self.state_rate

    def withholding_rate(self, issuer_country: str) -> float:
        if issuer_country in self.treaty_rates:
            return self.treaty_rates[issuer_country]
        return self.withholding.get(issuer_country, 0.0)

    def after_tax_coupon(self, coupon: float, asset_type: AssetType, issuer_country: str = "US") -> float:
        return coupon * (1.0 - self.coupon_tax_rate(asset_type, issuer_country)) * (1.0 - self.withholding_rate(issuer_country))

    def taxable_equivalent_yield(self, muni_yield: float) -> float:
        t = self.ordinary_rate + (0.0 if self.muni_exempt_state else self.state_rate)
        return muni_yield / (1.0 - t) if t < 1 else float("inf")

    def transaction_tax(self, notional: float, market: str) -> float:
        return notional * self.transaction_taxes.get(market, 0.0)


__all__ = ["TaxProfile"]
