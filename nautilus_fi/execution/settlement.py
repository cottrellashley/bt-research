"""Settlement conventions by asset type and currency.

US corporates and Treasuries settle T+1 (since May 2024); many non-US bonds remain T+2; leveraged
loans settle T+7 or later; specified-pool MBS and other securitized paper typically T+3 (TBA monthly
SIFMA class dates are out of scope and should be supplied explicitly through ``settlement_lag_days``).
"""

from __future__ import annotations

from datetime import date

from nautilus_fi.instruments.enums import AssetType

_USD_LAGS = {
    AssetType.SOVEREIGN: 1, AssetType.BILL: 1, AssetType.AGENCY: 1, AssetType.SUPRANATIONAL: 1,
    AssetType.CORPORATE: 1, AssetType.MUNICIPAL: 1, AssetType.COVERED: 1, AssetType.PREFERRED: 1,
    AssetType.LINKER: 1, AssetType.LOAN: 7, AssetType.MBS: 3, AssetType.CMBS: 3, AssetType.ABS: 3,
    AssetType.CLO: 3, AssetType.OTHER: 2,
}
_T1_SOVEREIGN_CCY = {"USD", "CAD", "GBP", "JPY", "MXN", "ARS"}


def default_settlement_lag(asset_type: AssetType, currency: str = "USD") -> int:
    ccy = (currency or "USD").upper()
    if ccy in {"USD", "CAD"}:
        return _USD_LAGS.get(asset_type, 2)
    if asset_type in (AssetType.SOVEREIGN, AssetType.BILL) and ccy in _T1_SOVEREIGN_CCY:
        return 1
    if asset_type == AssetType.LOAN:
        return 10
    if asset_type.is_securitized:
        return 3
    return 2


def settle_date(bond, trade_date: date) -> date:
    return bond.settle_date(trade_date)


__all__ = ["default_settlement_lag", "settle_date"]
