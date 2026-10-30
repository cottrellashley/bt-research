"""Versioned market conventions (blueprint sections 5, 11, 16): dated settlement rules and coupon
entitlement rules by market. Conventions change through time (US corporate settlement moved to T+1
on 28 May 2024), so every rule carries an effective interval and the engine asks for the rule in
force on the trade date instead of applying today's lag to all history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from nautilus_fi.instruments.calendar import BondCalendar, get_calendar
from nautilus_fi.instruments.enums import AssetType

GOVT_TYPES = (AssetType.SOVEREIGN, AssetType.BILL, AssetType.LINKER)
CORP_TYPES = (AssetType.CORPORATE, AssetType.AGENCY, AssetType.SUPRANATIONAL, AssetType.MUNICIPAL, AssetType.COVERED, AssetType.PREFERRED)
SECURITIZED_TYPES = (AssetType.MBS, AssetType.CMBS, AssetType.ABS, AssetType.CLO)


@dataclass(frozen=True)
class SettlementRule:
    market: str
    lag_business_days: int
    asset_types: tuple[AssetType, ...] | None = None
    currency: str | None = None
    effective_from: date = date(1900, 1, 1)
    effective_to: date | None = None
    calendar_name: str = "WEEKENDS"
    source: str = ""
    approximation: bool = False

    def applies(self, asset_type: AssetType, currency: str, trade_date: date) -> bool:
        if self.asset_types is not None and asset_type not in self.asset_types:
            return False
        if self.currency is not None and self.currency != currency:
            return False
        if trade_date < self.effective_from:
            return False
        if self.effective_to is not None and trade_date >= self.effective_to:
            return False
        return True

    @property
    def specificity(self) -> int:
        return (2 if self.asset_types is not None else 0) + (1 if self.currency is not None else 0)


DEFAULT_SETTLEMENT_RULES: list[SettlementRule] = [
    # US corporates, agencies, munis, preferreds: T+3 -> T+2 (5 Sep 2017) -> T+1 (28 May 2024)   [SEC 2024]
    SettlementRule("US", 3, CORP_TYPES, "USD", date(1900, 1, 1), date(2017, 9, 5), "US_BOND", "SEC T+2 rule 2017"),
    SettlementRule("US", 2, CORP_TYPES, "USD", date(2017, 9, 5), date(2024, 5, 28), "US_BOND", "SEC T+2 rule 2017"),
    SettlementRule("US", 1, CORP_TYPES, "USD", date(2024, 5, 28), None, "US_BOND", "SEC T+1 transition 28 May 2024"),
    # US Treasuries and bills: regular way T+1 throughout the relevant history
    SettlementRule("US", 1, GOVT_TYPES, "USD", date(1900, 1, 1), None, "US_BOND", "Treasury regular way T+1"),
    # specified pool MBS / ABS / CMBS / CLO secondary: typically T+3 (TBA class dates handled separately)
    SettlementRule("US", 3, SECURITIZED_TYPES, "USD", date(1900, 1, 1), None, "US_BOND", "secondary securitized convention", approximation=True),
    # leveraged loans: par T+7 target (LSTA); delayed compensation beyond it
    SettlementRule("US", 7, (AssetType.LOAN,), "USD", date(1900, 1, 1), None, "US_BOND", "LSTA par settlement target", approximation=True),
    # Canada mirrored the US move on 27 May 2024
    SettlementRule("CA", 2, None, "CAD", date(2017, 9, 5), date(2024, 5, 27), "WEEKENDS", "CCMA"),
    SettlementRule("CA", 1, None, "CAD", date(2024, 5, 27), None, "WEEKENDS", "CCMA T+1"),
    # Euro area: T+2 for governments and credit (Eurosystem / CSDR)
    SettlementRule("EU", 2, None, "EUR", date(1900, 1, 1), None, "WEEKENDS", "CSDR / ICSD standard"),
    # UK: gilts T+1; sterling corporates T+2
    SettlementRule("UK", 1, GOVT_TYPES, "GBP", date(1900, 1, 1), None, "WEEKENDS", "DMO gilt convention"),
    SettlementRule("UK", 2, None, "GBP", date(1900, 1, 1), None, "WEEKENDS", "sterling credit convention"),
    # Japan: JGB T+2 -> T+1 on 1 May 2018
    SettlementRule("JP", 2, GOVT_TYPES, "JPY", date(1900, 1, 1), date(2018, 5, 1), "WEEKENDS", "JSDA"),
    SettlementRule("JP", 1, GOVT_TYPES, "JPY", date(2018, 5, 1), None, "WEEKENDS", "JSDA T+1 2018"),
    SettlementRule("JP", 2, None, "JPY", date(1900, 1, 1), None, "WEEKENDS", "JSDA corporate"),
    # Australia, Switzerland, Scandinavia, EM Eurobonds (ICSD): T+2
    SettlementRule("AU", 2, None, "AUD", date(1900, 1, 1), None, "WEEKENDS", "ASX/Austraclear"),
    SettlementRule("CH", 2, None, "CHF", date(1900, 1, 1), None, "WEEKENDS", "SIX"),
    SettlementRule("ICSD", 2, None, None, date(1900, 1, 1), None, "WEEKENDS", "Euroclear/Clearstream default", approximation=True),
]


class SettlementConventions:
    def __init__(self, rules: list[SettlementRule] | None = None) -> None:
        self.rules = list(rules or DEFAULT_SETTLEMENT_RULES)

    def rule(self, asset_type: AssetType, currency: str, trade_date: date) -> SettlementRule:
        candidates = [r for r in self.rules if r.applies(asset_type, currency, trade_date)]
        if not candidates:
            return SettlementRule("UNKNOWN", 2, None, None, approximation=True, source="fallback T+2")
        return max(candidates, key=lambda r: (r.specificity, r.effective_from))

    def settle_date(self, asset_type: AssetType, currency: str, trade_date: date, calendar: BondCalendar | None = None, override_lag: int | None = None) -> date:
        r = self.rule(asset_type, currency, trade_date)
        cal = calendar or get_calendar(r.calendar_name)
        return cal.add_business_days(trade_date, override_lag if override_lag is not None else r.lag_business_days)


class EntitlementMode(str, Enum):
    PAYMENT_DATE_HOLDER = "PAYMENT_DATE_HOLDER"    # US corporates / Treasuries: due bills pass the coupon to the settled holder at payment
    RECORD_DATE_HOLDER = "RECORD_DATE_HOLDER"      # registered holder on the record date, no due bills
    EX_DATE_HOLDER = "EX_DATE_HOLDER"              # gilt style ex-coupon period with negative accrued


@dataclass(frozen=True)
class EntitlementRule:
    market: str
    mode: EntitlementMode
    record_lag_calendar_days: int = 15
    ex_coupon_business_days: int = 0
    negative_accrued_in_ex_period: bool = False
    source: str = ""

    def cutoff(self, payment_date: date, calendar: BondCalendar) -> date:
        """Date on which (settled) holdings determine entitlement to the payment."""
        if self.mode == EntitlementMode.PAYMENT_DATE_HOLDER:
            return payment_date
        if self.mode == EntitlementMode.RECORD_DATE_HOLDER:
            from datetime import timedelta

            return payment_date - timedelta(days=self.record_lag_calendar_days)
        return calendar.add_business_days(payment_date, -self.ex_coupon_business_days)

    def in_ex_period(self, settle: date, payment_date: date, calendar: BondCalendar) -> bool:
        return self.mode == EntitlementMode.EX_DATE_HOLDER and self.cutoff(payment_date, calendar) <= settle < payment_date


DEFAULT_ENTITLEMENT_RULES = {
    "US": EntitlementRule("US", EntitlementMode.PAYMENT_DATE_HOLDER, 15, 0, False, "DTC due bills"),
    "CA": EntitlementRule("CA", EntitlementMode.PAYMENT_DATE_HOLDER, 15, 0, False),
    "EU": EntitlementRule("EU", EntitlementMode.PAYMENT_DATE_HOLDER, 1, 0, False, "ICSD record date = payment - 1"),
    "UK_GILT": EntitlementRule("UK_GILT", EntitlementMode.EX_DATE_HOLDER, 7, 7, True, "DMO ex-dividend 7 business days"),
    "UK": EntitlementRule("UK", EntitlementMode.PAYMENT_DATE_HOLDER, 15, 0, False),
    "JP": EntitlementRule("JP", EntitlementMode.RECORD_DATE_HOLDER, 1, 0, False),
    "ICSD": EntitlementRule("ICSD", EntitlementMode.PAYMENT_DATE_HOLDER, 1, 0, False),
}


def entitlement_rule(market: str) -> EntitlementRule:
    return DEFAULT_ENTITLEMENT_RULES.get(market.upper(), DEFAULT_ENTITLEMENT_RULES["ICSD"])


def market_for(asset_type: AssetType, currency: str, country: str = "US") -> str:
    ccy = currency.upper()
    if ccy == "USD" and country.upper() in ("US", ""):
        return "US"
    if ccy == "GBP":
        return "UK_GILT" if asset_type in GOVT_TYPES else "UK"
    if ccy == "EUR":
        return "EU"
    if ccy == "JPY":
        return "JP"
    if ccy == "CAD":
        return "CA"
    return "ICSD"


__all__ = [
    "CORP_TYPES",
    "DEFAULT_ENTITLEMENT_RULES",
    "DEFAULT_SETTLEMENT_RULES",
    "GOVT_TYPES",
    "SECURITIZED_TYPES",
    "EntitlementMode",
    "EntitlementRule",
    "SettlementConventions",
    "SettlementRule",
    "entitlement_rule",
    "market_for",
]
