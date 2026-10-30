"""Point-in-time custom data types carried through the Nautilus message bus.

Every type carries two timestamps (Nautilus convention):

* ``ts_event``  when the thing happened (trade execution time, evaluated price snapshot time)
* ``ts_init``   when it became **knowable** (TRACE dissemination = execution + 15 minutes, vendor
  publication time, rating announcement). The backtest engine replays by ``ts_init``, so a strategy
  can never see a datum before the market did.
"""

from __future__ import annotations

from nautilus_trader.model import CustomData, DataType, InstrumentId, register_custom_data_class
from nautilus_trader.model.custom import customdataclass


@customdataclass
class BondMark:
    """Daily mark from the pricing waterfall (clean prices per 100 current face)."""

    instrument_id: InstrumentId
    clean_mid: float
    clean_bid: float
    clean_ask: float
    factor: float = 1.0          # factor / index ratio applying on the mark date
    accrued: float = 0.0         # per 100 current face, for the settle date of a trade on the mark date
    tier: str = "NONE"           # MarkTier value
    quality: float = 0.0         # 0..1 confidence (e.g. BVAL score / 10)
    source: str = ""
    ytm: float = float("nan")
    mod_duration: float = float("nan")
    spread_duration: float = float("nan")
    z_spread_bp: float = float("nan")
    stale_days: int = 0

    @property
    def dirty_mid(self) -> float:
        return self.clean_mid + self.accrued


@customdataclass
class BondTrade:
    """A cleaned transaction print (TRACE-style)."""

    instrument_id: InstrumentId
    price: float
    size: float                  # face amount actually traded (may be hidden by the cap at ts_init)
    public_size: float           # face amount visible when disseminated (capped)
    side: str = "D"              # B = customer buy, S = customer sell, D = dealer-dealer
    source: str = "TRACE"
    report_lag_ns: int = 0


@customdataclass
class RatingUpdate:
    instrument_id: InstrumentId
    agency: str
    rating: str
    score: int                   # 1 = AAA ... 22 = D


@customdataclass
class IndexFixing:
    index_name: str
    rate: float                  # annual decimal


@customdataclass
class FactorUpdate:
    instrument_id: InstrumentId
    factor: float
    paydown_price: float = 100.0


@customdataclass
class CorporateAction:
    """Calls, puts, tenders, defaults, recoveries, exchange offers (identifier changes)."""

    instrument_id: InstrumentId
    action: str                  # CALL | PUT | TENDER | DEFAULT | RECOVERY | MATURITY | EXCHANGE
    effective_date: str          # ISO date the action takes economic effect
    price: float = float("nan")  # redemption / recovery price per 100 current face
    note: str = ""


@customdataclass
class LiquiditySnapshot:
    """What the execution simulator believed about a bond on a trading day."""

    instrument_id: InstrumentId
    volume_5d: float
    trade_count_20d: int
    days_since_trade: int
    quote_score: float
    tradeable_bid_face: float    # face the simulator would let you sell today
    tradeable_ask_face: float    # face the simulator would let you buy today
    half_spread_bp: float        # indicative half spread for a normal institutional size
    p_fill_bid: float
    p_fill_ask: float


@customdataclass
class StressIndex:
    name: str
    value: float
    regime: str = "NORMAL"       # NORMAL | ELEVATED | STRESS


@customdataclass
class PortfolioSnapshot:
    """Ledger NAV and risk snapshot published once per marking time."""

    nav: float
    cash: float
    market_value: float
    accrued: float
    unsettled_cash: float
    duration: float
    spread_duration: float
    dts: float
    n_positions: int
    day_return: float


ALL_TYPES = [BondMark, BondTrade, RatingUpdate, IndexFixing, FactorUpdate, CorporateAction, LiquiditySnapshot, StressIndex, PortfolioSnapshot]
_REGISTERED = False


def register_types() -> None:
    """Register all custom data classes with Nautilus (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    for cls in ALL_TYPES:
        register_custom_data_class(cls)
    _REGISTERED = True


def data_type_of(cls_or_obj) -> DataType:
    cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
    return DataType(cls.__name__)


def wrap(obj) -> CustomData:
    """Wrap a custom data object for ``engine.add_data`` / ``publish_data``."""
    return CustomData(data_type_of(obj), obj)


__all__ = [
    "ALL_TYPES",
    "BondMark",
    "BondTrade",
    "CorporateAction",
    "FactorUpdate",
    "IndexFixing",
    "LiquiditySnapshot",
    "PortfolioSnapshot",
    "RatingUpdate",
    "StressIndex",
    "data_type_of",
    "register_types",
    "wrap",
]
