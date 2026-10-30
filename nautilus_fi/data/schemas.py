"""Input data contracts (pandas column schemas) and validation.

All vendor-specific adapters (BVAL, ICE, TRACE, Intex, Mergent, Bloomberg BQL ...) should produce
frames in these shapes; everything downstream is vendor neutral. Every price-like dataset carries a
``publish_ts`` (knowledge time). When a vendor does not give one, stamp it conservatively (e.g. ICE
end-of-day evaluations at 16:00 ET + publication delay).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Schema:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    description: str = ""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: missing required columns {missing}; have {list(df.columns)}")
        return df


REFERENCE = Schema(
    "reference",
    required=("symbol", "issuer", "issue_date", "maturity", "coupon_rate"),
    optional=(
        "coupon_type", "frequency", "day_count", "currency", "asset_type", "principal_type", "name", "sector",
        "country", "seniority", "rating_initial", "dated_date", "first_coupon_date", "min_piece", "increment",
        "amount_outstanding", "is_144a", "is_benchmark", "float_index", "float_spread_bp", "float_reset_lag_days",
        "call_schedule", "put_schedule", "sinking_schedule", "original_term_months", "projected_cpr",
        "settlement_lag_days", "calendar_name", "cusip", "isin", "figi", "bbg_id", "ticker",
    ),
    description="Bond terms (Mergent FISD / Bloomberg DES style). maturity empty for perpetuals.",
)
EVALUATED_PRICES = Schema(
    "evaluated_prices",
    required=("date", "instrument_id", "mid"),
    optional=("bid", "ask", "score", "source", "publish_ts"),
    description="Vendor evaluated prices (BVAL, ICE, LSEG). score: vendor quality (BVAL 1-10).",
)
COMPOSITE_QUOTES = Schema(
    "composite_quotes",
    required=("ts", "instrument_id", "bid", "ask"),
    optional=("bid_size", "ask_size", "source", "publish_ts"),
    description="Composite / executable quotes (CP+, Ai-Price, CEP, dealer runs).",
)
TRADES = Schema(
    "trades",
    required=("instrument_id", "exec_ts", "available_ts", "price", "size", "public_size", "side"),
    optional=("size_revealed_ts",),
    description="Cleaned transactions (see nautilus_fi.data.trace.clean_trace).",
)
CURVES = Schema(
    "curves",
    required=("date", "curve", "tenor", "rate"),
    optional=("kind", "publish_ts"),
    description="Par yields (kind='par') or zero rates (kind='zero') per date and tenor (years).",
)
FIXINGS = Schema("fixings", required=("date", "index", "rate"), optional=("publish_ts",))
FACTORS = Schema("factors", required=("date", "instrument_id", "factor"), optional=("paydown_price", "publish_ts"))
RATINGS = Schema("ratings", required=("date", "instrument_id", "rating"), optional=("agency", "publish_ts"))
CORPORATE_ACTIONS = Schema(
    "corporate_actions",
    required=("instrument_id", "action", "effective_date"),
    optional=("price", "announce_ts", "note"),
    description="CALL / PUT / TENDER / DEFAULT / RECOVERY / EXCHANGE with effective dates and prices.",
)
STRESS = Schema("stress", required=("date", "value"), optional=("name", "publish_ts"))
BENCHMARK = Schema("benchmark", required=("date", "total_return"), optional=("name", "duration", "oas"))
INDEX_RATIOS = Schema("index_ratios", required=("date", "instrument_id", "index_ratio"), optional=("publish_ts",))

ALL_SCHEMAS = {
    s.name: s
    for s in (REFERENCE, EVALUATED_PRICES, COMPOSITE_QUOTES, TRADES, CURVES, FIXINGS, FACTORS, RATINGS, CORPORATE_ACTIONS, STRESS, BENCHMARK, INDEX_RATIOS)
}


def validate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    return ALL_SCHEMAS[name].validate(df)


__all__ = ["ALL_SCHEMAS", "BENCHMARK", "COMPOSITE_QUOTES", "CORPORATE_ACTIONS", "CURVES", "EVALUATED_PRICES", "FACTORS", "FIXINGS", "INDEX_RATIOS", "RATINGS", "REFERENCE", "STRESS", "TRADES", "Schema", "validate"]
