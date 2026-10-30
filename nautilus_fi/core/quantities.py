"""Typed quantities (blueprint section 5: "give every quantity a meaning").

A price is not a number: it has a quotation basis (percent of par, decimal, discount yield, yield,
spread, futures points, upfront), a clean/dirty kind, a principal basis (current, original or indexed
face), a currency, a side and a time. A notional is original, current, indexed, committed, drawn or a
reference amount. Conversions are explicit and need their inputs; mismatches raise ``UnitError``
before any portfolio calculation runs.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

CURRENCY_PRECISION = {"JPY": 0, "KRW": 0, "HUF": 2, "USD": 2, "EUR": 2, "GBP": 2, "CHF": 2, "AUD": 2, "CAD": 2, "BRL": 2, "MXN": 2, "ZAR": 2, "CNY": 2, "INR": 2}


class UnitError(ValueError):
    """A quantity was combined with an incompatible unit, basis or currency."""


class QuoteBasis(str, Enum):
    PCT_OF_PAR = "PCT_OF_PAR"              # 98.50 per 100 of principal
    DECIMAL = "DECIMAL"                    # 0.9850 per 1 of principal
    DISCOUNT_YIELD = "DISCOUNT_YIELD"      # bills / CP: annualised discount on an ACT/360 (or ACT/365) basis
    YIELD = "YIELD"                        # quoted by yield; needs the instrument's price function
    SPREAD_BP = "SPREAD_BP"                # quoted as a spread to a named reference
    DISCOUNT_MARGIN_BP = "DISCOUNT_MARGIN_BP"
    POINTS = "POINTS"                      # futures / options points (decimalised)
    UPFRONT_PCT = "UPFRONT_PCT"            # CDS points upfront (% of notional, protection buyer pays)
    NAV_PER_SHARE = "NAV_PER_SHARE"
    PRICE_PER_SHARE = "PRICE_PER_SHARE"


class PriceKind(str, Enum):
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    NOT_APPLICABLE = "NA"


class PrincipalBasis(str, Enum):
    CURRENT = "CURRENT"                    # per 100 of current (factor-adjusted) principal: the bond market default
    ORIGINAL = "ORIGINAL"                  # per 100 of original principal (already factor-adjusted price)
    INDEXED = "INDEXED"                    # linker real price per 100 of indexed principal


class Side(str, Enum):
    BID = "BID"
    MID = "MID"
    ASK = "ASK"
    TRADE = "TRADE"
    EVAL = "EVAL"
    MODEL = "MODEL"
    NA = "NA"


class Compounding(str, Enum):
    SIMPLE = "SIMPLE"
    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"
    MONTHLY = "MONTHLY"
    CONTINUOUS = "CONTINUOUS"

    @property
    def periods(self) -> int:
        return {"ANNUAL": 1, "SEMI_ANNUAL": 2, "QUARTERLY": 4, "MONTHLY": 12}.get(self.value, 0)


class NotionalKind(str, Enum):
    ORIGINAL = "ORIGINAL"
    CURRENT = "CURRENT"
    INDEXED = "INDEXED"
    COMMITTED = "COMMITTED"
    DRAWN = "DRAWN"
    UNDRAWN = "UNDRAWN"
    REFERENCE = "REFERENCE"                # derivative notional (no principal exchange)
    SHARES = "SHARES"


def currency_precision(currency: str) -> int:
    return CURRENCY_PRECISION.get(currency.upper(), 2)


@dataclass(frozen=True)
class Money:
    amount: float
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.currency, str) or len(self.currency) != 3:
            raise UnitError(f"Invalid currency code {self.currency!r}")
        if not math.isfinite(self.amount):
            raise UnitError("Money amount must be finite")

    def _check(self, other: Money) -> None:
        if not isinstance(other, Money):
            raise UnitError("Money can only be combined with Money")
        if other.currency != self.currency:
            raise UnitError(f"Currency mismatch {self.currency} vs {other.currency}; convert explicitly")

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.amount - other.amount, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.amount, self.currency)

    def scale(self, k: float) -> Money:
        return Money(self.amount * k, self.currency)

    def rounded(self) -> Money:
        p = currency_precision(self.currency)
        return Money(round(self.amount + 0.0, p), self.currency)

    def is_zero(self, tol: float | None = None) -> bool:
        tol = tol if tol is not None else 0.5 * 10 ** (-currency_precision(self.currency))
        return abs(self.amount) < tol

    @staticmethod
    def zero(currency: str) -> Money:
        return Money(0.0, currency)

    def __str__(self) -> str:
        return f"{self.amount:,.{currency_precision(self.currency)}f} {self.currency}"


@dataclass(frozen=True)
class Price:
    value: float
    basis: QuoteBasis = QuoteBasis.PCT_OF_PAR
    kind: PriceKind = PriceKind.CLEAN
    principal_basis: PrincipalBasis = PrincipalBasis.CURRENT
    currency: str | None = None
    side: Side = Side.NA
    ts_ns: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise UnitError("Price value must be finite")

    # ------------------------------------------------------------------ conversions
    def to_pct_of_par(
        self,
        *,
        accrued_pct: float | None = None,
        days_to_maturity: int | None = None,
        year_basis: int = 360,
        yield_to_price: Callable[[float], float] | None = None,
        conversion_factor: float | None = None,
    ) -> Price:
        """Convert to a CLEAN percent-of-par price. Each basis needs its own explicit inputs."""
        b = self.basis
        if b == QuoteBasis.PCT_OF_PAR:
            clean = self.value
            if self.kind == PriceKind.DIRTY:
                if accrued_pct is None:
                    raise UnitError("Dirty price needs accrued_pct to produce a clean price")
                clean = self.value - accrued_pct
        elif b == QuoteBasis.DECIMAL:
            clean = self.value * 100.0
            if self.kind == PriceKind.DIRTY:
                if accrued_pct is None:
                    raise UnitError("Dirty price needs accrued_pct to produce a clean price")
                clean -= accrued_pct
        elif b == QuoteBasis.DISCOUNT_YIELD:
            if days_to_maturity is None:
                raise UnitError("Discount yield needs days_to_maturity (and year_basis) to produce a price")
            clean = 100.0 * (1.0 - self.value * days_to_maturity / year_basis)
        elif b in (QuoteBasis.YIELD, QuoteBasis.SPREAD_BP, QuoteBasis.DISCOUNT_MARGIN_BP):
            if yield_to_price is None:
                raise UnitError(f"{b.value} quotes need the instrument's yield_to_price function")
            clean = yield_to_price(self.value)
        elif b == QuoteBasis.POINTS:
            if conversion_factor is None:
                raise UnitError("Futures points need a conversion_factor to map to a bond price")
            clean = self.value * conversion_factor
        elif b == QuoteBasis.UPFRONT_PCT:
            clean = 100.0 - self.value  # protection seller's economic "price" of the reference obligation
        else:
            raise UnitError(f"No percent-of-par conversion for basis {b.value}")
        return Price(clean, QuoteBasis.PCT_OF_PAR, PriceKind.CLEAN, self.principal_basis, self.currency, self.side, self.ts_ns, self.source)

    def with_side(self, side: Side) -> Price:
        return Price(self.value, self.basis, self.kind, self.principal_basis, self.currency, side, self.ts_ns, self.source)

    def describe(self) -> str:
        return f"{self.value:.6g} {self.basis.value}/{self.kind.value}/{self.principal_basis.value} {self.currency or ''} {self.side.value}".strip()


@dataclass(frozen=True)
class Notional:
    amount: float
    kind: NotionalKind
    currency: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount):
            raise UnitError("Notional must be finite")


@dataclass(frozen=True)
class Rate:
    value: float
    compounding: Compounding = Compounding.ANNUAL
    day_count: str = "ACT/360"
    tenor: str | None = None
    reference: str | None = None

    def to_compounding(self, target: Compounding, t_years: float = 1.0) -> Rate:
        """Convert the annualised rate to another compounding convention (same day count)."""
        if target == self.compounding:
            return self
        # growth factor over t years
        if self.compounding == Compounding.SIMPLE:
            g = 1.0 + self.value * t_years
        elif self.compounding == Compounding.CONTINUOUS:
            g = math.exp(self.value * t_years)
        else:
            m = self.compounding.periods
            g = (1.0 + self.value / m) ** (m * t_years)
        if target == Compounding.SIMPLE:
            r = (g - 1.0) / t_years
        elif target == Compounding.CONTINUOUS:
            r = math.log(g) / t_years
        else:
            m = target.periods
            r = m * (g ** (1.0 / (m * t_years)) - 1.0)
        return Rate(r, target, self.day_count, self.tenor, self.reference)


@dataclass(frozen=True)
class Spread:
    bp: float
    kind: str                              # G | I | Z | OAS | DM | ASW | CDS
    reference_curve: str

    def __post_init__(self) -> None:
        if self.kind not in {"G", "I", "Z", "OAS", "DM", "ASW", "CDS"}:
            raise UnitError(f"Unknown spread kind {self.kind}")


@dataclass(frozen=True)
class RiskMeasure:
    name: str
    value: float
    currency: str
    shock_bp: float
    unit: str                              # e.g. "currency per bp", "years", "pct NAV"
    sign_convention: str = "up-shock loss positive"
    model_version: str = ""


@dataclass(frozen=True)
class Observation:
    """One market observation with the four times the blueprint requires (section 9)."""

    subject: str                           # instrument or index identifier
    field: str                             # e.g. "clean_mid", "fixing", "factor"
    value: float
    economic_ns: int                       # when the thing happened / was struck
    available_ns: int                      # when the simulated actor could know it
    publication_ns: int | None = None      # source publication or dissemination
    ingestion_ns: int | None = None        # when our database received it
    source: str = ""
    vintage: str = ""                      # version identifier of the source dataset
    quality: float | None = None
    revision_of: str | None = None         # id of the observation this one corrects
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.available_ns < self.economic_ns:
            raise UnitError("An observation cannot be available before it exists (available_ns < economic_ns)")
        if self.publication_ns is not None and self.available_ns < self.publication_ns:
            raise UnitError("available_ns cannot precede publication_ns")

    @property
    def id(self) -> str:
        key = f"{self.subject}|{self.field}|{self.value!r}|{self.economic_ns}|{self.available_ns}|{self.source}|{self.vintage}|{self.revision_of}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------- valuation
def market_value(
    price: Price,
    notional: Notional,
    *,
    factor: float = 1.0,
    index_ratio: float = 1.0,
    accrued_pct_current: float = 0.0,
    currency: str | None = None,
) -> Money:
    """Dirty market value with enforced basis consistency (blueprint section 17).

    * ``PCT_OF_PAR``/``DECIMAL`` prices only; convert other bases first.
    * Price on **current** principal with a **current** notional: factor is *not* applied again.
    * Price on current principal with an **original** notional: current = original x factor.
    * Price on **original** principal (factor-adjusted price): only valid with an original notional.
    * ``INDEXED`` (linker real price): principal is scaled by ``index_ratio``.
    """
    ccy = currency or price.currency or notional.currency
    if price.currency and price.currency != notional.currency:
        raise UnitError(f"Price currency {price.currency} differs from notional currency {notional.currency}")
    if price.basis == QuoteBasis.DECIMAL:
        px = price.value * 100.0
    elif price.basis == QuoteBasis.PCT_OF_PAR:
        px = price.value
    else:
        raise UnitError(f"market_value needs a percent-of-par or decimal price, got {price.basis.value}; convert first")
    if price.kind == PriceKind.DIRTY and accrued_pct_current:
        raise UnitError("Dirty price and separate accrued both supplied; pass one")
    accrued = 0.0 if price.kind == PriceKind.DIRTY else accrued_pct_current
    pb, nk = price.principal_basis, notional.kind
    if pb == PrincipalBasis.CURRENT:
        if nk == NotionalKind.CURRENT:
            current = notional.amount
        elif nk == NotionalKind.ORIGINAL:
            current = notional.amount * factor
        else:
            raise UnitError(f"Current-principal price cannot value a {nk.value} notional")
        return Money(current * (px + accrued) / 100.0, ccy)
    if pb == PrincipalBasis.ORIGINAL:
        if nk != NotionalKind.ORIGINAL:
            raise UnitError("A factor-adjusted (original-principal) price requires an ORIGINAL notional; applying the factor twice would understate value")
        return Money(notional.amount * px / 100.0 + notional.amount * factor * accrued / 100.0, ccy)
    if pb == PrincipalBasis.INDEXED:
        if nk not in (NotionalKind.ORIGINAL, NotionalKind.INDEXED):
            raise UnitError("An indexed (real) price needs an ORIGINAL or INDEXED notional")
        base = notional.amount if nk == NotionalKind.ORIGINAL else notional.amount / index_ratio
        return Money(base * index_ratio * (px + accrued) / 100.0, ccy)
    raise UnitError(f"Unsupported principal basis {pb}")


def validate_price_for_instrument(price: Price, expected_basis: QuoteBasis, expected_principal: PrincipalBasis) -> None:
    """Fail fast when a feed delivers the wrong quotation convention for an instrument."""
    if price.basis != expected_basis:
        raise UnitError(f"Instrument expects {expected_basis.value} quotes, got {price.basis.value}")
    if price.principal_basis != expected_principal:
        raise UnitError(f"Instrument expects prices per 100 {expected_principal.value} principal, got {price.principal_basis.value}")


__all__ = [
    "CURRENCY_PRECISION",
    "Compounding",
    "Money",
    "Notional",
    "NotionalKind",
    "Observation",
    "Price",
    "PriceKind",
    "PrincipalBasis",
    "QuoteBasis",
    "Rate",
    "RiskMeasure",
    "Side",
    "Spread",
    "UnitError",
    "currency_precision",
    "market_value",
    "validate_price_for_instrument",
]
