"""Enumerations shared across the fixed income model."""

from __future__ import annotations

from enum import Enum, IntEnum


class AssetType(str, Enum):
    """Broad fixed income asset type. Drives default settlement, cost and liquidity parameters."""

    SOVEREIGN = "SOVEREIGN"          # Treasuries, gilts, bunds, JGBs
    BILL = "BILL"                    # discount instruments: T-bills, CP, CDs
    AGENCY = "AGENCY"                # agency debentures
    SUPRANATIONAL = "SUPRANATIONAL"
    CORPORATE = "CORPORATE"          # IG and HY corporates (rating distinguishes)
    MUNICIPAL = "MUNICIPAL"
    COVERED = "COVERED"
    PREFERRED = "PREFERRED"          # preferreds / hybrids / perpetuals
    LOAN = "LOAN"                    # leveraged loans
    LINKER = "LINKER"                # inflation-linked (TIPS, linkers)
    MBS = "MBS"                      # agency / non-agency pass-throughs
    CMBS = "CMBS"
    ABS = "ABS"
    CLO = "CLO"
    OTHER = "OTHER"

    @property
    def is_securitized(self) -> bool:
        return self in {AssetType.MBS, AssetType.CMBS, AssetType.ABS, AssetType.CLO}

    @property
    def is_government(self) -> bool:
        return self in {AssetType.SOVEREIGN, AssetType.BILL, AssetType.AGENCY, AssetType.SUPRANATIONAL, AssetType.LINKER}


class CouponType(str, Enum):
    FIXED = "FIXED"
    FLOATING = "FLOATING"
    ZERO = "ZERO"
    STEP = "STEP"                      # scheduled coupon changes (step-ups / step-downs)
    INFLATION_LINKED = "INFLATION_LINKED"


class PrincipalType(str, Enum):
    BULLET = "BULLET"
    SINKING = "SINKING"                # scheduled sinking fund
    AMORTIZING = "AMORTIZING"          # factor-driven (MBS/ABS/CLO/loans)
    PERPETUAL = "PERPETUAL"


class DayCount(str, Enum):
    THIRTY_360 = "30/360"              # US (NASD) 30/360, standard for US corporates and agencies
    THIRTY_E_360 = "30E/360"           # Eurobond basis
    ACT_ACT_ICMA = "ACT/ACT ICMA"      # US Treasuries, most government bonds
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    ACT_360 = "ACT/360"                # money market, floaters, SOFR
    ACT_365F = "ACT/365F"              # GBP, AUD money markets


class Frequency(IntEnum):
    ZERO = 0
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12

    @property
    def months(self) -> int:
        if self == Frequency.ZERO:
            return 0
        return 12 // int(self)


class BusinessDayConvention(str, Enum):
    UNADJUSTED = "UNADJUSTED"
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"
    PRECEDING = "PRECEDING"


class Seniority(str, Enum):
    SENIOR_SECURED = "SENIOR_SECURED"
    SENIOR_UNSECURED = "SENIOR_UNSECURED"
    SUBORDINATED = "SUBORDINATED"
    JUNIOR = "JUNIOR"
    PREFERRED = "PREFERRED"
    UNKNOWN = "UNKNOWN"


class MarkTier(str, Enum):
    """Which tier of the pricing waterfall produced a mark. Lower rank is better."""

    TRADE = "TRADE"                    # recent institutional-size transaction
    COMPOSITE = "COMPOSITE"            # executable / composite quote (CP+, Ai-Price, CEP, dealer runs)
    EVALUATED = "EVALUATED"            # vendor evaluated price with direct observations
    MODEL = "MODEL"                    # matrix / factor model (incl. low-score evaluated prices)
    STALE_ROLLED = "STALE_ROLLED"      # last mark rolled forward with a factor model
    NONE = "NONE"

    @property
    def rank(self) -> int:
        return list(MarkTier).index(self)


# Rating scale: a single numeric score across agencies (1 = AAA ... 22 = D). Convenient for
# bucketing, point-in-time rating history and cost model inputs.
_RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]
_MOODYS = {
    "Aaa": "AAA", "Aa1": "AA+", "Aa2": "AA", "Aa3": "AA-", "A1": "A+", "A2": "A", "A3": "A-",
    "Baa1": "BBB+", "Baa2": "BBB", "Baa3": "BBB-", "Ba1": "BB+", "Ba2": "BB", "Ba3": "BB-",
    "B1": "B+", "B2": "B", "B3": "B-", "Caa1": "CCC+", "Caa2": "CCC", "Caa3": "CCC-", "Ca": "CC", "C": "C",
}


def rating_to_score(rating: str | None) -> int | None:
    """Map an S&P/Fitch or Moody's rating string to a 1..22 score (None if unrated)."""
    if rating is None:
        return None
    r = str(rating).strip()
    if r in {"", "NR", "WR", "WD", "N/A"}:
        return None
    r = _MOODYS.get(r, r.upper())
    if r not in _RATING_ORDER:
        return None
    return _RATING_ORDER.index(r) + 1


def score_to_rating(score: float | None) -> str | None:
    if score is None:
        return None
    score = int(round(score))
    if score < 1 or score > len(_RATING_ORDER):
        return None
    return _RATING_ORDER[score - 1]


def is_investment_grade(score: int | None) -> bool | None:
    if score is None:
        return None
    return score <= 10


def rating_bucket(score: int | None) -> str:
    """Coarse bucket used by cost / liquidity models and matrix pricing."""
    if score is None:
        return "NR"
    if score <= 4:
        return "AAA-AA"
    if score <= 7:
        return "A"
    if score <= 10:
        return "BBB"
    if score <= 13:
        return "BB"
    if score <= 16:
        return "B"
    return "CCC-D"
