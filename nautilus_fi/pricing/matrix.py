"""Matrix pricing: spread curves per issuer / sector-rating bucket, fitted each date from good marks.

The target bond is priced by interpolating the comparable curve at its duration, adding a liquidity
premium, and discounting its cashflows off the risk-free curve. Issuer curves shrink toward their
sector-rating bucket curve, which shrinks toward the global curve, so thin buckets degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.instruments.analytics import price_from_spread
from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.cashflows import MarketInputs
from nautilus_fi.instruments.curves import SpreadCurve, ZeroCurve
from nautilus_fi.instruments.enums import rating_bucket


@dataclass
class SpreadObservation:
    bond: BondSpec
    z_spread_bp: float
    duration: float
    weight: float = 1.0
    rating_score: int | None = None


@dataclass
class LiquidityPremiumConfig:
    base_bp: float = 5.0
    small_issue_bp: float = 15.0          # amount outstanding below threshold
    small_issue_threshold: float = 300e6
    old_issue_bp: float = 5.0             # more than 3 years since issue
    hy_extra_bp: float = 15.0
    securitized_extra_bp: float = 20.0


@dataclass
class MatrixPricer:
    knots: tuple[float, ...] = (0.5, 2.0, 4.0, 7.0, 10.0, 20.0)
    bucket_shrinkage: float = 3.0
    issuer_shrinkage: float = 5.0
    min_issuer_obs: int = 2
    premium: LiquidityPremiumConfig = field(default_factory=LiquidityPremiumConfig)
    _global: SpreadCurve | None = field(default=None, repr=False)
    _buckets: dict[str, SpreadCurve] = field(default_factory=dict, repr=False)
    _issuers: dict[str, SpreadCurve] = field(default_factory=dict, repr=False)
    as_of: date | None = None

    @staticmethod
    def bucket_key(bond: BondSpec, rating_score: int | None) -> str:
        return f"{bond.asset_type.value}|{bond.sector or 'ALL'}|{rating_bucket(rating_score)}"

    def fit(self, as_of: date, observations: list[SpreadObservation]) -> MatrixPricer:
        self.as_of = as_of
        self._buckets.clear()
        self._issuers.clear()
        if not observations:
            self._global = None
            return self
        d = [o.duration for o in observations]
        s = [o.z_spread_bp for o in observations]
        w = [o.weight for o in observations]
        self._global = SpreadCurve.fit(d, s, w, knots=self.knots, shrinkage=1.0, key="GLOBAL")
        by_bucket: dict[str, list[SpreadObservation]] = {}
        by_issuer: dict[str, list[SpreadObservation]] = {}
        for o in observations:
            by_bucket.setdefault(self.bucket_key(o.bond, o.rating_score), []).append(o)
            by_issuer.setdefault(o.bond.issuer, []).append(o)
        for key, obs in by_bucket.items():
            self._buckets[key] = SpreadCurve.fit([o.duration for o in obs], [o.z_spread_bp for o in obs], [o.weight for o in obs], knots=self.knots, prior=self._global, shrinkage=self.bucket_shrinkage, key=key)
        for issuer, obs in by_issuer.items():
            if len(obs) < self.min_issuer_obs:
                continue
            prior = self._buckets.get(self.bucket_key(obs[0].bond, obs[0].rating_score), self._global)
            self._issuers[issuer] = SpreadCurve.fit([o.duration for o in obs], [o.z_spread_bp for o in obs], [o.weight for o in obs], knots=self.knots, prior=prior, shrinkage=self.issuer_shrinkage, key=issuer)
        return self

    def liquidity_premium_bp(self, bond: BondSpec, rating_score: int | None, as_of: date) -> float:
        p = self.premium
        bp = p.base_bp
        if bond.amount_outstanding is not None and bond.amount_outstanding < p.small_issue_threshold:
            bp += p.small_issue_bp
        if bond.age_years(as_of) > 3.0:
            bp += p.old_issue_bp
        if rating_score is not None and rating_score > 10:
            bp += p.hy_extra_bp
        if bond.asset_type.is_securitized:
            bp += p.securitized_extra_bp
        return bp

    def spread_bp(self, bond: BondSpec, duration: float, rating_score: int | None, as_of: date | None = None) -> tuple[float, str] | None:
        """Comparable spread (incl. liquidity premium) and the level it came from."""
        as_of = as_of or self.as_of
        curve = self._issuers.get(bond.issuer)
        level = "ISSUER"
        if curve is None:
            curve = self._buckets.get(self.bucket_key(bond, rating_score))
            level = "BUCKET"
        if curve is None:
            curve = self._global
            level = "GLOBAL"
        if curve is None:
            return None
        return curve.spread(duration) + self.liquidity_premium_bp(bond, rating_score, as_of or date.today()), level

    def price(self, bond: BondSpec, settle: date, curve: ZeroCurve, duration: float, rating_score: int | None, inputs: MarketInputs | None = None) -> tuple[float, float, str] | None:
        """Clean price per 100 current face, the spread used, and the comparable level."""
        res = self.spread_bp(bond, duration, rating_score, settle)
        if res is None:
            return None
        spread, level = res
        px = price_from_spread(bond, settle, curve, spread, inputs)
        return px, spread, level

    def bucket_spread(self, bond: BondSpec, duration: float, rating_score: int | None) -> float | None:
        curve = self._buckets.get(self.bucket_key(bond, rating_score)) or self._global
        return curve.spread(duration) if curve else None


__all__ = ["LiquidityPremiumConfig", "MatrixPricer", "SpreadObservation"]
