"""The market object: currency/collateral discount curves, projection curves, sovereign and repo
curves, credit and survival curves, inflation, FX and volatility. A product declares the subset it
needs; the dependency validator checks it before a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from nautilus_fi.instruments.curves import ZeroCurve


class CurveKind(str, Enum):
    DISCOUNT_OIS = "DISCOUNT_OIS"        # collateral / overnight discounting by currency (SOFR, ESTR, SONIA)
    PROJECTION = "PROJECTION"            # term or overnight index projection
    SOVEREIGN = "SOVEREIGN"              # government par / zero curve (spread quotation reference)
    REPO = "REPO"                        # financing curve
    CREDIT = "CREDIT"                    # issuer / sector spread curve (bp vs tenor)
    SURVIVAL = "SURVIVAL"                # hazard / survival for CDS and default risk
    INFLATION = "INFLATION"              # real / breakeven curve
    FX_BASIS = "FX_BASIS"
    VOL = "VOL"


class MissingMarketData(LookupError):
    pass


@dataclass(frozen=True)
class CurveRef:
    name: str
    kind: CurveKind
    currency: str
    index: str | None = None
    collateral: str | None = None
    entity: str | None = None


@dataclass
class MarketState:
    as_of: date
    curves: dict[str, ZeroCurve] = field(default_factory=dict)
    refs: dict[str, CurveRef] = field(default_factory=dict)
    status: dict[str, str] = field(default_factory=dict)        # curve name -> CalibrationStatus value
    fx: dict[str, float] = field(default_factory=dict)           # pair -> spot
    fixings: dict[str, float] = field(default_factory=dict)      # index -> latest fixing
    hazards: dict[str, float] = field(default_factory=dict)      # entity -> flat hazard
    recoveries: dict[str, float] = field(default_factory=dict)
    vols: dict[str, float] = field(default_factory=dict)         # e.g. "USD-HW-SIGMA"
    cpi: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    stress: float = 0.0
    lineage: dict[str, tuple[str, ...]] = field(default_factory=dict)   # curve -> input sources

    # ------------------------------------------------------------------ registration
    def add_curve(self, ref: CurveRef, curve: ZeroCurve, status: str = "OK", lineage: tuple[str, ...] = ()) -> None:
        self.refs[ref.name] = ref
        self.curves[ref.name] = curve
        self.status[ref.name] = status
        self.lineage[ref.name] = lineage

    def _find(self, kind: CurveKind, currency: str, **match) -> ZeroCurve:
        for name, ref in self.refs.items():
            if ref.kind == kind and ref.currency == currency and all(getattr(ref, k) == v for k, v in match.items() if v is not None):
                if self.status.get(name) == "FAILED":
                    raise MissingMarketData(f"curve {name} failed calibration on {self.as_of}; no silent reuse")
                return self.curves[name]
        raise MissingMarketData(f"no {kind.value} curve for {currency} {match} on {self.as_of}")

    # ------------------------------------------------------------------ typed lookups
    def discount(self, currency: str, collateral: str | None = None) -> ZeroCurve:
        return self._find(CurveKind.DISCOUNT_OIS, currency, collateral=collateral)

    def projection(self, index: str, currency: str) -> ZeroCurve:
        try:
            return self._find(CurveKind.PROJECTION, currency, index=index)
        except MissingMarketData:
            return self.discount(currency)     # single-curve fallback is explicit here, not implicit elsewhere

    def sovereign(self, currency: str) -> ZeroCurve:
        return self._find(CurveKind.SOVEREIGN, currency)

    def repo(self, currency: str) -> ZeroCurve:
        try:
            return self._find(CurveKind.REPO, currency)
        except MissingMarketData:
            return self.discount(currency)

    def credit(self, entity: str, currency: str = "USD") -> ZeroCurve:
        return self._find(CurveKind.CREDIT, currency, entity=entity)

    def is_stale(self, name: str) -> bool:
        return self.status.get(name) == "STALE"

    def requirements_met(self, refs: list[CurveRef]) -> list[str]:
        problems = []
        for r in refs:
            try:
                self._find(r.kind, r.currency, index=r.index, collateral=r.collateral, entity=r.entity)
            except MissingMarketData as e:
                problems.append(str(e))
        return problems

    # ------------------------------------------------------------------ scenarios
    def shocked(self, shocks_bp: dict[str, float] | None = None, fx_shocks: dict[str, float] | None = None, hazard_multipliers: dict[str, float] | None = None) -> MarketState:
        """Static valuation shock: positions fixed, curves/fx/hazards moved (blueprint section 21)."""
        new = MarketState(self.as_of, dict(self.curves), dict(self.refs), dict(self.status), dict(self.fx), dict(self.fixings), dict(self.hazards), dict(self.recoveries), dict(self.vols), dict(self.cpi), self.stress, dict(self.lineage))
        for name, bp in (shocks_bp or {}).items():
            if name in new.curves:
                new.curves[name] = new.curves[name].shifted(bp)
            elif name == "*":
                new.curves = {k: v.shifted(bp) for k, v in new.curves.items()}
        for pair, pct in (fx_shocks or {}).items():
            if pair in new.fx:
                new.fx[pair] *= 1.0 + pct
        for ent, m in (hazard_multipliers or {}).items():
            if ent in new.hazards:
                new.hazards[ent] *= m
        return new


__all__ = ["CurveKind", "CurveRef", "MarketState", "MissingMarketData"]
