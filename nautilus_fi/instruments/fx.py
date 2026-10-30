"""Multi-currency support: FX fixing policy, rate store with forward points, conversions, and a
rolling forward hedge program with hedged / unhedged return decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.instruments.derivatives import FXForward
from nautilus_fi.instruments.schedule import add_months


@dataclass(frozen=True)
class FXFixingPolicy:
    source: str = "WMR"
    time_label: str = "16:00 London"
    publish_lag_minutes: int = 15


class FXRates:
    """Spot and forward points by pair (``EURUSD`` = USD per EUR). Lookups are as-of (last value on or before)."""

    def __init__(self, policy: FXFixingPolicy | None = None, points_scale: float = 1e4) -> None:
        self.policy = policy or FXFixingPolicy()
        self.spot: dict[str, dict[date, float]] = {}
        self.points: dict[tuple[str, int], dict[date, float]] = {}   # (pair, tenor months) -> date -> forward points
        self.points_scale = points_scale

    def add_spot(self, pair: str, series: dict[date, float]) -> None:
        self.spot.setdefault(pair, {}).update(series)

    def add_points(self, pair: str, tenor_months: int, series: dict[date, float]) -> None:
        self.points.setdefault((pair, tenor_months), {}).update(series)

    @staticmethod
    def _asof(series: dict[date, float], d: date) -> float | None:
        keys = [k for k in series if k <= d]
        return series[max(keys)] if keys else None

    def rate(self, pair: str, d: date) -> float | None:
        if pair[:3] == pair[3:]:
            return 1.0
        v = self._asof(self.spot.get(pair, {}), d)
        if v is not None:
            return v
        inv = self._asof(self.spot.get(pair[3:] + pair[:3], {}), d)
        if inv:
            return 1.0 / inv
        # cross through USD
        a, b = pair[:3], pair[3:]
        ra, rb = self.rate(a + "USD", d) if a != "USD" else 1.0, self.rate(b + "USD", d) if b != "USD" else 1.0
        if ra is None or rb is None or (a != "USD" and ra == self.rate(a + "USD", d) is None):
            return None
        return ra / rb if rb else None

    def convert(self, amount: float, from_ccy: str, to_ccy: str, d: date) -> float:
        if from_ccy == to_ccy:
            return amount
        r = self.rate(from_ccy + to_ccy, d)
        if r is None:
            raise LookupError(f"No FX rate for {from_ccy}{to_ccy} on {d}")
        return amount * r

    def forward_rate(self, pair: str, d: date, tenor_months: int) -> float | None:
        s = self.rate(pair, d)
        pts = self._asof(self.points.get((pair, tenor_months), {}), d)
        if s is None or pts is None:
            return None
        return s + pts / self.points_scale


@dataclass
class HedgeProgram:
    """Rolling forward hedge of foreign-currency market values back to the base currency."""

    base_currency: str = "USD"
    hedge_ratio: float = 1.0
    tenor_months: int = 1
    rebalance_threshold: float = 0.02      # re-hedge when the hedge drifts more than this (fraction of exposure)
    hedges: dict[str, FXForward] = field(default_factory=dict)
    _n: int = 0

    def required_hedges(self, d: date, exposures: dict[str, float], rates: FXRates) -> list[FXForward]:
        """``exposures[ccy]`` = market value in that currency. Returns the forwards to open (replacing stale ones)."""
        new: list[FXForward] = []
        for ccy, mv in exposures.items():
            if ccy == self.base_currency or abs(mv) < 1e-9:
                continue
            pair = ccy + self.base_currency
            fwd = rates.forward_rate(pair, d, self.tenor_months)
            if fwd is None:
                raise LookupError(f"no forward for {pair} on {d}")
            target = -self.hedge_ratio * mv             # sell the foreign currency forward
            existing = self.hedges.get(ccy)
            if existing is not None and existing.settlement > d and abs(existing.notional_base - target) <= self.rebalance_threshold * abs(mv):
                continue
            self._n += 1
            contract = FXForward(f"FXF-{ccy}-{self._n}", pair, target, fwd, add_months(d, self.tenor_months))
            self.hedges[ccy] = contract
            new.append(contract)
        return new

    def due_for_roll(self, d: date) -> list[FXForward]:
        return [h for h in self.hedges.values() if h.settlement <= d]


def hedged_return_decomposition(local_return: float, fx_spot_return: float, forward_carry: float, hedge_ratio: float = 1.0) -> dict:
    """Unhedged = (1 + local)(1 + fx) - 1. Hedged ~= local + hedge_ratio x carry + (1 - h) x fx + interaction."""
    unhedged = (1.0 + local_return) * (1.0 + fx_spot_return) - 1.0
    hedge_pnl = hedge_ratio * (forward_carry - fx_spot_return)          # short forward gains when spot falls vs forward
    hedged = unhedged + hedge_pnl
    return {"unhedged": unhedged, "hedged": hedged, "local": local_return, "fx_spot": fx_spot_return, "carry": forward_carry, "hedge_pnl": hedge_pnl, "residual": hedged - (local_return + hedge_ratio * forward_carry + (1.0 - hedge_ratio) * fx_spot_return)}


__all__ = ["FXFixingPolicy", "FXRates", "HedgeProgram", "hedged_return_decomposition"]
