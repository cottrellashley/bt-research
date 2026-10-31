"""The pricing waterfall: best available observation per bond-date, with the tier recorded.

Tier order (see ``MarkTier``):

1. **TRADE**: size-weighted price of institutional-size prints within the lookback, adjusted for side
   (customer buys sit above mid by roughly a half spread, customer sells below).
2. **COMPOSITE**: executable / composite quote mid; bid and ask kept for the execution simulator.
3. **EVALUATED**: vendor evaluation whose quality score clears the threshold.
4. **MODEL**: matrix / factor model price (also low-score evaluations, which are model prices in all
   but name).
5. **STALE_ROLLED**: yesterday's mark rolled forward with a factor model (never carried flat).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.enums import MarkTier


@dataclass
class TradeObs:
    ts: object
    price: float
    size: float
    side: str = "D"      # B customer buy, S customer sell, D inter-dealer


@dataclass
class QuoteObs:
    bid: float
    ask: float
    ts: object = None
    source: str = "COMPOSITE"


@dataclass
class EvaluatedObs:
    mid: float
    bid: float | None = None
    ask: float | None = None
    score: float | None = None   # vendor quality, e.g. BVAL 1-10
    source: str = "EVALUATED"


@dataclass(slots=True)                                   # one per bond-day: slots halve the memory
class MarkResult:
    clean_mid: float
    clean_bid: float
    clean_ask: float
    tier: MarkTier
    quality: float
    source: str = ""
    stale_days: int = 0
    n_trades: int = 0

    @property
    def half_spread_bp(self) -> float:
        return (self.clean_ask - self.clean_bid) / 2.0 / self.clean_mid * 1e4 if self.clean_mid else 0.0


@dataclass
class MarkObservation:
    trades: list[TradeObs] = field(default_factory=list)
    composite: QuoteObs | None = None
    evaluated: EvaluatedObs | None = None
    model_price: float | None = None
    last_mark: MarkResult | None = None
    rolled_last_mark: float | None = None   # last mark rolled to today by the factor model


@dataclass
class WaterfallConfig:
    trade_min_size: float = 100_000.0
    trade_min_count: int = 1
    trade_side_adjust: bool = True
    evaluated_min_score: float = 6.0          # BVAL caps score at 5 without direct observations
    max_stale_days: int = 10                  # beyond this a rolled mark degrades to quality 0.05
    quality_trade: float = 1.0
    quality_composite: float = 0.85
    quality_model: float = 0.35
    prefer_model_after_stale_days: int = 3


class PricingWaterfall:
    def __init__(self, config: WaterfallConfig | None = None, half_spread_bp: Callable[[BondSpec, date], float] | None = None) -> None:
        self.config = config or WaterfallConfig()
        self._half_spread_bp = half_spread_bp or (lambda bond, d: 25.0)

    def _synth_bid_ask(self, bond: BondSpec, d: date, mid: float) -> tuple[float, float]:
        hs = self._half_spread_bp(bond, d) / 1e4 * mid
        return mid - hs, mid + hs

    def mark(self, bond: BondSpec, d: date, obs: MarkObservation) -> MarkResult | None:
        cfg = self.config
        hs_bp = self._half_spread_bp(bond, d)
        # 1. trades
        trades = [t for t in obs.trades if t.size >= cfg.trade_min_size and t.price > 0]
        if len(trades) >= cfg.trade_min_count:
            px = np.array([t.price for t in trades])
            sz = np.array([t.size for t in trades])
            if cfg.trade_side_adjust:
                adj = np.array([1.0 if t.side == "B" else -1.0 if t.side == "S" else 0.0 for t in trades])
                px = px * (1.0 - adj * hs_bp / 1e4)
            mid = float(np.average(px, weights=sz))
            bid, ask = self._synth_bid_ask(bond, d, mid)
            if obs.composite is not None and obs.composite.bid < obs.composite.ask:
                bid, ask = min(bid, obs.composite.bid), max(ask, obs.composite.ask)
            quality = min(1.0, cfg.quality_trade * min(1.0, 0.5 + 0.25 * len(trades)))
            return MarkResult(mid, bid, ask, MarkTier.TRADE, quality, "TRADES", 0, len(trades))
        # 2. composite
        if obs.composite is not None and obs.composite.bid > 0 and obs.composite.ask >= obs.composite.bid:
            mid = 0.5 * (obs.composite.bid + obs.composite.ask)
            return MarkResult(mid, obs.composite.bid, obs.composite.ask, MarkTier.COMPOSITE, cfg.quality_composite, obs.composite.source)
        # 3. evaluated (good score)
        ev = obs.evaluated
        if ev is not None and ev.mid > 0 and (ev.score is None or ev.score >= cfg.evaluated_min_score):
            bid, ask = (ev.bid, ev.ask) if ev.bid and ev.ask else self._synth_bid_ask(bond, d, ev.mid)
            quality = (ev.score / 10.0) if ev.score is not None else 0.6
            return MarkResult(ev.mid, bid, ask, MarkTier.EVALUATED, quality, ev.source)
        # 4. model: low-score evaluation or matrix price (prefer the vendor's comparables model if present)
        candidates: list[MarkResult] = []
        if ev is not None and ev.mid > 0:
            bid, ask = (ev.bid, ev.ask) if ev.bid and ev.ask else self._synth_bid_ask(bond, d, ev.mid)
            candidates.append(MarkResult(ev.mid, bid, ask, MarkTier.MODEL, (ev.score or 3.0) / 10.0, ev.source + "(low score)"))
        if obs.model_price is not None and obs.model_price > 0:
            bid, ask = self._synth_bid_ask(bond, d, obs.model_price)
            candidates.append(MarkResult(obs.model_price, bid, ask, MarkTier.MODEL, cfg.quality_model, "MATRIX"))
        # 5. stale roll
        if obs.last_mark is not None and obs.rolled_last_mark is not None and obs.rolled_last_mark > 0:
            stale = obs.last_mark.stale_days + 1
            q = max(0.05, obs.last_mark.quality * (0.85 ** stale))
            bid, ask = self._synth_bid_ask(bond, d, obs.rolled_last_mark)
            rolled = MarkResult(obs.rolled_last_mark, bid, ask, MarkTier.STALE_ROLLED, q, "ROLL", stale)
            if candidates and (stale > cfg.prefer_model_after_stale_days or stale > cfg.max_stale_days):
                return candidates[0]
            if not candidates:
                return rolled
            # young stale roll of a good mark beats a model price
            return rolled if obs.last_mark.tier.rank <= MarkTier.EVALUATED.rank else candidates[0]
        return candidates[0] if candidates else None


__all__ = ["EvaluatedObs", "MarkObservation", "MarkResult", "PricingWaterfall", "QuoteObs", "TradeObs", "WaterfallConfig"]
