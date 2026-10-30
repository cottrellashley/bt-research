"""Execution as a process (blueprint section 15): creation, approval, routing, quotes, acceptance,
partial fills, cancellation, expiry and settlement, with lot rules, restricted lists, trading hours,
inventory access and short availability. Liquidity is matched only against opportunities accessible
while the order is active, volume is never allocated twice, and costs are decomposed as reference +
spread + impact + fees without a second delay charge."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from nautilus_fi.execution.assumptions import ExecutionAssumptions
from nautilus_fi.execution.costs import CostModel
from nautilus_fi.execution.liquidity import TYPICAL_SIZE, LiquidityModel, LiquidityStats
from nautilus_fi.instruments.bond import BondSpec


class OrderState(str, Enum):
    CREATED = "CREATED"
    APPROVED = "APPROVED"
    ROUTED = "ROUTED"
    QUOTED = "QUOTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class VenueType(str, Enum):
    RFQ = "RFQ"
    PORTFOLIO_TRADE = "PORTFOLIO_TRADE"
    ALL_TO_ALL = "ALL_TO_ALL"
    VOICE = "VOICE"
    PRIMARY = "PRIMARY"
    EXCHANGE = "EXCHANGE"


class EvidenceLevel(str, Enum):
    DAILY_EVALUATIONS = "DAILY_EVALUATIONS"          # assumed costs, delays and fills with broad sensitivities
    TRANSACTIONS = "TRANSACTIONS"                    # market activity, indicative size and cost distributions
    TIMESTAMPED_QUOTES = "TIMESTAMPED_QUOTES"        # quoted terms for the observed context
    FULL_ORDERS_RFQ = "FULL_ORDERS_RFQ"              # conditional fill, rejection and slippage behaviour


class ReferencePolicy(str, Enum):
    PRIOR_CLOSE = "PRIOR_CLOSE"                      # knowable reference: last published mark before the window
    INTRADAY_QUOTE = "INTRADAY_QUOTE"                # timestamped quotes available at the window
    SAME_DAY_CLOSE_RETROSPECTIVE = "SAME_DAY_CLOSE_RETROSPECTIVE"   # fill-day close: retrospective feasibility only


@dataclass
class ProcessConfig:
    approval_delay_business_days: int = 0
    reference_policy: ReferencePolicy = ReferencePolicy.PRIOR_CLOSE
    evidence_level: EvidenceLevel = EvidenceLevel.DAILY_EVALUATIONS
    venue: VenueType = VenueType.RFQ
    portfolio_trade_min_lines: int = 20
    n_dealers: int = 5
    dealer_response_base: float = 0.75          # response probability for an active bond
    quote_dispersion_bp: float = 5.0            # dispersion of dealer quotes around the model half spread
    dealer_markup_bp: float = 0.0               # explicit markup on top of the spread
    platform_fee_bp: float = 0.0
    transaction_tax_bp: float = 0.0
    trading_hours: tuple[int, int] = (8, 17)    # venue hours (UTC labels) for RFQ
    restricted: set[str] = field(default_factory=set)
    counterparty_limits: dict[str, float] = field(default_factory=dict)   # dealer -> max face per day
    allow_shorts: bool = False


@dataclass
class OrderRequest:
    order_id: str
    symbol: str
    side: str                       # BUY | SELL
    face: float
    limit_clean: float | None
    created_ns: int
    created_date: date
    protocol: str = "RFQ"
    tags: tuple[str, ...] = ()
    basket_lines: int = 1
    is_short: bool = False


@dataclass
class ExecutionContext:
    d: date
    now_ns: int
    reference_clean: float | None
    reference_source: str
    indicative_bid: float | None
    indicative_ask: float | None
    stats: LiquidityStats
    stress: float
    rating_score: int | None
    duration: float | None
    factor: float = 1.0
    borrow_available: float | None = None
    new_issue_reoffer: float | None = None
    new_issue_allocation: float = 0.0


@dataclass
class DealerQuote:
    dealer: str
    responded: bool
    clean: float | None
    size: float
    firm: bool = True


@dataclass
class CostDecomposition:
    reference_clean: float
    reference_source: str
    half_spread_bp: float
    impact_bp: float
    markup_bp: float
    platform_fee_bp: float
    tax_bp: float

    @property
    def total_bp(self) -> float:
        return self.half_spread_bp + self.impact_bp + self.markup_bp + self.platform_fee_bp + self.tax_bp

    def fill_clean(self, side: str) -> float:
        sign = 1.0 if side == "BUY" else -1.0
        return self.reference_clean * (1.0 + sign * (self.half_spread_bp + self.impact_bp + self.markup_bp) / 1e4)

    def explain(self) -> str:
        return f"ref {self.reference_clean:.4f} ({self.reference_source}) + spread {self.half_spread_bp:.1f}bp + impact {self.impact_bp:.1f}bp + markup {self.markup_bp:.1f}bp + fees {self.platform_fee_bp + self.tax_bp:.1f}bp"


@dataclass
class ProcessOutcome:
    state: OrderState
    fills: list[tuple[float, float]]           # (clean price, face)
    reason: str
    venue: VenueType
    evidence: EvidenceLevel
    quotes: list[DealerQuote] = field(default_factory=list)
    cost: CostDecomposition | None = None
    p_fill: float = 0.0
    draw: float = 0.0
    offered_face: float = 0.0
    transitions: list[OrderState] = field(default_factory=list)


class LiquidityBudget:
    """Accessible volume per bond-day shared across orders: offers consume it, unfilled offers are released."""

    def __init__(self) -> None:
        self.capacity: dict[tuple[str, date], float] = {}
        self.consumed: dict[tuple[str, date], float] = {}
        self.offers: dict[str, tuple[str, date, float]] = {}

    def set_capacity(self, symbol: str, d: date, face: float) -> None:
        self.capacity[(symbol, d)] = face

    def available(self, symbol: str, d: date) -> float:
        return max(self.capacity.get((symbol, d), float("inf")) - self.consumed.get((symbol, d), 0.0), 0.0)

    def offer(self, order_id: str, symbol: str, d: date, face: float) -> float:
        prev = self.offers.pop(order_id, None)
        if prev is not None:
            self.consumed[(prev[0], prev[1])] -= prev[2]
        take = min(face, self.available(symbol, d))
        if take > 0:
            self.consumed[(symbol, d)] = self.consumed.get((symbol, d), 0.0) + take
            self.offers[order_id] = (symbol, d, take)
        return take

    def settle(self, order_id: str, filled_face: float) -> None:
        """Replace the offer by the realised fill (release the unfilled remainder)."""
        prev = self.offers.pop(order_id, None)
        if prev is None:
            return
        symbol, d, offered = prev
        self.consumed[(symbol, d)] += filled_face - offered

    def release_all_offers(self) -> None:
        for order_id in list(self.offers):
            self.settle(order_id, 0.0)


class OrderProcessSimulator:
    def __init__(self, liquidity_model: LiquidityModel, cost_model: CostModel, assumptions: ExecutionAssumptions, config: ProcessConfig | None = None, budget: LiquidityBudget | None = None) -> None:
        self.liquidity_model = liquidity_model
        self.cost_model = cost_model
        self.assumptions = assumptions
        self.config = config or ProcessConfig()
        self.budget = budget or LiquidityBudget()
        self.dealer_usage: dict[tuple[str, date], float] = {}

    def _rng(self, order_id: str, d: date, salt: str = "") -> random.Random:
        return random.Random(f"{self.assumptions.seed}|{order_id}|{d.isoformat()}|{salt}")

    def evaluate(self, bond: BondSpec, req: OrderRequest, ctx: ExecutionContext, calendar) -> ProcessOutcome:
        cfg, a = self.config, self.assumptions
        transitions = [OrderState.CREATED]
        venue = VenueType.PORTFOLIO_TRADE if (req.protocol == "PORTFOLIO_TRADE" or req.basket_lines >= cfg.portfolio_trade_min_lines) else VenueType(req.protocol) if req.protocol in VenueType.__members__ else VenueType.RFQ
        if req.protocol == "NEW_ISSUE":
            venue = VenueType.PRIMARY
        # 1. eligibility
        if req.symbol in cfg.restricted:
            return ProcessOutcome(OrderState.REJECTED, [], "restricted list", venue, cfg.evidence_level, transitions=transitions + [OrderState.REJECTED])
        hour = pd.Timestamp(ctx.now_ns, tz="UTC").hour
        if not (cfg.trading_hours[0] <= hour < cfg.trading_hours[1]):
            return ProcessOutcome(OrderState.CREATED, [], "outside trading hours", venue, cfg.evidence_level, transitions=transitions)
        if req.is_short and not cfg.allow_shorts:
            return ProcessOutcome(OrderState.REJECTED, [], "short selling not permitted by mandate", venue, cfg.evidence_level, transitions=transitions + [OrderState.REJECTED])
        if req.is_short and ctx.borrow_available is not None and ctx.borrow_available < req.face:
            return ProcessOutcome(OrderState.REJECTED, [], f"borrow unavailable ({ctx.borrow_available:,.0f} < {req.face:,.0f})", venue, cfg.evidence_level, transitions=transitions + [OrderState.REJECTED])
        # 2. approval delay (operational)
        delay = max(cfg.approval_delay_business_days, a.min_delay_business_days)
        if delay > 0 and calendar.business_days_between(req.created_date, ctx.d) < delay + 1:
            return ProcessOutcome(OrderState.CREATED, [], f"awaiting approval ({delay}bd)", venue, cfg.evidence_level, transitions=transitions)
        transitions += [OrderState.APPROVED, OrderState.ROUTED]
        if ctx.reference_clean is None or ctx.reference_clean <= 0:
            return ProcessOutcome(OrderState.ROUTED, [], "no reference price", venue, cfg.evidence_level, transitions=transitions)
        # 3. primary market
        if venue == VenueType.PRIMARY:
            if ctx.new_issue_reoffer is None or req.side != "BUY":
                return ProcessOutcome(OrderState.REJECTED, [], "new issue not available", venue, cfg.evidence_level, transitions=transitions + [OrderState.REJECTED])
            alloc = bond.round_face(req.face * ctx.new_issue_allocation)
            cost = CostDecomposition(ctx.new_issue_reoffer, "REOFFER", 0.0, 0.0, 0.0, 0.0, 0.0)
            return ProcessOutcome(OrderState.PARTIALLY_FILLED if alloc < req.face else OrderState.FILLED, [(ctx.new_issue_reoffer, alloc)] if alloc > 0 else [], "primary allocation", venue, cfg.evidence_level, cost=cost, p_fill=1.0, offered_face=alloc, transitions=transitions + [OrderState.FILLED])
        # 4. fill probability and accessible size
        decision = self.liquidity_model.evaluate(bond, ctx.stats, req.side, ctx.d, ctx.stress, req.face, a)
        p_fill = decision.p_fill if a.apply_fill_probability else 1.0
        rng = self._rng(req.order_id, ctx.d)
        draw = rng.random()
        if draw > p_fill:
            return ProcessOutcome(OrderState.ROUTED, [], f"no accessible liquidity ({decision.reason})", venue, cfg.evidence_level, p_fill=p_fill, draw=draw, transitions=transitions)
        cap = decision.max_face if a.apply_size_cap else float("inf")
        self.budget.set_capacity(req.symbol, ctx.d, cap)
        offered = self.budget.offer(req.order_id, req.symbol, ctx.d, req.face)
        offered = bond.round_face(offered) if offered != float("inf") else req.face
        if offered <= 0:
            return ProcessOutcome(OrderState.ROUTED, [], "daily accessible volume exhausted", venue, cfg.evidence_level, p_fill=p_fill, draw=draw, transitions=transitions)
        # 5. quotes
        transitions.append(OrderState.QUOTED)
        typical = TYPICAL_SIZE.get(bond.asset_type, 1_000_000.0)
        hs = 0.0 if a.fill_at_mid else self.cost_model.half_spread_bp(bond, ctx.stats, offered, req.side, ctx.d, ctx.stress, venue.value, ctx.rating_score, a, ctx.duration)
        impact = 0.0
        if not a.fill_at_mid and offered > typical:
            base_hs = self.cost_model.half_spread_bp(bond, ctx.stats, typical, req.side, ctx.d, ctx.stress, venue.value, ctx.rating_score, a, ctx.duration)
            impact, hs = max(hs - base_hs, 0.0), min(hs, base_hs)
        quotes: list[DealerQuote] = []
        best: DealerQuote | None = None
        if venue in (VenueType.RFQ, VenueType.ALL_TO_ALL, VenueType.VOICE) and not a.fill_at_mid:
            resp_p = min(1.0, cfg.dealer_response_base * (0.5 + ctx.stats.quote_score)) * max(0.2, 1.0 - 0.5 * ctx.stress)
            for k in range(cfg.n_dealers):
                r = self._rng(req.order_id, ctx.d, f"dealer{k}")
                responded = r.random() < resp_p
                dealer = f"D{k + 1}"
                limit = cfg.counterparty_limits.get(dealer)
                used = self.dealer_usage.get((dealer, ctx.d), 0.0)
                if limit is not None and used >= limit:
                    responded = False
                if not responded:
                    quotes.append(DealerQuote(dealer, False, None, 0.0))
                    continue
                dispersion = r.gauss(0.0, cfg.quote_dispersion_bp)
                sign = 1.0 if req.side == "BUY" else -1.0
                px = ctx.reference_clean * (1.0 + sign * max(hs + dispersion, 0.0) / 1e4)
                size = min(offered, max(typical * (0.5 + r.random()), bond.min_piece))
                quotes.append(DealerQuote(dealer, True, px, size))
            responded = [q for q in quotes if q.responded]
            if not responded:
                self.budget.settle(req.order_id, 0.0)
                return ProcessOutcome(OrderState.QUOTED, [], "no dealer responded", venue, cfg.evidence_level, quotes, p_fill=p_fill, draw=draw, transitions=transitions)
            best = min(responded, key=lambda q: q.clean) if req.side == "BUY" else max(responded, key=lambda q: q.clean)
            fill_px = best.clean
            fill_face = min(offered, best.size)
            self.dealer_usage[(best.dealer, ctx.d)] = self.dealer_usage.get((best.dealer, ctx.d), 0.0) + fill_face
        else:
            fill_px = ctx.reference_clean * (1.0 + (1.0 if req.side == "BUY" else -1.0) * (hs + impact) / 1e4)
            fill_face = offered
        # 6. limit check: the best quote must be inside the limit
        if req.limit_clean is not None and ((req.side == "BUY" and fill_px > req.limit_clean + 1e-9) or (req.side == "SELL" and fill_px < req.limit_clean - 1e-9)):
            self.budget.settle(req.order_id, 0.0)
            return ProcessOutcome(OrderState.QUOTED, [], f"best quote {fill_px:.4f} outside limit {req.limit_clean:.4f}", venue, cfg.evidence_level, quotes, p_fill=p_fill, draw=draw, transitions=transitions)
        fill_face = bond.round_face(fill_face)
        if fill_face <= 0:
            self.budget.settle(req.order_id, 0.0)
            return ProcessOutcome(OrderState.QUOTED, [], "size below minimum piece", venue, cfg.evidence_level, quotes, p_fill=p_fill, draw=draw, transitions=transitions)
        cost = CostDecomposition(ctx.reference_clean, ctx.reference_source, hs if a.fill_at_mid is False else 0.0, impact, cfg.dealer_markup_bp, cfg.platform_fee_bp, cfg.transaction_tax_bp)
        state = OrderState.FILLED if fill_face >= req.face - 1e-9 else OrderState.PARTIALLY_FILLED
        return ProcessOutcome(state, [(fill_px, fill_face)], "filled" if state == OrderState.FILLED else "partial fill", venue, cfg.evidence_level, quotes, cost, p_fill, draw, fill_face, transitions + [OrderState.ACCEPTED, state])


__all__ = ["CostDecomposition", "DealerQuote", "EvidenceLevel", "ExecutionContext", "LiquidityBudget", "OrderProcessSimulator", "OrderRequest", "OrderState", "ProcessConfig", "ProcessOutcome", "ReferencePolicy", "VenueType"]
