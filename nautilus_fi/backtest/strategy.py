"""Base class for fixed income strategies.

Strategies see the market only through a ``DecisionContext`` that the engine fills as information
becomes available; there is no handle to the environment or its history. Orders are GTD limit orders
built by the portfolio constructor (netting live orders, reserving settled cash, enforcing typed
constraints and forced-action policies), submitted at the decision time so they rest until the next
execution window.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from nautilus_trader.model import OrderSide, TimeInForce
from nautilus_trader.trading import Strategy, StrategyConfig

from nautilus_fi.backtest.context import StrategyServices
from nautilus_fi.core.context import DecisionContext
from nautilus_fi.execution.windows import NS_PER_DAY
from nautilus_fi.instruments.analytics import BondAnalytics, analyse
from nautilus_fi.instruments.bond import BondSpec, face_to_units
from nautilus_fi.instruments.curves import ZeroCurve
from nautilus_fi.portfolio.construction import OrderIntent, TargetPortfolio
from nautilus_fi.universe.universe import UniverseFilter


def _num(x) -> float:
    if x is None:
        return 0.0
    return float(x.as_double()) if hasattr(x, "as_double") else float(x)


class FixedIncomeStrategyConfig(StrategyConfig):
    def __init__(self, *, rebalance: str = "M", limit_tolerance_bp: float | None = None, max_order_age_business_days: int | None = None, default_protocol: str = "RFQ", log_decisions: bool = False, cash_buffer: float = 0.01, **_kwargs) -> None:
        super().__init__()
        self.rebalance = rebalance
        self.limit_tolerance_bp = limit_tolerance_bp
        self.max_order_age_business_days = max_order_age_business_days
        self.default_protocol = default_protocol
        self.log_decisions = log_decisions
        self.cash_buffer = cash_buffer


class FixedIncomeStrategy(Strategy):
    TIMER = "fi-decision"

    def __init__(self, config: FixedIncomeStrategyConfig | None = None) -> None:
        super().__init__(config or FixedIncomeStrategyConfig())
        self.fi_config: FixedIncomeStrategyConfig = config or FixedIncomeStrategyConfig()
        self.ctx: DecisionContext | None = None
        self.services: StrategyServices | None = None
        self._last_rebalance: date | None = None
        self._decision_count = 0
        self.decision_log: list[dict] = []

    # ------------------------------------------------------------------ wiring
    def attach(self, ctx: DecisionContext, services: StrategyServices) -> None:
        self.ctx = ctx
        self.services = services

    @property
    def today(self) -> date:
        return self.ctx.today

    # ------------------------------------------------------------------ lifecycle
    def on_start(self) -> None:
        assert self.ctx is not None and self.services is not None, "attach() must be called before the engine starts"
        first = self.services.schedule.decision_ns(self.ctx.today) if self.ctx.now_ns else None
        start_ns = first if first is not None else self.clock.timestamp_ns()
        now = self.clock.timestamp_ns()
        while start_ns < now:
            start_ns += NS_PER_DAY
        self.clock.set_timer_ns(self.TIMER, NS_PER_DAY, start_time_ns=start_ns, callback=None, fire_immediately=True)
        self.on_fi_start()

    def on_fi_start(self) -> None:
        """Hook for subclasses."""

    def on_time_event(self, event) -> None:
        if event.name != self.TIMER:
            self.on_fi_timer(event)
            return
        d = pd.Timestamp(event.ts_event, tz="UTC").date()
        if not self.services.calendar.is_business_day(d):
            return
        if self.ctx.now_ns < event.ts_event:
            self.ctx._set_now(event.ts_event)
        if self.ctx.funding_shortfall > 0:
            self.on_funding_shortfall(d, self.ctx.funding_shortfall)
        self.on_daily(d)
        if self._rebalance_due(d):
            self._decision_count += 1
            if self.fi_config.log_decisions:
                self.log.info(f"[{d}] decision #{self._decision_count}")
            self.on_decision(d)
            self._last_rebalance = d

    def on_fi_timer(self, event) -> None:
        """Hook for subclass timers."""

    def on_funding_shortfall(self, d: date, shortfall: float) -> None:
        """Default liquidation rule when the funding policy could not source cash: sell the most liquid
        holdings (by recent volume, bullet first) until the shortfall plus a 5% margin is covered.
        Override to implement a mandate-specific rule."""
        need = shortfall * 1.05
        pending = sum(o.leaves * (self.dirty_value_per_face(o.instrument) or 0.0) for o in self.ctx.open_orders() if o.side == "SELL")
        need -= pending
        if need <= 0:
            return
        rows = []
        for sym, face in self.holdings_face().items():
            if face <= 0:
                continue
            liq = self.ctx.liquidity(sym)
            bond = self.services.bonds.get(sym)
            dv = self.dirty_value_per_face(sym)
            if bond is None or not dv:
                continue
            rows.append((-(liq.volume_5d if liq else 0.0), 1 if bond.asset_type.is_securitized else 0, sym, face, dv))
        for _, _, sym, face, dv in sorted(rows):
            if need <= 0:
                break
            sell_face = self.services.bonds[sym].round_face(min(face, need / dv))
            if sell_face <= 0:
                continue
            self.order_face(sym, -sell_face, protocol="RFQ", tags=["funding=LIQUIDATE_BY_RULE"])
            need -= sell_face * dv

    def on_daily(self, d: date) -> None:
        """Hook: every trading day at the decision time."""

    def on_decision(self, d: date) -> None:
        """Override: generate orders on rebalance dates."""

    def _rebalance_due(self, d: date) -> bool:
        rule = self.fi_config.rebalance
        if self._last_rebalance is None:
            return True
        if isinstance(rule, int) or str(rule).isdigit():
            return self.services.calendar.business_days_between(self._last_rebalance, d) >= int(rule)
        rule = str(rule).upper()
        if rule == "D":
            return True
        if rule == "W":
            return d.isocalendar()[1] != self._last_rebalance.isocalendar()[1] or d.year != self._last_rebalance.year
        if rule == "M":
            return (d.year, d.month) != (self._last_rebalance.year, self._last_rebalance.month)
        if rule == "Q":
            return (d.year, (d.month - 1) // 3) != (self._last_rebalance.year, (self._last_rebalance.month - 1) // 3)
        raise ValueError(f"Unknown rebalance rule {rule!r}")

    # ------------------------------------------------------------------ point-in-time reads
    def mark(self, symbol: str):
        return self.ctx.mark(symbol)

    def liquidity(self, symbol: str):
        return self.ctx.liquidity(symbol)

    def rating_score(self, symbol: str) -> int | None:
        r = self.ctx.rating_score(symbol)
        if r is not None:
            return r
        b = self.services.bonds.get(symbol)
        return b.rating_score_initial if b else None

    @property
    def curve(self) -> ZeroCurve | None:
        c = self.ctx.curve()
        if c is None:
            return None
        return ZeroCurve(self.today, list(c.tenors), list(c.interest_rates), c.curve_name) if hasattr(c, "interest_rates") else c

    @property
    def stress(self) -> float:
        return self.ctx.stress

    def fixing(self, index: str) -> float | None:
        return self.ctx.fixing(index)

    def bond(self, symbol: str) -> BondSpec:
        if self.services.terms_known_as_of is not None:
            return self.services.terms_known_as_of(symbol, self.today)
        return self.services.bonds[symbol]

    def universe(self, filt: UniverseFilter | None = None) -> list[BondSpec]:
        """Bonds alive today for which a mark has been received, filtered on point-in-time attributes."""
        filt = filt or UniverseFilter()
        out = []
        defaulted = {a.instrument_id.symbol.value for a in self.ctx.actions() if getattr(a, "action", "") == "DEFAULT"}
        for sym in self.ctx.universe():
            b = self.services.bonds.get(sym)
            if b is None or (filt.require_mark and self.ctx.mark(sym) is None):
                continue
            if filt.exclude_defaulted and sym in defaulted:
                continue
            if (self.today - b.issue_date).days < filt.min_age_days:
                continue
            if filt.asset_types and b.asset_type not in filt.asset_types:
                continue
            if filt.currencies and b.currency not in filt.currencies:
                continue
            score = self.rating_score(sym)
            if filt.min_rating_score is not None and (score is None or score < filt.min_rating_score):
                continue
            if filt.max_rating_score is not None and (score is None or score > filt.max_rating_score):
                continue
            yrs = b.remaining_years(self.today)
            if filt.min_years_to_maturity is not None and yrs is not None and yrs < filt.min_years_to_maturity:
                continue
            if filt.max_years_to_maturity is not None and yrs is not None and yrs > filt.max_years_to_maturity:
                continue
            if filt.min_amount_outstanding is not None and (b.amount_outstanding or 0) < filt.min_amount_outstanding:
                continue
            if filt.custom is not None and not filt.custom(b, self.today):
                continue
            ok, _ = self.ctx.can_trade(sym)
            if not ok:
                continue
            out.append(b)
        return out

    def analytics(self, symbol: str) -> BondAnalytics | None:
        mk = self.ctx.mark(symbol)
        if mk is None:
            return None
        bond = self.bond(symbol)
        return analyse(bond, self.services.settle_date(bond, self.today), mk.clean_mid, self.curve, self.services.market_inputs(symbol, self.today), mk.factor)

    # ------------------------------------------------------------------ portfolio state
    def holdings_face(self) -> dict[str, float]:
        return {sym: p.quantity for sym, p in self.ctx.positions().items() if abs(p.quantity) > 0}

    def open_order_face(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for o in self.ctx.open_orders():
            out[o.instrument] = out.get(o.instrument, 0.0) + (o.leaves if o.side == "BUY" else -o.leaves)
        return out

    def cash(self) -> float:
        return self.ctx.cash(self.services.base_currency).available

    def nav(self) -> float:
        return self.ctx.nav or self.ctx.cash(self.services.base_currency).settled

    def dirty_value_per_face(self, symbol: str) -> float | None:
        mk = self.ctx.mark(symbol)
        if mk is None:
            return None
        fx = self.ctx.fx(self.services.bonds[symbol].currency) or 1.0
        return (mk.clean_mid + mk.accrued) / 100.0 * fx

    # ------------------------------------------------------------------ orders
    def order_face(self, symbol: str, face: float, limit_price: float | None = None, protocol: str | None = None, expire_business_days: int | None = None, tags: list[str] | None = None):
        """GTD limit order for a signed current-face amount (buy > 0, sell < 0)."""
        bond = self.bond(symbol)
        mk = self.ctx.mark(symbol)
        if mk is None:
            self.log.warning(f"No mark received yet for {symbol}; order skipped")
            return None
        ok, reason = self.ctx.can_trade(symbol)
        if not ok:
            self.log.warning(f"{symbol} not tradable: {reason}")
            return None
        face = bond.round_face(face)
        if face == 0.0:
            return None
        factor = mk.factor or 1.0
        units = face_to_units(abs(face) / factor)
        if units <= 0:
            return None
        side = OrderSide.BUY if face > 0 else OrderSide.SELL
        instrument = self.cache.instrument(bond.instrument_id(self.services.venue))
        if limit_price is None:
            tol_bp = self.fi_config.limit_tolerance_bp
            if tol_bp is None:
                tol_bp = 500.0 if self.services.assumptions.fill_at_mid else max(10.0, 2.0 * (mk.clean_ask - mk.clean_bid) / 2.0 / mk.clean_mid * 1e4)
            clean_limit = mk.clean_ask * (1.0 + tol_bp / 1e4) if side == OrderSide.BUY else mk.clean_bid * (1.0 - tol_bp / 1e4)
            limit_price = clean_limit * factor
        from nautilus_trader.model import Price, Quantity

        price = Price(round(limit_price, bond.price_precision), bond.price_precision)
        d = self.today
        age = expire_business_days or self.fi_config.max_order_age_business_days or self.services.assumptions.max_order_age_business_days or 5
        expire_ns = self.services.schedule.window_ns(self.services.calendar.add_business_days(d, age)) + 3_600_000_000_000
        order_tags = [f"protocol={protocol or self.fi_config.default_protocol}", *(tags or [])]
        order = self.order_factory.limit(instrument.id, side, Quantity(units, 0), price, time_in_force=TimeInForce.GTD, expire_time=expire_ns, tags=order_tags)
        self.services.record_decision(str(order.client_order_id), mk.clean_mid, d)
        self.decision_log.append({"date": d, "symbol": symbol, "side": side.name, "face": face, "limit": limit_price, "order_id": str(order.client_order_id)})
        self.submit_order(order)
        return order

    def cancel_open_orders(self, symbol: str | None = None) -> None:
        seen = set()
        for o in self.ctx.open_orders(symbol):
            iid = self.services.bonds[o.instrument].instrument_id(self.services.venue)
            if iid not in seen:
                seen.add(iid)
                self.cancel_all_orders(iid)

    def _position_rows(self) -> pd.DataFrame:
        rows = []
        nav = self.nav() or 1.0
        for sym, p in self.ctx.positions().items():
            b = self.services.bonds.get(sym)
            mk = self.ctx.mark(sym)
            if b is None:
                continue
            dv = self.dirty_value_per_face(sym) or 1.0
            rows.append({"symbol": sym, "issuer": b.issuer, "sector": b.sector, "country": b.country, "currency": b.currency, "seniority": b.seniority.value, "rating_score": self.rating_score(sym), "weight": p.quantity * dv / nav, "duration": (mk.mod_duration if mk and mk.mod_duration == mk.mod_duration else 0.0), "spread_duration": (mk.spread_duration if mk and mk.spread_duration == mk.spread_duration else 0.0), "dts": ((mk.spread_duration * mk.z_spread_bp) if mk and mk.spread_duration == mk.spread_duration and mk.z_spread_bp == mk.z_spread_bp else 0.0), "liquidity_rank": 1 if (self.ctx.liquidity(sym) and self.ctx.liquidity(sym).volume_5d > 0) else 3})
        return pd.DataFrame(rows)

    def rebalance_to_target_faces(self, targets: dict[str, float], protocol: str | None = None, cancel_existing: bool = True) -> list:
        """Trade toward target current-face amounts through the portfolio constructor."""
        if cancel_existing:
            self.cancel_open_orders()
        live = [] if cancel_existing else [{"order_id": o.order_id, "symbol": o.instrument, "side": o.side, "leaves_face": o.leaves} for o in self.ctx.open_orders()]
        dirty = {s: self.dirty_value_per_face(s) for s in set(targets) | set(self.holdings_face()) if self.dirty_value_per_face(s)}
        forced: list[OrderIntent] = []
        if self.services.forced_actions is not None:
            forced = self.services.forced_actions.evaluate(self.today, self.holdings_face(), {s: self.rating_score(s) for s in self.holdings_face()}, years_to_maturity={s: (self.services.bonds[s].remaining_years(self.today) or 99.0) for s in self.holdings_face()})
        res = self.services.constructor.build(TargetPortfolio(faces=targets), self.services.bonds, self.nav(), dirty, self.holdings_face(), live, self.cash(), self._position_rows(), protocol or self.fi_config.default_protocol, forced)
        for v in res.violations_post:
            if v.severity == "HARD":
                self.log.warning(f"hard constraint after construction: {v.describe()}")
        orders = []
        for intent in res.intents:
            for rid in intent.replaces:
                try:
                    from nautilus_trader.model import ClientOrderId

                    self.cancel_order(ClientOrderId(rid))
                except Exception:  # noqa: BLE001
                    pass
            o = self.order_face(intent.symbol, intent.face if intent.side == "BUY" else -intent.face, protocol=intent.protocol)
            if o is not None:
                orders.append(o)
        return orders

    def rebalance_to_weights(self, weights: dict[str, float], nav: float | None = None, protocol: str | None = None, cancel_existing: bool = True) -> list:
        nav = nav or self.nav()
        targets: dict[str, float] = {}
        for symbol, w in weights.items():
            dv = self.dirty_value_per_face(symbol)
            if dv is None or dv <= 0 or symbol not in self.services.bonds:
                continue
            targets[symbol] = self.services.bonds[symbol].round_face(w * nav / dv)
        return self.rebalance_to_target_faces(targets, protocol=protocol, cancel_existing=cancel_existing)


__all__ = ["FixedIncomeStrategy", "FixedIncomeStrategyConfig"]
