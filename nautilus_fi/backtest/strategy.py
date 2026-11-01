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
from nautilus_fi.domain.calendars import RebalanceCalendar, parse_calendar
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
    """``rebalance``: a calendar string (``D``, ``N``, ``W``, ``W-WED``, ``M``/``BMS``, ``BME``, ``Q``/``BQS``,
    ``BQE``) or a ``RebalanceCalendar`` (index calendars, explicit dates, ``Triggered``).
    ``start_with_decision``: decide on the first trading day regardless of the calendar."""

    def __init__(self, *, rebalance: str | int | RebalanceCalendar = "BMS", start_with_decision: bool = True, limit_tolerance_bp: float | None = None, max_order_age_business_days: int | None = None, default_protocol: str = "RFQ", log_decisions: bool = False, cash_buffer: float = 0.01, **_kwargs) -> None:
        super().__init__()
        self.rebalance = rebalance
        self.start_with_decision = start_with_decision
        self.limit_tolerance_bp = limit_tolerance_bp
        self.max_order_age_business_days = max_order_age_business_days
        self.default_protocol = default_protocol
        self.log_decisions = log_decisions
        self.cash_buffer = cash_buffer


class FixedIncomeStrategy(Strategy):
    TIMER = "fi-decision"
    FLUSH_TIMER = "fi-overlay-flush"

    def __init__(self, config: FixedIncomeStrategyConfig | None = None) -> None:
        super().__init__(config or FixedIncomeStrategyConfig())
        self.fi_config: FixedIncomeStrategyConfig = config or FixedIncomeStrategyConfig()
        self.ctx: DecisionContext | None = None
        self.services: StrategyServices | None = None
        self._last_rebalance: date | None = None
        self._decision_count = 0
        self.decision_log: list[dict] = []
        self.calendar: RebalanceCalendar = parse_calendar(self.fi_config.rebalance)
        self.construction_log: list[dict] = []          # every constructor run: targets, violations, relaxations
        self.trigger_log: list[dict] = []
        self._trigger_skipped = False
        self._last_signal_ranks: pd.Series | None = None

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
        if self.services.margin_venue is not None:                  # overlay orders go out just before the next window, after margin is funded
            flush = self.services.schedule.window_ns(self.ctx.today) - 60_000_000_000
            while flush < now:
                flush += NS_PER_DAY
            self.clock.set_timer_ns(self.FLUSH_TIMER, NS_PER_DAY, start_time_ns=flush, callback=None, fire_immediately=True)
        self._overlay_queue: list[tuple[str, int]] = []
        self.on_fi_start()

    def on_fi_start(self) -> None:
        """Hook for subclasses."""

    def on_time_event(self, event) -> None:
        if event.name == self.FLUSH_TIMER:
            self._flush_overlays()
            return
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
            self._trigger_skipped = False
            self.on_decision(d)
            if not self._trigger_skipped:
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
        cal = self.services.calendar
        last = self._last_rebalance
        if last is None and self.fi_config.start_with_decision:
            return True
        prev_td = cal.previous_business_day(d)
        next_td = cal.add_business_days(d, 1)
        elapsed = cal.business_days_between(last, d) if last is not None else None
        return self.calendar.is_due(d, last, prev_td, next_td, elapsed)

    def _trigger_allows(self, target_weights: dict[str, float]) -> bool:
        """Triggered calendars: rebalance only when the proposed change exceeds the threshold."""
        cal = self.calendar
        if not cal.triggered or self._last_rebalance is None:
            return True
        if cal.measure == "signal":
            values = self.ctx.field_slice(cal.field) if cal.field else {}
            ranks = pd.Series(values, dtype=float).rank(pct=True)
            prev = self._last_signal_ranks
            change = float((ranks - prev).abs().reindex(ranks.index.intersection(prev.index)).mean()) if prev is not None and len(ranks) else float("inf")
        else:
            from nautilus_fi.domain.conventions import TurnoverDefinition, turnover

            nav = self.nav() or 1.0
            current = {s: f * (self.dirty_value_per_face(s) or 0.0) / nav for s, f in self.holdings_face().items()}
            change = turnover(current, target_weights, TurnoverDefinition.HALF_SUM)
        ok = change >= cal.threshold
        self.trigger_log.append({"date": self.today, "measure": cal.measure, "value": change, "threshold": cal.threshold, "rebalanced": ok})
        if not ok:
            self._trigger_skipped = True
        return ok

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
        """Spendable cash: settled, unreserved cash plus - when the funding policy finances leverage by
        repo - the repo capacity against eligible holdings."""
        return self.ctx.cash(self.services.base_currency).available + float(self.ctx.field("funding.repo_capacity", self.services.base_currency, 0.0) or 0.0)

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

    # ------------------------------------------------------------------ overlays (margin venue)
    def overlay_positions(self) -> dict[str, float]:
        """Overlay / synthetic-short units held on the margin venue (as published to the context)."""
        return {k: float(v) for k, v in self.ctx.field_slice("overlay.units").items() if v}

    def overlay_price(self, symbol: str) -> float | None:
        return self.ctx.field("overlay.price", symbol)

    def overlay_exposures(self) -> pd.DataFrame:
        book = self.services.overlay_book
        return book.exposures_frame(self.today) if book is not None else pd.DataFrame()

    def order_overlay(self, symbol: str, units: float, expire_business_days: int = 5, tags: list[str] | None = None):
        """Marketable limit order on the margin venue; fills at the next execution window."""
        venue = self.services.margin_venue
        if venue is None:
            self.log.warning("no margin venue configured (BacktestSettings.margin)")
            return None
        units = round(units)
        px = self.overlay_price(symbol)
        if units == 0 or px is None:
            return None
        from nautilus_trader.model import InstrumentId, Price, Quantity, Symbol, Venue

        iid = InstrumentId(Symbol(symbol), Venue(venue))
        inst = self.cache.instrument(iid)
        if inst is None:
            self.log.warning(f"overlay {symbol} is not a margin-venue instrument")
            return None
        side = OrderSide.BUY if units > 0 else OrderSide.SELL
        limit = px * (1.05 if units > 0 else 0.95)
        expire_ns = self.services.schedule.window_ns(self.services.calendar.add_business_days(self.today, expire_business_days)) + 3_600_000_000_000
        order = self.order_factory.limit(iid, side, Quantity(abs(units), 0), Price(round(limit, inst.price_precision), inst.price_precision), time_in_force=TimeInForce.GTD, expire_time=expire_ns, tags=["overlay", *(tags or [])])
        self.decision_log.append({"date": self.today, "symbol": symbol, "side": side.name, "face": float(units), "limit": limit, "order_id": str(order.client_order_id), "venue": venue})
        self.submit_order(order)
        return order

    def rebalance_overlays(self, notionals: dict[str, float], nav: float | None = None) -> list[tuple[str, int]]:
        """Trade overlay contracts to target notionals (fractions of NAV, signed); unlisted holdings go
        to zero. Collateral for the targets' initial margin is prefunded now; the orders are submitted
        just before the next execution window (where they fill either way)."""
        venue = self.services.margin_venue
        if venue is None:
            self.log.warning("no margin venue configured (BacktestSettings.margin)")
            return []
        from nautilus_trader.model import InstrumentId, Symbol, Venue

        nav = nav or self.nav()
        held = self.overlay_positions()
        book = self.services.overlay_book
        need = 0.0
        self._overlay_queue = []
        for sym in sorted(set(notionals) | set(held)):
            iid = InstrumentId(Symbol(sym), Venue(venue))
            inst = self.cache.instrument(iid)
            px = self.overlay_price(sym)
            if inst is None or not px:
                continue
            mult = float(inst.multiplier.as_double()) if hasattr(inst.multiplier, "as_double") else float(inst.multiplier)
            target = round(notionals.get(sym, 0.0) * nav / (px * mult))
            need += abs(target) * px * mult * (book.contracts[sym].margin_init if book is not None and sym in book.contracts else 0.05)
            self.cancel_all_orders(iid)
            delta = target - round(held.get(sym, 0.0))
            if delta:
                self._overlay_queue.append((sym, delta))
        if self.services.prefund_margin is not None and need > 0:
            self.services.prefund_margin(need)
        return list(self._overlay_queue)

    def _flush_overlays(self) -> None:
        queue, self._overlay_queue = getattr(self, "_overlay_queue", []), []
        for sym, units in queue:
            self.order_overlay(sym, units)

    def cancel_open_orders(self, symbol: str | None = None) -> None:
        seen = set()
        for o in self.ctx.open_orders(symbol):
            iid = self.services.bonds[o.instrument].instrument_id(self.services.venue)
            if iid not in seen:
                seen.add(iid)
                self.cancel_all_orders(iid)

    def _row(self, sym: str, weight: float) -> dict | None:
        b = self.services.bonds.get(sym)
        if b is None:
            return None
        mk = self.ctx.mark(sym)
        ok = mk is not None
        dur = mk.mod_duration if ok and mk.mod_duration == mk.mod_duration else 0.0
        sdur = mk.spread_duration if ok and mk.spread_duration == mk.spread_duration else 0.0
        z = mk.z_spread_bp if ok and mk.z_spread_bp == mk.z_spread_bp else 0.0
        liq = self.ctx.liquidity(sym)
        return {"symbol": sym, "issuer": b.issuer, "sector": b.sector, "country": b.country, "currency": b.currency, "seniority": b.seniority.value, "rating_score": self.rating_score(sym), "weight": weight, "duration": dur, "spread_duration": sdur, "dts": sdur * z, "liquidity_rank": 1 if (liq and liq.volume_5d > 0) else 3}

    def _position_rows(self) -> pd.DataFrame:
        nav = self.nav() or 1.0
        rows = [self._row(sym, p.quantity * (self.dirty_value_per_face(sym) or 1.0) / nav) for sym, p in self.ctx.positions().items()]
        return pd.DataFrame([r for r in rows if r is not None])

    def _candidate_rows(self, symbols) -> pd.DataFrame:
        rows = [self._row(sym, 0.0) for sym in symbols]
        return pd.DataFrame([r for r in rows if r is not None])

    def benchmark_weights(self) -> dict[str, float]:
        """Benchmark weights as published to the context (empty when the run has no benchmark)."""
        return {k: float(v) for k, v in self.ctx.field_slice(self.services.benchmark_field).items() if v and v > 0}

    def _benchmark_rows(self) -> pd.DataFrame | None:
        bw = self.benchmark_weights()
        if not bw:
            return None
        rows = [self._row(sym, w) for sym, w in bw.items()]
        return pd.DataFrame([r for r in rows if r is not None])

    def rebalance_to_target_faces(self, targets: dict[str, float], protocol: str | None = None, cancel_existing: bool = True) -> list:
        """Trade toward target current-face amounts through the portfolio constructor."""
        if self.calendar.triggered:
            nav = self.nav() or 1.0
            if not self._trigger_allows({s: f * (self.dirty_value_per_face(s) or 0.0) / nav for s, f in targets.items()}):
                return []
            if self.calendar.measure == "signal" and self.calendar.field:
                self._last_signal_ranks = pd.Series(self.ctx.field_slice(self.calendar.field), dtype=float).rank(pct=True)
        if cancel_existing:
            self.cancel_open_orders()
        live = [] if cancel_existing else [{"order_id": o.order_id, "symbol": o.instrument, "side": o.side, "leaves_face": o.leaves} for o in self.ctx.open_orders()]
        dirty = {s: self.dirty_value_per_face(s) for s in set(targets) | set(self.holdings_face()) if self.dirty_value_per_face(s)}
        forced: list[OrderIntent] = []
        bench = self.benchmark_weights()
        if self.services.forced_actions is not None:
            removed = {s for s in self.holdings_face() if s not in bench} if bench else None            # index removals need published membership
            forced = self.services.forced_actions.evaluate(self.today, self.holdings_face(), {s: self.rating_score(s) for s in self.holdings_face()}, removed_from_index=removed, years_to_maturity={s: (self.services.bonds[s].remaining_years(self.today) or 99.0) for s in self.holdings_face()})
        conv = self.services.conventions
        res = self.services.constructor.build(TargetPortfolio(faces=targets), self.services.bonds, self.nav(), dirty, self.holdings_face(), live, self.cash(), self._position_rows(), protocol or self.fi_config.default_protocol, forced, candidate_rows=self._candidate_rows(set(targets) - set(self.holdings_face())), benchmark_rows=self._benchmark_rows(), turnover_definition=conv.turnover.value if conv is not None else "HALF_SUM")
        self.construction_log += res.records(self.today)
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
