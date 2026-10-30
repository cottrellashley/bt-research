"""The market environment: everything the simulator and ledger know about the market, per date.

``MarketEnvironmentBuilder`` is the offline pipeline: schema-conforming frames in, a point-in-time
environment out. It runs the pricing waterfall (tier and evidence per bond-date), calibrates curves
with explicit status (never a silent reuse), fits matrix spread curves, rolls stale marks, derives
liquidity statistics both by close (for marks) and by execution window (for fills), validates the
dependency graph (fixings for floaters, CPI for linkers, factors for amortizers) and emits the
Nautilus data stream. The window quote sent to the venue follows the declared reference policy:
by default the *prior close*, so a 10:00 fill never references a 16:00 price of the same day.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from nautilus_fi.core.registry import DependencyGraph, Node, SupportMatrix, UnsupportedCombination
from nautilus_fi.data import schemas
from nautilus_fi.data.reference import reference_from_dataframe
from nautilus_fi.data.trace import daily_liquidity_stats
from nautilus_fi.data.types import (
    BondMark,
    CorporateAction,
    FactorUpdate,
    IndexFixing,
    LiquiditySnapshot,
    RatingUpdate,
    StressIndex,
    wrap,
)
from nautilus_fi.execution.costs import ParametricCostModel
from nautilus_fi.execution.fill_model import NewIssueTerms
from nautilus_fi.execution.liquidity import TYPICAL_SIZE, LiquidityStats, VolumeGatedLiquidityModel
from nautilus_fi.execution.windows import ExecutionWindowSchedule
from nautilus_fi.instruments.analytics import BondAnalytics, accrued_interest, analyse
from nautilus_fi.instruments.bond import BondSpec, face_to_units
from nautilus_fi.instruments.calendar import BondCalendar, get_calendar
from nautilus_fi.instruments.cashflows import MarketInputs, factor_at
from nautilus_fi.instruments.conventions import (
    EntitlementRule,
    SettlementConventions,
    entitlement_rule,
    market_for,
)
from nautilus_fi.instruments.curves import ZeroCurve
from nautilus_fi.instruments.enums import CouponType, MarkTier, PrincipalType, rating_to_score
from nautilus_fi.instruments.fx import FXRates
from nautilus_fi.market.calibration import CalibrationLog, CalibrationStatus, CurveCalibrator
from nautilus_fi.market.state import CurveKind, CurveRef, MarketState
from nautilus_fi.pricing.marks import EvidenceType
from nautilus_fi.pricing.matrix import MatrixPricer, SpreadObservation
from nautilus_fi.pricing.stale_roll import RollInputs, roll_mark
from nautilus_fi.pricing.waterfall import (
    EvaluatedObs,
    MarkObservation,
    MarkResult,
    PricingWaterfall,
    QuoteObs,
    TradeObs,
)
from nautilus_fi.support import default_support_matrix

TIER_EVIDENCE = {MarkTier.TRADE: EvidenceType.TRANSACTION, MarkTier.COMPOSITE: EvidenceType.COMPOSITE, MarkTier.EVALUATED: EvidenceType.EVALUATED, MarkTier.MODEL: EvidenceType.MODEL, MarkTier.STALE_ROLLED: EvidenceType.STALE, MarkTier.NONE: EvidenceType.STALE}


@dataclass
class ActionRecord:
    symbol: str
    action: str
    effective_date: date
    price: float | None
    announce_ts: pd.Timestamp
    note: str = ""


@dataclass
class MarketEnvironment:
    bonds: dict[str, BondSpec]
    dates: list[date]
    schedule: ExecutionWindowSchedule
    calendar: BondCalendar
    curves: dict[date, ZeroCurve]
    marks: dict[tuple[str, date], MarkResult]
    analytics_by_key: dict[tuple[str, date], BondAnalytics]
    liquidity: dict[tuple[str, date], LiquidityStats]
    stress_series: dict[date, float]
    fixings: dict[str, list[tuple[date, float]]]
    factors: dict[str, list[tuple[date, float]]]
    index_ratios: dict[str, list[tuple[date, float]]]
    ratings: dict[str, list[tuple[date, int]]]
    actions: list[ActionRecord]
    benchmark: pd.Series | None = None
    new_issues: dict[str, NewIssueTerms] = field(default_factory=dict)
    curve_name: str = "GOVT"
    liquidity_window: dict[tuple[str, date], LiquidityStats] = field(default_factory=dict)
    window_quotes: dict[tuple[str, date], tuple[float, float]] = field(default_factory=dict)
    reference_policy: str = "PRIOR_CLOSE"
    calibration_log: CalibrationLog = field(default_factory=CalibrationLog)
    curve_status: dict[date, str] = field(default_factory=dict)
    ois_curves: dict[date, ZeroCurve] = field(default_factory=dict)
    max_stale_curve_days: int = 5
    fx_rates: FXRates = field(default_factory=FXRates)
    cpi: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    cpi_base: dict[str, float] = field(default_factory=dict)
    daily_fixings: dict[str, dict[date, float]] = field(default_factory=dict)
    index_values: dict[str, dict[date, float]] = field(default_factory=dict)
    tranche_payments: dict[tuple[str, date], tuple[float, float]] = field(default_factory=dict)
    support: SupportMatrix = field(default_factory=default_support_matrix)
    settlement: SettlementConventions = field(default_factory=SettlementConventions)
    validation_problems: list[str] = field(default_factory=list)
    unsupported_symbols: set[str] = field(default_factory=set)
    base_currency: str = "USD"
    _dates_set: set[date] = field(default_factory=set, repr=False)
    _terminations: dict[str, tuple[date, str]] = field(default_factory=dict, repr=False)
    _defaults: dict[str, date] = field(default_factory=dict, repr=False)
    _actions_by_date: dict[date, list[ActionRecord]] = field(default_factory=dict, repr=False)
    _stale_recorded: set[date] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._dates_set = set(self.dates)
        for a in self.actions:
            self._actions_by_date.setdefault(a.effective_date, []).append(a)
            if a.action == "DEFAULT":
                self._defaults[a.symbol] = min(a.effective_date, self._defaults.get(a.symbol, a.effective_date))
            if a.action in ("CALL", "PUT", "TENDER", "MATURITY", "RECOVERY"):
                cur = self._terminations.get(a.symbol)
                if cur is None or a.effective_date < cur[0]:
                    self._terminations[a.symbol] = (a.effective_date, a.action)
        for s, b in self.bonds.items():
            if b.maturity is not None and s not in self._terminations:
                self._terminations[s] = (b.maturity, "MATURITY")

    # ---- dates ---------------------------------------------------------------------------
    @property
    def start_date(self) -> date:
        return self.dates[0]

    @property
    def end_date(self) -> date:
        return self.dates[-1]

    def is_trading_date(self, d: date) -> bool:
        return d in self._dates_set

    def prev_trading_date(self, d: date) -> date | None:
        i = bisect_left(self.dates, d)
        return self.dates[i - 1] if i > 0 else None

    # ---- lookups -------------------------------------------------------------------------
    def mark(self, symbol: str, d: date) -> MarkResult | None:
        return self.marks.get((symbol, d))

    def previous_mark(self, symbol: str, d: date) -> tuple[date, MarkResult] | None:
        p = self.prev_trading_date(d)
        while p is not None:
            m = self.marks.get((symbol, p))
            if m is not None:
                return p, m
            p = self.prev_trading_date(p)
        return None

    def reference(self, symbol: str, d: date, policy: str | None = None) -> tuple[float, float, float, str] | None:
        """Execution reference (clean mid, bid, ask, source) under the reference policy."""
        pol = policy or self.reference_policy
        if pol == "INTRADAY_QUOTE" and (symbol, d) in self.window_quotes:
            bid, ask = self.window_quotes[(symbol, d)]
            return 0.5 * (bid + ask), bid, ask, "INTRADAY_QUOTE"
        if pol == "SAME_DAY_CLOSE_RETROSPECTIVE":
            m = self.marks.get((symbol, d))
            return (m.clean_mid, m.clean_bid, m.clean_ask, "SAME_DAY_CLOSE(retrospective)") if m else None
        prev = self.previous_mark(symbol, d)
        if prev is None:
            return None
        pd_, m = prev
        return m.clean_mid, m.clean_bid, m.clean_ask, f"PRIOR_CLOSE({pd_})"

    def analytics(self, symbol: str, d: date) -> BondAnalytics | None:
        return self.analytics_by_key.get((symbol, d))

    def liquidity_stats(self, symbol: str, d: date, at: str = "window") -> LiquidityStats:
        src = self.liquidity_window if at == "window" else self.liquidity
        return src.get((symbol, d), LiquidityStats(amount_outstanding=self.bonds[symbol].amount_outstanding if symbol in self.bonds else None))

    def stress(self, d: date) -> float:
        return self.stress_series.get(d, 0.0)

    def curve(self, d: date) -> ZeroCurve | None:
        """Sovereign curve for ``d``; stale carry within ``max_stale_curve_days`` is recorded, beyond it None."""
        if d in self.curves:
            return self.curves[d]
        prior = [k for k in self.curves if k <= d]
        if not prior:
            return None
        last = max(prior)
        age = (d - last).days
        if age > self.max_stale_curve_days:
            if d not in self._stale_recorded:
                self._stale_recorded.add(d)
                self.curve_status[d] = CalibrationStatus.FAILED.value
            return None
        if d not in self._stale_recorded:
            self._stale_recorded.add(d)
            self.curve_status[d] = CalibrationStatus.STALE.value
            _, recipe = CurveCalibrator.carry_forward(self.curves[last], d, self.max_stale_curve_days)
            self.calibration_log.add(recipe)
        return self.curves[last].with_as_of(d)

    def market_state(self, d: date) -> MarketState:
        ms = MarketState(d, stress=self.stress(d))
        c = self.curve(d)
        if c is not None:
            ms.add_curve(CurveRef(self.curve_name, CurveKind.SOVEREIGN, self.base_currency), c, self.curve_status.get(d, "OK"), ("curves",))
        ois = self.ois_curves.get(d)
        if ois is not None:
            ms.add_curve(CurveRef(f"{self.base_currency}-OIS", CurveKind.DISCOUNT_OIS, self.base_currency), ois, "OK", ("ois",))
        for index in self.fixings:
            v = self.fixing(index, d)
            if v is not None:
                ms.fixings[index] = v
        for pair in self.fx_rates.spot:
            r = self.fx_rates.rate(pair, d)
            if r is not None:
                ms.fx[pair] = r
        return ms

    @staticmethod
    def _last_leq(series: list[tuple[date, float]] | None, d: date):
        if not series:
            return None
        keys = [k for k, _ in series]
        i = bisect_right(keys, d)
        return series[i - 1][1] if i > 0 else None

    def fixing(self, index: str, d: date) -> float | None:
        v = self._last_leq(self.fixings.get(index), d)
        if v is None and index in self.daily_fixings:
            keys = [k for k in self.daily_fixings[index] if k <= d]
            return self.daily_fixings[index][max(keys)] if keys else None
        return v

    def fx(self, currency: str, d: date) -> float | None:
        """Units of base currency per one unit of ``currency``."""
        if currency == self.base_currency:
            return 1.0
        return self.fx_rates.rate(currency + self.base_currency, d)

    def factor(self, symbol: str, d: date) -> float:
        bond = self.bonds.get(symbol)
        if bond is None:
            return 1.0
        if bond.coupon_type == CouponType.INFLATION_LINKED:
            if bond.inflation and bond.inflation.index in self.cpi and symbol in self.cpi_base:
                return factor_at(bond, d, self.market_inputs(symbol, d))
            v = self._last_leq(self.index_ratios.get(symbol), d)
            return float(v) if v is not None else 1.0
        if bond.principal_type == PrincipalType.AMORTIZING:
            v = self._last_leq(self.factors.get(symbol), d)
            return float(v) if v is not None else factor_at(bond, d, None)
        if bond.principal_type == PrincipalType.SINKING:
            return factor_at(bond, d, None)
        return 1.0

    def rating_score(self, symbol: str, d: date) -> int | None:
        v = self._last_leq([(k, float(s)) for k, s in self.ratings.get(symbol, [])], d)
        if v is not None:
            return int(v)
        bond = self.bonds.get(symbol)
        return bond.rating_score_initial if bond else None

    def market_inputs(self, symbol: str, d: date) -> MarketInputs:
        """Point-in-time inputs for cashflow projection: only history knowable on ``d``."""
        bond = self.bonds[symbol]
        fix = None
        daily = None
        if bond.floating is not None:
            idx = bond.floating.index
            series = self.fixings.get(idx, [])
            fix = {idx: {k: v for k, v in series if k <= d}}
            if bond.floating.transition is not None:
                nidx = bond.floating.transition.new_index
                fix[nidx] = {k: v for k, v in self.fixings.get(nidx, []) if k <= d}
            daily = {k: {dd: v for dd, v in s.items() if dd <= d} for k, s in self.daily_fixings.items()} or None
        facs = {k: v for k, v in self.factors.get(symbol, []) if k <= d} or None
        ratios = {k: v for k, v in self.index_ratios.get(symbol, []) if k <= d} or None
        dflt = self._defaults.get(symbol)
        cpi = self.cpi.get(bond.inflation.index) if (bond.inflation and bond.coupon_type == CouponType.INFLATION_LINKED) else None
        return MarketInputs(fixings=fix, realized_factors=facs, index_ratios=ratios, default_date=dflt if (dflt and dflt <= d) else None, daily_fixings=daily, index_values=self.index_values or None, cpi=cpi, cpi_base=self.cpi_base.get(symbol), known_as_of=d)

    def entitlement_rule(self, bond: BondSpec) -> EntitlementRule:
        return entitlement_rule(market_for(bond.asset_type, bond.currency, bond.country))

    def settle_date(self, bond: BondSpec, trade_date: date) -> date:
        if bond.settlement_lag_days is not None:
            return bond.settle_date(trade_date)
        return self.settlement.settle_date(bond.asset_type, bond.currency, trade_date, bond.calendar)

    def evidence(self, symbol: str, d: date) -> EvidenceType:
        m = self.marks.get((symbol, d))
        return TIER_EVIDENCE.get(m.tier, EvidenceType.STALE) if m else EvidenceType.STALE

    def is_structured(self, symbol: str) -> bool:
        return any(k[0] == symbol for k in self.tranche_payments)

    def actions_effective(self, d: date) -> list[ActionRecord]:
        return self._actions_by_date.get(d, [])

    def is_defaulted(self, symbol: str, d: date) -> bool:
        dd = self._defaults.get(symbol)
        return dd is not None and dd <= d

    def is_terminated(self, symbol: str, d: date) -> bool:
        t = self._terminations.get(symbol)
        return t is not None and t[0] <= d

    def termination(self, symbol: str) -> tuple[date, str] | None:
        return self._terminations.get(symbol)

    def alive_symbols(self, d: date) -> list[str]:
        return [s for s, b in self.bonds.items() if b.is_alive(d) and not self.is_terminated(s, d) and s not in self.unsupported_symbols]

    # ---- Nautilus objects ----------------------------------------------------------------
    def instruments(self, venue: str = "OTC") -> list:
        return [b.to_instrument(venue) for b in self.bonds.values()]

    def nautilus_data(self, venue: str = "OTC") -> list:
        from nautilus_trader.model import (
            InstrumentClose,
            InstrumentCloseType,
            MarkPriceUpdate,
            Price,
            Quantity,
            QuoteTick,
        )

        out: list = []
        ids = {s: b.instrument_id(venue) for s, b in self.bonds.items()}
        precision = {s: b.price_precision for s, b in self.bonds.items()}
        last_rating: dict[str, int] = {}
        last_factor: dict[str, float] = {}
        fixing_dates: dict[str, set[date]] = {k: {d for d, _ in v} for k, v in self.fixings.items()}
        sched = self.schedule

        def px(symbol: str, value: float) -> Price:
            p = precision[symbol]
            return Price(max(round(value, p), 10 ** (-p)), p)

        for d in self.dates:
            w_ns, m_ns, p_ns = sched.window_ns(d), sched.mark_ns(d), sched.publish_ns(d)
            for s in self.alive_symbols(d):
                bond = self.bonds[s]
                ref = self.reference(s, d)
                if ref is not None:
                    f_prev = self.factor(s, self.prev_trading_date(d) or d)
                    stats = self.liquidity_stats(s, d, "window")
                    typical = TYPICAL_SIZE.get(bond.asset_type, 1_000_000.0)
                    size_units = max(1, face_to_units(max(typical, stats.volume_5d / 5.0)))
                    out.append(QuoteTick(ids[s], px(s, ref[1] * f_prev), px(s, ref[2] * f_prev), Quantity(size_units, 0), Quantity(size_units, 0), w_ns, w_ns))
                    out.append(wrap(LiquiditySnapshot(ts_event=w_ns, ts_init=w_ns, instrument_id=ids[s], volume_5d=stats.volume_5d, trade_count_20d=int(stats.trade_count_20d), days_since_trade=int(min(stats.days_since_trade, 10_000)), quote_score=stats.quote_score, tradeable_bid_face=0.0, tradeable_ask_face=0.0, half_spread_bp=(ref[2] - ref[1]) / 2.0 / ref[0] * 1e4 if ref[0] else 0.0, p_fill_bid=0.0, p_fill_ask=0.0)))
                mk = self.marks.get((s, d))
                if mk is None:
                    continue
                f = self.factor(s, d)
                an = self.analytics_by_key.get((s, d))
                out.append(MarkPriceUpdate(ids[s], px(s, mk.clean_mid * f), m_ns, p_ns))
                accrued = 0.0 if self.is_defaulted(s, d) else accrued_interest(bond, self.settle_date(bond, d), self.market_inputs(s, d))
                out.append(wrap(BondMark(ts_event=m_ns, ts_init=p_ns, instrument_id=ids[s], clean_mid=mk.clean_mid, clean_bid=mk.clean_bid, clean_ask=mk.clean_ask, factor=f, accrued=accrued, tier=mk.tier.value, quality=mk.quality, source=mk.source,
                                         ytm=float(an.ytm) if an and an.ytm is not None else float("nan"), mod_duration=float(an.mod_duration) if an and an.mod_duration is not None else float("nan"), spread_duration=float(an.spread_duration) if an and an.spread_duration is not None else float("nan"), z_spread_bp=float(an.z_spread_bp) if an and an.z_spread_bp is not None else float("nan"), stale_days=int(mk.stale_days))))
                r = self.rating_score(s, d)
                if r is not None and last_rating.get(s) != r:
                    last_rating[s] = r
                    out.append(wrap(RatingUpdate(ts_event=m_ns, ts_init=p_ns, instrument_id=ids[s], agency="COMPOSITE", rating=str(r), score=int(r))))
                if bond.is_factor_driven and last_factor.get(s) != f:
                    last_factor[s] = f
                    out.append(wrap(FactorUpdate(ts_event=m_ns, ts_init=p_ns, instrument_id=ids[s], factor=f, paydown_price=bond.amortization.paydown_price if bond.amortization else 100.0)))
            c = self.curves.get(d)
            if c is not None:
                out.append(wrap(c.to_nautilus(m_ns, p_ns)))
            stress = self.stress_series.get(d)
            if stress is not None:
                out.append(wrap(StressIndex(ts_event=m_ns, ts_init=p_ns, name="STRESS", value=float(stress), regime="STRESS" if stress > 0.6 else "ELEVATED" if stress > 0.25 else "NORMAL")))
            for index, dset in fixing_dates.items():
                if d in dset:
                    out.append(wrap(IndexFixing(ts_event=m_ns, ts_init=p_ns, index_name=index, rate=float(self.fixing(index, d)))))
            for a in self._actions_by_date.get(d, []):
                if a.action in ("CALL", "PUT", "TENDER", "MATURITY", "RECOVERY") and a.price is not None and a.symbol in ids:
                    out.append(InstrumentClose(ids[a.symbol], px(a.symbol, a.price * self.factor(a.symbol, d)), InstrumentCloseType.CONTRACT_EXPIRED, m_ns, m_ns))
        for a in self.actions:
            if a.symbol not in ids:
                continue
            ts = int(_utc(a.announce_ts).value)
            out.append(wrap(CorporateAction(ts_event=ts, ts_init=ts, instrument_id=ids[a.symbol], action=a.action, effective_date=a.effective_date.isoformat(), price=float(a.price) if a.price is not None else float("nan"), note=a.note)))
        return out

    # ---- frames --------------------------------------------------------------------------
    def marks_frame(self) -> pd.DataFrame:
        rows = []
        for (s, d), m in self.marks.items():
            an = self.analytics_by_key.get((s, d))
            rows.append({"date": pd.Timestamp(d), "symbol": s, "clean_mid": m.clean_mid, "clean_bid": m.clean_bid, "clean_ask": m.clean_ask, "tier": m.tier.value, "evidence": TIER_EVIDENCE.get(m.tier, EvidenceType.STALE).value, "quality": m.quality, "stale_days": m.stale_days, "factor": self.factor(s, d), "ytm": an.ytm if an else None, "mod_duration": an.mod_duration if an else None, "z_spread_bp": an.z_spread_bp if an else None})
        return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True) if rows else pd.DataFrame()

    def coverage_frame(self) -> pd.DataFrame:
        """Missingness / staleness by date and asset type (blueprint section 10)."""
        rows = []
        for d in self.dates:
            alive = self.alive_symbols(d)
            for at in sorted({self.bonds[s].asset_type.value for s in alive}):
                syms = [s for s in alive if self.bonds[s].asset_type.value == at]
                marked = [s for s in syms if (s, d) in self.marks]
                tiers = pd.Series([self.marks[(s, d)].tier.value for s in marked]).value_counts().to_dict() if marked else {}
                rows.append({"date": pd.Timestamp(d), "asset_type": at, "alive": len(syms), "marked": len(marked), "no_observation": len(syms) - len(marked), "stress": self.stress(d), "curve_status": self.curve_status.get(d, "OK" if d in self.curves else "MISSING"), **{f"tier_{k}": v for k, v in tiers.items()}})
        return pd.DataFrame(rows)


# ============================================================================================ builder
def _date(v) -> date:
    return pd.Timestamp(v).date()


def _utc(v) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


@dataclass
class AnalyticsConfig:
    every_n_days: int = 1
    krd_tenors: tuple[float, ...] | None = None


class MarketEnvironmentBuilder:
    def __init__(self, reference: dict[str, BondSpec] | pd.DataFrame, start: date | str, end: date | str, calendar: BondCalendar | str = "US_BOND", schedule: ExecutionWindowSchedule | None = None, waterfall: PricingWaterfall | None = None, matrix: MatrixPricer | None = None, cost_model: ParametricCostModel | None = None, analytics: AnalyticsConfig | None = None, curve_name: str = "GOVT", roll_stale_marks: bool = True, reference_policy: str = "PRIOR_CLOSE", max_stale_curve_days: int = 5, strict_dependencies: bool = True, support: SupportMatrix | None = None, base_currency: str = "USD") -> None:
        self.bonds = reference if isinstance(reference, dict) else reference_from_dataframe(reference)
        self.start, self.end = _date(start), _date(end)
        self.calendar = calendar if isinstance(calendar, BondCalendar) else get_calendar(calendar)
        self.schedule = schedule or ExecutionWindowSchedule()
        self.cost_model = cost_model or ParametricCostModel()
        self.waterfall = waterfall or PricingWaterfall(half_spread_bp=self._indicative_half_spread)
        self.matrix = matrix or MatrixPricer()
        self.analytics_cfg = analytics or AnalyticsConfig()
        self.curve_name = curve_name
        self.roll_stale = roll_stale_marks
        self.reference_policy = reference_policy
        self.max_stale_curve_days = max_stale_curve_days
        self.strict = strict_dependencies
        self.support = support or default_support_matrix()
        self.base_currency = base_currency
        self.evaluated = self.composite = self.trades = self.curves_df = self.fixings_df = self.daily_fixings_df = None
        self.factors_df = self.ratios_df = self.ratings_df = self.actions_df = self.stress_df = self.benchmark_df = self.cpi_df = self.fx_df = self.tranche_df = None
        self.new_issues: dict[str, NewIssueTerms] = {}
        self._liq_model = VolumeGatedLiquidityModel()
        self.calibrator = CurveCalibrator()

    # ---- inputs (fluent) ---------------------------------------------------------------
    def with_evaluated_prices(self, df):
        self.evaluated = schemas.EVALUATED_PRICES.validate(df)
        return self

    def with_composite_quotes(self, df):
        self.composite = schemas.COMPOSITE_QUOTES.validate(df)
        return self

    def with_trades(self, df):
        self.trades = schemas.TRADES.validate(df)
        return self

    def with_curves(self, df):
        self.curves_df = schemas.CURVES.validate(df)
        return self

    def with_fixings(self, df, daily: bool = False):
        if daily:
            self.daily_fixings_df = schemas.FIXINGS.validate(df)
        else:
            self.fixings_df = schemas.FIXINGS.validate(df)
        return self

    def with_factors(self, df):
        self.factors_df = schemas.FACTORS.validate(df)
        return self

    def with_index_ratios(self, df):
        self.ratios_df = schemas.INDEX_RATIOS.validate(df)
        return self

    def with_cpi(self, df):
        """Columns: index, year, month, value, publish_ts (optional)."""
        self.cpi_df = df
        return self

    def with_fx(self, df):
        """Columns: date, pair (e.g. EURUSD), rate; optional tenor_months + points."""
        self.fx_df = df
        return self

    def with_ratings(self, df):
        self.ratings_df = schemas.RATINGS.validate(df)
        return self

    def with_corporate_actions(self, df):
        self.actions_df = schemas.CORPORATE_ACTIONS.validate(df)
        return self

    def with_stress(self, df):
        self.stress_df = schemas.STRESS.validate(df)
        return self

    def with_benchmark(self, df):
        self.benchmark_df = schemas.BENCHMARK.validate(df)
        return self

    def with_tranche_payments(self, df):
        """Deal-level allocations per tranche: columns symbol, pay_date, interest_per_100, principal_per_100 (per 100 original face)."""
        self.tranche_df = df
        return self

    def with_new_issues(self, terms: dict[str, NewIssueTerms]):
        self.new_issues = dict(terms)
        return self

    # ---- helpers -------------------------------------------------------------------------
    def _indicative_half_spread(self, bond: BondSpec, d: date) -> float:
        from nautilus_fi.execution.assumptions import NET_FULL

        typical = TYPICAL_SIZE.get(bond.asset_type, 1_000_000.0)
        return self.cost_model.half_spread_bp(bond, LiquidityStats(quote_score=0.7), typical, "SELL", d, 0.0, "RFQ", bond.rating_score_initial, NET_FULL)

    @staticmethod
    def _series_by_key(df: pd.DataFrame | None, key: str, value: str) -> dict[str, list[tuple[date, float]]]:
        out: dict[str, list[tuple[date, float]]] = {}
        if df is None or df.empty:
            return out
        for k, g in df.groupby(key):
            g = g.sort_values("date")
            out[str(k)] = [(_date(r["date"]), float(r[value])) for _, r in g.iterrows()]
        return out

    def _build_curves(self, dates: list[date], log: CalibrationLog) -> tuple[dict[date, ZeroCurve], dict[date, ZeroCurve], dict[date, str]]:
        curves: dict[date, ZeroCurve] = {}
        ois: dict[date, ZeroCurve] = {}
        status: dict[date, str] = {}
        if self.curves_df is None or self.curves_df.empty:
            return curves, ois, status
        df = self.curves_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        kind_col = df["kind"] if "kind" in df else pd.Series("par", index=df.index)
        for (d, cname), g in df.groupby(["date", "curve"]):
            if d < dates[0] or d > dates[-1]:
                continue
            g = g.sort_values("tenor")
            kind = str(kind_col.loc[g.index[0]]).lower() if "kind" in df else "par"
            tenors = g["tenor"].astype(float).to_numpy()
            rates = g["rate"].astype(float).to_numpy()
            target = ois if str(cname).upper().endswith("OIS") else curves
            if kind == "zero":
                try:
                    target[d] = ZeroCurve(d, tenors, rates, str(cname))
                    if target is curves:
                        status[d] = CalibrationStatus.OK.value
                except Exception as e:  # noqa: BLE001
                    status[d] = CalibrationStatus.FAILED.value
                    from nautilus_fi.market.calibration import CalibrationRecipe

                    log.add(CalibrationRecipe(str(cname), d, "zero-input", status=CalibrationStatus.FAILED, message=str(e)))
            else:
                curve, recipe = self.calibrator.bootstrap_par(str(cname), d, [{"tenor": float(t), "quote": float(r), "source": "input"} for t, r in zip(tenors, rates)], info_cutoff=d.isoformat())
                log.add(recipe)
                if curve is not None:
                    target[d] = curve
                if target is curves:
                    status[d] = recipe.status.value
        return curves, ois, status

    def _actions(self) -> list[ActionRecord]:
        out: list[ActionRecord] = []
        explicit: set[str] = set()
        if self.actions_df is not None and not self.actions_df.empty:
            for _, r in self.actions_df.iterrows():
                if str(r["action"]).upper() in ("CALL", "PUT", "TENDER", "MATURITY", "RECOVERY"):
                    explicit.add(str(r["instrument_id"]))
        for s_, b in self.bonds.items():
            if b.maturity is not None and self.start <= b.maturity <= self.end and s_ not in explicit:
                eff = self.calendar.adjust(b.maturity, b.business_day_convention)
                ann = pd.Timestamp(self.schedule.publish_ns(self.calendar.previous_business_day(eff)), tz="UTC")
                out.append(ActionRecord(s_, "MATURITY", eff, 100.0, ann, "scheduled maturity"))
        if self.actions_df is None or self.actions_df.empty:
            return out
        for _, r in self.actions_df.iterrows():
            sym = str(r["instrument_id"])
            eff = self.calendar.adjust(_date(r["effective_date"]), self.bonds[sym].business_day_convention) if sym in self.bonds else _date(r["effective_date"])
            ann = r.get("announce_ts")
            if ann is None or (isinstance(ann, float) and np.isnan(ann)) or pd.isna(ann):
                ann_ts = pd.Timestamp(self.schedule.publish_ns(self.calendar.previous_business_day(eff)), tz="UTC")
            else:
                ann_ts = _utc(ann)
            price = r.get("price")
            price = None if price is None or (isinstance(price, float) and np.isnan(price)) else float(price)
            out.append(ActionRecord(sym, str(r["action"]).upper(), eff, price, ann_ts, str(r.get("note", "") or "")))
        return out

    def _validate_dependencies(self, env: MarketEnvironment) -> list[str]:
        g = DependencyGraph()
        g.add_node(Node("model:hw1f-trinomial", "model", detail="callable exercise/valuation"))
        for index in env.fixings:
            g.add_node(Node(f"fixings:{index}", "data", units="rate", available=True))
        for index in env.daily_fixings:
            g.add_node(Node(f"daily_fixings:{index}", "data", units="rate", available=True))
        for index in env.cpi:
            g.add_node(Node(f"cpi:{index}", "data", units="index", available=True))
        for s in env.factors:
            g.add_node(Node(f"factors:{s}", "data", units="fraction", available=True))
        for s in env.index_ratios:
            g.add_node(Node(f"index_ratios:{s}", "data", units="ratio", available=True))
        for s, b in self.bonds.items():
            if not b.is_alive(self.end) and (b.maturity and b.maturity < self.start):
                continue
            g.add_node(Node(s, "product"))
            if b.coupon_type == CouponType.FLOATING and b.floating is not None:
                idx = b.floating.index
                if b.floating.transition is not None and b.floating.transition.effective_date <= self.end:
                    nidx = b.floating.transition.new_index
                    g.require(s, f"daily_fixings:{nidx}" if b.floating.transition.new_method.value != "IN_ADVANCE" else f"fixings:{nidx}", "rate", "post-transition coupon needs new index fixings")
                if b.floating.method == "IN_ADVANCE":
                    g.require(s, f"fixings:{idx}", "rate", "floating coupon needs term fixings")
                else:
                    g.require(s, f"daily_fixings:{idx}", "rate", "in-arrears coupon needs daily fixings")
            if b.coupon_type == CouponType.INFLATION_LINKED:
                if b.inflation and b.inflation.index in env.cpi:
                    g.require(s, f"cpi:{b.inflation.index}", "index", "linker needs CPI history")
                else:
                    g.require(s, f"index_ratios:{s}", "ratio", "linker needs index ratios or CPI")
            if b.is_callable:
                g.require(s, "model:hw1f-trinomial", "", "callable needs an exercise/valuation model")
        problems = g.validate()
        return problems

    # ---- build -----------------------------------------------------------------------------
    def build(self) -> MarketEnvironment:
        dates = self.calendar.business_days(self.start, self.end)
        log = CalibrationLog()
        curves, ois, curve_status = self._build_curves(dates, log)
        fixings = self._series_by_key(self.fixings_df, "index", "rate")
        daily_fixings: dict[str, dict[date, float]] = {}
        if self.daily_fixings_df is not None and not self.daily_fixings_df.empty:
            for idx, g in self.daily_fixings_df.groupby("index"):
                daily_fixings[str(idx)] = {_date(r["date"]): float(r["rate"]) for _, r in g.iterrows()}
        factors = self._series_by_key(self.factors_df, "instrument_id", "factor")
        ratios = self._series_by_key(self.ratios_df, "instrument_id", "index_ratio")
        cpi: dict[str, dict[tuple[int, int], float]] = {}
        if self.cpi_df is not None and not self.cpi_df.empty:
            for idx, g in self.cpi_df.groupby("index"):
                cpi[str(idx)] = {(int(r["year"]), int(r["month"])): float(r["value"]) for _, r in g.iterrows()}
        fx = FXRates()
        if self.fx_df is not None and not self.fx_df.empty:
            for pair, g in self.fx_df.groupby("pair"):
                fx.add_spot(str(pair), {_date(r["date"]): float(r["rate"]) for _, r in g.iterrows() if pd.isna(r.get("tenor_months", np.nan))})
                for ten, gg in g[g.get("tenor_months", pd.Series(np.nan, index=g.index)).notna()].groupby("tenor_months") if "tenor_months" in g else []:
                    fx.add_points(str(pair), int(ten), {_date(r["date"]): float(r["points"]) for _, r in gg.iterrows()})
        ratings: dict[str, list[tuple[date, int]]] = {}
        if self.ratings_df is not None and not self.ratings_df.empty:
            for s, g in self.ratings_df.groupby("instrument_id"):
                ratings[str(s)] = [(_date(r["date"]), rating_to_score(r["rating"]) or 0) for _, r in g.sort_values("date").iterrows() if rating_to_score(r["rating"]) is not None]
        actions = self._actions()
        stress_series = {_date(r["date"]): float(r["value"]) for _, r in self.stress_df.iterrows()} if self.stress_df is not None and not self.stress_df.empty else {}
        benchmark = None
        if self.benchmark_df is not None and not self.benchmark_df.empty:
            b = self.benchmark_df.copy()
            benchmark = pd.Series(b["total_return"].astype(float).to_numpy(), index=pd.to_datetime(b["date"]), name="benchmark")
        tranche_payments: dict[tuple[str, date], tuple[float, float]] = {}
        if self.tranche_df is not None and not self.tranche_df.empty:
            for _, r in self.tranche_df.iterrows():
                tranche_payments[(str(r["symbol"]), _date(r["pay_date"]))] = (float(r.get("interest_per_100", 0.0)), float(r.get("principal_per_100", 0.0)))
        cpi_base: dict[str, float] = {}
        for s, b in self.bonds.items():
            if b.coupon_type == CouponType.INFLATION_LINKED and b.inflation and b.inflation.index in cpi:
                from nautilus_fi.instruments import linkers as lk

                conv = lk.CONVENTIONS.get(b.inflation.convention, lk.US_TIPS)
                try:
                    cpi_base[s] = lk.base_cpi_for(cpi[b.inflation.index], b.dated_date, conv)
                except Exception:  # noqa: BLE001
                    pass

        env = MarketEnvironment(bonds=self.bonds, dates=dates, schedule=self.schedule, calendar=self.calendar, curves=curves, marks={}, analytics_by_key={}, liquidity={}, stress_series=stress_series, fixings=fixings, factors=factors, index_ratios=ratios, ratings=ratings, actions=actions, benchmark=benchmark, new_issues=self.new_issues, curve_name=self.curve_name,
                                reference_policy=self.reference_policy, calibration_log=log, curve_status=curve_status, ois_curves=ois, max_stale_curve_days=self.max_stale_curve_days, fx_rates=fx, cpi=cpi, cpi_base=cpi_base, daily_fixings=daily_fixings, tranche_payments=tranche_payments, support=self.support, base_currency=self.base_currency)

        problems = self._validate_dependencies(env)
        env.validation_problems = problems
        if problems:
            if self.strict:
                raise UnsupportedCombination("Run rejected by dependency validation:\n  - " + "\n  - ".join(problems))
            for p in problems:
                sym = p.split(" ")[0]
                if sym in self.bonds:
                    env.unsupported_symbols.add(sym)

        # observations indexed by (symbol, date)
        ev_by: dict[tuple[str, date], EvaluatedObs] = {}
        if self.evaluated is not None and not self.evaluated.empty:
            ev = self.evaluated
            pub = pd.to_datetime(ev["publish_ts"], utc=True) if "publish_ts" in ev else None
            for i, r in enumerate(ev.itertuples(index=False)):
                d = _date(r.date)
                if pub is not None and pub.iloc[i].value > self.schedule.publish_ns(d):
                    continue
                ev_by[(str(r.instrument_id), d)] = EvaluatedObs(float(r.mid), getattr(r, "bid", None) if "bid" in ev else None, getattr(r, "ask", None) if "ask" in ev else None, float(r.score) if "score" in ev and not pd.isna(r.score) else None, str(getattr(r, "source", "EVALUATED")) if "source" in ev else "EVALUATED")
        comp_by: dict[tuple[str, date], QuoteObs] = {}
        if self.composite is not None and not self.composite.empty:
            for r in self.composite.itertuples(index=False):
                ts = _utc(r.ts)
                d = ts.date()
                comp_by[(str(r.instrument_id), d)] = QuoteObs(float(r.bid), float(r.ask), r.ts, str(getattr(r, "source", "COMPOSITE")) if "source" in self.composite else "COMPOSITE")
                if ts.time() <= self.schedule.window_time:
                    env.window_quotes[(str(r.instrument_id), d)] = (float(r.bid), float(r.ask))
        trades_by: dict[tuple[str, date], list[TradeObs]] = {}
        liq = liq_w = pd.DataFrame()
        if self.trades is not None and not self.trades.empty:
            t = self.trades
            avail = pd.to_datetime(t["available_ts"], utc=True)
            for i, r in enumerate(t.itertuples(index=False)):
                d = avail.iloc[i].date()
                if avail.iloc[i].value > self.schedule.mark_ns(d):
                    continue
                size = float(r.public_size) if "public_size" in t else float(r.size)
                trades_by.setdefault((str(r.instrument_id), d), []).append(TradeObs(r.exec_ts, float(r.price), size, str(r.side)))
            didx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
            liq = daily_liquidity_stats(t, didx)
            liq_w = daily_liquidity_stats(t, didx, cutoff_time=self.schedule.window_time)
        liq_by = {(str(r.instrument_id), _date(r.date)): (float(r.volume_5d), int(r.trade_count_20d), int(r.days_since_trade)) for r in liq.itertuples(index=False)} if not liq.empty else {}
        liqw_by = {(str(r.instrument_id), _date(r.date)): (float(r.volume_5d), int(r.trade_count_20d), int(r.days_since_trade)) for r in liq_w.itertuples(index=False)} if not liq_w.empty else {}

        prev_marks: dict[str, MarkResult] = {}
        last_mark_date: dict[str, date] = {}
        prev_analytics: dict[str, tuple[date, BondAnalytics]] = {}
        prev_curve: ZeroCurve | None = None
        prev_bucket_spread: dict[str, float] = {}
        for d in dates:
            curve = env.curve(d)
            alive = env.alive_symbols(d)
            day_marks: dict[str, MarkResult] = {}
            pending: list[str] = []
            for s in alive:
                bond = self.bonds[s]
                mk = self.waterfall.mark(bond, d, MarkObservation(trades=trades_by.get((s, d), []), composite=comp_by.get((s, d)), evaluated=ev_by.get((s, d))))
                if mk is not None and mk.tier.rank <= MarkTier.EVALUATED.rank:
                    day_marks[s] = mk
                else:
                    pending.append(s)
            obs_for_fit: list[SpreadObservation] = []
            for s, mk in day_marks.items():
                an = self._analytics(env, s, d, mk, curve, prev_analytics)
                if an is not None and an.z_spread_bp is not None and an.mod_duration is not None and curve is not None:
                    obs_for_fit.append(SpreadObservation(self.bonds[s], an.z_spread_bp, an.mod_duration, max(mk.quality, 0.05), env.rating_score(s, d)))
            if curve is not None and obs_for_fit:
                self.matrix.fit(d, obs_for_fit)
            for s in pending:
                bond = self.bonds[s]
                last = prev_marks.get(s)
                last_an = prev_analytics.get(s, (None, None))[1]
                model_px = None
                if curve is not None and obs_for_fit:
                    dur = last_an.mod_duration if last_an and last_an.mod_duration is not None else (bond.remaining_years(d) or 5.0) * 0.8
                    res = self.matrix.price(bond, env.settle_date(bond, d), curve, dur, env.rating_score(s, d), env.market_inputs(s, d))
                    if res is not None and res[0] > 0:
                        model_px = res[0]
                rolled = None
                if self.roll_stale and last is not None and curve is not None and prev_curve is not None:
                    bucket_now = self.matrix.bucket_spread(bond, last_an.mod_duration if last_an and last_an.mod_duration else 5.0, env.rating_score(s, d)) if obs_for_fit else None
                    rolled = roll_mark(last.clean_mid, last_an.mod_duration if last_an else None, last_an.spread_duration if last_an else None, bond.remaining_years(d), last_an.ytm if last_an else None, bond.coupon_rate_on(d), RollInputs(prev_curve, curve, prev_bucket_spread.get(s), bucket_now, max(1, (d - last_mark_date.get(s, d)).days)), int(bond.frequency) or 2)
                    if bucket_now is not None:
                        prev_bucket_spread[s] = bucket_now
                elif last is not None:
                    rolled = last.clean_mid
                mk = self.waterfall.mark(bond, d, MarkObservation(trades=trades_by.get((s, d), []), composite=comp_by.get((s, d)), evaluated=ev_by.get((s, d)), model_price=model_px, last_mark=last, rolled_last_mark=rolled))
                if mk is not None:
                    day_marks[s] = mk
                    self._analytics(env, s, d, mk, curve, prev_analytics)
            for s, mk in day_marks.items():
                env.marks[(s, d)] = mk
                prev_marks[s] = mk
                last_mark_date[s] = d
                an = env.analytics_by_key.get((s, d))
                if an is not None and curve is not None and an.z_spread_bp is not None and an.mod_duration is not None:
                    bs = self.matrix.bucket_spread(self.bonds[s], an.mod_duration, env.rating_score(s, d))
                    if bs is not None:
                        prev_bucket_spread[s] = bs
                quality = mk.quality if mk.tier in (MarkTier.EVALUATED, MarkTier.MODEL) else (0.9 if mk.tier in (MarkTier.TRADE, MarkTier.COMPOSITE) else 0.2)
                v, n, since = liq_by.get((s, d), (0.0, 0, 9_999))
                env.liquidity[(s, d)] = LiquidityStats(v, n, since, quality, self.bonds[s].amount_outstanding, mk.tier == MarkTier.COMPOSITE)
                vw, nw, sw = liqw_by.get((s, d), (0.0, 0, 9_999))
                env.liquidity_window[(s, d)] = LiquidityStats(vw, nw, sw, quality, self.bonds[s].amount_outstanding, (s, d) in env.window_quotes)
            if curve is not None:
                prev_curve = curve
        return env

    def _analytics(self, env: MarketEnvironment, s: str, d: date, mk: MarkResult, curve: ZeroCurve | None, prev: dict[str, tuple[date, BondAnalytics]]) -> BondAnalytics | None:
        bond = self.bonds[s]
        last = prev.get(s)
        n = self.analytics_cfg.every_n_days
        if last is not None and n > 1 and (d - last[0]).days < n and abs(last[1].clean_price - mk.clean_mid) < 1.0:
            an = last[1]
        else:
            try:
                an = analyse(bond, env.settle_date(bond, d), mk.clean_mid, curve, env.market_inputs(s, d), env.factor(s, d), self.analytics_cfg.krd_tenors)
            except Exception:  # noqa: BLE001 - one bond must not break the build
                return None
            prev[s] = (d, an)
        env.analytics_by_key[(s, d)] = an
        return an


__all__ = ["TIER_EVIDENCE", "ActionRecord", "AnalyticsConfig", "MarketEnvironment", "MarketEnvironmentBuilder"]
