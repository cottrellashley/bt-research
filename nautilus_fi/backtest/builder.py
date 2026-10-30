"""Wire the engines into a Nautilus ``BacktestEngine`` and run under a manifest (blueprint section 25).

The same sequence every time: validate configuration, support and dependency graph; freeze data and
model references in the manifest; build the engine; run; record computational status separately
from the economic outcome; hash outputs; return a result that carries its manifest, event log,
ledger, reconciliation, evidence and invariant checks.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.common import LoggerConfig, LogLevel
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.model import AccountType, BookType, Currency, Money, OmsType, TraderId, Venue
from nautilus_trader.portfolio import PortfolioConfig
from nautilus_trader.risk import RiskEngineConfig

from nautilus_fi.accounting.cashflow_module import FixedIncomeCashflowModule
from nautilus_fi.accounting.financing import CashRateModel, FinancingTerms
from nautilus_fi.accounting.ledger import FixedIncomeLedger
from nautilus_fi.accounting.postings import AccountingSettings
from nautilus_fi.backtest.actor import FixedIncomeEnvironmentActor
from nautilus_fi.backtest.context import FixedIncomeContext, StrategyServices
from nautilus_fi.backtest.environment import MarketEnvironment
from nautilus_fi.backtest.result import FixedIncomeResult
from nautilus_fi.backtest.strategy import FixedIncomeStrategy
from nautilus_fi.core.assumptions import AssumptionSet
from nautilus_fi.core.context import DecisionContext
from nautilus_fi.core.events import EventLog
from nautilus_fi.core.manifest import (
    BenchmarkContract,
    ComputationalStatus,
    DeployabilityClaim,
    EconomicOutcome,
    ExperimentManifest,
    RunMode,
)
from nautilus_fi.core.registry import SupportStatus, UnsupportedCombination
from nautilus_fi.core.time import DataVintage
from nautilus_fi.data.types import register_types
from nautilus_fi.execution.assumptions import NET_FULL, ExecutionAssumptions
from nautilus_fi.execution.costs import CostModel, ParametricCostModel
from nautilus_fi.execution.fill_model import FixedIncomeFillModel
from nautilus_fi.execution.liquidity import LiquidityModel, VolumeGatedLiquidityModel
from nautilus_fi.execution.process import ProcessConfig, ReferencePolicy
from nautilus_fi.funding.policy import FundingPolicy
from nautilus_fi.instruments.enums import AssetType
from nautilus_fi.lifecycle.state_machine import PRECEDENCE
from nautilus_fi.portfolio.constraints import Constraint, ConstraintSet
from nautilus_fi.portfolio.construction import PortfolioConstructor
from nautilus_fi.portfolio.forced_actions import ForcedActionPolicy
from nautilus_fi.universe.universe import PointInTimeUniverse, UniverseFilter

FAMILY = {AssetType.SOVEREIGN: ("SOVEREIGN", "fixed_coupon"), AssetType.BILL: ("BILL", "discount_yield_quotation"), AssetType.AGENCY: ("SOVEREIGN", "fixed_coupon"), AssetType.SUPRANATIONAL: ("SOVEREIGN", "fixed_coupon"), AssetType.CORPORATE: ("CORPORATE", "fixed_step_zero"), AssetType.MUNICIPAL: ("MUNICIPAL", "tax_treatment"), AssetType.COVERED: ("CORPORATE", "fixed_step_zero"), AssetType.PREFERRED: ("PREFERRED", "discretionary_call"), AssetType.LOAN: ("LOAN", "drawn_undrawn_fees"), AssetType.LINKER: ("LINKER", "indexation_us_tips"), AssetType.MBS: ("MBS", "pool_factors_paydown"), AssetType.CMBS: ("CMBS", "tranche_cashflows"), AssetType.ABS: ("ABS", "tranche_cashflows"), AssetType.CLO: ("CLO", "waterfall_triggers_reinvestment"), AssetType.OTHER: ("CORPORATE", "fixed_step_zero")}


@dataclass
class BacktestSettings:
    venue: str = "OTC"
    starting_cash: float = 100_000_000.0
    base_currency: str = "USD"
    account_type: str = "CASH"
    leverage: float = 1.0
    trader_id: str = "FI-BACKTESTER-001"
    log_level: str | None = None
    risk_bypass: bool = False
    use_mark_prices: bool = True
    include_nautilus_reports: bool = True
    include_marks: bool = False
    run_mode: RunMode = RunMode.HISTORICAL_REPLAY
    deployability: DeployabilityClaim = DeployabilityClaim.RETROSPECTIVE_RECONSTRUCTION
    hypothesis: str = ""
    owner: str = ""
    allow_support: SupportStatus = SupportStatus.EXPERIMENTAL
    terminal_policy: str = "MARK_AT_END"
    funding_policy: FundingPolicy = field(default_factory=FundingPolicy)
    process_config: ProcessConfig | None = None
    constraints: list[Constraint] = field(default_factory=list)
    forced_actions: ForcedActionPolicy | None = None
    accounting: AccountingSettings = field(default_factory=AccountingSettings)
    benchmark: BenchmarkContract | None = None
    restricted: set[str] = field(default_factory=set)
    assumptions_set: AssumptionSet | None = None


class FixedIncomeBacktest:
    def __init__(self, env: MarketEnvironment, strategy: FixedIncomeStrategy | Callable[[], FixedIncomeStrategy], assumptions: ExecutionAssumptions = NET_FULL, liquidity_model: LiquidityModel | None = None, cost_model: CostModel | None = None, cash_rate: CashRateModel | None = None, financing: FinancingTerms | None = None, settings: BacktestSettings | None = None, name: str | None = None) -> None:
        self.env = env
        self._strategy = strategy
        self.assumptions = assumptions
        self.liquidity_model = liquidity_model or VolumeGatedLiquidityModel()
        self.cost_model = cost_model or ParametricCostModel()
        self.cash_rate = cash_rate
        self.financing = financing
        self.settings = settings or BacktestSettings()
        self.name = name or assumptions.name
        self.engine: BacktestEngine | None = None
        self.ledger: FixedIncomeLedger | None = None
        self.fill_model: FixedIncomeFillModel | None = None
        self.manifest: ExperimentManifest | None = None
        self.ctx: FixedIncomeContext | None = None

    # ------------------------------------------------------------------ manifest
    def _build_manifest(self) -> ExperimentManifest:
        s, env = self.settings, self.env
        m = ExperimentManifest(hypothesis=s.hypothesis, run_mode=s.run_mode, deployability=s.deployability, universe=f"{len(env.bonds)} instruments / {len(env.dates)} trading days", start=env.start_date.isoformat(), end=env.end_date.isoformat(), base_currency=s.base_currency, capital=s.starting_cash, benchmark=s.benchmark, decision_schedule=f"decision {env.schedule.decision_time} after publication {env.schedule.mark_publish_time}; execution window {env.schedule.window_time} next business day", event_precedence=tuple(k.value for k, _ in sorted(PRECEDENCE.items(), key=lambda kv: kv[1])), reference_policy=env.reference_policy, terminal_policy=s.terminal_policy, owner=s.owner, seeds={"execution": self.assumptions.seed})
        m.models = {"waterfall": "PricingWaterfall@1", "matrix": "MatrixPricer@1", "stale_roll": "roll_mark@1", "liquidity": f"{type(self.liquidity_model).__name__}@1", "cost": f"{type(self.cost_model).__name__}@1", "execution": "OrderProcessSimulator@1", "pricing": "DiscountingEngine@1.0 / HullWhiteTreeEngine@0.1", "accounting": "StandardAccountingPolicy@1", "assumptions": self.assumptions.name}
        m.add_data("marks", env.marks_frame(), knowledge_time=env.end_date.isoformat(), availability_policy="publish time (ts_init) per mark; prior-close execution reference" if env.reference_policy == "PRIOR_CLOSE" else env.reference_policy)
        m.add_data("coverage", env.coverage_frame(), availability_policy="derived")
        aset = s.assumptions_set or AssumptionSet()
        aset.default("execution.cost_multiplier", self.assumptions.cost_multiplier, "x", "ExecutionAssumptions")
        aset.default("execution.min_delay_business_days", self.assumptions.min_delay_business_days, "days", "ExecutionAssumptions")
        aset.default("execution.max_order_age_business_days", self.assumptions.max_order_age_business_days, "days", "ExecutionAssumptions")
        aset.default("execution.apply_fill_probability", self.assumptions.apply_fill_probability, "bool", "ExecutionAssumptions")
        aset.default("execution.reference_policy", env.reference_policy, "policy", "MarketEnvironmentBuilder")
        aset.default("funding.allow_bridge", s.funding_policy.allow_bridge, "bool", "FundingPolicy")
        aset.default("funding.settle_before_spend", s.funding_policy.settle_before_spend, "bool", "FundingPolicy")
        aset.default("curves.max_stale_days", env.max_stale_curve_days, "days", "MarketEnvironment")
        aset.default("marks.prefer_model_after_stale_days", 3, "days", "WaterfallConfig")
        if hasattr(self.cost_model, "base_bp"):
            for k, v in self.cost_model.base_bp.items():
                aset.product_default(k.value, "cost.base_half_spread_bp", v, "bp", source="ParametricCostModel default (uncalibrated)", low=v * 0.5, high=v * 2.0)
        if hasattr(self.liquidity_model, "volume_threshold"):
            aset.default("liquidity.volume_threshold", self.liquidity_model.volume_threshold, "face", "VolumeGatedLiquidityModel", low=100_000, high=1_000_000)
            aset.default("liquidity.cap_fraction", self.liquidity_model.cap_fraction, "fraction of 5d volume", "VolumeGatedLiquidityModel", low=0.05, high=0.5)
        m.assumptions = aset.export_resolved().to_dict("records")
        families = {FAMILY.get(b.asset_type, ("CORPORATE", "fixed_step_zero")) for b in env.bonds.values()}
        caps = [env.support.status(f, feat, run_mode=s.run_mode.value) for f, feat in sorted(families)]
        caps.append(env.support.status("PLATFORM", "point_in_time_replay"))
        caps.append(env.support.status("PLATFORM", "execution_process"))
        m.support = [{"product_family": c.product_family, "feature": c.feature, "status": c.status.value, "approximation": c.approximation, "validation_case": c.validation_case, "notes": c.notes} for c in caps]
        return m

    # ------------------------------------------------------------------ build
    def build(self) -> BacktestEngine:
        register_types()
        s = self.settings
        self.manifest = self._build_manifest()
        problems = self.manifest.validate()
        if problems:
            self.manifest.computational_status = ComputationalStatus.INVALID_CONFIGURATION
            self.manifest.messages += problems
            raise UnsupportedCombination("Invalid configuration: " + "; ".join(problems))
        blocked = [c for c in self.manifest.support if SupportStatus(c["status"]).rank > s.allow_support.rank]
        if blocked:
            self.manifest.computational_status = ComputationalStatus.UNSUPPORTED_FEATURES
            self.manifest.messages += [f"{c['product_family']}/{c['feature']}: {c['status']}" for c in blocked]
            raise UnsupportedCombination("Unsupported features requested: " + "; ".join(self.manifest.messages))
        if self.env.validation_problems and self.env.unsupported_symbols:
            self.manifest.messages.append(f"{len(self.env.unsupported_symbols)} instruments excluded by dependency validation")
        logging = LoggerConfig(bypass_logging=True) if s.log_level is None else LoggerConfig(stdout_level=getattr(LogLevel, s.log_level.upper()))
        engine = BacktestEngine(BacktestEngineConfig(trader_id=TraderId(s.trader_id), logging=logging, portfolio=PortfolioConfig(use_mark_prices=s.use_mark_prices), risk_engine=RiskEngineConfig(bypass=s.risk_bypass), run_analysis=False))
        venue = Venue(s.venue)
        ccy = Currency.from_str(s.base_currency)
        event_log = EventLog()
        multi_ccy = any(b.currency != s.base_currency for b in self.env.bonds.values())
        self.fill_model = FixedIncomeFillModel(self.env, self.assumptions, self.liquidity_model, self.cost_model, s.process_config or ProcessConfig(reference_policy=ReferencePolicy(self.env.reference_policy), restricted=set(s.restricted)), new_issues=self.env.new_issues, event_log=event_log)
        self.ledger = FixedIncomeLedger(self.env, s.base_currency, self.cash_rate, self.financing, s.funding_policy, s.accounting, event_log, self.fill_model)
        if s.funding_policy.settle_before_spend:
            self.fill_model.available_cash = self.ledger.available_cash
        module = FixedIncomeCashflowModule(self.ledger)
        margin = s.account_type.upper() == "MARGIN"
        balances = [Money(s.starting_cash, ccy)] + ([Money(0, Currency.from_str(c)) for c in sorted({b.currency for b in self.env.bonds.values()} - {s.base_currency})] if multi_ccy else [])
        engine.add_venue(venue=venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN if margin else AccountType.CASH, base_currency=None if multi_ccy else ccy, starting_balances=balances, default_leverage=Decimal(str(s.leverage)) if margin else None, fill_model=self.fill_model, fee_model=FixedFeeModel(Money(0, ccy)), modules=[module], book_type=BookType.L1_MBP, support_gtd_orders=True, allow_cash_borrowing=multi_ccy)
        for inst in self.env.instruments(s.venue):
            engine.add_instrument(inst)
        engine.add_data(self.env.nautilus_data(s.venue))
        self.ledger.bind(engine.cache, venue)
        self.fill_model.cache = engine.cache
        universe = PointInTimeUniverse(self.env)
        decision = DecisionContext("PM", DataVintage(None), self.env.support, dict(self.manifest.models), s.base_currency, s.run_mode.value)
        decision._bind(universe_fn=lambda d: universe.symbols(d, UniverseFilter(require_mark=False)), tradable_fn=lambda sym, d: (False, "terminated") if self.env.is_terminated(sym, d) else (False, "unsupported") if sym in self.env.unsupported_symbols else (True, "ok"), restricted=set(s.restricted))
        services = StrategyServices(self.env.bonds, self.env.calendar, self.env.schedule, self.assumptions, s.venue, s.base_currency, PortfolioConstructor(ConstraintSet(list(s.constraints)), cash_buffer=0.01, settle_before_spend=s.funding_policy.settle_before_spend), s.forced_actions, self.env.market_inputs, self.env.settle_date, lambda oid, mid, d: self.ledger.decision_marks.__setitem__(oid, (mid, d)))
        self.ctx = FixedIncomeContext(self.env, self.ledger, self.fill_model, self.assumptions, universe, decision, services, self.manifest, event_log, s.venue, s.base_currency, set(s.restricted))
        engine.add_actor(FixedIncomeEnvironmentActor(self.ctx))
        strategy = self._strategy() if callable(self._strategy) and not isinstance(self._strategy, FixedIncomeStrategy) else self._strategy
        strategy.attach(decision, services)
        engine.add_strategy(strategy)
        self.engine = engine
        return engine

    # ------------------------------------------------------------------ run
    def run(self) -> FixedIncomeResult:
        t0 = time.perf_counter()
        engine = self.build()
        s = self.settings
        venue = Venue(s.venue)
        m = self.manifest
        acct = fills = positions = None
        try:
            engine.run()
            m.computational_status = ComputationalStatus.COMPLETED
            if s.include_nautilus_reports:
                acct = engine.generate_account_report(venue)
                fills = engine.generate_order_fills_report()
                positions = engine.generate_positions_report()
        except Exception as e:  # noqa: BLE001 - the failure is recorded, never a silent performance chart
            m.computational_status = ComputationalStatus.ENGINE_FAILURE
            m.messages.append(f"{type(e).__name__}: {e}")
        finally:
            engine.dispose()
        m.timings["run_seconds"] = time.perf_counter() - t0
        nav = self.ledger.nav_frame()
        if m.computational_status == ComputationalStatus.COMPLETED:
            if self.ledger.funding_default:
                m.economic_outcome = EconomicOutcome.FUNDING_DEFAULT
            elif not nav.empty and nav["nav"].iloc[-1] <= 0:
                m.economic_outcome = EconomicOutcome.INSOLVENT
            else:
                m.economic_outcome = EconomicOutcome.SOLVENT
        for name, df in (("nav", nav), ("fills", self.ledger.fills_frame()), ("cashflows", self.ledger.cashflows_frame())):
            m.record_output(name, df)
        m.output_hashes["event_chain"] = self.ledger.events.chain_hash
        result = FixedIncomeResult(name=self.name, assumptions=self.assumptions, nav=nav, fills=self.ledger.fills_frame(), cashflows=self.ledger.cashflows_frame(), holdings=self.ledger.holdings_frame(), decisions=self.fill_model.decisions_frame(), account_report=acct, fills_report=fills, positions_report=positions, benchmark=self.env.benchmark, marks=self.env.marks_frame() if s.include_marks else None,
                                   manifest=m, events=self.ledger.events_frame(), entries=self.ledger.entries_frame(), reconciliation=self.ledger.reconciliation_frame(), support=pd.DataFrame(m.support), coverage=self.env.coverage_frame(), funding_events=pd.DataFrame(self.ledger.funding_events), calibration=self.env.calibration_log.to_frame(), funding_rejects=pd.DataFrame(self.fill_model.funding_rejects), event_log=self.ledger.events, ledger=self.ledger.ledger,
                                   fx_end={c: (self.env.fx(c, self.env.end_date) or 1.0) for c in {b.currency for b in self.env.bonds.values()} if c != s.base_currency})
        return result


__all__ = ["BacktestSettings", "FixedIncomeBacktest"]
