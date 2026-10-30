"""Engine-side wiring handle and the services a strategy may use (no environment access)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from nautilus_fi.accounting.ledger import FixedIncomeLedger
from nautilus_fi.backtest.environment import MarketEnvironment
from nautilus_fi.core.context import DecisionContext
from nautilus_fi.core.events import EventLog
from nautilus_fi.core.manifest import ExperimentManifest
from nautilus_fi.execution.assumptions import ExecutionAssumptions
from nautilus_fi.execution.fill_model import FixedIncomeFillModel
from nautilus_fi.execution.windows import ExecutionWindowSchedule
from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.calendar import BondCalendar
from nautilus_fi.instruments.cashflows import MarketInputs
from nautilus_fi.portfolio.construction import PortfolioConstructor
from nautilus_fi.portfolio.forced_actions import ForcedActionPolicy
from nautilus_fi.universe.universe import PointInTimeUniverse


@dataclass
class StrategyServices:
    """What a strategy is allowed to touch besides its ``DecisionContext``: contract terms, calendars,
    the portfolio constructor, and point-in-time callbacks whose results are knowable at the decision time."""

    bonds: dict[str, BondSpec]
    calendar: BondCalendar
    schedule: ExecutionWindowSchedule
    assumptions: ExecutionAssumptions
    venue: str
    base_currency: str
    constructor: PortfolioConstructor
    forced_actions: ForcedActionPolicy | None
    market_inputs: Callable[[str, date], MarketInputs]
    settle_date: Callable[[BondSpec, date], date]
    record_decision: Callable[[str, float, date], None]
    terms_known_as_of: Callable[[str, date], BondSpec] | None = None


@dataclass
class FixedIncomeContext:
    """Engine-side bundle (never handed to strategies)."""

    env: MarketEnvironment
    ledger: FixedIncomeLedger
    fill_model: FixedIncomeFillModel
    assumptions: ExecutionAssumptions
    universe: PointInTimeUniverse
    decision: DecisionContext
    services: StrategyServices
    manifest: ExperimentManifest
    event_log: EventLog
    venue: str = "OTC"
    base_currency: str = "USD"
    restricted: set[str] = field(default_factory=set)


__all__ = ["FixedIncomeContext", "StrategyServices"]
