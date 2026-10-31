"""Run modes, deployability claims, status and the experiment manifest (blueprint sections 3, 22, 25).

A run must identify its mode and prevent inconsistent combinations (a counterfactual that changes
prepayments but keeps historical balances). Computational status is recorded separately from the
economic outcome: a completed run may demonstrate insolvency, which is a valid finding; an engine
failure must never produce an ordinary performance chart.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum

import pandas as pd


class RunMode(str, Enum):
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"                # observed market, modelled decisions/fills -> conditional implementability
    HISTORICAL_COUNTERFACTUAL = "HISTORICAL_COUNTERFACTUAL"  # declared subset of drivers changed -> sensitivity
    STOCHASTIC_SIMULATION = "STOCHASTIC_SIMULATION"        # joint future paths -> model-conditional distribution
    VALUATION_COMPARISON = "VALUATION_COMPARISON"          # same contract, alternative engines -> model disagreement
    FORECAST_EVALUATION = "FORECAST_EVALUATION"            # archived forecasts vs outcomes


class DeployabilityClaim(str, Enum):
    HISTORICALLY_DEPLOYABLE = "HISTORICALLY_DEPLOYABLE"        # methods, data access and operations existed at the time
    RETROSPECTIVE_RECONSTRUCTION = "RETROSPECTIVE_RECONSTRUCTION"  # today's model tested on old data


class ComputationalStatus(str, Enum):
    COMPLETED = "COMPLETED"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    UNSUPPORTED_FEATURES = "UNSUPPORTED_FEATURES"
    MISSING_REQUIRED_INPUTS = "MISSING_REQUIRED_INPUTS"
    ENGINE_FAILURE = "ENGINE_FAILURE"
    NOT_RUN = "NOT_RUN"


class EconomicOutcome(str, Enum):
    SOLVENT = "SOLVENT"
    FUNDING_DEFAULT = "FUNDING_DEFAULT"
    INSOLVENT = "INSOLVENT"
    MANDATE_INFEASIBLE = "MANDATE_INFEASIBLE"
    UNDETERMINED = "UNDETERMINED"


class ProbabilityMeasure(str, Enum):
    RISK_NEUTRAL = "RISK_NEUTRAL"
    REAL_WORLD = "REAL_WORLD"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class CounterfactualSpec:
    """Dependency rules for counterfactuals (section 3): changed drivers must recompute or freeze
    every dependent channel explicitly; anything else is rejected as inconsistent."""

    changed_drivers: tuple[str, ...] = ()
    recompute: tuple[str, ...] = ()
    frozen: tuple[str, ...] = ()

    # known downstream channels per driver
    DEPENDENCIES = {
        "rates": ("coupons_floating", "prepayments", "call_exercise", "marks", "margin", "financing"),
        "spreads": ("marks", "margin", "forced_sales", "defaults"),
        "prepayments": ("pool_balances", "coupons", "marks", "reinvestment"),
        "defaults": ("cashflows", "marks", "recoveries", "margin"),
        "fx": ("marks", "hedge_pnl", "margin"),
        "volatility": ("marks", "call_exercise"),
    }

    def validate(self) -> list[str]:
        problems = []
        declared = set(self.recompute) | set(self.frozen)
        for d in self.changed_drivers:
            for ch in self.DEPENDENCIES.get(d, ()):
                if ch not in declared:
                    problems.append(f"driver {d!r} affects {ch!r}: declare it in recompute or frozen")
        both = set(self.recompute) & set(self.frozen)
        if both:
            problems.append(f"channels both recomputed and frozen: {sorted(both)}")
        return problems


def hash_frame(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    h = pd.util.hash_pandas_object(df.reset_index(drop=True).astype(str), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def hash_object(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_sha(path: str | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # pragma: no cover
        return None


def capture_environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("nautilus_trader", "numpy", "pandas", "pyarrow", "QuantLib"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "?")
        except Exception:
            env[pkg] = None
    try:
        from nautilus_fi.version import __version__

        env["nautilus_fi"] = __version__
    except Exception:  # pragma: no cover
        pass
    return env


@dataclass
class DataVersion:
    name: str
    rows: int
    content_hash: str
    knowledge_time: str | None = None
    vintage: str = ""
    source: str = ""
    availability_policy: str = ""


@dataclass
class BenchmarkContract:
    name: str
    return_type: str = "TOTAL_RETURN"       # TOTAL_RETURN | PRICE_RETURN
    currency: str = "USD"
    hedged: bool = False
    reinvestment: str = "DAILY"
    fees_bp: float = 0.0
    rebalance: str = "MONTHLY"
    constituents: str = "PUBLISHED"         # PUBLISHED | SIMULATED_INVESTABLE
    notes: str = ""


@dataclass
class ExperimentManifest:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    hypothesis: str = ""
    run_mode: RunMode = RunMode.HISTORICAL_REPLAY
    deployability: DeployabilityClaim = DeployabilityClaim.RETROSPECTIVE_RECONSTRUCTION
    measure: ProbabilityMeasure = ProbabilityMeasure.NOT_APPLICABLE
    universe: str = ""
    start: str = ""
    end: str = ""
    base_currency: str = "USD"
    capital: float = 0.0
    benchmark: BenchmarkContract | None = None
    decision_schedule: str = ""
    event_precedence: tuple[str, ...] = ()
    reference_policy: str = ""
    terminal_policy: str = "MARK_AT_END"
    data_versions: list[DataVersion] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)          # role -> model@version
    assumptions: list[dict] = field(default_factory=list)
    support: list[dict] = field(default_factory=list)
    counterfactual: CounterfactualSpec | None = None
    seeds: dict[str, int] = field(default_factory=dict)
    conventions: dict = field(default_factory=dict)                 # domain.conventions.Conventions.to_dict()
    spec: dict = field(default_factory=dict)                        # research spec summary and hash when run from a spec
    code_version: str | None = field(default_factory=git_sha)
    environment: dict = field(default_factory=capture_environment)
    owner: str = ""
    output_hashes: dict[str, str] = field(default_factory=dict)
    computational_status: ComputationalStatus = ComputationalStatus.NOT_RUN
    economic_outcome: EconomicOutcome = EconomicOutcome.UNDETERMINED
    messages: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def validate(self) -> list[str]:
        problems = []
        if self.run_mode == RunMode.HISTORICAL_COUNTERFACTUAL:
            if self.counterfactual is None:
                problems.append("counterfactual run needs a CounterfactualSpec")
            else:
                problems += self.counterfactual.validate()
        if self.run_mode == RunMode.STOCHASTIC_SIMULATION and self.measure == ProbabilityMeasure.NOT_APPLICABLE:
            problems.append("stochastic simulation must label its probability measure")
        if not self.start or not self.end:
            problems.append("start and end dates are required")
        return problems

    def add_data(self, name: str, df: pd.DataFrame, knowledge_time: str | None = None, vintage: str = "", source: str = "", availability_policy: str = "") -> DataVersion:
        dv = DataVersion(name, len(df) if df is not None else 0, hash_frame(df), knowledge_time, vintage, source, availability_policy)
        self.data_versions.append(dv)
        return dv

    def record_output(self, name: str, df: pd.DataFrame) -> None:
        self.output_hashes[name] = hash_frame(df)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["run_mode"] = self.run_mode.value
        d["deployability"] = self.deployability.value
        d["measure"] = self.measure.value
        d["computational_status"] = self.computational_status.value
        d["economic_outcome"] = self.economic_outcome.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @property
    def config_hash(self) -> str:
        d = self.to_dict()
        for k in ("run_id", "created_utc", "output_hashes", "computational_status", "economic_outcome", "messages", "timings", "environment", "code_version"):
            d.pop(k, None)
        return hash_object(d)


__all__ = [
    "BenchmarkContract",
    "ComputationalStatus",
    "CounterfactualSpec",
    "DataVersion",
    "DeployabilityClaim",
    "EconomicOutcome",
    "ExperimentManifest",
    "ProbabilityMeasure",
    "RunMode",
    "capture_environment",
    "git_sha",
    "hash_frame",
    "hash_object",
]
