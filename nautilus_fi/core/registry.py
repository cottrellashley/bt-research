"""Support matrix, model registration and dependency validation (blueprint sections 1 and 8).

A product is *supported* only when its terms, lifecycle, data, valuation, execution, accounting and
validation are supported for the requested use. Every capability carries a status that users see
before a run and in its results. Each model declares its contract; the dependency graph of a run is
validated before any computation (missing fixings for a floater, no exercise model for a callable,
unit mismatches, unresolved cycles) and unsupported combinations are rejected.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class SupportStatus(str, Enum):
    VALIDATED = "VALIDATED"
    SUPPORTED_WITH_APPROXIMATION = "SUPPORTED_WITH_APPROXIMATION"
    EXPERIMENTAL = "EXPERIMENTAL"
    UNSUPPORTED = "UNSUPPORTED"

    @property
    def rank(self) -> int:
        return list(SupportStatus).index(self)


@dataclass(frozen=True)
class Capability:
    product_family: str                 # e.g. CORPORATE, CLO, TBA, IRS
    feature: str                        # e.g. "partial_call", "waterfall_triggers", "fx_hedge"
    status: SupportStatus
    jurisdictions: tuple[str, ...] = ("US",)
    currencies: tuple[str, ...] = ("USD",)
    period: tuple[str, str] | None = None            # (from, to) ISO dates of validated history
    run_modes: tuple[str, ...] = ("HISTORICAL_REPLAY",)
    data_requirements: tuple[str, ...] = ()
    model: str = ""
    approximation: str = ""                          # what is approximated, if status says so
    validation_case: str = ""                        # the critical validation fixture that covers it
    notes: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.product_family, self.feature)


class SupportMatrix:
    def __init__(self) -> None:
        self._caps: dict[tuple[str, str], Capability] = {}

    def declare(self, cap: Capability) -> Capability:
        self._caps[cap.key] = cap
        return cap

    def status(self, product_family: str, feature: str, jurisdiction: str = "US", currency: str = "USD", run_mode: str = "HISTORICAL_REPLAY") -> Capability:
        cap = self._caps.get((product_family, feature))
        if cap is None:
            return Capability(product_family, feature, SupportStatus.UNSUPPORTED, notes="not declared")
        if cap.jurisdictions and jurisdiction not in cap.jurisdictions and "*" not in cap.jurisdictions:
            return Capability(product_family, feature, SupportStatus.UNSUPPORTED, notes=f"jurisdiction {jurisdiction} not covered")
        if cap.currencies and currency not in cap.currencies and "*" not in cap.currencies:
            return Capability(product_family, feature, SupportStatus.UNSUPPORTED, notes=f"currency {currency} not covered")
        if cap.run_modes and run_mode not in cap.run_modes and "*" not in cap.run_modes:
            return Capability(product_family, feature, SupportStatus.UNSUPPORTED, notes=f"run mode {run_mode} not covered")
        return cap

    def check(self, requested: Iterable[tuple[str, str]], jurisdiction: str = "US", currency: str = "USD", run_mode: str = "HISTORICAL_REPLAY", allow: SupportStatus = SupportStatus.EXPERIMENTAL) -> list[Capability]:
        """Return the capabilities for the requested features; raise if any is below ``allow``."""
        found = [self.status(f, feat, jurisdiction, currency, run_mode) for f, feat in requested]
        blocked = [c for c in found if c.status.rank > allow.rank]
        if blocked:
            raise UnsupportedCombination("; ".join(f"{c.product_family}/{c.feature}: {c.status.value} ({c.notes})" for c in blocked))
        return found

    def to_frame(self) -> pd.DataFrame:
        rows = [{"product_family": c.product_family, "feature": c.feature, "status": c.status.value, "jurisdictions": ",".join(c.jurisdictions), "currencies": ",".join(c.currencies), "run_modes": ",".join(c.run_modes), "data_requirements": ",".join(c.data_requirements), "model": c.model, "approximation": c.approximation, "validation_case": c.validation_case, "notes": c.notes} for c in self._caps.values()]
        return pd.DataFrame(rows).sort_values(["product_family", "feature"]).reset_index(drop=True) if rows else pd.DataFrame()

    def families(self) -> list[str]:
        return sorted({c.product_family for c in self._caps.values()})

    def __len__(self) -> int:
        return len(self._caps)


class UnsupportedCombination(ValueError):
    pass


# ------------------------------------------------------------------------------------------ models
@dataclass
class ModelRegistration:
    """Blueprint section 8: the registration contract every extension must declare."""

    model_id: str
    version: str
    kind: str                                   # pricing | behavior | execution | funding | curve | mark | risk | cashflow | scenario
    supported_features: tuple[str, ...] = ()
    run_modes: tuple[str, ...] = ("HISTORICAL_REPLAY",)
    input_schema: dict[str, str] = field(default_factory=dict)      # input name -> unit / type
    output_units: dict[str, str] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()           # other model ids / data requirements
    parameters: dict[str, object] = field(default_factory=dict)
    calibration_method: str = ""
    training_cutoff: str | None = None           # ISO date: data after this must not have informed parameters
    missing_data_behavior: str = "reject"        # reject | approximate | carry
    random_streams: tuple[str, ...] = ()
    numerical_tolerances: dict[str, float] = field(default_factory=dict)
    expected_runtime: str = ""
    diagnostics: tuple[str, ...] = ()
    reference_fixtures: tuple[str, ...] = ()
    status: SupportStatus = SupportStatus.EXPERIMENTAL
    outputs_unavailable: tuple[str, ...] = ()    # explicitly unavailable outputs (not zero)

    @property
    def ref(self) -> str:
        return f"{self.model_id}@{self.version}"


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelRegistration] = {}

    def register(self, reg: ModelRegistration) -> ModelRegistration:
        self._models[reg.ref] = reg
        self._models.setdefault(reg.model_id, reg)  # latest alias
        return reg

    def get(self, ref: str) -> ModelRegistration:
        if ref not in self._models:
            raise KeyError(f"Unknown model {ref}")
        return self._models[ref]

    def to_frame(self) -> pd.DataFrame:
        rows = [{"model": r.ref, "kind": r.kind, "status": r.status.value, "features": ",".join(r.supported_features), "dependencies": ",".join(r.dependencies), "calibration": r.calibration_method, "training_cutoff": r.training_cutoff, "missing_data": r.missing_data_behavior} for k, r in self._models.items() if "@" in k]
        return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ dependency graph
@dataclass
class Node:
    id: str
    kind: str                                   # product | model | data | convention | curve
    units: str = ""
    available: bool = True
    detail: str = ""


@dataclass
class Edge:
    src: str                                    # requirer
    dst: str                                    # requirement
    required_units: str = ""
    reason: str = ""


class DependencyGraph:
    """Validate a run's dependency graph before execution (blueprint section 8)."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def require(self, src: str, dst: str, required_units: str = "", reason: str = "") -> Edge:
        e = Edge(src, dst, required_units, reason)
        self.edges.append(e)
        return e

    def validate(self) -> list[str]:
        problems: list[str] = []
        for e in self.edges:
            if e.src not in self.nodes:
                problems.append(f"unknown requirer {e.src}")
            dst = self.nodes.get(e.dst)
            if dst is None:
                problems.append(f"{e.src} requires missing dependency {e.dst} ({e.reason})")
                continue
            if not dst.available:
                problems.append(f"{e.src} requires {e.dst} which is unavailable: {dst.detail or 'no data'} ({e.reason})")
            if e.required_units and dst.units and e.required_units != dst.units:
                problems.append(f"{e.src} expects {e.dst} in {e.required_units} but it is provided in {dst.units}")
        # cycle detection (Kahn)
        indeg = {n: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for e in self.edges:
            if e.src in self.nodes and e.dst in self.nodes:
                adj[e.src].append(e.dst)
                indeg[e.dst] += 1
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if seen < len(self.nodes):
            problems.append("dependency cycle detected; register a coupled solver with convergence diagnostics")
        return problems

    def assert_valid(self) -> None:
        problems = self.validate()
        if problems:
            raise UnsupportedCombination("Run rejected by dependency validation:\n  - " + "\n  - ".join(problems))


__all__ = ["Capability", "DependencyGraph", "Edge", "ModelRegistration", "ModelRegistry", "Node", "SupportMatrix", "SupportStatus", "UnsupportedCombination"]
