"""The restricted decision context (blueprint section 4: make information access enforceable).

Strategies and allocators receive this object, never the market environment. It carries the actor,
the decision clock, the data vintage, positions and orders visible to that actor, the approved model
versions, and point-in-time accessors whose contents are updated by the engine only as information
becomes available. There is deliberately no handle back to the environment or the full history.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from nautilus_fi.core.registry import Capability, SupportMatrix
from nautilus_fi.core.time import DataVintage, date_of


class InformationBoundaryError(PermissionError):
    pass


@dataclass
class VisiblePosition:
    instrument: str
    quantity: float                 # face / units (signed)
    currency: str
    settled_quantity: float         # quantity whose trades have settled
    cost: float
    carrying: float
    accrued: float
    factor: float = 1.0


@dataclass
class VisibleOrder:
    order_id: str
    instrument: str
    side: str
    quantity: float
    leaves: float
    limit: float | None
    status: str
    submitted_ns: int
    expires_ns: int | None


@dataclass
class CashState:
    currency: str
    settled: float
    unsettled_receivable: float
    unsettled_payable: float
    reserved: float                 # reserved for live orders / margin
    collateral_posted: float

    @property
    def available(self) -> float:
        """Spendable without bridge financing: settled cash less committed payables and reservations."""
        return self.settled - self.unsettled_payable - self.reserved

    @property
    def projected(self) -> float:
        return self.settled + self.unsettled_receivable - self.unsettled_payable - self.reserved


_BUILTIN_KINDS = frozenset({"mark", "liquidity", "rating", "fixing", "curve", "factor", "index_ratio", "action", "stress", "fx"})


class DecisionContext:
    """Point-in-time view handed to strategies. All state is pushed by the engine; reads are restricted."""

    def __init__(self, actor: str, vintage: DataVintage, support: SupportMatrix | None = None, approved_models: dict[str, str] | None = None, base_currency: str = "USD", run_mode: str = "HISTORICAL_REPLAY", max_history: int | None = 520) -> None:
        self.actor = actor
        self.vintage = vintage
        self.support = support or SupportMatrix()
        self.approved_models = dict(approved_models or {})
        self.base_currency = base_currency
        self.run_mode = run_mode
        self.max_history = max_history                  # observations kept per instrument / index (None = unbounded)
        self._now_ns: int = 0
        self._marks: dict[str, Any] = {}
        self._mark_history: dict[str, deque] = {}
        self._liquidity: dict[str, Any] = {}
        self._ratings: dict[str, int] = {}
        self._fixings: dict[str, tuple[int, float]] = {}
        self._fixing_history: dict[str, deque] = {}
        self._curves: dict[str, Any] = {}
        self._factors: dict[str, float] = {}
        self._index_ratios: dict[str, float] = {}
        self._actions: list[Any] = []
        self._stress: float = 0.0
        self._fx: dict[str, float] = {}
        self._positions: dict[str, VisiblePosition] = {}
        self._orders: dict[str, VisibleOrder] = {}
        self._cash: dict[str, CashState] = {}
        self._universe_fn: Callable[[date], list[str]] | None = None
        self._tradable_fn: Callable[[str, date], tuple[bool, str]] | None = None
        self._terms_fn: Callable[[str, date], Any] | None = None
        self._nav: float | None = None
        self._funding_shortfall: float = 0.0
        self._restricted: set[str] = set()
        self._fields: dict[str, dict[str, Any]] = {}
        self._rows: dict[str, tuple] = {}
        self._field_ts: dict[str, int] = {}
        self.log: list[str] = []

    # ------------------------------------------------------------------ engine-side updates
    def _set_now(self, ns: int) -> None:
        if ns < self._now_ns:
            raise InformationBoundaryError("Decision clock cannot move backwards")
        self._now_ns = ns

    def _publish(self, kind: str, key: str, value: Any, available_ns: int) -> None:
        if available_ns > self._now_ns:
            raise InformationBoundaryError(f"{kind} {key} published to the context before it is available ({available_ns} > {self._now_ns})")
        if kind == "mark":
            self._marks[key] = value
            h = self._mark_history.get(key)
            if h is None:
                h = self._mark_history[key] = deque(maxlen=self.max_history)
            h.append(value)
        elif kind == "liquidity":
            self._liquidity[key] = value
        elif kind == "rating":
            self._ratings[key] = int(value)
        elif kind == "fixing":
            self._fixings[key] = (available_ns, float(value))
            h = self._fixing_history.get(key)
            if h is None:
                h = self._fixing_history[key] = deque(maxlen=self.max_history)
            h.append((available_ns, float(value)))
        elif kind == "curve":
            self._curves[key] = value
        elif kind == "factor":
            self._factors[key] = float(value)
        elif kind == "index_ratio":
            self._index_ratios[key] = float(value)
        elif kind == "action":
            self._actions.append(value)
        elif kind == "stress":
            self._stress = float(value)
        elif kind == "fx":
            self._fx[key] = float(value)
        else:                                            # any other kind is a generic field (signal, vendor analytic, index weight ...)
            self._materialise(kind)
            self._rows.pop(kind, None)
            self._fields.setdefault(kind, {})[key] = value

    def _publish_row(self, field: str, ids, values, available_ns: int) -> None:
        """Publish a cross-section by reference (an id index and an aligned value array): O(1) per
        field per day; dictionaries are only built when a strategy reads the field."""
        if available_ns > self._now_ns:
            raise InformationBoundaryError(f"field {field} published to the context before it is available ({available_ns} > {self._now_ns})")
        self._rows[field] = (ids, values)
        self._fields.pop(field, None)
        self._field_ts[field] = available_ns

    def _materialise(self, name: str) -> dict[str, Any]:
        if name in self._rows and name not in self._fields:
            import numpy as np

            ids, vals = self._rows[name]
            ok = np.isfinite(vals)
            self._fields[name] = dict(zip(ids[ok], vals[ok].tolist()))
        return self._fields.get(name, {})

    def field_row(self, name: str):
        """(ids, values) of a field published by reference, or None."""
        return self._rows.get(name)

    def _publish_slice(self, field: str, values: dict[str, Any], available_ns: int, replace: bool = True) -> None:
        """Publish a whole cross-section of a generic field at once (one call per field per day)."""
        if available_ns > self._now_ns:
            raise InformationBoundaryError(f"field {field} published to the context before it is available ({available_ns} > {self._now_ns})")
        if field in _BUILTIN_KINDS:
            for k, v in values.items():
                self._publish(field, k, v, available_ns)
            return
        if not replace:
            self._materialise(field)
        self._rows.pop(field, None)
        if replace:
            self._fields[field] = dict(values)
        else:
            self._fields.setdefault(field, {}).update(values)
        self._field_ts[field] = available_ns

    def _set_portfolio(self, positions: dict[str, VisiblePosition], orders: dict[str, VisibleOrder], cash: dict[str, CashState], nav: float | None) -> None:
        self._positions, self._orders, self._cash, self._nav = dict(positions), dict(orders), dict(cash), nav

    def _set_funding(self, shortfall: float) -> None:
        self._funding_shortfall = max(0.0, float(shortfall))

    @property
    def funding_shortfall(self) -> float:
        """Base-currency cash the funding policy could not source (repo exhausted): the strategy is
        expected to raise it by selling, most liquid first unless it overrides the rule."""
        return self._funding_shortfall

    def _bind(self, universe_fn=None, tradable_fn=None, terms_fn=None, restricted: set[str] | None = None) -> None:
        self._universe_fn, self._tradable_fn, self._terms_fn = universe_fn, tradable_fn, terms_fn
        self._restricted = set(restricted or ())

    # ------------------------------------------------------------------ strategy-side reads
    @property
    def now_ns(self) -> int:
        return self._now_ns

    @property
    def today(self) -> date:
        return date_of(self._now_ns)

    def mark(self, instrument: str):
        return self._marks.get(instrument)

    def mark_history(self, instrument: str, n: int | None = None) -> list:
        h = list(self._mark_history.get(instrument, ()))
        return h[-n:] if n else h

    def liquidity(self, instrument: str):
        return self._liquidity.get(instrument)

    def rating_score(self, instrument: str) -> int | None:
        return self._ratings.get(instrument)

    def fixing(self, index: str) -> float | None:
        v = self._fixings.get(index)
        return v[1] if v else None

    def fixing_history(self, index: str) -> list[tuple[int, float]]:
        return list(self._fixing_history.get(index, []))

    def curve(self, name: str = "GOVT"):
        return self._curves.get(name)

    def curves(self) -> dict[str, Any]:
        return dict(self._curves)

    def factor(self, instrument: str) -> float:
        return self._factors.get(instrument, 1.0)

    def index_ratio(self, instrument: str) -> float:
        return self._index_ratios.get(instrument, 1.0)

    def actions(self, instrument: str | None = None) -> list:
        return [a for a in self._actions if instrument is None or getattr(a, "symbol", getattr(a, "instrument", None)) == instrument]

    @property
    def stress(self) -> float:
        return self._stress

    def fx(self, currency: str) -> float | None:
        return 1.0 if currency == self.base_currency else self._fx.get(currency)

    def positions(self) -> dict[str, VisiblePosition]:
        return dict(self._positions)

    def position(self, instrument: str) -> VisiblePosition | None:
        return self._positions.get(instrument)

    def open_orders(self, instrument: str | None = None) -> list[VisibleOrder]:
        return [o for o in self._orders.values() if instrument is None or o.instrument == instrument]

    def cash(self, currency: str | None = None) -> CashState:
        ccy = currency or self.base_currency
        return self._cash.get(ccy, CashState(ccy, 0.0, 0.0, 0.0, 0.0, 0.0))

    @property
    def nav(self) -> float | None:
        return self._nav

    def universe(self) -> list[str]:
        return self._universe_fn(self.today) if self._universe_fn else list(self._marks)

    def terms(self, instrument: str):
        """Contract terms effective today (effective-time query); amendments after today are invisible."""
        return self._terms_fn(instrument, self.today) if self._terms_fn else None

    def can_trade(self, instrument: str) -> tuple[bool, str]:
        if instrument in self._restricted:
            return False, "restricted list"
        if self._tradable_fn:
            return self._tradable_fn(instrument, self.today)
        return True, "ok"

    # ------------------------------------------------------------------ generic fields
    def field(self, name: str, key: str, default: Any = None) -> Any:
        """Latest published value of a generic field for one entity (instrument, index, curve ...)."""
        return self._materialise(name).get(key, default)

    def field_slice(self, name: str) -> dict[str, Any]:
        """The latest published cross-section of a generic field."""
        return dict(self._materialise(name))

    def field_names(self) -> list[str]:
        return sorted(set(self._fields) | set(self._rows))

    def field_time(self, name: str) -> int | None:
        return self._field_ts.get(name)

    def panel(self, names: list[str], keys: list[str] | None = None):
        """Cross-section of several generic fields as a DataFrame (rows: entities, columns: fields)."""
        import pandas as pd

        cols = {n: self._materialise(n) for n in names}
        idx = keys if keys is not None else sorted(set().union(*[set(v) for v in cols.values()])) if cols else []
        return pd.DataFrame({n: [v.get(k) for k in idx] for n, v in cols.items()}, index=idx, dtype=float)

    def support_status(self, product_family: str, feature: str) -> Capability:
        return self.support.status(product_family, feature, run_mode=self.run_mode)

    def note(self, message: str) -> None:
        self.log.append(f"[{self.today}] {message}")

    def __getattr__(self, name: str):
        if name in {"env", "environment", "ledger", "engine", "cache"}:
            raise InformationBoundaryError(f"Strategies cannot access {name!r}; use the point-in-time accessors")
        raise AttributeError(name)


__all__ = ["CashState", "DecisionContext", "InformationBoundaryError", "VisibleOrder", "VisiblePosition"]
_ = field
