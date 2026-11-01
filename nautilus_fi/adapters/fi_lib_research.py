"""fi-lib-research engine: timing studies with constant exposure (gap analysis §5.4).

Conventions: the ``fi_lib_research`` profile (trades T+1, first earns T+2). The engine's own
choices, as kernel settings:

* ``engine``: constant exposure (the traded weights are held, ``Strategy(hold="constant")``),
  additive P&L and a fixed cost per change of position (``PerChange``);
* ``input_field``: its input kinds (level, return, log return, cumulative return) as declared
  field forms, converted once on read;
* ``align``: its alignment modes for two legs (inner, strict, outer);
* ``validation_gate``: ``ValidationGate``'s P&L-gated selection.
"""

from __future__ import annotations

from nautilus_fi.adapters.parity import ParityCase, Tolerance
from nautilus_fi.kernel.data import FieldInfo

PROFILE = "fi_lib_research"
REQUIREMENTS = ("constant exposure", "additive P&L", "fixed cost per change", "input kinds", "alignment modes", "ValidationGate")
INPUT_KINDS = {"level": "price", "price": "price", "return": "simple", "simple": "simple", "log_return": "log", "log": "log", "cumulative": "cumulative", "cumulative_return": "cumulative"}


def engine(strategy, cost_per_change: float | None = None):
    """``strategy`` as fi-lib-research runs it: constant exposure, additive P&L under the
    profile's timing, and (``cost_per_change``) a fixed amount per change of position."""
    import dataclasses

    from nautilus_fi.kernel.mechanics import PerChange

    s = strategy.under(PROFILE)
    s = s.with_(hold="constant", metrics=dataclasses.replace(s.metrics, pnl="ADDITIVE"))
    if cost_per_change is not None:
        s = s.with_(execution=dataclasses.replace(s.execution, cost=PerChange(cost_per_change)))
    return s


def input_field(name: str, kind: str, description: str = "", **kw) -> FieldInfo:
    """A field declared with fi-lib-research's input kind (level, return, log_return,
    cumulative): the kernel reads it as a simple return wherever a return is realised."""
    key = kind.lower()
    if key not in INPUT_KINDS:
        raise ValueError(f"unknown input kind {kind!r}; known: {sorted(INPUT_KINDS)}")
    return FieldInfo(name, description or f"{kind} input", form=INPUT_KINDS[key], **kw)


def align(total, rate, mode: str = "inner"):
    """Two legs aligned the engine's way: ``inner`` (common dates, the dropped ones reported),
    ``strict`` (any mismatch raises) or ``outer`` (every date, zero where a leg has none)."""
    from nautilus_fi.kernel.returns import align_legs

    return align_legs(total, rate, mode)


def validation_gate(results, primary: str = "information_ratio", tolerance: float = 0.1, pnl: str = "annual_return", floor: float = 0.0):
    """``ValidationGate``: candidates within ``tolerance`` of the best ``primary`` score, then the
    best ``pnl`` above ``floor`` (raises when none clears it)."""
    from nautilus_fi.kernel.validation import select_gated

    return select_gated(results, primary=primary, tolerance=tolerance, pnl=pnl, floor=floor)


PARITY = [ParityCase("engine cases", "fi_lib_research", "tests/test_backtesting_engine.py", (Tolerance("returns", bp_per_year=1.0, min_corr=0.9999),), notes="each test case's expected P&L")]

__all__ = ["INPUT_KINDS", "PARITY", "PROFILE", "REQUIREMENTS", "align", "engine", "input_field", "validation_gate"]
