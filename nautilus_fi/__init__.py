"""
nautilus_fi: a generic fixed income research and backtesting platform built on Nautilus Trader v2.

The research kernel (``nautilus_fi.kernel``, re-exported here) is the ground everything works up
from: strategies are Python objects (data sources, expressions, universes, indices, construction,
optimiser terms and constraints, execution, costs, financing, conventions) and run at Paper
fidelity (vectorised panels) or Implement fidelity (the event engine)::

    import nautilus_fi as nfi
    s = nfi.Strategy("value", universe=nfi.Index("LUACTRUU"), signal=nfi.Field("oas_bp").zscore(by="sector"),
                     portfolio=nfi.TopN(100), benchmark=nfi.Index("LUACTRUU"))
    print(nfi.compare(s, s.with_(portfolio=nfi.TopN(50), name="concentrated"), source=data).explain())

Underneath, the implementation simulator separates the four engines a credible fixed income
backtest needs:

* ``nautilus_fi.data``       point-in-time data layer (knowledge-time stamped inputs)
* ``nautilus_fi.pricing``    pricing / marking waterfall (what a position is worth today)
* ``nautilus_fi.execution``  execution simulator (what you could actually have traded, when, how much)
* ``nautilus_fi.accounting`` portfolio accounting ledger (every cashflow the real portfolio would see)

``nautilus_fi.backtest`` wires them into a Nautilus ``BacktestEngine``.
"""

from nautilus_fi.version import __version__


def __getattr__(name: str):
    """Kernel names on the package (``nfi.Strategy``, ``nfi.run`` ...), imported on first use."""
    import importlib

    kernel = importlib.import_module("nautilus_fi.kernel")
    try:
        return getattr(kernel, name)
    except AttributeError:
        raise AttributeError(f"module 'nautilus_fi' has no attribute {name!r}") from None


def __dir__():
    import importlib

    return sorted(set(globals()) | set(importlib.import_module("nautilus_fi.kernel").__all__))


__all__ = ["__version__"]
