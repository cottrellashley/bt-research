"""
nautilus_fi: a generic fixed income paper-portfolio backtester built on Nautilus Trader v2.

The package separates the four engines a credible fixed income backtest needs:

* ``nautilus_fi.data``       point-in-time data layer (knowledge-time stamped inputs)
* ``nautilus_fi.pricing``    pricing / marking waterfall (what a position is worth today)
* ``nautilus_fi.execution``  execution simulator (what you could actually have traded, when, how much)
* ``nautilus_fi.accounting`` portfolio accounting ledger (every cashflow the real portfolio would see)

``nautilus_fi.backtest`` wires them into a Nautilus ``BacktestEngine``.
"""

from nautilus_fi.version import __version__

__all__ = ["__version__"]
