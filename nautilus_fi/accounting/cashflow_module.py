"""Nautilus ``SimulationModule`` that drives the ledger once per engine timestamp and returns the
account adjustments (coupons, paydowns, accrued settlement, cash interest, financing)."""

from __future__ import annotations

from nautilus_trader.backtest import SimulationModule

from nautilus_fi.accounting.ledger import FixedIncomeLedger


class FixedIncomeCashflowModule(SimulationModule):
    def __init__(self, ledger: FixedIncomeLedger) -> None:
        super().__init__()
        self.ledger = ledger
        self.failures: list[str] = []

    def pre_process(self, data) -> None:  # market data hook (unused)
        return None

    def process(self, ts_now: int, context):
        return self.ledger.on_timestamp(ts_now, list(context.positions))

    def acknowledge(self, outcomes) -> None:
        for o in outcomes:
            text = str(o)
            if "Failed" in text or "Err" in text:
                self.failures.append(text)

    def log_diagnostics(self) -> None:
        return None

    def reset(self) -> None:
        return None


class MarginVenueModule(SimulationModule):
    """Applies the ledger's queued margin-account adjustments (collateral transfers, borrow fees)."""

    def __init__(self, link) -> None:
        super().__init__()
        self.link = link

    def pre_process(self, data) -> None:
        return None

    def process(self, ts_now: int, context):
        return self.link.take_margin_adjustments()

    def acknowledge(self, outcomes) -> None:
        return None

    def log_diagnostics(self) -> None:
        return None

    def reset(self) -> None:
        return None


__all__ = ["FixedIncomeCashflowModule", "MarginVenueModule"]
