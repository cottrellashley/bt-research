"""Cash and financing rate models for the ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class CashRateModel:
    def rate(self, d: date) -> float:
        raise NotImplementedError


@dataclass
class FlatCashRate(CashRateModel):
    annual_rate: float = 0.0

    def rate(self, d: date) -> float:
        return self.annual_rate


@dataclass
class IndexCashRate(CashRateModel):
    """Cash earns an index fixing (e.g. SOFR) plus a spread; borrowing pays index plus borrow spread."""

    env: object
    index: str = "SOFR"
    spread_bp: float = -10.0

    def rate(self, d: date) -> float:
        fixing = self.env.fixing(self.index, d)
        return (fixing or 0.0) + self.spread_bp / 1e4


@dataclass
class FinancingTerms:
    """Repo / margin financing for negative cash balances (leverage)."""

    borrow_spread_bp: float = 50.0      # over the cash rate
    haircut: float = 0.02               # informational; margin accounts enforce leverage in Nautilus
    day_count_basis: int = 360


__all__ = ["CashRateModel", "FinancingTerms", "FlatCashRate", "IndexCashRate"]
