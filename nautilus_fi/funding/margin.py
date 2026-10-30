"""Derivative margin and collateral: initial and variation margin, thresholds, minimum transfer
amounts, independent amounts, eligible collateral with haircuts, netting sets and dispute lags.
Collateralised-to-market (CTM) keeps exposure on the balance sheet with collateral transfers;
settled-to-market (STM) settles the exposure daily (futures-style)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

import pandas as pd


@dataclass
class CSATerms:
    counterparty: str
    netting_set: str
    threshold: float = 0.0
    minimum_transfer: float = 250_000.0
    independent_amount: float = 0.0
    eligible_collateral: tuple[str, ...] = ("CASH",)
    haircuts: dict[str, float] = field(default_factory=lambda: {"CASH": 0.0, "UST": 0.02, "CORP_IG": 0.08})
    dispute_lag_days: int = 1
    remuneration_bp: float = 0.0           # over the cash index on posted cash collateral
    mode: str = "CTM"                      # CTM | STM
    rounding: float = 10_000.0
    currency: str = "USD"


class IMModel(Protocol):
    def initial_margin(self, netting_set: str, positions: list[dict]) -> float: ...


@dataclass
class PercentNotionalIM:
    pct: float = 0.02

    def initial_margin(self, netting_set: str, positions: list[dict]) -> float:
        return sum(abs(p.get("notional", 0.0)) for p in positions) * self.pct


@dataclass
class ShockIM:
    """Max loss over a declared shock set using each position's sensitivities (a SIMM-like proxy)."""

    rate_shocks_bp: tuple[float, ...] = (-50.0, 50.0)
    spread_shocks_bp: tuple[float, ...] = (-100.0, 100.0)
    fx_shocks: tuple[float, ...] = (-0.08, 0.08)

    def initial_margin(self, netting_set: str, positions: list[dict]) -> float:
        worst = 0.0
        for r in self.rate_shocks_bp:
            for s in self.spread_shocks_bp:
                for f in self.fx_shocks:
                    pnl = sum(-(p.get("dv01", 0.0) * r) - (p.get("cs01", 0.0) * s) + p.get("fx_delta", 0.0) * f for p in positions)
                    worst = min(worst, pnl)
        return -worst


@dataclass
class MarginCall:
    netting_set: str
    counterparty: str
    kind: str                     # VM | IM
    amount: float                 # positive: we post; negative: we receive
    call_date: date
    due_date: date
    currency: str


class MarginEngine:
    def __init__(self, csas: dict[str, CSATerms], im_model: IMModel | None = None, fixing: Callable[[str, date], float | None] | None = None) -> None:
        self.csas = csas
        self.im_model = im_model or PercentNotionalIM()
        self.posted_vm: dict[str, float] = {k: 0.0 for k in csas}     # net collateral we have posted (+) / received (-)
        self.posted_im: dict[str, float] = {k: 0.0 for k in csas}
        self.calls: list[MarginCall] = []
        self._fixing = fixing

    def _round(self, amount: float, csa: CSATerms) -> float:
        if csa.rounding <= 0:
            return amount
        import math

        return math.copysign(math.ceil(abs(amount) / csa.rounding) * csa.rounding, amount)

    def collateral_value(self, csa: CSATerms, collateral_type: str, market_value: float) -> float:
        if collateral_type not in csa.eligible_collateral:
            raise ValueError(f"{collateral_type} is not eligible collateral under {csa.counterparty} CSA")
        return market_value * (1.0 - csa.haircuts.get(collateral_type, 0.0))

    def process(self, d: date, exposures: dict[str, float], positions_by_set: dict[str, list[dict]] | None = None) -> list[MarginCall]:
        """``exposures[netting_set]`` = our net MTM (positive = counterparty owes us). Returns new calls."""
        new: list[MarginCall] = []
        for ns, csa in self.csas.items():
            mtm = exposures.get(ns, 0.0)
            if csa.mode == "STM":
                # settled to market: the full change settles as cash (no collateral balance remains)
                required = 0.0
            else:
                # we must post when MTM is negative beyond the threshold; we receive when positive
                exposure_to_cover = -mtm
                if exposure_to_cover > csa.threshold:
                    required = exposure_to_cover - csa.threshold
                elif exposure_to_cover < -csa.threshold:
                    required = exposure_to_cover + csa.threshold
                else:
                    required = 0.0
            delta = required - self.posted_vm[ns]
            if abs(delta) >= csa.minimum_transfer and csa.mode == "CTM":
                amt = self._round(delta, csa)
                self.posted_vm[ns] += amt
                new.append(MarginCall(ns, csa.counterparty, "VM", amt, d, d + timedelta(days=csa.dispute_lag_days), csa.currency))
            if positions_by_set is not None:
                im_req = self.im_model.initial_margin(ns, positions_by_set.get(ns, [])) + csa.independent_amount
                im_delta = im_req - self.posted_im[ns]
                if abs(im_delta) >= csa.minimum_transfer:
                    self.posted_im[ns] += im_delta
                    new.append(MarginCall(ns, csa.counterparty, "IM", im_delta, d, d + timedelta(days=csa.dispute_lag_days), csa.currency))
        self.calls.extend(new)
        return new

    def remuneration(self, ns: str, d: date, days: int = 1) -> float:
        """Interest earned (paid) on cash collateral posted (received)."""
        csa = self.csas[ns]
        base = self._fixing(f"{csa.currency}_OIS", d) if self._fixing else 0.0
        rate = (base or 0.0) + csa.remuneration_bp / 1e4
        return (self.posted_vm[ns] + self.posted_im[ns]) * rate * days / 360.0

    def total_posted(self, currency: str | None = None) -> float:
        return sum(self.posted_vm[k] + self.posted_im[k] for k, csa in self.csas.items() if currency is None or csa.currency == currency)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.calls])


__all__ = ["CSATerms", "IMModel", "MarginCall", "MarginEngine", "PercentNotionalIM", "ShockIM"]
