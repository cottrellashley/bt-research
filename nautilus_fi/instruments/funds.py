"""Bond funds and ETFs as holdings: shares, distributions, NAV versus market price, fund costs and
access mechanism. The market-traded return differs from the underlying basket."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class FundInstrument:
    symbol: str
    kind: str                                   # ETF | MUTUAL_FUND
    expense_ratio_bp: float = 20.0
    access: str = "SECONDARY"                   # SECONDARY | CREATION_UNIT | PRIMARY
    creation_unit_shares: int = 25_000
    nav: dict[date, float] = field(default_factory=dict)
    market_price: dict[date, float] = field(default_factory=dict)
    distributions: dict[date, float] = field(default_factory=dict)   # per share, ex-date keyed

    def premium_discount(self, d: date) -> float | None:
        if d in self.nav and d in self.market_price and self.nav[d]:
            return self.market_price[d] / self.nav[d] - 1.0
        return None

    def total_return(self, start: date, end: date, basis: str = "MARKET") -> float:
        series = self.market_price if (basis == "MARKET" and self.kind == "ETF") else self.nav
        dates = sorted(k for k in series if start <= k <= end)
        if len(dates) < 2:
            return 0.0
        p0, p1 = series[dates[0]], series[dates[-1]]
        dist = sum(v for k, v in self.distributions.items() if dates[0] < k <= dates[-1])
        return (p1 + dist) / p0 - 1.0

    def tradable_quantity(self, shares: float) -> float:
        if self.access == "CREATION_UNIT":
            return round(shares / self.creation_unit_shares) * self.creation_unit_shares
        return float(round(shares))


__all__ = ["FundInstrument"]
