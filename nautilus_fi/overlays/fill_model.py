"""Fill model for the margin venue: an order fills only at an execution window after the decision
that created it, at the window quote (prior close +/- the contract's half spread), in full."""

from __future__ import annotations

from nautilus_trader.execution import FillModel
from nautilus_trader.model import BookOrder, BookType, OrderBook, OrderSide, Quantity


class OverlayFillModel(FillModel):
    def __init__(self, schedule) -> None:
        super().__init__()
        self.schedule = schedule
        self.clock = None
        self.fills: list[dict] = []
        self._seq = 0

    def is_limit_filled(self) -> bool:
        return True

    def is_slipped(self) -> bool:
        return False

    def fill_limit_inside_spread(self) -> bool:
        return False

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        book = OrderBook(instrument.id, BookType.L2_MBP)
        if self.clock is None or best_bid is None or best_ask is None:
            return book
        now = self.clock.timestamp_ns()
        if order.ts_init >= now or not self.schedule.is_window(now):
            return book
        self._seq += 1
        side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        px = best_ask if order.side == OrderSide.BUY else best_bid
        book.add(BookOrder(side, px, Quantity(order.leaves_qty.as_double() if hasattr(order.leaves_qty, "as_double") else float(order.leaves_qty), 0), self._seq), 0, self._seq, now)
        self.fills.append({"ts": now, "order_id": str(order.client_order_id), "symbol": instrument.id.symbol.value, "side": "BUY" if order.side == OrderSide.BUY else "SELL", "price": float(px.as_double() if hasattr(px, "as_double") else px)})
        return book


__all__ = ["OverlayFillModel"]
