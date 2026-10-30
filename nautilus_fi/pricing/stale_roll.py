"""Roll stale marks forward with a factor model instead of carrying them flat.

    dP ~= -D_mod * dY * P - D_spread * dS * P + pull-to-par carry

where ``dY`` is the move in the risk-free par yield at the bond's horizon and ``dS`` the move in the
bond's sector-rating spread. Flat carry understates volatility and correlation; this keeps rolled
marks moving with the market while flagging them as ``STALE_ROLLED``.
"""

from __future__ import annotations

from dataclasses import dataclass

from nautilus_fi.instruments.curves import ZeroCurve


@dataclass
class RollInputs:
    curve_prev: ZeroCurve
    curve_now: ZeroCurve
    sector_spread_prev_bp: float | None
    sector_spread_now_bp: float | None
    days: int = 1


def roll_mark(
    last_clean: float,
    mod_duration: float | None,
    spread_duration: float | None,
    horizon_years: float | None,
    ytm: float | None,
    coupon_rate: float,
    inputs: RollInputs,
    freq: int = 2,
) -> float:
    if last_clean <= 0:
        return last_clean
    horizon = max(horizon_years or 1.0, 1.0 / max(freq, 1))
    dy = inputs.curve_now.par_yield(horizon, freq) - inputs.curve_prev.par_yield(horizon, freq)
    ds = 0.0
    if inputs.sector_spread_prev_bp is not None and inputs.sector_spread_now_bp is not None:
        ds = (inputs.sector_spread_now_bp - inputs.sector_spread_prev_bp) / 1e4
    d_mod = mod_duration if mod_duration is not None else horizon * 0.85
    d_spr = spread_duration if spread_duration is not None else d_mod
    dt = inputs.days / 365.25
    carry = ((ytm if ytm is not None else coupon_rate) * last_clean - coupon_rate * 100.0) * dt
    new_px = last_clean * (1.0 - d_mod * dy - d_spr * ds) + carry
    return max(new_px, 0.01)


__all__ = ["RollInputs", "roll_mark"]
