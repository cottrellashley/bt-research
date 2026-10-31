"""Mechanics as objects: how a decision becomes a trade, what trading costs, what cash earns and
leverage pays, what a missing return means, how excess returns compound, what turnover is and how
statistics annualise.

Each is a small immutable object that changes how the engine behaves for one strategy:

* ``NextClose(lag=1, cost=HalfSpread() * 1.5 + SqrtImpact(10))``: decide on the close, trade
  ``lag`` trading days later at the close, first return the day after (``SameClose()`` trades on
  the close the signal sees). ``participation=Participation(0.1)`` caps each day's trade at 10% of
  ADV and carries the rest.
* Cost algebra: ``HalfSpread``, ``FixedCost``, ``SqrtImpact``, ``FieldCost`` (any expression in bp),
  ``NoCost``, combined with ``+`` and scaled by a number or an expression (``HalfSpread() *
  (1 + (TimeSeries("vix") > 30))`` doubles costs in stress).
* ``Cash("SOFR", borrow_spread_bp=25, short_fee_bp=30)`` / ``NoCash()``.
* ``ZeroFill()``, ``DropDates(0.5)``, ``Refill("tr_alt")``, ``Fail()``.
* ``VendorExcess("xr")``, ``Compounded("tsy_r")``, ``Geometric``, ``DailySum``.
* ``HalfSum()``, ``TwoWay()``, ``OneWayBuys()``; ``Calendar365()``, ``Calendar36525()``,
  ``Trading252()``; ``Metrics(...)`` bundles them for statistics.

The ``Conventions`` profiles of the legacy frameworks are presets of these objects
(``profile("MRI_v1")``), so a strategy can be run under another framework's conventions with
``strategy.under("MRI_v1")``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nautilus_fi.domain.conventions import (
    PROFILES,
    Annualisation,
    CashReturn,
    Conventions,
    ExcessReturnMethod,
    MissingReturnPolicy,
    Timing,
    TurnoverDefinition,
    WeightDrift,
    years_spanned,
)
from nautilus_fi.kernel.data import MissingDataError
from nautilus_fi.kernel.expr import EvalContext, Expr, _fmt


# ====================================================================================== costs
class CostModel:
    """Cost of trading, as a fraction of traded value, per instrument on a date. ``unit(ctx, t,
    trade, nav)`` may depend on the trade size (weights); ``linear`` is the size-free part the
    optimiser sees (nonlinear terms at a reference size)."""

    def unit(self, ctx: EvalContext, t: int, trade: np.ndarray | None, nav: float) -> np.ndarray:
        raise NotImplementedError

    def linear(self, ctx: EvalContext, t: int, nav: float, ref: float = 0.002) -> np.ndarray:
        return self.unit(ctx, t, np.full(ctx.N, ref), nav)

    def total(self, ctx: EvalContext, t: int, trade: np.ndarray, nav: float) -> float:
        a = np.abs(trade)
        nz = a > 0
        if not nz.any():
            return 0.0
        u = self.unit(ctx, t, a, nav)
        return float(np.nansum(a[nz] * u[nz]))

    def names(self) -> set[str]:
        return set()

    def optional_names(self) -> set[str]:
        """Fields used when present (alternatives: ADV or a size field)."""
        return set()

    def __add__(self, other: CostModel) -> CostModel:
        return SumCost((self, other))

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __mul__(self, k) -> CostModel:
        return ScaledCost(self, k)

    __rmul__ = __mul__


@dataclass(frozen=True)
class HalfSpread(CostModel):
    """Half the bid/ask spread (a bp field) per unit traded; ``fallback_bp`` fills gaps (None:
    a missing spread on a traded name is an error)."""

    field: str = "half_spread_bp"
    fallback_bp: float | None = None

    def names(self):
        return {self.field}

    def unit(self, ctx, t, trade, nav):
        if self.field not in ctx.panel:
            raise MissingDataError(f"cost model HalfSpread needs field {self.field!r}, which the source does not have; use execution=NextClose(cost=FixedCost(bp)) or NoCost(), or HalfSpread(field=...)")
        v = np.asarray(ctx.field(self.field)[t], dtype=np.float64)
        if np.isnan(v).any():
            if self.fallback_bp is None:
                med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
                v = np.where(np.isnan(v), med, v)
            else:
                v = np.where(np.isnan(v), self.fallback_bp, v)
        return v / 1e4

    def __repr__(self) -> str:
        args = [] if self.field == "half_spread_bp" else [f"field={self.field!r}"]
        if self.fallback_bp is not None:
            args.append(f"fallback_bp={self.fallback_bp:g}")
        return f"HalfSpread({', '.join(args)})"


@dataclass(frozen=True)
class FixedCost(CostModel):
    bp: float = 5.0

    def unit(self, ctx, t, trade, nav):
        return np.full(ctx.N, self.bp / 1e4)

    def __repr__(self) -> str:
        return f"FixedCost({self.bp:g})"


@dataclass(frozen=True)
class NoCost(CostModel):
    def unit(self, ctx, t, trade, nav):
        return np.zeros(ctx.N)

    def __repr__(self) -> str:
        return "NoCost()"


@dataclass(frozen=True, eq=False)
class FieldCost(CostModel):
    """Any expression in bp per unit traded (e.g. a vendor liquidity score mapped to a cost)."""

    expr: Expr = None
    fallback_bp: float = 0.0

    def names(self):
        return self.expr.names()

    def unit(self, ctx, t, trade, nav):
        v = np.asarray(self.expr.evaluate(ctx.full())[t], dtype=np.float64)
        return np.where(np.isfinite(v), v, self.fallback_bp) / 1e4

    def __repr__(self) -> str:
        return f"FieldCost({self.expr!r})"


@dataclass(frozen=True)
class SqrtImpact(CostModel):
    """Market impact ``coef_bp x sqrt(traded value / ADV)`` per unit traded. ADV is a field
    (``adv``, currency) or ``adv_fraction`` of a size field (``amt_out``) when the source has no
    volume data; ``cap_bp`` bounds the impact of a single trade."""

    coef_bp: float = 10.0
    adv: str = "adv"
    adv_fraction: float | None = 0.002
    size_field: str = "amt_out"
    cap_bp: float | None = 500.0

    def optional_names(self):
        return {self.adv, self.size_field}

    def _adv(self, ctx, t) -> np.ndarray:
        if self.adv in ctx.panel:
            return np.asarray(ctx.field(self.adv)[t], dtype=np.float64)
        if self.adv_fraction is not None and self.size_field in ctx.panel:
            return np.asarray(ctx.field(self.size_field)[t], dtype=np.float64) * self.adv_fraction
        raise MissingDataError(f"SqrtImpact needs an ADV field {self.adv!r} or a size field {self.size_field!r} (with adv_fraction)")

    def unit(self, ctx, t, trade, nav):
        adv = self._adv(ctx, t)
        size = np.abs(trade if trade is not None else np.zeros(ctx.N)) * nav
        with np.errstate(divide="ignore", invalid="ignore"):
            bp = self.coef_bp * np.sqrt(size / adv)
        bp = np.where(np.isfinite(bp), bp, 0.0 if self.cap_bp is None else self.cap_bp)
        if self.cap_bp is not None:
            bp = np.minimum(bp, self.cap_bp)
        return bp / 1e4

    def __repr__(self) -> str:
        return f"SqrtImpact({self.coef_bp:g}{'' if self.adv == 'adv' else ', adv=' + repr(self.adv)})"


@dataclass(frozen=True, eq=False)
class SumCost(CostModel):
    parts: tuple = ()

    def names(self):
        return set().union(*[p.names() for p in self.parts]) if self.parts else set()

    def optional_names(self):
        return set().union(*[p.optional_names() for p in self.parts]) if self.parts else set()

    def unit(self, ctx, t, trade, nav):
        return sum((p.unit(ctx, t, trade, nav) for p in self.parts), np.zeros(ctx.N))

    def linear(self, ctx, t, nav, ref=0.002):
        return sum((p.linear(ctx, t, nav, ref) for p in self.parts), np.zeros(ctx.N))

    def __add__(self, other):
        return SumCost((*self.parts, other))

    def __repr__(self) -> str:
        return " + ".join(repr(p) for p in self.parts)


@dataclass(frozen=True, eq=False)
class ScaledCost(CostModel):
    model: CostModel = None
    factor: object = 1.0                                  # number or expression (per instrument and date)

    def names(self):
        return self.model.names() | (self.factor.names() if isinstance(self.factor, Expr) else set())

    def optional_names(self):
        return self.model.optional_names()

    def _k(self, ctx, t):
        if isinstance(self.factor, Expr):
            v = np.asarray(self.factor.evaluate(ctx.full())[t], dtype=np.float64)
            return np.where(np.isfinite(v), v, 1.0)
        return float(self.factor)

    def unit(self, ctx, t, trade, nav):
        return self.model.unit(ctx, t, trade, nav) * self._k(ctx, t)

    def linear(self, ctx, t, nav, ref=0.002):
        return self.model.linear(ctx, t, nav, ref) * self._k(ctx, t)

    def __repr__(self) -> str:
        return f"{self.model!r} * {_fmt(self.factor)}"


# ====================================================================================== execution
@dataclass(frozen=True)
class Participation:
    """Trade at most ``max_fraction`` of ADV per instrument per day; the rest of the target
    carries to the next day (a new decision replaces it)."""

    max_fraction: float = 0.1
    adv: str = "adv"
    adv_fraction: float | None = 0.002
    size_field: str = "amt_out"

    def optional_names(self):
        return {self.adv, self.size_field}

    def capacity(self, ctx: EvalContext, t: int, nav: float) -> np.ndarray:
        if self.adv in ctx.panel:
            adv = np.asarray(ctx.field(self.adv)[t], dtype=np.float64)
        elif self.adv_fraction is not None and self.size_field in ctx.panel:
            adv = np.asarray(ctx.field(self.size_field)[t], dtype=np.float64) * self.adv_fraction
        else:
            raise MissingDataError(f"Participation needs an ADV field {self.adv!r} or a size field {self.size_field!r}")
        return np.where(np.isfinite(adv), adv * self.max_fraction / max(nav, 1.0), np.inf)

    def __repr__(self) -> str:
        return f"Participation({self.max_fraction:g})"


@dataclass(frozen=True)
class Execution:
    """Decide on the close of T; trade on the close of T + ``lag`` (trading days); the new weights
    earn from the period after. ``lag=1`` is the canonical convention (first return T+2)."""

    lag: int = 1
    cost: CostModel = field(default_factory=HalfSpread)
    participation: Participation | None = None

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError("execution lag must be >= 0 trading days")

    @property
    def return_lag_days(self) -> int:
        return self.lag + 1

    def names(self) -> set[str]:
        return self.cost.names()

    def optional_names(self) -> set[str]:
        return self.cost.optional_names() | (self.participation.optional_names() if self.participation is not None else set())

    def describe(self) -> str:
        return f"decide on the close, trade {self.lag} trading day(s) later on the close, first return T+{self.lag + 1}; costs {self.cost!r}" + (f"; {self.participation!r}" if self.participation else "")

    def __repr__(self) -> str:
        cost = "" if isinstance(self.cost, HalfSpread) and self.cost == HalfSpread() else f", cost={self.cost!r}"
        part = "" if self.participation is None else f", participation={self.participation!r}"
        if self.lag == 0:
            return f"SameClose({cost.lstrip(', ')}{part})"
        return f"NextClose(lag={self.lag}{cost}{part})"


def NextClose(lag: int = 1, cost: CostModel | None = None, participation: Participation | None = None) -> Execution:
    return Execution(int(lag), cost if cost is not None else HalfSpread(), participation)


def SameClose(cost: CostModel | None = None, participation: Participation | None = None) -> Execution:
    return Execution(0, cost if cost is not None else HalfSpread(), participation)


# ====================================================================================== financing
@dataclass(frozen=True)
class Financing:
    """What uninvested cash earns (a date series of per-period returns; None: zero), what negative
    cash pays above it, and the fee on short positions."""

    series: str | None = "cash"
    borrow_spread_bp: float = 0.0
    short_fee_bp: float = 30.0

    def names(self) -> set[str]:
        return {self.series} if self.series else set()

    def __repr__(self) -> str:
        if self.series is None:
            return f"NoCash(borrow_spread_bp={self.borrow_spread_bp:g}, short_fee_bp={self.short_fee_bp:g})"
        extra = "".join([f", borrow_spread_bp={self.borrow_spread_bp:g}" if self.borrow_spread_bp else "", f", short_fee_bp={self.short_fee_bp:g}" if self.short_fee_bp != 30.0 else ""])
        return f"Cash({self.series!r}{extra})"


def Cash(series: str = "cash", borrow_spread_bp: float = 0.0, short_fee_bp: float = 30.0) -> Financing:
    return Financing(series, float(borrow_spread_bp), float(short_fee_bp))


def NoCash(borrow_spread_bp: float = 0.0, short_fee_bp: float = 30.0) -> Financing:
    return Financing(None, float(borrow_spread_bp), float(short_fee_bp))


# ====================================================================================== missing returns
@dataclass(frozen=True)
class MissingReturns:
    policy: MissingReturnPolicy = MissingReturnPolicy.FAIL
    threshold: float = 0.5
    refill: str | None = None

    def names(self) -> set[str]:
        return {self.refill} if self.refill else set()

    def __repr__(self) -> str:
        p = self.policy
        if p == MissingReturnPolicy.ZERO:
            return "ZeroFill()"
        if p == MissingReturnPolicy.DROP_DATE:
            return f"DropDates({self.threshold:g})"
        if p == MissingReturnPolicy.REFILL:
            return f"Refill({self.refill!r})"
        return "Fail()"


def ZeroFill() -> MissingReturns:
    return MissingReturns(MissingReturnPolicy.ZERO)


def DropDates(threshold: float = 0.5) -> MissingReturns:
    return MissingReturns(MissingReturnPolicy.DROP_DATE, float(threshold))


def Refill(field: str, threshold: float = 0.5) -> MissingReturns:
    return MissingReturns(MissingReturnPolicy.REFILL, float(threshold), field)


def Fail() -> MissingReturns:
    return MissingReturns(MissingReturnPolicy.FAIL)


# ====================================================================================== excess returns
@dataclass(frozen=True)
class ExcessReturn:
    method: ExcessReturnMethod = ExcessReturnMethod.COMPOUNDED
    vendor: str | None = "xr"
    treasury: str | None = "tsy_r"
    vendor_first: bool = True

    def names(self) -> set[str]:
        return {n for n in (self.vendor, self.treasury) if n}

    def __repr__(self) -> str:
        m = self.method
        if m == ExcessReturnMethod.VENDOR:
            return f"VendorExcess({self.vendor!r})"
        name = {ExcessReturnMethod.COMPOUNDED: "Compounded", ExcessReturnMethod.GEOMETRIC: "Geometric", ExcessReturnMethod.DAILY_SUM: "DailySum"}[m]
        return f"{name}({self.treasury!r}{'' if self.vendor_first or m == ExcessReturnMethod.DAILY_SUM else ', vendor_first=False'})"


def VendorExcess(field: str = "xr") -> ExcessReturn:
    return ExcessReturn(ExcessReturnMethod.VENDOR, field, None, True)


def Compounded(treasury: str = "tsy_r", vendor: str | None = "xr", vendor_first: bool = True) -> ExcessReturn:
    return ExcessReturn(ExcessReturnMethod.COMPOUNDED, vendor, treasury, vendor_first)


def Geometric(treasury: str = "tsy_r", vendor: str | None = "xr", vendor_first: bool = True) -> ExcessReturn:
    return ExcessReturn(ExcessReturnMethod.GEOMETRIC, vendor, treasury, vendor_first)


def DailySum(treasury: str = "tsy_r", vendor: str | None = "xr") -> ExcessReturn:
    return ExcessReturn(ExcessReturnMethod.DAILY_SUM, vendor, treasury, False)


# ====================================================================================== turnover & annualisation
def HalfSum() -> TurnoverDefinition:
    return TurnoverDefinition.HALF_SUM


def TwoWay() -> TurnoverDefinition:
    return TurnoverDefinition.TWO_WAY


def OneWayBuys() -> TurnoverDefinition:
    return TurnoverDefinition.ONE_WAY_BUYS


def Calendar365() -> Annualisation:
    return Annualisation.CALENDAR_365


def Calendar36525() -> Annualisation:
    return Annualisation.CALENDAR_365_25


def Trading252() -> Annualisation:
    return Annualisation.TRADING_252


def turnover_vectors(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    d = after - before
    a = float(np.abs(d).sum())
    return {"ONE_WAY_BUYS": float(d[d > 0].sum()), "HALF_SUM": 0.5 * a, "TWO_WAY": a}


# ====================================================================================== metrics
STAT_KEYS = ("total_return", "annual_return", "annual_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate", "gross_annual_return", "cost_drag", "turnover", "turnover_one_way_buys", "turnover_half_sum", "turnover_two_way", "n_rebalances", "avg_names", "avg_cash_weight", "max_gross_exposure", "benchmark_annual_return", "active_annual_return", "tracking_error", "information_ratio", "excess_annual_return", "years", "periods_per_year")


@dataclass(frozen=True)
class Metrics:
    """How statistics are computed and stated. One function, ``stats``, serves every engine, so
    Paper and Implement results report the same keys on the same basis."""

    annualisation: Annualisation = Annualisation.CALENDAR_365
    turnover: TurnoverDefinition = TurnoverDefinition.HALF_SUM
    excess: ExcessReturn = field(default_factory=ExcessReturn)
    risk_free: str | None = None                   # series for Sharpe on excess-of-cash returns (None: raw returns)

    def names(self) -> set[str]:
        return self.excess.names() | ({self.risk_free} if self.risk_free else set())

    def __repr__(self) -> str:
        parts = []
        if self.annualisation != Annualisation.CALENDAR_365:
            parts.append(f"annualisation={self.annualisation.value!r}")
        if self.turnover != TurnoverDefinition.HALF_SUM:
            parts.append(f"turnover={self.turnover.value!r}")
        if self.excess != ExcessReturn():
            parts.append(f"excess={self.excess!r}")
        if self.risk_free:
            parts.append(f"risk_free={self.risk_free!r}")
        return f"Metrics({', '.join(parts)})"

    def years(self, index: pd.Index) -> float:
        return years_spanned(index, self.annualisation)

    def periods_per_year(self, index: pd.Index) -> float:
        from nautilus_fi.domain.conventions import periods_per_year

        return periods_per_year(index, self.annualisation)

    def stats(self, returns: pd.Series, gross: pd.Series | None = None, costs: pd.Series | None = None, benchmark: pd.Series | None = None, risk_free: pd.Series | None = None, trades: pd.DataFrame | None = None, daily: pd.DataFrame | None = None, excess: pd.Series | None = None) -> dict:
        r = pd.Series(returns, dtype=float).dropna()
        out: dict = {"annualisation": self.annualisation.value, "turnover_definition": self.turnover.value}
        if r.empty:
            return out
        years = self.years(r.index)
        ppy = self.periods_per_year(r.index)
        nav = (1.0 + r).cumprod()
        growth = float(nav.iloc[-1])
        out["total_return"] = growth - 1.0
        out["annual_return"] = growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")
        sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
        out["annual_vol"] = sd * math.sqrt(ppy) if np.isfinite(sd) else float("nan")
        ex = r - pd.Series(risk_free).reindex(r.index).fillna(0.0) if risk_free is not None else r
        sdx = float(ex.std(ddof=1)) if len(ex) > 1 else 0.0
        out["sharpe"] = float(ex.mean() / sdx * math.sqrt(ppy)) if sdx else float("nan")
        down = ex[ex < 0]
        dd = float(np.sqrt((down ** 2).sum() / max(len(ex), 1)))
        out["sortino"] = float(ex.mean() / dd * math.sqrt(ppy)) if dd else float("nan")
        mdd = float((nav / nav.cummax() - 1.0).min())
        out["max_drawdown"] = mdd
        out["calmar"] = float(out["annual_return"] / abs(mdd)) if mdd < 0 and np.isfinite(out["annual_return"]) else float("nan")
        out["hit_rate"] = float((r > 0).mean())
        out["years"] = years
        out["periods_per_year"] = ppy
        if gross is not None:
            g = pd.Series(gross, dtype=float).reindex(r.index).dropna()
            gg = float((1.0 + g).prod())
            out["gross_annual_return"] = gg ** (1.0 / years) - 1.0 if years > 0 and gg > 0 else float("nan")
        if costs is not None:
            out["cost_drag"] = float(pd.Series(costs).reindex(r.index).fillna(0.0).sum() / years) if years else float("nan")
        if trades is not None and len(trades):
            for k in ("ONE_WAY_BUYS", "HALF_SUM", "TWO_WAY"):
                if k in trades:
                    out[f"turnover_{k.lower()}"] = float(trades[k].sum() / years) if years else float("nan")
            out["turnover"] = out.get(f"turnover_{self.turnover.value.lower()}", float("nan"))
            out["n_rebalances"] = len(trades)
        elif trades is not None:
            out.update({"turnover": 0.0, "turnover_one_way_buys": 0.0, "turnover_half_sum": 0.0, "turnover_two_way": 0.0, "n_rebalances": 0})
        if daily is not None and len(daily):
            if "n_names" in daily:
                out["avg_names"] = float(daily["n_names"].reindex(r.index).mean())
            if "cash_weight" in daily:
                out["avg_cash_weight"] = float(daily["cash_weight"].reindex(r.index).mean())
            if "gross_exposure" in daily:
                out["max_gross_exposure"] = float(daily["gross_exposure"].max())
        if benchmark is not None:
            b = pd.Series(benchmark, dtype=float).reindex(r.index)
            if b.isna().any():
                raise ValueError(f"benchmark returns missing on {int(b.isna().sum())} dates")
            bg = float((1.0 + b).prod())
            out["benchmark_total_return"] = bg - 1.0
            out["benchmark_annual_return"] = bg ** (1.0 / years) - 1.0 if years > 0 and bg > 0 else float("nan")
            out["active_annual_return"] = out["annual_return"] - out["benchmark_annual_return"]
            act = r - b
            te = float(act.std(ddof=1) * math.sqrt(ppy)) if len(act) > 1 else float("nan")
            out["tracking_error"] = te
            out["information_ratio"] = float(act.mean() * ppy / te) if te else float("nan")
            bvar = float(b.var(ddof=1)) if len(b) > 1 else 0.0
            out["beta"] = float(r.cov(b) / bvar) if bvar else float("nan")
        if excess is not None:
            e = pd.Series(excess, dtype=float).reindex(r.index).dropna()
            if len(e):
                eg = float((1.0 + e).prod())
                out["excess_annual_return"] = eg ** (1.0 / years) - 1.0 if years > 0 and eg > 0 else float("nan")
        return out


# ====================================================================================== conventions bridge
def mechanics_from_conventions(conv: Conventions) -> dict:
    """The mechanics a ``Conventions`` profile implies (for ``Strategy.under``)."""
    mr = MissingReturns(conv.missing_returns, conv.bad_date_threshold)
    exc = ExcessReturn(conv.excess_return, "xr", "tsy_r", conv.prefer_vendor_excess)
    return {"execution_lag": conv.timing.trade_offset_days, "missing": mr, "metrics": Metrics(conv.annualisation, conv.turnover, exc), "benchmark_drift": conv.benchmark_drift, "schedule": conv.rebalance, "cash_return": conv.cash_return, "profile": conv.name, "confirmed": conv.confirmed}


def conventions_of(execution: Execution, missing: MissingReturns, metrics: Metrics, schedule, benchmark_drift: WeightDrift = WeightDrift.DRIFT, financing: Financing | None = None, name: str = "kernel", confirmed: bool = True) -> Conventions:
    """The ``Conventions`` record of a strategy's mechanics (manifests, ledger, official gates)."""
    cash = CashReturn.ZERO if financing is not None and financing.series is None else CashReturn.OVERNIGHT
    return Conventions(name=name, timing=Timing(execution.lag + 1), excess_return=metrics.excess.method, prefer_vendor_excess=metrics.excess.vendor_first, benchmark_drift=benchmark_drift, turnover=metrics.turnover, annualisation=metrics.annualisation, missing_returns=missing.policy, bad_date_threshold=missing.threshold, rebalance=str(schedule), cash_return=cash, confirmed=confirmed)


def profile(name: str) -> dict:
    """Strategy fields that reproduce a legacy framework's conventions, for ``Strategy.with_``."""
    conv = PROFILES.get(name)
    if conv is None:
        raise KeyError(f"unknown conventions profile {name!r}; known: {sorted(PROFILES)}")
    m = mechanics_from_conventions(conv)
    return {"missing": m["missing"], "metrics": m["metrics"], "execution_lag": m["execution_lag"], "profile": name}


__all__ = [
    "STAT_KEYS",
    "Calendar365",
    "Calendar36525",
    "Cash",
    "Compounded",
    "CostModel",
    "DailySum",
    "DropDates",
    "ExcessReturn",
    "Execution",
    "Fail",
    "FieldCost",
    "Financing",
    "FixedCost",
    "Geometric",
    "HalfSpread",
    "HalfSum",
    "Metrics",
    "MissingReturns",
    "NextClose",
    "NoCash",
    "NoCost",
    "OneWayBuys",
    "Participation",
    "Refill",
    "SameClose",
    "ScaledCost",
    "SqrtImpact",
    "SumCost",
    "Trading252",
    "TwoWay",
    "VendorExcess",
    "ZeroFill",
    "conventions_of",
    "mechanics_from_conventions",
    "profile",
    "turnover_vectors",
]
