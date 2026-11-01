"""Named, recorded conventions: the settings fixed income backtesting frameworks disagree on.

Every choice that changes a reported number is an explicit field here rather than a hard-coded
behaviour: when a decision first earns, how excess returns compound over a period, whether benchmark
weights drift between reconstitutions, what turnover means, how statistics annualise, what happens
to a missing return, which calendar rebalances and what cash earns. A ``Conventions`` object is
recorded in every manifest and printed next to every statistic.

Named profiles reproduce legacy frameworks so their numbers can be matched first and then compared
with the canonical profile (``Conventions.profile("MRI_v1")`` ...). Profiles marked ``unconfirmed``
encode what the framework review could read; confirm them against golden outputs before relying on
parity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum

import numpy as np
import pandas as pd


class ExcessReturnMethod(str, Enum):
    VENDOR = "VENDOR"              # use the published excess return series as given (compounded where needed)
    COMPOUNDED = "COMPOUNDED"      # (prod(1 + TR) - 1) - (prod(1 + TSY) - 1) over the period
    GEOMETRIC = "GEOMETRIC"        # prod(1 + TR) / prod(1 + TSY) - 1
    DAILY_SUM = "DAILY_SUM"        # sum of daily excess returns (legacy; drifts from vendor figures in stress)


class WeightDrift(str, Enum):
    DRIFT = "DRIFT"                # weights move with returns between reconstitutions
    STATIC = "STATIC"              # weights held at the reconstitution values until the next one


class TurnoverDefinition(str, Enum):
    ONE_WAY_BUYS = "ONE_WAY_BUYS"  # sum of weight increases
    HALF_SUM = "HALF_SUM"          # half the sum of absolute weight changes
    TWO_WAY = "TWO_WAY"            # sum of absolute weight changes (round trip)


class Annualisation(str, Enum):
    TRADING_252 = "TRADING_252"            # 252 / 52 / 12 periods per year by observation frequency
    CALENDAR_365 = "CALENDAR_365"          # years = calendar days / 365
    CALENDAR_365_25 = "CALENDAR_365_25"    # years = calendar days / 365.25


class MissingReturnPolicy(str, Enum):
    ZERO = "ZERO"                  # a missing return on a held asset is a zero return
    DROP_DATE = "DROP_DATE"        # bad dates leave the run (portfolio, benchmark and statistics); other gaps are zero-filled and counted
    REFILL = "REFILL"              # missing values are taken from a refill source; FAIL if still missing
    FAIL = "FAIL"                  # any missing return on a held asset stops the run


class ExitPolicy(str, Enum):
    SELL = "SELL"                  # a held name that leaves the universe is sold at the next rebalance, outside any turnover budget
    EXCLUDE = "EXCLUDE"            # it may only shrink; selling it comes first within the turnover budget
    HOLD_NOT_BUY = "HOLD_NOT_BUY"  # it may be held (or trimmed) but never bought


class CashReturn(str, Enum):
    ZERO = "ZERO"
    OVERNIGHT = "OVERNIGHT"        # overnight index (SOFR, ESTR, SONIA), accrued per period
    BILL_INDEX = "BILL_INDEX"      # a bill index total return series


@dataclass(frozen=True)
class Timing:
    """Trading days from the information date (last data a decision may use) to the end of the first
    period whose return the new weights earn. ``return_lag_days = 2``: data to close T, trade at the
    close of T+1, first return T+1 -> T+2. ``1``: trade on the close the signal sees. On panels of
    another frequency the lag counts panel periods (months for a monthly panel)."""

    return_lag_days: int = 2

    def __post_init__(self) -> None:
        if self.return_lag_days < 1:
            raise ValueError("return_lag_days must be >= 1: a decision cannot earn the return of its own information date")

    @property
    def trade_offset_days(self) -> int:
        """Trading days from the information date to the trade (close) date."""
        return self.return_lag_days - 1


@dataclass(frozen=True)
class Conventions:
    name: str = "canonical"
    timing: Timing = field(default_factory=Timing)
    excess_return: ExcessReturnMethod = ExcessReturnMethod.COMPOUNDED
    prefer_vendor_excess: bool = True                    # use vendor excess returns where supplied
    benchmark_drift: WeightDrift = WeightDrift.DRIFT
    turnover: TurnoverDefinition = TurnoverDefinition.HALF_SUM
    annualisation: Annualisation = Annualisation.CALENDAR_365
    missing_returns: MissingReturnPolicy = MissingReturnPolicy.FAIL
    bad_date_threshold: float = 0.5                      # share of a cross-section missing or exactly zero
    bad_date_zeros_only: bool = False                    # count only exactly-zero returns as bad (missing ones are gaps, not a bad date)
    rebalance: str = "BME"
    cash_return: CashReturn = CashReturn.OVERNIGHT
    exits: ExitPolicy = ExitPolicy.SELL                  # held names that leave the universe
    confirmed: bool = True
    notes: str = ""

    # ------------------------------------------------------------------ construction
    def with_(self, **changes) -> Conventions:
        if "timing" in changes and isinstance(changes["timing"], int):
            changes["timing"] = Timing(changes["timing"])
        return replace(self, **changes)

    @staticmethod
    def profile(name: str) -> Conventions:
        try:
            return PROFILES[name]
        except KeyError as e:
            raise KeyError(f"unknown conventions profile {name!r}; known: {sorted(PROFILES)}") from e

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timing"] = self.timing.return_lag_days
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return d

    def describe(self) -> str:
        return (f"{self.name}: lag {self.timing.return_lag_days}d, excess {self.excess_return.value}{' (vendor first)' if self.prefer_vendor_excess else ''}, "
                f"benchmark {self.benchmark_drift.value}, turnover {self.turnover.value}, annualisation {self.annualisation.value}, "
                f"missing {self.missing_returns.value}, rebalance {self.rebalance}, cash {self.cash_return.value}, exits {self.exits.value}")

    # ------------------------------------------------------------------ statistics
    def years(self, index: pd.Index) -> float:
        return years_spanned(index, self.annualisation)

    def periods_per_year(self, index: pd.Index) -> float:
        return periods_per_year(index, self.annualisation)


PROFILES: dict[str, Conventions] = {
    "canonical": Conventions(),
    "official": Conventions(name="official", missing_returns=MissingReturnPolicy.FAIL, annualisation=Annualisation.CALENDAR_365, notes="official backtest records: 365 / calendar days"),
    "fi_lib_research": Conventions(name="fi_lib_research", timing=Timing(2), confirmed=False, notes="trades T+1, first earns T+2; excess-return formula unreadable in the review, confirm"),
    "EUR_IG": Conventions(name="EUR_IG", timing=Timing(2), confirmed=False, notes="EUR IG LightGBM: trades T+1, first earns T+2"),
    "MRI_v1": Conventions(name="MRI_v1", timing=Timing(1), excess_return=ExcessReturnMethod.DAILY_SUM, prefer_vendor_excess=False, missing_returns=MissingReturnPolicy.ZERO, confirmed=False, notes="earns close T -> T+1 on the close its signal sees; sums daily excess returns"),
    "erin_beta_plus": Conventions(name="erin_beta_plus", timing=Timing(1), benchmark_drift=WeightDrift.STATIC, turnover=TurnoverDefinition.ONE_WAY_BUYS, missing_returns=MissingReturnPolicy.REFILL, bad_date_threshold=0.8, bad_date_zeros_only=True, confirmed=False, notes="T-1 inputs earn the T return; published daily benchmark weights, never drifted; one-way buys turnover; dates with >= 80% exactly-zero returns refilled (Erin refills from the next orphan date within 4 days: name the refill field)"),
    "fi_lib_backtesting": Conventions(name="fi_lib_backtesting", benchmark_drift=WeightDrift.DRIFT, turnover=TurnoverDefinition.TWO_WAY, missing_returns=MissingReturnPolicy.ZERO, confirmed=False, notes="benchmark weights drift between reconstitutions; round-trip turnover"),
    "paperport": Conventions(name="paperport", annualisation=Annualisation.TRADING_252, missing_returns=MissingReturnPolicy.ZERO, confirmed=False, notes="252 periods / sqrt(252); turnover definition unreadable in the review, confirm"),
}


_SIGNATURE = ("timing", "excess_return", "prefer_vendor_excess", "benchmark_drift", "turnover", "annualisation", "missing_returns", "bad_date_threshold", "bad_date_zeros_only", "cash_return")


def unconfirmed_profile(conv: Conventions) -> str | None:
    """The unconfirmed legacy profile whose distinguishing conventions (where it differs from the
    canonical profile) ``conv`` carries, whatever ``conv`` is called. Confirmation belongs to the
    resolved values, so relabelling a legacy profile's mechanics does not confirm them."""
    base = PROFILES["canonical"]
    for name, prof in PROFILES.items():
        if prof.confirmed:
            continue
        diff = [f for f in _SIGNATURE if getattr(prof, f) != getattr(base, f)]
        if diff and all(getattr(conv, f) == getattr(prof, f) for f in diff):
            return name
    return None


# ====================================================================================== statistics
def _spacing_days(index: pd.Index) -> float:
    idx = pd.DatetimeIndex(index)
    if len(idx) < 2:
        return 1.0
    return float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float)))


def _trading_ppy(index: pd.Index) -> float:
    s = _spacing_days(index)
    if s <= 4:
        return 252.0
    if s <= 10:
        return 52.0
    if s <= 40:
        return 12.0
    if s <= 120:
        return 4.0
    return 1.0


def years_spanned(index: pd.Index, annualisation: Annualisation) -> float:
    """Years covered by a return series indexed by period end. Calendar bases count the days from the
    first period's start (one median spacing before the first index date) to the last period end."""
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    if n == 0:
        return float("nan")
    if annualisation == Annualisation.TRADING_252:
        return n / _trading_ppy(idx)
    days = (idx[-1] - idx[0]).days + _spacing_days(idx) if n > 1 else _spacing_days(idx)
    return days / (365.0 if annualisation == Annualisation.CALENDAR_365 else 365.25)


def periods_per_year(index: pd.Index, annualisation: Annualisation) -> float:
    if annualisation == Annualisation.TRADING_252:
        return _trading_ppy(index)
    y = years_spanned(index, annualisation)
    return len(index) / y if y and y > 0 else float("nan")


def annualised_return(returns: pd.Series, conventions: Conventions) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    y = conventions.years(r.index)
    growth = float(np.prod(1.0 + r.to_numpy()))
    return growth ** (1.0 / y) - 1.0 if y > 0 and growth > 0 else float("nan")


def annualised_vol(returns: pd.Series, conventions: Conventions) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(conventions.periods_per_year(r.index)))


def sharpe_ratio(returns: pd.Series, conventions: Conventions, cash: pd.Series | None = None) -> float:
    r = pd.Series(returns).dropna()
    if cash is not None:
        r = r - pd.Series(cash).reindex(r.index).fillna(0.0)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    if not sd:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(conventions.periods_per_year(r.index)))


def performance_stats(returns: pd.Series, conventions: Conventions, cash: pd.Series | None = None, benchmark: pd.Series | None = None) -> dict:
    """Headline statistics, all on one stated basis."""
    r = pd.Series(returns).dropna()
    out: dict = {"conventions": conventions.name, "annualisation": conventions.annualisation.value}
    if r.empty:
        return out
    nav = (1.0 + r).cumprod()
    out["total_return"] = float(nav.iloc[-1] - 1.0)
    out["annual_return"] = annualised_return(r, conventions)
    out["annual_vol"] = annualised_vol(r, conventions)
    out["sharpe"] = sharpe_ratio(r, conventions, cash)
    out["max_drawdown"] = float((nav / nav.cummax() - 1.0).min())
    out["periods_per_year"] = conventions.periods_per_year(r.index)
    out["years"] = conventions.years(r.index)
    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index)
        if b.isna().any():
            raise ValueError(f"benchmark returns missing on {int(b.isna().sum())} dates; apply a MissingReturnPolicy first")
        active = r - b
        out["benchmark_total_return"] = float((1.0 + b).prod() - 1.0)
        out["active_annual_return"] = out["annual_return"] - annualised_return(b, conventions)
        out["tracking_error"] = annualised_vol(active, conventions)
        out["information_ratio"] = float(active.mean() * conventions.periods_per_year(r.index) / out["tracking_error"]) if out["tracking_error"] else float("nan")
    return out


# ====================================================================================== turnover
def turnover(before: pd.Series | dict, after: pd.Series | dict, definition: TurnoverDefinition = TurnoverDefinition.HALF_SUM) -> float:
    """Turnover between two weight vectors (fractions of NAV) under a named definition."""
    b = pd.Series(before, dtype=float)
    a = pd.Series(after, dtype=float)
    idx = b.index.union(a.index)
    delta = a.reindex(idx).fillna(0.0) - b.reindex(idx).fillna(0.0)
    if definition == TurnoverDefinition.ONE_WAY_BUYS:
        return float(delta.clip(lower=0.0).sum())
    if definition == TurnoverDefinition.HALF_SUM:
        return float(0.5 * delta.abs().sum())
    return float(delta.abs().sum())


def turnover_all(before, after) -> dict[str, float]:
    return {d.value: turnover(before, after, d) for d in TurnoverDefinition}


# ====================================================================================== excess returns
def compound_excess(total: pd.Series | np.ndarray, treasury: pd.Series | np.ndarray | None = None, method: ExcessReturnMethod = ExcessReturnMethod.COMPOUNDED, vendor_excess: pd.Series | np.ndarray | None = None) -> float:
    """Excess return over a period from per-period total returns and duration-matched Treasury returns.

    ``VENDOR`` compounds the vendor's per-period excess returns geometrically when no other rule is
    given; ``DAILY_SUM`` reproduces frameworks that add daily excess returns (it understates or
    overstates the vendor figure when returns are large, e.g. October 2008)."""
    if method == ExcessReturnMethod.VENDOR:
        if vendor_excess is None:
            raise ValueError("VENDOR excess return method needs the vendor excess series")
        v = np.asarray(vendor_excess, dtype=float)
        return float(np.prod(1.0 + v) - 1.0)
    tr = np.asarray(total, dtype=float)
    if treasury is None:
        if method == ExcessReturnMethod.DAILY_SUM and vendor_excess is not None:
            return float(np.sum(np.asarray(vendor_excess, dtype=float)))
        raise ValueError(f"{method.value} excess return needs the Treasury return series")
    ts = np.asarray(treasury, dtype=float)
    if method == ExcessReturnMethod.DAILY_SUM:
        return float(np.sum(tr - ts))
    if method == ExcessReturnMethod.GEOMETRIC:
        return float(np.prod(1.0 + tr) / np.prod(1.0 + ts) - 1.0)
    return float((np.prod(1.0 + tr) - 1.0) - (np.prod(1.0 + ts) - 1.0))


def excess_method(method: ExcessReturnMethod, prefer_vendor: bool, has_vendor: bool) -> ExcessReturnMethod:
    """The method that applies: vendor excess returns replace the computed ones when supplied and
    preferred (never under ``DAILY_SUM``, which reproduces frameworks that sum their own)."""
    if has_vendor and prefer_vendor and method != ExcessReturnMethod.DAILY_SUM:
        return ExcessReturnMethod.VENDOR
    return method


def annualised_excess(total, treasury, method: ExcessReturnMethod, years: float, vendor_excess=None, prefer_vendor: bool = True) -> float:
    """Annual excess return over ``years`` from per-period legs, under the convention:

    * ``VENDOR``: the compounded vendor excess series, annualised;
    * ``COMPOUNDED``: each leg compounded and annualised, then differenced (``g_tr ** (1/y) - g_tsy ** (1/y)``);
    * ``GEOMETRIC``: the annualised ratio of the compounded legs;
    * ``DAILY_SUM``: the summed per-period differences, per year (legacy).

    Every engine's statistics call this, so a reported excess return never compounds per-period
    differences (which is none of the four)."""
    if not years or not np.isfinite(years) or years <= 0:
        return float("nan")
    method = excess_method(method, prefer_vendor, vendor_excess is not None)
    if method == ExcessReturnMethod.VENDOR:
        if vendor_excess is None:
            raise ValueError("VENDOR excess return method needs the vendor excess series")
        g = float(np.prod(1.0 + np.asarray(vendor_excess, dtype=float)))
        return g ** (1.0 / years) - 1.0 if g > 0 else float("nan")
    if treasury is None:
        if method == ExcessReturnMethod.DAILY_SUM and vendor_excess is not None:
            return float(np.sum(np.asarray(vendor_excess, dtype=float)) / years)
        raise ValueError(f"{method.value} excess return needs the Treasury return series")
    tr, ts = np.asarray(total, dtype=float), np.asarray(treasury, dtype=float)
    if method == ExcessReturnMethod.DAILY_SUM:
        return float(np.sum(tr - ts) / years)
    gt, gs = float(np.prod(1.0 + tr)), float(np.prod(1.0 + ts))
    if gt <= 0 or gs <= 0:
        return float("nan")
    if method == ExcessReturnMethod.GEOMETRIC:
        return (gt / gs) ** (1.0 / years) - 1.0
    return gt ** (1.0 / years) - gs ** (1.0 / years)


def excess_return_series(total: pd.Series, treasury: pd.Series | None, conventions: Conventions, freq: str | None = None, vendor_excess: pd.Series | None = None) -> pd.Series:
    """Per-period excess returns, optionally aggregated to ``freq`` (e.g. "ME") under the convention
    (each period's legs compounded separately: ``compound_excess``)."""
    method = excess_method(conventions.excess_return, conventions.prefer_vendor_excess, vendor_excess is not None)
    total = pd.Series(total).astype(float)
    tsy = pd.Series(treasury).reindex(total.index).astype(float) if treasury is not None else None
    ven = pd.Series(vendor_excess).reindex(total.index).astype(float) if vendor_excess is not None else None
    if freq is None:
        if method == ExcessReturnMethod.VENDOR:
            return ven.rename("excess")
        if tsy is None:
            raise ValueError("excess returns need Treasury returns or vendor excess returns")
        if method == ExcessReturnMethod.GEOMETRIC:
            return ((1.0 + total) / (1.0 + tsy) - 1.0).rename("excess")
        return (total - tsy).rename("excess")
    groups = total.groupby(pd.Grouper(freq=freq))
    out = {}
    for k, g in groups:
        if g.empty:
            continue
        out[k] = compound_excess(g.to_numpy(), None if tsy is None else tsy.loc[g.index].to_numpy(), method, None if ven is None else ven.loc[g.index].to_numpy())
    return pd.Series(out, name="excess", dtype=float)


# ====================================================================================== missing returns
@dataclass
class MissingReturnReport:
    policy: str
    bad_dates: list = field(default_factory=list)
    filled_zero: int = 0
    refilled: int = 0
    dropped_dates: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)     # (date, asset) still missing

    def to_dict(self) -> dict:
        return {"policy": self.policy, "bad_dates": [str(d) for d in self.bad_dates], "filled_zero": self.filled_zero, "refilled": self.refilled, "dropped_dates": [str(d) for d in self.dropped_dates], "unresolved": len(self.unresolved)}


class MissingReturnsError(ValueError):
    pass


def detect_bad_dates(returns: pd.DataFrame, threshold: float = 0.5, alive: pd.DataFrame | None = None, zeros_only: bool = False) -> list:
    """Dates where at least ``threshold`` of the live cross-section is missing or exactly zero
    (vendor holiday mismatches, stale snapshots); ``zeros_only``: exactly-zero returns only."""
    if returns.empty:
        return []
    live = alive.reindex_like(returns).fillna(False).astype(bool) if alive is not None else pd.DataFrame(True, index=returns.index, columns=returns.columns)
    flagged = (returns == 0.0) if zeros_only else (returns.isna() | (returns == 0.0))
    bad_share = (flagged & live).sum(axis=1) / live.sum(axis=1).replace(0, np.nan)
    return [d for d, s in bad_share.items() if s >= threshold and live.loc[d].sum() > 1]


def apply_missing_policy(returns: pd.DataFrame | pd.Series, policy: MissingReturnPolicy, held: pd.DataFrame | None = None, refill: pd.DataFrame | pd.Series | None = None, threshold: float = 0.5, alive: pd.DataFrame | None = None) -> tuple[pd.DataFrame | pd.Series, MissingReturnReport]:
    """Resolve missing returns under a policy. ``held`` (same shape, bool) restricts the check to
    positions that are actually held; ``alive`` marks where an asset exists (missing outside it is not
    a gap). Returns the cleaned frame and a report recorded with the run."""
    is_series = isinstance(returns, pd.Series)
    df = returns.to_frame() if is_series else returns.copy()
    rep = MissingReturnReport(policy.value)
    exists = alive.reindex_like(df).fillna(False).astype(bool) if alive is not None else pd.DataFrame(True, index=df.index, columns=df.columns)
    rep.bad_dates = detect_bad_dates(df.where(exists), threshold, exists) if df.shape[1] > 1 else []
    relevant = df.isna() & exists
    if held is not None:
        relevant &= held.reindex_like(df).fillna(False).astype(bool)
    if policy == MissingReturnPolicy.ZERO:
        rep.filled_zero = int(relevant.to_numpy().sum())
        df = df.mask(relevant, 0.0)
    elif policy == MissingReturnPolicy.DROP_DATE:
        drop = set(rep.bad_dates) | set(df.index[relevant.any(axis=1)])
        rep.dropped_dates = sorted(drop)
        df = df.drop(index=list(drop))
    elif policy == MissingReturnPolicy.REFILL:
        src = refill.to_frame() if isinstance(refill, pd.Series) else refill
        if src is not None:
            src = src.reindex(index=df.index, columns=df.columns)
            bad = pd.DataFrame(np.repeat(df.index.isin(rep.bad_dates)[:, None], df.shape[1], axis=1), index=df.index, columns=df.columns)
            fill_mask = (relevant | (bad & exists)) & src.notna()
            rep.refilled = int(fill_mask.to_numpy().sum())
            df = df.mask(fill_mask, src)
        still = df.isna() & exists
        if held is not None:
            still &= held.reindex_like(df).fillna(False).astype(bool)
        rep.unresolved = [(d, c) for d, c in zip(*np.nonzero(still.to_numpy()))]
        if rep.unresolved:
            raise MissingReturnsError(f"{len(rep.unresolved)} returns still missing after refill (first at {df.index[rep.unresolved[0][0]]}, {df.columns[rep.unresolved[0][1]]})")
    else:
        if relevant.to_numpy().any():
            r, c = np.nonzero(relevant.to_numpy())
            raise MissingReturnsError(f"{len(r)} missing returns on held assets under FAIL (first at {df.index[r[0]]}, {df.columns[c[0]]}); choose ZERO, DROP_DATE or REFILL explicitly")
    if is_series:
        return df.iloc[:, 0], rep
    return df, rep


__all__ = [
    "PROFILES",
    "Annualisation",
    "CashReturn",
    "Conventions",
    "ExcessReturnMethod",
    "ExitPolicy",
    "MissingReturnPolicy",
    "MissingReturnReport",
    "MissingReturnsError",
    "Timing",
    "TurnoverDefinition",
    "WeightDrift",
    "annualised_excess",
    "annualised_return",
    "annualised_vol",
    "apply_missing_policy",
    "compound_excess",
    "detect_bad_dates",
    "excess_method",
    "excess_return_series",
    "performance_stats",
    "periods_per_year",
    "sharpe_ratio",
    "turnover",
    "turnover_all",
    "unconfirmed_profile",
    "years_spanned",
]
