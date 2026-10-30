"""Synthetic fixed income market generator.

Produces schema-conforming frames for every input the environment builder accepts so the whole stack
runs end to end without vendor data: Treasuries, bills and commercial paper, IG and HY corporates (some
callable), term and compounded-in-arrears SOFR floaters, leveraged loans, hybrid perpetuals, tax-exempt
municipals, hard-currency EM sovereigns, EUR corporates with an FX series, MBS-style amortizing pools,
CPI-driven and index-ratio linkers, CLO / ABS / CMBS deals whose tranche payments and factors come from
the internal waterfall engine, a default with recovery, calls, rating changes, TRACE-like prints with a
15-minute dissemination lag, evaluated prices with quality scores, an OIS curve, a stress regime and a
market-value weighted gross benchmark.

The generator is for plumbing and tests, not for drawing conclusions about markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nautilus_fi.data.trace import TraceCleaningConfig, clean_trace
from nautilus_fi.execution.costs import BASE_HALF_SPREAD_BP
from nautilus_fi.instruments.analytics import accrued_interest, price_from_spread
from nautilus_fi.instruments.bond import (
    AmortizationTerms,
    BondSpec,
    CallOption,
    FloatingTerms,
    InflationTerms,
)
from nautilus_fi.instruments.calendar import us_bond_market_calendar
from nautilus_fi.instruments.cashflows import MarketInputs
from nautilus_fi.instruments.curves import NelsonSiegelSvensson, ZeroCurve
from nautilus_fi.instruments.enums import (
    AssetType,
    CouponType,
    DayCount,
    Frequency,
    PrincipalType,
    Seniority,
    rating_bucket,
    rating_to_score,
    score_to_rating,
)
from nautilus_fi.instruments.linkers import US_TIPS, base_cpi_for
from nautilus_fi.instruments.money_market import commercial_paper
from nautilus_fi.instruments.schedule import add_months
from nautilus_fi.structured.collateral import CollateralPool, PoolAssumptions
from nautilus_fi.structured.engine import InternalWaterfallEngine
from nautilus_fi.structured.waterfall import Deal, DealState, Tranche

BUCKET_SPREAD_BP = {"AAA-AA": 45.0, "A": 75.0, "BBB": 130.0, "BB": 300.0, "B": 450.0, "CCC-D": 900.0, "NR": 200.0}
SECTOR_OFFSET_BP = {"FIN": 10.0, "IND": 0.0, "UTL": -10.0, "ENE": 25.0, "TMT": 5.0}


@dataclass
class SyntheticMarketConfig:
    start: date = date(2023, 1, 3)
    end: date = date(2024, 12, 31)
    seed: int = 7
    n_treasuries: int = 5
    n_bills: int = 1
    n_ig: int = 24
    n_hy: int = 10
    n_frn: int = 4
    n_mbs: int = 2
    n_linkers: int = 1
    callable_share: float = 0.35
    default_one_hy: bool = True
    stress_episode: bool = True
    tenors: tuple[float, ...] = (0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)
    # extended product set (all optional; zero disables)
    n_frn_arrears: int = 1          # SOFR compounded-in-arrears floaters (daily fixings)
    n_loans: int = 2                # leveraged loans: SOFR + spread, index floor, T+7 settlement
    n_hybrids: int = 1              # perpetual callable hybrids / preferreds
    n_munis: int = 2                # tax-exempt municipals
    n_em_sov: int = 2               # hard-currency (USD) EM sovereigns
    n_eur_corp: int = 2             # EUR corporates (adds an EURUSD spot + 3m points series)
    n_cp: int = 1                   # commercial paper maturing inside the window
    clo_deal: bool = True           # one CLO (5 debt tranches) run through the internal waterfall
    abs_deal: bool = True           # one sequential-pay auto ABS (2 debt tranches), monthly
    cmbs_deal: bool = True          # one CMBS conduit (3 debt tranches), balloon collateral
    linkers_use_cpi: bool = True    # first linker indexed off a CPI series (others off index ratios)
    ois_curve: bool = True          # add a USD-OIS discount curve (GOVT - 5bp)


@dataclass
class SyntheticMarket:
    config: SyntheticMarketConfig
    bonds: dict[str, BondSpec]
    reference: pd.DataFrame
    evaluated_prices: pd.DataFrame
    trades: pd.DataFrame
    curves: pd.DataFrame
    fixings: pd.DataFrame
    factors: pd.DataFrame
    index_ratios: pd.DataFrame
    ratings: pd.DataFrame
    corporate_actions: pd.DataFrame
    stress: pd.DataFrame
    benchmark: pd.DataFrame
    true_prices: pd.DataFrame
    liquidity_bucket: dict[str, int] = field(default_factory=dict)
    cpi: pd.DataFrame = field(default_factory=pd.DataFrame)
    fx: pd.DataFrame = field(default_factory=pd.DataFrame)
    tranche_payments: pd.DataFrame = field(default_factory=pd.DataFrame)
    deals: dict[str, dict] = field(default_factory=dict)

    def build_environment(self, **builder_kwargs):
        from nautilus_fi.backtest.environment import MarketEnvironmentBuilder

        b = MarketEnvironmentBuilder(self.bonds, self.config.start, self.config.end, **builder_kwargs)
        b.with_evaluated_prices(self.evaluated_prices).with_trades(self.trades).with_curves(self.curves)
        b.with_fixings(self.fixings).with_fixings(self.fixings, daily=True).with_ratings(self.ratings).with_corporate_actions(self.corporate_actions)
        b.with_stress(self.stress).with_benchmark(self.benchmark)
        if not self.factors.empty:
            b.with_factors(self.factors)
        if not self.index_ratios.empty:
            b.with_index_ratios(self.index_ratios)
        if not self.cpi.empty:
            b.with_cpi(self.cpi)
        if not self.fx.empty:
            b.with_fx(self.fx)
        if not self.tranche_payments.empty:
            b.with_tranche_payments(self.tranche_payments)
        return b.build()


def _rng_choice(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


def generate_synthetic_market(cfg: SyntheticMarketConfig | None = None) -> SyntheticMarket:
    cfg = cfg or SyntheticMarketConfig()
    rng = np.random.default_rng(cfg.seed)
    cal = us_bond_market_calendar()
    dates = cal.business_days(cfg.start, cfg.end)
    n_days = len(dates)
    tenors = np.array(cfg.tenors)

    # ---------------------------------------------------------------- rates: NSS random walk
    beta0, beta1, beta2 = 0.045, -0.012, -0.01
    curve_rows, zero_curves = [], {}
    for i, d in enumerate(dates):
        if i > 0:
            beta0 += rng.normal(0, 0.0004)
            beta1 += rng.normal(0, 0.0003)
            beta2 += rng.normal(0, 0.0003)
        nss = NelsonSiegelSvensson(beta0, beta1, beta2, 0.0, 1.8, 10.0)
        par = np.asarray(nss.yield_(tenors))
        zero_curves[d] = ZeroCurve(d, tenors, par, "GOVT")   # treat as zero rates for truth & inputs (kind=zero)
        for t, r in zip(tenors, par):
            curve_rows.append({"date": pd.Timestamp(d), "curve": "GOVT", "tenor": float(t), "rate": float(r), "kind": "zero"})
            if cfg.ois_curve:
                curve_rows.append({"date": pd.Timestamp(d), "curve": "USD-OIS", "tenor": float(t), "rate": float(r) - 0.0005, "kind": "zero"})
    curves_df = pd.DataFrame(curve_rows)

    # ---------------------------------------------------------------- credit factor & stress
    s = 1.0
    credit_factor, stress_vals = [], []
    episode_start = n_days // 2 if cfg.stress_episode else None
    for i in range(n_days):
        shock = rng.normal(0, 0.012)
        if episode_start is not None and episode_start <= i < episode_start + 25:
            shock += 0.05
        if episode_start is not None and episode_start + 25 <= i < episode_start + 70:
            shock -= 0.02
        s = max(0.5, s * (1 - 0.02) + 0.02 * 1.0 + shock)  # mean-reverting to 1
        credit_factor.append(s)
        stress_vals.append(max(0.0, (s - 1.0) / 1.2))
    stress_df = pd.DataFrame({"date": [pd.Timestamp(d) for d in dates], "value": stress_vals, "name": "SYNTH_STRESS"})

    # ---------------------------------------------------------------- fixings (SOFR ~ 3m rate - 5bp)
    sofr = {d: float(zero_curves[d].zero(0.25)) - 0.0005 for d in dates}
    fix_rows = [{"date": pd.Timestamp(d), "index": "SOFR", "rate": r} for d, r in sofr.items()]
    # pre-history so floaters issued earlier have fixings
    for k in range(1, 400):
        d0 = cfg.start - timedelta(days=k)
        fix_rows.append({"date": pd.Timestamp(d0), "index": "SOFR", "rate": sofr[dates[0]] + rng.normal(0, 0.0005)})
    fixings_df = pd.DataFrame(fix_rows)

    # ---------------------------------------------------------------- CPI (monthly, published ~12th of the following month)
    cpi_rows: list[dict] = []
    cpi_hist: dict[tuple[int, int], float] = {}
    level = 290.0
    m = add_months(date(cfg.start.year, cfg.start.month, 1), -30)
    while m <= add_months(cfg.end, 1):
        level *= 1 + 0.03 / 12 + rng.normal(0, 0.0012)
        cpi_hist[(m.year, m.month)] = level
        pub = add_months(m, 1).replace(day=12)
        cpi_rows.append({"index": "CPI-U", "year": m.year, "month": m.month, "value": level, "publish_ts": pd.Timestamp(pub, tz="UTC") + pd.Timedelta(hours=12, minutes=30)})
        m = add_months(m, 1)
    cpi_df = pd.DataFrame(cpi_rows)

    # ---------------------------------------------------------------- FX: EURUSD spot (random walk) and 3m forward points
    fx_rows: list[dict] = []
    eurusd = 1.08
    eurusd_hist: dict[date, float] = {}
    if cfg.n_eur_corp > 0:
        for d in dates:
            eurusd *= 1 + rng.normal(0, 0.004)
            eurusd_hist[d] = eurusd
            fx_rows.append({"date": pd.Timestamp(d), "pair": "EURUSD", "rate": eurusd, "tenor_months": np.nan, "points": np.nan})
            fx_rows.append({"date": pd.Timestamp(d), "pair": "EURUSD", "rate": np.nan, "tenor_months": 3, "points": (sofr[d] - 0.02) * 0.25 * eurusd * 1e4})   # covered-interest-parity style points (pips)
    fx_df = pd.DataFrame(fx_rows, columns=["date", "pair", "rate", "tenor_months", "points"])

    # ---------------------------------------------------------------- bonds
    bonds: dict[str, BondSpec] = {}
    meta: dict[str, dict] = {}   # per-bond synthetic parameters
    sectors = list(SECTOR_OFFSET_BP)

    def issue_before(years_max: float) -> date:
        return cfg.start - timedelta(days=int(rng.integers(30, int(365 * years_max))))

    k = 0
    for i in range(cfg.n_treasuries):
        mat_years = [2, 3, 5, 7, 10, 30][i % 6]
        issue = issue_before(1.5)
        mat = add_months(issue, mat_years * 12)
        c = round(float(zero_curves[dates[0]].par_yield(mat_years)) * 800) / 800
        sym = f"UST{mat_years:02d}Y{k:03d}"
        bonds[sym] = BondSpec(sym, "UST", issue, mat, c, day_count=DayCount.ACT_ACT_ICMA, asset_type=AssetType.SOVEREIGN, min_piece=100, increment=100, amount_outstanding=40e9, rating_initial="AAA", is_benchmark=True, sector="GOVT")
        meta[sym] = {"spread0": 0.0, "idio_vol": 0.0, "liq": 1, "noise_bp": 0.5, "score": 9.8}
        k += 1
    for i in range(cfg.n_bills):
        issue = cfg.start + timedelta(days=int(rng.integers(5, 40)))
        mat = issue + timedelta(days=182)
        sym = f"BILL{k:03d}"
        bonds[sym] = BondSpec(sym, "UST", issue, mat, 0.0, coupon_type=CouponType.ZERO, day_count=DayCount.ACT_360, asset_type=AssetType.BILL, min_piece=100, increment=100, amount_outstanding=60e9, rating_initial="AAA", sector="GOVT")
        meta[sym] = {"spread0": 0.0, "idio_vol": 0.0, "liq": 1, "noise_bp": 0.3, "score": 9.9}
        k += 1
    for grp, n in (("IG", cfg.n_ig), ("HY", cfg.n_hy)):
        for i in range(n):
            rating = _rng_choice(rng, ["AA", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]) if grp == "IG" else _rng_choice(rng, ["BB+", "BB", "BB-", "B+", "B", "B-"])
            sector = _rng_choice(rng, sectors)
            issue = issue_before(6.0) if rng.random() < 0.85 else cfg.start + timedelta(days=int(rng.integers(30, 300)))
            mat_years = int(_rng_choice(rng, [3, 5, 5, 7, 7, 10, 10, 30])) if grp == "IG" else int(_rng_choice(rng, [5, 7, 8]))
            mat = add_months(issue, mat_years * 12)
            score = rating_to_score(rating)
            spread0 = (BUCKET_SPREAD_BP[rating_bucket(score)] + SECTOR_OFFSET_BP[sector]) * (1 + rng.normal(0, 0.15))
            base_curve = zero_curves[dates[0]]
            c = round((float(base_curve.par_yield(min(mat_years, 30))) + spread0 / 1e4 + rng.normal(0, 0.002)) * 800) / 800
            sym = f"{grp}{k:03d}"
            calls = []
            if rng.random() < cfg.callable_share:
                first_call = add_months(issue, max(12, (mat_years - 3) * 12 if mat_years > 4 else 24))
                calls = [CallOption(first_call, 100.0 + c * 100 / 2), CallOption(add_months(first_call, 12), 100.0 + c * 100 / 4), CallOption(add_months(first_call, 24), 100.0)]
            liq = int(_rng_choice(rng, [1, 1, 2, 2, 2, 3, 3])) if grp == "IG" else int(_rng_choice(rng, [1, 2, 2, 3, 3]))
            amount = float(_rng_choice(rng, [300e6, 500e6, 750e6, 1e9, 1.5e9, 2.5e9]))
            bonds[sym] = BondSpec(sym, f"{sector}CO{k:02d}", issue, mat, c, asset_type=AssetType.CORPORATE, sector=sector, rating_initial=rating, call_schedule=calls, amount_outstanding=amount, is_144a=(grp == "HY" and rng.random() < 0.5), tags={"high_yield": grp == "HY"})
            meta[sym] = {"spread0": spread0, "idio_vol": 0.02 if grp == "IG" else 0.04, "liq": liq, "noise_bp": {1: 4.0, 2: 12.0, 3: 35.0}[liq] * (1.5 if grp == "HY" else 1.0), "score": {1: 8.5, 2: 6.5, 3: 4.0}[liq]}
            k += 1
    for i in range(cfg.n_frn):
        issue = issue_before(2.0)
        mat = add_months(issue, int(_rng_choice(rng, [3, 5])) * 12)
        spread_bp = float(_rng_choice(rng, [60, 90, 120, 150]))
        sym = f"FRN{k:03d}"
        bonds[sym] = BondSpec(sym, f"BANK{k:02d}", issue, mat, 0.0, coupon_type=CouponType.FLOATING, frequency=Frequency.QUARTERLY, day_count=DayCount.ACT_360, asset_type=AssetType.CORPORATE, sector="FIN", rating_initial="A", floating=FloatingTerms("SOFR", spread_bp), amount_outstanding=1e9)
        meta[sym] = {"spread0": spread_bp, "idio_vol": 0.015, "liq": 2, "noise_bp": 8.0, "score": 7.0, "frn": True}
        k += 1
    for i in range(cfg.n_mbs):
        issue = cfg.start - timedelta(days=int(rng.integers(400, 1500)))
        mat = add_months(issue, 360)
        c = float(_rng_choice(rng, [0.03, 0.035, 0.045, 0.055, 0.06]))
        sym = f"POOL{k:03d}"
        bonds[sym] = BondSpec(sym, "FNMA", issue, mat, c, frequency=Frequency.MONTHLY, day_count=DayCount.THIRTY_360, asset_type=AssetType.MBS, principal_type=PrincipalType.AMORTIZING, amortization=AmortizationTerms(original_term_months=360, projected_cpr=0.07), min_piece=1000, increment=1, amount_outstanding=5e9, rating_initial="AAA", sector="MBS")
        meta[sym] = {"spread0": 50.0, "idio_vol": 0.01, "liq": 1, "noise_bp": 3.0, "score": 9.0, "mbs": True}
        k += 1
    for i in range(cfg.n_linkers):
        issue = issue_before(2.0)
        mat = add_months(issue, 120)
        sym = f"TIPS{k:03d}"
        bonds[sym] = BondSpec(sym, "UST", issue, mat, 0.0125, coupon_type=CouponType.INFLATION_LINKED, day_count=DayCount.ACT_ACT_ICMA, asset_type=AssetType.LINKER, inflation=InflationTerms("CPI-U"), min_piece=100, increment=100, amount_outstanding=20e9, rating_initial="AAA", sector="GOVT")
        meta[sym] = {"spread0": -230.0, "idio_vol": 0.005, "liq": 1, "noise_bp": 1.5, "score": 9.5, "linker": True, "cpi": bool(cfg.linkers_use_cpi and i == 0)}
        k += 1
    for i in range(cfg.n_frn_arrears):
        issue = issue_before(1.5)
        mat = add_months(issue, 36)
        spread_bp = float(_rng_choice(rng, [45, 70, 95]))
        sym = f"FRNA{k:03d}"
        bonds[sym] = BondSpec(sym, f"BANK{k:02d}", issue, mat, 0.0, coupon_type=CouponType.FLOATING, frequency=Frequency.QUARTERLY, day_count=DayCount.ACT_360, asset_type=AssetType.CORPORATE, sector="FIN", rating_initial="A+", floating=FloatingTerms("SOFR", spread_bp, method="COMPOUNDED_IN_ARREARS", lookback_days=2), amount_outstanding=1.5e9, tags={"in_arrears": True})
        meta[sym] = {"spread0": spread_bp, "idio_vol": 0.012, "liq": 2, "noise_bp": 6.0, "score": 7.5, "frn": True, "arrears": True}
        k += 1
    for i in range(cfg.n_loans):
        issue = issue_before(3.0)
        mat = add_months(issue, 84)
        spread_bp = float(_rng_choice(rng, [325, 375, 425, 500]))
        rating = _rng_choice(rng, ["BB-", "B+", "B", "B-"])
        sym = f"LOAN{k:03d}"
        bonds[sym] = BondSpec(sym, f"LBO{k:02d}", issue, mat, 0.0, coupon_type=CouponType.FLOATING, frequency=Frequency.QUARTERLY, day_count=DayCount.ACT_360, asset_type=AssetType.LOAN, sector=_rng_choice(rng, sectors), seniority=Seniority.SENIOR_SECURED, rating_initial=rating, floating=FloatingTerms("SOFR", spread_bp, floor=0.005, floor_on_index=True), min_piece=1_000_000, increment=1, amount_outstanding=float(_rng_choice(rng, [500e6, 750e6, 1.2e9])), calendar_name="US_BOND", tags={"loan": True, "high_yield": True})
        meta[sym] = {"spread0": spread_bp, "idio_vol": 0.03, "liq": 3, "noise_bp": 20.0, "score": 5.0, "frn": True, "loan": True}
        k += 1
    for i in range(cfg.n_hybrids):
        issue = issue_before(3.0)
        first_call = add_months(issue, 60)
        c = round((float(zero_curves[dates[0]].par_yield(5)) + 0.035) * 800) / 800
        sym = f"HYB{k:03d}"
        bonds[sym] = BondSpec(sym, f"BANK{k:02d}", issue, None, c, asset_type=AssetType.PREFERRED, sector="FIN", seniority=Seniority.SUBORDINATED, rating_initial="BB+", call_schedule=[CallOption(first_call, 100.0), CallOption(add_months(first_call, 60), 100.0)], amount_outstanding=1e9, tags={"hybrid": True, "reset_spread_bp": 350.0, "coupon_deferrable": True})
        meta[sym] = {"spread0": 380.0, "idio_vol": 0.035, "liq": 2, "noise_bp": 18.0, "score": 6.0, "hybrid": True}
        k += 1
    for i in range(cfg.n_munis):
        issue = issue_before(5.0)
        mat_years = int(_rng_choice(rng, [10, 15, 20, 30]))
        mat = add_months(issue, mat_years * 12)
        sym = f"MUNI{k:03d}"
        state_ = _rng_choice(rng, ["CA", "NY", "TX", "IL", "FL"])
        bonds[sym] = BondSpec(sym, f"STATE-{state_}", issue, mat, 0.05, asset_type=AssetType.MUNICIPAL, sector="MUNI", rating_initial=_rng_choice(rng, ["AAA", "AA+", "AA", "AA-"]), call_schedule=[CallOption(add_months(issue, 120), 100.0)] if mat_years > 10 else [], min_piece=5_000, increment=5_000, amount_outstanding=float(_rng_choice(rng, [100e6, 250e6, 500e6])), tags={"tax_exempt": True, "state": state_})
        meta[sym] = {"spread0": -60.0, "idio_vol": 0.01, "liq": 3, "noise_bp": 10.0, "score": 6.5, "muni": True}
        k += 1
    for i in range(cfg.n_em_sov):
        issue = issue_before(4.0)
        mat_years = int(_rng_choice(rng, [7, 10, 30]))
        mat = add_months(issue, mat_years * 12)
        country = _rng_choice(rng, ["BR", "MX", "ID", "ZA", "TR", "CO"])
        rating = _rng_choice(rng, ["BBB-", "BB+", "BB", "BB-", "B+"])
        spread0 = (BUCKET_SPREAD_BP[rating_bucket(rating_to_score(rating))] + 40.0) * (1 + rng.normal(0, 0.1))
        c = round((float(zero_curves[dates[0]].par_yield(min(mat_years, 30))) + spread0 / 1e4) * 800) / 800
        sym = f"EMSOV{k:03d}"
        bonds[sym] = BondSpec(sym, f"REPUBLIC-{country}", issue, mat, c, asset_type=AssetType.SOVEREIGN, sector="EM_SOV", country=country, rating_initial=rating, min_piece=200_000, increment=1_000, amount_outstanding=float(_rng_choice(rng, [1e9, 1.5e9, 2.5e9])), is_144a=False, tags={"em_hard_currency": True, "high_yield": rating_to_score(rating) >= rating_to_score("BB+")})
        meta[sym] = {"spread0": spread0, "idio_vol": 0.035, "liq": 2, "noise_bp": 12.0, "score": 7.0, "em": True}
        k += 1
    for i in range(cfg.n_eur_corp):
        issue = issue_before(4.0)
        mat_years = int(_rng_choice(rng, [4, 6, 8]))
        mat = add_months(issue, mat_years * 12)
        rating = _rng_choice(rng, ["A", "A-", "BBB+", "BBB"])
        spread0 = (BUCKET_SPREAD_BP[rating_bucket(rating_to_score(rating))] - 10.0) * (1 + rng.normal(0, 0.1))
        c = round((float(zero_curves[dates[0]].par_yield(mat_years)) - 0.015 + spread0 / 1e4) * 800) / 800
        sym = f"EUR{k:03d}"
        bonds[sym] = BondSpec(sym, f"EUCO{k:02d}", issue, mat, max(c, 0.00125), frequency=Frequency.ANNUAL, day_count=DayCount.ACT_ACT_ICMA, currency="EUR", country="DE", asset_type=AssetType.CORPORATE, sector=_rng_choice(rng, sectors), rating_initial=rating, calendar_name="WEEKENDS", min_piece=100_000, increment=1_000, amount_outstanding=float(_rng_choice(rng, [500e6, 750e6, 1e9])), tags={"eur": True})
        meta[sym] = {"spread0": spread0, "idio_vol": 0.02, "liq": 2, "noise_bp": 8.0, "score": 7.5, "eur": True}
        k += 1
    for i in range(cfg.n_cp):
        issue = cfg.start + timedelta(days=int(rng.integers(3, 20)))
        sym = f"CP{k:03d}"
        bonds[sym] = commercial_paper(sym, f"FIN-CP{k:02d}", issue, issue + timedelta(days=90))
        bonds[sym].rating_initial = "A+"          # long-term scale stand-in for A-1/P-1 paper
        bonds[sym].amount_outstanding = 2e9
        meta[sym] = {"spread0": 25.0, "idio_vol": 0.0, "liq": 1, "noise_bp": 1.0, "score": 9.0, "cp": True}
        k += 1

    # ---------------------------------------------------------------- structured deals through the internal waterfall engine
    tranche_rows: list[dict] = []
    deal_meta: dict[str, dict] = {}
    deal_factor_rows: list[dict] = []
    deal_factor_hist: dict[str, dict[date, float]] = {}
    engine = InternalWaterfallEngine()

    def run_deal(deal_id: str, asset_type: AssetType, tranche_terms: list[tuple[str, float, float | None, float, str]], pool: CollateralPool, assumptions: PoolAssumptions, closing: date, reinvest_months: int, legal_final_years: int, frequency: int, rating_by_tranche: dict[str, str], spread_noise: dict[str, float], liq_by_tranche: dict[str, int]) -> None:
        """tranche_terms: (name, share, fixed coupon or None, spread_bp, DEBT|EQUITY)."""
        nonlocal k
        total = pool.balance
        tranches = []
        for j, (name, share, fixed, spread_bp, kind) in enumerate(tranche_terms):
            tranches.append(Tranche(name, total * share, fixed, None if fixed is not None else "SOFR", spread_bp, j + 1, kind, oc_trigger=(1.08 + 0.02 * j if kind == "DEBT" and j >= 1 else None), ic_trigger=(1.10 if kind == "DEBT" and j >= 1 else None), deferrable=(kind == "DEBT" and j >= 3)))
        deal = Deal(deal_id, tranches, reinvestment_end=add_months(closing, reinvest_months) if reinvest_months else None, payment_frequency=frequency)
        state = DealState.initial(deal)
        months = 12 // frequency
        legal_final = add_months(closing, legal_final_years * 12)
        syms = {}
        for t in deal.debt_tranches():
            sym = f"{deal_id}{t.name}"
            floating = FloatingTerms("SOFR", t.spread_bp) if t.coupon_fixed is None else None
            bonds[sym] = BondSpec(sym, deal_id, closing, legal_final, t.coupon_fixed or 0.0, coupon_type=CouponType.FLOATING if floating else CouponType.FIXED, frequency={4: Frequency.QUARTERLY, 12: Frequency.MONTHLY, 2: Frequency.SEMI_ANNUAL}[frequency], day_count=DayCount.ACT_360 if floating else DayCount.THIRTY_360, asset_type=asset_type, principal_type=PrincipalType.AMORTIZING, amortization=AmortizationTerms(paydown_price=100.0), floating=floating, seniority=Seniority.SENIOR_SECURED if t.seniority == 1 else Seniority.SUBORDINATED, rating_initial=rating_by_tranche[t.name], sector=asset_type.value, min_piece=250_000, increment=1_000, amount_outstanding=t.original_balance, tags={"deal": deal_id, "tranche": t.name, "seniority": t.seniority})
            meta[sym] = {"spread0": t.spread_bp + spread_noise[t.name] if floating else spread_noise[t.name], "idio_vol": 0.01 + 0.008 * t.seniority, "liq": liq_by_tranche[t.name], "noise_bp": 4.0 * t.seniority, "score": max(4.0, 9.0 - 1.2 * t.seniority), "frn": bool(floating), "structured": True}
            syms[t.name] = sym
            deal_factor_hist[sym] = {closing: 1.0}
            k += 1
        period_start = closing
        while period_start < cfg.end:
            period_end = add_months(period_start, months)
            fix_date = max((d for d in sofr if d <= period_start), default=dates[0])
            results, flows, _snap = engine.project(deal, state, pool, period_start, 1, assumptions, {"SOFR": sofr[fix_date]})
            for fl in flows:
                if fl.tranche not in syms:
                    continue
                sym = syms[fl.tranche]
                orig = next(t.original_balance for t in deal.tranches if t.name == fl.tranche)
                deal_factor_hist[sym][period_end] = fl.factor_end
                if period_end >= cfg.start:
                    tranche_rows.append({"symbol": sym, "pay_date": pd.Timestamp(period_end), "interest_per_100": fl.interest / orig * 100.0, "principal_per_100": fl.principal / orig * 100.0, "deferred_per_100": fl.deferred / orig * 100.0, "source": fl.source})
                    deal_factor_rows.append({"date": pd.Timestamp(period_end), "instrument_id": sym, "factor": fl.factor_end, "paydown_price": 100.0, "publish_ts": pd.Timestamp(period_end, tz="UTC") + pd.Timedelta(hours=15)})
            period_start = period_end
        deal_meta[deal_id] = {"asset_type": asset_type.value, "tranches": list(syms.values()), "closing": closing, "reinvestment_end": deal.reinvestment_end, "legal_final": legal_final, "pool_balance": total, "assumptions": assumptions.__dict__, "engine": f"{engine.name}@{engine.version}"}

    if cfg.clo_deal:
        closing = add_months(date(cfg.start.year, cfg.start.month, 15), -12)
        run_deal(f"CLO{k:03d}", AssetType.CLO, [("AAA", 0.62, None, 150.0, "DEBT"), ("AA", 0.10, None, 210.0, "DEBT"), ("A", 0.07, None, 270.0, "DEBT"), ("BBB", 0.06, None, 400.0, "DEBT"), ("BB", 0.05, None, 725.0, "DEBT"), ("EQ", 0.10, None, 0.0, "EQUITY")],
                 CollateralPool(500e6, sofr[dates[0]] + 0.036, 72, amortizing=False), PoolAssumptions(cpr=0.22, cdr=0.02, severity=0.35, recovery_lag_months=9, reinvestment_spread_bp=0.0), closing, 18, 12, 4,
                 {"AAA": "AAA", "AA": "AA", "A": "A", "BBB": "BBB-", "BB": "BB-"}, {"AAA": 5.0, "AA": 8.0, "A": 12.0, "BBB": 20.0, "BB": 40.0}, {"AAA": 2, "AA": 2, "A": 3, "BBB": 3, "BB": 3})
    if cfg.abs_deal:
        closing = add_months(date(cfg.start.year, cfg.start.month, 20), -6)
        run_deal(f"ABS{k:03d}", AssetType.ABS, [("A", 0.86, 0.052, 0.0, "DEBT"), ("B", 0.09, 0.066, 0.0, "DEBT"), ("EQ", 0.05, None, 0.0, "EQUITY")],
                 CollateralPool(300e6, 0.071, 60, amortizing=True), PoolAssumptions(cpr=0.15, cdr=0.025, severity=0.5, recovery_lag_months=6), closing, 0, 6, 12,
                 {"A": "AAA", "B": "BBB"}, {"A": 70.0, "B": 220.0}, {"A": 2, "B": 3})
    if cfg.cmbs_deal:
        closing = add_months(date(cfg.start.year, cfg.start.month, 10), -24)
        run_deal(f"CMBS{k:03d}", AssetType.CMBS, [("A", 0.70, 0.048, 0.0, "DEBT"), ("B", 0.15, 0.056, 0.0, "DEBT"), ("C", 0.10, 0.068, 0.0, "DEBT"), ("EQ", 0.05, None, 0.0, "EQUITY")],
                 CollateralPool(400e6, 0.056, 84, amortizing=False), PoolAssumptions(cpr=0.03, cdr=0.03, severity=0.4, recovery_lag_months=12), closing, 0, 10, 4,
                 {"A": "AAA", "B": "A-", "C": "BB"}, {"A": 110.0, "B": 260.0, "C": 650.0}, {"A": 2, "B": 3, "C": 3})
    tranche_df = pd.DataFrame(tranche_rows, columns=["symbol", "pay_date", "interest_per_100", "principal_per_100", "deferred_per_100", "source"])

    # ---------------------------------------------------------------- factor paths (MBS) and index ratios
    factor_rows, ratio_rows = [], []
    factor_hist: dict[str, dict[date, float]] = {}
    ratio_hist: dict[str, dict[date, float]] = {}
    for sym, b in bonds.items():
        if meta[sym].get("mbs"):
            f = 1.0
            hist = {}
            m = b.issue_date
            while m <= cfg.end:
                age = (m.year - b.issue_date.year) * 12 + (m.month - b.issue_date.month)
                rate_level = float(zero_curves[dates[0]].zero(5.0)) if m < cfg.start else float(zero_curves[min(dates, key=lambda d: abs((d - m).days))].zero(5.0))
                incentive = b.coupon_rate - rate_level
                cpr = min(0.6, max(0.02, 0.06 + 2.5 * incentive))
                smm = 1 - (1 - cpr) ** (1 / 12)
                r = b.coupon_rate / 12
                sched = ((1 + r) ** 360 - (1 + r) ** (age + 1)) / ((1 + r) ** 360 - (1 + r) ** age) if age < 359 else 0.0
                f = f * sched * (1 - smm)
                hist[m] = f
                if m >= cfg.start:
                    factor_rows.append({"date": pd.Timestamp(m), "instrument_id": sym, "factor": f, "paydown_price": 100.0, "publish_ts": pd.Timestamp(m, tz="UTC") + pd.Timedelta(hours=14)})
                m = add_months(m, 1)
            factor_hist[sym] = hist
        if meta[sym].get("cpi"):
            meta[sym]["cpi_base"] = base_cpi_for(cpi_hist, b.dated_date, US_TIPS)
            continue
        if meta[sym].get("linker"):
            ratio = 1.08
            hist = {}
            m = cfg.start - timedelta(days=200)
            while m <= cfg.end:
                ratio *= 1 + 0.025 / 12 + rng.normal(0, 0.001)
                hist[m] = ratio
                if m >= cfg.start - timedelta(days=40):
                    ratio_rows.append({"date": pd.Timestamp(m), "instrument_id": sym, "index_ratio": ratio})
                m = add_months(m, 1)
            ratio_hist[sym] = hist
    factors_df = pd.DataFrame(factor_rows + deal_factor_rows, columns=["date", "instrument_id", "factor", "paydown_price", "publish_ts"])
    factor_hist.update(deal_factor_hist)
    ratios_df = pd.DataFrame(ratio_rows, columns=["date", "instrument_id", "index_ratio"])

    # ---------------------------------------------------------------- ratings, default, calls
    rating_rows, action_rows = [], []
    current_score = {sym: rating_to_score(b.rating_initial) for sym, b in bonds.items()}
    for sym, b in bonds.items():
        rating_rows.append({"date": pd.Timestamp(cfg.start - timedelta(days=1)), "instrument_id": sym, "rating": b.rating_initial, "agency": "COMPOSITE"})
    default_sym = None
    hy_syms = [s_ for s_ in bonds if bonds[s_].tags.get("high_yield")]
    if cfg.default_one_hy and hy_syms:
        default_sym = hy_syms[int(rng.integers(0, len(hy_syms)))]
    default_idx = int(n_days * 0.62)
    default_date = dates[default_idx]
    recovery_date = cal.add_business_days(default_date, 45)
    if default_sym is not None:
        action_rows.append({"instrument_id": default_sym, "action": "DEFAULT", "effective_date": pd.Timestamp(default_date), "price": 38.0, "announce_ts": pd.Timestamp(default_date, tz="UTC") + pd.Timedelta(hours=8), "note": "missed coupon"})
        action_rows.append({"instrument_id": default_sym, "action": "RECOVERY", "effective_date": pd.Timestamp(recovery_date), "price": 41.0, "announce_ts": pd.Timestamp(recovery_date, tz="UTC") - pd.Timedelta(days=5), "note": "workout distribution"})
        for j in (default_idx - 60, default_idx - 20):
            if j > 0:
                current_score[default_sym] = min(22, current_score[default_sym] + 3)
                rating_rows.append({"date": pd.Timestamp(dates[j]), "instrument_id": default_sym, "rating": score_to_rating(current_score[default_sym]), "agency": "COMPOSITE"})
    for sym in bonds:
        if sym == default_sym or bonds[sym].asset_type.is_government:
            continue
        if rng.random() < 0.25:
            j = int(rng.integers(20, n_days - 20))
            current_score[sym] = int(np.clip(current_score[sym] + int(_rng_choice(rng, [-1, 1, 1, 2])), 1, 21))
            rating_rows.append({"date": pd.Timestamp(dates[j]), "instrument_id": sym, "rating": score_to_rating(current_score[sym]), "agency": "COMPOSITE"})
    ratings_df = pd.DataFrame(rating_rows)

    # ---------------------------------------------------------------- true prices, evaluated prices, trades, benchmark
    price_rows, trade_rows, true_rows = [], [], []
    idio = {sym: 0.0 for sym in bonds}
    bench_rows = []
    prev_dirty: dict[str, float] = {}
    called_at: dict[str, date] = {}
    msg = 0
    for i, d in enumerate(dates):
        curve = zero_curves[d]
        cf = credit_factor[i]
        stress = stress_vals[i]
        bench_num = bench_den = 0.0
        for sym, b in bonds.items():
            m = meta[sym]
            if not b.is_alive(d) or (sym in called_at and d >= called_at[sym]):
                continue
            if sym == default_sym and d >= recovery_date:
                continue
            idio[sym] = idio[sym] * 0.97 + rng.normal(0, m["idio_vol"])
            spread_bp = m["spread0"] * cf * (1 + idio[sym]) if m["spread0"] > 0 else m["spread0"] * (1 + 0.3 * idio[sym])
            settle = b.settle_date(d)
            known_cpi = {ym: v for ym, v in cpi_hist.items() if add_months(date(ym[0], ym[1], 1), 1).replace(day=12) <= d} if m.get("cpi") else None
            inputs = MarketInputs(
                fixings={"SOFR": {dd: r for dd, r in sofr.items() if dd <= d}} if m.get("frn") else None,
                realized_factors={dd: f for dd, f in factor_hist.get(sym, {}).items() if dd <= d} or None,
                index_ratios={dd: r for dd, r in ratio_hist.get(sym, {}).items() if dd <= d} or None,
                flat_forward_rate=sofr[d] if m.get("frn") else None,
                daily_fixings={"SOFR": {dd: r for dd, r in sofr.items() if dd <= d}} if m.get("arrears") else None,
                cpi=known_cpi, cpi_base=m.get("cpi_base"), known_as_of=d,
            )
            if sym == default_sym and d >= default_date:
                clean = 38.0 + rng.normal(0, 0.5)
            else:
                try:
                    clean = price_from_spread(b, settle, curve, spread_bp, inputs)
                except Exception:
                    continue
                if clean <= 0:
                    continue
            # callable: exercised when deep in the money at a call date
            if b.call_schedule and sym not in called_at:
                for c in b.call_schedule:
                    if 0 <= (c.date - d).days <= 35 and clean > c.price + 1.5 and c.date <= cfg.end:
                        called_at[sym] = c.date
                        action_rows.append({"instrument_id": sym, "action": "CALL", "effective_date": pd.Timestamp(c.date), "price": c.price, "announce_ts": pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=14), "note": "called"})
                        break
            factor = 1.0
            if m.get("mbs") or m.get("structured"):
                known = [f for dd, f in sorted(factor_hist[sym].items()) if dd <= d]
                factor = known[-1] if known else 1.0
            if m.get("cpi"):
                factor = __import__("nautilus_fi.instruments.cashflows", fromlist=["factor_at"]).factor_at(b, settle, inputs)
            elif m.get("linker"):
                known = [r for dd, r in ratio_hist[sym].items() if dd <= d]
                factor = known[-1] if known else 1.0
            accrued = 0.0 if (sym == default_sym and d >= default_date) else accrued_interest(b, settle, inputs)
            true_rows.append({"date": pd.Timestamp(d), "instrument_id": sym, "clean": clean, "spread_bp": spread_bp, "factor": factor})
            hs_bp = BASE_HALF_SPREAD_BP[b.asset_type] * (1.5 if m["liq"] == 3 else 1.0) * (1 + 1.2 * stress)
            noise = rng.normal(0, m["noise_bp"] / 1e4 * clean) * (1 + stress)
            mid = clean + noise
            price_rows.append({"date": pd.Timestamp(d), "instrument_id": sym, "mid": mid, "bid": mid * (1 - hs_bp / 1e4), "ask": mid * (1 + hs_bp / 1e4), "score": float(np.clip(m["score"] + rng.normal(0, 0.4) - 2.0 * stress, 1, 10)), "source": "SYNTH_EVAL", "publish_ts": pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=16, minutes=20)})
            lam = {1: 5.0, 2: 1.0, 3: 0.12}[m["liq"]] * (1 + 0.6 * stress) * (0.35 if (m.get("mbs") or m.get("structured")) else 1.0) * (0.0 if (m.get("loan") or m.get("eur")) else 1.0)
            n_tr = int(rng.poisson(lam))
            for _ in range(n_tr):
                side = _rng_choice(rng, ["B", "S", "D"])
                adj = {"B": 1.0, "S": -1.0, "D": 0.0}[side]
                px = clean * (1 + adj * hs_bp / 1e4) + rng.normal(0, m["noise_bp"] / 2e4 * clean)
                size = float(np.exp(rng.normal(np.log(1_000_000 if b.asset_type in (AssetType.CORPORATE, AssetType.MUNICIPAL, AssetType.PREFERRED) else 10_000_000 if b.asset_type.is_government else 2_000_000), 0.9)))
                size = max(10_000.0, round(size / 10_000) * 10_000)
                exec_ts = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=int(rng.integers(9, 15)), minutes=int(rng.integers(0, 60)))
                msg += 1
                trade_rows.append({"instrument_id": sym, "exec_ts": exec_ts, "report_ts": exec_ts + pd.Timedelta(minutes=int(rng.integers(1, 15))), "price": px, "size": size, "side": side, "status": "T", "msg_id": str(msg)})
            dirty = (clean + accrued) * factor
            if sym in prev_dirty and b.amount_outstanding:
                coupon_cash = 0.0
                p = b.current_period(d)
                if p is not None and p.accrual_start == d and b.coupon_type != CouponType.ZERO:
                    coupon_cash = b.coupon_rate_on(d) * 100 / max(int(b.frequency), 1) * factor
                ret = (dirty + coupon_cash) / prev_dirty[sym] - 1.0
                w = b.amount_outstanding
                bench_num += w * ret
                bench_den += w
            prev_dirty[sym] = dirty
        bench_rows.append({"date": pd.Timestamp(d), "total_return": bench_num / bench_den if bench_den else 0.0, "name": "SYNTH_AGG"})

    trades_raw = pd.DataFrame(trade_rows)
    trades_df = clean_trace(trades_raw, config=TraceCleaningConfig(min_size=0.0), is_high_yield={s_: bool(bonds[s_].tags.get("high_yield")) for s_ in bonds}) if len(trades_raw) else pd.DataFrame(columns=["instrument_id", "exec_ts", "available_ts", "price", "size", "public_size", "size_revealed_ts", "side"])
    from nautilus_fi.data.reference import reference_to_dataframe

    return SyntheticMarket(
        config=cfg, bonds=bonds, reference=reference_to_dataframe(bonds), evaluated_prices=pd.DataFrame(price_rows), trades=trades_df,
        curves=curves_df, fixings=fixings_df, factors=factors_df, index_ratios=ratios_df, ratings=ratings_df,
        corporate_actions=pd.DataFrame(action_rows, columns=["instrument_id", "action", "effective_date", "price", "announce_ts", "note"]),
        stress=stress_df, benchmark=pd.DataFrame(bench_rows), true_prices=pd.DataFrame(true_rows),
        liquidity_bucket={s_: meta[s_]["liq"] for s_ in bonds},
        cpi=cpi_df if cfg.linkers_use_cpi else pd.DataFrame(), fx=fx_df, tranche_payments=tranche_df, deals=deal_meta,
    )


__all__ = ["BUCKET_SPREAD_BP", "SyntheticMarket", "SyntheticMarketConfig", "generate_synthetic_market"]
