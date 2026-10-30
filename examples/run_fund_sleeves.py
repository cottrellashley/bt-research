"""Run every fund-style sleeve on the extended synthetic market and print the platform's view of
each: performance, implementation shortfall, evidence quality, funding, invariants and the support
statuses that apply to the products the sleeve holds.

Usage: python examples/run_fund_sleeves.py [--quick] [--sleeve NAME ...] [--ladder] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from a checkout without installing
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fund_sleeves import SLEEVES

from nautilus_fi.backtest import AnalyticsConfig, FixedIncomeBacktest, compare, run_execution_ladder
from nautilus_fi.execution import GROSS, NET_FULL, NET_SPREAD
from nautilus_fi.instruments.money_market import AmortizedCostPolicy
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


def money_market_shadow(result) -> dict:
    """Amortized-cost shadow NAV versus market NAV for the money market sleeve (2a-7 style check):
    each lot is carried at straight-line amortized cost from its purchase price to par at maturity."""
    pol = AmortizedCostPolicy()
    h = result.holdings
    if h.empty:
        return {}
    fills = result.fills[(result.fills["protocol"] != "EXPIRATION") & (result.fills["side"] == "BUY")]
    first_buy = {r.symbol: (float(r.clean_price), pd.Timestamp(r.date).date()) for r in fills.sort_values("date", ascending=False).itertuples()}
    bonds = {s: b for s, b in result.extra.get("bonds", {}).items()} if result.extra.get("bonds") else {}
    worst = 0.0
    for d, g in h.groupby("date"):
        as_of = pd.Timestamp(d).date()
        mkt = float(g["market_value"].sum())
        amort = 0.0
        for r in g.itertuples():
            px, bought = first_buy.get(r.symbol, (float(r.clean_price), as_of))
            maturity = bonds[r.symbol].maturity if r.symbol in bonds else None
            amort += r.face * (pol.amortized_price(px, bought, maturity, as_of) if maturity else r.clean_price) / 100.0
        dev = pol.shadow_deviation(mkt, amort) if amort else 0.0
        worst = max(worst, abs(dev))
    return {"max_shadow_deviation": round(worst, 6), "breaches_half_pct": bool(worst > 0.005)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sleeve", nargs="*", default=None)
    ap.add_argument("--ladder", action="store_true", help="run GROSS / NET_SPREAD / NET_FULL for each sleeve")
    ap.add_argument("--out", default=None, help="folder for report bundles")
    args = ap.parse_args()
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 12, 29) if args.quick else date(2024, 12, 31), seed=7, n_ig=12 if args.quick else 24, n_hy=5 if args.quick else 10)
    t0 = time.time()
    market = generate_synthetic_market(cfg)
    env = market.build_environment(analytics=AnalyticsConfig(every_n_days=1))
    print(f"environment: {len(env.bonds)} instruments, {len(env.dates)} days, built in {time.time() - t0:.1f}s")
    print(pd.Series({b.asset_type.value: 0 for b in env.bonds.values()}).index.tolist(), "\n")
    print("deals:", {k: v["tranches"] for k, v in market.deals.items()}, "\n")
    names = args.sleeve or list(SLEEVES)
    results = {}
    for name in names:
        spec = SLEEVES[name]
        t1 = time.time()
        if args.ladder:
            ladder = run_execution_ladder(env, lambda spec=spec: spec.make(f"{spec.name}-001"), [GROSS, NET_SPREAD, NET_FULL], settings=spec.backtest_settings())
            for k, r in ladder.items():
                results[f"{name}/{k}"] = r
            res = ladder["NET_FULL"]
        else:
            res = FixedIncomeBacktest(env, spec.make(f"{spec.name}-001"), NET_FULL, settings=spec.backtest_settings(), name=name).run()
            results[name] = res
        st = res.stats()
        print(f"[{name:<24}] {spec.archetype:<45} status={st['status']:<10} outcome={st['outcome']:<8} ret={st.get('total_return', float('nan')):+.4f} fills={st.get('n_fills', 0):>3} model_share={st.get('avg_model_marked_share', float('nan')):.3f} ({time.time() - t1:.1f}s)")
        if args.out:
            res.bundle().export(Path(args.out) / name)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n=== comparison ===")
        print(compare(results).round(4).to_string())
        for name, res in results.items():
            if "/" in name and not name.endswith("NET_FULL"):
                continue
            print(f"\n--- {name}: evidence (share of market value by valuation evidence class) ---")
            print(res.evidence_table().round(3).to_string())
            inv = res.invariants()
            if not inv.empty:
                print(f"invariants: {int((inv['passed'] == True).sum())}/{len(inv)} passed")
            cf = res.cashflows
            if not cf.empty:
                print("cashflows:", cf.groupby("type")["amount"].agg(["count", "sum"]).round(0).to_dict("index"))
            spec = SLEEVES[name.split("/")[0]]
            sup = res.support[res.support["product_family"].isin(spec.asset_families + ("PLATFORM",))]
            print(sup[["product_family", "feature", "status", "approximation"]].to_string(index=False))
            if name.startswith("MONEY_MARKET"):
                res.extra["bonds"] = env.bonds
                print("money market shadow NAV:", money_market_shadow(res))


if __name__ == "__main__":
    main()
