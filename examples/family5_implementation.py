"""Family 5 - implementation and live proposals.

One research spec, three stages: Paper on the event engine's own marks, Implement through orders,
fills at bid / ask dirty, minimum pieces and increments, dated settlement, funding and the ledger,
the reconciliation between them, and a live trade proposal from the same spec with CRD / QOE order
files, a PORT position upload, the pre-trade validation report and a position / cash reconciliation.

    python examples/family5_implementation.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from nautilus_fi.backtest import BacktestSettings, FixedIncomeBacktest
from nautilus_fi.backtest.spec_strategy import SpecStrategy, SpecStrategyConfig
from nautilus_fi.construct import ConstructionSpec, EqualWeight
from nautilus_fi.domain import Conventions, MissingReturnPolicy, Timing
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec, UniverseRule
from nautilus_fi.execution import NET_FULL
from nautilus_fi.paper import PaperEngine, PaperSettings
from nautilus_fi.paper.reconcile import paper_data_from_environment, reconcile
from nautilus_fi.propose import (
    QOE_ORDERS,
    build_proposal,
    reconcile_cash,
    reconcile_positions,
    write_orders,
    write_port_upload,
    write_prep_report,
)
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


def main() -> None:
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 9, 29), n_treasuries=3, n_bills=0, n_ig=24, n_hy=0, n_frn=0, n_mbs=0, n_linkers=0, n_frn_arrears=0, n_loans=0, n_hybrids=0, n_munis=0, n_em_sov=0, n_eur_corp=0, n_cp=0, clo_deal=False, abs_deal=False, cmbs_deal=False, default_one_hy=False)
    env = generate_synthetic_market(cfg).build_environment()
    spec = ResearchSpec("wide_spread", FieldSignal("oas_bp"), ConstructionSpec(EqualWeight(top=10, invested=0.95), [{"kind": "weight", "max": 0.12}]), Conventions(timing=Timing(1), missing_returns=MissingReturnPolicy.ZERO), calendar="BME", universe=UniverseRule(field_rules=[("oasd", ">", 1.0)]))
    data = paper_data_from_environment(env)
    paper = PaperEngine(spec, data, PaperSettings(record_daily_weights=True, initial_nav=50e6)).run()
    impl = FixedIncomeBacktest(env, SpecStrategy(spec, SpecStrategyConfig(rebalance="BME", universe_from_benchmark=False)), NET_FULL, settings=BacktestSettings(starting_cash=50e6, conventions=spec.conventions)).run()
    rec = reconcile(paper, impl, data.panel)
    s = rec.summary
    print(f"paper gross {s['paper_gross_total']:+.2%} | paper net {s['paper_net_total']:+.2%} | implemented {s['implemented_total']:+.2%} (fill rate {impl.stats().get('fill_rate', float('nan')):.0%})")
    print(f"shortfall {s['shortfall_vs_gross']:+.2%} = holdings gap {s['holdings_gap']:+.2%} + cash {s['cash']:+.2%} + execution {s['execution']:+.2%} + residual {s['residual']:+.2%}")
    # ---- live proposal from the same spec on the last date, against the implemented holdings
    d = env.end_date
    out = Path("output/family5")
    out.mkdir(parents=True, exist_ok=True)
    last = impl.holdings[impl.holdings["date"] == impl.holdings["date"].max()]
    holdings = dict(zip(last["symbol"], last["face"]))
    marks = {sym: env.mark(sym, d) for sym in env.bonds if env.mark(sym, d) is not None}
    prices = pd.DataFrame({"bid": [m.clean_bid for m in marks.values()], "ask": [m.clean_ask for m in marks.values()], "mid": [m.clean_mid for m in marks.values()], "accrued": [env.accrued(sym, d) for sym in marks], "price_date": d}, index=list(marks))
    target = paper.weights[paper.weights["date"] == paper.weights["date"].max()].set_index("instrument_id")["weight"]
    cash = float(impl.nav["cash"].iloc[-1])
    prop = build_proposal(target, holdings, prices, env.bonds, cash, d, portfolio="FI-CORE", constraints=spec.construction.constraints, spec_hash=spec.hash)
    write_orders(prop, out / "crd_orders.csv")
    write_orders(prop, out / "qoe_orders.csv", QOE_ORDERS)
    write_port_upload(prop, out / "port_upload.csv", env.bonds)
    write_prep_report(prop, out / "prep_validation.csv")
    print(f"proposal: {prop.summary()}")
    print(f"files: {sorted(p.name for p in out.iterdir())}")
    after = prop.holdings_after.set_index("symbol")["face"]
    print("position reconciliation (proposal vs itself):", reconcile_positions(after, after)["status"].value_counts().to_dict(), "| cash:", reconcile_cash(prop.cash_after, prop.cash_after)["status"])


if __name__ == "__main__":
    main()
