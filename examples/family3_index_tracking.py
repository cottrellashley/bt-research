"""Family 3 - index-tracking credit portfolios at index scale.

A custom benchmark (the index ex-financials, issuer-capped at 3%, drifting between reconstitutions),
an optimiser that maximises a value tilt while minimising active OASD and DTS by sector, under the
constraint grammar: active issuer and sector bands, OASD within +/-0.10y, a 10% monthly turnover
budget, and - with ``--milp`` - a minimum holding (semi-continuous weights). The run covers thousands
of bonds over years of daily accounting in minutes.

    python examples/family3_index_tracking.py [--full] [--milp]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_fi.benchmarks import ConstituentBenchmark, CustomBenchmark
from nautilus_fi.construct import ConstructionSpec, Objective, Optimise, OptimiserSettings, TrackingTerm
from nautilus_fi.domain import Conventions
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
from nautilus_fi.paper import PaperData, PaperEngine
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


def main(full: bool = False, milp: bool = False) -> None:
    t0 = time.perf_counter()
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=9000 if full else 3000, start="2015-01-02", end="2024-12-31" if full else "2019-12-31", signal_noise=3.0, n_defaults=5))
    parent = ConstituentBenchmark("IDX", s.membership)
    bench = CustomBenchmark("IDX ex-FIN 3% capped", parent, s.attributes, exclude={"sector": ["FIN"]}, issuer_cap=0.03)
    data = PaperData(s.panel, s.attributes, s.cash, bench)
    constraints = [
        {"kind": "active_group", "attribute": "issuer", "max": 0.005},
        {"kind": "active_group", "attribute": "sector", "min": -0.02, "max": 0.02},
        {"kind": "active_exposure", "attribute": "oasd", "min": -0.10, "max": 0.10},
        {"kind": "active_exposure", "attribute": "dts", "min": -0.05, "max": 0.05, "relative": True},
        {"kind": "turnover", "max": 0.10, "definition": "HALF_SUM"},
        {"kind": "weight", "max": 0.02},
    ]
    if milp:
        constraints.append({"kind": "min_holding", "min": 0.0005})
    spec = ResearchSpec("index_plus_value", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(Optimise(Objective(alpha=0.0003, tracking=TrackingTerm(("oasd", "dts"), ("sector",))), OptimiserSettings(time_limit=30)), constraints), Conventions())
    print(f"universe: {len(s.ids):,} bonds x {len(s.dates):,} days ({int(s.panel['alive'].sum()):,} live bond-days), generated in {time.perf_counter() - t0:.1f}s")
    t1 = time.perf_counter()
    res = PaperEngine(spec, data).run()
    st = res.stats()
    c = res.construction
    print(f"run: {time.perf_counter() - t1:.1f}s, {len(c)} rebalances, median solve {c['opt_solve_seconds'].median():.2f}s, statuses {c['opt_status'].value_counts().to_dict()}")
    print(f"active {st['active_annual_return']:+.2%}  TE {st['tracking_error']:.2%}  IR {st['information_ratio']:.2f}  turnover {st['turnover_annual']:.0%}/yr ({st['conventions']}: {Conventions().turnover.value})  names {st['avg_names']:.0f}  hard violations {st['hard_violations']}")
    print(f"active OASD at rebalances: max |{res.exposures['active_oasd'].loc[res.trades['trade_date']].abs().max():.3f}| years")


if __name__ == "__main__":
    main(full="--full" in sys.argv, milp="--milp" in sys.argv)
