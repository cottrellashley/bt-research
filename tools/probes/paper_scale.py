"""Panel engine at index scale: 9,000 bonds x 10 years daily, monthly optimiser rebalances."""
import resource
import sys
import time

from nautilus_fi.benchmarks import ConstituentBenchmark
from nautilus_fi.construct import (
    ConstructionSpec,
    Objective,
    Optimise,
    OptimiserSettings,
    Quantile,
    TrackingTerm,
)
from nautilus_fi.domain import Conventions
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
from nautilus_fi.paper import PaperData, PaperEngine
from nautilus_fi.paper.evaluate import EvaluationSettings, evaluate_signal
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel

n, years = int(sys.argv[1]), int(sys.argv[2])
t0 = time.perf_counter()
d = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=n, start="2015-01-02", end=f"{2014 + years}-12-31", signal_noise=3.0, n_defaults=10))
t1 = time.perf_counter()
print(f"generated {len(d.dates)} days x {n} bonds in {t1 - t0:.1f}s, panel {d.panel.nbytes() / 1e9:.2f} GB, alive bond-days {int(d.panel['alive'].sum()):,}", flush=True)
data = PaperData(d.panel, d.attributes, d.cash, ConstituentBenchmark("IDX", d.membership), d.overlays)
t = time.perf_counter(); ev = evaluate_signal(d.panel, "signal.value", Conventions(), EvaluationSettings(horizons=(21, 63)), d.attributes)
print(f"evaluate: {time.perf_counter() - t:.1f}s", ev.summary()[["mean_ic", "ic_tstat_nw", "spread_annualised"]].round(4).to_dict("index"), flush=True)
q5 = ResearchSpec("value_q5", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(Quantile(5, weighting="benchmark")), Conventions())
t = time.perf_counter(); r = PaperEngine(q5, data).run()
print(f"paper quintile: {time.perf_counter() - t:.1f}s", {k: round(v, 4) for k, v in r.stats().items() if k in ("annual_return", "active_annual_return", "tracking_error", "turnover_annual", "avg_names")}, flush=True)
opt = ResearchSpec("index_plus", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(Optimise(Objective(alpha=0.0003, tracking=TrackingTerm(fields=("oasd", "dts"), by=("sector",))), OptimiserSettings(time_limit=30)), [{"kind": "active_group", "attribute": "issuer", "max": 0.005}, {"kind": "active_group", "attribute": "sector", "min": -0.02, "max": 0.02}, {"kind": "active_exposure", "attribute": "oasd", "min": -0.1, "max": 0.1}, {"kind": "turnover", "max": 0.08}, {"kind": "weight", "max": 0.02}]), Conventions())
t = time.perf_counter(); o = PaperEngine(opt, data).run()
c = o.construction
print(f"paper optimiser: {time.perf_counter() - t:.1f}s for {len(c)} rebalances (median solve {c['opt_solve_seconds'].median():.2f}s)", {k: round(v, 4) for k, v in o.stats().items() if k in ("annual_return", "active_annual_return", "tracking_error", "information_ratio", "turnover_annual", "avg_names", "hard_violations")}, flush=True)
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
