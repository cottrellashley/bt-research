"""Kernel scale probe: vectorised engine at index scale (9k bonds x 10y daily panels)."""
import resource
import sys
import time

import nautilus_fi as nfi
from nautilus_fi.kernel.engine import prepare, simulate

n = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
years = int(sys.argv[2]) if len(sys.argv) > 2 else 10
t = time.time()
src = nfi.SyntheticSource(n_bonds=n, start="2015-01-02", end=f"{2015 + years - 1}-12-31", stress_start="2016-01-15")
print(f"generate {n} bonds x {years}y: {time.time() - t:.1f}s")
ix = nfi.Index("SYNTH_IG")
q_m = nfi.Strategy("q5_monthly", universe=ix, signal=nfi.Field("signal.value").zscore(by="sector"), portfolio=nfi.Quantile(5, by="sector"), benchmark=ix, missing=nfi.ZeroFill())
q_d = q_m.with_(schedule="D", name="q5_daily")
opt = nfi.Strategy("index_plus", universe=ix, signal=nfi.Field("signal.value").zscore(by="sector"), benchmark=ix, missing=nfi.ZeroFill(),
                   portfolio=nfi.Optimise(1e-3 * nfi.Alpha() - 2e-3 * nfi.TrackingError(by=["sector"], fields=["oasd", "dts"]) - nfi.Costs(),
                                          nfi.ActiveWeight("issuer") <= 0.005, nfi.ActiveWeight("sector").within(0.02), nfi.ActiveExposure("oasd").within(0.1), nfi.Turnover() <= 0.1))
t = time.time(); prep = prepare([q_m, q_d, opt], src); print(f"prepare: {time.time() - t:.1f}s ({prep.info['panel_mb']} MB panel)")
for s in (q_m, q_d, opt):
    t = time.time(); r = simulate(s, prep); el = time.time() - t
    tm = r.diagnostics["timings"]
    print(f"{s.name:12s} {el:6.1f}s  universe {tm['universe_seconds']:.1f} signal {tm['signal_seconds']:.1f} targets {tm.get('targets_seconds', 0):.1f} loop {tm['loop_seconds']:.1f} analytics {tm['analytics_seconds']:.1f} | rebalances {r.stats()['n_rebalances']} names {r.stats()['avg_names']:.0f} | {r.headline()[:80]}")
grid = {"portfolio.n": [3, 4, 5, 8, 10], "execution.lag": [0, 1]}
for workers in (1, 3):
    sess = nfi.Session(source=src, name=f"probe{workers}")
    sess._prepared, sess._prep_source, sess._prep_strategies = prep, src, [q_m]
    t = time.time(); sw = sess.sweep(q_m, grid, workers=workers); print(f"sweep 10 variants, {sw.mode}: {time.time() - t:.1f}s")
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
