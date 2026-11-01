"""When do sweep workers pay? Optimiser variants (expensive per variant) sequential vs processes."""
import sys
import time

import nautilus_fi as nfi

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
src = nfi.SyntheticSource(n_bonds=n, start="2015-01-02", end="2019-12-31")
ix = nfi.Index("SYNTH_IG")
base = nfi.Strategy("ip", universe=ix, signal=nfi.Field("signal.value").zscore(by="sector"), benchmark=ix, missing=nfi.ZeroFill(),
                    portfolio=nfi.Optimise(1e-3 * nfi.Alpha() - 2e-3 * nfi.TrackingError(by=["sector"]) - nfi.Costs(), nfi.ActiveWeight("issuer") <= 0.005, nfi.ActiveWeight("sector").within(0.02), nfi.Turnover() <= 0.1))
variants = {f"to_{x:g}": base.with_(portfolio=base.portfolio.replace(nfi.Turnover() <= x), name=f"to_{x:g}") for x in (0.02, 0.04, 0.06, 0.08, 0.12, 0.16)}
for workers in (1, 3):
    t = time.time(); sw = nfi.Session(source=src).sweep(variants, workers=workers); print(f"{len(variants)} optimiser variants, {sw.mode}: {time.time() - t:.1f}s", flush=True)
