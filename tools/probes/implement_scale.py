"""Event-engine scale with vendor analytics: build and run cost per bond-day, and memory."""
import resource
import sys
import time

from nautilus_fi.backtest import BacktestSettings, FixedIncomeBacktest, MarketEnvironmentBuilder
from nautilus_fi.backtest.spec_strategy import SpecStrategy, SpecStrategyConfig
from nautilus_fi.construct import BenchmarkPlusTilts, ConstructionSpec
from nautilus_fi.domain import Conventions
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
from nautilus_fi.execution import NET_SPREAD
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel, to_market_inputs

n, years, source = int(sys.argv[1]), float(sys.argv[2]), (sys.argv[3] if len(sys.argv) > 3 else "VENDOR")
end = f"{2015 + int(years) - (1 if years == int(years) else 0)}-12-31" if years >= 1 else "2015-06-30"
d = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=n, start="2015-01-02", end=end))
inp = to_market_inputs(d)
t0 = time.perf_counter(); r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
b = MarketEnvironmentBuilder(inp["reference"], d.dates[0], d.dates[-1], strict_dependencies=False)
b.with_evaluated_prices(inp["evaluated_prices"]).with_curves(inp["curves"]).with_ratings(inp["ratings"]).with_fields(inp["fields"]).with_benchmark_constituents(inp["benchmark_constituents"], "IDX")
if source != "COMPUTED":
    b.with_analytics(inp["analytics"], source)
env = b.build(); t1 = time.perf_counter(); r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
bd = len(env.marks)
spec = ResearchSpec("tilt", FieldSignal("signal.value", zscore_by="sector"), ConstructionSpec(BenchmarkPlusTilts(0.5, max_active=0.005)), Conventions(), calendar="BME")
res = FixedIncomeBacktest(env, SpecStrategy(spec, SpecStrategyConfig(rebalance="BME")), NET_SPREAD, settings=BacktestSettings(starting_cash=500e6, holdings_frequency="M", stream_days=63)).run(); t2 = time.perf_counter()
print(f"n={n} years={years} source={source} bond_days={bd} build={t1 - t0:.1f}s ({(t1 - t0) / bd * 1e3:.3f} ms/bond-day) run={t2 - t1:.1f}s ({(t2 - t1) / bd * 1e3:.3f} ms/bond-day) env_rss+={(r1 - r0) / 1024:.0f}MB status={res.status}/{res.outcome} positions_end={int(res.nav['n_positions'].iloc[-1])} unprocessed={res.extra.get('unprocessed_fills')}")
