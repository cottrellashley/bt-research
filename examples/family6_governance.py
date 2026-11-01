"""Family 6 - research governance, end to end.

Pre-register a hypothesis; run a parameter search in the panel engine with every trial in the
ledger; compute the deflated Sharpe ratio from the ledger's trial variance and PBO over the variant
matrix; promote the selected spec to CANDIDATE; show the OFFICIAL gate refusing it until an Implement
run and its reconciliation exist; then run them and promote it. Every record is hash-chained.

    python examples/family6_governance.py
"""

from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_fi.backtest import BacktestSettings, FixedIncomeBacktest
from nautilus_fi.backtest.spec_strategy import SpecStrategy, SpecStrategyConfig
from nautilus_fi.construct import ConstructionSpec, EqualWeight
from nautilus_fi.domain import Conventions, MissingReturnPolicy, Timing
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec, UniverseRule
from nautilus_fi.execution import NET_SPREAD
from nautilus_fi.governance import GateFailure, RecordLevel, RecordRegistry, TrialLedger
from nautilus_fi.paper import PaperEngine, PaperSettings
from nautilus_fi.paper.reconcile import paper_data_from_environment, reconcile
from nautilus_fi.research import Hypothesis
from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market


def main() -> None:
    root = Path("output/family6")
    shutil.rmtree(root, ignore_errors=True)
    ledger, registry = TrialLedger(root / "ledger"), RecordRegistry(root / "records", min_years=0.5)     # a nine-month window: the track-record threshold (default one year) is the team's to set
    hyp = "F6-WIDE"
    ledger.register(Hypothesis(hyp, "wide spreads compensate beyond their risk", "IG corporates", "BME 16:45", "OAS rank", "none", 50e6, "12% names", "1m", "NET_SPREAD", "sharpe < 0.3", ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-09-29"), ("2030-01-01", "2030-12-31")))
    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 9, 29), n_treasuries=3, n_bills=0, n_ig=24, n_hy=0, n_frn=0, n_mbs=0, n_linkers=0, n_frn_arrears=0, n_loans=0, n_hybrids=0, n_munis=0, n_em_sov=0, n_eur_corp=0, n_cp=0, clo_deal=False, abs_deal=False, cmbs_deal=False, default_one_hy=False)
    env = generate_synthetic_market(cfg).build_environment()
    data = paper_data_from_environment(env)
    conv = Conventions(timing=Timing(1), missing_returns=MissingReturnPolicy.FAIL)
    make = lambda n: ResearchSpec(f"wide_top{n}", FieldSignal("oas_bp"), ConstructionSpec(EqualWeight(top=n, invested=0.95)), conv, "BME", UniverseRule(field_rules=[("oasd", ">", 1.0)]), hyp, {"top": n})
    results = {n: PaperEngine(make(n), data, PaperSettings(record_daily_weights=True, initial_nav=50e6)).run(ledger=ledger) for n in (4, 6, 8, 10, 12)}
    best_n = max(results, key=lambda n: results[n].stats()["sharpe"])
    best, spec = results[best_n], make(best_n)
    dsr = ledger.deflated_sharpe(hyp, best.returns.iloc[1:])
    print(f"{ledger.n_trials(hyp)} trials; selected top {best_n} (Sharpe {best.stats()['sharpe']:.2f}); DSR {dsr['dsr']:.2f} with trial variance {dsr['var_trial_sr_source']}; PBO {ledger.pbo(hyp, n_splits=4)['pbo']:.2f}")
    cand = registry.submit(RecordLevel.CANDIDATE, best, spec, ledger)
    print(f"CANDIDATE {cand.record_id}: {sum(v['ok'] for v in cand.checks.values())}/{len(cand.checks)} checks pass")
    try:
        registry.submit(RecordLevel.OFFICIAL, best, spec, ledger)
    except GateFailure as e:
        print(f"OFFICIAL refused: {sorted(e.failed)}")
    impl = FixedIncomeBacktest(env, SpecStrategy(spec, SpecStrategyConfig(rebalance="BME", universe_from_benchmark=False)), NET_SPREAD, settings=BacktestSettings(starting_cash=50e6, conventions=conv, hypothesis=hyp)).run()
    ledger.record_result(impl, spec, stage="IMPLEMENT", hypothesis_id=hyp)
    rec = reconcile(best, impl, data.panel)
    official = registry.submit(RecordLevel.OFFICIAL, best, spec, ledger, implement=impl, reconciliation=rec.summary, data_hashes={d.name: d.content_hash for d in impl.manifest.data_versions}, point_in_time=True)     # a synthetic market is generated once, never revised
    print(f"OFFICIAL {official.record_id}: implemented {official.implement['total_return']:+.2%} vs paper {official.paper['total_return']:+.2%}; chain verified {registry.verify_chain()}; record at {root / 'records' / (official.record_id + '.md')}")


if __name__ == "__main__":
    main()
