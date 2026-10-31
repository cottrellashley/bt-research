"""Family 2 - cross-sectional signal evaluation.

A value signal on bond-level forward excess returns: rank IC by horizon and its decay, quintile
returns, Q5-Q1 with Newey-West t-statistics and block-bootstrap intervals, the same after
neutralising sector and spread duration, and the evaluation recorded against a pre-registered
hypothesis in the trial ledger.

    python examples/family2_signal_evaluation.py [--full]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nautilus_fi.domain import Conventions
from nautilus_fi.governance import TrialLedger
from nautilus_fi.paper.evaluate import EvaluationSettings, evaluate_signal
from nautilus_fi.research import Hypothesis
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


def main(full: bool = False) -> None:
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=9000 if full else 1500, start="2015-01-02", end="2024-12-31" if full else "2019-12-31", signal_noise=3.0))
    ledger = TrialLedger(Path("output/family2_ledger"))
    hyp = "F2-VALUE"
    if ledger.hypothesis(hyp) is None:
        ledger.register(Hypothesis(hyp, "idiosyncratic spread dislocations mean-revert", "IG index members", "BME", "signal.value", "index", 0.0, "none", "1m", "none", "IC < 0.02 or t < 3", ("2015-01-01", "2018-12-31"), ("2019-01-01", "2019-12-31"), ("2030-01-01", "2030-12-31")))
    rep = evaluate_signal(s.panel, "signal.value", Conventions(), EvaluationSettings(horizons=(1, 5, 21, 63), neutralise_groups=("sector",), neutralise_exposures=("oasd",)), s.attributes, ledger=ledger, hypothesis_id=hyp)
    print(rep.summary()[["n_dates", "mean_ic", "ic_tstat_nw", "hit_rate", "spread_annualised", "spread_tstat_nw", "spread_ci_low", "spread_ci_high", "monotonicity", "mean_ic_neutral"]].round(4).to_string())
    print("quintile mean forward excess returns:\n", rep.quantile_means().round(5).to_string())
    print(f"top-quintile turnover per rebalance: {rep.top_turnover.mean():.1%}; evaluations recorded: {len(ledger.trials(hyp, 'EVALUATE'))}")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
