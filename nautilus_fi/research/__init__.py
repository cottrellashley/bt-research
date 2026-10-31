"""Research design and statistical integrity (blueprint sections 19 and 21)."""

from nautilus_fi.research.experiment import ExperimentRecord, ExperimentRegistry, HoldoutViolation, Hypothesis
from nautilus_fi.research.statistics import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    min_track_record_length,
    newey_west_mean_se,
    newey_west_sharpe_se,
    newey_west_tstat,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from nautilus_fi.research.stress import (
    STANDARD_SCENARIOS,
    StressDimension,
    StressKind,
    StressScenario,
    joint_scenarios,
    reverse_stress,
    uncertainty_envelope,
)
from nautilus_fi.research.walkforward import Split, purged_train_indices, walk_forward_splits

__all__ = [
    "STANDARD_SCENARIOS",
    "ExperimentRecord",
    "ExperimentRegistry",
    "HoldoutViolation",
    "Hypothesis",
    "Split",
    "StressDimension",
    "StressKind",
    "StressScenario",
    "block_bootstrap_ci",
    "deflated_sharpe_ratio",
    "joint_scenarios",
    "min_track_record_length",
    "newey_west_mean_se",
    "newey_west_sharpe_se",
    "newey_west_tstat",
    "probability_of_backtest_overfitting",
    "purged_train_indices",
    "reverse_stress",
    "sharpe_ratio",
    "uncertainty_envelope",
    "walk_forward_splits",
]
