"""Family 1 - timing an index excess-return series.

The index is a tradable return stream (no terms): exposure from 0.5x to 2x driven by trailing
excess-return momentum, an explicit lag in trading days, per-change costs, and walk-forward selection
of the look-back with purging and embargo. The same decider runs under the canonical conventions
(trade T+1, earn T+2) and the MRI v1 profile (trade on the close the signal sees) to show what the
convention is worth. Then the chosen spec runs in the event engine as a financed total-return swap
on a margin venue.

    python examples/family1_index_timing.py [--quick]
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from nautilus_fi.data.panel import Panel
from nautilus_fi.domain import Conventions, MissingReturnPolicy
from nautilus_fi.domain.spec import Decider, ResearchSpec, Target
from nautilus_fi.governance import TrialLedger, walk_forward
from nautilus_fi.paper import PaperData, PaperEngine, PaperSettings
from nautilus_fi.research import Hypothesis, walk_forward_splits
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


class Momentum(Decider):
    """Exposure to the index: 2x when trailing excess return is positive, 0.5x otherwise."""

    def __init__(self, lookback: int = 63, high: float = 2.0, low: float = 0.5) -> None:
        self.lookback, self.high, self.low = lookback, high, low

    def decide(self, view):
        h = view.history("xr", self.lookback)
        if len(h) < self.lookback:
            return Target(weights=pd.Series({"IDX": 1.0}))
        trailing = float((1 + h["IDX"].fillna(0)).prod() - 1)
        return Target(weights=pd.Series({"IDX": self.high if trailing > 0 else self.low}))


def index_panel(quick: bool) -> tuple[Panel, pd.Series]:
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=400 if quick else 1500, start="2015-01-02", end="2018-12-31" if quick else "2022-12-30", stress_start="2016-01-15"))
    idx = s.index
    p = Panel(s.dates, pd.Index(["IDX"]), {"tr": idx[["tr"]].to_numpy(), "xr": idx[["xr"]].to_numpy(), "half_spread_bp": np.full((len(idx), 1), 1.0), "alive": np.ones((len(idx), 1))})
    return p, s.cash


def main(quick: bool = True) -> None:
    panel, cash = index_panel(quick)
    data = PaperData(panel, cash=cash, excess_field="xr", treasury_field=None)
    for name in ("canonical", "MRI_v1"):
        conv = Conventions.profile(name).with_(missing_returns=MissingReturnPolicy.ZERO, rebalance="D")
        res = PaperEngine(ResearchSpec(f"momentum[{name}]", Momentum(), conventions=conv), data, PaperSettings(financing_spread_bp=25.0)).run()
        s = res.stats()
        print(f"{name:10s} lag {conv.timing.return_lag_days}d  annual {s['annual_return']:+.2%}  vol {s['annual_vol']:.2%}  Sharpe {s['sharpe']:.2f}  turnover {s['turnover_annual']:.1f}x  costs {s['cost_drag_annual']:.2%}")
    ledger = TrialLedger(Path("output/family1_ledger"))
    hyp = "F1-MOMENTUM"
    if ledger.hypothesis(hyp) is None:
        ledger.register(Hypothesis(hyp, "index excess returns trend", "IG index", "daily close", "trailing excess return", "cash", 100e6, "0.5x-2x", "1d", "1bp per change", "Sharpe < 0.3", ("2015-01-01", "2016-12-31"), ("2017-01-01", "2017-12-31"), ("2030-01-01", "2030-12-31")))
    conv = Conventions().with_(missing_returns=MissingReturnPolicy.ZERO, rebalance="D")
    make = lambda p: ResearchSpec(f"momentum_{p['lookback']}", Momentum(p["lookback"]), conventions=conv, params=p)
    end = date(2018, 12, 31) if quick else date(2022, 12, 30)
    wf = walk_forward(make, [{"lookback": 21}, {"lookback": 63}, {"lookback": 126}], data, walk_forward_splits(date(2015, 1, 1), end, 12, 6, 6, 6), label_horizon_days=126, embargo_days=5, ledger=ledger, hypothesis_id=hyp)
    print(wf.folds[["split", "selected", "validation_score", "test_score", "purged_val"]].to_string(index=False))
    oos = wf.stats(conv)
    print(f"out of sample: annual {oos['annual_return']:+.2%}, Sharpe {oos['sharpe']:.2f}; trials in ledger {ledger.n_trials(hyp, ('VALIDATION', 'OOS'))}; DSR {ledger.deflated_sharpe(hyp, wf.oos_returns, 252, ('VALIDATION',))['dsr']:.2f}")
    implement_as_trs(quick)


def implement_as_trs(quick: bool) -> None:
    """The index exposure in the event engine: a total-return swap (long earns the index, pays cash +
    50bp) on a margin venue next to a cash sleeve of Treasuries."""
    from nautilus_fi.accounting.margin import MarginSettings
    from nautilus_fi.backtest import (
        BacktestSettings,
        FixedIncomeBacktest,
        FixedIncomeStrategy,
        FixedIncomeStrategyConfig,
    )
    from nautilus_fi.execution import NET_SPREAD
    from nautilus_fi.overlays import OverlayBook, index_trs
    from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market

    mkt = generate_synthetic_market(SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 6, 30), n_treasuries=3, n_bills=1, n_ig=0, n_hy=0, n_frn=0, n_mbs=0, n_linkers=0, n_frn_arrears=0, n_loans=0, n_hybrids=0, n_munis=0, n_em_sov=0, n_eur_corp=0, n_cp=0, clo_deal=False, abs_deal=False, cmbs_deal=False, default_one_hy=False))
    env = mkt.build_environment()
    dates = pd.to_datetime(env.dates)
    level = pd.Series(100 * np.cumprod(1 + np.random.default_rng(3).normal(0.0002, 0.0025, len(dates))), index=dates)
    book = OverlayBook()
    book.add(index_trs("IG-XR-TRS", level, 0.05, 50.0, exposures=pd.DataFrame({"oasd": 6.8}, index=dates)))

    class Timed(FixedIncomeStrategy):
        def on_decision(self, d):
            hist = [self.overlay_price("IG-XR-TRS") or 100.0]
            self._px = getattr(self, "_px", []) + hist
            px = self._px
            exposure = 2.0 if len(px) > 21 and px[-1] > px[-21] else 0.5
            govts = [b.symbol for b in self.universe() if b.asset_type.value in ("SOVEREIGN", "BILL")]
            self.rebalance_to_weights({s: 0.85 / len(govts) for s in govts})
            self.rebalance_overlays({"IG-XR-TRS": exposure})

    res = FixedIncomeBacktest(env, Timed(FixedIncomeStrategyConfig(rebalance="W")), NET_SPREAD, settings=BacktestSettings(starting_cash=100e6, margin=MarginSettings(collateral=5e6, book=book))).run()
    nav = res.nav
    print(f"event engine: {res.status}/{res.outcome}, NAV {nav['nav'].iloc[-1]:,.0f}, max overlay notional {nav['overlay_gross_notional'].max():,.0f}, margin equity {nav['margin_equity'].iloc[-1]:,.0f}, invariants {bool(res.invariants()['passed'].all())}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv or len(sys.argv) == 1)
