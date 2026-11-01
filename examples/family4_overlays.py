"""Family 4 - derivatives, overlays and relative value.

Panel engine: a credit portfolio with a Treasury-future duration hedge sized by the construction
spec (DV01 / OAD neutral), a CDX overlay, and DTS-neutral long-short pairs. Event engine: the same
kind of book on a margin venue - an index TRS, a Treasury future held to expiry, a synthetic short of
a cash bond (financing received, coupon paid, borrow fee charged) - with collateral moving between
the cash and margin accounts and one ledger of record.

    python examples/family4_overlays.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from nautilus_fi.benchmarks import ConstituentBenchmark
from nautilus_fi.construct import ConstructionSpec, EqualWeight, HedgeOverlay, PairsSizing
from nautilus_fi.domain import Conventions, MissingReturnPolicy
from nautilus_fi.domain.spec import Decider, FieldSignal, ResearchSpec, Target
from nautilus_fi.paper import PaperData, PaperEngine
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


class PairsAndCDX(Decider):
    """Long the cheapest and short the richest bond in each sector, DTS-neutral, plus a CDX overlay."""

    def decide(self, view):
        sig = view.field("signal.value").reindex(view.universe).dropna()
        sect = view.attributes.reindex(sig.index)["sector"]
        pairs = [(g.idxmax(), g.idxmin()) for _, g in sig.groupby(sect) if len(g) > 1]
        return Target(scores=pd.Series({p: 1.0 for pair in pairs for p in pair}), overlays=pd.Series({"CDX_IG": 0.2}), note=str(pairs))


def paper_overlays() -> None:
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=600, start="2015-01-02", end="2018-12-31"))
    data = PaperData(s.panel, s.attributes, s.cash, ConstituentBenchmark("IDX", s.membership), s.overlays)
    conv = Conventions(missing_returns=MissingReturnPolicy.ZERO)
    plain = PaperEngine(ResearchSpec("credit", FieldSignal("signal.value"), ConstructionSpec(EqualWeight(top=40)), conv), data).run()
    hedged = PaperEngine(ResearchSpec("credit_hedged", FieldSignal("signal.value"), ConstructionSpec(EqualWeight(top=40), overlay=HedgeOverlay(("TY",), ("oad",), {"oad": 0.0})), conv), data).run()
    rates = s.treasury["r_10"]
    for r in (plain, hedged):
        st = r.stats()
        print(f"{r.name:14s} annual {st['annual_return']:+.2%} vol {st['annual_vol']:.2%} corr(10y) {r.returns.corr(rates.reindex(r.returns.index)):+.2f} OAD {r.exposures['oad'].iloc[-1]:.2f}y TY notional {r.overlays.loc[r.overlays['instrument_id'] == 'TY', 'notional'].iloc[-1] if not r.overlays.empty else 0:+.2f}")

    class PairsOnly(PairsAndCDX):
        def decide(self, view):
            t = super().decide(view)
            sig = view.field("signal.value").reindex(view.universe).dropna()
            sect = view.attributes.reindex(sig.index)["sector"]
            pairs = [(g.idxmax(), g.idxmin()) for _, g in sig.groupby(sect) if len(g) > 1]
            dts = view.field("dts")
            w = pd.Series(0.0, index=view.universe)
            for lo, sh in pairs:
                if np.isfinite(dts.get(lo, np.nan)) and np.isfinite(dts.get(sh, np.nan)) and dts[sh] > 0:
                    w[lo] += 0.05
                    w[sh] -= 0.05 * dts[lo] / dts[sh]
            return Target(weights=w[w != 0], overlays=t.overlays)

    rv = PaperEngine(ResearchSpec("pairs_cdx", PairsOnly(), conventions=conv), data).run()
    st = rv.stats()
    print(f"pairs + CDX    annual {st['annual_return']:+.2%} vol {st['annual_vol']:.2%} gross {rv.daily['gross_exposure'].mean():.2f} net {rv.daily['net_exposure'].mean():+.2f} DTS {rv.exposures['dts'].abs().mean():.1f}")
    sized = ConstructionSpec(PairsSizing([("B00001", "B00002")], "dts", 0.05), long_only=False)
    print("PairsSizing (DTS-neutral) hedge ratio method available:", sized.method.name)


def event_engine_overlays() -> None:
    from nautilus_fi.accounting.margin import MarginSettings
    from nautilus_fi.backtest import (
        BacktestSettings,
        FixedIncomeBacktest,
        FixedIncomeStrategy,
        FixedIncomeStrategyConfig,
    )
    from nautilus_fi.execution import NET_SPREAD
    from nautilus_fi.overlays import OverlayBook, index_trs, treasury_future
    from nautilus_fi.synthetic import SyntheticMarketConfig, generate_synthetic_market

    cfg = SyntheticMarketConfig(start=date(2023, 1, 3), end=date(2023, 5, 31), n_treasuries=3, n_bills=0, n_ig=10, n_hy=0, n_frn=0, n_mbs=0, n_linkers=0, n_frn_arrears=0, n_loans=0, n_hybrids=0, n_munis=0, n_em_sov=0, n_eur_corp=0, n_cp=0, clo_deal=False, abs_deal=False, cmbs_deal=False, default_one_hy=False)
    env = generate_synthetic_market(cfg).build_environment()
    dates = pd.to_datetime(env.dates)
    book = OverlayBook()
    book.add(index_trs("IG-TRS", pd.Series(100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0002, 0.002, len(dates))), index=dates), 0.05, 50.0, exposures=pd.DataFrame({"oad": 7.0}, index=dates)))
    book.add(treasury_future("TYM3", pd.Series(110 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, len(dates))), index=dates), date(2023, 4, 20), oad=6.5))
    corp = [s for s, b in env.bonds.items() if b.asset_type.value == "CORPORATE"]

    class Book(FixedIncomeStrategy):
        def on_decision(self, d):
            self.rebalance_to_weights({s: 0.2 for s in corp[:4]})
            targets = {"IG-TRS": 0.5, corp[5] + "-TRS": -0.05}
            if d < date(2023, 4, 15):
                targets["TYM3"] = -0.3
            self.rebalance_overlays(targets)

    res = FixedIncomeBacktest(env, Book(FixedIncomeStrategyConfig(rebalance="BMS")), NET_SPREAD, settings=BacktestSettings(starting_cash=100e6, margin=MarginSettings(collateral=1e6, book=book, synthetic_shorts={corp[5]}))).run()
    nav = res.nav
    pos = res.positions_report
    print(f"event engine: {res.status}/{res.outcome}; NAV {nav['nav'].iloc[-1]:,.0f}; margin equity {nav['margin_equity'].iloc[-1]:,.0f} vs initial margin {nav['margin_required'].iloc[-1]:,.0f}; collateral transfers {len(res.extra['margin_transfers'])}; borrow fees {res.extra['borrow_fees']['fee'].sum() if len(res.extra['borrow_fees']) else 0:,.2f}")
    print(pos[pos["instrument_id"].astype(str).str.endswith(".MARGIN")][["instrument_id", "side", "quantity", "realized_pnl"]].to_string(index=False))
    print("invariants:", res.invariants()[["invariant", "passed"]].to_dict("records"))


if __name__ == "__main__":
    paper_overlays()
    event_engine_overlays()
