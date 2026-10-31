"""A PM's "what if" loop, the way a research agent runs it.

The agent has a data-access tool (here a function over a synthetic warehouse) and the kernel. It
wraps the tool as a data source once, builds the PM's current strategy as Python objects, and then
answers "if I change my strategy this way, how would it perform?" with ``compare(...).explain()``.
It sweeps the parameter the PM is unsure about, reads the selection statistics the session keeps
for every trial, runs the chosen variant through the event engine, and submits it through the
record gates.

    python examples/agent_workflow.py [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nautilus_fi as nfi
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


def warehouse(quick: bool):
    """Stand-in for the company data platform: long tables per field, reference data, series."""
    d = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=600 if quick else 2500, start="2015-01-02", end="2017-12-29" if quick else "2019-12-31", stress_start="2016-01-15"))
    long = d.panel.to_long()
    return d, long


def make_source(quick: bool):
    d, long = warehouse(quick)

    def data_tool(field: str, start, end, ids):
        """What the agent's data-access tool does: one query per field and window."""
        rows = long[(long["date"] >= start) & (long["date"] <= end)] if start is not None else long
        return rows[["date", "instrument_id", field]].dropna()

    tool = nfi.CallableSource(data_tool, [nfi.field_info(f) for f in d.panel.fields], attributes=d.attributes,
                              series={"cash": d.cash, "index.tr": d.index["tr"]}, constituents={"IG": d.membership},
                              name="warehouse", version="warehouse:2026-10-30")
    return nfi.CachedSource(tool), d                                # years of data load once; later windows are slices


def event_environment(d, ids, start, end):
    """Contract terms, evaluated prices, curves and ratings for the event engine (the warehouse's
    reference and pricing tables; here generated from the synthetic universe)."""
    import pandas as pd

    from nautilus_fi.data.loaders import builder_from_frames
    from nautilus_fi.synthetic.panels import to_market_inputs

    return builder_from_frames(to_market_inputs(d, list(ids)), pd.Timestamp(start).date(), pd.Timestamp(end).date()).build()


def main(quick: bool = True) -> None:
    src, d = make_source(quick)
    sess = nfi.session(source=src, name="credit-value-what-ifs")
    print(src.catalog().describe(entity="asset").splitlines()[0])
    print("spread fields:", [f.name for f in src.catalog().search("spread")])

    # ---- the PM's current strategy, as objects
    base = nfi.Strategy(
        "ig_value",
        universe=nfi.Index("IG") & (nfi.Field("oasd") > 1),
        signal=nfi.Field("signal.value").zscore(by="sector"),
        portfolio=nfi.Optimise(1e-3 * nfi.Alpha() - 2e-3 * nfi.TrackingError(by=["sector"], fields=["oasd", "dts"]) - nfi.Costs(),
                               nfi.ActiveWeight("issuer") <= 0.01, nfi.ActiveWeight("sector").within(0.03),
                               nfi.ActiveExposure("oasd").within(0.15), nfi.Turnover() <= 0.15),
        benchmark=nfi.Index("IG"),
        execution=nfi.NextClose(1, cost=nfi.HalfSpread() + nfi.SqrtImpact(8)),
        missing=nfi.ZeroFill(),
        capital=500e6)
    print("\n" + base.describe())
    issues = base.validate(src)
    print("validation:", issues or "clean")

    # ---- the PM's questions, one object change each
    questions = {
        "tighter turnover (5%)": base.with_(portfolio=base.portfolio.replace(nfi.Turnover() <= 0.05), name="turnover_5"),
        "duration-neutral signal": base.with_(signal=base.signal.neutralise("oasd", by="sector"), name="dur_neutral"),
        "no financials in the benchmark": base.with_(benchmark=nfi.Index("IG").exclude(sector="FIN"), universe=nfi.Index("IG").exclude(sector="FIN") & (nfi.Field("oasd") > 1), name="ex_fin"),
        "costs 50% higher": base.with_(execution=nfi.NextClose(1, cost=(nfi.HalfSpread() + nfi.SqrtImpact(8)) * 1.5), name="costs_x1.5"),
        "slower execution (2 days)": base.set("execution.lag", 2).with_(name="lag_2"),
    }
    for question, variant in questions.items():
        cmp = sess.compare(base, variant)
        e = cmp.explain()
        print(f"\n=== {question}")
        print(e.text())

    # ---- the parameter the PM is unsure about: how hard to track
    sweep = sess.sweep(lambda k: base.with_(portfolio=base.portfolio.with_objective(1e-3 * nfi.Alpha() - k * nfi.TrackingError(by=["sector"], fields=["oasd", "dts"]) - nfi.Costs()), name=f"te_penalty_{k:g}"), {"k": [5e-4, 2e-3, 8e-3]}, metric="information_ratio")
    print("\n" + sweep.table()[["information_ratio", "tracking_error", "turnover"]].to_string())
    sel = sweep.selection()
    print(f"best {sel['best']}: IR {sel['value']:.2f}; deflated Sharpe {sel['deflated_sharpe']['dsr']:.2f} over {sel['deflated_sharpe']['ledger_trials']} session trials; PBO {sel['pbo']['pbo']:.2f}")

    # ---- does the signal itself rank bonds?
    rep = nfi.evaluate(base.signal, universe=base.universe, horizons=(5, 21, 63))
    print("\n" + rep.summary()[["mean_ic", "ic_tstat_nw", "spread_annualised"]].round(3).to_string())

    # ---- the chosen variant through the event engine (a small sleeve keeps the example quick)
    sleeve = questions["duration-neutral signal"].with_(universe=nfi.Index("IG") & (nfi.Field("oasd") > 1) & nfi.Field("amt_out").top(40), portfolio=nfi.TopN(15), capital=50e6, name="dur_neutral_sleeve")
    window = ("2017-01-03", "2017-06-30")
    paper = sess.run(sleeve, window=window)
    held = sorted(set(paper.weights["instrument_id"]))
    implemented = sess.run(sleeve, window=window, fidelity="implement", env=event_environment(d, held, *window))
    rec = nfi.reconcile(paper, implemented, source=src)
    print(f"\npaper {paper.headline()}\nimplement {implemented.headline()}")
    print("paper -> implement:", {k: round(v, 4) for k, v in rec.summary.items() if k in ("paper_gross_total", "implemented_total", "holdings_gap", "execution", "residual")})
    record = sess.promote(paper, "EXPLORATORY")
    print(f"\n{sess.summary()}; record {record.record_id} ({record.level})")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv or len(sys.argv) == 1)
