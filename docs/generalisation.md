# One platform for the six fixed income backtesting families

The review of nautilus-fi against seventeen fixed income backtesting frameworks found a strong cash-bond
implementation simulator (ledger, settlement, entitlement, funding, marks kept apart from fills) and
gaps in everything the research families need: index and return-series assets, vendor analytics,
signal panels, benchmarks, optimisation, derivatives, governance, live outputs, real data and scale.
This document describes how the platform closes those gaps and where each piece lives.

## The shape: one spec, four stages

```
                 ResearchSpec  (hashed into manifests and the trial ledger)
   decider · ConstructionSpec (method + constraint grammar + overlay) · UniverseRule · calendar · Conventions
        │                  │                        │                          │
   ┌────▼─────┐      ┌─────▼──────┐         ┌───────▼────────┐         ┌───────▼───────┐
   │ Evaluate │      │   Paper    │         │   Implement    │         │    Propose    │
   │ panels   │      │ panel      │◄───────►│ Nautilus +     │         │ trade list,   │
   │ IC, Qn   │      │ engine     │reconcile│ ledger         │         │ CRD/QOE, PORT │
   └────┬─────┘      └─────┬──────┘         └───────┬────────┘         └───────┬───────┘
        └──────────────────┴──────────────┬─────────┴──────────────────────────┘
                 governance: trial ledger · DSR / PBO · walk-forward · official records
   data plane: bitemporal field store + point-in-time store ← loaders (SQL / Snowflake / FIAtlas / files)
```

* **Evaluate** (`paper.evaluate`): cross-sectional signal research on panels.
* **Paper** (`paper.engine`): a vectorised portfolio simulation on return panels. It needs no contract
  terms, and 9,000 bonds × 10 years runs in seconds to minutes.
* **Implement** (`backtest`, `execution`, `accounting`, `funding`, `overlays`): the original event engine,
  generalised. It adds generic fields, vendor analytics, benchmarks, terms-free bonds, a margin venue,
  repo-financed leverage and streamed replay.
* **Propose** (`propose`): the same construction spec rendered as executable orders.

A strategy that should run everywhere is a `Decider` (`domain.spec`): `decide(view) -> Target`. The
`DecisionView` is the point-in-time cross-section:

* the Paper engine builds it from panels, with availability lags applied;
* the event engine builds it from the restricted `DecisionContext` (`backtest.spec_strategy.ContextView`).

The `Target` carries one of three things: scores for the construction method, final weights, or overlay
notionals. `paper.reconcile` decomposes the return gap between a spec's Paper and Implement runs into:

* holdings gap: delay, partial fills, expiries, rounding;
* cash;
* modelled versus actual execution costs;
* a residual.

## How each gap closed

| Assessment row | Now | Where |
| --- | --- | --- |
| Ledger, settlement, entitlement, funding, marks vs fills | Kept. Lifecycle work now runs only for held, traded or owed bonds; fills are processed incrementally; expiration ids are deterministic so event-chain hashes reproduce | `accounting.ledger` |
| Manifests and reproducibility | Kept. Manifests now also record conventions and spec; streamed and one-shot runs hash identically | `core.manifest`, `backtest.builder` |
| Walk-forward, purging, DSR, PBO, bootstrap run automatically | Every run can record itself in the trial ledger. DSR uses the ledger's trial count and cross-trial Sharpe variance; PBO uses the variant matrix. Walk-forward purges and embargoes in trading days and records every validation and out-of-sample trial | `governance` |
| Vendor OAS / OASD / DTS / KRD / returns as inputs | `with_analytics(df, source=VENDOR \| VENDOR_ELSE_COMPUTED \| COMPARE \| COMPUTED)`; COMPARE records vendor-versus-computed differences. Vendor returns and any other per-bond series go in as fields | `backtest.environment`, `domain.fields` |
| Terms-free mode | Paper needs only return panels. In Implement, a `terms_source="VENDOR"` bond uses vendor prices, vendor accrued, vendor analytics and supplied cashflows | `paper`, `instruments.bond`, `accounting.ledger` |
| Index or return series as a tradable asset | Any return stream is an asset in Paper; in Implement, an index total-return swap is a financed `PerpetualContract` on the margin venue | `paper.engine`, `overlays` |
| Daily index constituents and weights | `with_benchmark_constituents`: daily weights (drifting or static) are published as `benchmark.weight`, feed `universe_from_benchmark`, and give index-removal forced sales | `benchmarks`, `backtest.actor`, `backtest.strategy` |
| Benchmark object (custom, capped, sector subset, drifting or not) | `ConstituentBenchmark` / `PublishedIndex`, `CustomBenchmark` (filters, issuer and name caps, reweighting), `CompositeBenchmark`, `ReturnSeriesBenchmark` | `benchmarks` |
| Arbitrary signal panel | Any field reaches the context once per day by reference (`_publish_row`); `DecisionContext.field / field_slice / panel` read it | `core.context`, `data.panel` |
| Optimiser (LP / MILP / QP, TE, CVaR) | HiGHS via highspy: alpha, costs, L1 bucket tracking, factor-model tracking error (QP), CVaR (LP), soft constraints, an `lp_round` strategy for large mixed-integer problems, and relaxation when infeasible | `construct.optimiser` |
| Benchmark-relative constraints | One constraint grammar, enforced by the optimiser and post-checked for every method. In Implement the portfolio constructor now checks new buys, scales trades to turnover budgets, evaluates benchmark bands and active exposures, and logs every pre and post violation into the result | `construct.grammar`, `portfolio` |
| Binning, quantile, pair, DV01 / DTS sizing | `Quantile` (within groups, long-short), `RankWeights`, `BenchmarkPlusTilts`, `PairsSizing`, `HedgeOverlay` | `construct.methods` |
| Futures, CDS / CDX, FX forwards, IRS tradable | Margin venue: `FinancedContract` (TRS, CDX, carry instruments) and `DatedContract` (futures, FX forwards). Collateral is booked in the ledger, and it is prefunded or swept between venues by rule | `overlays`, `accounting.margin` |
| Shorts, leverage, overlays outside the capital budget | Paper: native. Implement: shorts as synthetic bond TRS (borrow fee charged), leverage financed by repo (`FundingPolicy(repo_for_leverage=True)`), overlays on the margin venue | `paper.engine`, `overlays`, `funding.policy` |
| Rebalance calendars | `BME`, `BMS`, `W-WED`, quarter starts and ends, every N days, index calendars, explicit dates, and `Triggered` (turnover or signal change), all in both engines | `domain.calendars` |
| Excess return vs Treasuries | A named `ExcessReturnMethod` (VENDOR / COMPOUNDED / GEOMETRIC / DAILY_SUM). Paper reports vendor-weighted and Treasury-leg excess returns; `FixedIncomeResult.excess_returns` uses vendor bond excess returns or a duration-matched leg | `domain.conventions`, `paper.engine`, `backtest.result` |
| Scale (9k bonds × 10y daily, monthly runs) | Paper: generate 22 s, evaluate 1 s, quintile run 4.5 s, monthly-optimised index tracking 20 s; 5.8 GB peak. Implement: 0.046 ms per bond-day build with vendor analytics (from 0.58), 0.12 ms per bond-day run (from 0.20), streamed in windows | `nautilus-fi-probes/paper_scale.py`, `implement_scale.py` |
| Live proposals (QOE / CRD), PORT / PREP | `build_proposal` / `propose_from_spec`: bid/ask dirty pricing, settlement dates, pre-trade checks, `ColumnMap` writers, position and cash reconciliation | `propose` |
| Real data loaders (Snowflake / FIAtlas) | `SQLLoader` on any DB-API driver (`:start`/`:end` parameters, `::` casts left alone), `snowflake_loader` (environment-variable credentials, key-pair or SSO), a FIAtlas catalogue template, and file loaders. Every pull goes to the point-in-time and field stores | `data.loaders` |

## Conventions: named, recorded, per profile

Every setting that changes a number is a field of `domain.conventions.Conventions`. It is recorded in
manifests, the trial ledger and official records, and printed with every statistic.

| Convention | Options | Canonical | Legacy profiles |
| --- | --- | --- | --- |
| Timing (`Timing.return_lag_days`) | trading days from the information date to the end of the first earned period | 2 (trade T+1, earn T+2) | `fi_lib_research`, `EUR_IG` = 2; `MRI_v1`, `erin_beta_plus` = 1 |
| Excess return | VENDOR, COMPOUNDED, GEOMETRIC, DAILY_SUM | COMPOUNDED, vendor first | `MRI_v1` = DAILY_SUM |
| Benchmark weights | DRIFT, STATIC | DRIFT | `fi_lib_backtesting` = DRIFT; `erin_beta_plus` = STATIC |
| Turnover | ONE_WAY_BUYS, HALF_SUM, TWO_WAY (all three always reported) | HALF_SUM | `erin_beta_plus` = ONE_WAY_BUYS; `fi_lib_backtesting` = TWO_WAY |
| Annualisation | TRADING_252, CALENDAR_365, CALENDAR_365_25 | CALENDAR_365 (official records require it) | `paperport` = TRADING_252 |
| Missing returns | ZERO, DROP_DATE, REFILL (with a bad-date detector), FAIL | FAIL | most = ZERO; `erin_beta_plus` = REFILL |
| Rebalance | any calendar string or object | BME | |
| Cash | ZERO, OVERNIGHT, BILL_INDEX | OVERNIGHT | |

Profiles marked `confirmed=False` encode only what the review could read. In particular, the
excess-return formulas of fi-lib-research and the paperport turnover definition were unreadable in the
photos. Confirm them against golden outputs; the CANDIDATE gate refuses unconfirmed conventions.

## Using it

```python
from nautilus_fi.benchmarks import ConstituentBenchmark
from nautilus_fi.construct import ConstructionSpec, Objective, Optimise, TrackingTerm
from nautilus_fi.domain import Conventions
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
from nautilus_fi.governance import TrialLedger
from nautilus_fi.paper import PaperData, PaperEngine

spec = ResearchSpec("index_plus", FieldSignal("signal.value", zscore_by="sector"),
                    ConstructionSpec(Optimise(Objective(alpha=3e-4, tracking=TrackingTerm(("oasd", "dts"), ("sector",)))),
                                     [{"kind": "active_group", "attribute": "sector", "min": -0.02, "max": 0.02},
                                      {"kind": "active_exposure", "attribute": "oasd", "min": -0.1, "max": 0.1},
                                      {"kind": "turnover", "max": 0.1}]),
                    Conventions(), hypothesis_id="H1")
paper = PaperEngine(spec, PaperData(panel, attributes, cash, ConstituentBenchmark("IDX", constituents))).run(ledger=TrialLedger("ledger/"))
# the same spec through orders, fills, settlement and the ledger:
from nautilus_fi.backtest import FixedIncomeBacktest, BacktestSettings
from nautilus_fi.backtest.spec_strategy import SpecStrategy
implemented = FixedIncomeBacktest(env, SpecStrategy(spec), settings=BacktestSettings(conventions=spec.conventions)).run()
```

There is one example per family in `examples/` (`family1_index_timing.py` … `family6_governance.py`).

## What is still approximated or yours to confirm

* Execution parameters (RFQ panel, liquidity gate, cost tables) are scenario defaults until calibrated
  on your own TRACE / RFQ history, as before.
* Margin-venue carry is settled daily as Nautilus funding on a supplied price series. CDX defaults must
  be in the supplied prices; futures rolls are the strategy's decision; FX forwards are cash-settled at
  the last price.
* Order and position file layouts are column maps. Confirm them against the CRD, QOE, PORT and PREP
  import specifications.
* Snowflake / FIAtlas table and column names in `fiatlas_catalog.example.json` are templates. Point them
  at the deployed views.
* Legacy parity: the profiles reproduce the conventions; matching each framework's numbers needs its
  golden outputs.
* Implement at full index scale: build and run are fast with vendor analytics. Per-bond-day objects
  still cost about 1–2 KB, so very large universes need memory sized accordingly, or the Paper engine.
