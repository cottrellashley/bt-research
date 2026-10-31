# Architecture

`nautilus_fi` is a fixed income backtesting platform built on Nautilus Trader v2. Nautilus supplies
the event loop, nanosecond clock, order lifecycle (GTD limit orders resting across days), the
matching engine hook, cash accounts, positions, message bus and reports. Everything a bond needs
that an equity does not (knowledge-time data, marks separate from fills, dated settlement, coupon
entitlement, factors, waterfalls, defaults, funding, a ledger of record, run manifests) lives in this
package and is wired into Nautilus at five points.

| Nautilus primitive | Used for | Where |
| --- | --- | --- |
| custom data (`@customdataclass`) replayed by `ts_init` | marks, liquidity, ratings, fixings, factors, actions, stress, curves delivered at their **availability** time | `data.types`, `backtest.environment.nautilus_data` |
| `MarkPriceUpdate` + `PortfolioConfig(use_mark_prices=True)` | Nautilus positions valued off the pricing waterfall, never off fill quotes | `backtest.environment` |
| `FillModel.get_orderbook_for_fill_simulation` | one synthetic liquidity book per order per window: the order-process simulator (RFQ panel / voice / portfolio trade), liquidity budget, cost decomposition and the settle-before-spend funding gate | `execution.fill_model`, `execution.process` |
| `SimulationModule.process()` returning `Money` | the ledger of record runs on every engine timestamp and pushes cash effects (coupons, paydowns, accrued, interest, FX conversions, repo) into the Nautilus account | `accounting.cashflow_module`, `accounting.ledger` |
| `InstrumentClose(CONTRACT_EXPIRED)` | maturities, calls, tenders, recoveries close positions at the redemption price | `backtest.environment` |
| `DataActor` + `Strategy` on the shared clock | the environment actor publishes data into the strategy's restricted `DecisionContext`; strategies see nothing else | `backtest.actor`, `backtest.strategy`, `core.context` |

## Layers

```
core/          typed quantities, four-timestamp TimePoint, append-only EventLog with lineage, double-entry Ledger,
               SupportMatrix / ModelRegistry / DependencyGraph, AssumptionSet, ExperimentManifest, DecisionContext
instruments/   BondSpec and terms (fixed, step, floating in-advance / compounded in arrears / index, linkers, sinkers,
               amortizers, perpetuals), day counts, schedules, calendars, curves, cashflow projection, analytics,
               dated settlement and entitlement conventions, loans, hybrids, distressed, money market, tax, funds,
               derivatives (futures, IRS, CDS, FX forwards) and FX rates / hedge programs
lifecycle/     bond lifecycle state machine with event precedence; corrections and restatement bridge
funding/       settlement queue, reservations, cash view, repo, securities borrow, margin, funding policy
structured/    collateral pools (realized remittances or projected), deal waterfall with OC/IC tests, internal
               deal engine and the external-engine adapter contract, TBA / dollar rolls
market/        market state (curves by kind, surfaces, fixings, FX) and calibration recipes with explicit status
pricing/       waterfall marks (TRADE > COMPOSITE > EVALUATED > MODEL > STALE_ROLLED), matrix pricing, stale roll,
               behaviour models (PSA, refinance S-curve, issuer call policy, default model), discounting and
               Hull-White trinomial engines, four-value marks with evidence classification, QuantLib adapter
execution/     windows, assumption ladders (GROSS / NET_SPREAD / NET_FULL), liquidity, costs, order process
               simulator, fill model, settlement helpers
accounting/    posting policy (every event type), ledger of record, attribution, returns, cashflow module
portfolio/     typed constraints, portfolio constructor (nets live orders, reserves cash), forced actions, risk
research/      experiments, walk-forward with purging, block bootstrap / Newey-West / deflated Sharpe / PBO, stress
validation/    invariants (temporal isolation, duplicate delivery, conservation, lot regrouping, pinned replay,
               unfilled orders, failed settlement, payments reconcile) and tolerances
reports/       report bundle (performance, implementation, evidence, risk, attribution, robustness, support,
               invariants, unknowns) and order / payment explainers
backtest/      MarketEnvironmentBuilder, FixedIncomeBacktest (manifest, validation, wiring, status), strategy base,
               environment actor, results, scenarios
synthetic/     synthetic market generator covering every product family; index-scale panel universe
domain/        conventions (timing, excess return, drift, turnover, annualisation, missing returns), rebalance
               calendars, field definitions, ResearchSpec / Decider / DecisionView / Target
benchmarks/    constituent, custom, composite and return-series benchmarks
construct/     cross-sections, signal transforms, constraint grammar, construction methods, HiGHS optimiser
paper/         panel engine (Paper), signal evaluation (Evaluate), Paper / Implement reconciliation
governance/    trial ledger, walk-forward with purge / embargo, gated official records
overlays/      margin-venue contracts and fill model; accounting/margin.py books them in the ledger
propose/       trade proposals, OMS / PORT writers, pre-trade validation, reconciliation
data/loaders/  SQL, Snowflake, FIAtlas catalogue and file loaders into the point-in-time stores
```

The panel engine and the event engine share the domain model: a `ResearchSpec` runs in both (the
event engine through `backtest.spec_strategy.SpecStrategy`, whose `ContextView` is built only from the
restricted context), and `paper.reconcile` decomposes the difference. See `docs/generalisation.md`.

## A trading day

```
10:00  execution window   one QuoteTick per alive bond at the PRIOR-CLOSE reference x factor; resting GTD orders are
                          matched against the book the fill model builds for each order: funding gate (settled cash
                          less committed payables, less what this window already offered), liquidity budget shared
                          across orders in the same bond, RFQ panel / voice / portfolio-trade process, cost
                          decomposition (half spread, size, regime, protocol, delay drift kept separate)
10:00+ ledger             fills -> trade postings at clean + accrued (negative accrued in ex-coupon periods),
                          settlement queue on the dated convention (US corporates T+3 -> T+2 on 5 Sep 2017 -> T+1
                          on 28 May 2024; Treasuries T+1; securitized T+3; loans T+7; EUR T+2 ...), FX conversion
                          at spot through the translation account for non-base purchases
every ts                  settlements due, coupon entitlements (the first trade date whose settlement no longer
                          carries the coupon crystallises a receivable for the start-of-day holder), daily accrual,
                          coupon / tranche payments, factor paydowns, calls / maturities / defaults / recoveries,
                          cash interest and financing, funding assessment and breach ladder
16:00  marks struck       ts_event of marks, curves, stress, ratings
16:30  marks published    ts_init: MarkPriceUpdate reaches the portfolio, data reaches the DecisionContext
16:30  ledger snapshot    NAV = assets - liabilities from the ledger; evidence shares, duration / DTS, settled and
                          unsettled cash, coupon receivables, repo, reconciliation against Nautilus cash
16:35  PortfolioSnapshot  published on the bus
16:45  decision           strategy timer; orders go through the portfolio constructor and rest to the next window
```

Orders submitted at 16:45 cannot fill on the 10:00 quotes of the same day (`order.ts_init >= now`
returns an empty book), and the execution reference is the prior close, so a signal never trades
against its own print.

## Instrument representation

Nautilus v2 rejects multiplier instruments on cash accounts. Bonds are therefore `Equity`
instruments quoted per 100 with quantity in units of 100 face. Factor instruments (MBS / ABS / CLO
tranches, sinkers, linkers) carry original face and a factor-adjusted price so notional always equals
current-face market value. Strategies work in face amounts (`order_face`, `holdings_face`,
`rebalance_to_weights`) and never see the representation. Multi-currency runs use a Nautilus
multi-currency cash account with `allow_cash_borrowing`; the ledger converts base cash at spot when
a foreign purchase settles and the funding gate counts convertible base cash.

## The ledger of record

`FixedIncomeLedger` (`accounting/ledger.py`) is a balanced double-entry ledger on
`core.ledger.Ledger` with the posting policy in `accounting/postings.py`. Each posting cites the
event that caused it (`core.events.EventLog`, hash-chained). Accounts: settled cash, unsettled
receivables / payables, coupon receivable, securities at market, accrued interest, contractual claims
(defaulted paper), collateral, financing payable, capital, realized / unrealized P&L, FX translation,
income and expense lines. Invariants checked in results: conservation (assets - liabilities =
equity), payments reconcile with events, unfilled orders change nothing, and the identity
`NAV = cash (incl. unsettled, net of funding repo) + dirty market value + entitled coupons` holds on
every snapshot (asserted in the tests).

Nautilus keeps trade-date cash and positions; the ledger reconciles the Nautilus base-currency
balance against its own trade-date cash view on every snapshot (`result.reconciliation`).

## Run manifests, status and outcome

`FixedIncomeBacktest.build()` creates an `ExperimentManifest` (hypothesis, run mode, deployability
claim, universe, dates, capital, benchmark contract, decision schedule, event precedence, reference
and terminal policies, model versions, data versions with content hashes and availability policies,
resolved assumptions with ranges, support rows for the product families in the universe, seeds),
validates it, refuses unsupported features (`BacktestSettings.allow_support`) and records the
dependency-validation outcome of the environment. `run()` records `computational_status`
(COMPLETED / ENGINE_FAILURE / INVALID_CONFIGURATION / UNSUPPORTED_FEATURES) separately from
`economic_outcome` (SOLVENT / FUNDING_DEFAULT / INSOLVENT), hashes outputs and the event chain.
`FixedIncomeResult.bundle()` renders the report bundle with headline numbers and the list of
unknowns.

## Funding

Spendable cash is settled cash less payables already committed to pending settlements less
reservations (settle before spend). The fill gate also subtracts what the current window has
already offered to other orders. A breach runs the policy's ladder: hold buys (gate), source repo
against eligible holdings (government, agency, IG credit, agency MBS; haircut and eligible share from
the policy; interest accrues to the financing payable), ask the strategy to liquidate by rule
(`FixedIncomeStrategy.on_funding_shortfall`, default most-liquid-first), and declare a funding
default only when the shortfall persists beyond `failure_after_business_days`.

## Structured products

Deals run once through `structured.engine.InternalWaterfallEngine` (collateral cash from realized
remittances where observed, projected from CPR / CDR / severity assumptions otherwise, never both for
one period; fees, interest by seniority with OC / IC diversion, reinvestment then sequential or
pro-rata principal, equity residual, conservation checked). Tranche allocations become
`tranche_payments` (interest and principal per 100 original face by pay date) and tranche factors.
The ledger pays held tranches from those allocations instead of from a coupon schedule. The external
engine adapter is the contract for Intex-style engines; it is not integrated.

## Extending

* Vendor data: produce the frames in `docs/data_contracts.md` and feed `MarketEnvironmentBuilder`.
* Models: subclass `LiquidityModel`, `CostModel`, `PricingWaterfall`, `CashRateModel`, register
  model versions in the `ModelRegistry`, and declare capabilities in `support.py`.
* Strategies: subclass `FixedIncomeStrategy`; use only the `DecisionContext` (`self.ctx`) and the
  services the engine hands you (`self.services`); override `on_funding_shortfall` for a mandate
  specific liquidation rule.
* Derivative overlays: `BacktestSettings(margin=MarginSettings(collateral, book=OverlayBook(...)))`
  adds a MARGIN venue; contracts are `overlays.FinancedContract` (index TRS, CDX, synthetic bond TRS,
  carry instruments: daily carry as Nautilus funding) or `overlays.DatedContract` (futures, FX
  forwards); strategies trade them with `rebalance_overlays({symbol: notional fraction of NAV})`.
* Research specs: implement a `domain.spec.Decider` and run the same `ResearchSpec` in
  `paper.PaperEngine` and in the event engine through `SpecStrategy`.
