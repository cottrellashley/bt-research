# Blueprint delivery status

Section by section against *Fixed Income Backtesting Blueprint*. "Wired" means the mechanism runs
inside `FixedIncomeBacktest`; "module" means the code and its unit tests exist but the backtest loop
does not call it yet. Statuses follow the support matrix vocabulary.

| Blueprint area | Status | Where | Notes |
| --- | --- | --- | --- |
| Typed quantities, four timestamps, append-only event log with lineage | VALIDATED, wired | `core.quantities`, `core.time`, `core.events` | every ledger posting cites its event; hash-chained log in results |
| Double-entry ledger of record, conservation | VALIDATED, wired | `core.ledger`, `accounting.postings`, `accounting.ledger` | NAV identity and conservation asserted in tests |
| Run modes, manifests, status vs outcome, output hashes | VALIDATED, wired | `core.manifest`, `backtest.builder` | HISTORICAL_REPLAY exercised; COUNTERFACTUAL is EXPERIMENTAL; STOCHASTIC_SIMULATION unsupported |
| Support matrix, model registry, dependency validation | VALIDATED, wired | `core.registry`, `support.py`, `MarketEnvironmentBuilder._validate_dependencies` | strict mode refuses; lenient mode excludes and reports |
| Assumption sets with precedence and ranges | VALIDATED, wired | `core.assumptions` | resolved values recorded in the manifest |
| Restricted decision context | VALIDATED, wired | `core.context`, `backtest.actor`, `backtest.strategy` | no environment access from strategies |
| Dated settlement conventions | VALIDATED, wired | `instruments.conventions` | US T+1 transition, Treasuries, securitized, loans, EUR, UK, JP, CA, AU, CH |
| Coupon entitlement (payment-date holder, record date, ex-date with negative accrued) | VALIDATED, wired | `accounting.ledger._process_entitlements`, `instruments.conventions` | receivable crystallised before payment; written off on default |
| Floating conventions (in advance, compounded in arrears with lookback / lockout, simple average, index), benchmark transition | VALIDATED, wired | `instruments.floating`, `instruments.cashflows` | partial-knowledge determinations flagged |
| Inflation indexation (US TIPS, UK 3m / 8m, EUR, CA) | VALIDATED (US) / APPROXIMATION (others), wired | `instruments.linkers` | CPI publication lag respected; projected prints flagged |
| Step, sinking, PIK, make-whole, tender, partial call | VALIDATED / APPROXIMATION, wired through actions | `instruments.bond`, `accounting.postings` | make-whole and tender use rule-based exercise, no price model |
| Loans (fees, delayed compensation, amendments, transfer restrictions) | VALIDATED / APPROXIMATION | `instruments.loans`, `instruments.terms_history` | T+7 settlement and index floors wired; delayed compensation is a module |
| Hybrids / preferreds (deferral, discretionary call, write-down) | VALIDATED / APPROXIMATION | `instruments.hybrids` | perpetuals trade and accrue; deferral economics are a module |
| Distressed lifecycle (default, claims, recovery, exchange) | VALIDATED, wired | `instruments.distressed`, `lifecycle.state_machine`, `accounting.ledger` | claims carried at recovery marks; exchange posting exists |
| Money market (amortized cost, liquidity fees) | APPROXIMATION | `instruments.money_market` | shadow NAV check demonstrated in the sleeve runner |
| Municipal tax treatment | APPROXIMATION | `instruments.tax` | after-tax proceeds; no AMT / de minimis |
| Funds (NAV vs market) | APPROXIMATION | `instruments.funds` | |
| MBS factors, paydowns, prepayment behaviour models | VALIDATED (factors) / APPROXIMATION (models) | `accounting.ledger`, `pricing.behavior` | realized factors drive the ledger; PSA / S-curve are valuation inputs |
| TBA delivery and dollar rolls | APPROXIMATION, module | `structured.tba` | not wired to the ledger |
| Deal waterfall (OC / IC, reinvestment, sequential / pro-rata, equity), tranche payments | APPROXIMATION, wired | `structured.*`, `synthetic.generator`, `accounting.ledger` | CLO / ABS / CMBS deals run through the internal engine; deferred tranche interest is reported, not capitalised |
| External deal engines (Intex) | EXPERIMENTAL | `structured.engine.ExternalEngineAdapter` | adapter contract only |
| Treasury futures, IRS, CDS, FX forwards | APPROXIMATION, module | `instruments.derivatives`, `accounting.postings` | valuation and margin / VM postings exist; not tradable inside the Nautilus venue (needs a margin venue) |
| FX conversion and translation | VALIDATED, wired | `accounting.ledger._fx_fund`, `instruments.fx` | spot conversion through the translation account; hedge programs are a module |
| Repo, securities lending, margin | APPROXIMATION | `funding.repo`, `funding.borrow`, `funding.margin`, `accounting.ledger` | funding repo under the breach ladder is wired; strategy-driven repo / shorts are modules with hooks (`borrow_available`) |
| Funding policy: settle before spend, bridge, breach ladder | VALIDATED, wired | `funding.policy`, `funding.cash`, `accounting.ledger._process_funding` | hold buys -> repo -> liquidate by rule -> declare failure |
| Marks: four values, evidence classes, lineage, stale roll | VALIDATED, wired | `pricing.marks`, `pricing.waterfall`, `pricing.stale_roll` | evidence shares in results and bundles |
| Calibration recipes with explicit status | VALIDATED, wired | `market.calibration`, `MarketEnvironment.curve` | STALE / FAILED recorded per date |
| Pricing engines (discounting, Hull-White trinomial for callables) | VALIDATED / EXPERIMENTAL | `pricing.engine`, `pricing.quantlib_adapter` | HW tree is 0.1; exercise in replay comes from realized call actions |
| Execution process (RFQ panel, voice, portfolio trades, liquidity budget, cost decomposition) | APPROXIMATION, wired | `execution.process`, `execution.fill_model` | scenario parameters until calibrated on own order history |
| Prior-close reference, window-time liquidity | VALIDATED, wired | `MarketEnvironment.reference`, `data.trace.daily_liquidity_stats` | |
| Portfolio construction, constraints, forced actions, risk | VALIDATED, wired | `portfolio.*`, `backtest.strategy` | constraints are typed; soft violations logged |
| Attribution and returns | VALIDATED, module + partial wiring | `accounting.attribution`, `accounting.returns`, `result.attribution()` | result attribution is income / execution / price; the full decomposition is a module |
| Research: walk-forward with purging, bootstrap, Newey-West, deflated Sharpe, PBO, stress | VALIDATED, module | `research.*` | call around `FixedIncomeBacktest`; not automated inside the builder |
| Invariants | VALIDATED, partly wired | `validation.invariants`, `result.invariants()` | conservation, payments reconcile, unfilled orders run in results; temporal isolation, duplicate delivery, lot regrouping, pinned replay, failed settlement are suites to call in tests |
| Reports: bundle, explain order / payment, unknowns | VALIDATED, wired | `reports.bundle`, `result.bundle()` | |
| Corrections / restatement bridge | VALIDATED, module | `lifecycle.corrections` | environment rebuild on restatement is manual |

## Not built

* Stochastic simulation run mode; auction / reopening events; covenants; municipal extraordinary
  redemptions; convertibles.
* A margin venue for exchange-traded and OTC derivatives inside Nautilus (valuation exists).
* Vendor adapters (BVAL / ICE / TRACE / Intex loaders): the data contracts are defined, the loaders
  are the user's.
* Calibration of the execution process on the desk's own RFQ / TRACE history: hooks and scenario
  defaults exist, the fitted model does not.
