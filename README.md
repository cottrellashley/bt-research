# bt-research: a fixed income backtesting platform on Nautilus Trader v2

`nautilus_fi` turns [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) (v2, Rust
core + PyO3) into a research-grade paper-portfolio backtester for the whole fixed income universe:
governments, bills and commercial paper, investment grade and high yield credit (fixed, step,
floating in advance or compounded in arrears, callable, sinking, PIK), municipals, hybrids and
preferreds, leveraged loans, hard-currency EM sovereigns, non-base-currency bonds, inflation linkers,
agency MBS and the tranches of CLO / ABS / CMBS deals. It implements the design in
*Fixed Income Backtesting Blueprint* and *Backtesting Fixed Income Paper Portfolios on Real Data*:
knowledge-time data, marks kept apart from fills, dated settlement and entitlement conventions, a
double-entry ledger of record, explicit funding, run manifests with support statuses, and reports
that say what is validated, what is approximated and what is unknown.

| Blueprint engine | What it does here | Package |
| --- | --- | --- |
| Core | typed quantities, four timestamps, append-only event log with lineage, double-entry ledger, support matrix, model registry, dependency graph, assumption sets, run manifests, restricted decision context | `nautilus_fi.core` |
| Instruments and conventions | every bond term type, day counts, schedules, curves, cashflow projection and analytics, dated settlement rules (US T+3 → T+2 → T+1 on 28 May 2024, Treasuries, securitized, loans, EUR / UK / JP ...), coupon entitlement rules (payment-date holder, record date, ex-date with negative accrued), loans, hybrids, distressed, money market, tax, funds, derivatives, FX | `nautilus_fi.instruments`, `nautilus_fi.lifecycle` |
| Point-in-time data | custom data replayed at availability time, never-overwrite snapshot store, TRACE cleaning with dissemination lags, window-time liquidity, CPI by publication date | `nautilus_fi.data` |
| Market, pricing and marks | market state and calibration recipes with explicit status, tiered waterfall marks with evidence classes and lineage, matrix pricing, stale roll, behaviour models (PSA, refinance S-curve, issuer call policy, default model), discounting and Hull-White engines | `nautilus_fi.market`, `nautilus_fi.pricing` |
| Structured | collateral pools (realized or projected, never both), deal waterfall with OC / IC tests and reinvestment, internal deal engine, external engine adapter contract, TBA / dollar rolls | `nautilus_fi.structured` |
| Execution process | prior-close execution reference, order-process simulator (RFQ panel, voice, portfolio trades), liquidity budget shared across orders, cost decomposition without delay double count, GTD orders resting across windows, settle-before-spend funding gate | `nautilus_fi.execution` |
| Funding | settlement queue, reservations, cash view, repo, borrow, margin, funding policy with breach ladder (hold buys → repo → liquidate by rule → declare failure) | `nautilus_fi.funding` |
| Accounting | posting policy for every event type, ledger of record (NAV = assets − liabilities), entitlement before payment, claims and recoveries, FX translation, attribution and return measures | `nautilus_fi.accounting` |
| Portfolio | typed hard / soft constraints with recorded relaxations, constructor that nets live orders and reserves cash, forced-action policies, risk | `nautilus_fi.portfolio` |
| Research and validation | walk-forward with purging, block bootstrap, Newey-West, deflated Sharpe, PBO, stress; invariants (temporal isolation, duplicate delivery, conservation, lot regrouping, pinned replay, unfilled orders, failed settlement, payments reconcile) | `nautilus_fi.research`, `nautilus_fi.validation` |
| Reports | report bundle (performance, implementation, evidence, risk, attribution, robustness, support, invariants, unknowns), order and payment explainers | `nautilus_fi.reports` |
| Orchestration | `MarketEnvironmentBuilder`, `FixedIncomeBacktest` (manifest, validation, wiring, computational status vs economic outcome), strategy base, scenarios | `nautilus_fi.backtest` |

Honest status per blueprint area is in [docs/blueprint_status.md](docs/blueprint_status.md); the
product / feature support matrix is in [docs/support_matrix.md](docs/support_matrix.md). Every run
carries the rows that apply to it.

## Install

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python --pre -e ".[dev]"      # nautilus_trader 2.0.0rc8+
uv pip install --python .venv/bin/python QuantLib                # optional: callable OAS adapter
```

Nautilus v2 is a release candidate; `nautilus_trader<2` (the Cython v1 line) is **not** supported.
On macOS older than 26 pin `nautilus_trader==2.0.0rc8` (the v1.231 wheels need macOS 26). The
example scripts add the checkout to `sys.path`, so they also run without the editable install.

## Quick start

```bash
python examples/run_fund_sleeves.py --quick    # 13 fund-style sleeves on a 51-instrument synthetic market
python examples/run_carry_strategy.py --quick  # one strategy under GROSS vs NET_SPREAD vs NET_FULL
python examples/run_cost_scenarios.py          # cost stress x1/x2/x3 and a capacity sweep
pytest                                         # 92 tests, several minutes (environment builds dominate)
```

A strategy is a `FixedIncomeStrategy` that sees the market only through its `DecisionContext`
(`self.ctx`): marks, liquidity, ratings, curves, fixings, stress, actions, positions, open orders,
cash states, NAV and the funding shortfall, each delivered when it became available. Orders go
through the portfolio constructor (netting live orders, reserving settled cash, typed constraints,
forced actions) and rest as GTD limit orders until the next execution window.

```python
from nautilus_fi.backtest import FixedIncomeStrategy, FixedIncomeStrategyConfig, FixedIncomeBacktest, BacktestSettings
from nautilus_fi.execution import NET_FULL
from nautilus_fi.universe import UniverseFilter
from nautilus_fi.instruments import AssetType

class CheapIG(FixedIncomeStrategy):
    def on_decision(self, d):                       # fires at 16:45 on rebalance dates, after marks are published
        bonds = self.universe(UniverseFilter(asset_types={AssetType.CORPORATE}, max_rating_score=10))
        picks = sorted(bonds, key=lambda b: -self.mark(b.symbol).z_spread_bp)[:20]
        self.rebalance_to_weights({b.symbol: 0.9 / len(picks) for b in picks})   # GTD limits, fill t+1 or later

result = FixedIncomeBacktest(env, CheapIG(FixedIncomeStrategyConfig(rebalance="M")), NET_FULL,
                             settings=BacktestSettings(starting_cash=100e6, hypothesis="cheap IG carry")).run()
print(result.status, result.outcome)             # COMPLETED SOLVENT
print(result.summary())                          # returns, costs, fill rate, evidence shares, funding, reconciliation
print(result.evidence_table())                   # share of market value and turnover by valuation evidence class
print(result.invariants())                       # conservation, payments reconcile, unfilled orders
open("run.md", "w").write(result.bundle().to_markdown())
```

`env` is a `MarketEnvironment` built once from your data (`MarketEnvironmentBuilder`, see
[docs/data_contracts.md](docs/data_contracts.md)) or from the synthetic generator; every strategy
and execution variant reuses it. The environment build is the slow part (about 1 ms per
instrument-day); a Nautilus run takes a couple of seconds.

## Fund-style sleeves

`examples/fund_sleeves.py` holds thirteen mandate archetypes (AAA CLO, mezzanine CLO, agency MBS,
securitized income, ultrashort, IG corporate, EM hard currency, multi-sector income, core-plus,
short-duration flexible, high yield, developed-world hedged, money market) as bucketed
`FixedIncomeStrategy` subclasses with constraints and forced-action policies. On the 2023 synthetic
market (51 instruments including a CLO, an ABS and a CMBS deal run through the internal waterfall
engine, loans, a hybrid perpetual, munis, EM sovereigns, EUR credit funded at spot, a CPI-indexed
linker, commercial paper) all thirteen complete, stay solvent and keep the ledger balanced every day:

| Sleeve | Return | Fill rate | Avg delay (bd) | Spread cost (bp) | Model / stale share | Ledger balanced |
| --- | --- | --- | --- | --- | --- | --- |
| CLO_AAA | +0.9% | 0.06 | 3.1 | 53.6 | 7.5% | yes |
| CLO_MEZZ | −0.2% | 0.03 | 3.2 | 105.5 | 94.5% | yes |
| AGENCY_MBS | +13.0% | 0.39 | 2.1 | 3.0 | 0.0% | yes |
| SECURITIZED_INCOME | +6.9% | 0.49 | 2.0 | 36.8 | 7.6% | yes |
| ULTRASHORT | +3.2% | 0.62 | 2.1 | 11.3 | 1.3% | yes |
| CORPORATE_IG | +6.3% | 0.57 | 2.3 | 21.5 | 9.8% | yes |
| EM_HARD_CURRENCY | +30.6% | 0.87 | 1.4 | 0.2 | 3.3% | yes |
| MULTI_SECTOR_INCOME | +10.4% | 0.77 | 1.9 | 52.1 | 22.3% | yes |
| CORE_PLUS | +13.2% | 0.96 | 1.7 | 18.8 | 7.0% | yes |
| SHORT_DURATION_FLEX | +4.5% | 0.75 | 2.0 | 18.2 | 6.5% | yes |
| HIGH_YIELD | +2.3% | 0.21 | 2.5 | 66.1 | 27.4% | yes |
| DEVELOPED_WORLD_HEDGED | +5.8% | 0.73 | 1.2 | 0.9 | 0.2% | yes |
| MONEY_MARKET | +1.3% | 1.00 | 1.5 | 0.0 | 0.6% | yes |

The numbers come from a synthetic market and say nothing about real markets. What they show is the
platform doing its job: structured sleeves fill slowly and are marked mostly by model (the 94.5%
model share on mezzanine CLO is the warning the blueprint asks for), the gate and the liquidity
budget bite where paper is scarce, and every sleeve reconciles to the cent against Nautilus cash.

## The execution ladder

`examples/run_carry_strategy.py --quick` (68 synthetic bonds, 2023, monthly carry / value picks):

```
                        GROSS    NET_SPREAD    NET_FULL
total_return           0.1019       0.0823      0.0746
fill_rate              1.0000       0.9024      0.6049
avg_delay_days         1.0000       2.1429      2.4498
spread_cost_bp         0.0007      22.4527     21.4706
delay_drift_bp         0.0000      -6.7413     -4.7153
avg_model_marked_share 0.0868       0.0909      0.0483
ledger_balanced        True         True        True
```

Gross, net of spread, and net of spread + volume-gated fills + delays + expiries: the "where does
the alpha disappear" view, with spread cost measured against the prior-close reference and delay
drift reported as attribution rather than charged twice.

## Representation choices worth knowing

* Bonds are Nautilus `Equity` instruments quoted per 100 with quantity in units of 100 face;
  factor instruments carry original face and a factor-adjusted price. Nautilus v2 does not allow
  multiplier instruments on cash accounts, and cash accounting is the whole point. Strategies never
  see this: `order_face`, `holdings_face`, `rebalance_to_weights`.
* Maturities, calls, tenders and recoveries are `InstrumentClose` events at the redemption price;
  the ledger books accrued due at redemption, stops accrual on default, carries claims at recovery
  marks and writes off entitled-but-unpaid coupons.
* Coupon entitlement crystallises on the first trade date whose settlement no longer carries the
  coupon (ex-date for gilt-style markets, record date where applicable), for the start-of-day holder.
  No catch-up coupons, no coupons to same-day buyers; the receivable is part of NAV until paid.
* Orders submitted at the decision time can only fill at the next execution window (t+1 earliest),
  against the prior-close reference plus the simulator's process, never against the signal's print.
* Spendable cash is settled cash less committed payables less reservations; non-base purchases
  convert base cash at spot through the FX translation account; funding breaches run the policy's
  ladder and a funding default is an economic outcome, not a crash.
* Every run has a manifest: data hashes, model versions, resolved assumptions with ranges, support
  rows, seeds, output hashes and the event-chain hash. `computational_status` (COMPLETED,
  ENGINE_FAILURE, INVALID_CONFIGURATION, UNSUPPORTED_FEATURES) is separate from `economic_outcome`
  (SOLVENT, FUNDING_DEFAULT, INSOLVENT).

## Plugging in real data

Produce the frames in [docs/data_contracts.md](docs/data_contracts.md) from your vendors (evaluated
prices, composite quotes, cleaned TRACE through `clean_trace`, curves and OIS, term and daily
fixings, factors, CPI, FX, trustee tranche payments, terms, rating history, actions, a stress index,
index returns) and store each pull in `PointInTimeStore`. Then:

```python
builder = MarketEnvironmentBuilder(reference_df, "2019-01-02", "2024-12-31", strict_dependencies=True)
builder.with_evaluated_prices(bval).with_composite_quotes(cp_plus).with_trades(trace_clean).with_curves(curves)
builder.with_fixings(term_sofr).with_fixings(sofr_daily, daily=True).with_factors(factors).with_cpi(cpi).with_fx(fx)
builder.with_tranche_payments(trustee).with_ratings(ratings).with_corporate_actions(actions).with_stress(stress)
env = builder.build()          # refuses instruments whose required data is missing (or excludes them when lenient)
```

Calibrate `ParametricCostModel`, `VolumeGatedLiquidityModel` and the RFQ panel parameters on your
own TRACE / RFQ history (the defaults encode stylised facts, not your desk), then validate simulated
fill rates, delays and slippage against realised TCA. The support matrix marks the execution process
as SUPPORTED_WITH_APPROXIMATION until that is done.

## Layout

```
nautilus_fi/
  core/          quantities, time, events, ledger, registry, assumptions, manifest, context
  instruments/   terms, conventions, calendars, day counts, schedules, curves, cashflows, analytics, floating,
                 linkers, loans, hybrids, distressed, money_market, tax, funds, derivatives, fx, terms_history
  lifecycle/     state machine, corrections
  funding/       cash, repo, borrow, margin, policy
  structured/    collateral, waterfall, engine, tba
  market/        state, calibration
  pricing/       waterfall, matrix, stale_roll, behavior, engine, marks, quantlib_adapter
  execution/     windows, assumptions, liquidity, costs, process, fill_model, settlement
  accounting/    postings, ledger, attribution, returns, cashflow_module, financing
  portfolio/     constraints, construction, forced_actions, risk
  research/      experiment, walkforward, statistics, stress
  validation/    invariants, tolerances
  reports/       bundle
  backtest/      environment, builder, strategy, actor, context, result, scenarios
  synthetic/     generator (every product family, three structured deals)
  support.py     the declared support matrix
examples/        fund_sleeves, strategies, run_fund_sleeves, run_carry_strategy, run_cost_scenarios
tests/           92 tests: core, products, lifecycle/funding, market/pricing, execution/portfolio,
                 research/validation, redemption accounting, fund sleeves, end to end
docs/            architecture, data contracts, bias checklist, support matrix, blueprint status
```

## Known limits

See [docs/blueprint_status.md](docs/blueprint_status.md) for the full list. The short version:
derivatives, TBA delivery, FX hedge programs, securities lending and strategy-driven repo exist as
valuation and posting modules but are not tradable inside the Nautilus venue; the external deal
engine is an adapter contract; the stochastic simulation run mode, auction events, covenants,
extraordinary municipal redemptions and convertibles are not built; vendor loaders and the
calibration of the execution process on your own order history are yours.
