# Data contracts

All inputs are pandas frames; the schema-validated ones go through `nautilus_fi.data.schemas`.
Every price-like dataset should carry a `publish_ts` (knowledge time); when missing, the builder
treats the datum as knowable at the mark publish time of its date. Store every vendor pull in
`PointInTimeStore` with its retrieval time and read "as of" a knowledge time when rebuilding history.
The manifest records each dataset's row count, content hash, knowledge time and availability policy.

| Dataset | Builder method | Required columns | Optional | Typical source |
| --- | --- | --- | --- | --- |
| `reference` | constructor | symbol, issuer, issue_date, maturity (empty = perpetual), coupon_rate | coupon_type, frequency, day_count, currency, country, asset_type, principal_type, seniority, sector, rating_initial, min_piece, increment, amount_outstanding, float_index, float_spread_bp, float_method (IN_ADVANCE / COMPOUNDED_IN_ARREARS / SIMPLE_AVERAGE_IN_ARREARS / INDEX), lookback_days, floor, floor_on_index, inflation_index, inflation_convention (US_TIPS / UK_GILT_3M / UK_GILT_8M / EUR_OATEI / CA_RRB), call_schedule (JSON [[date, price], ...]), original_term_months, projected_cpr, settlement_lag_days, cusip, isin, figi, tags | Mergent FISD, Bloomberg DES, deal documents |
| `evaluated_prices` | `with_evaluated_prices` | date, instrument_id, mid | bid, ask, score (1-10), source, publish_ts | BVAL, ICE, LSEG |
| `composite_quotes` | `with_composite_quotes` | ts, instrument_id, bid, ask | bid_size, ask_size, source | CP+, Ai-Price, CEP, dealer runs (quotes before the window time also feed the execution reference) |
| `trades` | `with_trades` | instrument_id, exec_ts, available_ts, price, size, public_size, side | size_revealed_ts | TRACE via `clean_trace` |
| `curves` | `with_curves` | date, curve, tenor (years), rate | kind (`par` default or `zero`), publish_ts; a curve whose name ends in `OIS` becomes the discount curve | Treasury par, SOFR OIS |
| `fixings` | `with_fixings(df)` | date, index, rate | publish_ts | term / in-advance fixings (SOFR term, EURIBOR) |
| daily fixings | `with_fixings(df, daily=True)` | date, index, rate | publish_ts | overnight rates for compounded-in-arrears coupons (SOFR, SONIA) |
| `factors` | `with_factors` | date, instrument_id, factor | paydown_price, publish_ts | Bloomberg MTG_FACTOR, Intex, trustee reports |
| `index_ratios` | `with_index_ratios` | date, instrument_id, index_ratio | publish_ts | Treasury / DMO linker tables |
| `cpi` | `with_cpi` | index, year, month, value | publish_ts | BLS CPI-U, ONS RPI / CPIH, Eurostat HICPxT |
| `fx` | `with_fx` | date, pair (e.g. EURUSD), rate | tenor_months + points rows for forward points | WM/Refinitiv fixings |
| `tranche_payments` | `with_tranche_payments` | symbol, pay_date, interest_per_100, principal_per_100 | deferred_per_100, source (REALIZED / PROJECTED) | trustee remittance reports, deal engine output |
| `ratings` | `with_ratings` | date, instrument_id, rating | agency, publish_ts | rating histories |
| `corporate_actions` | `with_corporate_actions` | instrument_id, action (CALL / PUT / TENDER / DEFAULT / RECOVERY / MATURITY / EXCHANGE), effective_date | price, announce_ts, note | FISD, Bloomberg CACS, trustee notices |
| `stress` | `with_stress` | date, value | name, publish_ts | CDX / index OAS / VIX normalised (0 calm, ~1 March 2020) |
| `benchmark` | `with_benchmark` | date, total_return | name, duration, oas | index vendors |
| new issues | `with_new_issues` | `NewIssueTerms` per symbol | reoffer price, allocation fraction, issue date | syndicate data |

Maturities are generated automatically for every bond maturing inside the window unless an explicit
action exists. Dependency validation refuses (strict mode) or excludes (lenient mode) instruments
whose data the terms require but the environment lacks: in-advance floaters need term fixings,
in-arrears floaters need daily fixings, linkers need CPI or index ratios, callables need the
exercise model, and benchmark transitions need the new index. The outcome is in
`env.validation_problems` and `env.unsupported_symbols`, and in the manifest messages.

## TRACE cleaning (`nautilus_fi.data.trace.clean_trace`)

Map raw columns with `columns={...}`; the function removes cancellations / corrections (and the
messages they reference), reversals, inter-dealer double reports, when-issued and special-price
trades, odd lots below `min_size`, and prices deviating more than `max_abs_return` from the
neighbouring median. It stamps `available_ts` (report time, else execution + 15 minutes), caps
`public_size` at the dissemination cap (5mm IG / 1mm HY by default) and records when the true size
became public (`size_revealed_ts`, Enhanced TRACE lag). `daily_liquidity_stats(..., cutoff_time)`
computes window-time statistics that use only prints disseminated before the execution window.

## Point-in-time semantics inside the engine

* Custom data is replayed by `ts_init`. `BondMark.ts_event` is the mark time, `ts_init` the publish
  time; `BondTrade.ts_init` is the dissemination time; CPI prints carry their publication date.
* Strategies hold a `DecisionContext` filled by the environment actor as data becomes available.
  It exposes marks, liquidity, ratings, curves, fixings, stress, actions, positions, open orders,
  cash states, NAV and the funding shortfall; attribute access to the environment, ledger, engine or
  cache raises.
* The fill model and ledger read the environment for *today* only (they are the market). Execution
  references are the prior close (or window-time composite quotes), never the fill-day close.
* Terms histories (`instruments.terms_history`) and the restatement bridge
  (`lifecycle.corrections`) keep announcement and effective dates apart for amended terms.
