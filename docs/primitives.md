# Platform primitives

The generic capabilities several legacy frameworks need (section 4 of the 2026-10-30 gap analysis),
by area. Each is tested in `tests/test_primitives.py`; test names start with the section number
(`test_43_*` for timing). Framework-specific behaviour lives in the adapters
([adapters.md](adapters.md)); live proposals and the operational interface are in
[operations.md](operations.md).

## 4.1 Return construction and data contracts

* **Declared input kinds.** `FieldInfo(form=...)`: `"simple"` (the default), `"log"`, `"price"`
  (levels) or `"cumulative"` (additive cumulative returns). Wherever a return is realised, the
  kernel converts it once (`expm1`, percentage change, difference). A log return read as simple
  is a declaration error, not a silent conversion.
* **Bond total return from prices.** `returns.total_return_from_prices(clean, accrued, factor,
  ...)`: `((clean + accrued)_t f_t + coupon f_{t-1} + (f_{t-1} - f_t) 100) / ((clean +
  accrued)_{t-1} f_{t-1}) - 1`. Coupons are inferred from accrued drops on dates that are not
  defaults; distressed accrued is written off as a loss.
* **MTD de-cumulation.** `decumulate_mtd` turns month-to-date series into daily returns
  (`(1 + mtd_t) / (1 + mtd_{t-1}) - 1`, the month's first day equal to its MTD).
  `mtd_from_daily` goes back, keeping `tr = ir + xr` on every date.
* **Leg alignment.** `align_legs(total, rate, mode)`: `"inner"` keeps the common dates and warns,
  `"strict"` raises, `"outer"` carries zero (reported). `common_dates(*series)` gives the
  intersection.
* **Returns from expressions.** `Strategy(returns=SpreadReturn())`, or any expression such as
  `Field("oas") * YearFraction() - Field("oasd") * Field("oas").diff()`. The panel gets a
  `returns:<hash>` column and every engine reads it as the realised return.
* **Resampling.** `resample_returns(r, "ME")` compounds to a lower frequency, optionally on the
  dates common to several sleeves.
* **Forward returns for evaluation.** `forward_returns(tr, horizons, ir, lag, alive,
  membership)` gives forward total, rate and excess returns per bond and horizon, with `exit`,
  `missing` and `left_index` flags.
* **Availability windows.** `FieldInfo.start` / `end` are enforced. A rule that reads a field
  before its history starts raises unless it is guarded with `Available(name)`, for example
  `~Available("esg") | (Field("esg") > 3)`.
* **Dated attributes.** `FrameSource(..., categories={"sector": wide_label_frame})` reclassifies
  over time. Groups, constraints and reports read the label as of each date.

## 4.2 Return and valuation models

* `SpreadReturn(oas, duration)`: the realised spread proxy `OAS dt - OASD dOAS`.
  `spread_return(...)` splits it into carry and spread legs, with the treatment of exits.
* `duration_convexity_return(carry, D, C, ds)`: `carry - D ds + C ds^2 / 2`.
* `constant_maturity_returns(yields)`: government constant-maturity returns,
  `-ModDur_{t-1} dy + accrual`. `carry_rolldown(yields, tenor, horizons_months=(1, 3))` gives
  carry and roll-down at 1m and 3m.
* `fx_hedged_returns(local, spot, forward, ratio=1.0, roll=None)`: `(1 + r) h_{t-1} / h_t - 1`
  with `h = F / S`. `roll="ME"` runs a hedge programme: each forward is held from one roll to the
  next and `ratio` sets the share hedged. `unhedged_returns` converts at spot and `hedge_carry`
  gives the annualised forward premium. A multi-currency universe runs in the base currency: give
  each bond its base-currency return (hedged or not) as the return field, and use
  `CurrencyExposure` for the currency limits.
* **Attribution.** `kernel.analytics.attribution(result, source, by="sector", fx=None,
  active=False)` splits each period's return into rates, carry, spread, FX, residual, cash,
  financing, overlays and costs. The legs add up exactly to the net return in every period, and
  the groups add up to the portfolio. With `active=True` it attributes active return instead.

## 4.3 Timing and schedules

* `NextClose(0)` raises: same-bar trading must be written `SameClose()`.
* `Expr.ffill(days=7)` sets staleness in calendar days: MRI v2 allows 5, Erin 7.
* **Calendars:**
  * `YearEnd()` (`"Y"`, `"A"`, `"BYE"`) and `YearStart()`;
  * `Weekly(2, every=2)`: biweekly Wednesday;
  * `NthBusinessDay(2)`: the second business day of each month;
  * `AnchoredWeekly(offsets=(7, 14, 21), min_gap_days=7)`: anchors plus offsets, skipping
    dates too close to the next anchor;
  * `IndexDates("LUACTRUU")`: the index's reconstitution dates, read from the source.
* **Interim rebalances.** `WithInterim(fixed, interim=Triggered(...), portfolio=...)` adds
  triggered rebalances between fixed dates, each with its own construction.
  `Triggered(threshold, "score_drop")` fires on a fall in the book's portfolio-weighted score.
* **Names.** `"M"`, `"W"` and `"Q"` warn: write `BMS` / `BME`, `WS` and `BQS`, as in pandas.
  `"2W-WED"` is parsed.
* **Per-event turnover caps.** `AnnualTurnoverBudget(annual, month_end_share=0.4,
  events_per_month=3)`.
* **Lag stress.** `Session.lag_stress(strategy)` gives the metric at execution lags 0-3, and
  `RecordRegistry(min_lag_stress=...)` gates promotion on the worst of them.

## 4.4 Universe and benchmark

* **Index mappings.** `data/loaders/fiatlas_catalog.example.json` lists LUACTRUU, LF98TRUU,
  LECPTREU, LP01TREU, JBCDCOMP, LD08TRUU and LEC1TREU, each with a `benchmark_constituents_<code>`
  dataset. All are flagged "to confirm" against the platform.
* `PairedIndex(construction, returns)`: Erin's `_STATS` / `RU` pair. Decisions see one index,
  returns the other.
* **Redemptions.** `Index(..., redemptions="pro_rata" | "cash")`, also `drift_weights(...,
  redemptions=)`.
* **Replication.** `replication_report(benchmark, published, source, tolerance_bp=20)` tests the
  rebuilt index against the provider's published returns. This is the non-circular test.
* **Reweighting.** `Benchmark.reweight("market_value", field=..., schedule="BME")` (or
  `"equal"`), reconstituted on its own schedule.
* **Small groups.** `ActiveWeight("sector").within(0.02).skip_below(0.01)` skips a group the
  index holds less than 1% of.
* **Formula filters.** `BenchmarkStat("oas_bp", benchmark)` is the benchmark's weighted average
  on the date. `Field("oas").cs_quantile(0.9)` is the cross-sectional percentile within the
  universe. They are the `@bench.X` and `@universe.X.pN` of Erin's filter DSL.

## 4.6 Construction methods

* `Stratified(cells, per_cell)`: the rule-based constructor, with maturity x rating cells
  matched to the index.
* `Quantile(..., buffer=, keep_weights=)`: drift-aware quantiles with buffers.
* `ZScoreRule(z, entry, exit, stop, max_hold)`: a per-name state machine on the signal's path.
  `returns.pair_spread_pnl(a, b, beta="rolling")` gives pairs in spread units with a lagged
  rolling OLS beta, and `equal_move_sizing` sizes the legs.
* `Rules(base, rules)`: per-name actions by precedence (sell > exclude > underweight > hold /
  buymax / buy).
* **Hedges.**
  * `Hedge(..., tolerance=, lot=, cost_per_contract=, contract_notional=)`.
  * `KRDWaterfall(buckets, targets)` hedges longest bucket first, with targets as numbers or
    `fn(cs)`.
  * `Hedge(..., joint=True)` solves the hedge with the book (see 4.9).
* Positions in factor units: `Bounds(names=..., lo=..., hi=...)` takes expressions. The assetiq
  adapter maps x = w x DTS.

## 4.7 Optimiser

* **Tracking error.** `TEV(model) <= x`, as a hard limit, solved by risk-aversion bisection (a
  QP), or by cutting planes with integers or a user risk term. `TargetTE(model, x)` spends a
  budget. `MaxIR(base, model, levels)` and `MaxSharpe(...)` search the frontier.
* **Minimum trade.** `MinTrade(w)`: each trade is zero or at least `w` of NAV (a MILP).
* **Tie-break.** `Optimise(..., tie_break=-ActiveShare())` runs a second stage at no loss in the
  first.
* **Per-name limits.**
  * `Bounds(lo, hi, names)`, `Pin({id: w})`, `NoSell(names)` and `NoBuy(names)`.
  * `TradeLimit(max_trade)`: capacity, for example `Field("adv") * 0.1 / nav`; a band that needs
    more binds at the capacity.
* **Bounds that change by date.** Any bound may be an expression or `fn(cs)`, for example
  `ActiveExposure("dts", relative=True) <= Field("mri_ratio") - 1`.
* **Subsets and relative caps.**
  * `ActiveWeight("issuer").where(Field("sector") == "BANKING") <= 0`.
  * `GroupWeight("issuer").cap(0.04, 3.0)` caps each issuer at min(4%, 3 x its index weight).
* **Currency.** `CurrencyExposure("EUR", base="USD")` counts cash in its currency.
* **Relaxation ladder.** `OptimiserSettings(on_infeasible="relax", relax_tiers=(1.5, 2.0),
  relax_soft=True)`. The ladder grandfathers breaches the book already has, widens non-strict
  limits by the tiers, then softens them. Each step is recorded per rebalance. Strict limits are
  never relaxed. `diagnose_conflicts` names the limits that clash. `on_infeasible="push"` keeps the
  book and retries the next day.
* **Turnover.** Forced exits are outside the budget. `IndexTurnover(benchmark, window,
  allowance)` sets a budget of the index's own turnover plus an allowance.
* **Sparsity.** `Sparse(base, names=(40, 60))` uses IRL1 to hit a holdings band.
* Dense risk blocks for up to 1,500 names (`DENSE_RISK_MAX_NAMES`) keep PCA-model QPs stable.

## 4.8 Costs, execution and financing

* **Cost models:**
  * `PerChange(amount)`: a fixed ticket per position change;
  * `BuyOnly(model, min_trade_bp)`: cost on buys above a threshold, with a per-trade audit in
    `result.diagnostics["cost_audit"]`;
  * `DurationSpreadCost(spread_bp, duration, price)`: OASD x spread x price, where the spread may
    be an expression;
  * `TableCost(table, keys, buckets)`: rating x sector x size tables.
* **Breakeven.** `Session.breakeven(strategy, multipliers, baseline=)` gives the cost multiplier
  where the edge crosses zero.
* **Additive P&L.** `Metrics(pnl="ADDITIVE")`: returns summed, NAV = 1 + cumulative sum.
* **Shorts.** `Financing(short_rate="series")` funds shorts from a rate series.
* **Futures tick costs.** `Hedge(cost_per_contract=, contract_notional=)`.
* **Price tolerance.** `Execution(price_tolerance_bp=, price_field=)`: trades whose price moved
  beyond tolerance since the decision are skipped and counted.

## 4.9 Derivatives and overlays

* **Futures and CDS:**
  * `continuous_futures(...)`: rolls a set number of days before first notice, back- or
    ratio-adjusted;
  * `empirical_dv01(...)`: a rolling regression;
  * `cds_returns(spread_bp, risky_duration, roll_dates, coupon_bp)`: `-RD_{t-1} dS + c dt`, with
    zero spread change on IMM rolls and the premium as quarterly flows.
* **FX hedging programmes.** `fx_hedged_returns(..., ratio, roll)` (see 4.2).
* **Hedges inside the optimisation.** `Hedge(instruments, fields, joint=True, cost_bp=,
  max_gross=)` turns the overlay notionals into decision variables of the portfolio's `Optimise`.
  Its exposure limits count the overlays' exposure, each unit traded costs `cost_bp`, and the
  post-trade checks see the hedge. It works in Paper and Implement.

## 4.10 Metrics and reporting

`kernel.analytics`:

* drawdowns and capture:
  * `relative_drawdown`;
  * `capture`: up / down capture, Jensen alpha, beta, correlation, skew, appraisal ratio;
* period views:
  * `windows`: YTD, 1Y, 3Y, 5Y, since inception, OOS and trailing;
  * `period_table`;
* timing:
  * `timing_stats`: flips per year, time over / neutral / under, rolling Sharpe;
  * `breakdowns`: by regime, drawdown episode, whipsaw;
* trading:
  * `hit_rates`: on active return and from trades;
  * `window_stats`: per holding window;
  * `trade_blotter`;
* risk and attribution:
  * `ex_ante_risk`: TEV, contributions, marginal TEV, diversification, budget use, active carry;
  * `carry_spread_split`, `factor_contributions` and `attribution`;
* data and signals:
  * `data_coverage`;
  * `signal_extras`: IC uplift, Q5 - Q1 as mean, median and winsorised mean, monotonicity,
    equal-issuer weighting, neutralised outcomes, control matching, DTS-normalised returns;
* writers: `write_json` and `write_sql`, with as-of stamps and staleness flags.

`beta` is in `STAT_KEYS`.

## 4.11 Research and validation

`kernel.validation`:

* `walk_forward(session, builder, grid, splits, label_horizon, embargo)`: purged and embargoed,
  on the kernel;
* `event_splits`;
* `select_gated`: P&L-gated selection;
* `placebos`: shuffled, noise and anti signals;
* `preregistered`: subperiods, contamination, shadow verdicts;
* `label_periods`: IS / validation / OOS;
* `eligible` and `leaderboard`: sweep gates.

## 4.12 Reproducibility and provenance

* **Saved configs.** `Strategy.to_code()`, `Strategy.from_code(code, fingerprint)`,
  `strategy.save(path)` and `Strategy.load(path)` rebuild to the same fingerprint. Custom code
  refuses to be saved this way.
* **Trial records.** Each trial records its data version, code version, run quality
  (`Result.quality()`) and, with `Session(keep_inputs=True)`, its raw inputs.
* **Regime records.** `RunSettings(record_fields=(...))` stores the regime per rebalance.
* **Idempotent trials.** Trial ids are deterministic (no UUID), and
  `TrialLedger.rebuild_index()` rebuilds the index.

## 4.13 Engine parity

* **One result.** Implement fills the same `Result` (exposures, construction log, excess legs)
  and takes `forced_actions`.
* **Constant exposure.** `Strategy(hold="constant")` holds the traded weights between rebalances
  (fi-lib-research), without charging the implicit daily rebalance. It is Paper only: Implement
  holds face, so its weights drift.
* **Decisions from a kernel strategy.** `decide_at(strategy, source, as_of, current, nav)` runs
  one decision: the same universe, signal, construction, exits and overlay sizing a backtest
  runs. Proposals use it (`propose_from_strategy`).
* **Index scale.** Implement sizes its bond-days before building (about 2 KB each,
  `BYTES_PER_BOND_DAY`). A run that does not fit the memory available raises `MemoryError`,
  naming a sample, a shorter window or Paper. Implement is the validation tier.

## 4.14 Live proposals

See [operations.md](operations.md):

* sizing at bid / ask;
* rounding rules;
* minimum trades, caps and cash targets;
* check severities;
* headerless TSV and futures lines;
* PM feedback;
* external books in units or notional;
* T-1 analytics;
* fill reconciliation.
