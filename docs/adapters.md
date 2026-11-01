# Legacy framework adapters

Behaviour specific to one legacy framework stays out of the kernel's core (§5 of the gap
analysis). Each framework has an adapter in `nautilus_fi/adapters/`:

* its convention `PROFILE`;
* small named components built from kernel primitives;
* the `REQUIREMENTS` they cover;
* the `PARITY` cases that must reproduce the framework's own outputs.

`adapters.status()` lists them. Tests: `tests/test_adapters.py`.

**Every adapter is unconfirmed.** The components are built and tested on synthetic data. Parity
needs each framework's golden outputs and agreed tolerances (decision §9.2), and neither is in
this repository.

| order | adapter | framework | profile | parity target |
| --- | --- | --- | --- | --- |
| 1 | `fi_lib_backtesting` | fiq_backtester | `fi_lib_backtesting` | `audit/returns_audit.md` (US IG 2015-2025, US HY 2016-2025), EM corporates run |
| 2 | `erin` | fi-backtest-experimentation-erin | `erin_beta_plus` | canonical v2: 147HYC, 147IGC, 147SHY, JTJSHY |
| 3 | `mri` | fi-mdl-macrocredit (MRI v1 / v2) | `MRI_v1` | stored tables: US HY, US IG, EU IG, EU HY |
| 4 | `lta_ucits` | lta-ucits (THFA / TABC) | canonical | the fund's stored backtest |
| 5 | `fi_lib_research` | fi-lib-research | `fi_lib_research` | `tests/test_backtesting_engine.py` cases |
| 6 | `assetiq` | assetiq | canonical | stored allocations |
| - | `eur_ig_lgbm` | EUR IG LightGBM | `EUR_IG` | stored backtest |
| - | `composite` | new composite score / xrv | canonical | stored quintile tables |
| - | `credit_pairs` | credit-pairs | canonical | stored pair P&L |
| - | `globalrates` | globalrates | canonical | stored country returns |
| - | `systematicpm` | fi-lib-systematicpm | - | requirements not captured |

## What each adapter adds

* **erin.**
  * `stats_ru_benchmark`: the `_STATS` / `RU` pair.
  * `start_date_field` and `guard_duplicate_dates`.
  * `orphan_refill`: bad dates (at least 80% exactly-zero returns) refilled from the next orphan
    date within 4 days. Give its frame to `under("erin_beta_plus", refill=...)`; a bad date with
    nothing to refill from raises.
  * `composite_score_2`: Composite Score 2.0 on 0-100. With two or more null components it
    carries the last valid score within 7 days, else 19. No row scores 50. It has a standalone
    mode.
  * `mri_ratio` / `mri_dts_band`: a clamped piecewise-linear target ratio. Staleness is a hard
    error, overrides are dated, and the band changes by date.
  * `parse_filter` / `filters` / `filter_audit`: SELL, EXCLUDE and TICKER_UNDERWEIGHT rules with
    `@bench.X` and `@universe.X.pN`, audited per rebalance.
  * `analyst_scores`: decoded from -2 to +2, with 14-day staleness.
  * `lp_construction`: a soft group-deviation penalty, IRL1 sparsity and a one-way-buys turnover
    tier ladder.
  * `schedule` and `event_cap`.
  * `costs`: buy-only OASD x 3 / 10bp, with a 1bp threshold, audited.
  * `credit_futures`, `treasury_waterfall` and `live_proposal`, which writes the headerless QOE
    file.
  * `apply_tuning` and `eligible`.
* **fi_lib_backtesting.**
  * `benchmark` and `replication`: drift between reconstitutions, ±20bp/yr against published
    returns.
  * `constraint(metric, "absolute" | "ratio" | "offset", lo, hi, by, min_benchmark_weight)` and
    `group_limit`.
  * `turnover_budget` and `relaxation`, which can push to the next day.
  * `rules` and `score_drop`, with interim rebalances.
  * Futures and CDS returns, `fx_hedged`, and `external_book`.
* **mri.**
  * `PriceAdjustCost` / `cost(universe)`: 5 / 20 / 10 / 25bp as a signed execution-price
    adjustment, with a log-difference mode. The mapping to universes is to confirm.
  * `additive_metrics`.
  * `v2_signal`: the OR rule, inputs carried at most 5 calendar days.
  * `decimal_guard`.
  * `WINDOW_PRESETS` are to be filled from the v2 config.
* **fi_lib_research.**
  * `engine`: constant exposure, additive P&L and a cost per change.
  * `input_field`: the input kinds.
  * `align`: the alignment modes.
  * `validation_gate`.
* **eur_ig_lgbm.**
  * `symmetric_tilt`: 1 ± k s, clipped and stepped.
  * `oof_calibrate`: out-of-fold Platt, centred at 0.5.
  * `seed_ensemble` and `fit_blend`: training rows only.
  * `timing_metrics`.
* **composite.**
  * `forward_excess` and `investible`.
  * `quintile_spread`: mean, median, winsorised, monotonicity, equal-issuer.
  * `nw_tstats`: at 13 and 26 lags.
  * `bootstrap` (moving block) and `acceptance` (pre-registered).
* **assetiq.**
  * `to_factor` / `from_factor`: x = w x DTS.
  * `objective`: MaxSharpe or MaxIR.
  * `tev_band`: the frontier searched between the floor and the budget.
  * `hierarchy` and `tilt_bands`.
  * `kde_copula_scenarios` / `KDECopulaScenarios`, for `CVaR`.
  * `risk_report`.
* **lta_ucits.**
  * `two_stage` and `min_trade`.
  * `issuer_limit`: min(4%, 3 x index) [?].
  * `banks_at_index`.
  * `esg_screen`: eligible / grandfathered / excluded.
  * `grandfathered`, `flow_construction`, `turnover_budget` (index plus allowance) and
    `common_start`.
  * `paired_t`.
  * `tabc_hedge`: Bund futures, TE ≤ 0.85% [?].
* **credit_pairs.**
  * `pair_pnl`, `imm_dates` and `zscore_rules`.
  * `dv01_neutral` and `etf_shares`.
  * `adf`, with MacKinnon p-values, and `bh_fdr`.
* **globalrates.**
  * `constant_maturity`, `carry_rolldown` and `fx_hedged`.
  * `country_calendar`: US bond market, TARGET and UK calendars.

## Confirming an adapter

1. **Load the golden output.** `load_golden(path)` returns the output with its content hash.
2. **Run the case.** `run_parity(case, source, golden, sha)` builds the strategy, runs the case's
   window and compares: returns as an annualised difference in bp and a correlation, statistics,
   weights, or table columns.
3. **Collect the evidence.** `profile_confirmation(profile, reports, required=[...])` raises
   unless every required case passed on a hashed golden file, and returns the evidence.
4. **Confirm by review.** Commit the evidence with the change that sets `confirmed=True` on the
   profile in `domain/conventions.py`. The harness never sets it: a profile is confirmed by a
   reviewed change, not by a file a run leaves behind.

The tolerances in each `PARITY` list are proposals until §9.2 agrees them.
