# Gap analysis: status

Where each item of the 2026-10-30 gap analysis (revision `10dd477`) stands. Every fix has a
regression test in `tests/test_gap_analysis.py` that fails on `10dd477`. The test names start with the
item's ID.

Status values:

* **fixed**: the required control is in place and tested.
* **partial**: the generic part is done; what is missing is stated.
* **not reproduced**: the code already does what the analysis asked; the evidence is given.
* **open**: not started. The note says what it needs.

## P0: correctness (D1–D10)

| ID | Status | What changed |
| --- | --- | --- |
| D1 | fixed | `EvalContext.realised(name)` reads returns, excess-return legs and refills without the availability lag. It is used by the engine, benchmarks (`returns`, drift), Implement contributions and the legacy Paper engine. Benchmark weights from a lagged weight field are lagged only for decisions (`Benchmark.known_weights`). |
| D2 | fixed | A cash band means `1 - sum(w)` in `[min, max]`, and `min` defaults to 0 unless an explicit `Budget` or `Leverage` constraint allows more than NAV. The optimiser and the post-check (`grammar.cash_floor`) agree. |
| D3 | fixed | `annualised_excess` (`domain.conventions`) gives every engine's statistics: VENDOR compounds the vendor series, COMPOUNDED annualises each leg and differences them, GEOMETRIC uses their ratio, DAILY_SUM the summed differences. `Result.excess(freq)` aggregates the legs through `compound_excess`. |
| D4 | fixed | Every row of the table: benchmark gaps, cash and risk-free gaps, off-calendar series (compounded or carried to the next trading day), returns that stop without an exit (missing data unless an `alive` field or a maturity / call / exit date explains them), percent-scale returns (`DataQualityError`), NAV at zero (`InsolvencyError`), `HalfSpread` fallbacks (counted), duplicate rows and conflicting panels (raise), `DropDates` (removed from portfolio, benchmark and statistics in both engines, which agree), and evaluation windows with gaps (excluded and counted). `fill_count` covers every fill, and the gate refuses any. |
| D5 | fixed | `Strategy(exits="SELL" \| "EXCLUDE" \| "HOLD_NOT_BUY")`, recorded in the conventions. SELL forces leavers to zero outside the turnover budget and reports `forced_sells`; EXCLUDE sells them first within the budget; HOLD_NOT_BUY keeps them. These rules apply in both engines and in Implement. An empty universe, or a signal with no values, keeps the book and logs `aborted`. |
| D6 | fixed | Relaxing an infeasible problem records which limits were softened. The rebalance logs `rebalance_relaxed` or `infeasible_kept`, `stats()` counts them, and promotion refuses them. `OptimiserSettings(on_infeasible="fail")` raises. *The body of D6 is not in the photos; the fix follows its heading and §4.7.* |
| D7 | fixed | Reconstitution uses what exists at its close. As-of reads are strictly before the cutoff (`field_store.AS_OF_INCLUSIVE = False`, overridable per read). The walk-forward purge adds the execution lag, and `fit()` sees only data before the window it is scored on. `ffill()` and `cumsum()` load all history. `Explicit` defaults to `on_or_after`. Panels use one trading calendar (other dates snap to the next trading day). Turning off look-ahead checks is recorded and refused at promotion. Deciders are look-ahead checked on truncated panels. |
| D8 | fixed | Results are cached under the fingerprint, data version, window, fidelity and the settings that change numbers (`settings_key`). Unversioned sources are cached in memory only. Trials record the data version and the universe, and the data version is part of a trial's identity. Unnamed persistent sessions are the root's `default` thread. |
| D9 | fixed | `erin_beta_plus`: the bad-date threshold is 80% exactly-zero returns (`zeros_only`), and `under(..., refill=)` names the refill field (it no longer fails with "no None"). Published daily weights are used as published (STATIC freezes only constituent files). `under()` sets the drift on derived and blended benchmarks, and `conventions()` reports it. START_DATE-labelled returns are declared with `FieldInfo(label="start")` and realise one period later. Erin's refill from the next orphan date within 4 days is `adapters.erin.orphan_refill`; its frame is the refill field. Under `Refill`, a bad date with nothing to refill from now raises instead of keeping its zeros, and partly refilled dates are reported (`bad_dates_partly_refilled`). *Confirming the profile needs Erin's golden outputs (§5).* |
| D10 | fixed | Turnover statistics exclude the build from cash (`initial_build_turnover`). `round_face` gives the minimum piece plus whole increments. Proposals size at the traded side and round once, and report targets below the minimum piece. Missing or NaN minimum pieces and increments take explicit defaults that are recorded and warned. The first-day accrual uses actual days. Carry includes roll-down at constant yield, with the dead terms removed. **Implement turnover and the factor: not reproduced.** Fills record current face (`face = orig_face * factor`, `accounting/ledger.py`), so `face × clean / 100` is already the traded value. |

## P0: governance (G1–G14)

| ID | Status | What changed |
| --- | --- | --- |
| G1 | fixed | DSR and PBO cover the search, `TrialLedger.scope(universe, window, data_version)`. That means every counted trial on the same universe, an overlapping window and the same data, whatever hypothesis it was filed under, or none. Relabelled copies with identical returns count once. |
| G2 | fixed | Confirmation is computed from the values (`unconfirmed_profile`). Mechanics carrying an unconfirmed profile's distinguishing conventions are unconfirmed under any name. |
| G3 | fixed | `register_new` stamps the registration time and freezes the content by hash. The gate checks `hypothesis_frozen`. |
| G4 | fixed | OFFICIAL needs the Implement run's fingerprint to equal the Paper one, a reconciliation with its required fields, and a daily correlation and residual within tolerance. Implement manifests now record the spec they ran (`manifest.spec["hash"]`). Within a trial the Paper record is used for the counts and statistics, so an Implement run checks the trial rather than adding a draw. |
| G5 | fixed | `RecordRegistry(min_dsr=0.95, max_pbo=0.5, min_years=1.0)` is enforced by default; PBO is gated from four distinct trials, with CSCV blocks of at least 40 observations. `promote(..., min_dsr=None)` records a value without gating it. **The values are placeholders for §9.5.** |
| G6 | fixed | The holdout gate checks the trials, and a second look by the same hypothesis raises (the registry already did that). Trial records are now idempotent (§4.12), but re-running a trial that viewed the holdout still raises: it is a second look, not a duplicate write. |
| G7 | fixed | One record per (strategy, data, window) and level (`result_key`). |
| G8 | fixed | The chain head goes in an anchor file (`anchor_path`, which can live on separate storage), and appends happen under the lock. A deleted tail fails `verify_chain()`, and `submit` refuses to append to a chain that does not verify. `adopt_chain()` anchors a pre-existing chain after an audit. |
| G9 | fixed | Registration checks and writes in one locked step. Six concurrent writers give one success and five refusals. |
| G10 | fixed | `Session.run` refuses an unregistered hypothesis before running. |
| G11 | fixed | An unreadable trial file raises `LedgerError`. Only in-flight temporary files are skipped. |
| G12 | fixed | Failure criteria must parse as `<statistic> <op> <number>` joined by `or`. They are evaluated at promotion, and text that cannot be checked fails the gate. |
| G13 | fixed | One definition of a trial: a distinct (strategy, data version) at PAPER, VALIDATION or IMPLEMENT stage, outside the holdout. `n_trials`, the DSR and PBO all use it. |
| G14 | fixed | `DataSource.point_in_time()`: synthetic data and `FieldStoreSource(mode="first")` are True, `mode="latest"` is False, and in-memory, SQL and callable sources are whatever they declare. OFFICIAL requires True. Legacy Paper results, which carry no source, take the caller's evidence (`submit(..., point_in_time=True)`). |

## Section 7: documentation claims

| Claim | Status |
| --- | --- |
| A missing half-spread is an error | true now (D4); fallbacks are counted |
| COMPOUNDED is the canonical excess return | true now (D3) |
| DROP_DATE removes bad dates | true now, in both engines (D4) |
| Deduplication across restarts | true for a session name; unnamed persistent sessions use `default` (D8); `docs/kernel.md` says which |
| `under(profile)` applies drift and rebalance rules | docs corrected: drift on every index, schedule unchanged, cash per profile |
| Promotion thresholds and holdout | gates rewritten (G1–G14); `docs/kernel.md` lists them |
| Live proposals use bid/ask dirty pricing | true now: sizing and pricing at the traded side (D10) |
| "PORT / PREP" covered | renamed `write_pretrade_report`; docs say Bloomberg PORT / PREP reconciliation is not covered |
| One `Result` schema across fidelities | Implement now fills exposures, the construction log and excess legs (§4.13) |
| The kernel reproduces the Paper engine exactly | true now, including `Triggered(field=...)` and `DropDates` (tests) |
| Every expression is point in time | true now (D1); docs separate availability from realisation |
| `Triggered` measures one-way turnover | `Triggered(turnover=...)` names the definition, HALF_SUM by default |
| Carry includes roll-down | true now (D10) |
| Scale figures from probe scripts | the scripts are in `tools/probes/` |
| Developed-world hedged sleeve | renamed `DEVELOPED_WORLD_UNHEDGED` (spot conversion, no forward hedge) |
| Benchmark / index rows VALIDATED | downgraded to SUPPORTED_WITH_APPROXIMATION (circular synthetic check); `docs/support_matrix.md` regenerated by `tools/support_matrix.py` |
| family1 "per-change costs" | wording corrected: bp of traded value |

## Section 4: platform primitives

All buildable rows are done. [primitives.md](primitives.md) is the reference. The tests are in
`tests/test_primitives.py`, named by section, and §4.14 is also in [operations.md](operations.md).

| Section | Status | What is in place |
| --- | --- | --- |
| 4.1 Returns and data contracts | fixed | Declared forms (`FieldInfo(form=)`), bond TR from prices with coupons and write-offs, MTD de-cumulation, leg alignment modes, returns from expressions, resampling, bond-level forward returns with flags, enforced availability windows with `Available`, dated categories. |
| 4.2 Return and valuation models | fixed | `SpreadReturn`, duration / convexity sleeves, constant-maturity returns with carry and roll-down, FX-hedged returns with hedge programmes (`ratio`, `roll`), and `analytics.attribution`. Its legs (rates, carry, spread, FX, residual, cash, financing, overlays, costs) add up exactly, by group and active. |
| 4.3 Timing and schedules | fixed | `NextClose(0)` refused, calendar-day `ffill(days=)`, yearly / biweekly / n-th business day / anchored weekly calendars, `IndexDates`, `WithInterim` with a score-drop trigger, pandas names (the old ones warn), `AnnualTurnoverBudget`, lag stress and its gate. |
| 4.4 Universe and benchmark | fixed | The seven index mappings in the FIAtlas template (to confirm), `PairedIndex`, cash or pro-rata redemptions, `replication_report` against published returns, MV reweighting on its own schedule, `skip_below`, `BenchmarkStat` and `cs_quantile` filters, and a tested investible filter. *Real replication needs the provider's returns.* |
| 4.5 Signal handling | not captured | The section's body is not in the photos. |
| 4.6 Construction methods | fixed | `Stratified`, quantile buffers and kept weights, spread-unit pairs, `ZScoreRule`, `Rules` with precedence, hedge tolerance / lots / tick costs, `KRDWaterfall`, factor-unit bounds. |
| 4.7 Optimiser | fixed | TE limits (bisection or cuts), `TargetTE`, `MaxIR` / `MaxSharpe`, `MinTrade`, tie-break stage, `Bounds` / `Pin` / `NoSell` / `NoBuy` / `TradeLimit` (capacity), bounds that change by date, subsets, relative caps, `CurrencyExposure`, the recorded relaxation ladder with strict limits, conflicts and push, forced exits outside the budget, `IndexTurnover`, `Sparse`. The dense risk blocks also fixed a latent QP failure with PCA models. |
| 4.8 Costs, execution and financing | fixed | `PerChange`, `BuyOnly` with an audit, `DurationSpreadCost`, `TableCost`, `Session.breakeven`, additive P&L, short funding from a series, futures tick costs, Paper price tolerance. |
| 4.9 Derivatives and overlays | fixed | Futures rolls and continuous series, empirical DV01, CDS / CDX returns with IMM rolls, FX hedge programmes, and multi-currency universes run in the base currency (per-name hedged or unhedged returns plus `CurrencyExposure`). `Hedge(joint=True)` solves the hedge with the book in Paper and Implement. |
| 4.10 Metrics and reporting | fixed | `kernel.analytics`: relative drawdown, capture, windows, period tables, timing statistics, breakdowns, hit rates, per-window statistics, ex-ante risk, carry / spread split, factor contributions, attribution, blotter, coverage, signal extras, JSON / SQL writers. |
| 4.11 Research and validation | fixed | `kernel.validation`: kernel walk-forward (purged, embargoed), event splits, P&L-gated selection, placebos, pre-registered evaluation, labelled periods, sweep gates and leaderboard. |
| 4.12 Reproducibility | fixed | `to_code` / `from_code` / `save` / `load` to the same fingerprint, data and code versions, run quality and raw inputs per trial, regime records, deterministic trial ids with `rebuild_index`. |
| 4.13 Engine parity | fixed | One result schema, forced actions in Implement, `Strategy(hold="constant")`, `decide_at` / `propose_from_strategy`, the Implement memory pre-flight (`MemoryError` naming a sample, a shorter window or Paper). `Triggered(field=)` already matched Paper. |
| 4.14 Live proposals | fixed | Bid / ask sizing, rounding rules, minimum trades (bps of NAV and face), caps, cash target and bounds, severities (blocker / approval_required / warning), MV reconciliation, signal and analytics coverage, futures lines and readiness, headerless TSV, PM feedback, cash-flow mode, external books in units / notional / market value, T-1 analytics, fill reconciliation. |

## Section 5: legacy adapters

Built, not confirmed. [adapters.md](adapters.md) describes each one. `nautilus_fi/adapters/` has
an adapter per framework, made of named components over the kernel. They cover the items in
§5.1–§5.9 that could be read. Tests on synthetic data are in `tests/test_adapters.py`.

A parity harness (`adapters.parity`) compares a run with the framework's golden output, and
`profile_confirmation` produces the evidence for a reviewed change to `confirmed=True`. **No
profile is confirmed.** That needs each framework's golden outputs (none are in this repository)
and the tolerances of §9.2. `systematicpm` has no adapter, because its requirements were not
captured.

## Section 6: operational scope

The library side is done (see [operations.md](operations.md)):

* the stable target-book export (`target_book`, versioned schema, content hash);
* the reconciliation interface (`reconcile_book` with a named gold copy, adjustment rows left
  out and zero-against-nonzero blocks; `reconcile_returns` in bp; `reconcile_fills`);
* a `BookStore` protocol with a file implementation;
* official-record `links` to the operational store, with the strategy's config and per-name
  versions.

**Open, as decision §9.1:** whether bt-research also runs the service (versioned configs per
period, persisted positions and overrides, restartable runs, PORT BBU exports) or hands that to
paperport / systematicpm through this interface.

## Section 9: decisions still yours

* **9.1 Scope:** library only, or the operational paper-portfolio service as well.
* **9.2 Parity:** golden outputs per framework and the tolerances they must meet. The adapters'
  `PARITY` lists propose some.
* **9.4–9.5 Defaults chosen here, for review:**
  * as-of reads strictly before the cutoff;
  * exits SELL;
  * promotion thresholds DSR 0.95, PBO 0.5 (from four trials) and a one-year track record;
  * lag stress not gated unless set;
  * reconciliation tolerances: correlation 0.8, residual 2%;
  * proposal severities as in `propose.live`.
* **Readings to confirm:** values the review gave from photos, marked [?] or "to confirm" in the
  code. They include the MRI cost-to-universe mapping, the lta-ucits issuer limit and TABC TE
  cap, the Erin anchored-weekly offsets, the FIAtlas index codes and the QOE column layout.
