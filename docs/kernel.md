# The research kernel

The kernel (`nautilus_fi.kernel`, re-exported as `import nautilus_fi as nfi`) is the ground the
platform works up from. A strategy is a set of Python objects, and those objects decide what the
engines do: no configuration files, no switch statements over settings. The intended user is a
research agent running in the cloud next to the company data platform, answering a PM's
question: "if I change my strategy this way, how would it perform, and why?"

```python
import nautilus_fi as nfi

nfi.session("~/research/ig-value", source=company_data)

base = nfi.Strategy(
    "ig_value",
    universe=nfi.Index("LUACTRUU") & (nfi.Field("rating_score") <= 10) & (nfi.Field("oasd") > 1),
    signal=nfi.Field("oas_bp").zscore(by="sector").neutralise("oasd"),
    portfolio=nfi.Optimise(1e-3 * nfi.Alpha() - 2e-3 * nfi.TrackingError(by=["sector"]) - nfi.Costs(),
                           nfi.ActiveWeight("issuer") <= 0.005, nfi.ActiveWeight("sector").within(0.02),
                           nfi.ActiveExposure("oasd").within(0.10), nfi.Turnover() <= 0.10),
    benchmark=nfi.Index("LUACTRUU").exclude(sector="FIN").cap(issuer=0.03),
    schedule=nfi.MonthEnd(),
    execution=nfi.NextClose(lag=1, cost=nfi.HalfSpread() * 1.5 + nfi.SqrtImpact(10)),
    financing=nfi.Cash("SOFR"))

variant = base.with_(portfolio=base.portfolio.replace(nfi.Turnover() <= 0.05), name="turnover_5")
cmp = nfi.compare(base, variant, window=("2010", "2024"))
print(cmp.explain())
```

`examples/agent_workflow.py` runs this loop end to end on a synthetic warehouse in about 9 seconds:

1. five PM questions, compared and explained;
2. a sweep with selection statistics;
3. a signal evaluation;
4. the chosen variant through the event engine, with the reconciliation;
5. a record submission.

## Layers

| Layer | Objects | Module |
| --- | --- | --- |
| Data seam | `DataSource`, `DataRequest`, `Catalog`, `FieldInfo`; `PanelSource`, `FrameSource`, `CallableSource`, `SQLSource`, `FieldStoreSource`, `SyntheticSource`, `CompositeSource`, `MappedSource`, `CachedSource` | `kernel.data` |
| Expressions | `Field`, `Attribute`, `TimeSeries`, arithmetic, predicates, time-series and cross-sectional transforms, `combine`, `where`, `apply`, `cross_sectional`, `check_lookahead` | `kernel.expr` |
| Universe and benchmarks | `Index` (members and weights), `.exclude/.only/.where/.cap/.reweight`, `Blend`, `ReturnSeries`, `All` | `kernel.benchmark` |
| Construction | `EqualWeight`, `TopN`, `Quantile`, `RankWeighted`, `BenchmarkWeight`, `Tilt`, `Hold`, `Pairs`; weightings; `.cap/.subject_to/.checked`; `Optimise` with terms and constraints; `Hedge` | `kernel.portfolio` |
| Mechanics | `NextClose`/`SameClose`, `Participation`, cost algebra (`HalfSpread`, `FixedCost`, `SqrtImpact`, `FieldCost`), `Cash`/`NoCash`, `ZeroFill`/`DropDates`/`Refill`/`Fail`, excess-return methods, turnover definitions, annualisation, `Metrics` | `kernel.mechanics` |
| Strategy | `Strategy` (immutable, `with_`, `set`, `under`, `diff`, `fingerprint`, `describe`, `requirements`, `validate`) | `kernel.strategy` |
| Engines | `prepare` + `simulate` (Paper, vectorised); `run_implement` (Implement, the event engine); one `Result` schema | `kernel.engine`, `kernel.implement`, `kernel.result` |
| Research threads | `Session` (trial ledger, result cache, shared data, sweeps, DSR / PBO, `promote`) | `kernel.session` |
| Agent API | `session`, `catalog`, `run`, `compare`, `sweep`, `evaluate`, `explain`, `promote`, `vocabulary` | `kernel.api` |

## Plugging in the data platform

The contract is small: `catalog()` (what exists) and `load(DataRequest) -> Panel` (fields for
dates and instruments). Static attributes, date series, index constituents, overlay instruments, a
`version()` and an event-engine `environment()` are optional. An agent's data-access tool needs no
class at all:

```python
def data_tool(field, start, end, ids):            # the agent's tool: one query per field and window
    return warehouse.query(field, start, end, ids)   # long (date, instrument_id, value) or wide (dates x ids)

src = nfi.CachedSource(
    nfi.CallableSource(data_tool, nfi.fields_from_dictionary(warehouse.data_dictionary()),
                       attributes=warehouse.reference(), series={"SOFR": warehouse.series("SOFR")},
                       constituents=lambda index, start, end: warehouse.index_members(index, end)),
    root="~/.nfi-cache")
```

* **Availability.** `FieldInfo.lag` is how many trading days after its date a value may be used.
  Vendor analytics published the next morning have lag 1. `Field` applies the lag when it reads,
  so every expression is point in time by construction; `Field("x", lag=0)` on a lag-1 field
  raises `LookaheadError`. The lag governs decisions only: realised returns are read without it
  (see "Data: what is known, what is realised" below). `FieldStoreSource` reads true knowledge
  times: a request's `as_of` gives the latest snapshot known strictly before it, and
  `mode="first"` gives what a live process saw before revisions (the only mode that counts as
  point-in-time for official records).
* **Discovery.** `catalog().search("spread duration")` matches names, descriptions and tags.
  An unknown name raises `UnknownFieldError` with close matches. `fields_from_dictionary(df)`
  turns the platform's data dictionary into the catalogue.
* **Years of data.** `CachedSource` keeps one block per field covering the widest window
  requested; narrower windows are views. With a `root`, blocks are `.npy` files read
  memory-mapped, so many processes share one copy in the page cache. A warehouse that cannot
  version its data gets a snapshot id pinned on first use; results stay reproducible against the
  cache until `refresh()`. `FieldStore` reads push date and entity filters into the Parquet scan.
* **Several systems.** `CompositeSource(analytics, prices, reference)` routes each field to the
  source that has it. `MappedSource(src, cusip_to_isin)` maps identifiers and reports what did
  not map.

## Expressions

An expression is a graph. The engine evaluates it once over the whole dates × instruments panel
with numpy, caches every node by its structural key, and hands each rebalance its row. Variants
that share a universe, benchmark or signal compute it once.

* **Time series** (look back only): `lag`, `diff`, `pct_change`, `rolling(n).mean/std/sum/min/max/median/zscore`, `ewm(hl).mean/std`, `ffill`, `cumsum`.
* **Cross-sectional** (per date, within the strategy's universe, within groups with `by=`): `rank`, `zscore`, `demean`, `winsor`, `neutralise(*exposures, by=)` (residual of a per-date regression; groups by demeaning), `bucket(edges | n=)`, `top(n)`, `scale`.
* **Groups** (`by=`): an attribute (`"sector"`), a list, or a category expression (`Field("rating_score").bucket([4, 7, 10])`).
* **Predicates**: comparisons, `isin`, `between`, `notna`, `&`, `|`, `~`. A missing value never satisfies a predicate.
* **Custom Python**: `expr.apply(fn)` (vectorised arrays) and `expr.cs(fn)` (one date's cross-section as a Series). `cs` runs only on decision dates unless a time-series transform sits above it.

**Look-ahead is checked two ways.**

* *Structurally:* `lookahead() > 0` means a lead or negative lag, and the strategy is refused.
* *Empirically, for custom code:* `check_lookahead` re-evaluates the expression on panels
  truncated at three dates. Any value that changes when later data are removed used later data.
  Full-sample demeaning inside `apply` is caught this way. The verdict is cached per expression.

## Construction and the optimiser

* **Path-independent methods** (`EqualWeight`, `TopN` without a buffer, `Quantile`,
  `RankWeighted`, `BenchmarkWeight`, `Tilt`, `Hold`) are computed for every rebalance date in one
  vectorised pass.
* **Path-dependent methods** run per date on a `KernelCross` cross-section: the optimiser, buffers,
  `subject_to` projections, deciders and triggered schedules. A `KernelCross` can still evaluate
  any kernel expression at its date. A test asserts that both paths give identical results.

`Optimise(objective, *constraints)` compiles to the HiGHS model of `construct.optimiser`
(LP / MILP / QP, `lp_round` for large mixed-integer problems, relaxation when infeasible).

* **Objective.** It is built from terms with numbers, `+` and `-`.
  * Linear terms take any sign: `Alpha()` (or `Alpha(expr)`) and `Linear(expr)`.
  * Penalties need negative coefficients: `Costs()`, `TrackingError(by, fields)`, `ActiveShare()`, `Risk(model)` (QP; `PCARisk(window)` estimates a factor model from history) and `CVaR(beta)`.
* **Constraints.** These are comparisons on measures: `Weight`, `ActiveWeight(group)`, `GroupWeight`, `Exposure`, `ActiveExposure(relative=)`, `Turnover(definition)`, `Leverage`, `Budget`, `CashWeight`, `Names`, `CVaRLimit`, `MinHolding`, and `LinearMeasure(expr)` for custom constraints.
  * `.soft(penalty)` makes a constraint soft.
  * `opt.replace(Turnover() <= 0.05)` swaps the constraint in the same slot; `.add`, `.without(Turnover)`, `.with_objective`, `.with_settings` edit the rest.

Every rebalance logs the method's diagnostics, which constraints bound, and any violations. A
heuristic method is post-checked against `.checked(...)` constraints; `.subject_to(...)` projects
it onto them (nearest feasible portfolio in L1).

More constraints and constructions are in [primitives.md](primitives.md):

* tracking-error limits (`TEV(model) <= x`, `TargetTE`, `MaxIR`, `MaxSharpe`);
* `MinTrade` and a tie-break stage;
* per-name `Bounds`, `Pin`, `NoSell`, `NoBuy` and `TradeLimit`;
* bounds that change by date, subsets (`.where`) and relative caps (`GroupWeight.cap`);
* the recorded relaxation ladder, and `Sparse`;
* `Stratified`, `ZScoreRule` and `Rules`.

`Hedge(..., joint=True)` makes the overlays decision variables of the optimiser, so its exposure
limits count the hedge.

## Mechanics and conventions

The named conventions of the legacy frameworks (`domain.conventions`) are presets of these
objects:

* `strategy.under("MRI_v1")` re-runs a strategy with that framework's execution timing,
  missing-return, excess-return, turnover and annualisation rules, the benchmark weight drift
  (applied to every `Index` in the benchmark, derived and blended ones included), and zero cash
  when the profile says cash earns nothing. The schedule, costs, universe, signal and portfolio stay
  as they are: rebalancing is a strategy choice, not a convention. A profile that refills bad dates
  (`erin_beta_plus`) needs the refill field: `under("erin_beta_plus", refill="tr_vendor_b")`.
* `strategy.conventions()` is the record that manifests, the ledger and the official gates read.
  It is computed from the values: mechanics that carry an unconfirmed legacy profile's
  distinguishing conventions are unconfirmed whatever the strategy is called, so
  `under("paperport").with_(profile="kernel")` is still unconfirmed.
* Profiles still marked `confirmed=False` cannot be promoted to CANDIDATE.
* `Strategy(hold="constant")` holds the traded weights between rebalances instead of letting them
  drift (fi-lib-research's constant exposure), at Paper fidelity.

## Data: what is known, what is realised

* **Availability vs realisation.** A field's `lag` is when a value may inform a decision:
  `Field("oas_bp")` on row *t* reads the value published by *t*. Returns are outcomes: the engine
  reads the return field, the excess-return legs and refills *realised* (unlagged, aligned to the
  period they cover), so declaring `tr` with `lag=1` changes what a momentum signal sees, not the
  P&L. A field stamped with the start of its period declares `FieldInfo(..., label="start")`.
* **One trading calendar.** A panel's rows are the return field's dates; a field or date series
  dated between them (a Saturday month end) moves to the next trading day, compounded for returns
  and carried for levels, and the move is counted (`missing["snapped_observations"]`).
* **Missing data never fills silently.** The strategy's `MissingReturns` applies to held
  instruments, benchmark members, the cash series and the risk-free series alike; under `Fail()`
  (the default) a gap stops the run. A held instrument whose returns just stop is missing data unless
  the source says it exits (an `alive` field, or a maturity / call / exit date). `DropDates(threshold)`
  removes bad dates from the run (portfolio, benchmark and statistics; cash accrues over them).
  Every fill is counted in `result.missing` and `stats()["data_fills"]`, and the promotion gate
  refuses any. `HalfSpread` raises for a traded name with no spread unless given `fallback_bp`.
  Duplicate `(date, id)` rows and two panels disagreeing on a field raise; returns whose median
  absolute value looks like percentages raise; a NAV that reaches zero stops the run.
* **Exits and outages.** `Strategy(exits="SELL")` (default) sells a held name that leaves the
  universe at the next rebalance, outside any turnover budget, and reports it as a forced sale;
  `"EXCLUDE"` lets it only shrink (sold first within the budget); `"HOLD_NOT_BUY"` keeps it without
  buying more. A rebalance whose universe is empty, or whose signal has no value for any name in
  it, keeps the book and logs `aborted`.
* **Look-ahead.** Expressions are checked structurally and, for custom code, on truncated panels;
  custom deciders are checked on truncated panels too. Unbounded look-backs (`ffill()`,
  `cumsum()`) load all history. Knowledge-time reads are strictly before the cutoff
  (`data.field_store.AS_OF_INCLUSIVE`). `RunSettings(check_lookahead=False)` is recorded and the
  result cannot be promoted.

## Strategy identity

`Strategy` is frozen.

* **Editing:**
  * `with_(**changes)` returns a changed copy;
  * `set("execution.lag", 2)` changes a nested value by path, which is what sweeps use;
  * `diff(other)` lists what changed, component by component.
* **`fingerprint`** hashes the canonical description of every component, including:
  * the source code (or bytecode), closure and defaults of custom functions;
  * the source of decider classes;
  * the kernel version.

  Results cache under it, so editing a lambda never serves a stale result.
* **Reading and checking:**
  * `describe()` prints the strategy as code and prose;
  * `requirements(catalog)` lists the data it needs;
  * `validate(source)` reports missing data, look-ahead and inconsistent parts before a long run.
* **Escape hatch.** `Strategy(decider=MyDecider(), fields=(...), lookback=n)` runs any `Decider`
  (`decide(view) -> Target`). `Strategy.from_spec(research_spec)` runs a legacy `ResearchSpec`.
  The kernel reproduces the Paper engine's returns for it exactly, including under `DropDates`
  and `Triggered(field=...)`; tests assert this.

## Fidelities and results

* **Paper fidelity.** `simulate` runs vectorised expressions, vectorised targets where possible,
  and one loop over days for drift, costs, financing, overlays, participation limits and missing
  returns.
* **Implement fidelity.** `run_implement` runs the same strategy through the Nautilus event
  engine: RFQ execution, bid/ask dirty fills, minimum pieces, dated settlement, funding and the
  ledger. The universe and signal reach it point in time, and the kernel construction runs on the
  event engine's cross-section. `reconcile(paper, implement, source=...)` decomposes the gap into
  holdings gap, cash, execution and residual.

Both return a `Result` with one schema.

* **Series:** returns, gross, costs, NAV, benchmark returns, daily exposures, rebalances with
  turnover under all three definitions, weights, and the excess-return legs (Treasury and vendor).
* **Breakdowns:** contributions by instrument and by group (with benchmark weights and
  contributions), the construction log and violations. Implement builds exposures, groups,
  contributions and excess legs from the event engine's holdings, and its construction log from
  the event strategy's decisions.
* **Statistics:** `stats()` comes from one function (`Metrics.stats`), so Paper and Implement
  report the same keys on the same basis. Excess returns aggregate under the stated method
  (`COMPOUNDED`: each leg compounded, then differenced; `DAILY_SUM` only when a profile asks for it).
  Turnover statistics exclude the initial build from cash (reported as `initial_build_turnover`).
  Implement adds `implement_*` keys such as fill rate, spread cost, delay drift and solvency.

`compare(...)` lines results up. `explain(variant, base)` reports:

1. the strategy diff;
2. the headline table;
3. whether the return difference is distinguishable from noise (Newey-West t-statistic and a
   block-bootstrap interval);
4. a per-year arithmetic decomposition into trading costs, holdings by group and by name, and
   cash and financing;
5. average exposure and group-weight differences;
6. which constraints bound and how often;
7. the difference by calendar year;
8. a note when a change altered nothing.

## Live decisions and proposals

`decide_at(strategy, source, as_of, current, nav)` runs one decision from a live book: the same
universe, signal, construction, exit policy and overlay sizing as a backtest.
`propose.propose_from_strategy` turns it into sized, checked orders with severities and futures
lines. It also handles PM feedback and fills. `propose.target_book` and `reconcile_book` are the
interface to an operational store. See [operations.md](operations.md).

## Sessions and governance

* **Trials.** Every new run in a `Session` is a trial in the ledger. Each trial is one atomic file
  carrying the session, fingerprint, description, statistics and returns, so many agents can write
  at once. The hypothesis registry merges under a lock.
* **Deduplication.** Within a session name, the same strategy on the same window and data version
  is one trial, however often it runs; a persistent session without a name is its root's `default`
  thread, so restarts deduplicate. Results are cached under the strategy, the data version, the
  window and every run setting that changes numbers; a source without `version()` is cached in
  memory only. Trials record the data version and the universe.
* **Selection statistics.** A trial is a distinct (strategy, data version) at Paper, Validation or
  Implement stage outside the holdout; relabelled copies with identical returns count once.
  `deflated_sharpe(result)` and `pbo()` use that one definition.
* **Sweeps.** `sweep(base, {"portfolio.n": [...]})` or `sweep(builder, {"k": [...]})` runs in
  this process on the shared cache, or in `workers` subprocesses (`kernel.worker`) that read one
  memory-mapped snapshot. Variants whose custom code lives in a script run in this process.
* **Promotion.** `promote(result, "CANDIDATE")` submits through the official-record gates
  (`governance.records`):
  * the hypothesis: registered before its first trial (the registry stamps the time and freezes
    the content by hash), the trial in the ledger, and machine-checkable failure criteria
    (`"sharpe < 0.3 or max_drawdown < -15%"`) that the result does not meet;
  * conventions confirmed by their values; no filled or altered data; no hard violations or relaxed
    constructions; look-ahead checks on; at least `min_years` of track record;
  * DSR and PBO over the whole search the result came from: every trial on the same universe,
    overlapping window and data version, whatever hypothesis it was filed under, or none
    (`min_dsr=0.95`, `max_pbo=0.5` above four trials, `min_years=1`: placeholders until the team
    sets them);
  * the holdout untouched; one record per result and level.

  OFFICIAL also needs an Implement run of the same fingerprint that completed solvent with
  manifest hashes, a reconciliation with its content and within tolerance, and a source that
  declares point-in-time inputs (`source.point_in_time()`). The record chain keeps its head in an
  anchor file, so a deleted tail fails `verify_chain()`.

## Performance

Synthetic universe, one core unless stated. Reproduce with `tools/probes/kernel_scale.py 9000 10`
and `tools/probes/kernel_sweep_parallel.py`.

| Run | Time |
| --- | --- |
| 3,000 bonds × 5y, daily quintile by sector | 0.9 s (the previous Paper engine: 32.9 s) |
| 3,000 bonds × 5y, monthly quintile | 0.5 s (previous: 2.0 s) |
| 9,000 bonds × 10y (1.1 GB panel), monthly quintile by sector | 2.7 s |
| 9,000 bonds × 10y, daily quintile (2,495 rebalances) | 4.2 s |
| 9,000 bonds × 10y, monthly index-plus optimiser (120 HiGHS solves) | 18.1 s |
| 9,000 bonds × 10y, 10 heuristic variants | 15.7 s in one process (shared cache); 23.4 s in 3 workers |
| 3,000 bonds × 5y, 6 optimiser variants | 20.9 s in one process; 8.1 s in 3 workers |
| Agent workflow example (5 questions, sweep, evaluation, Implement run, promotion) | 8.7 s |

Peak memory for a 9k × 10y session is 6.1 GB; each sweep worker adds about 3 GB. Workers pay a
start-up cost: each loads the snapshot and re-evaluates the shared universe, benchmark and signal.
So use `workers` when variants are expensive (optimiser sweeps), and the shared in-process cache
when they are cheap. The post-run analytics (group tables, exposures, contributions) are about
half the time of a heuristic run; the optimiser's solves dominate an optimised one.

## Legacy frameworks

Behaviour specific to one legacy framework is in `nautilus_fi.adapters`, one module per
framework. Each module has the framework's convention profile, small components built from kernel
primitives, and parity cases against the framework's golden outputs. `adapters.status()` lists
them; [adapters.md](adapters.md) says how a profile gets confirmed.

## Extending

* **A data source.** Subclass `DataSource` (`catalog`, `load`; optionally `attributes`, `series`,
  `constituents`, `environment`, `version`), or use `CallableSource`.
* **An expression.** Use `apply` / `cs` for one-off code. For a reusable node, subclass `Expr`
  with `_eval(ctx) -> (T, N)` and `_params()`. Set `cross_sectional = True` if it works per date
  within the universe; time-series nodes evaluate their children with `ctx.full()`.
* **A construction method.** Subclass `Construction` with `batch(B) -> (R, n)` (vectorised over
  rebalance dates) and/or `weights(cs) -> (Series, diagnostics)`. Set `path_dependent = True` if
  it needs the current portfolio. `names()` declares the data it reads.
* **A cost model.** Subclass `CostModel` with `unit(ctx, t, trade, nav)` (cost per unit traded).
  `linear` is what the optimiser sees.
* **A decision rule the kernel cannot express.** Write a `Decider` and pass it with
  `Strategy(decider=..., fields=..., lookback=...)`.

## Still yours to supply or confirm

The kernel removes the structural gaps, but not the need for real inputs:

* the company data platform behind the data seam;
* calibrated cost and impact parameters from your own trading history;
* golden outputs from the legacy frameworks, to run each adapter's parity cases
  (`adapters.parity`) and confirm its convention profile;
* contract terms and evaluated prices for Implement fidelity outside the synthetic universe.
