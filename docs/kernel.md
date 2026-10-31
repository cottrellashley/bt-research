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
  raises `LookaheadError`. `FieldStoreSource` reads true knowledge times: a request's `as_of`
  gives the latest snapshot known then, and `mode="first"` gives what a live process saw before
  revisions.
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

## Mechanics and conventions

The named conventions of the legacy frameworks (`domain.conventions`) are presets of these
objects:

* `strategy.under("MRI_v1")` re-runs a strategy with that framework's timing, missing-return,
  excess-return, turnover, annualisation and benchmark-drift rules. Costs, universe and signal stay
  the same.
* `strategy.conventions()` is the record that manifests, the ledger and the official gates read.
* Profiles still marked `confirmed=False` cannot be promoted to CANDIDATE.

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
  The kernel reproduces the Paper engine's returns for it exactly; a test asserts this.

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
  turnover under all three definitions, weights.
* **Breakdowns:** contributions by instrument and by group (with benchmark weights and
  contributions), the construction log and violations.
* **Statistics:** `stats()` comes from one function (`Metrics.stats`), so Paper and Implement
  report the same keys on the same basis. Implement adds `implement_*` keys such as fill rate,
  spread cost, delay drift and solvency.

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

## Sessions and governance

* **Trials.** Every new run in a `Session` is a trial in the ledger. Each trial is one atomic file
  carrying the session, fingerprint, description, statistics and returns, so many agents can write
  at once. The hypothesis registry merges under a lock.
* **Deduplication.** The same strategy on the same window is one trial, however often it runs and
  across session restarts.
* **Selection statistics.** `deflated_sharpe(result)` deflates by the session's distinct trials;
  `pbo()` covers the session or a sweep.
* **Sweeps.** `sweep(base, {"portfolio.n": [...]})` or `sweep(builder, {"k": [...]})` runs in
  this process on the shared cache, or in `workers` subprocesses (`kernel.worker`) that read one
  memory-mapped snapshot. Variants whose custom code lives in a script run in this process.
* **Promotion.** `promote(result, "CANDIDATE")` submits through the official-record gates: a
  hypothesis registered first, the trial in the ledger, confirmed conventions, no silently
  zero-filled returns, no hard violations, and DSR from the ledger. OFFICIAL additionally needs an
  Implement run and its reconciliation.

## Performance

Synthetic universe, one core unless stated.

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
* golden outputs from the legacy frameworks, to confirm their convention profiles;
* contract terms and evaluated prices for Implement fidelity outside the synthetic universe.
