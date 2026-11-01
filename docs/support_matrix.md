# Support matrix

Generated from `nautilus_fi.support.default_support_matrix()` by `tools/support_matrix.py`; every run's manifest carries the rows that apply to the products it holds, and `BacktestSettings.allow_support` refuses runs that need anything above the allowed status (default: EXPERIMENTAL allowed, UNSUPPORTED refused).

| Status | Meaning | Count |
| --- | --- | --- |
| VALIDATED | mechanics covered by a fixture or invariant test named in the row | 41 |
| SUPPORTED_WITH_APPROXIMATION | runs, with the simplification stated in the row | 33 |
| EXPERIMENTAL | code path exists, not validated end to end | 2 |
| UNSUPPORTED | refused by the builder | 5 |

| Family | Feature | Status | Approximation / notes | Validation case |
| --- | --- | --- | --- | --- |
| ABS | tranche_cashflows | SUPPORTED_WITH_APPROXIMATION | same waterfall engine as CLO |  |
| BILL | discount_yield_quotation | VALIDATED |  | bill yield conversion; maturity at par |
| CDS | single_name_and_index | SUPPORTED_WITH_APPROXIMATION | flat hazard per tenor; quarterly accrual-on-default; CDX tradable on the margin venue (price 100 - upfront, running coupon as daily carry; defaults in the supplied price) | credit event payout; index factor |
| CLO | external_engine | EXPERIMENTAL | adapter contract only (Intex not integrated) |  |
| CLO | waterfall_triggers_reinvestment | SUPPORTED_WITH_APPROXIMATION | simplified OC/IC tests, sequential/pro-rata principal, equity residual | trigger breach redirects cash; conservation |
| CMBS | tranche_cashflows | SUPPORTED_WITH_APPROXIMATION | same waterfall engine; no loan-level CMBS specifics |  |
| CONVERTIBLE | conversion | UNSUPPORTED | equity path dependency not modelled |  |
| CORPORATE | benchmark_transition | VALIDATED |  | LIBOR to SOFR with spread adjustment |
| CORPORATE | covenants | UNSUPPORTED |  |  |
| CORPORATE | fixed_step_zero | VALIDATED |  | coupon entitlement on fill day; redemption |
| CORPORATE | floating_compounded_in_arrears | VALIDATED |  | lookback/lockout weighting; partial-knowledge status |
| CORPORATE | floating_in_advance | VALIDATED |  | fixing lag across holidays |
| CORPORATE | full_call_put | VALIDATED |  | call at price; put by policy |
| CORPORATE | make_whole_call | SUPPORTED_WITH_APPROXIMATION | exercise rule only; no make-whole price model |  |
| CORPORATE | partial_call | VALIDATED |  | partial call cash and position reconciliation |
| CORPORATE | pik | VALIDATED |  | PIK capitalisation without cash |
| CORPORATE | sinking_fund | SUPPORTED_WITH_APPROXIMATION | scheduled fractions, no lottery |  |
| CORPORATE | tender_offer | SUPPORTED_WITH_APPROXIMATION | actor policy threshold; proration cap only | tender with proration |
| DISTRESSED | default_lifecycle | VALIDATED |  | missed payment -> grace -> default -> distributions -> discharge |
| DISTRESSED | exchange_restructuring | VALIDATED |  | old debt becomes cash plus new debt and equity |
| FUND | nav_vs_market | SUPPORTED_WITH_APPROXIMATION | price/NAV series supplied; no creation mechanics |  |
| FUTURES | treasury_futures | SUPPORTED_WITH_APPROXIMATION | CTD by implied repo among supplied basket; VM settled daily; tradable on the margin venue as dated contracts closed at the last price (rolls are the strategy's) | daily VM without duplicate P&L |
| FX | forwards_and_hedging | SUPPORTED_WITH_APPROXIMATION | forward points by tenor; rolling program; forwards tradable on the margin venue as dated contracts, cash-settled at the last price | hedged vs unhedged decomposition |
| GOVERNANCE | official_records | VALIDATED |  | gates refuse missing Implement / reconciliation / unconfirmed conventions; hash-chained registry |
| GOVERNANCE | trial_ledger | VALIDATED |  | every run recorded; DSR uses the ledger's trial variance; PBO over the variant matrix; holdout guard |
| GOVERNANCE | walk_forward | VALIDATED |  | purge and embargo in trading days; validation and out-of-sample trials recorded |
| IRS | valuation_margin | SUPPORTED_WITH_APPROXIMATION | single-curve fallback when no projection curve; flat forwards per period; tradable on the margin venue as a financed contract on a supplied price (NPV) series | par swap values to zero; VM/CTM postings |
| KERNEL | construction_methods | VALIDATED |  | vectorised and per-date paths give identical returns; caps, projections and checks |
| KERNEL | cost_models | SUPPORTED_WITH_APPROXIMATION | half spread, fixed, square-root impact and expression costs with default parameters until calibrated on own TCA; the optimiser sees the linear part | cost algebra; impact rises with size; participation spreads trades over days |
| KERNEL | custom_code | SUPPORTED_WITH_APPROXIMATION | custom functions are checked empirically on three truncated panels, not proven causal; closures over external data are not seen | causal custom code passes; full-sample demeaning refused |
| KERNEL | data_seam | VALIDATED |  | frame / callable / composite / mapped / SQL / field-store sources; memory-mapped cache slices; pinned snapshot versions; as-of reads |
| KERNEL | explain | VALIDATED |  | net difference = holdings + cash and financing + costs + residual; identical results flagged; binding constraints reported |
| KERNEL | expressions | VALIDATED |  | z-score / rank / demean by group, rolling and differences equal pandas references; neutralised residual orthogonal to the exposure |
| KERNEL | implement_fidelity | SUPPORTED_WITH_APPROXIMATION | needs an event-engine environment (terms, evaluated prices) from the source or supplied; execution parameters as in the execution process row | the same strategy runs solvent in the event engine with the paper statistics' keys; reconciliation decomposes the gap |
| KERNEL | index_expressions | SUPPORTED_WITH_APPROXIMATION | the published-return check uses a synthetic index built with the same formula (circular); replication against a real index is not yet shown | constituents reproduce the synthetic index return; exclusions and issuer caps hold at every reconstitution |
| KERNEL | lookahead_checks | VALIDATED |  | leads refused structurally; full-sample code caught on truncated panels; availability lags cannot be undercut |
| KERNEL | optimiser_expressions | VALIDATED |  | issuer, sector, exposure, weight and turnover limits hold at every rebalance; custom linear constraints bind; penalties need negative coefficients |
| KERNEL | sessions_and_sweeps | VALIDATED |  | one trial per strategy and window across restarts; concurrent ledger writers; parallel sweeps equal sequential; record gates |
| KERNEL | strategy_identity | VALIDATED |  | fingerprint changes with custom code and parameters, not names; diff by component; path edits |
| KERNEL | vectorised_engine | VALIDATED |  | identical returns to the panel engine for the same research spec; long-short financing; overlays; triggered schedules |
| LINKER | indexation_uk_eur_ca | SUPPORTED_WITH_APPROXIMATION | lag/interpolation per convention; no seasonality |  |
| LINKER | indexation_us_tips | VALIDATED |  | Canadian interpolation; known vs projected CPI; maturity floor |
| LOAN | amendments | VALIDATED |  | effective vs announcement dated terms |
| LOAN | delayed_compensation | SUPPORTED_WITH_APPROXIMATION | LSTA formula simplified (interest less cost of carry) |  |
| LOAN | drawn_undrawn_fees | VALIDATED |  | commitment fee; drawdown without price |
| LOAN | transfer_restrictions | SUPPORTED_WITH_APPROXIMATION | flag only; no consent workflow |  |
| MBS | pool_factors_paydown | VALIDATED |  | paydown reduces principal and interest without double counting |
| MBS | prepayment_models | SUPPORTED_WITH_APPROXIMATION | PSA / refinance-incentive S-curve; no loan-level behaviour |  |
| MONEY_MARKET | amortized_cost_liquidity_fee | SUPPORTED_WITH_APPROXIMATION | straight-line amortized cost; 2a-7 style fee rule |  |
| MUNICIPAL | extraordinary_redemption | UNSUPPORTED |  |  |
| MUNICIPAL | tax_treatment | SUPPORTED_WITH_APPROXIMATION | investor profile rates; no AMT or de minimis rules | after-tax proceeds |
| PLATFORM | benchmark_objects | SUPPORTED_WITH_APPROXIMATION | index replication checked only against a synthetic index built with the same formula (circular); needs real index returns within the agreed tolerance | constituent weights reconstruct the synthetic index; drift vs static; custom caps and subsets |
| PLATFORM | benchmark_relative_constraints | VALIDATED |  | new buys checked, turnover budgets scale trades, benchmark bands and active exposures evaluated and logged |
| PLATFORM | conventions_registry | VALIDATED |  | turnover / annualisation / excess-return / missing-return definitions; legacy profiles recorded |
| PLATFORM | data_loaders | SUPPORTED_WITH_APPROXIMATION | Snowflake / FIAtlas mappings are templates until pointed at the deployed views | SQL mapping, scaling, knowledge-time stamping into the point-in-time stores |
| PLATFORM | dated_settlement | VALIDATED |  | US T+2 -> T+1 on 28 May 2024 |
| PLATFORM | double_entry_ledger | VALIDATED |  | 1,030 fixture; conservation |
| PLATFORM | execution_process | SUPPORTED_WITH_APPROXIMATION | RFQ panel and liquidity gate are scenario assumptions until calibrated on own order history |  |
| PLATFORM | historical_counterfactual | EXPERIMENTAL | dependency rules validated; driver recomputation partial |  |
| PLATFORM | margin_venue_overlays | SUPPORTED_WITH_APPROXIMATION | financed contracts (TRS, CDX, synthetic bond TRS) settle carry daily as Nautilus funding; futures / FX forwards close at the last price; collateral moves by rule | NAV identity with margin equity; collateral location does not change NAV |
| PLATFORM | optimiser | VALIDATED |  | bands, turnover, min holding, cardinality, factor tracking error, CVaR; infeasible problems relax or hold |
| PLATFORM | panel_engine | VALIDATED |  | benchmark replication without costs tracks to 1e-9; oracle signal earns nothing with a two-day lag |
| PLATFORM | paper_implement_reconciliation | VALIDATED |  | holdings gap + cash + execution + residual equals the shortfall |
| PLATFORM | point_in_time_replay | VALIDATED |  | replay by availability time; temporal isolation invariant |
| PLATFORM | rebalance_calendars | VALIDATED |  | BME / BMS / W-WED / index calendars / triggered rebalancing in both engines |
| PLATFORM | repo_financed_leverage | SUPPORTED_WITH_APPROXIMATION | repo drawn at settlement against eligible holdings at cash rate + spread; single funding agreement | gross exposure above NAV stays solvent; financing booked |
| PLATFORM | signal_evaluation | VALIDATED |  | IC / quantile spread on a known signal; noise IC near zero; forward windows start after the trade |
| PLATFORM | stochastic_simulation | UNSUPPORTED |  |  |
| PLATFORM | streamed_replay | VALIDATED |  | streamed and one-shot runs give identical NAV and event-chain hashes |
| PLATFORM | synthetic_shorts | SUPPORTED_WITH_APPROXIMATION | short cash bonds as bond TRS on the margin venue: financing received, coupon paid, borrow fee charged; no recall or specials | cash-venue short rejected; TRS short carried with borrow fee |
| PLATFORM | terms_free_instruments | SUPPORTED_WITH_APPROXIMATION | vendor accrued and supplied cashflows replace terms; no matrix pricing or stale roll without terms | supplied coupon paid under the entitlement rule; NAV identity |
| PLATFORM | trade_proposals | SUPPORTED_WITH_APPROXIMATION | CRD / QOE / PORT layouts are column maps to confirm against each system's import specification | bid / ask dirty pricing, settlement dates, pre-trade checks, position reconciliation |
| PLATFORM | vendor_analytics | VALIDATED |  | vendor OAS / OAD / OASD used; COMPARE records differences |
| PREFERRED | coupon_deferral | VALIDATED |  | cumulative vs non-cumulative deferral |
| PREFERRED | discretionary_call | VALIDATED |  | no assumed redemption at first call |
| PREFERRED | writedown_conversion | SUPPORTED_WITH_APPROXIMATION | trigger on supplied metric; no capital model |  |
| REPO | financing | SUPPORTED_WITH_APPROXIMATION | haircut, term, rollover, margin; no specials curve | collateral coupon passes during repo |
| SECURITIES_LENDING | borrow | SUPPORTED_WITH_APPROXIMATION | availability callback, fee, recall | manufactured payment |
| SOVEREIGN | auction_reopening | UNSUPPORTED | auction allocation events not modelled |  |
| SOVEREIGN | fixed_coupon | VALIDATED |  | par identity, ACT/ACT accrued, maturity fixture |
| TBA | delivery_and_rolls | SUPPORTED_WITH_APPROXIMATION | SIFMA class dates supplied as data; approximate fallback flagged | forward converts to delivered pools |
