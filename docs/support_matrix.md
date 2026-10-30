# Support matrix

Generated from `nautilus_fi.support.default_support_matrix()`; every run's manifest carries the rows that apply to the products it holds, and `BacktestSettings.allow_support` refuses runs that need anything above the allowed status (default: EXPERIMENTAL allowed, UNSUPPORTED refused).

| Status | Meaning | Count |
| --- | --- | --- |
| VALIDATED | mechanics covered by a fixture or invariant test named in the row | 20 |
| SUPPORTED_WITH_APPROXIMATION | runs, with the simplification stated in the row | 22 |
| EXPERIMENTAL | code path exists, not validated end to end | 2 |
| UNSUPPORTED | refused by the builder | 5 |

| Family | Feature | Status | Approximation / notes | Validation case |
| --- | --- | --- | --- | --- |
| ABS | tranche_cashflows | SUPPORTED_WITH_APPROXIMATION | same waterfall engine as CLO |  |
| BILL | discount_yield_quotation | VALIDATED |  | bill yield conversion; maturity at par |
| CDS | single_name_and_index | SUPPORTED_WITH_APPROXIMATION | flat hazard per tenor; quarterly accrual-on-default | credit event payout; index factor |
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
| FUTURES | treasury_futures | SUPPORTED_WITH_APPROXIMATION | CTD by implied repo among supplied basket; VM settled daily | daily VM without duplicate P&L |
| FX | forwards_and_hedging | SUPPORTED_WITH_APPROXIMATION | forward points by tenor; rolling program | hedged vs unhedged decomposition |
| IRS | valuation_margin | SUPPORTED_WITH_APPROXIMATION | single-curve fallback when no projection curve; flat forwards per period | par swap values to zero; VM/CTM postings |
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
| PLATFORM | dated_settlement | VALIDATED |  | US T+2 -> T+1 on 28 May 2024 |
| PLATFORM | double_entry_ledger | VALIDATED |  | 1,030 fixture; conservation |
| PLATFORM | execution_process | SUPPORTED_WITH_APPROXIMATION | RFQ panel and liquidity gate are scenario assumptions until calibrated on own order history |  |
| PLATFORM | historical_counterfactual | EXPERIMENTAL | dependency rules validated; driver recomputation partial |  |
| PLATFORM | point_in_time_replay | VALIDATED |  | replay by availability time; temporal isolation invariant |
| PLATFORM | stochastic_simulation | UNSUPPORTED |  |  |
| PREFERRED | coupon_deferral | VALIDATED |  | cumulative vs non-cumulative deferral |
| PREFERRED | discretionary_call | VALIDATED |  | no assumed redemption at first call |
| PREFERRED | writedown_conversion | SUPPORTED_WITH_APPROXIMATION | trigger on supplied metric; no capital model |  |
| REPO | financing | SUPPORTED_WITH_APPROXIMATION | haircut, term, rollover, margin; no specials curve | collateral coupon passes during repo |
| SECURITIES_LENDING | borrow | SUPPORTED_WITH_APPROXIMATION | availability callback, fee, recall | manufactured payment |
| SOVEREIGN | auction_reopening | UNSUPPORTED | auction allocation events not modelled |  |
| SOVEREIGN | fixed_coupon | VALIDATED |  | par identity, ACT/ACT accrued, maturity fixture |
| TBA | delivery_and_rolls | SUPPORTED_WITH_APPROXIMATION | SIFMA class dates supplied as data; approximate fallback flagged | forward converts to delivered pools |
