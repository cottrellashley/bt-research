"""The platform's support matrix: every product family and feature with its status, approximation
and the validation case that covers it. Users see this before a run and in its results."""

from __future__ import annotations

from nautilus_fi.core.registry import Capability, SupportMatrix, SupportStatus

V, A, X, U = SupportStatus.VALIDATED, SupportStatus.SUPPORTED_WITH_APPROXIMATION, SupportStatus.EXPERIMENTAL, SupportStatus.UNSUPPORTED


def default_support_matrix() -> SupportMatrix:
    sm = SupportMatrix()
    decl = sm.declare
    # cash securities
    decl(Capability("SOVEREIGN", "fixed_coupon", V, ("US", "*"), ("USD", "EUR", "GBP", "JPY", "CAD", "AUD"), validation_case="par identity, ACT/ACT accrued, maturity fixture"))
    decl(Capability("BILL", "discount_yield_quotation", V, validation_case="bill yield conversion; maturity at par"))
    decl(Capability("SOVEREIGN", "auction_reopening", U, notes="auction allocation events not modelled"))
    decl(Capability("CORPORATE", "fixed_step_zero", V, validation_case="coupon entitlement on fill day; redemption"))
    decl(Capability("CORPORATE", "floating_in_advance", V, data_requirements=("fixings",), validation_case="fixing lag across holidays"))
    decl(Capability("CORPORATE", "floating_compounded_in_arrears", V, data_requirements=("daily_fixings",), validation_case="lookback/lockout weighting; partial-knowledge status"))
    decl(Capability("CORPORATE", "benchmark_transition", V, validation_case="LIBOR to SOFR with spread adjustment"))
    decl(Capability("CORPORATE", "partial_call", V, validation_case="partial call cash and position reconciliation"))
    decl(Capability("CORPORATE", "full_call_put", V, validation_case="call at price; put by policy"))
    decl(Capability("CORPORATE", "tender_offer", A, approximation="actor policy threshold; proration cap only", validation_case="tender with proration"))
    decl(Capability("CORPORATE", "sinking_fund", A, approximation="scheduled fractions, no lottery"))
    decl(Capability("CORPORATE", "pik", V, validation_case="PIK capitalisation without cash"))
    decl(Capability("CORPORATE", "make_whole_call", A, approximation="exercise rule only; no make-whole price model"))
    decl(Capability("CORPORATE", "covenants", U))
    decl(Capability("LINKER", "indexation_us_tips", V, validation_case="Canadian interpolation; known vs projected CPI; maturity floor"))
    decl(Capability("LINKER", "indexation_uk_eur_ca", A, jurisdictions=("UK", "EU", "CA"), currencies=("GBP", "EUR", "CAD"), approximation="lag/interpolation per convention; no seasonality"))
    decl(Capability("MUNICIPAL", "tax_treatment", A, approximation="investor profile rates; no AMT or de minimis rules", validation_case="after-tax proceeds"))
    decl(Capability("MUNICIPAL", "extraordinary_redemption", U))
    decl(Capability("LOAN", "drawn_undrawn_fees", V, validation_case="commitment fee; drawdown without price"))
    decl(Capability("LOAN", "delayed_compensation", A, approximation="LSTA formula simplified (interest less cost of carry)"))
    decl(Capability("LOAN", "amendments", V, validation_case="effective vs announcement dated terms"))
    decl(Capability("LOAN", "transfer_restrictions", A, approximation="flag only; no consent workflow"))
    decl(Capability("DISTRESSED", "default_lifecycle", V, validation_case="missed payment -> grace -> default -> distributions -> discharge"))
    decl(Capability("DISTRESSED", "exchange_restructuring", V, validation_case="old debt becomes cash plus new debt and equity"))
    decl(Capability("PREFERRED", "coupon_deferral", V, validation_case="cumulative vs non-cumulative deferral"))
    decl(Capability("PREFERRED", "discretionary_call", V, validation_case="no assumed redemption at first call"))
    decl(Capability("PREFERRED", "writedown_conversion", A, approximation="trigger on supplied metric; no capital model"))
    # structured and overlays
    decl(Capability("MBS", "pool_factors_paydown", V, data_requirements=("factors",), validation_case="paydown reduces principal and interest without double counting"))
    decl(Capability("MBS", "prepayment_models", A, approximation="PSA / refinance-incentive S-curve; no loan-level behaviour"))
    decl(Capability("TBA", "delivery_and_rolls", A, data_requirements=("tba_settlement_calendar",), approximation="SIFMA class dates supplied as data; approximate fallback flagged", validation_case="forward converts to delivered pools"))
    decl(Capability("CLO", "waterfall_triggers_reinvestment", A, approximation="simplified OC/IC tests, sequential/pro-rata principal, equity residual", validation_case="trigger breach redirects cash; conservation"))
    decl(Capability("CLO", "external_engine", X, notes="adapter contract only (Intex not integrated)"))
    decl(Capability("ABS", "tranche_cashflows", A, approximation="same waterfall engine as CLO"))
    decl(Capability("CMBS", "tranche_cashflows", A, approximation="same waterfall engine; no loan-level CMBS specifics"))
    decl(Capability("IRS", "valuation_margin", A, approximation="single-curve fallback when no projection curve; flat forwards per period", validation_case="par swap values to zero; VM/CTM postings"))
    decl(Capability("CDS", "single_name_and_index", A, approximation="flat hazard per tenor; quarterly accrual-on-default", validation_case="credit event payout; index factor"))
    decl(Capability("FUTURES", "treasury_futures", A, approximation="CTD by implied repo among supplied basket; VM settled daily", validation_case="daily VM without duplicate P&L"))
    decl(Capability("FX", "forwards_and_hedging", A, approximation="forward points by tenor; rolling program", validation_case="hedged vs unhedged decomposition"))
    decl(Capability("REPO", "financing", A, approximation="haircut, term, rollover, margin; no specials curve", validation_case="collateral coupon passes during repo"))
    decl(Capability("SECURITIES_LENDING", "borrow", A, approximation="availability callback, fee, recall", validation_case="manufactured payment"))
    decl(Capability("MONEY_MARKET", "amortized_cost_liquidity_fee", A, approximation="straight-line amortized cost; 2a-7 style fee rule"))
    decl(Capability("FUND", "nav_vs_market", A, approximation="price/NAV series supplied; no creation mechanics"))
    decl(Capability("CONVERTIBLE", "conversion", U, notes="equity path dependency not modelled"))
    # platform
    decl(Capability("PLATFORM", "point_in_time_replay", V, validation_case="replay by availability time; temporal isolation invariant"))
    decl(Capability("PLATFORM", "double_entry_ledger", V, validation_case="1,030 fixture; conservation"))
    decl(Capability("PLATFORM", "dated_settlement", V, validation_case="US T+2 -> T+1 on 28 May 2024"))
    decl(Capability("PLATFORM", "execution_process", A, approximation="RFQ panel and liquidity gate are scenario assumptions until calibrated on own order history"))
    decl(Capability("PLATFORM", "historical_counterfactual", X, run_modes=("HISTORICAL_COUNTERFACTUAL",), notes="dependency rules validated; driver recomputation partial"))
    decl(Capability("PLATFORM", "stochastic_simulation", U, run_modes=("STOCHASTIC_SIMULATION",)))
    return sm


__all__ = ["default_support_matrix"]
