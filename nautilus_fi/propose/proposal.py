"""Propose: turn a target portfolio into a trade proposal a desk can execute.

The same construction output the engines simulate becomes real orders: target weights against
current holdings (from PORT or the custodian), each trade sized and priced at the side it pays (ask
dirty for buys, bid dirty for sells) and rounded once to the bond's minimum piece and increment,
with settlement dates from the dated settlement conventions and the cash impact per settlement
date. ``validate`` runs the platform's pre-trade checks: pieces and increments, restricted names,
missing or stale prices, shorts, cash by settlement date, and the construction grammar's
constraints on the post-trade portfolio. Reconciliation to Bloomberg PORT / PREP is not covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from nautilus_fi.instruments.bond import BondSpec
from nautilus_fi.instruments.conventions import SettlementConventions


@dataclass
class TradeProposal:
    as_of: date
    portfolio: str
    nav: float
    cash_before: float
    lines: pd.DataFrame                    # one row per order
    holdings_after: pd.DataFrame           # symbol, face, weight
    validation: pd.DataFrame               # check, scope, symbol, status, detail
    spec_hash: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """No blocker (approvals may still be required: ``approvals``)."""
        if self.validation.empty:
            return True
        if "severity" in self.validation:
            return not (self.validation["severity"] == "blocker").any()
        return not (self.validation["status"] == "FAIL").any()

    @property
    def blockers(self) -> pd.DataFrame:
        v = self.validation
        return v[v["severity"] == "blocker"] if "severity" in v else v[v["status"] == "FAIL"]

    @property
    def approvals(self) -> pd.DataFrame:
        v = self.validation
        return v[v["severity"] == "approval_required"] if "severity" in v else v.iloc[0:0]

    @property
    def cash_after(self) -> float:
        return float(self.cash_before + (self.lines["cash"].sum() if len(self.lines) else 0.0))

    def summary(self) -> dict:
        v = self.validation
        return {"as_of": str(self.as_of), "portfolio": self.portfolio, "nav": self.nav, "n_orders": len(self.lines), "buys": float(-self.lines.loc[self.lines["side"] == "BUY", "cash"].sum()) if len(self.lines) else 0.0, "sells": float(self.lines.loc[self.lines["side"] == "SELL", "cash"].sum()) if len(self.lines) else 0.0, "cash_before": self.cash_before, "cash_after": self.cash_after, "fails": int((v["status"] == "FAIL").sum()) if len(v) else 0, "warnings": int((v["status"] == "WARN").sum()) if len(v) else 0}


def _ident(b: BondSpec) -> tuple[str, str]:
    if b.identifiers.cusip:
        return b.identifiers.cusip, "CUSIP"
    if b.identifiers.isin:
        return b.identifiers.isin, "ISIN"
    return b.symbol, "TICKER"


def build_proposal(target_weights: pd.Series | dict, holdings_face: pd.Series | dict, prices: pd.DataFrame, bonds: dict[str, BondSpec], cash: float, as_of: date, nav: float | None = None, portfolio: str = "PORTFOLIO", settlement: SettlementConventions | None = None, restricted: set[str] | None = None, min_trade_face: float = 0.0, max_price_age_days: int = 3, constraints=None, benchmark: pd.Series | None = None, attributes: pd.DataFrame | None = None, allow_shorts: bool = False, spec_hash: str | None = None, round_up_fraction: float | None = None, min_trade_bps: float = 0.0, max_position_weight: float | None = None, cash_flow: float | None = None, reported_nav: float | None = None, mv_tolerance: float = 0.001, cash_bounds: tuple | None = None, cash_target: float | None = None, cash_target_tolerance: float = 0.005, analytics: pd.DataFrame | None = None, coverage_min: dict | None = None, carried: pd.DataFrame | None = None, trade_overrides: dict | None = None, severities: dict | None = None) -> TradeProposal:
    """``prices``: index symbol, columns ``bid`` / ``ask`` (clean, per 100 current face), optional
    ``mid``, ``accrued`` (per 100), ``factor``, ``price_date``. ``target_weights``: fractions of NAV,
    valued at dirty mid. Each trade is sized from the value change at the price it pays (ask dirty
    for buys, bid dirty for sells) and rounded once: buys down (never above the target cash), trims
    to the nearest tradable amount, full sales exactly; a residual below the minimum piece is closed
    and a target below it is reported, not dropped silently. ``round_up_fraction``: a new buy of at
    least that share of the minimum piece buys the minimum piece. ``min_trade_bps``: trades worth
    less than this share of NAV are skipped; ``max_position_weight`` caps targets;
    ``cash_flow``: an inflow (> 0) only buys and an outflow (< 0) only sells; ``reported_nav``:
    the book's NAV must agree within ``mv_tolerance``; ``cash_bounds``: (min, max) cash after
    the trades as a share of NAV; ``cash_target``: the share of NAV kept in cash (targets summing
    to more are scaled down to 1 - cash_target, and cash after the trades is reconciled to it
    within ``cash_target_tolerance``). ``analytics``: values by symbol (the signal, OASD, ...)
    whose coverage of the target book, by weight, must reach ``coverage_min`` (default 95% for
    each column); ``carried``: the report of analytics carried from T-1 (``carry_analytics``).
    ``trade_overrides``: symbol -> signed face that replaces the sized trade (PM feedback; 0: no
    trade). Every check carries a severity: blocker, approval_required, warning or info
    (``severities`` overrides the defaults by check name)."""
    inputs = {k: v for k, v in locals().items() if k != "inputs"}             # kept for apply_feedback
    settlement = settlement or SettlementConventions()
    restricted = set(restricted or ())
    tw = pd.Series(target_weights, dtype=float)
    if cash_target is not None and tw.clip(lower=0.0).sum() > 1.0 - cash_target:
        tw = tw * (1.0 - cash_target) / tw.clip(lower=0.0).sum()
    if max_position_weight is not None:
        tw = tw.clip(upper=float(max_position_weight))
    overrides = {str(k): float(v) for k, v in (trade_overrides or {}).items()}
    hf = pd.Series(holdings_face, dtype=float)
    px = prices.copy()
    if "mid" not in px:
        px["mid"] = (px["bid"] + px["ask"]) / 2.0
    for col, default in (("accrued", 0.0), ("factor", 1.0)):
        if col not in px:
            px[col] = default
    dirty_mid = (px["mid"] + px["accrued"]) / 100.0 * px["factor"]
    computed_nav = float((hf * dirty_mid.reindex(hf.index).fillna(0.0)).sum() + cash)
    if nav is None:
        nav = computed_nav
    checks: list[dict] = []
    if reported_nav is not None and abs(computed_nav - reported_nav) > mv_tolerance * abs(reported_nav):
        checks.append({"check": "mv_reconciliation", "scope": "portfolio", "symbol": None, "status": "FAIL", "detail": f"holdings and cash value to {computed_nav:,.0f}, the book reports {reported_nav:,.0f}"})
    rows = []
    syms = sorted(set(tw.index) | set(hf.index) | set(overrides))
    for sym in syms:
        b = bonds.get(sym)
        if b is None:
            checks.append({"check": "known_security", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "no reference terms"})
            continue
        cur = float(hf.get(sym, 0.0))
        if sym not in px.index or not np.isfinite(px.at[sym, "bid"]) or not np.isfinite(px.at[sym, "ask"]):
            if abs(float(tw.get(sym, 0.0))) > 0 or cur:
                checks.append({"check": "price_available", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "no bid/ask"})
            continue
        dm = float(dirty_mid[sym])
        acc = float(px.at[sym, "accrued"])
        fct = float(px.at[sym, "factor"])
        want = float(tw.get(sym, 0.0)) * nav                                    # target value, at mid
        change = want - cur * dm
        closing = abs(want) < 1e-9 and abs(cur) > 0
        if sym in overrides:                                                    # a PM's size: validated, not resized
            if abs(overrides[sym]) < 1e-9:
                continue
            change = overrides[sym]
            closing = change < 0 and abs(change) >= abs(cur) - 1e-9 and cur > 0
        if not closing and abs(change) < 1e-9:
            continue
        side = "BUY" if change > 0 else "SELL"
        clean = float(px.at[sym, "ask"] if side == "BUY" else px.at[sym, "bid"])
        dirty = clean + acc
        d_side = dirty / 100.0 * fct                                            # sized at the price the trade pays: ask for buys, bid for sells
        if sym in overrides:
            face = abs(cur) if closing else abs(overrides[sym])
        elif closing:
            face = abs(cur)
        else:
            raw = abs(change) / d_side if d_side > 0 else 0.0
            face = abs(b.tradable(raw, "down" if side == "BUY" else "nearest"))       # once: buys never overspend, trims to the nearest piece
            if face == 0.0 and round_up_fraction is not None and side == "BUY" and raw >= round_up_fraction * b.min_piece:
                face = b.min_piece
            rest = abs(cur) - face if side == "SELL" else abs(cur) + face
            if side == "SELL" and 0 < rest < b.min_piece - 1e-9:                 # an untradable residual: close it
                checks.append({"check": "residual_below_min_piece", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"selling {face:,.0f} would leave {rest:,.0f} below the minimum piece {b.min_piece:,.0f}; closing the position"})
                face, closing = abs(cur), True
            if face == 0.0:
                checks.append({"check": "below_min_piece", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"{'target' if not cur else 'change'} of {raw:,.0f} face is below the minimum piece {b.min_piece:,.0f}: not traded"})
                continue
            if face < min_trade_face:
                checks.append({"check": "below_min_trade", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"trade {face:,.0f} below the minimum trade {min_trade_face:,.0f}; skipped"})
                continue
        if sym not in overrides and cash_flow is not None and ((cash_flow > 0 and side == "SELL") or (cash_flow < 0 and side == "BUY")):
            checks.append({"check": "cash_flow_mode", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"{side.lower()} skipped: a {'inflow buys' if cash_flow > 0 else 'outflow sells'} only"})
            continue
        if sym not in overrides and min_trade_bps and face * d_side < min_trade_bps / 1e4 * nav and not closing:
            checks.append({"check": "below_min_trade", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"trade worth {face * d_side:,.0f} is below {min_trade_bps:g}bp of NAV; skipped"})
            continue
        target_face = (cur + face) if side == "BUY" else (0.0 if closing else cur - face)
        settle = settlement.settle_date(b.asset_type, b.currency, as_of, b.calendar) if b.settlement_lag_days is None else b.settle_date(as_of)
        ident, id_type = _ident(b)
        cash_amt = (-1.0 if side == "BUY" else 1.0) * face * dirty / 100.0 * fct
        rows.append({"order_id": f"{portfolio}-{as_of:%Y%m%d}-{len(rows) + 1:04d}", "portfolio": portfolio, "symbol": sym, "security_id": ident, "id_type": id_type, "side": side, "face": face, "price_clean": clean, "accrued": acc, "price_dirty": dirty, "factor": fct, "cash": cash_amt, "currency": b.currency, "trade_date": as_of, "settle_date": settle, "min_piece": b.min_piece, "increment": b.increment, "weight_before": cur * dm / nav if nav else 0.0, "weight_after": target_face * dm / nav if nav else 0.0, "reason": "PM size" if sym in overrides else "rebalance to target", "instrument": "BOND", "notional": face / fct if fct else face})
        if sym in restricted and side == "BUY":
            checks.append({"check": "restricted_list", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "buy of a restricted name"})
        if face < b.min_piece and not closing:
            checks.append({"check": "min_piece", "scope": "line", "symbol": sym, "status": "FAIL", "detail": f"{face:,.0f} below minimum piece {b.min_piece:,.0f}"})
        if not closing and not b.is_tradable(face):
            checks.append({"check": "increment", "scope": "line", "symbol": sym, "status": "FAIL", "detail": f"{face:,.0f} is not the minimum piece {b.min_piece:,.0f} plus whole increments of {b.increment:,.0f}"})
        if not allow_shorts and target_face < -1e-9:
            checks.append({"check": "no_shorts", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "target below zero"})
        if "price_date" in px and pd.notna(px.at[sym, "price_date"]) and (as_of - pd.Timestamp(px.at[sym, "price_date"]).date()).days > max_price_age_days:
            checks.append({"check": "stale_price", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"price from {pd.Timestamp(px.at[sym, 'price_date']).date()}"})
    lines = pd.DataFrame(rows)
    if len(lines):
        lines = lines.sort_values(["side", "symbol"], ascending=[False, True]).reset_index(drop=True)      # sells first
        by_settle = lines.groupby("settle_date")["cash"].sum().sort_index().cumsum() + cash
        for d_, bal in by_settle.items():
            if bal < -1e-6:
                checks.append({"check": "cash_by_settlement", "scope": "portfolio", "symbol": None, "status": "FAIL", "detail": f"cash {bal:,.0f} on {d_}"})
    after = hf.copy()
    for _, r in lines.iterrows():
        after[r["symbol"]] = after.get(r["symbol"], 0.0) + (r["face"] if r["side"] == "BUY" else -r["face"])
    after = after[after.abs() > 1e-9]
    w_after = (after * dirty_mid.reindex(after.index).fillna(0.0)) / nav if nav else after * 0.0
    holdings_after = pd.DataFrame({"symbol": after.index, "face": after.to_numpy(), "weight": w_after.to_numpy()})
    if constraints:
        from nautilus_fi.construct.cross_section import CrossSection
        from nautilus_fi.construct.grammar import check_constraints, parse_constraints

        ids = pd.Index(sorted(set(after.index) | set((benchmark if benchmark is not None else pd.Series(dtype=float)).index)))
        cs = CrossSection(pd.Timestamp(as_of), ids, attributes.reindex(ids) if attributes is not None else pd.DataFrame(index=ids), pd.Series(hf, dtype=float).mul(dirty_mid).div(nav).reindex(ids).fillna(0.0), benchmark if benchmark is not None else pd.Series(0.0, index=ids))
        for v in check_constraints(w_after.reindex(ids).fillna(0.0), cs, parse_constraints(constraints)):
            checks.append({"check": f"constraint:{v['constraint']}", "scope": "portfolio", "symbol": v["group"], "status": "FAIL" if v["severity"] == "HARD" else "WARN", "detail": f"{v['value']:.4g} vs {v['side']} {v['bound']:.4g}"})
    if cash_target is not None and nav:
        after_cash = float(cash + (lines["cash"].sum() if len(lines) else 0.0)) / nav
        off = after_cash - cash_target
        checks.append({"check": "cash_target", "scope": "portfolio", "symbol": None, "status": "PASS" if abs(off) <= cash_target_tolerance else "WARN", "detail": f"cash after trades {after_cash:.2%} against a target of {cash_target:.2%} ({off:+.2%})"})
    if analytics is not None and len(tw):
        a = analytics.reindex(tw.index)
        wabs = tw.abs()
        for col in a.columns:
            need = float((coverage_min or {}).get(col, 0.95))
            num = a[col].notna() & a[col].map(lambda x: not isinstance(x, float) or np.isfinite(x))
            share = float(wabs[num].sum() / wabs.sum()) if wabs.sum() > 0 else 1.0
            if share < need:
                missing_ = list(a.index[~num & (wabs > 0)][:5])
                checks.append({"check": f"coverage:{col}", "scope": "portfolio", "symbol": None, "status": "FAIL", "detail": f"{col} covers {share:.1%} of the target book by weight, below {need:.0%} (missing e.g. {missing_})"})
    if carried is not None and len(carried):
        for r in carried.itertuples():
            checks.append({"check": "analytics_carried", "scope": "line", "symbol": r.symbol, "status": "WARN", "detail": f"{r.field} carried from {r.from_date}"})
    if cash_bounds is not None and nav:
        after_cash = float(cash + (lines["cash"].sum() if len(lines) else 0.0)) / nav
        lo, hi = cash_bounds
        if lo is not None and after_cash < lo:
            checks.append({"check": "cash_min", "scope": "portfolio", "symbol": None, "status": "FAIL", "detail": f"cash after trades {after_cash:.2%} below {lo:.2%}"})
        if hi is not None and after_cash > hi:
            checks.append({"check": "cash_max", "scope": "portfolio", "symbol": None, "status": "WARN", "detail": f"cash after trades {after_cash:.2%} above {hi:.2%}"})
    validation = pd.DataFrame(checks, columns=["check", "scope", "symbol", "status", "detail"])
    if validation.empty:
        validation = pd.DataFrame([{"check": "all", "scope": "portfolio", "symbol": None, "status": "PASS", "detail": f"{len(lines)} orders validated"}])
    from nautilus_fi.propose.live import severity_of

    validation["severity"] = [severity_of(c, st, severities) for c, st in zip(validation["check"], validation["status"], strict=True)]
    return TradeProposal(as_of, portfolio, float(nav), float(cash), lines, holdings_after, validation, spec_hash, {"inputs": inputs})


def propose_from_spec(spec, data, as_of, holdings_face: pd.Series | dict, prices: pd.DataFrame, bonds: dict[str, BondSpec], cash: float, nav: float | None = None, **kwargs) -> TradeProposal:
    """Run a spec's decider and construction on the panel as of ``as_of`` and propose the trades."""
    from nautilus_fi.construct.cross_section import CrossSection
    from nautilus_fi.domain.spec import as_target
    from nautilus_fi.paper.engine import _PanelView

    panel = data.panel
    i = panel.date_pos(as_of)
    hf = pd.Series(holdings_face, dtype=float)
    px = prices.copy()
    if "mid" not in px:
        px["mid"] = (px["bid"] + px["ask"]) / 2.0
    dirty = (px["mid"] + px.get("accrued", 0.0)) / 100.0 * px.get("factor", 1.0)
    nav = nav if nav is not None else float((hf * dirty.reindex(hf.index).fillna(0.0)).sum() + cash)
    w = (hf * dirty.reindex(hf.index).fillna(0.0) / nav).astype(float)
    bench_w = data.benchmark.weight_matrix(panel, data.return_field)[i] if data.benchmark is not None else np.zeros(len(panel.ids))
    alive = panel["alive"][i] > 0 if "alive" in panel else np.isfinite(panel[data.return_field][i])
    elig = (bench_w > 0) if data.benchmark is not None else alive
    held = panel.ids.isin(w.index)
    cand = np.nonzero(elig | held | (bench_w > 0))[0]
    attrs = data.attributes.reindex(panel.ids) if data.attributes is not None else pd.DataFrame(index=panel.ids)
    view = _PanelView(panel, i, cand, panel.ids[np.nonzero(elig & alive)[0]], w, pd.Series(bench_w[bench_w > 0], index=panel.ids[bench_w > 0]), attrs.iloc[cand], nav)
    if spec.universe is not None:
        view._universe = spec.universe.apply(view, view.universe)
    target = as_target(spec.decider.decide(view), spec.construction is not None)
    if target.weights is not None:
        tw = target.weights
    else:
        cs = CrossSection.build(view.date, view.universe, view.fields(panel.fields).join(view.attributes, how="left", rsuffix="_attr"), w, view.benchmark, target.scores, view.field(data.cost_field) if data.cost_field and view.has(data.cost_field) else None, None, None, nav)
        tw = spec.construction.build(cs).weights
    return build_proposal(tw, hf, prices, bonds, cash, pd.Timestamp(as_of).date(), nav, spec_hash=spec.hash, constraints=spec.construction.constraints if spec.construction is not None else None, benchmark=view.benchmark, attributes=view.attributes, **kwargs)


def propose_from_strategy(strategy, source, as_of, holdings_face: pd.Series | dict, prices: pd.DataFrame, bonds: dict[str, BondSpec], cash: float, nav: float | None = None, symbols: dict[str, str] | None = None, coverage_fields: tuple[str, ...] = (), last_decision=None, futures: dict | None = None, futures_prices: pd.Series | dict | None = None, current_contracts: pd.Series | dict | None = None, **kwargs) -> TradeProposal:
    """A kernel ``Strategy``'s trades at ``as_of`` from the book: ``decide_at`` runs the universe,
    signal, construction, exit policy and overlay sizing a backtest runs, once, from the book's
    weights (holdings at dirty mid), and ``build_proposal`` sizes and checks the orders. After
    rounding, the strategy's own constraints are checked on the post-trade book. ``symbols``:
    book symbol -> the source's instrument id, where they differ. ``coverage_fields``: analytics
    whose coverage of the target book is checked (the signal always is). ``futures`` (symbol ->
    ``FutureTerms``) with ``futures_prices``: the overlay's notionals become futures lines, from
    ``current_contracts``. ``last_decision``: the previous decision date (a ``Triggered``
    schedule's threshold). The decision is kept in ``extra["decision"]``."""
    from nautilus_fi.kernel.engine import decide_at
    from nautilus_fi.propose.live import add_futures, append_checks, contracts_from_overlays

    hf = pd.Series(holdings_face, dtype=float)
    px = prices.copy()
    if "mid" not in px:
        px["mid"] = (px["bid"] + px["ask"]) / 2.0
    dirty = (px["mid"] + px.get("accrued", 0.0)) / 100.0 * px.get("factor", 1.0)
    nav = nav if nav is not None else float((hf * dirty.reindex(hf.index).fillna(0.0)).sum() + cash)
    to_id = dict(symbols or {})
    to_sym = {v: k for k, v in to_id.items()}
    w = hf * dirty.reindex(hf.index).fillna(0.0) / nav
    w.index = [to_id.get(str(x), str(x)) for x in w.index]
    ov_now = None
    if futures and current_contracts is not None and futures_prices is not None:
        fp = pd.Series(futures_prices, dtype=float)
        ov_now = {k: n * futures[k].exposure(float(fp[k])) / nav for k, n in pd.Series(current_contracts, dtype=float).items() if k in futures and k in fp.index}
    d = decide_at(strategy, source, as_of, w, nav, overlays=ov_now, last_decision=last_decision, fields=tuple(coverage_fields))
    tw = d.weights.copy()
    tw.index = [to_sym.get(str(x), str(x)) for x in tw.index]
    an = d.analytics.reindex(d.weights.index)
    an.index = tw.index
    p = build_proposal(tw, hf, prices, bonds, cash, pd.Timestamp(as_of).date(), nav, spec_hash=strategy.fingerprint, analytics=an, **kwargs)
    p.extra.update({"decision": d, "symbols": to_id, "schedule": str(strategy.resolved().schedule)})
    p.validation = append_checks(p.validation, strategy_checks(p), kwargs.get("severities"))
    if futures:
        if futures_prices is None:
            raise ValueError("futures lines need futures_prices")
        contracts = contracts_from_overlays(d.overlays, nav, futures, futures_prices)
        p = add_futures(p, contracts, futures, futures_prices, current_contracts, severities=kwargs.get("severities"))
    return p


def strategy_checks(p: TradeProposal) -> list[dict]:
    """The checks a kernel strategy's proposal adds to the line checks: a decision that kept the
    book or relaxed limits (approval), an off-schedule run, and the strategy's own constraints on
    the post-trade book after rounding."""
    d = p.extra.get("decision")
    if d is None:
        return []
    to_id = p.extra.get("symbols") or {}
    checks = []
    if d.action != "rebalance":
        checks.append({"check": "decision", "scope": "portfolio", "symbol": None, "status": "WARN", "detail": f"{d.action}: {d.log.get('reason') or d.log.get('note') or d.log.get('relaxed_constraints') or 'the book is kept'}"})
    if not d.scheduled:
        checks.append({"check": "schedule", "scope": "portfolio", "symbol": None, "status": "WARN", "detail": f"{d.as_of.date()} is not a decision date of {p.extra.get('schedule')}"})
    after = p.holdings_after.set_index("symbol")["weight"] if len(p.holdings_after) else pd.Series(dtype=float)
    after.index = [to_id.get(str(x), str(x)) for x in after.index]
    for v in d.check(after):
        checks.append({"check": f"constraint:{v['constraint']}", "scope": "portfolio", "symbol": v.get("group"), "status": "FAIL" if v.get("severity") == "HARD" else "WARN", "detail": f"after rounding {v['value']:.4g} vs {v['side']} {v['bound']:.4g}"})
    return checks


__all__ = ["TradeProposal", "build_proposal", "propose_from_spec", "propose_from_strategy", "strategy_checks"]
