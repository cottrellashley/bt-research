"""Propose: turn a target portfolio into a trade proposal a desk can execute.

The same construction output the engines simulate becomes real orders: target weights against
current holdings (from PORT or the custodian), rounded to each bond's minimum piece and increment,
priced at bid (sells) / ask (buys) dirty, with settlement dates from the dated settlement
conventions and the cash impact per settlement date. ``validate`` runs the pre-trade checks (the
PREP role): pieces and increments, restricted names, missing or stale prices, shorts, cash by
settlement date, and the construction grammar's constraints on the post-trade portfolio.
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
        return self.validation.empty or not (self.validation["status"] == "FAIL").any()

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


def build_proposal(target_weights: pd.Series | dict, holdings_face: pd.Series | dict, prices: pd.DataFrame, bonds: dict[str, BondSpec], cash: float, as_of: date, nav: float | None = None, portfolio: str = "PORTFOLIO", settlement: SettlementConventions | None = None, restricted: set[str] | None = None, min_trade_face: float = 0.0, max_price_age_days: int = 3, constraints=None, benchmark: pd.Series | None = None, attributes: pd.DataFrame | None = None, allow_shorts: bool = False, spec_hash: str | None = None) -> TradeProposal:
    """``prices``: index symbol, columns ``bid`` / ``ask`` (clean, per 100 current face), optional
    ``mid``, ``accrued`` (per 100), ``factor``, ``price_date``. ``target_weights``: fractions of NAV."""
    settlement = settlement or SettlementConventions()
    restricted = set(restricted or ())
    tw = pd.Series(target_weights, dtype=float)
    hf = pd.Series(holdings_face, dtype=float)
    px = prices.copy()
    if "mid" not in px:
        px["mid"] = (px["bid"] + px["ask"]) / 2.0
    for col, default in (("accrued", 0.0), ("factor", 1.0)):
        if col not in px:
            px[col] = default
    dirty_mid = (px["mid"] + px["accrued"]) / 100.0 * px["factor"]
    if nav is None:
        nav = float((hf * dirty_mid.reindex(hf.index).fillna(0.0)).sum() + cash)
    checks: list[dict] = []
    rows = []
    syms = sorted(set(tw.index) | set(hf.index))
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
        target_face = b.round_face(float(tw.get(sym, 0.0)) * nav / dm) if dm > 0 else 0.0
        delta = target_face - cur
        if abs(delta) < 1e-9:
            continue
        closing = abs(target_face) < 1e-9
        if not closing and abs(delta) < max(b.min_piece, min_trade_face):
            checks.append({"check": "below_min_trade", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"change {delta:,.0f} below the minimum trade; skipped"})
            continue
        face = abs(b.round_face(delta)) if not closing else abs(cur)
        if face <= 0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        clean = float(px.at[sym, "ask"] if side == "BUY" else px.at[sym, "bid"])
        acc = float(px.at[sym, "accrued"])
        fct = float(px.at[sym, "factor"])
        dirty = clean + acc
        settle = settlement.settle_date(b.asset_type, b.currency, as_of, b.calendar) if b.settlement_lag_days is None else b.settle_date(as_of)
        ident, id_type = _ident(b)
        cash_amt = (-1.0 if side == "BUY" else 1.0) * face * dirty / 100.0 * fct
        rows.append({"order_id": f"{portfolio}-{as_of:%Y%m%d}-{len(rows) + 1:04d}", "portfolio": portfolio, "symbol": sym, "security_id": ident, "id_type": id_type, "side": side, "face": face, "price_clean": clean, "accrued": acc, "price_dirty": dirty, "factor": fct, "cash": cash_amt, "currency": b.currency, "trade_date": as_of, "settle_date": settle, "min_piece": b.min_piece, "increment": b.increment, "weight_before": cur * dm / nav if nav else 0.0, "weight_after": target_face * dm / nav if nav else 0.0, "reason": "rebalance to target"})
        if sym in restricted and side == "BUY":
            checks.append({"check": "restricted_list", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "buy of a restricted name"})
        if face < b.min_piece and not closing:
            checks.append({"check": "min_piece", "scope": "line", "symbol": sym, "status": "FAIL", "detail": f"{face:,.0f} below minimum piece {b.min_piece:,.0f}"})
        if b.increment and (round(face / b.increment) * b.increment - face) > 1e-6 and not closing:
            checks.append({"check": "increment", "scope": "line", "symbol": sym, "status": "FAIL", "detail": f"{face:,.0f} not a multiple of {b.increment:,.0f}"})
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
    validation = pd.DataFrame(checks, columns=["check", "scope", "symbol", "status", "detail"])
    if validation.empty:
        validation = pd.DataFrame([{"check": "all", "scope": "portfolio", "symbol": None, "status": "PASS", "detail": f"{len(lines)} orders validated"}])
    return TradeProposal(as_of, portfolio, float(nav), float(cash), lines, holdings_after, validation, spec_hash)


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


__all__ = ["TradeProposal", "build_proposal", "propose_from_spec"]
