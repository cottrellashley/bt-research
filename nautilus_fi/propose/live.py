"""Live proposals: check severities, futures lines, PM feedback, external books and fills.

A proposal's checks carry one of four severities: ``blocker`` (the file must not be sent),
``approval_required`` (it may be sent once a PM or compliance approves), ``warning`` and ``info``.
``severity_of`` maps a check and its status to one; ``build_proposal(severities=...)`` overrides it
per check. Futures lines (``add_futures``) are sized in whole contracts from overlay notionals and
checked for readiness: terms, a fresh price, first notice and margin. ``apply_feedback`` re-runs a
proposal with the PM's actions (approve, reject, defer, resize). External books state holdings in
face, original notional, units or market value (``book_to_face``); vendor analytics published after
the close are carried from T-1 with a warning (``carry_analytics``); and fills are reconciled to
the orders (``reconcile_fills``), not only positions and cash.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

SEVERITIES = ("blocker", "approval_required", "warning", "info", "approved")

# check (or "prefix:") -> severity by status; anything else: FAIL -> blocker, WARN -> warning, PASS -> info
_RULES: dict[str, dict[str, str]] = {
    "constraint:": {"FAIL": "blocker", "WARN": "approval_required"},       # hard / soft limits after rounding
    "coverage:": {"FAIL": "approval_required"},                             # signal or analytics missing for part of the book
    "stale_price": {"WARN": "approval_required"},
    "futures_roll": {"WARN": "approval_required"},
    "futures_price_stale": {"WARN": "approval_required"},
    "cash_max": {"WARN": "warning"},
    "pm_feedback": {"PASS": "info", "WARN": "info"},
    "decision": {"WARN": "approval_required"},                              # the strategy kept the book (aborted, infeasible, pushed)
    "schedule": {"WARN": "warning"},                                        # as_of is not one of the schedule's decision dates
}


def severity_of(check: str, status: str, overrides: dict | None = None) -> str:
    """The severity of a check with ``status`` (PASS / WARN / FAIL): ``overrides`` by check name
    (or ``"prefix:"``) first, then the defaults."""
    for rules in (overrides or {}, _RULES):
        rule = rules.get(check)
        if rule is None and ":" in check:
            rule = rules.get(check.split(":", 1)[0] + ":")
        if rule is not None:
            if isinstance(rule, str):
                return _valid(rule)
            if status in rule:
                return _valid(rule[status])
    return {"FAIL": "blocker", "WARN": "warning"}.get(str(status).upper(), "info")


def _valid(sev: str) -> str:
    if sev not in SEVERITIES:
        raise ValueError(f"unknown severity {sev!r}; known: {SEVERITIES}")
    return sev


def append_checks(validation: pd.DataFrame, checks: list[dict], severities: dict | None = None) -> pd.DataFrame:
    """``validation`` with more checks (and their severities); the placeholder "all passed" row
    goes once a check that did not pass is added."""
    if not checks:
        return validation
    v = pd.DataFrame(checks, columns=["check", "scope", "symbol", "status", "detail"])
    v["severity"] = [severity_of(c, st, severities) for c, st in zip(v["check"], v["status"], strict=True)]
    out = pd.concat([validation, v], ignore_index=True)
    if (v["status"] != "PASS").any():
        out = out[out["check"] != "all"].reset_index(drop=True)
    return out


# ------------------------------------------------------------------------------------------ futures
@dataclass(frozen=True)
class FutureTerms:
    """What an order line for a listed future needs. ``contract_notional``: currency face per
    contract (100,000 for a US 10y note future); the exposure of one contract is
    ``contract_notional * price / 100``."""

    symbol: str
    contract_notional: float
    currency: str = "USD"
    first_notice: date | None = None          # roll before it (or before last trade, for cash-settled contracts)
    last_trade: date | None = None
    security_id: str | None = None            # the OMS identifier (e.g. a Bloomberg ticker)
    id_type: str = "TICKER"
    margin_init: float = 0.0                  # initial margin per contract, in currency

    def exposure(self, price: float) -> float:
        return self.contract_notional * price / 100.0


def contracts_from_overlays(overlays: pd.Series | dict, nav: float, futures: dict[str, FutureTerms], prices: pd.Series | dict) -> pd.Series:
    """Overlay notionals as shares of NAV (a kernel ``Decision.overlays``) -> contracts, unrounded."""
    ov = pd.Series(overlays, dtype=float)
    px = pd.Series(prices, dtype=float)
    out = {}
    for k, x in ov.items():
        if k not in futures or k not in px.index or not np.isfinite(px[k]):
            out[k] = np.nan
            continue
        out[k] = x * nav / futures[k].exposure(float(px[k]))
    return pd.Series(out, dtype=float)


def add_futures(proposal, target_contracts: pd.Series | dict, futures: dict[str, FutureTerms], prices: pd.Series | dict, current: pd.Series | dict | None = None, price_dates: dict | None = None, roll_days: int = 5, max_price_age_days: int = 1, severities: dict | None = None):
    """``proposal`` with futures lines: whole contracts from ``target_contracts`` against
    ``current`` (rounded to the nearest contract, reported when rounding moves it), and readiness
    checks: known terms, a price (fresh within ``max_price_age_days``), not inside ``roll_days``
    business days of first notice (or past it), and initial margin covered by cash after the
    trades. Futures trade without cash at the trade; margin is reported, not booked."""
    tgt = pd.Series(target_contracts, dtype=float)
    cur = pd.Series(current if current is not None else {}, dtype=float)
    px = pd.Series(prices, dtype=float)
    as_of = proposal.as_of
    checks, rows = [], []
    n0 = len(proposal.lines)
    margin = 0.0
    for sym in sorted(set(tgt.index) | set(cur.index)):
        want, have = float(tgt.get(sym, 0.0)), float(cur.get(sym, 0.0))
        terms = futures.get(sym)
        if terms is None:
            checks.append({"check": "futures_terms", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "no contract terms (notional, first notice)"})
            continue
        if not np.isfinite(want):
            checks.append({"check": "futures_price", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "no price to size the hedge"})
            continue
        n = float(np.round(want))
        if abs(n - want) > 1e-9:
            checks.append({"check": "futures_lot", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"{want:,.2f} contracts rounded to {n:,.0f}"})
        p = float(px.get(sym, np.nan))
        if not np.isfinite(p):
            checks.append({"check": "futures_price", "scope": "line", "symbol": sym, "status": "FAIL", "detail": "no price"})
            continue
        if price_dates and sym in price_dates and (as_of - pd.Timestamp(price_dates[sym]).date()).days > max_price_age_days:
            checks.append({"check": "futures_price_stale", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"price from {pd.Timestamp(price_dates[sym]).date()}"})
        notice = terms.first_notice or terms.last_trade
        if notice is not None and n != 0:
            left = int(np.busday_count(as_of, notice))
            if left <= 0:
                checks.append({"check": "futures_expiry", "scope": "line", "symbol": sym, "status": "FAIL", "detail": f"first notice / last trade {notice} has passed: hold the next contract"})
            elif left <= roll_days:
                checks.append({"check": "futures_roll", "scope": "line", "symbol": sym, "status": "WARN", "detail": f"first notice in {left} business days ({notice}): roll into the next contract"})
        margin += abs(n) * terms.margin_init
        trade = n - have
        if abs(trade) < 1e-9:
            continue
        side = "BUY" if trade > 0 else "SELL"
        e = terms.exposure(p)
        rows.append({"order_id": f"{proposal.portfolio}-{as_of:%Y%m%d}-{n0 + len(rows) + 1:04d}", "portfolio": proposal.portfolio, "symbol": sym, "security_id": terms.security_id or sym, "id_type": terms.id_type, "side": side, "face": abs(trade), "price_clean": p, "accrued": 0.0, "price_dirty": p, "factor": 1.0, "cash": 0.0, "currency": terms.currency, "trade_date": as_of, "settle_date": as_of, "min_piece": 1.0, "increment": 1.0, "weight_before": have * e / proposal.nav if proposal.nav else 0.0, "weight_after": n * e / proposal.nav if proposal.nav else 0.0, "reason": "overlay", "instrument": "FUTURE", "notional": abs(trade) * terms.contract_notional, "contracts": abs(trade)})
    if margin > 0 and margin > proposal.cash_after + 1e-6:
        checks.append({"check": "futures_margin", "scope": "portfolio", "symbol": None, "status": "FAIL", "detail": f"initial margin {margin:,.0f} exceeds cash after trades {proposal.cash_after:,.0f}"})
    lines = pd.concat([proposal.lines, pd.DataFrame(rows)], ignore_index=True) if rows else proposal.lines
    validation = append_checks(proposal.validation, checks, severities)
    extra = {**proposal.extra, "futures": {"target_contracts": tgt, "current": cur, "futures": futures, "prices": px, "price_dates": price_dates, "roll_days": roll_days, "max_price_age_days": max_price_age_days, "severities": severities}, "futures_margin": margin}
    return replace(proposal, lines=lines, validation=validation, extra=extra)


# ------------------------------------------------------------------------------------------ PM feedback
FEEDBACK_ACTIONS = ("approve", "reject", "defer", "resize")


def apply_feedback(proposal, feedback: pd.DataFrame | list[dict]):
    """The proposal re-run with a PM's actions. ``feedback`` rows: ``symbol`` and ``action``
    (approve | reject | defer | resize, with ``face`` for a resize and an optional ``side``), or
    ``check`` and ``action="approve"`` for a portfolio-level approval; optional ``comment``. A
    rejected or deferred line is dropped, a resized one is validated at the PM's size (not
    re-rounded), and approvals turn the matching ``approval_required`` checks into ``approved``.
    Unlisted lines stand as proposed. The actions are recorded as ``pm_feedback`` checks."""
    from nautilus_fi.propose.proposal import build_proposal

    inputs = proposal.extra.get("inputs")
    if inputs is None:
        raise ValueError("the proposal does not keep its inputs (not built by build_proposal): rebuild it, then apply the feedback")
    fb = pd.DataFrame(feedback)
    if fb.empty:
        return proposal
    if "action" not in fb:
        raise ValueError("feedback needs an 'action' column")
    lines = proposal.lines.set_index("symbol") if len(proposal.lines) else pd.DataFrame(columns=["side", "face", "instrument"])
    fut = proposal.extra.get("futures")
    fut_tgt = fut["target_contracts"].copy() if fut else pd.Series(dtype=float)
    fut_cur = fut["current"] if fut else pd.Series(dtype=float)
    overrides = dict(inputs.get("trade_overrides") or {})
    notes, ok_syms, ok_checks, deferred = [], set(), set(), []
    for r in fb.to_dict("records"):
        action = str(r["action"]).lower()
        if action not in FEEDBACK_ACTIONS:
            raise ValueError(f"unknown feedback action {r['action']!r}; known: {FEEDBACK_ACTIONS}")
        sym = r.get("symbol")
        sym = None if sym is None or (isinstance(sym, float) and np.isnan(sym)) else str(sym)
        comment = r.get("comment")
        comment = "" if comment is None or (isinstance(comment, float) and np.isnan(comment)) else f": {comment}"
        if sym is None:
            check = r.get("check")
            if action != "approve" or check is None or (isinstance(check, float) and np.isnan(check)):
                raise ValueError("a feedback row without a symbol approves a check: {'check': ..., 'action': 'approve'}")
            ok_checks.add(str(check))
            notes.append({"check": "pm_feedback", "scope": "portfolio", "symbol": None, "status": "PASS", "detail": f"approved {check}{comment}"})
            continue
        is_future = sym in fut_tgt.index or sym in fut_cur.index
        if sym not in lines.index and action != "resize":
            raise KeyError(f"{sym} has no line in the proposal (resize adds one: face and side)")
        face = r.get("face")
        if action == "resize" and (face is None or not np.isfinite(float(face)) or float(face) < 0):
            raise ValueError(f"resize {sym}: give the new face (contracts for a future), >= 0")
        side = r.get("side") if isinstance(r.get("side"), str) else (lines.at[sym, "side"] if sym in lines.index else None)
        if action == "resize" and side not in ("BUY", "SELL"):
            raise ValueError(f"resize {sym}: give the side (BUY or SELL) for a new line")
        if is_future:
            have = float(fut_cur.get(sym, 0.0))
            if action in ("reject", "defer"):
                fut_tgt[sym] = have
            elif action == "resize":
                fut_tgt[sym] = have + (float(face) if side == "BUY" else -float(face))
        elif action in ("reject", "defer"):
            overrides[sym] = 0.0
        elif action == "resize":
            overrides[sym] = float(face) if side == "BUY" else -float(face)
        if action == "approve":
            ok_syms.add(sym)
        if action == "defer":
            deferred.append(sym)
        notes.append({"check": "pm_feedback", "scope": "line", "symbol": sym, "status": "PASS", "detail": f"{action}" + (f" to {float(face):,.0f} {side}" if action == "resize" else "") + comment})
    new = build_proposal(**{**inputs, "trade_overrides": overrides})
    if proposal.extra.get("decision") is not None:                      # a kernel strategy's proposal: its checks again, on the new book
        from nautilus_fi.propose.proposal import strategy_checks

        new.extra.update({k: proposal.extra[k] for k in ("decision", "symbols", "schedule") if k in proposal.extra})
        new.validation = append_checks(new.validation, strategy_checks(new), inputs.get("severities"))
    if fut:
        new = add_futures(new, fut_tgt, fut["futures"], fut["prices"], fut["current"], fut["price_dates"], fut["roll_days"], fut["max_price_age_days"], fut["severities"])
    v = new.validation.copy()
    hit = (v["severity"] == "approval_required") & (v["symbol"].isin(ok_syms) | v["check"].isin(ok_checks))
    v.loc[hit, "severity"] = "approved"
    n = pd.DataFrame(notes)
    n["severity"] = "info"
    v = pd.concat([v, n], ignore_index=True)
    return replace(new, validation=v, extra={**new.extra, "feedback": fb, "deferred": deferred})


# ------------------------------------------------------------------------------------------ external books
BOOK_UNITS = ("face", "notional", "units", "market_value")


def book_to_face(book: pd.DataFrame, factors: pd.Series | dict | None = None, unit_size: pd.Series | dict | float | None = None, prices: pd.DataFrame | None = None) -> pd.Series:
    """An external book's holdings as current face by symbol. ``book``: ``symbol``, ``quantity``
    and ``unit``: ``face`` (current face), ``notional`` (original face: times the pool factor),
    ``units`` (times ``unit_size``: a column, a mapping or one number; an ETF share or a bond
    unit) or ``market_value`` (divided by the dirty price per unit of face from ``prices``:
    ``bid`` / ``ask`` / ``mid``, ``accrued`` and ``factor`` as ``build_proposal`` reads them).
    Several rows of one symbol are summed."""
    b = book.copy()
    if "unit" not in b:
        b["unit"] = "face"
    b["unit"] = b["unit"].str.lower()
    bad = sorted(set(b["unit"]) - set(BOOK_UNITS))
    if bad:
        raise ValueError(f"unknown book unit(s) {bad}; known: {BOOK_UNITS}")
    fct = pd.Series(factors if factors is not None else {}, dtype=float)
    out = np.zeros(len(b))
    for i, r in enumerate(b.itertuples(index=False)):
        q = float(r.quantity)
        if r.unit == "face":
            out[i] = q
        elif r.unit == "notional":
            f = getattr(r, "factor", None)
            f = float(f) if f is not None and np.isfinite(float(f)) else float(fct.get(r.symbol, np.nan))
            if not np.isfinite(f):
                raise ValueError(f"{r.symbol}: a notional holding needs the pool factor (book column 'factor' or factors=)")
            out[i] = q * f
        elif r.unit == "units":
            u = getattr(r, "unit_size", None)
            if u is None or not np.isfinite(float(u)):
                u = unit_size.get(r.symbol) if isinstance(unit_size, (dict, pd.Series)) else unit_size
            if u is None or not np.isfinite(float(u)):
                raise ValueError(f"{r.symbol}: a holding in units needs the unit size (book column 'unit_size' or unit_size=)")
            out[i] = q * float(u)
        else:
            if prices is None or r.symbol not in prices.index:
                raise ValueError(f"{r.symbol}: a holding by market value needs its price (prices=)")
            p = prices.loc[r.symbol]
            mid = float(p["mid"]) if "mid" in p and np.isfinite(p["mid"]) else (float(p["bid"]) + float(p["ask"])) / 2.0
            dirty = (mid + float(p.get("accrued", 0.0))) / 100.0 * float(p.get("factor", 1.0))
            out[i] = q / dirty
    return pd.Series(out, index=b["symbol"].astype(str).to_numpy()).groupby(level=0).sum()


def lines_in_book_units(proposal, units: dict[str, str] | str = "face", unit_size: pd.Series | dict | float | None = None) -> pd.DataFrame:
    """The order lines with ``quantity`` and ``unit`` in an external book's terms (the reverse of
    ``book_to_face``): ``face``, ``notional`` (face / factor) or ``units`` (face / unit size)."""
    lines = proposal.lines.copy()
    if lines.empty:
        return lines.assign(quantity=pd.Series(dtype=float), unit=pd.Series(dtype=str))
    unit = lines["symbol"].map(lambda s: units.get(s, "face") if isinstance(units, dict) else units)
    q = lines["face"].astype(float).copy()
    for i, (s, u) in enumerate(zip(lines["symbol"], unit, strict=True)):
        if u == "notional":
            q.iloc[i] = lines["face"].iloc[i] / lines["factor"].iloc[i] if lines["factor"].iloc[i] else np.nan
        elif u == "units":
            size = unit_size.get(s) if isinstance(unit_size, (dict, pd.Series)) else unit_size
            if size is None:
                raise ValueError(f"{s}: units need unit_size=")
            q.iloc[i] = lines["face"].iloc[i] / float(size)
        elif u != "face":
            raise ValueError(f"unknown unit {u!r}: face, notional or units")
    lines["quantity"] = q
    lines["unit"] = unit
    return lines


def carry_analytics(today: pd.DataFrame, previous: pd.DataFrame, as_of, previous_date, max_age_days: int = 1, fields=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T-1 analytics for what T has not published yet (vendor analytics land after the close):
    the missing cells of ``today`` (symbol x field) are filled from ``previous`` when
    ``previous_date`` is at most ``max_age_days`` business days before ``as_of``. Returns the
    filled frame and a report (symbol, field, from_date) for ``build_proposal(carried=...)``,
    which records each carry as a warning. Older analytics are not carried."""
    as_of, prev_d = pd.Timestamp(as_of).date(), pd.Timestamp(previous_date).date()
    age = int(np.busday_count(prev_d, as_of))
    cols = list(fields) if fields is not None else list(today.columns)
    out = today.copy()
    for c in cols:
        if c not in out:
            out[c] = np.nan
    rows = []
    if 0 < age <= max_age_days:
        prev = previous.reindex(index=out.index.union(previous.index))
        out = out.reindex(prev.index)
        for c in cols:
            if c not in prev:
                continue
            gap = out[c].isna() & prev[c].notna()
            out.loc[gap, c] = prev.loc[gap, c]
            rows += [{"symbol": s, "field": c, "from_date": prev_d} for s in out.index[gap]]
    return out, pd.DataFrame(rows, columns=["symbol", "field", "from_date"])


# ------------------------------------------------------------------------------------------ fills
def reconcile_fills(proposal, fills: pd.DataFrame, face_tolerance: float = 0.5, price_tolerance_bp: float = 25.0) -> pd.DataFrame:
    """Fills against the orders. ``fills``: ``order_id`` (or ``symbol``), ``side``, ``face`` (or
    contracts) and ``price`` (clean); several fills of one order are summed at their average
    price. One row per order and per fill that matches no order: ordered, filled, remaining,
    average price, limit, slippage in bp (positive: worse than the limit: paid more, sold for
    less), ``status`` FILLED | PARTIAL | UNFILLED | OVERFILLED | WRONG_SIDE | UNEXPECTED and
    ``price_status`` OK | PRICE_BREAK (beyond ``price_tolerance_bp``)."""
    lines = proposal.lines
    f = fills.copy()
    if "order_id" not in f:
        if not len(lines):
            f["order_id"] = None
        else:
            by_sym = dict(zip(lines["symbol"], lines["order_id"], strict=True))
            f["order_id"] = f["symbol"].map(by_sym)
    f["face"] = f["face"].astype(float)
    f["value"] = f["face"] * f["price"].astype(float)
    rows = []
    known = set(lines["order_id"]) if len(lines) else set()
    for r in lines.itertuples() if len(lines) else ():
        g = f[f["order_id"] == r.order_id]
        side_ok = g["side"].eq(r.side) if "side" in g else pd.Series(True, index=g.index)
        filled = float(g.loc[side_ok, "face"].sum())
        wrong = float(g.loc[~side_ok, "face"].sum())
        vwap = float(g.loc[side_ok, "value"].sum() / filled) if filled > 0 else np.nan
        slip = (vwap / r.price_clean - 1.0) * 1e4 * (1 if r.side == "BUY" else -1) if filled > 0 and r.price_clean else np.nan
        if wrong > 0:
            status = "WRONG_SIDE"
        elif filled <= face_tolerance:
            status = "UNFILLED"
        elif filled > r.face + face_tolerance:
            status = "OVERFILLED"
        elif filled < r.face - face_tolerance:
            status = "PARTIAL"
        else:
            status = "FILLED"
        rows.append({"order_id": r.order_id, "symbol": r.symbol, "side": r.side, "ordered": float(r.face), "filled": filled, "remaining": float(r.face) - filled, "avg_price": vwap, "limit_price": float(r.price_clean), "slippage_bp": slip, "status": status, "price_status": "PRICE_BREAK" if np.isfinite(slip) and slip > price_tolerance_bp else "OK"})
    extra = f[~f["order_id"].isin(known)]
    for r in extra.itertuples():
        rows.append({"order_id": r.order_id, "symbol": getattr(r, "symbol", None), "side": getattr(r, "side", None), "ordered": 0.0, "filled": float(r.face), "remaining": -float(r.face), "avg_price": float(r.price), "limit_price": np.nan, "slippage_bp": np.nan, "status": "UNEXPECTED", "price_status": "OK"})
    return pd.DataFrame(rows, columns=["order_id", "symbol", "side", "ordered", "filled", "remaining", "avg_price", "limit_price", "slippage_bp", "status", "price_status"])


def positions_after_fills(holdings_face: pd.Series | dict, fills: pd.DataFrame) -> pd.Series:
    """Face by symbol after the fills (``symbol``, ``side``, ``face``): the book to reconcile the
    custodian's or PORT's positions against."""
    h = pd.Series(holdings_face, dtype=float).copy()
    for r in fills.itertuples():
        h[r.symbol] = h.get(r.symbol, 0.0) + (float(r.face) if r.side == "BUY" else -float(r.face))
    return h[h.abs() > 1e-9]


__all__ = ["BOOK_UNITS", "FEEDBACK_ACTIONS", "SEVERITIES", "FutureTerms", "add_futures", "append_checks", "apply_feedback", "book_to_face", "carry_analytics", "contracts_from_overlays", "lines_in_book_units", "positions_after_fills", "reconcile_fills", "severity_of"]
