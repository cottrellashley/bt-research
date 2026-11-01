"""Output files for the order management and portfolio systems, and their reconciliation.

Layouts are column maps, not hard-coded formats: the defaults below follow common CRD order-import,
QOE order-file and Bloomberg PORT position-upload shapes, and each desk's template is a
``ColumnMap`` (output column -> proposal field, or a constant with ``=value``). Confirm the maps
against the systems' import specifications before sending files to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from nautilus_fi.propose.proposal import TradeProposal


@dataclass
class ColumnMap:
    name: str
    columns: dict[str, str]                 # output column -> source column, or "=constant"
    side_codes: dict[str, str] = field(default_factory=lambda: {"BUY": "BUY", "SELL": "SELL"})
    date_format: str = "%Y-%m-%d"
    header: bool = True                     # False: a headerless file with these columns in this order
    sep: str = ","                          # "\t" for tab-separated imports

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col, src in self.columns.items():
            if src.startswith("="):
                out[col] = src[1:]
            elif src == "side":
                out[col] = df["side"].map(self.side_codes)
            elif src in df:
                v = df[src]
                out[col] = v.map(lambda x: x.strftime(self.date_format) if hasattr(x, "strftime") else x)
            else:
                raise KeyError(f"{self.name}: source column {src!r} not in the proposal ({list(df.columns)})")
        return out


CRD_ORDERS = ColumnMap("CRD", {"ORDER_ID": "order_id", "ACCOUNT": "portfolio", "SEC_ID": "security_id", "SEC_ID_TYPE": "id_type", "TRANS_TYPE": "side", "TARGET_QTY": "face", "LIMIT_PRICE": "price_clean", "PRICE_TYPE": "=CLEAN", "CURRENCY": "currency", "TRADE_DATE": "trade_date", "SETTLE_DATE": "settle_date", "COMMENTS": "reason"}, {"BUY": "BUYL", "SELL": "SELLL"})
QOE_ORDERS = ColumnMap("QOE", {"OrderRef": "order_id", "Portfolio": "portfolio", "Identifier": "security_id", "IdentifierType": "id_type", "Side": "side", "Nominal": "face", "CleanPrice": "price_clean", "DirtyPrice": "price_dirty", "Accrued": "accrued", "Ccy": "currency", "TradeDate": "trade_date", "SettleDate": "settle_date"}, {"BUY": "B", "SELL": "S"})
QOE_TSV = ColumnMap("QOE_TSV", {"OrderRef": "order_id", "Portfolio": "portfolio", "Identifier": "security_id", "IdentifierType": "id_type", "AssetType": "instrument", "Side": "side", "Quantity": "face", "CleanPrice": "price_clean", "Ccy": "currency", "TradeDate": "trade_date", "SettleDate": "settle_date"}, {"BUY": "B", "SELL": "S"}, "%Y%m%d", header=False, sep="\t")   # fixed columns, no header: confirm against the QOE import spec
PORT_POSITIONS = ColumnMap("PORT", {"Portfolio Name": "portfolio", "Security ID": "security_id", "ID Type": "id_type", "Position": "face", "Date": "as_of"}, date_format="%Y%m%d")


def write_orders(proposal: TradeProposal, path: str | Path, layout: ColumnMap = CRD_ORDERS, header: bool | None = None, sep: str | None = None, blocked_ok: bool = False) -> Path:
    """The order file in ``layout`` (its header and separator unless ``header`` / ``sep`` say
    otherwise). A proposal with a blocker is not written unless ``blocked_ok``."""
    if not proposal.ok and not blocked_ok:
        raise ValueError(f"the proposal has blockers ({sorted(set(proposal.blockers['check']))}): resolve them, or write with blocked_ok=True for review")
    path = Path(path)
    lines = proposal.lines
    if "instrument" not in lines and len(lines):
        lines = lines.assign(instrument="BOND")
    layout.apply(lines).to_csv(path, index=False, header=layout.header if header is None else header, sep=layout.sep if sep is None else sep)
    return path


def write_port_upload(proposal: TradeProposal, path: str | Path, bonds: dict, layout: ColumnMap = PORT_POSITIONS) -> Path:
    """Post-trade positions in the PORT upload layout (face per security)."""
    from nautilus_fi.propose.proposal import _ident

    h = proposal.holdings_after.copy()
    h["portfolio"] = proposal.portfolio
    h["as_of"] = proposal.as_of
    ids = [(_ident(bonds[s]) if s in bonds else (s, "TICKER")) for s in h["symbol"]]
    h["security_id"] = [i[0] for i in ids]
    h["id_type"] = [i[1] for i in ids]
    path = Path(path)
    layout.apply(h).to_csv(path, index=False)
    return path


def write_pretrade_report(proposal: TradeProposal, path: str | Path) -> Path:
    """Pre-trade validation report (one row per check) plus a markdown summary next to it. This is
    the platform's own pre-trade check, not a Bloomberg PREP export: reconciling to PORT / PREP is
    not covered."""
    path = Path(path)
    proposal.validation.to_csv(path, index=False)
    s = proposal.summary()
    sev = proposal.validation["severity"].value_counts().to_dict() if "severity" in proposal.validation else {}
    md = [f"# Pre-trade validation: {proposal.portfolio} {proposal.as_of}", "", f"* orders: {s['n_orders']} (buys {s['buys']:,.0f}, sells {s['sells']:,.0f})", f"* cash: {s['cash_before']:,.0f} -> {s['cash_after']:,.0f}", f"* blockers: {sev.get('blocker', 0)}, approvals required: {sev.get('approval_required', 0)}, warnings: {sev.get('warning', 0)}", f"* spec: {proposal.spec_hash}", "", "| check | scope | symbol | status | severity | detail |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.check} | {r.scope} | {r.symbol or ''} | {r.status} | {getattr(r, 'severity', '')} | {r.detail} |" for r in proposal.validation.itertuples()]
    path.with_suffix(".md").write_text("\n".join(md))
    return path


def read_positions(path: str | Path, layout: ColumnMap = PORT_POSITIONS, symbol_map: dict[str, str] | None = None) -> pd.Series:
    """Positions file (e.g. a PORT or custodian extract in ``layout``) -> face by symbol."""
    df = pd.read_csv(path)
    id_col = next(k for k, v in layout.columns.items() if v == "security_id")
    face_col = next(k for k, v in layout.columns.items() if v == "face")
    syms = df[id_col].astype(str).map(lambda x: (symbol_map or {}).get(x, x))
    return pd.Series(df[face_col].astype(float).to_numpy(), index=syms).groupby(level=0).sum()


def reconcile_positions(model: pd.Series | dict, actual: pd.Series | dict, tolerance_face: float = 0.5) -> pd.DataFrame:
    m, a = pd.Series(model, dtype=float), pd.Series(actual, dtype=float)
    idx = m.index.union(a.index)
    out = pd.DataFrame({"model": m.reindex(idx), "actual": a.reindex(idx)})
    out["difference"] = out["actual"].fillna(0.0) - out["model"].fillna(0.0)
    out["status"] = "MATCH"
    out.loc[out["difference"].abs() > tolerance_face, "status"] = "BREAK"
    out.loc[out["model"].isna(), "status"] = "MISSING_IN_MODEL"
    out.loc[out["actual"].isna(), "status"] = "MISSING_IN_ACTUAL"
    return out.reset_index(names="symbol")


def reconcile_cash(model_cash: float, actual_cash: float, tolerance: float = 1.0) -> dict:
    diff = float(actual_cash - model_cash)
    return {"model": float(model_cash), "actual": float(actual_cash), "difference": diff, "status": "MATCH" if abs(diff) <= tolerance else "BREAK"}


__all__ = ["CRD_ORDERS", "PORT_POSITIONS", "QOE_ORDERS", "QOE_TSV", "ColumnMap", "read_positions", "reconcile_cash", "reconcile_positions", "write_orders", "write_port_upload", "write_pretrade_report"]
