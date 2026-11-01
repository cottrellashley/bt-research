"""The library side of an operational portfolio service: a stable target-book export, book
reconciliation and the interface a store implements.

Whether bt-research runs the paper-portfolio service itself or hands off to one (paperport,
systematicpm), the hand-off is the same three things:

* ``target_book``: what the strategy wants held, in a fixed, versioned schema
  (``TARGET_BOOK_COLUMNS``) that ``write_target_book`` / ``read_target_book`` round-trip with a
  content hash, so a consumer can refuse a file it does not understand;
* ``reconcile_book``: the model book against the store's (or PORT's) book, with the gold copy
  named, adjustment rows excluded and a zero-vs-nonzero mismatch blocking; ``reconcile_returns``
  compares return series in bp (an internal series against a PREP-style export);
* ``BookStore``: the protocol the operational store implements (positions, cash, fills, saving a
  target book). ``FileBookStore`` implements it on a directory, for tests and local use.

Official records link to the store through ``links`` (``governance.LINK_KEYS``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

TARGET_BOOK_VERSION = "bt-research/target-book/1"
TARGET_BOOK_COLUMNS = ("as_of", "portfolio", "strategy", "fingerprint", "symbol", "security_id", "id_type", "instrument", "currency", "current_face", "target_face", "trade_face", "current_weight", "target_weight", "price_dirty_mid", "record_id")
_NUMERIC = ("current_face", "target_face", "trade_face", "current_weight", "target_weight", "price_dirty_mid")


def target_book(source, strategy: str | None = None, record_id: str | None = None, portfolio: str | None = None) -> pd.DataFrame:
    """The target book in the stable schema, from a ``TradeProposal`` (faces, weights and the
    trades) or a kernel ``Decision`` (weights only: faces are blank). One row per instrument held
    before or after."""
    from nautilus_fi.kernel.engine import Decision

    if isinstance(source, Decision):
        idx = source.weights.index.union(source.current.index)
        cur = source.current.reindex(idx).fillna(0.0)
        tgt = source.weights.reindex(idx).fillna(0.0)
        df = pd.DataFrame({"symbol": idx.astype(str), "current_weight": cur.to_numpy(), "target_weight": tgt.to_numpy()})
        as_of = source.as_of.date()
        fp = None
    else:
        p = source
        before = pd.Series(p.extra.get("inputs", {}).get("holdings_face", {}), dtype=float)
        after = p.holdings_after.set_index("symbol")
        lines = p.lines.set_index("symbol") if len(p.lines) else pd.DataFrame(columns=["face", "side", "security_id", "id_type", "instrument", "currency", "price_dirty"])
        idx = before.index.union(after.index).union(lines.index)
        prices = p.extra.get("inputs", {}).get("prices")
        mid = None
        if prices is not None:
            px = prices.copy()
            if "mid" not in px:
                px["mid"] = (px["bid"] + px["ask"]) / 2.0
            mid = ((px["mid"] + px.get("accrued", 0.0)) / 100.0 * px.get("factor", 1.0)).reindex(idx)
        trade = pd.Series(0.0, index=idx)
        if len(lines):
            trade.loc[lines.index] = np.where(lines["side"] == "BUY", 1.0, -1.0) * lines["face"].astype(float)
        bonds = p.extra.get("inputs", {}).get("bonds", {})
        from nautilus_fi.propose.proposal import _ident

        ids = [(_ident(bonds[s]) if s in bonds else (s, "TICKER")) for s in idx]
        df = pd.DataFrame({"symbol": idx.astype(str), "security_id": [i[0] for i in ids], "id_type": [i[1] for i in ids], "instrument": [lines.at[s, "instrument"] if s in lines.index and "instrument" in lines else "BOND" for s in idx], "currency": [bonds[s].currency if s in bonds else None for s in idx], "current_face": before.reindex(idx).fillna(0.0).to_numpy(), "target_face": after["face"].reindex(idx).fillna(0.0).to_numpy(), "trade_face": trade.to_numpy(), "target_weight": after["weight"].reindex(idx).fillna(0.0).to_numpy(), "price_dirty_mid": mid.to_numpy() if mid is not None else np.nan})
        df["current_weight"] = df["current_face"] * df["price_dirty_mid"] / p.nav if p.nav else np.nan
        as_of = p.as_of
        fp = p.spec_hash
        portfolio = portfolio or p.portfolio
    df["as_of"] = str(as_of)
    df["portfolio"] = portfolio
    df["strategy"] = strategy
    df["fingerprint"] = fp
    df["record_id"] = record_id
    for c in TARGET_BOOK_COLUMNS:
        if c not in df:
            df[c] = np.nan if c in _NUMERIC else None
    return df[list(TARGET_BOOK_COLUMNS)].sort_values("symbol").reset_index(drop=True)


def _digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, float_format="%.12g").encode()).hexdigest()


def write_target_book(book: pd.DataFrame, path: str | Path) -> Path:
    """The book as CSV with a ``.schema.json`` beside it: the schema version, the columns and the
    content hash (a consumer checks all three)."""
    check_target_book(book)
    path = Path(path)
    book.to_csv(path, index=False, float_format="%.12g")
    path.with_suffix(".schema.json").write_text(json.dumps({"schema": TARGET_BOOK_VERSION, "columns": list(TARGET_BOOK_COLUMNS), "rows": len(book), "sha256": _digest(book)}, indent=2))
    return path


def read_target_book(path: str | Path) -> pd.DataFrame:
    """A target book written by ``write_target_book``; refuses another schema version, other
    columns or changed content."""
    path = Path(path)
    meta_path = path.with_suffix(".schema.json")
    if not meta_path.exists():
        raise ValueError(f"{path}: no {meta_path.name}: not a target-book export")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema") != TARGET_BOOK_VERSION:
        raise ValueError(f"{path}: schema {meta.get('schema')!r}, this library reads {TARGET_BOOK_VERSION!r}")
    df = pd.read_csv(path, dtype={"symbol": str, "security_id": str, "as_of": str, "record_id": str, "fingerprint": str, "portfolio": str, "strategy": str})
    df = df.astype({c: object for c in TARGET_BOOK_COLUMNS if c not in _NUMERIC}).where(df.notna(), None)
    for c in _NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    check_target_book(df)
    if _digest(df) != meta.get("sha256"):
        raise ValueError(f"{path}: the content does not match its recorded hash: the file was changed after export")
    return df


def check_target_book(book: pd.DataFrame) -> None:
    cols = tuple(book.columns)
    if cols != TARGET_BOOK_COLUMNS:
        raise ValueError(f"target book columns {list(cols)} are not the schema's {list(TARGET_BOOK_COLUMNS)}")
    if book["symbol"].duplicated().any():
        raise ValueError(f"duplicate symbols in the target book: {sorted(book.loc[book['symbol'].duplicated(), 'symbol'])[:5]}")


def reconcile_book(model: pd.Series | dict | pd.DataFrame, actual: pd.Series | dict | pd.DataFrame, gold: str = "actual", tolerance_face: float = 0.5, exclude_kinds: tuple[str, ...] = ("ADJUSTMENT",)) -> pd.DataFrame:
    """Model face against the store's (or PORT's) face, by symbol. A frame side may carry a
    ``kind`` column: rows of ``exclude_kinds`` (manual adjustments) are left out of the comparison
    and of any upload. ``gold`` names the side that is right when they differ (``actual``: PORT
    as the gold copy; the model is re-based to it). Status per symbol: MATCH, BREAK (both held,
    face differs), BLOCK (one side holds it and the other holds zero or nothing: an upload must
    stop) and ``action`` says what follows (none, rebase_model, correct_store, investigate)."""
    m, a = _faces(model, exclude_kinds), _faces(actual, exclude_kinds)
    idx = m.index.union(a.index)
    out = pd.DataFrame({"model": m.reindex(idx), "actual": a.reindex(idx)})
    mz = out["model"].fillna(0.0).abs() <= tolerance_face
    az = out["actual"].fillna(0.0).abs() <= tolerance_face
    out["difference"] = out["actual"].fillna(0.0) - out["model"].fillna(0.0)
    out["status"] = "MATCH"
    out.loc[(out["difference"].abs() > tolerance_face) & ~mz & ~az, "status"] = "BREAK"
    out.loc[mz != az, "status"] = "BLOCK"
    fix = "rebase_model" if gold == "actual" else "correct_store"
    out["action"] = np.where(out["status"] == "MATCH", "none", np.where(out["status"] == "BLOCK", "investigate", fix))
    out.attrs["blocked"] = bool((out["status"] == "BLOCK").any())
    out.attrs["gold"] = gold
    return out.reset_index(names="symbol")


def _faces(x, exclude_kinds) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        df = x
        if "kind" in df:
            df = df[~df["kind"].astype(str).str.upper().isin([k.upper() for k in exclude_kinds])]
        return df.groupby(df["symbol"].astype(str))["face"].sum().astype(float)
    return pd.Series(x, dtype=float)


def reconcile_returns(internal: pd.Series, external: pd.Series, tolerance_bp: float = 1.0) -> dict:
    """Return series against an external one (a PREP-style export), on common dates, in bp:
    mean, median, mean absolute and maximum absolute daily error, the share of days beyond
    ``tolerance_bp``, the cumulative difference and the correlation."""
    i, e = internal.align(external, join="inner")
    ok = i.notna() & e.notna()
    i, e = i[ok].astype(float), e[ok].astype(float)
    err = (i - e) * 1e4
    if not len(err):
        return {"n": 0}
    return {"n": len(err), "mean_bp": float(err.mean()), "median_bp": float(err.median()), "mean_abs_bp": float(err.abs().mean()), "max_abs_bp": float(err.abs().max()), "worst_date": str(err.abs().idxmax().date()) if hasattr(err.abs().idxmax(), "date") else str(err.abs().idxmax()), "share_beyond_tolerance": float((err.abs() > tolerance_bp).mean()), "cumulative_diff_bp": float(((1 + i).prod() - (1 + e).prod()) * 1e4), "correlation": float(i.corr(e)) if len(err) > 2 else None}


@runtime_checkable
class BookStore(Protocol):
    """What bt-research needs from an operational store (paperport, systematicpm, or its own)."""

    def positions(self, portfolio: str, as_of) -> pd.Series: ...           # face by symbol
    def cash(self, portfolio: str, as_of) -> float: ...
    def fills(self, portfolio: str, as_of) -> pd.DataFrame: ...             # order_id, symbol, side, face, price
    def save_target(self, book: pd.DataFrame) -> str: ...                  # returns the store's id for it


class FileBookStore:
    """``BookStore`` on a directory: ``<root>/<portfolio>/<as_of>/{positions,fills}.csv``,
    ``cash.json`` and ``target_book.csv`` (create-only: an existing target book is never
    overwritten)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _dir(self, portfolio: str, as_of) -> Path:
        return self.root / portfolio / str(pd.Timestamp(as_of).date())

    def positions(self, portfolio: str, as_of) -> pd.Series:
        f = self._dir(portfolio, as_of) / "positions.csv"
        if not f.exists():
            return pd.Series(dtype=float)
        df = pd.read_csv(f, dtype={"symbol": str})
        return _faces(df, ("ADJUSTMENT",))

    def cash(self, portfolio: str, as_of) -> float:
        f = self._dir(portfolio, as_of) / "cash.json"
        return float(json.loads(f.read_text())["cash"]) if f.exists() else 0.0

    def fills(self, portfolio: str, as_of) -> pd.DataFrame:
        f = self._dir(portfolio, as_of) / "fills.csv"
        return pd.read_csv(f, dtype={"symbol": str, "order_id": str}) if f.exists() else pd.DataFrame(columns=["order_id", "symbol", "side", "face", "price"])

    def save_target(self, book: pd.DataFrame) -> str:
        check_target_book(book)
        as_of, portfolio = str(book["as_of"].iloc[0]), str(book["portfolio"].iloc[0])
        d = self._dir(portfolio, as_of)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "target_book.csv"
        if path.exists():
            raise FileExistsError(f"{path} exists: target books are create-only (a re-run writes a new as-of or portfolio)")
        write_target_book(book, path)
        return f"{portfolio}/{as_of}/target_book"

    def write_positions(self, portfolio: str, as_of, positions: pd.Series | pd.DataFrame, cash: float | None = None) -> None:
        d = self._dir(portfolio, as_of)
        d.mkdir(parents=True, exist_ok=True)
        df = positions if isinstance(positions, pd.DataFrame) else pd.DataFrame({"symbol": positions.index, "face": positions.to_numpy()})
        df.to_csv(d / "positions.csv", index=False)
        if cash is not None:
            (d / "cash.json").write_text(json.dumps({"cash": float(cash)}))

    def write_fills(self, portfolio: str, as_of, fills: pd.DataFrame) -> None:
        d = self._dir(portfolio, as_of)
        d.mkdir(parents=True, exist_ok=True)
        fills.to_csv(d / "fills.csv", index=False)


__all__ = ["TARGET_BOOK_COLUMNS", "TARGET_BOOK_VERSION", "BookStore", "FileBookStore", "check_target_book", "read_target_book", "reconcile_book", "reconcile_returns", "target_book", "write_target_book"]
