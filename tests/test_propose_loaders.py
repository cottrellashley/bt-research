"""Trade proposals, OMS / PORT files, reconciliation, and the data loaders (SQLite stands in for
Snowflake / FIAtlas)."""

import sqlite3
from datetime import date

import pandas as pd
import pytest

from nautilus_fi.benchmarks import ConstituentBenchmark
from nautilus_fi.construct import ConstructionSpec, Quantile
from nautilus_fi.data import PointInTimeStore
from nautilus_fi.data.field_store import FieldStore
from nautilus_fi.data.loaders import (
    BLOOMBERG_INDEX_MEMBERS,
    DatasetMapping,
    SQLLoader,
    builder_from_frames,
    canonicalise,
    fiatlas_catalog,
    load_bloomberg_index_members,
    load_into_stores,
)
from nautilus_fi.domain.spec import FieldSignal, ResearchSpec
from nautilus_fi.instruments import BondSpec
from nautilus_fi.instruments.bond import Identifiers
from nautilus_fi.paper import PaperData
from nautilus_fi.propose import (
    QOE_ORDERS,
    build_proposal,
    propose_from_spec,
    read_positions,
    reconcile_cash,
    reconcile_positions,
    write_orders,
    write_port_upload,
    write_pretrade_report,
)
from nautilus_fi.synthetic.panels import SyntheticPanelConfig, generate_synthetic_panel


def _bonds():
    return {s: BondSpec(s, iss, date(2020, 1, 15), date(2030, 1, 15), 0.05, sector=sec, min_piece=1000, increment=1000, identifiers=Identifiers(cusip=f"12345{i}AB1")) for i, (s, iss, sec) in enumerate([("A", "I1", "FIN"), ("B", "I1", "FIN"), ("C", "I2", "IND"), ("D", "I3", "UTL")])}


def test_proposal_prices_validates_and_writes(tmp_path):
    bonds = _bonds()
    prices = pd.DataFrame({"bid": [99.5, 101.0, 98.0, 100.0], "ask": [99.8, 101.3, 98.4, 100.4], "accrued": [1.2, 0.5, 2.0, 0.1], "price_date": [date(2024, 6, 3)] * 3 + [date(2024, 5, 20)]}, index=list(bonds))
    attrs = pd.DataFrame({"sector": [b.sector for b in bonds.values()]}, index=list(bonds))
    prop = build_proposal({"A": 0.3, "B": 0.25, "D": 0.4}, {"A": 200_000, "C": 300_000}, prices, bonds, cash=500_000.0, as_of=date(2024, 6, 3), portfolio="FI-CORE", restricted={"D"}, constraints=[{"kind": "group", "attribute": "sector", "max": 0.5}], attributes=attrs)
    lines = prop.lines
    assert next(iter(lines["side"])) == "SELL" and set(lines["symbol"]) == {"A", "B", "C", "D"}
    buy = lines[lines["symbol"] == "B"].iloc[0]
    assert buy["price_clean"] == 101.3 and buy["price_dirty"] == pytest.approx(101.8) and buy["cash"] < 0 and buy["face"] % 1000 == 0
    assert buy["settle_date"] == date(2024, 6, 4)                                  # US corporates settle T+1 after 28 May 2024
    checks = set(prop.validation["check"])
    assert {"restricted_list", "constraint:group:sector", "stale_price"} <= checks and not prop.ok
    with pytest.raises(ValueError, match="blockers"):
        write_orders(prop, tmp_path / "blocked.csv")                               # a blocked proposal is not written by default
    crd = write_orders(prop, tmp_path / "crd.csv", blocked_ok=True)
    qoe = write_orders(prop, tmp_path / "qoe.csv", QOE_ORDERS, blocked_ok=True)
    assert pd.read_csv(crd)["TRANS_TYPE"].isin(["BUYL", "SELLL"]).all() and pd.read_csv(qoe)["Side"].isin(["B", "S"]).all()
    port = write_port_upload(prop, tmp_path / "port.csv", bonds)
    write_pretrade_report(prop, tmp_path / "pretrade.csv")
    assert (tmp_path / "pretrade.md").exists()
    back = read_positions(port, symbol_map={b.identifiers.cusip: s for s, b in bonds.items()})
    rec = reconcile_positions(prop.holdings_after.set_index("symbol")["face"], back)
    assert (rec["status"] == "MATCH").all()
    broken = reconcile_positions({"A": 100.0}, {"A": 90.0, "Z": 5.0})
    assert set(broken["status"]) == {"BREAK", "MISSING_IN_MODEL"} and reconcile_cash(100.0, 100.4)["status"] == "MATCH"


def test_proposal_cash_by_settlement_date_fails_when_short():
    bonds = _bonds()
    prices = pd.DataFrame({"bid": [99.5] * 4, "ask": [99.8] * 4}, index=list(bonds))
    prop = build_proposal({"A": 0.9}, {}, prices, bonds, cash=100_000.0, as_of=date(2024, 6, 3), nav=1_000_000.0)
    assert "cash_by_settlement" in set(prop.validation["check"])


def test_propose_from_spec_uses_the_same_construction():
    s = generate_synthetic_panel(SyntheticPanelConfig(n_bonds=120, start="2015-01-02", end="2015-06-30", seed=3))
    data = PaperData(s.panel, s.attributes, s.cash, ConstituentBenchmark("IDX", s.membership))
    spec = ResearchSpec("q5", FieldSignal("signal.value"), ConstructionSpec(Quantile(5), [{"kind": "weight", "max": 0.2}]))
    d = s.dates[-1]
    i = s.panel.date_pos(d)
    px = pd.DataFrame({"mid": s.panel["price_clean"][i], "bid": s.panel["price_clean"][i] * 0.999, "ask": s.panel["price_clean"][i] * 1.001}, index=s.panel.ids).dropna()
    bonds = {k: BondSpec(k, s.attributes.at[k, "issuer"], s.attributes.at[k, "issue_date"], s.attributes.at[k, "maturity"], float(s.attributes.at[k, "coupon"])) for k in px.index}
    prop = propose_from_spec(spec, data, d, {}, px, bonds, cash=10e6)
    assert len(prop.lines) > 5 and (prop.lines["side"] == "BUY").all() and prop.spec_hash == spec.hash


@pytest.fixture()
def vendor_db():
    con = sqlite3.connect(":memory:")
    pd.DataFrame({"CUSIP": ["V1", "V2"], "ISSUER": ["A", "B"], "ISSUE_DATE": ["2020-01-15", "2021-03-01"], "MATURITY": ["2030-01-15", "2031-03-01"], "COUPON": [5.0, 4.0]}).to_sql("REF", con, index=False)
    days = pd.bdate_range("2024-01-02", "2024-02-29")
    pd.DataFrame([{"D": str(d.date()), "CUSIP": c, "PX": 100.0, "OAS_PCT": 1.2, "LOAD_TS": f"{d.date()} 16:10:00"} for d in days for c in ("V1", "V2")]).to_sql("PRICES", con, index=False)
    return con


def test_sql_loader_canonicalises_and_stamps_knowledge_time(vendor_db, tmp_path):
    ref = DatasetMapping("reference", "SELECT * FROM REF", {"symbol": "CUSIP", "issuer": "ISSUER", "issue_date": "ISSUE_DATE", "maturity": "MATURITY", "coupon_rate": "COUPON"}, scale={"coupon_rate": 0.01}, date_column="issue_date")
    px = DatasetMapping("evaluated_prices", "SELECT * FROM PRICES WHERE D BETWEEN :start AND :end AND CUSIP != :skip", {"date": "D", "instrument_id": "CUSIP", "mid": "PX"}, knowledge_time_column="LOAD_TS", params={"skip": "NONE"})
    an = DatasetMapping("analytics", "PRICES", {"date": "D", "instrument_id": "CUSIP", "oas_bp": "OAS_PCT"}, scale={"oas_bp": 100.0})
    loader = SQLLoader(lambda: vendor_db)
    frames = load_into_stores(loader, [ref, px, an], "2024-01-01", "2024-01-31", PointInTimeStore(tmp_path / "pit"), FieldStore(tmp_path / "fields"), knowledge_time="2024-02-01")
    assert frames["reference"]["coupon_rate"].tolist() == [0.05, 0.04] and frames["analytics"]["oas_bp"].iloc[0] == pytest.approx(120.0)
    assert pd.to_datetime(frames["evaluated_prices"]["date"]).max() <= pd.Timestamp("2024-01-31") and "publish_ts" in frames["evaluated_prices"]
    assert len(loader.pulls) == 3 and PointInTimeStore(tmp_path / "pit").read("evaluated_prices").shape[0] == len(frames["evaluated_prices"])
    assert FieldStore(tmp_path / "fields").fields() == ["oas_bp"]
    env = builder_from_frames(frames, "2024-01-02", "2024-01-31").build()
    assert len(env.marks) > 0 and env.analytics_source == "VENDOR_ELSE_COMPUTED"
    with pytest.raises(ValueError):
        canonicalise(pd.DataFrame({"X": [1]}), px)


def test_paramstyles_and_casts():
    loader = SQLLoader(lambda: None, paramstyle="pyformat")
    sql, params = loader._bind("SELECT x::date FROM t WHERE d >= :start AND d <= :end", {"start": "a", "end": "b"})
    assert sql == "SELECT x::date FROM t WHERE d >= %(start)s AND d <= %(end)s" and params == {"start": "a", "end": "b"}
    q, p = SQLLoader(lambda: None)._bind("SELECT * FROM t WHERE d >= :start AND c = :code AND d <= :end", {"start": 1, "end": 2, "code": "X"})
    assert q.count("?") == 3 and p == [1, "X", 2]


def test_fiatlas_catalogue_and_bloomberg_members(tmp_path):
    cat = fiatlas_catalog()
    assert {"reference", "evaluated_prices", "analytics", "fields", "benchmark_constituents", "ratings", "curves"} <= set(cat)
    assert "FIATLAS.CREDIT." in cat["analytics"].query and cat["benchmark_constituents"].params["index_code"] == "LUACTRUU"
    f = tmp_path / "members.csv"
    pd.DataFrame({"Date": ["2024-01-31"] * 3, "ID_CUSIP": ["A", "B", "C"], "Weight": [50.0, 30.0, 25.0]}).to_csv(f, index=False)
    m = load_bloomberg_index_members(f, BLOOMBERG_INDEX_MEMBERS)
    assert m["weight"].sum() == pytest.approx(1.0) and list(m.columns) == ["date", "instrument_id", "weight"]
