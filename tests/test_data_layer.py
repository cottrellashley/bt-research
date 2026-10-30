"""Point-in-time store, TRACE cleaning and custom data types."""

import pandas as pd
import pytest

from nautilus_fi.data import (
    BondMark,
    PointInTimeStore,
    TraceCleaningConfig,
    clean_trace,
    daily_liquidity_stats,
    reference_from_dataframe,
    reference_to_dataframe,
    register_types,
    wrap,
)
from nautilus_fi.instruments import BondSpec, CallOption


def test_pit_store_as_of_semantics(tmp_path):
    st = PointInTimeStore(tmp_path)
    first = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 2), "instrument_id": ["A", "B"], "mid": [99.0, 101.0]})
    revised = first.assign(mid=[99.5, 101.0])
    st.write("evaluated_prices", first, knowledge_time=pd.Timestamp("2024-01-02 21:00", tz="UTC"))
    st.write("evaluated_prices", revised, knowledge_time=pd.Timestamp("2024-01-05 21:00", tz="UTC"))
    with pytest.raises(FileExistsError):
        st.write("evaluated_prices", revised, knowledge_time=pd.Timestamp("2024-01-05 21:00", tz="UTC"))
    assert st.read("evaluated_prices", pd.Timestamp("2024-01-03", tz="UTC"))["mid"].tolist() == [99.0, 101.0]
    assert st.read("evaluated_prices")["mid"].tolist() == [99.5, 101.0]
    assert st.read_first_known("evaluated_prices", ["date", "instrument_id"])["mid"].tolist() == [99.0, 101.0]
    with pytest.raises(FileNotFoundError):
        st.read("evaluated_prices", pd.Timestamp("2023-12-31", tz="UTC"))


def test_trace_cleaning_filters_and_knowledge_time():
    raw = pd.DataFrame({
        "cusip": ["A"] * 6 + ["B"] * 2,
        "exec": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 10:05", "2024-01-02 10:05", "2024-01-02 11:00", "2024-01-02 12:00", "2024-01-02 13:00", "2024-01-02 10:00", "2024-01-02 10:00"]),
        "px": [99.0, 99.1, 99.1, 120.0, 99.2, 99.3, 101.0, 101.0],
        "qty": [500_000, 200_000, 200_000, 100_000, 7_000_000, 50_000, 1_000_000, 1_000_000],
        "side": ["B", "D", "D", "S", "S", "B", "D", "D"],
        "st": ["T", "T", "T", "T", "T", "C", "T", "T"],
        "msg": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "orig": [None, None, None, None, None, "1", None, None],
    })
    out = clean_trace(raw, columns={"instrument_id": "cusip", "exec_ts": "exec", "price": "px", "size": "qty", "side": "side", "status": "st", "msg_id": "msg", "orig_msg_id": "orig"}, config=TraceCleaningConfig(min_size=100_000))
    assert out["price"].tolist() == [99.1, 99.2, 101.0]          # cancel+original, duplicate inter-dealer, outlier, odd lot removed
    assert (out["available_ts"] - out["exec_ts"]).unique().tolist() == [pd.Timedelta(minutes=15)]
    assert out.loc[out["size"] == 7_000_000, "public_size"].iloc[0] == 5_000_000  # dissemination cap
    stats = daily_liquidity_stats(out, pd.bdate_range("2024-01-02", "2024-01-04"))
    a = stats[stats["instrument_id"] == "A"].sort_values("date")
    assert a["volume_5d"].tolist() == [5_200_000.0] * 3 and a["days_since_trade"].tolist() == [0, 1, 2]


def test_custom_types_and_reference_roundtrip():
    register_types()
    from nautilus_trader.model import InstrumentId

    m = BondMark(ts_event=1, ts_init=2, instrument_id=InstrumentId.from_str("X.OTC"), clean_mid=99.0, clean_bid=98.9, clean_ask=99.1, accrued=0.5, tier="EVALUATED", quality=0.7)
    assert m.dirty_mid == pytest.approx(99.5) and wrap(m).ts_init == 2
    assert BondMark.from_dict(m.to_dict()).tier == "EVALUATED"
    b = BondSpec(symbol="CORP1", issuer="ACME", issue_date=pd.Timestamp("2020-03-15").date(), maturity=pd.Timestamp("2030-03-15").date(), coupon_rate=0.05, call_schedule=[CallOption(pd.Timestamp("2027-03-15").date(), 102.0)])
    back = reference_from_dataframe(reference_to_dataframe([b]))["CORP1"]
    assert back.call_schedule == b.call_schedule and back.maturity == b.maturity and back.coupon_rate == 0.05
