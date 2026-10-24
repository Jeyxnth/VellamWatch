"""as_of strictness, get_model_info, and load-time validation of the contract v1.1 data files."""
import datetime as dt
import json

import pandas as pd
import pytest

from ml import features, rain, train
from ml.features import DataFormatError, MissingDataError
from ml.predict import get_model_info, predict_risk_at
from ml.tests.conftest import SMALL_BBOX, clone_data


# ---- as_of --------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["2023-12-03T00:00:00", "2023-12-03", "not a time", ""])
def test_as_of_without_offset_or_garbage_is_rejected(env, bad):
    with pytest.raises(ValueError) as e:
        predict_risk_at(*SMALL_BBOX, 24, as_of=bad)
    msg = str(e.value)
    assert "UTC offset" in msg or "valid timestamp" in msg
    assert "+05:30" in msg  # the message shows the expected format


def test_as_of_accepts_offsets_utc_and_aware_datetimes(env):
    a = predict_risk_at(*SMALL_BBOX, 24, as_of="2023-12-03T00:00:00+05:30")
    b = predict_risk_at(*SMALL_BBOX, 24, as_of="2023-12-02T18:30:00Z")           # same instant in UTC
    c = predict_risk_at(*SMALL_BBOX, 24, as_of=dt.datetime(2023, 12, 3, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))))
    assert a == b == c


def test_naive_env_as_of_is_rejected(env, monkeypatch):
    monkeypatch.setenv("ML_AS_OF", "2023-12-03T00:00:00")
    with pytest.raises(ValueError, match="ML_AS_OF"):
        predict_risk_at(*SMALL_BBOX, 24)


def test_predict_risk_signature_unchanged():
    import inspect
    from ml.predict import predict_risk
    assert list(inspect.signature(predict_risk).parameters) == ["min_lat", "min_lon", "max_lat", "max_lon", "horizon_h"]


def test_get_model_info(env):
    info = get_model_info()
    assert info["data_source"] == "synthetic"
    assert pd.Timestamp(info["trained_at"]).tzinfo is not None
    assert isinstance(info["data_dir"], str) and info["data_dir"]
    assert info["rain_stage"] == "policy targets, not fitted"


# ---- cells.geojson ----------------------------------------------------------------------------
def _edit_cells(env, name, fn):
    d = clone_data(env, name)
    path = d / "cells.geojson"
    gj = json.loads(path.read_text())
    fn(gj)
    path.write_text(json.dumps(gj))
    return d


def test_cells_missing_row_col_properties(env):
    def drop(gj):
        del gj["features"][5]["properties"]["row"]
    d = _edit_cells(env, "c_noprop", drop)
    with pytest.raises(DataFormatError, match="cells.geojson") as e:
        features.load_cells(d)
    assert "row" in str(e.value) and "col" in str(e.value)


def test_cells_non_contiguous_rows(env):
    def gap(gj):
        for f in gj["features"]:
            r = f["properties"]["row"]
            if r >= 10:
                f["properties"]["row"] = r + 1
                f["properties"]["cell_id"] = f"r{r + 1}_c{f['properties']['col']}"
    d = _edit_cells(env, "c_gap", gap)
    with pytest.raises(DataFormatError, match="cells.geojson.*contiguous"):
        features.load_cells(d)


def test_cells_row_col_disagree_with_cell_id(env):
    def swap(gj):
        gj["features"][3]["properties"]["col"] += 1
    d = _edit_cells(env, "c_mismatch", swap)
    with pytest.raises(DataFormatError, match="cells.geojson.*disagree"):
        features.load_cells(d)


def test_cells_row0_must_be_southernmost(env):
    def flip(gj):
        rows = max(f["properties"]["row"] for f in gj["features"])
        for f in gj["features"]:
            p = f["properties"]
            p["row"] = rows - p["row"]
            p["cell_id"] = f"r{p['row']}_c{p['col']}"
    d = _edit_cells(env, "c_flip", flip)
    with pytest.raises(DataFormatError, match="SOUTHERNMOST"):
        features.load_cells(d)


def test_valid_cells_load(env):
    cells = features.load_cells(env["data"])
    assert cells.row.min() == 0 and cells.col.min() == 0 and len(cells) == cells.cell_id.nunique()


# ---- rainfall.parquet -----------------------------------------------------------------------------
def _rewrite_rain(env, name, fn):
    d = clone_data(env, name)
    df = pd.read_parquet(d / "rainfall.parquet")
    fn(df).to_parquet(d / "rainfall.parquet", index=False)
    return d


def test_rain_naive_timestamps_rejected(env):
    d = _rewrite_rain(env, "r_naive", lambda df: df.assign(timestamp=df.timestamp.dt.tz_localize(None)))
    with pytest.raises(DataFormatError, match="rainfall.parquet.*no UTC offset"):
        rain.load_rainfall(d)


def test_rain_timestamps_must_be_on_the_hour(env):
    d = _rewrite_rain(env, "r_minutes", lambda df: df.assign(timestamp=df.timestamp + pd.Timedelta(minutes=30)))
    with pytest.raises(DataFormatError, match="rainfall.parquet.*not on the hour"):
        rain.load_rainfall(d)


def test_rain_same_instants_in_other_offset_are_accepted(env):
    # IST hour starts written as UTC (:30 past the UTC hour) are the same instants, so they are fine
    d = _rewrite_rain(env, "r_utc", lambda df: df.assign(timestamp=df.timestamp.dt.tz_convert("UTC")))
    assert str(rain.load_rainfall(d).timestamp.dt.tz) == "Asia/Kolkata"


def test_rain_true_utc_hour_starts_rejected(env):
    # :00 UTC is :30 IST, i.e. not an IST hour start
    d = _rewrite_rain(env, "r_utc_shift", lambda df: df.assign(
        timestamp=(df.timestamp.dt.tz_convert("UTC") + pd.Timedelta(minutes=30))))
    with pytest.raises(DataFormatError, match=r"not on the hour in \+05:30"):
        rain.load_rainfall(d)


def test_rain_missing_columns(env):
    d = _rewrite_rain(env, "r_cols", lambda df: df.drop(columns=["precip_mm"]))
    with pytest.raises(ValueError, match="precip_mm"):
        rain.load_rainfall(d)


# ---- events.csv and event coverage --------------------------------------------------------------------
def test_events_csv_without_offset_rejected(env):
    d = clone_data(env, "e_naive")
    (d / "events.csv").write_text("event,start,end\n2015,2015-12-01,2015-12-04\n2023,2023-12-02,2023-12-06\n")
    with pytest.raises(DataFormatError, match="events.csv.*no UTC offset"):
        features.event_windows(d)


def test_events_csv_missing_columns_and_bad_order(env):
    d = clone_data(env, "e_cols")
    (d / "events.csv").write_text("event,start\n2015,2015-12-01T00:00:00+05:30\n")
    with pytest.raises(MissingDataError, match="events.csv"):
        features.event_windows(d)
    (d / "events.csv").write_text("event,start,end\n2015,2015-12-04T00:00:00+05:30,2015-12-01T00:00:00+05:30\n")
    with pytest.raises(DataFormatError, match="ends"):
        features.event_windows(d)


def test_events_csv_read_with_exclusive_end(env):
    windows, prov = features.event_windows(env["data"])
    start, end = windows["2023"]
    assert start == pd.Timestamp("2023-12-02T00:00:00+05:30") and end == pd.Timestamp("2023-12-06T00:00:00+05:30")
    assert "SYNTHETIC" in prov


def test_event_coverage_gap_warns_but_does_not_crash(env, tmp_path):
    def drop_2015(df):
        return df[df.timestamp >= pd.Timestamp("2023-01-01T00:00:00+05:30")]
    d = _rewrite_rain(env, "r_gap", drop_2015)
    with pytest.warns(UserWarning, match="event '2015'"):
        meta = train.train(d, tmp_path / "m")
    assert meta["rain"]["fitted"] is False and meta["n_cells"] > 0


def test_unknown_hotspot_event_warns(env, tmp_path):
    d = clone_data(env, "e_unknown")
    hot = pd.read_csv(d / "hotspots.csv", dtype=str)
    hot.loc[:5, "event"] = "1999"
    hot.to_csv(d / "hotspots.csv", index=False)
    with pytest.warns(UserWarning, match="1999"):
        train.train(d, tmp_path / "m2")


def test_missing_events_csv_warns_and_uses_placeholder(env, tmp_path):
    d = clone_data(env, "e_missing")
    (d / "events.csv").unlink()
    with pytest.warns(UserWarning, match="events.csv not found"):
        meta = train.train(d, tmp_path / "m3")
    assert meta["rain"]["fitted"] is False
