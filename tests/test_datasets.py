"""Synthetic generators and the Covertype loader (tested on a small hand-made archive)."""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlguard.datasets import (
    COVERTYPE_NUMERIC,
    load_beijing,
    load_covertype,
    load_news,
    make_synthetic,
    make_synthetic_series,
)


def _covertype_rows(n_rows: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    table = np.zeros((n_rows, 55), dtype=np.int64)
    table[:, :10] = rng.integers(0, 4000, size=(n_rows, 10))
    table[np.arange(n_rows), 10 + rng.integers(0, 4, n_rows)] = 1
    table[np.arange(n_rows), 14 + rng.integers(0, 40, n_rows)] = 1
    table[:, 54] = rng.integers(1, 8, n_rows)
    return table


def _write(path: Path, table: np.ndarray) -> Path:
    text = "\n".join(",".join(map(str, row)) for row in table) + "\n"
    payload = gzip.compress(text.encode())
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("covtype.data.gz", payload)
            archive.writestr("covtype.info", "description")
    else:
        path.write_bytes(payload)
    return path


@pytest.mark.parametrize("maker", [make_synthetic, make_synthetic_series])
def test_synthetic_data_is_deterministic_and_well_formed(maker) -> None:  # type: ignore[no-untyped-def]
    first, again, other = maker(2_000, seed=3), maker(2_000, seed=3), maker(2_000, seed=4)
    assert_frame_equal(first.frame, again.frame)
    assert not first.frame.equals(other.frame)
    assert len(first.frame) == 2_000
    assert set(first.features) | {first.target} == set(first.frame.columns)
    assert first.target not in first.features
    assert set(first.frame[first.target].unique()) == {0, 1}
    assert [first.kind(name) for name in first.numeric] == ["numeric"] * len(first.numeric)
    assert [first.kind(name) for name in first.categorical] == ["categorical"] * len(
        first.categorical
    )


def test_synthetic_table_has_the_hard_cases_it_promises() -> None:
    frame = make_synthetic(50_000, seed=0).frame
    assert frame["x_normal"].isna().mean() == pytest.approx(0.02, abs=0.005)
    assert (frame["x_zero_inflated"] == 0).mean() == pytest.approx(0.3, abs=0.01)
    assert (frame["x_sentinel"] == -1).mean() == pytest.approx(0.05, abs=0.005)
    assert frame["x_count"].dtype == np.int64
    assert frame["c_large"].nunique() == 30
    assert 0.3 < frame["target"].mean() < 0.7


def test_series_features_use_only_the_past() -> None:
    dataset = make_synthetic_series(5_000, seed=0)
    frame = dataset.frame
    assert dataset.time_ordered
    lag = frame["lag_1"].to_numpy()
    # lag_mean_3 at time t is the mean of the three values before t: lag_1 at t, t-1 and t-2.
    np.testing.assert_allclose(
        frame["lag_mean_3"].to_numpy()[2:], (lag[2:] + lag[1:-1] + lag[:-2]) / 3
    )
    # lag_1 at t + 1 is the current value of the signal, whose sign is the target at t.
    np.testing.assert_array_equal(frame["target"].to_numpy()[:-1], (lag[1:] > 0).astype(int))
    # The AR(1) signal makes the past informative but not decisive.
    agreement = ((lag > 0).astype(int) == frame["target"]).mean()
    assert 0.6 < agreement < 0.9
    # The regime keeps its state most of the time and uses all four states.
    regime = frame["regime"].to_numpy()
    assert (regime[1:] == regime[:-1]).mean() == pytest.approx(0.925, abs=0.02)
    assert sorted(set(regime)) == ["r0", "r1", "r2", "r3"]


@pytest.mark.parametrize("name", ["covertype.zip", "covtype.data.gz"])
def test_covertype_loader_decodes_the_one_hot_groups(tmp_path: Path, name: str) -> None:
    table = _covertype_rows(50)
    dataset = load_covertype(_write(tmp_path / name, table), expected_rows=50)
    frame = dataset.frame
    assert dataset.numeric == COVERTYPE_NUMERIC
    assert dataset.categorical == ("wilderness_area", "soil_type")
    assert dataset.target == "cover_type"
    np.testing.assert_array_equal(frame[list(COVERTYPE_NUMERIC)].to_numpy(), table[:, :10])
    np.testing.assert_array_equal(frame["cover_type"].to_numpy(), table[:, 54])
    expected_area = [f"area_{index + 1}" for index in table[:, 10:14].argmax(axis=1)]
    expected_soil = [f"soil_{index + 1:02d}" for index in table[:, 14:54].argmax(axis=1)]
    assert frame["wilderness_area"].tolist() == expected_area
    assert frame["soil_type"].tolist() == expected_soil
    assert "CC BY 4.0" in dataset.source


def test_covertype_loader_rejects_damaged_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_covertype(tmp_path / "missing.zip")
    table = _covertype_rows(20)
    with pytest.raises(ValueError, match="expected 581012 rows"):
        load_covertype(_write(tmp_path / "short.zip", table))
    with pytest.raises(ValueError, match="55 columns"):
        load_covertype(_write(tmp_path / "narrow.zip", table[:, :54]), expected_rows=20)
    two_areas = table.copy()
    two_areas[0, 10:14] = 1
    with pytest.raises(ValueError, match="wilderness area"):
        load_covertype(_write(tmp_path / "areas.zip", two_areas), expected_rows=20)
    no_soil = table.copy()
    no_soil[0, 14:54] = 0
    with pytest.raises(ValueError, match="soil type"):
        load_covertype(_write(tmp_path / "soil.zip", no_soil), expected_rows=20)
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("readme.txt", "nothing")
    with pytest.raises(ValueError, match="exactly one"):
        load_covertype(tmp_path / "empty.zip")


# --------------------------------------------------------------------------- Online News Popularity

NEWS_CHANNELS = ("lifestyle", "entertainment", "bus", "socmed", "tech", "world")
NEWS_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def news_table(n_rows: int = 30, seed: int = 0) -> pd.DataFrame:
    """A table with the 61 columns of the real file, including their leading spaces."""
    rng = np.random.default_rng(seed)
    columns: dict[str, object] = {"url": [f"http://example.org/{index}" for index in range(n_rows)]}
    columns[" timedelta"] = np.arange(n_rows, 0, -1).astype(float)
    for index in range(44):
        columns[f" Feature_{index:02d}"] = rng.normal(size=n_rows)
    channel = rng.integers(0, 7, n_rows)  # 6 means "no channel"
    for position, name in enumerate(NEWS_CHANNELS):
        columns[f" data_channel_is_{name}"] = (channel == position).astype(float)
    day = rng.integers(0, 7, n_rows)
    for position, name in enumerate(NEWS_DAYS):
        columns[f" weekday_is_{name}"] = (day == position).astype(float)
    columns[" is_weekend"] = (day >= 5).astype(float)
    columns[" shares"] = rng.integers(1, 3_000, n_rows)
    table = pd.DataFrame(columns)
    table.attrs["channel"], table.attrs["day"] = channel, day
    return table


def write_news(path: Path, table: pd.DataFrame) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("OnlineNewsPopularity/OnlineNewsPopularity.names", "description")
        archive.writestr("OnlineNewsPopularity/OnlineNewsPopularity.csv", table.to_csv(index=False))
    return path


def test_news_loader_decodes_indicators_and_builds_the_binary_target(tmp_path: Path) -> None:
    table = news_table()
    dataset = load_news(write_news(tmp_path / "news.zip", table), expected_rows=30)
    frame = dataset.frame
    assert len(dataset.numeric) == 44
    assert dataset.numeric[0] == "feature_00"  # stripped and lower-cased
    assert dataset.categorical == ("channel", "weekday")
    assert set(frame.columns) == {*dataset.numeric, "channel", "weekday", "popular"}
    names = np.array([*NEWS_CHANNELS, "other"], dtype=object)
    assert frame["channel"].tolist() == names[table.attrs["channel"]].tolist()
    assert (
        frame["weekday"].tolist() == np.array(NEWS_DAYS, dtype=object)[table.attrs["day"]].tolist()
    )
    np.testing.assert_array_equal(frame["popular"], (table[" shares"] >= 1_400).astype(int))
    assert not dataset.time_ordered
    for leaked in ("url", "timedelta", "shares", "is_weekend"):
        assert leaked not in frame.columns  # neither identifiers nor the raw target are features


def test_news_loader_rejects_damaged_files(tmp_path: Path) -> None:
    table = news_table()
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_news(tmp_path / "missing.zip")
    with pytest.raises(ValueError, match="expected 39644 rows"):
        load_news(write_news(tmp_path / "short.zip", table))
    with pytest.raises(ValueError, match="unexpected columns"):
        load_news(
            write_news(tmp_path / "narrow.zip", table.drop(columns=[" Feature_00"])),
            expected_rows=30,
        )
    two_days = table.copy()
    two_days.loc[0, [" weekday_is_monday", " weekday_is_tuesday"]] = 1.0
    with pytest.raises(ValueError, match="weekday_is_"):
        load_news(write_news(tmp_path / "days.zip", two_days), expected_rows=30)
    two_channels = table.copy()
    two_channels.loc[0, [" data_channel_is_bus", " data_channel_is_tech"]] = 1.0
    with pytest.raises(ValueError, match="data_channel_is_"):
        load_news(write_news(tmp_path / "channels.zip", two_channels), expected_rows=30)
    holes = table.copy()
    holes.loc[0, " Feature_03"] = np.nan
    with pytest.raises(ValueError, match="without missing values"):
        load_news(write_news(tmp_path / "holes.zip", holes), expected_rows=30)
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("readme.txt", "nothing")
    with pytest.raises(ValueError, match=r"exactly one \.csv"):
        load_news(tmp_path / "empty.zip")


# --------------------------------------------------------------------------- Beijing air quality


def station_table(station: str, n_hours: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = pd.date_range("2013-03-01", periods=n_hours, freq="h")
    table = pd.DataFrame(
        {
            "No": np.arange(1, n_hours + 1),
            "year": time.year, "month": time.month, "day": time.day, "hour": time.hour,
            "PM2.5": rng.uniform(10, 150, n_hours).round(),
        }
    )  # fmt: skip
    for name in ("PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN"):
        table[name] = rng.uniform(0, 100, n_hours).round(1)
    table["wd"] = rng.choice(np.array(["N", "E", "S", "W"], dtype=object), n_hours)
    table["WSPM"] = rng.uniform(0, 5, n_hours).round(1)
    table["station"] = station
    return table


def write_beijing(path: Path, tables: dict[str, pd.DataFrame], *, nested: bool = True) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer if nested else path, "w") as archive:
        for station, table in tables.items():
            name = f"PRSA_Data_20130301-20170228/PRSA_Data_{station}_20130301-20170228.csv"
            archive.writestr(name, table.to_csv(index=False))
    if nested:
        with zipfile.ZipFile(path, "w") as outer:
            outer.writestr("PRSA2017_Data_20130301-20170228.zip", buffer.getvalue())
            outer.writestr("data.csv", "unrelated,file\n1,2\n")
    return path


@pytest.mark.parametrize("nested", [True, False])
def test_beijing_loader_orders_by_time_and_builds_past_only_features(
    tmp_path: Path, nested: bool
) -> None:
    north, south = station_table("North", seed=1), station_table("South", seed=2)
    north.loc[5, "PM2.5"] = np.nan  # an hour without a reading
    north.loc[7, "wd"] = None
    path = write_beijing(tmp_path / "beijing.zip", {"North": north, "South": south}, nested=nested)
    dataset = load_beijing(path, expected_stations=2)
    frame = dataset.frame
    assert dataset.time_ordered
    assert dataset.group_by == "station"
    assert dataset.target == "polluted"
    assert len(frame) == 119  # 2 x 60 hours minus the hour without a reading
    assert frame["station"].tolist()[:4] == ["North", "South", "North", "South"]  # hour by hour
    assert "pm25" not in frame.columns  # the current reading is the target, never a feature

    for name, table in (("North", north), ("South", south)):
        own = frame[frame["station"] == name].reset_index(drop=True)
        readings = table["PM2.5"]
        kept = readings.notna().to_numpy()
        np.testing.assert_array_equal(own["polluted"], (readings[kept] > 75).astype(int))
        # the reading one hour earlier, and the mean of the 24 hours before: nothing from now
        np.testing.assert_allclose(own["pm25_lag_1h"], readings.shift(1)[kept], equal_nan=True)
        expected_mean = readings.shift(1).rolling(24, min_periods=12).mean()[kept]
        np.testing.assert_allclose(own["pm25_lag_mean_24h"], expected_mean, equal_nan=True)
        np.testing.assert_allclose(own["temperature"], table["TEMP"][kept])
    assert frame["wind_direction"].isna().sum() == 1
    assert frame["wind_direction"].dropna().map(type).eq(str).all()


def test_beijing_loader_rejects_damaged_files(tmp_path: Path) -> None:
    good = station_table("North")
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_beijing(tmp_path / "missing.zip")
    with pytest.raises(ValueError, match="expected 12 station files"):
        load_beijing(write_beijing(tmp_path / "one.zip", {"North": good}))
    with pytest.raises(ValueError, match="gap-free hourly"):
        load_beijing(
            write_beijing(tmp_path / "gap.zip", {"North": good.drop(index=10)}), expected_stations=1
        )
    with pytest.raises(ValueError, match="lacks expected columns"):
        load_beijing(
            write_beijing(tmp_path / "cols.zip", {"North": good.drop(columns=["TEMP"])}),
            expected_stations=1,
        )
    mixed = good.copy()
    mixed.loc[0, "station"] = "Elsewhere"
    with pytest.raises(ValueError, match="exactly one station"):
        load_beijing(write_beijing(tmp_path / "mixed.zip", {"North": mixed}), expected_stations=1)
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("readme.txt", "nothing")
    with pytest.raises(ValueError, match="no station file"):
        load_beijing(tmp_path / "empty.zip", expected_stations=None)
