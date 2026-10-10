"""Datasets of the benchmark: two synthetic generators and the loader of the Covertype cache.

Nothing here touches the network. Public data is downloaded once by ``tools/download_data.py``
and read from the local cache.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """A table with the roles of its columns."""

    name: str
    frame: pd.DataFrame
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    time_ordered: bool = False
    source: str = "synthetic"
    #: For time-ordered tables with several interleaved series: the column that names the series.
    group_by: str | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def kind(self, feature: str) -> str:
        return "numeric" if feature in self.numeric else "categorical"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def make_synthetic(n_rows: int, seed: int) -> Dataset:
    """I.i.d. table with the shapes that make drift detection hard in practice.

    A bell-shaped feature with natural missing values, a heavy-tailed one, a zero-inflated one, an
    integer count, a feature with a ``-1`` sentinel, a small and a long-tailed categorical feature,
    and a binary target that depends on most of them.
    """
    rng = np.random.default_rng([seed, 101])
    x_normal = rng.normal(50.0, 10.0, n_rows)
    x_lognormal = rng.lognormal(0.0, 1.0, n_rows)
    x_zero_inflated = np.where(rng.random(n_rows) < 0.3, 0.0, rng.gamma(2.0, 3.0, n_rows))
    x_count = rng.poisson(4.0, n_rows).astype(np.int64)
    x_sentinel = np.where(rng.random(n_rows) < 0.05, -1.0, rng.uniform(0.0, 100.0, n_rows))
    small = np.array(["a", "b", "c", "d", "e"], dtype=object)
    c_small = rng.choice(small, n_rows, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    weights = 1.0 / np.arange(1, 31)
    large = np.array([f"k{index:02d}" for index in range(30)], dtype=object)
    c_large = rng.choice(large, n_rows, p=weights / weights.sum())
    effect = {"a": 0.0, "b": 0.5, "c": -0.5, "d": 1.0, "e": -1.0}
    logit = (
        0.08 * (x_normal - 50.0)
        + 0.6 * np.log(x_lognormal)
        + 0.15 * (x_count - 4.0)
        + 0.1 * (x_zero_inflated - 4.2)
        + 0.02 * (np.where(x_sentinel < 0, 50.0, x_sentinel) - 50.0)
        + np.vectorize(effect.get)(c_small)
    )
    target = (rng.random(n_rows) < _sigmoid(logit)).astype(np.int64)
    x_normal[rng.random(n_rows) < 0.02] = np.nan
    frame = pd.DataFrame(
        {
            "x_normal": x_normal,
            "x_lognormal": x_lognormal,
            "x_zero_inflated": x_zero_inflated,
            "x_count": x_count,
            "x_sentinel": x_sentinel,
            "c_small": c_small,
            "c_large": c_large,
            "target": target,
        }
    )
    return Dataset(
        "synthetic",
        frame,
        numeric=("x_normal", "x_lognormal", "x_zero_inflated", "x_count", "x_sentinel"),
        categorical=("c_small", "c_large"),
        target="target",
    )


def make_synthetic_series(n_rows: int, seed: int) -> Dataset:
    """Stationary time-ordered table for the ``time_travel`` fault.

    ``y`` is an AR(1) signal. The features ``lag_1`` and ``lag_mean_3`` use past values of ``y``
    only; the target says whether the current ``y`` is positive. Shifting a lag feature forward in
    time therefore hands the model the answer. ``regime`` is a categorical Markov chain that tends
    to keep its state and ``unrelated`` is white noise; neither carries information about the
    target.
    """
    rng = np.random.default_rng([seed, 102])
    noise = rng.normal(0.0, 1.0, n_rows + 3)
    y = np.empty(n_rows + 3)
    y[0] = noise[0]
    for step in range(1, n_rows + 3):
        y[step] = 0.8 * y[step - 1] + noise[step]
    current = y[3:]
    # A state is kept with probability 0.9, otherwise redrawn uniformly from the four states.
    redraw = rng.random(n_rows) >= 0.9
    redraw[0] = True
    drawn = rng.integers(0, 4, n_rows)
    last_redraw = np.maximum.accumulate(np.where(redraw, np.arange(n_rows), 0))
    regime = np.array(["r0", "r1", "r2", "r3"], dtype=object)[drawn[last_redraw]]
    frame = pd.DataFrame(
        {
            "lag_1": y[2:-1],
            "lag_mean_3": (y[2:-1] + y[1:-2] + y[:-3]) / 3.0,
            "unrelated": rng.normal(0.0, 1.0, n_rows),
            "regime": regime,
            "target": (current > 0).astype(np.int64),
        }
    )
    return Dataset(
        "synthetic_series",
        frame,
        numeric=("lag_1", "lag_mean_3", "unrelated"),
        categorical=("regime",),
        target="target",
        time_ordered=True,
    )


COVERTYPE_NUMERIC = (
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
)
COVERTYPE_ROWS = 581_012
_N_WILDERNESS, _N_SOIL = 4, 40


def load_covertype(path: str | Path, *, expected_rows: int | None = COVERTYPE_ROWS) -> Dataset:
    """Read the UCI Covertype archive (``covertype.zip`` or ``covtype.data.gz``) from disk.

    The two one-hot encoded groups (4 wilderness areas, 40 soil types) are decoded into two
    categorical columns. The file is validated: 55 integer columns, exactly one active indicator
    per group in every row, and the documented number of rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python tools/download_data.py covertype` first"
        )
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith("covtype.data.gz")]
            if len(members) != 1:
                raise ValueError(f"{path} does not contain exactly one covtype.data.gz")
            raw = gzip.decompress(archive.read(members[0]))
    else:
        raw = gzip.decompress(path.read_bytes())
    table = pd.read_csv(io.BytesIO(raw), header=None, dtype=np.int64).to_numpy()
    if table.shape[1] != 55:
        raise ValueError(f"expected 55 columns, found {table.shape[1]}")
    if expected_rows is not None and table.shape[0] != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {table.shape[0]}")
    wilderness = table[:, 10 : 10 + _N_WILDERNESS]
    soil = table[:, 14 : 14 + _N_SOIL]
    for name, block in (("wilderness area", wilderness), ("soil type", soil)):
        if not (np.isin(block, (0, 1)).all() and (block.sum(axis=1) == 1).all()):
            raise ValueError(f"{name} columns are not a valid one-hot encoding")
    frame = pd.DataFrame(table[:, :10], columns=list(COVERTYPE_NUMERIC))
    areas = np.array([f"area_{index + 1}" for index in range(_N_WILDERNESS)], dtype=object)
    soils = np.array([f"soil_{index + 1:02d}" for index in range(_N_SOIL)], dtype=object)
    frame["wilderness_area"] = areas[wilderness.argmax(axis=1)]
    frame["soil_type"] = soils[soil.argmax(axis=1)]
    frame["cover_type"] = table[:, 54]
    return Dataset(
        "covertype",
        frame,
        numeric=COVERTYPE_NUMERIC,
        categorical=("wilderness_area", "soil_type"),
        target="cover_type",
        source="UCI Covertype, CC BY 4.0, https://doi.org/10.24432/C50K5N",
    )


NEWS_ROWS = 39_644
NEWS_POPULAR_FROM = 1_400
_NEWS_CHANNELS = ("lifestyle", "entertainment", "bus", "socmed", "tech", "world")
_NEWS_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _single_member(archive: zipfile.ZipFile, suffix: str) -> bytes:
    members = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(members) != 1:
        raise ValueError(f"expected exactly one {suffix} file in the archive, found {len(members)}")
    return archive.read(members[0])


def _decode_one_hot(
    table: pd.DataFrame, prefix: str, levels: tuple[str, ...], other: str | None
) -> np.ndarray:
    """Turn indicator columns ``prefix + level`` into one categorical column.

    Every row must have exactly one active indicator, or none when ``other`` names the level for
    such rows.
    """
    block = table[[prefix + level for level in levels]].to_numpy()
    active = block.sum(axis=1)
    allowed = (0, 1) if other is not None else (1,)
    if not (np.isin(block, (0, 1)).all() and np.isin(active, allowed).all()):
        raise ValueError(f"columns {prefix}* are not a valid one-hot encoding")
    names = np.array([*levels, other], dtype=object)
    return names[np.where(active == 1, block.argmax(axis=1), len(levels))]


def load_news(path: str | Path, *, expected_rows: int | None = NEWS_ROWS) -> Dataset:
    """Read the UCI Online News Popularity archive from disk.

    The task is the binary one of the dataset's authors: an article is *popular* when it has at
    least 1,400 shares (the median). The URL and the age of the article (``timedelta``) are not
    features. The six channel indicators and the seven weekday indicators are decoded into two
    categorical columns (articles without a channel get ``other``); ``is_weekend`` is dropped
    because the weekday already says it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python tools/download_data.py news` first")
    with zipfile.ZipFile(path) as archive:
        table = pd.read_csv(io.BytesIO(_single_member(archive, ".csv")))
    table.columns = [str(column).strip() for column in table.columns]
    required = {"url", "timedelta", "shares", "is_weekend"}
    if len(table.columns) != 61 or not required <= set(table.columns):
        raise ValueError("unexpected columns: this is not the Online News Popularity table")
    if expected_rows is not None and len(table) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(table)}")
    channel = _decode_one_hot(table, "data_channel_is_", _NEWS_CHANNELS, "other")
    weekday = _decode_one_hot(table, "weekday_is_", _NEWS_WEEKDAYS, None)
    indicators = [f"data_channel_is_{name}" for name in _NEWS_CHANNELS]
    indicators += [f"weekday_is_{name}" for name in _NEWS_WEEKDAYS]
    dropped = ["url", "timedelta", "shares", "is_weekend", *indicators]
    numeric = tuple(column.lower() for column in table.columns if column not in dropped)
    frame = table.drop(columns=dropped)
    frame.columns = list(numeric)
    if frame.isna().any().any() or not all(dtype.kind in "if" for dtype in frame.dtypes):
        raise ValueError("the numeric features must be numbers without missing values")
    frame["channel"] = channel
    frame["weekday"] = weekday
    frame["popular"] = (table["shares"].to_numpy() >= NEWS_POPULAR_FROM).astype(np.int64)
    return Dataset(
        "news",
        frame,
        numeric=numeric,
        categorical=("channel", "weekday"),
        target="popular",
        source="UCI Online News Popularity, CC BY 4.0, https://doi.org/10.24432/C5NS3V",
    )


BEIJING_STATIONS = 12
BEIJING_POLLUTED_ABOVE = 75.0
_BEIJING_NUMERIC = {
    "PM10": "pm10", "SO2": "so2", "NO2": "no2", "CO": "co", "O3": "o3", "TEMP": "temperature",
    "PRES": "pressure", "DEWP": "dew_point", "RAIN": "rain", "WSPM": "wind_speed",
}  # fmt: skip


def load_beijing(path: str | Path, *, expected_stations: int | None = BEIJING_STATIONS) -> Dataset:
    """Read the UCI Beijing Multi-Site Air Quality archive (hourly, 12 stations) from disk.

    Rows are ordered by time and then by station, so a contiguous block of rows is a period of
    time over all stations. The target says whether PM2.5 exceeds 75 micrograms per cubic metre;
    hours without a PM2.5 reading are dropped. Two features describe the past of PM2.5 at the same
    station and nothing else: the reading one hour earlier and the mean of the 24 hours before.
    They are what the ``time_travel`` fault shifts into the future. Missing sensor readings are
    kept as missing values.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `python tools/download_data.py beijing` first"
        )
    with zipfile.ZipFile(path) as archive:
        nested = [name for name in archive.namelist() if name.endswith(".zip")]
        if len(nested) == 1:  # the UCI download wraps the data in a second archive
            archive = zipfile.ZipFile(io.BytesIO(archive.read(nested[0])))
        members = sorted(
            name for name in archive.namelist() if "PRSA_Data_" in name and name.endswith(".csv")
        )
        tables = [pd.read_csv(io.BytesIO(archive.read(name))) for name in members]
    if expected_stations is not None and len(tables) != expected_stations:
        raise ValueError(f"expected {expected_stations} station files, found {len(tables)}")
    if not tables:
        raise ValueError("the archive contains no station file")
    parts = []
    for table in tables:
        needed = {"year", "month", "day", "hour", "PM2.5", "wd", "station", *_BEIJING_NUMERIC}
        if not needed <= set(table.columns):
            raise ValueError("a station file lacks expected columns")
        if table["station"].nunique() != 1:
            raise ValueError("a station file must describe exactly one station")
        time = pd.to_datetime(table[["year", "month", "day", "hour"]])
        hours = time.diff().dropna()
        if not (hours == pd.Timedelta(hours=1)).all():
            raise ValueError("a station file is not a gap-free hourly series in time order")
        pm25 = table["PM2.5"]
        part = pd.DataFrame({"time": time, "station": table["station"].astype(object)})
        for source, name in _BEIJING_NUMERIC.items():
            part[name] = table[source].astype(np.float64)
        # Past values only: shift(1) excludes the current hour from both features.
        part["pm25_lag_1h"] = pm25.shift(1)
        part["pm25_lag_mean_24h"] = pm25.shift(1).rolling(24, min_periods=12).mean()
        part["wind_direction"] = table["wd"].astype(object).where(table["wd"].notna(), None)
        part["pm25"] = pm25
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame[frame["pm25"].notna()].sort_values(["time", "station"], kind="stable")
    frame["polluted"] = (frame["pm25"].to_numpy() > BEIJING_POLLUTED_ABOVE).astype(np.int64)
    numeric = (*_BEIJING_NUMERIC.values(), "pm25_lag_1h", "pm25_lag_mean_24h")
    categorical = ("wind_direction", "station")
    frame = frame[[*numeric, *categorical, "polluted"]].reset_index(drop=True)
    return Dataset(
        "beijing",
        frame,
        numeric=numeric,
        categorical=categorical,
        target="polluted",
        time_ordered=True,
        source="UCI Beijing Multi-Site Air Quality, CC BY 4.0, https://doi.org/10.24432/C5RK5G",
        group_by="station",
    )
