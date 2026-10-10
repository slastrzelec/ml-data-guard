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
