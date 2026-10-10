"""Synthetic generators and the Covertype loader (tested on a small hand-made archive)."""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from mlguard.datasets import (
    COVERTYPE_NUMERIC,
    load_covertype,
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
