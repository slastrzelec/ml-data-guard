"""Distribution shift between two batches, and training/serving skew row by row."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlguard.compare import (
    calibrate_thresholds,
    compare_batches,
    feature_skew,
    skew_anomalies,
)
from mlguard.datasets import make_synthetic, make_synthetic_series
from mlguard.inject import FAULTS
from mlguard.schema import Schema, infer_schema


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame, Schema]:
    frame = make_synthetic(60_000, seed=21).frame
    reference, fresh = frame.iloc[:40_000], frame.iloc[40_000:]
    return reference, fresh, infer_schema(reference, exclude=["target"])


def shifted(batch: pd.DataFrame, fault: str, feature: str, severity: float) -> pd.DataFrame:
    return FAULTS[fault].apply(batch, feature, severity, np.random.default_rng(0))


def test_clean_batches_rarely_alarm(data: tuple[pd.DataFrame, pd.DataFrame, Schema]) -> None:
    reference, fresh, schema = data
    calibration = calibrate_thresholds(reference, schema, batch_size=1_000, n_batches=300, seed=1)
    references, thresholds = calibration
    assert set(thresholds) == set(schema.names) == set(references)
    alarms = 0
    for start in range(0, 20_000, 1_000):
        batch = fresh.iloc[start : start + 1_000]
        alarms += len(compare_batches(reference, batch, schema, calibration=calibration))
    assert alarms <= 6  # 20 batches x 7 features at a 1% target: about 1.4 expected


@pytest.mark.parametrize(
    ("fault", "feature", "severity"),
    [
        ("pinned_value", "x_normal", 0.1),
        ("unit_change", "x_lognormal", 0.1),
        ("new_category", "c_small", 0.1),
        ("missing_rate_increase", "x_count", 0.1),
        ("mean_shift", "x_normal", 0.5),
    ],
)
def test_shifted_batches_alarm_on_the_shifted_feature(
    data: tuple[pd.DataFrame, pd.DataFrame, Schema], fault: str, feature: str, severity: float
) -> None:
    reference, fresh, schema = data
    batch = shifted(fresh.iloc[:2_000], fault, feature, severity)
    report = compare_batches(reference, batch, schema, n_batches=200)
    assert feature in [anomaly.feature for anomaly in report]
    found = next(anomaly for anomaly in report if anomaly.feature == feature)
    assert found.type.value == "DISTRIBUTION_SHIFT"
    assert found.statistic > found.threshold
    assert "calibrated for 2000 rows" in found.description


def test_calibration_does_not_see_the_batch_under_test(
    data: tuple[pd.DataFrame, pd.DataFrame, Schema],
) -> None:
    """Leakage rule: the limit a batch is compared with does not depend on that batch."""
    reference, fresh, schema = data
    clean = fresh.iloc[:1_000]
    broken = shifted(clean, "unit_change", "x_normal", 0.5)
    limits = []
    for batch in (clean, broken):
        report = compare_batches(reference, batch, schema, n_batches=200, seed=3)
        limits.append({anomaly.feature: anomaly.threshold for anomaly in report})
    _, expected = calibrate_thresholds(reference, schema, batch_size=1_000, n_batches=200, seed=3)
    assert limits[1]["x_normal"] == expected["x_normal"]
    for feature, limit in limits[0].items():
        assert limit == expected[feature]


def test_calibration_is_seeded_and_depends_on_the_batch_size(
    data: tuple[pd.DataFrame, pd.DataFrame, Schema],
) -> None:
    reference, _, schema = data
    _, first = calibrate_thresholds(reference, schema, batch_size=500, n_batches=200, seed=5)
    _, again = calibrate_thresholds(reference, schema, batch_size=500, n_batches=200, seed=5)
    _, larger = calibrate_thresholds(reference, schema, batch_size=5_000, n_batches=200, seed=5)
    assert first == again
    assert all(larger[name] < first[name] for name in first)  # more rows, less noise, lower limit


def test_a_fixed_threshold_is_used_as_given(
    data: tuple[pd.DataFrame, pd.DataFrame, Schema],
) -> None:
    reference, fresh, schema = data
    batch = fresh.iloc[:1_000]
    assert compare_batches(reference, batch, schema, threshold=0.5) == []
    everything = compare_batches(reference, batch, schema, threshold=0.0)
    assert {anomaly.feature for anomaly in everything} == set(schema.names)
    assert all("fixed" in anomaly.description for anomaly in everything)


def test_other_detectors_and_their_limits(data: tuple[pd.DataFrame, pd.DataFrame, Schema]) -> None:
    reference, fresh, schema = data
    batch = shifted(fresh.iloc[:2_000], "new_category", "c_small", 0.01)
    by_chi2 = compare_batches(reference, batch, schema, detector="chi2", n_batches=200)
    assert "c_small" in [anomaly.feature for anomaly in by_chi2]
    with pytest.raises(ValueError, match="unknown detector"):
        compare_batches(reference, batch, schema, detector="magic")
    with pytest.raises(ValueError, match="does not apply to categorical"):
        compare_batches(reference, batch, schema, detector="ks")
    numeric_only = infer_schema(reference[["x_normal"]])
    assert (
        compare_batches(reference, fresh.iloc[:2_000], numeric_only, detector="ks", n_batches=200)
        == []
    )


def test_missing_and_deprecated_features_are_left_to_the_validator(
    data: tuple[pd.DataFrame, pd.DataFrame, Schema],
) -> None:
    reference, fresh, schema = data
    batch = shifted(fresh.iloc[:1_000], "unit_change", "x_normal", 0.5).drop(columns=["x_count"])
    report = compare_batches(reference, batch, schema, n_batches=200)
    assert [anomaly.feature for anomaly in report] == ["x_normal"]


def test_invalid_comparison_input_is_rejected(
    data: tuple[pd.DataFrame, pd.DataFrame, Schema],
) -> None:
    reference, fresh, schema = data
    batch = fresh.iloc[:100]
    with pytest.raises(ValueError, match="non-empty"):
        compare_batches(reference, batch.iloc[:0], schema)
    with pytest.raises(ValueError, match="threshold"):
        compare_batches(reference, batch, schema, threshold=-1.0)
    with pytest.raises(ValueError, match="target_fpr"):
        compare_batches(reference, batch, schema, target_fpr=0.0)
    with pytest.raises(ValueError, match="at least 1"):
        compare_batches(reference, batch, schema, n_batches=50)
    with pytest.raises(ValueError, match="half of the"):
        compare_batches(reference.iloc[:150], batch, schema, n_batches=200)


# --------------------------------------------------------------------------- skew


def logs(n_rows: int = 2_000) -> pd.DataFrame:
    frame = make_synthetic_series(n_rows, seed=4).frame
    frame.insert(0, "row_id", np.arange(n_rows))
    return frame


def test_identical_logs_have_no_skew() -> None:
    training = logs()
    report = feature_skew(training, training.sample(frac=1.0, random_state=0), key="row_id")
    assert report.n_joined == 2_000
    assert (report.n_training_only, report.n_serving_only) == (0, 0)
    assert set(report.mismatches.values()) == {0}
    assert skew_anomalies(report) == []


def test_skew_finds_time_travel_that_the_distances_miss() -> None:
    """Acceptance criterion 3 of phase 2."""
    training = logs()
    serving = FAULTS["time_travel"].apply(training, "lag_1", 1, np.random.default_rng(0))
    schema = infer_schema(training, exclude=["row_id", "target"])
    assert compare_batches(training, serving, schema, threshold=0.01) == []
    report = feature_skew(training, serving, key="row_id")
    assert report.mismatches["lag_1"] == 1_999  # every row but the last one, which has no future
    assert report.mismatches["lag_mean_3"] == 0
    anomalies = skew_anomalies(report)
    assert [(a.type.value, a.feature) for a in anomalies] == [("FEATURE_SKEW", "lag_1")]
    assert anomalies[0].statistic == pytest.approx(1_999 / 2_000)


def test_skew_counts_rows_that_exist_on_one_side_only() -> None:
    training = logs(100)
    serving = training.iloc[10:].copy()
    serving.loc[len(training)] = serving.iloc[0].to_dict() | {"row_id": 999}
    report = feature_skew(training, serving, key="row_id")
    assert (report.n_joined, report.n_training_only, report.n_serving_only) == (90, 10, 1)


def test_skew_comparison_rules() -> None:
    training = pd.DataFrame(
        {
            "key": [1, 2, 3, 4, 5],
            "number": [1.0, 2.0, math.nan, 4.0, 5.0],
            "text": np.array(["a", "b", None, "1", "e"], dtype=object),
        }
    )
    serving = pd.DataFrame(
        {
            "key": [1, 2, 3, 4, 5],
            "number": [1.0 + 1e-12, 2.5, math.nan, math.nan, 5],
            "text": np.array(["a", "B", None, 1, "e"], dtype=object),
        }
    )
    report = feature_skew(training, serving, key="key")
    # float noise is not skew, two missing values are equal, missing vs value is skew
    assert report.mismatches["number"] == 2
    # case matters, and the text "1" is not the number 1
    assert report.mismatches["text"] == 2
    assert report.share("number") == pytest.approx(0.4)
    loose = feature_skew(training, serving, key="key", features=["number"], rtol=0.5)
    assert loose.mismatches == {"number": 1}
    assert [a.feature for a in skew_anomalies(report, max_share=0.4)] == []
    assert len(skew_anomalies(report, max_share=0.39)) == 2


def test_invalid_skew_input_is_rejected() -> None:
    training = logs(50)
    with pytest.raises(ValueError, match="no key column"):
        feature_skew(training, training, key="missing")
    with pytest.raises(ValueError, match="unique"):
        feature_skew(training, pd.concat([training, training]), key="row_id")
    with pytest.raises(ValueError, match="not present in both"):
        feature_skew(training, training.drop(columns=["lag_1"]), key="row_id", features=["lag_1"])
    with pytest.raises(ValueError, match="non-empty"):
        feature_skew(training, training.iloc[:0], key="row_id")
    with pytest.raises(ValueError, match="max_share"):
        skew_anomalies(feature_skew(training, training, key="row_id"), max_share=2.0)
    disjoint = training.assign(row_id=training["row_id"] + 1_000)
    nothing = skew_anomalies(feature_skew(training, disjoint, key="row_id"))
    assert [a.feature for a in nothing] == [None]
    assert "nothing could be compared" in nothing[0].description
