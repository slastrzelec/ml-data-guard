"""Single-batch validator: every anomaly has a minimal trigger and stays quiet on clean data."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mlguard.datasets import make_synthetic
from mlguard.inject import FAULTS
from mlguard.schema import FeatureSchema, Schema, infer_schema
from mlguard.validate import Anomaly, AnomalyType, has_errors, validate_batch

SCHEMA = Schema(
    features=(
        FeatureSchema("amount", "numeric", "float", min_presence=0.9, minimum=0.0, maximum=100.0),
        FeatureSchema("count", "numeric", "int", minimum=0, maximum=10),
        FeatureSchema("colour", "categorical", "string", domain=("blue", "red")),
        FeatureSchema("note", "categorical", "string", min_presence=0.0, domain_storage="none"),
        FeatureSchema("legacy", "numeric", "float", stage="deprecated"),
    ),
    ignored=("id",),
)


def clean(n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "amount": np.linspace(0.0, 100.0, n_rows),
            "count": np.arange(n_rows) % 11,
            "colour": np.array(["red", "blue"] * (n_rows // 2), dtype=object),
            "note": np.array(["free text"] * n_rows, dtype=object),
        }
    )


def types_of(report: list[Anomaly]) -> list[str]:
    return [anomaly.type.value for anomaly in report]


def test_a_clean_batch_has_no_anomaly() -> None:
    assert validate_batch(clean(), SCHEMA) == []


def test_empty_batch() -> None:
    report = validate_batch(clean().iloc[:0], SCHEMA)
    assert types_of(report) == ["EMPTY_BATCH"]
    assert report[0].level == "error"


def test_missing_feature_but_not_a_deprecated_or_ignored_one() -> None:
    report = validate_batch(clean().drop(columns=["amount", "id"]), SCHEMA)
    assert [(anomaly.type.value, anomaly.feature) for anomaly in report] == [
        ("MISSING_FEATURE", "amount")
    ]
    assert "deprecated" in report[0].suggestion


def test_unexpected_feature_is_a_warning() -> None:
    report = validate_batch(clean().assign(surprise=1), SCHEMA)
    assert [(a.type.value, a.feature, a.level) for a in report] == [
        ("UNEXPECTED_FEATURE", "surprise", "warning")
    ]
    assert not has_errors(report)
    assert has_errors(report, strict=True)


def test_type_mismatch_in_a_numeric_feature_is_reported_once() -> None:
    batch = clean()
    batch["amount"] = batch["amount"].astype(object)
    batch.loc[0:3, "amount"] = ["1.0", "2.0", "three", "4"]
    report = validate_batch(batch, SCHEMA)
    assert types_of(report) == ["TYPE_MISMATCH"]  # not also missing, not also out of range
    assert report[0].statistic == pytest.approx(4 / 20)
    assert "string" in report[0].description
    assert "three" not in report[0].description


def test_integer_feature_accepts_whole_floats_and_rejects_fractions() -> None:
    whole = clean().assign(count=lambda frame: frame["count"].astype(float))
    assert validate_batch(whole, SCHEMA) == []
    fractional = whole.copy()
    fractional.loc[0, "count"] = 2.5
    report = validate_batch(fractional, SCHEMA)
    assert types_of(report) == ["TYPE_MISMATCH"]
    assert "float" in report[0].description
    booleans = clean().assign(count=True)
    assert types_of(validate_batch(booleans, SCHEMA)) == ["TYPE_MISMATCH"]


def test_type_mismatch_in_a_categorical_feature() -> None:
    batch = clean().astype({"colour": object, "note": object})
    batch.loc[0, "colour"] = -1
    batch.loc[1, "note"] = 12.5
    report = validate_batch(batch, SCHEMA)
    assert [(a.type.value, a.feature) for a in report] == [
        (
            "TYPE_MISMATCH",
            "colour",
        ),  # a number in a text feature is a type problem, not a new value
        ("TYPE_MISMATCH", "note"),
    ]


def test_low_presence_respects_the_threshold() -> None:
    batch = clean()
    batch.loc[0:1, "amount"] = math.nan  # 10% missing: exactly at the limit
    assert validate_batch(batch, SCHEMA) == []
    batch.loc[2, "amount"] = math.nan
    report = validate_batch(batch, SCHEMA)
    assert types_of(report) == ["LOW_PRESENCE"]
    assert report[0].statistic == pytest.approx(0.85)
    assert report[0].threshold == 0.9
    assert "0.85" in report[0].suggestion
    batch = clean().assign(note=None)
    assert validate_batch(batch, SCHEMA) == []  # this feature may be entirely missing


def test_out_of_range_reports_the_observed_range() -> None:
    batch = clean()
    batch.loc[0, "amount"] = 250.0
    batch.loc[1, "amount"] = -5.0
    report = validate_batch(batch, SCHEMA)
    assert types_of(report) == ["OUT_OF_RANGE"]
    assert report[0].statistic == pytest.approx(2 / 20)
    assert "[-5, 250]" in report[0].description
    assert "[-5, 250]" in report[0].suggestion
    edges = clean()
    edges.loc[0, "amount"], edges.loc[1, "amount"] = 0.0, 100.0
    assert validate_batch(edges, SCHEMA) == []  # the bounds themselves are allowed


def test_unexpected_values_hide_the_values_unless_asked() -> None:
    batch = clean()
    batch.loc[0:2, "colour"] = ["green", "green", "secret-value"]
    report = validate_batch(batch, SCHEMA)
    assert types_of(report) == ["UNEXPECTED_VALUES"]
    assert report[0].statistic == pytest.approx(3 / 20)
    assert "2 distinct" in report[0].description
    assert "secret-value" not in json.dumps(report[0].to_dict())
    shown = validate_batch(batch, SCHEMA, show_values=True)
    assert "'green'" in shown[0].description
    assert "'secret-value'" in shown[0].description


def test_tolerances_allow_a_known_share_of_junk() -> None:
    tolerant = Schema(
        features=(
            FeatureSchema(
                "amount", "numeric", "float", minimum=0, maximum=100, max_out_of_range_fraction=0.1
            ),
            FeatureSchema(
                "colour",
                "categorical",
                "string",
                domain=("blue", "red"),
                max_unexpected_fraction=0.1,
            ),
        ),
        ignored=("id", "count", "note"),
    )
    batch = clean()
    batch.loc[0:1, "amount"] = 1e6
    batch.loc[0:1, "colour"] = "green"
    assert validate_batch(batch, tolerant) == []
    batch.loc[2, "amount"] = 1e6
    batch.loc[2, "colour"] = "green"
    assert types_of(validate_batch(batch, tolerant)) == ["OUT_OF_RANGE", "UNEXPECTED_VALUES"]


def test_every_anomaly_explains_itself() -> None:
    batch = clean().drop(columns=["count"]).assign(surprise=1)
    batch.loc[0, "amount"] = 250.0
    batch.loc[0, "colour"] = "green"
    report = validate_batch(batch, SCHEMA)
    assert set(types_of(report)) == {
        "MISSING_FEATURE", "UNEXPECTED_FEATURE", "OUT_OF_RANGE", "UNEXPECTED_VALUES",
    }  # fmt: skip
    for anomaly in report:
        entry = anomaly.to_dict()
        assert set(entry) == {
            "type", "feature", "level", "description", "statistic", "threshold", "suggestion",
        }  # fmt: skip
        assert entry["type"] in {member.value for member in AnomalyType}
        assert entry["description"]
        assert entry["suggestion"]
        json.dumps(entry)  # plain JSON


def test_a_batch_that_is_not_a_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="DataFrame"):
        validate_batch([{"amount": 1.0}], SCHEMA)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- with the injectors

DETECTED = {
    "dropped_feature": ("x_normal", 1, "MISSING_FEATURE"),
    "type_change": ("x_normal", 0.01, "TYPE_MISMATCH"),
    "new_category": ("c_small", 0.01, "UNEXPECTED_VALUES"),
    "out_of_domain_numeric": ("x_count", 0.01, "OUT_OF_RANGE"),
    "unit_change": ("x_normal", 0.01, "OUT_OF_RANGE"),
    "missing_rate_increase": ("x_lognormal", 0.01, "LOW_PRESENCE"),
}
# Faults that keep every value inside the schema: a job for the distribution comparison.
NOT_DETECTED = {
    "mean_shift": ("x_normal", 0.05),  # a larger shift can leave the inferred range
    "variance_change": ("x_normal", 0.05),
    "duplicate_rows": (None, 0.2),
    "tiny_noise": ("x_normal", 0.01),
    "pinned_value": ("x_sentinel", 0.2),  # -1 is a legal value of this feature
    "time_travel": ("x_normal", 3),
}


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, Schema]:
    frame = make_synthetic(6_000, seed=3).frame
    reference, batch = frame.iloc[:5_000], frame.iloc[5_000:]
    return batch, infer_schema(reference, exclude=["target"])


def test_the_catalogue_is_split_into_caught_and_not_caught() -> None:
    assert set(DETECTED) | set(NOT_DETECTED) == set(FAULTS)


@pytest.mark.parametrize("fault", sorted(DETECTED))
def test_schema_violations_of_the_injectors_are_caught(
    synthetic: tuple[pd.DataFrame, Schema], fault: str
) -> None:
    batch, schema = synthetic
    feature, severity, expected = DETECTED[fault]
    assert validate_batch(batch, schema) == []
    faulty = FAULTS[fault].apply(batch, feature, severity, np.random.default_rng(0))
    report = validate_batch(faulty, schema)
    assert [(anomaly.type.value, anomaly.feature) for anomaly in report] == [(expected, feature)]


@pytest.mark.parametrize("fault", sorted(NOT_DETECTED))
def test_faults_inside_the_schema_are_not_caught_by_the_validator(
    synthetic: tuple[pd.DataFrame, Schema], fault: str
) -> None:
    """Documented limit: these need the distribution comparison or the skew check."""
    batch, schema = synthetic
    feature, severity = NOT_DETECTED[fault]
    faulty = FAULTS[fault].apply(batch, feature, severity, np.random.default_rng(0))
    assert validate_batch(faulty, schema) == []


def test_a_sentinel_outside_the_range_is_caught() -> None:
    frame = make_synthetic(3_000, seed=3).frame
    schema = infer_schema(frame, exclude=["target"])
    faulty = FAULTS["pinned_value"].apply(frame, "x_lognormal", 0.01, np.random.default_rng(0))
    assert types_of(validate_batch(faulty, schema)) == ["OUT_OF_RANGE"]
