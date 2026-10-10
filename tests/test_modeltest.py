"""Model unit test: generated rows obey the schema, and hidden assumptions are exposed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from mlguard.datasets import make_synthetic
from mlguard.modeltest import generate_examples, run_model_test
from mlguard.schema import FeatureSchema, Schema, infer_schema
from mlguard.validate import validate_batch

SCHEMA = infer_schema(make_synthetic(3_000, seed=8).frame, exclude=["target"])


@settings(max_examples=40)
@given(st.integers(1, 300), st.integers(0, 2**32 - 1), st.sampled_from(["plain", "hashed", "none"]))
def test_generated_rows_satisfy_the_schema(n_rows: int, seed: int, domains: str) -> None:
    schema = infer_schema(make_synthetic(500, seed=8).frame, exclude=["target"], domains=domains)
    if domains == "hashed":  # hashed domains cannot be turned back into values
        return
    examples = generate_examples(schema, n_rows=n_rows, seed=seed)
    assert examples.shape == (n_rows, len(schema.features))
    assert validate_batch(examples, schema) == []


def test_generation_is_seeded() -> None:
    assert_frame_equal(
        generate_examples(SCHEMA, 100, seed=1), generate_examples(SCHEMA, 100, seed=1)
    )
    assert not generate_examples(SCHEMA, 100, seed=1).equals(generate_examples(SCHEMA, 100, seed=2))


def test_the_edges_of_every_feature_are_generated() -> None:
    examples = generate_examples(SCHEMA, 200, seed=0)
    for feature in SCHEMA.features:
        column = examples[feature.name]
        if feature.kind == "numeric":
            assert column.min() == pytest.approx(
                np.ceil(feature.minimum) if feature.dtype == "int" else feature.minimum
            )
            assert column.max() == pytest.approx(
                np.floor(feature.maximum) if feature.dtype == "int" else feature.maximum
            )
        else:
            assert set(column.dropna()) == set(feature.domain)
    assert (examples["x_sentinel"] == 0).any()  # zero is inside the range, so it is tried
    assert examples["x_normal"].isna().any()  # this feature may be missing
    assert not examples["x_lognormal"].isna().any()  # this one may not
    assert examples["x_count"].dtype == np.int64


def test_unbounded_free_text_and_deprecated_features() -> None:
    schema = Schema(
        features=(
            FeatureSchema("anything", "numeric", "float"),
            FeatureSchema("note", "categorical", "string", min_presence=0.5, domain_storage="none"),
            FeatureSchema("code", "categorical", "int", domain_storage="none"),
            FeatureSchema("flag", "categorical", "bool", domain_storage="none"),
            FeatureSchema("old", "numeric", "float", stage="deprecated"),
        )
    )
    examples = generate_examples(schema, 50, seed=0)
    assert list(examples.columns) == ["anything", "note", "code", "flag"]
    assert examples["anything"].min() == -1e6
    assert examples["anything"].max() == 1e6
    assert examples["note"].iloc[0] == ""
    assert 0 < examples["note"].isna().sum() <= 25
    assert validate_batch(examples, schema) == []


def test_a_function_with_a_hidden_assumption_fails_with_a_reproducing_row() -> None:
    def careless(batch: pd.DataFrame) -> np.ndarray:
        return np.log(batch["x_sentinel"].to_numpy())  # the schema allows -1 and 0

    result = run_model_test(careless, SCHEMA)
    assert not result
    assert "FloatingPointError" in result.problem
    assert result.failing_example["x_sentinel"] <= 0
    single = pd.DataFrame([result.failing_example])
    with np.errstate(all="raise"), pytest.raises(FloatingPointError):
        careless(single)


def test_a_robust_function_passes() -> None:
    def careful(batch: pd.DataFrame) -> np.ndarray:
        values = batch["x_sentinel"].to_numpy(dtype=float)
        return np.where(values > 0, np.log1p(np.abs(values)), 0.0)

    result = run_model_test(careful, SCHEMA)
    assert result
    assert result.problem == ""
    assert result.failing_example is None


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (
            lambda batch: batch["x_normal"].to_numpy() * 2,
            "NaN or infinite",
        ),  # missing values pass through
        (lambda batch: np.zeros(len(batch) + 1), "predictions for"),
        (lambda batch: batch["no_such_column"], "KeyError"),
        (lambda batch: [{"a": 1, "b": 2}[value] for value in batch["c_small"]], "KeyError"),
        (lambda batch: 1.0 / (batch["x_count"].to_numpy() - 3.0), "FloatingPointError"),
    ],
)
def test_kinds_of_failure(function, message: str) -> None:  # type: ignore[no-untyped-def]
    result = run_model_test(function, SCHEMA)
    assert not result
    assert message in result.problem


def test_a_training_step_that_returns_nothing_is_checked_for_exceptions_only() -> None:
    seen = []
    result = run_model_test(lambda batch: seen.append(len(batch)), SCHEMA, n_rows=30)
    assert result
    assert seen == [30]


def test_a_failure_that_needs_several_rows_has_no_single_reproducing_row() -> None:
    def needs_two_rows(batch: pd.DataFrame) -> np.ndarray:
        if len(batch) > 1:
            raise RuntimeError("more than one row")
        return np.zeros(len(batch))

    result = run_model_test(needs_two_rows, SCHEMA, n_rows=10)
    assert not result
    assert result.problem == "RuntimeError: more than one row"
    assert result.failing_example is None


def test_invalid_generation_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_rows"):
        generate_examples(SCHEMA, 0)
    impossible = Schema(features=(FeatureSchema("k", "numeric", "int", minimum=0.2, maximum=0.8),))
    with pytest.raises(ValueError, match="no integer"):
        generate_examples(impossible, 10)
