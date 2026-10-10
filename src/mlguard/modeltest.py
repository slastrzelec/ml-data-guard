"""Model unit test: does the code accept everything the schema allows?

Training and serving code is full of unwritten assumptions (``log(x)`` assumes ``x > 0``). The test
generates rows that are valid according to the schema, including its edges, feeds them to a user
function and reports the first thing that breaks, with a single synthetic row that reproduces it.
Either the code or the schema is then wrong, and both are cheap to fix before real data arrives.
"""

from __future__ import annotations

import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlguard.schema import FeatureSchema, Schema

#: Range used for a numeric feature whose schema has no bound on that side.
UNBOUNDED = 1e6


def _numeric_column(feature: FeatureSchema, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    low = -UNBOUNDED if feature.minimum is None else feature.minimum
    high = UNBOUNDED if feature.maximum is None else feature.maximum
    if feature.dtype == "int":
        low_int, high_int = int(np.ceil(low)), int(np.floor(high))
        if low_int > high_int:
            raise ValueError(f"feature {feature.name!r} has no integer in its range")
        values = rng.integers(low_int, high_int + 1, size=n_rows).astype(np.float64)
        edges = [low_int, high_int] + ([0] if low_int <= 0 <= high_int else [])
    else:
        values = rng.uniform(low, high, size=n_rows)
        edges = [low, high] + ([0.0] if low <= 0 <= high else [])
    # the edges of the range come first: they are where hidden assumptions break
    values[: len(edges)] = edges[:n_rows]
    return values


def _categorical_column(
    feature: FeatureSchema, n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    if feature.domain is not None and feature.domain_storage == "plain":
        domain = np.array(feature.domain, dtype=object)
        values = domain[rng.integers(0, len(domain), size=n_rows)]
        values[: len(domain)] = domain[:n_rows]  # every allowed value at least once
        return values
    # No usable domain (free text, or hashed values that cannot be turned back): invent values.
    if feature.dtype == "bool":
        return rng.integers(0, 2, size=n_rows).astype(bool).astype(object)
    if feature.dtype == "int":
        return rng.integers(-1000, 1000, size=n_rows).astype(object)
    letters = np.array(list(string.ascii_lowercase))
    words = ["".join(rng.choice(letters, size=int(rng.integers(1, 12)))) for _ in range(n_rows)]
    if n_rows:
        words[0] = ""  # the empty string is valid text
    return np.array(words, dtype=object)


def generate_examples(schema: Schema, n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Rows that the schema allows: random values plus the edges of every feature.

    Per feature: the range minimum, maximum and zero (when allowed), every plain domain value,
    and missing values when ``min_presence < 1`` (as many as the schema tolerates, at least one).
    Deterministic for a seed. Deprecated features are left out.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    columns: dict[str, Any] = {}
    for position, feature in enumerate(schema.active):
        rng = np.random.default_rng([seed, position])
        if feature.kind == "numeric":
            values: np.ndarray = _numeric_column(feature, n_rows, rng)
        else:
            values = _categorical_column(feature, n_rows, rng)
        n_missing = int(np.floor((1.0 - feature.min_presence) * n_rows + 1e-9))
        if n_missing:
            where = n_rows - 1 - rng.permutation(max(1, n_rows // 2))[:n_missing]
            values = values.astype(object if feature.kind == "categorical" else np.float64)
            values[where] = None if feature.kind == "categorical" else np.nan
        if feature.kind == "numeric" and feature.dtype == "int" and not n_missing:
            values = values.astype(np.int64)
        columns[feature.name] = values
    return pd.DataFrame(columns)


@dataclass(frozen=True)
class ModelTestResult:
    """Outcome of a model unit test."""

    passed: bool
    n_rows: int
    problem: str = ""
    #: one generated row that reproduces the problem on its own, when such a row exists
    failing_example: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.passed


def _problem(function: Callable[[pd.DataFrame], Any], rows: pd.DataFrame) -> str:
    """Empty string when ``function`` handles ``rows``, otherwise a description of what broke."""
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            output = function(rows)
    except Exception as error:  # the user's function may raise anything
        return f"{type(error).__name__}: {error}"
    if output is None:
        return ""
    try:
        array = np.asarray(output)
    except Exception as error:
        return f"output cannot be read as an array ({type(error).__name__})"
    if array.ndim == 0:
        return ""
    if array.shape[0] != len(rows):
        return f"returned {array.shape[0]} predictions for {len(rows)} rows"
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        return "output contains NaN or infinite values"
    return ""


def run_model_test(
    function: Callable[[pd.DataFrame], Any],
    schema: Schema,
    *,
    n_rows: int = 200,
    seed: int = 0,
) -> ModelTestResult:
    """Feed schema-valid rows to ``function`` and report the first failure.

    The function fails the test when it raises, when NumPy reports a floating-point error
    (division by zero, invalid value, overflow), when it returns NaN or infinity, or when the
    number of returned predictions differs from the number of rows. A function that returns
    nothing (a training step) is only checked for exceptions.

    When the whole table fails, each row is tried on its own; the first row that fails by itself
    is returned as ``failing_example``. The rows are synthetic and safe to show.
    """
    examples = generate_examples(schema, n_rows=n_rows, seed=seed)
    problem = _problem(function, examples)
    if not problem:
        return ModelTestResult(passed=True, n_rows=n_rows)
    for position in range(len(examples)):
        row = examples.iloc[[position]]
        if _problem(function, row):
            record = {
                name: (None if isinstance(value, float) and np.isnan(value) else value)
                for name, value in row.iloc[0].to_dict().items()
            }
            return ModelTestResult(False, n_rows, problem, record)
    return ModelTestResult(False, n_rows, problem, None)
