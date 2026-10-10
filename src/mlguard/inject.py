"""Fault injectors: turn a clean batch into a batch with one known, labelled data error.

Every injector is a pure function ``(batch, feature, severity, rng) -> faulty batch``:

* the input frame is never modified and the output never shares memory with it;
* the same seed gives the same output;
* ``severity == 0`` returns an equal copy, and a larger severity never repairs what a smaller one
  broke. For the faults measured in a fraction of rows the affected rows are nested: with the same
  seed, every row changed at severity ``s`` is changed in the same way at any larger severity;
* index, column order and all other columns are preserved (``dropped_feature`` removes one column,
  ``duplicate_rows`` rewrites whole rows).

What ``severity`` means depends on the fault and is recorded in ``FAULTS[name].severity_unit``.
A severity greater than zero always changes at least one eligible row, so a batch labelled as
faulty is never silently clean.

Injectors only see the batch they are given. They never look at reference data, thresholds or
detector output.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype, is_numeric_dtype

SeverityUnit = Literal["fraction_of_rows", "sigma", "relative_scale", "lead_steps", "binary"]
Target = Literal["numeric", "categorical", "any", "rows"]

# --------------------------------------------------------------------------- helpers


def _column(batch: pd.DataFrame, feature: str) -> pd.Series:
    if not isinstance(batch, pd.DataFrame):
        raise ValueError(f"batch must be a pandas DataFrame, got {type(batch).__name__}")
    if len(batch) == 0:
        raise ValueError("batch must not be empty")
    if feature not in batch.columns:
        raise ValueError(f"batch has no feature {feature!r}")
    return batch[feature]


def _is_numeric(series: pd.Series) -> bool:
    return bool(is_numeric_dtype(series.dtype)) and not is_bool_dtype(series.dtype)


def _numeric_column(batch: pd.DataFrame, feature: str, fault: str) -> pd.Series:
    series = _column(batch, feature)
    if not _is_numeric(series):
        raise ValueError(f"{fault} needs a numeric feature, {feature!r} is {series.dtype}")
    return series


def _fraction(severity: float) -> float:
    if not (isinstance(severity, int | float) and math.isfinite(severity) and 0 <= severity <= 1):
        raise ValueError(f"severity must be a fraction of rows in [0, 1], got {severity!r}")
    return float(severity)


def _non_negative(severity: float) -> float:
    if not (isinstance(severity, int | float) and math.isfinite(severity) and severity >= 0):
        raise ValueError(f"severity must be a finite non-negative number, got {severity!r}")
    return float(severity)


def _row_count(fraction: float, n_eligible: int) -> int:
    """Number of rows to change: at least one whenever the fraction is positive."""
    if fraction == 0 or n_eligible == 0:
        return 0
    return min(n_eligible, max(1, math.ceil(fraction * n_eligible - 1e-9)))


def _pick(eligible: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Positions of the rows to change; nested in ``fraction`` for a fixed generator state."""
    candidates = np.flatnonzero(eligible)
    order = rng.permutation(candidates)
    return np.sort(order[: _row_count(fraction, candidates.size)])


def _assign(values: np.ndarray, rows: np.ndarray, new: Any) -> np.ndarray:
    """Copy of ``values`` with ``new`` written to ``rows``, upcasting the dtype when needed."""
    if values.dtype == object:
        result = values.copy()
    else:
        result = values.astype(np.result_type(values.dtype, np.asarray(new).dtype), copy=True)
    result[rows] = new
    return result


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _with_column(batch: pd.DataFrame, feature: str, values: np.ndarray) -> pd.DataFrame:
    """Deep copy of ``batch`` with one column replaced.

    A text column keeps its string dtype as long as it still holds only text; anything else that
    is not numeric becomes ``object`` so that no value is converted behind the caller's back.
    """
    original = batch[feature].dtype
    if values.dtype == object:
        column = pd.Series(values, index=batch.index, name=feature, dtype=object)
        only_text = all(isinstance(value, str) or _is_missing(value) for value in values)
        if isinstance(original, pd.StringDtype) and only_text:
            column = column.astype(original)
    else:
        column = pd.Series(values, index=batch.index, name=feature)
    result = batch.copy(deep=True)
    result[feature] = column
    return result


def _spread(values: np.ndarray, feature: str, fault: str) -> tuple[float, float]:
    """Mean and standard deviation of the non-missing values; a constant feature is an error."""
    present = values[~np.isnan(values)]
    if present.size == 0:
        raise ValueError(f"{fault}: feature {feature!r} has no non-missing value")
    mean, std = float(present.mean()), float(present.std())
    if std == 0:
        raise ValueError(f"{fault}: feature {feature!r} is constant in this batch")
    return mean, std


# --------------------------------------------------------------------------- faults on one feature


def pinned_value(
    batch: pd.DataFrame,
    feature: str,
    severity: float,
    rng: np.random.Generator,
    *,
    sentinel: Hashable = -1,
) -> pd.DataFrame:
    """Set a fraction of the rows that do not already hold ``sentinel`` to ``sentinel``."""
    series = _column(batch, feature)
    fraction = _fraction(severity)
    kind = object if not _is_numeric(series) else None
    values = series.to_numpy(dtype=kind, copy=True)
    rows = _pick(~(series == sentinel).to_numpy(dtype=bool, na_value=False), fraction, rng)
    return _with_column(batch, feature, _assign(values, rows, sentinel))


def dropped_feature(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Remove the column. Binary fault: any positive severity drops it."""
    _column(batch, feature)
    result = batch.copy(deep=True)
    return result.drop(columns=[feature]) if _non_negative(severity) > 0 else result


def type_change(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Turn a fraction of the non-missing numbers into their string form (``3.5`` -> ``"3.5"``)."""
    series = _numeric_column(batch, feature, "type_change")
    rows = _pick(series.notna().to_numpy(), _fraction(severity), rng)
    if rows.size == 0:
        return batch.copy(deep=True)
    values = series.to_numpy(dtype=object, copy=True)
    values[rows] = [str(value) for value in values[rows]]
    return _with_column(batch, feature, values)


def new_category(
    batch: pd.DataFrame,
    feature: str,
    severity: float,
    rng: np.random.Generator,
    *,
    category: Hashable | None = None,
) -> pd.DataFrame:
    """Replace a fraction of rows with a category that does not occur in the batch.

    Works on categorical (non-float) features. Without ``category`` the new value is ``"__new__"``
    for text features and ``max + 1`` for integer-coded ones.
    """
    series = _column(batch, feature)
    if is_float_dtype(series.dtype):
        raise ValueError(f"new_category needs a categorical feature, {feature!r} is {series.dtype}")
    fraction = _fraction(severity)
    integer_coded = is_integer_dtype(series.dtype)
    values = series.to_numpy(dtype=None if integer_coded else object, copy=True)
    existing = set(series.dropna().unique().tolist())
    if category is None:
        category = int(series.max()) + 1 if integer_coded else "__new__"
        while not integer_coded and category in existing:
            category = f"{category}_"
    elif category in existing:
        raise ValueError(f"category {category!r} already occurs in feature {feature!r}")
    rows = _pick(np.ones(len(series), dtype=bool), fraction, rng)
    return _with_column(batch, feature, _assign(values, rows, category))


def out_of_domain_numeric(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Move a fraction of rows above the largest value in the batch (by 10-100% of its range)."""
    series = _numeric_column(batch, feature, "out_of_domain_numeric")
    fraction = _fraction(severity)
    values = series.to_numpy(copy=True)
    present = ~np.isnan(values.astype(np.float64))
    if not present.any():
        raise ValueError(f"out_of_domain_numeric: feature {feature!r} has no non-missing value")
    low, high = float(values[present].min()), float(values[present].max())
    span = high - low if high > low else max(abs(high), 1.0)
    rows = _pick(present, fraction, rng)
    # One draw per row, so that the values are nested in severity like the rows are.
    new = high + span * rng.uniform(0.1, 1.0, size=len(values))[rows]
    if is_integer_dtype(series.dtype):
        new = np.ceil(new).astype(values.dtype)
    return _with_column(batch, feature, _assign(values, rows, new))


def missing_rate_increase(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Set a fraction of the non-missing rows to missing (NaN for numbers, None for text).

    An integer feature becomes float, as it would after a real upstream change.
    """
    series = _column(batch, feature)
    rows = _pick(series.notna().to_numpy(), _fraction(severity), rng)
    if rows.size == 0:
        return batch.copy(deep=True)
    if _is_numeric(series):
        return _with_column(batch, feature, _assign(series.to_numpy(copy=True), rows, np.nan))
    values = series.to_numpy(dtype=object, copy=True)
    return _with_column(batch, feature, _assign(values, rows, None))


def unit_change(
    batch: pd.DataFrame,
    feature: str,
    severity: float,
    rng: np.random.Generator,
    *,
    factor: float = 1000.0,
) -> pd.DataFrame:
    """Multiply a fraction of the non-missing, non-zero rows by ``factor`` (e.g. km -> m)."""
    series = _numeric_column(batch, feature, "unit_change")
    if not math.isfinite(factor) or factor in (0, 1):
        raise ValueError(f"factor must be finite and different from 0 and 1, got {factor!r}")
    values = series.to_numpy(copy=True)
    as_float = values.astype(np.float64)
    rows = _pick(~np.isnan(as_float) & (as_float != 0), _fraction(severity), rng)
    scale = int(factor) if is_integer_dtype(series.dtype) and float(factor).is_integer() else factor
    return _with_column(batch, feature, _assign(values, rows, values[rows] * scale))


def _keep_integers(original: np.ndarray, moved: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round ``moved`` back to integers without bias (up with probability = fractional part).

    If rounding happens to undo every change, the value that lies closest to the middle between
    two integers is rounded the other way, so that a positive severity never returns the original
    column and no value ends up further than one from its exact target.
    """
    floor = np.floor(moved)
    fraction = moved - floor
    rounded = floor + (rng.random(moved.size) < fraction)
    if np.array_equal(rounded, original) and fraction.max() > 0:
        position = int(np.argmax(np.minimum(fraction, 1.0 - fraction)))
        rounded[position] = 2 * floor[position] + 1 - rounded[position]
    return rounded.astype(original.dtype)


def mean_shift(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Add ``severity`` batch standard deviations to every value.

    An integer feature stays integer: the shifted values are rounded at random so that the mean
    still moves by the requested amount on average (a count does not become 4.02).
    """
    series = _numeric_column(batch, feature, "mean_shift")
    sigmas = _non_negative(severity)
    values = series.to_numpy(dtype=np.float64, copy=True)
    _, std = _spread(values, feature, "mean_shift")
    if sigmas == 0:
        return batch.copy(deep=True)
    moved = values + sigmas * std
    if is_integer_dtype(series.dtype):
        return _with_column(batch, feature, _keep_integers(series.to_numpy(), moved, rng))
    return _with_column(batch, feature, moved)


def variance_change(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Stretch the values around the batch mean: the standard deviation grows by 1 + severity.

    An integer feature stays integer (unbiased random rounding, as in ``mean_shift``).
    """
    series = _numeric_column(batch, feature, "variance_change")
    growth = _non_negative(severity)
    values = series.to_numpy(dtype=np.float64, copy=True)
    mean, _ = _spread(values, feature, "variance_change")
    if growth == 0:
        return batch.copy(deep=True)
    moved = mean + (values - mean) * (1.0 + growth)
    if is_integer_dtype(series.dtype):
        return _with_column(batch, feature, _keep_integers(series.to_numpy(), moved, rng))
    return _with_column(batch, feature, moved)


def time_travel(
    batch: pd.DataFrame,
    feature: str,
    severity: float,
    rng: np.random.Generator,
    *,
    group_by: str | None = None,
) -> pd.DataFrame:
    """Give every row the feature value of the row ``severity`` steps later.

    Rows must be in time order (within each ``group_by`` group). This is what a join on the wrong
    timestamp does: a feature built from past information now contains information from the
    future. The values themselves are unchanged, only their alignment in time, so the marginal
    distribution of the feature stays almost the same. The last ``severity`` rows of each group have
    no future row and keep their own value.
    """
    series = _column(batch, feature)
    if not (isinstance(severity, int | float) and math.isfinite(severity) and severity >= 0):
        raise ValueError(f"severity must be a non-negative number of steps, got {severity!r}")
    if float(severity) != int(severity):
        raise ValueError(f"severity must be a whole number of steps, got {severity!r}")
    lead = int(severity)
    source = np.arange(len(batch))
    if group_by is None:
        groups = [source.copy()]
    else:
        _column(batch, group_by)
        grouped = batch.groupby(group_by, sort=False, dropna=False).indices
        groups = [np.sort(positions) for positions in grouped.values()]
    for positions in groups:
        if 0 < lead < positions.size:
            source[positions[:-lead]] = positions[lead:]
    return _with_column(batch, feature, series.to_numpy(copy=True)[source])


def tiny_noise(
    batch: pd.DataFrame, feature: str, severity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Negative control: resample a tiny fraction of rows from the feature's own values.

    Float features also get a jitter of 0.1 batch standard deviations, clipped to the range of the
    batch: the control must not invent values outside the feature's domain (a negative value for
    a feature that is never negative would be a real anomaly, not noise). A detector that alarms
    on this is too sensitive to be useful.
    """
    series = _column(batch, feature)
    fraction = _fraction(severity)
    numeric = _is_numeric(series)
    values = series.to_numpy(dtype=None if numeric else object, copy=True)
    rows = _pick(np.ones(len(series), dtype=bool), fraction, rng)
    # One draw per row, so that the values are nested in severity like the rows are.
    new = values[rng.integers(0, len(values), size=len(values))]
    if is_float_dtype(series.dtype):
        std = float(np.nanstd(values)) if not np.all(np.isnan(values)) else 0.0
        new = new + rng.normal(0.0, 0.1 * std, size=len(values))
        if not np.all(np.isnan(values)):
            new = np.clip(new, np.nanmin(values), np.nanmax(values))
    return _with_column(batch, feature, _assign(values, rows, new[rows]))


# --------------------------------------------------------------------------- faults on whole rows


def duplicate_rows(batch: pd.DataFrame, severity: float, rng: np.random.Generator) -> pd.DataFrame:
    """Overwrite a fraction of the rows with copies of other rows; the index and shape are kept.

    Every overwritten row is a copy of a *different* row of the input batch.
    """
    if not isinstance(batch, pd.DataFrame):
        raise ValueError(f"batch must be a pandas DataFrame, got {type(batch).__name__}")
    if len(batch) < 2:
        raise ValueError("duplicate_rows needs at least two rows")
    fraction = _fraction(severity)
    n_rows = len(batch)
    targets = _pick(np.ones(n_rows, dtype=bool), fraction, rng)
    # One donor per row, never the row itself, so that the copies are nested in severity.
    donors = (np.arange(n_rows) + rng.integers(1, n_rows, size=n_rows)) % n_rows
    source = np.arange(n_rows)
    source[targets] = donors[targets]
    result = batch.iloc[source].copy(deep=True)
    result.index = batch.index.copy()
    return result


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Fault:
    """A fault and the facts the benchmark needs to know about it."""

    name: str
    function: Callable[..., pd.DataFrame]
    target: Target
    severity_unit: SeverityUnit
    preserves_shape: bool = True
    needs_time_order: bool = False
    negative_control: bool = False

    def apply(
        self,
        batch: pd.DataFrame,
        feature: str | None,
        severity: float,
        rng: np.random.Generator,
        **parameters: Any,
    ) -> pd.DataFrame:
        """Inject this fault; ``feature`` is ignored by faults that rewrite whole rows."""
        if self.target == "rows":
            return self.function(batch, severity, rng, **parameters)
        if feature is None:
            raise ValueError(f"{self.name} needs a feature")
        return self.function(batch, feature, severity, rng, **parameters)


_FAULTS = (
    Fault("pinned_value", pinned_value, "any", "fraction_of_rows"),
    Fault("dropped_feature", dropped_feature, "any", "binary", preserves_shape=False),
    Fault("type_change", type_change, "numeric", "fraction_of_rows"),
    Fault("new_category", new_category, "categorical", "fraction_of_rows"),
    Fault("out_of_domain_numeric", out_of_domain_numeric, "numeric", "fraction_of_rows"),
    Fault("missing_rate_increase", missing_rate_increase, "any", "fraction_of_rows"),
    Fault("unit_change", unit_change, "numeric", "fraction_of_rows"),
    Fault("mean_shift", mean_shift, "numeric", "sigma"),
    Fault("variance_change", variance_change, "numeric", "relative_scale"),
    Fault("duplicate_rows", duplicate_rows, "rows", "fraction_of_rows"),
    Fault("time_travel", time_travel, "any", "lead_steps", needs_time_order=True),
    Fault("tiny_noise", tiny_noise, "any", "fraction_of_rows", negative_control=True),
)

#: Every fault of the catalogue (SPEC section 5), by name.
FAULTS: dict[str, Fault] = {fault.name: fault for fault in _FAULTS}
