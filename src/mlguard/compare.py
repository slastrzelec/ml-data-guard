"""Comparison of two tables: distribution shift per feature, and feature skew row by row.

``compare_batches`` asks "does this batch look like the reference?" with one of the detectors of
the benchmark. ``feature_skew`` asks "do the same rows carry the same feature values in training
and in serving?", which is the only check here that can see a feature computed from the wrong
point in time, and the only one that needs both versions of the same rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlguard.detectors import (
    CALIBRATED_DETECTORS,
    FeatureReference,
    encode_column,
    fit_reference,
    score_encoded,
    score_feature,
)
from mlguard.schema import Schema, is_missing
from mlguard.validate import Anomaly, AnomalyType


def _columns(frame: pd.DataFrame, label: str) -> dict[str, object]:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise ValueError(f"{label} must be a non-empty pandas DataFrame")
    return {str(column): column for column in frame.columns}


def _shared_features(schema: Schema, *frames: dict[str, object]) -> list[str]:
    """Active features present in every table (a missing column is the validator's finding)."""
    return [
        feature.name
        for feature in schema.active
        if all(feature.name in columns for columns in frames)
    ]


def _fit(reference: pd.DataFrame, schema: Schema, names: list[str]) -> dict[str, FeatureReference]:
    columns = _columns(reference, "reference")
    return {
        name: fit_reference(name, reference[columns[name]].to_numpy(), schema[name].kind)
        for name in names
    }


def _check_detector(detector: str, schema: Schema, names: list[str]) -> None:
    if detector not in CALIBRATED_DETECTORS["numeric"]:
        raise ValueError(
            f"unknown detector {detector!r}; choose from {CALIBRATED_DETECTORS['numeric']}"
        )
    unsupported = [
        name for name in names if detector not in CALIBRATED_DETECTORS[schema[name].kind]
    ]
    if unsupported:
        raise ValueError(
            f"detector {detector!r} does not apply to categorical features {unsupported}"
        )


def calibrate_thresholds(
    reference: pd.DataFrame,
    schema: Schema,
    *,
    batch_size: int,
    detector: str = "linf",
    target_fpr: float = 0.01,
    n_batches: int = 500,
    seed: int = 0,
) -> tuple[dict[str, FeatureReference], dict[str, float]]:
    """Reference statistics and thresholds from clean reference data alone.

    The reference rows are split in two halves with ``seed``. Statistics are fitted on the first
    half. The threshold of a feature is the ``1 - target_fpr`` quantile of the scores of
    ``n_batches`` clean batches of ``batch_size`` rows drawn from the second half. Nothing but the
    reference is used, so the batch that is later compared cannot influence its own threshold.
    """
    names = _shared_features(schema, _columns(reference, "reference"))
    _check_detector(detector, schema, names)
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    if n_batches * target_fpr < 1:
        raise ValueError("n_batches * target_fpr must be at least 1 to estimate the threshold")
    rng = np.random.default_rng([seed, 31])
    order = rng.permutation(len(reference))
    first, second = order[: len(order) // 2], order[len(order) // 2 :]
    if batch_size < 1 or batch_size > second.size:
        raise ValueError(
            f"batch size {batch_size} is not possible: calibration draws batches from half of the "
            f"reference ({second.size} rows)"
        )
    references = _fit(reference.iloc[first], schema, names)
    columns = _columns(reference, "reference")
    encoded = {
        name: encode_column(references[name], reference[columns[name]].to_numpy()) for name in names
    }
    scores: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(n_batches):
        rows = rng.choice(second, size=batch_size, replace=False)
        for name in names:
            scores[name].append(score_encoded(references[name], encoded[name][rows])[detector])
    thresholds = {
        name: float(np.nanquantile(np.array(values), 1.0 - target_fpr, method="higher"))
        for name, values in scores.items()
    }
    return references, thresholds


def compare_batches(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    schema: Schema,
    *,
    detector: str = "linf",
    threshold: float | None = None,
    target_fpr: float = 0.01,
    n_batches: int = 500,
    seed: int = 0,
    calibration: tuple[dict[str, FeatureReference], dict[str, float]] | None = None,
) -> list[Anomaly]:
    """``DISTRIBUTION_SHIFT`` anomalies of ``batch`` with respect to ``reference``.

    With ``threshold`` the same fixed limit is used for every feature and the whole reference is
    used for the statistics. Without it the limit is calibrated per feature (see
    ``calibrate_thresholds``) for batches of the size of ``batch``; this is the default because a
    fixed limit is only right for one batch size. To check many batches of one size, calibrate
    once and pass the result as ``calibration``.
    """
    batch_columns = _columns(batch, "batch")
    names = _shared_features(schema, _columns(reference, "reference"), batch_columns)
    _check_detector(detector, schema, names)
    if calibration is not None:
        references, thresholds = calibration
        how = f"calibrated at a {target_fpr:.1%} false-alarm target"
    elif threshold is None:
        references, thresholds = calibrate_thresholds(
            reference,
            schema,
            batch_size=len(batch),
            detector=detector,
            target_fpr=target_fpr,
            n_batches=n_batches,
            seed=seed,
        )
        how = f"calibrated for {len(batch)} rows at a {target_fpr:.1%} false-alarm target"
    else:
        if not (np.isfinite(threshold) and threshold >= 0):
            raise ValueError("threshold must be a non-negative number")
        references = _fit(reference, schema, names)
        thresholds = dict.fromkeys(names, float(threshold))
        how = "fixed"
    anomalies = []
    for name in names:
        if name not in references:
            continue
        score = score_feature(references[name], batch[batch_columns[name]].to_numpy())[detector]
        if score > thresholds[name]:
            anomalies.append(
                Anomaly(
                    AnomalyType.DISTRIBUTION_SHIFT,
                    name,
                    "error",
                    f"{detector} distance to the reference is {score:.4g}, "
                    f"above the limit {thresholds[name]:.4g} ({how})",
                    statistic=float(score),
                    threshold=thresholds[name],
                    suggestion="check the upstream source of this feature; if the change is "
                    "intended, refresh the reference data",
                )
            )
    return anomalies


@dataclass(frozen=True)
class SkewReport:
    """Result of a row-by-row comparison of training and serving features."""

    n_joined: int
    n_training_only: int
    n_serving_only: int
    #: feature -> number of joined rows whose values differ
    mismatches: dict[str, int]

    def share(self, feature: str) -> float:
        return self.mismatches[feature] / self.n_joined if self.n_joined else 0.0


def _different(left: np.ndarray, right: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    """Element-wise "differs": floats with a tolerance, everything else exactly and type-strict."""
    result = np.empty(left.size, dtype=bool)
    for position, (a, b) in enumerate(zip(left, right, strict=True)):
        missing_a, missing_b = is_missing(a), is_missing(b)
        if missing_a or missing_b:
            result[position] = missing_a != missing_b
            continue
        number_a = isinstance(a, int | float | np.number) and not isinstance(a, bool | np.bool_)
        number_b = isinstance(b, int | float | np.number) and not isinstance(b, bool | np.bool_)
        if number_a and number_b:
            result[position] = not np.isclose(float(a), float(b), rtol=rtol, atol=atol)
        else:
            result[position] = type(a) is not type(b) or a != b
    return result


def feature_skew(
    training: pd.DataFrame,
    serving: pd.DataFrame,
    *,
    key: str,
    features: list[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 0.0,
) -> SkewReport:
    """Join training and serving rows on ``key`` and count rows whose feature values differ.

    The key must be unique in both tables. ``features`` defaults to the columns the tables share.
    Two missing values are equal; a number never equals its text form.
    """
    training_columns, serving_columns = _columns(training, "training"), _columns(serving, "serving")
    for label, columns, frame in (
        ("training", training_columns, training),
        ("serving", serving_columns, serving),
    ):
        if key not in columns:
            raise ValueError(f"{label} data has no key column {key!r}")
        keys = frame[columns[key]]
        if keys.isna().any() or keys.duplicated().any():
            raise ValueError(f"key {key!r} must be present and unique in the {label} data")
    shared = [name for name in training_columns if name in serving_columns and name != key]
    if features is None:
        features = shared
    unknown = [name for name in features if name not in shared]
    if unknown:
        raise ValueError(f"features not present in both tables: {unknown}")
    left = training.set_index(training_columns[key])
    right = serving.set_index(serving_columns[key])
    joined = left.index.intersection(right.index)
    mismatches = {}
    for name in features:
        a = left.loc[joined, training_columns[name]].to_numpy(dtype=object)
        b = right.loc[joined, serving_columns[name]].to_numpy(dtype=object)
        mismatches[name] = int(_different(a, b, rtol, atol).sum())
    return SkewReport(
        n_joined=len(joined),
        n_training_only=len(left.index.difference(right.index)),
        n_serving_only=len(right.index.difference(left.index)),
        mismatches=mismatches,
    )


def skew_anomalies(report: SkewReport, *, max_share: float = 0.0) -> list[Anomaly]:
    """``FEATURE_SKEW`` anomalies for features whose mismatch share exceeds ``max_share``."""
    if not 0 <= max_share <= 1:
        raise ValueError("max_share must be between 0 and 1")
    anomalies = []
    if report.n_joined == 0:
        return [
            Anomaly(
                AnomalyType.FEATURE_SKEW,
                None,
                "error",
                "no key occurs in both tables, so nothing could be compared",
                suggestion="check that both tables use the same key",
            )
        ]
    for name, count in report.mismatches.items():
        share = count / report.n_joined
        if share > max_share:
            anomalies.append(
                Anomaly(
                    AnomalyType.FEATURE_SKEW,
                    name,
                    "error",
                    f"{count} of {report.n_joined} joined rows ({share:.2%}) have a different "
                    "value in training and in serving",
                    statistic=share,
                    threshold=max_share,
                    suggestion="the two pipelines compute this feature differently; compare "
                    "their code and the point in time each one reads",
                )
            )
    return anomalies
