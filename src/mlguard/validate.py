"""Single-batch validation: compare one batch with a schema and explain every violation.

An anomaly is a finding, not a verdict: it says what was observed, which expectation of the schema
it broke, and which change of the schema would accept it. Whether the data or the schema is wrong
is for a person to decide. Cell values are not printed unless the caller asks for them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd

from mlguard.schema import FeatureSchema, Schema, is_missing, plain_token, value_type

_EPSILON = 1e-12


class AnomalyType(StrEnum):
    EMPTY_BATCH = "EMPTY_BATCH"
    MISSING_FEATURE = "MISSING_FEATURE"
    UNEXPECTED_FEATURE = "UNEXPECTED_FEATURE"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    LOW_PRESENCE = "LOW_PRESENCE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    UNEXPECTED_VALUES = "UNEXPECTED_VALUES"
    DISTRIBUTION_SHIFT = "DISTRIBUTION_SHIFT"
    FEATURE_SKEW = "FEATURE_SKEW"


@dataclass(frozen=True)
class Anomaly:
    """One violated expectation."""

    type: AnomalyType
    feature: str | None
    level: Literal["error", "warning"]
    description: str
    statistic: float | None = None
    threshold: float | None = None
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        entry = asdict(self)
        entry["type"] = self.type.value
        return entry


def has_errors(anomalies: list[Anomaly], *, strict: bool = False) -> bool:
    """Whether a list of anomalies should fail a pipeline (warnings count only when strict)."""
    return any(strict or anomaly.level == "error" for anomaly in anomalies)


def _share(count: int, total: int) -> str:
    return f"{count} of {total} rows ({count / total:.2%})"


def _expected_type(feature: FeatureSchema, value: object) -> bool:
    kind = value_type(value)
    if feature.dtype == "float":
        return kind in ("int", "float")
    return kind == feature.dtype


def _check_feature(
    feature: FeatureSchema, values: np.ndarray, *, show_values: bool
) -> list[Anomaly]:
    anomalies = []
    n_rows = len(values)
    present = np.array([not is_missing(value) for value in values], dtype=bool)
    typed = np.array(
        [
            bool(here) and _expected_type(feature, value)
            for value, here in zip(values, present, strict=True)
        ],
        dtype=bool,
    )
    if feature.kind == "numeric" and feature.dtype == "int":
        # a float column is fine for an integer feature as long as every number is whole
        for position in np.flatnonzero(present & ~typed):
            value = values[position]
            if value_type(value) == "float" and float(value).is_integer():
                typed[position] = True

    mismatched = int((present & ~typed).sum())
    if mismatched:
        found = sorted({value_type(value) for value in values[present & ~typed]})
        anomalies.append(
            Anomaly(
                AnomalyType.TYPE_MISMATCH,
                feature.name,
                "error",
                f"{_share(mismatched, n_rows)} hold {', '.join(found)} values, "
                f"expected {feature.dtype}",
                statistic=mismatched / n_rows,
                threshold=0.0,
                suggestion=f"fix the upstream type, or change dtype of {feature.name!r}",
            )
        )

    presence = float(present.mean())
    if presence < feature.min_presence - _EPSILON:
        anomalies.append(
            Anomaly(
                AnomalyType.LOW_PRESENCE,
                feature.name,
                "error",
                f"{_share(n_rows - int(present.sum()), n_rows)} are missing; "
                f"at least {feature.min_presence:.1%} of the rows must have a value",
                statistic=presence,
                threshold=feature.min_presence,
                suggestion=f"lower min_presence of {feature.name!r} to "
                f"{np.floor(presence * 1000) / 1000:g}",
            )
        )

    checked = values[typed]
    if feature.kind == "numeric" and checked.size:
        numbers = checked.astype(np.float64)
        low = -np.inf if feature.minimum is None else feature.minimum
        high = np.inf if feature.maximum is None else feature.maximum
        outside = int(((numbers < low) | (numbers > high)).sum())
        if outside / n_rows > feature.max_out_of_range_fraction + _EPSILON:
            observed = f"[{numbers.min():g}, {numbers.max():g}]"
            anomalies.append(
                Anomaly(
                    AnomalyType.OUT_OF_RANGE,
                    feature.name,
                    "error",
                    f"{_share(outside, n_rows)} are outside [{low:g}, {high:g}]; "
                    f"observed range {observed}",
                    statistic=outside / n_rows,
                    threshold=feature.max_out_of_range_fraction,
                    suggestion=f"widen the range of {feature.name!r} to {observed}, or raise "
                    f"max_out_of_range_fraction to {outside / n_rows:.4g}",
                )
            )
    if feature.kind == "categorical" and feature.domain is not None and checked.size:
        unexpected = [value for value in checked if not feature.allows(value)]
        if len(unexpected) / n_rows > feature.max_unexpected_fraction + _EPSILON:
            distinct = sorted({plain_token(value) for value in unexpected}, key=repr)
            detail = f"{len(distinct)} distinct value(s) not in the domain"
            if show_values:
                detail += ": " + ", ".join(repr(value) for value in distinct[:5])
            anomalies.append(
                Anomaly(
                    AnomalyType.UNEXPECTED_VALUES,
                    feature.name,
                    "error",
                    f"{_share(len(unexpected), n_rows)} hold {detail}",
                    statistic=len(unexpected) / n_rows,
                    threshold=feature.max_unexpected_fraction,
                    suggestion=f"add the new value(s) to the domain of {feature.name!r}, or raise "
                    f"max_unexpected_fraction to {len(unexpected) / n_rows:.4g}",
                )
            )
    return anomalies


def validate_batch(
    batch: pd.DataFrame, schema: Schema, *, show_values: bool = False
) -> list[Anomaly]:
    """All anomalies of a batch with respect to a schema; an empty list means it conforms.

    ``show_values`` adds up to five unexpected categorical values to the description. It is off by
    default because category values can be personal data.
    """
    if not isinstance(batch, pd.DataFrame):
        raise ValueError(f"batch must be a pandas DataFrame, got {type(batch).__name__}")
    if len(batch) == 0:
        return [Anomaly(AnomalyType.EMPTY_BATCH, None, "error", "the batch has no rows")]
    columns = {str(column): column for column in batch.columns}
    anomalies: list[Anomaly] = []
    for feature in schema.active:
        if feature.name not in columns:
            anomalies.append(
                Anomaly(
                    AnomalyType.MISSING_FEATURE,
                    feature.name,
                    "error",
                    "the column is missing from the batch",
                    suggestion=f"restore the column, or mark {feature.name!r} as deprecated",
                )
            )
            continue
        values = batch[columns[feature.name]].to_numpy(dtype=object)
        anomalies.extend(_check_feature(feature, values, show_values=show_values))
    for name in columns:
        if name not in schema.names and name not in schema.ignored:
            anomalies.append(
                Anomaly(
                    AnomalyType.UNEXPECTED_FEATURE,
                    name,
                    "warning",
                    "the column is not described by the schema",
                    suggestion=f"add {name!r} to the schema, or stop sending it",
                )
            )
    return anomalies
