"""Inject one fault into a synthetic batch and show which rule notices it (SPEC section 17).

This is the engine behind ``POST /api/v1/demo``. It uses the benchmark's pieces (synthetic data,
fault registry, detectors, schema rule) and follows its leakage rules: the reference, calibration
and evaluation parts of the table are disjoint, statistics and the schema come from the reference
part, thresholds from clean batches of the calibration part, and the batch under test never
influences either. Nothing here touches user data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from mlguard.datasets import Dataset, make_synthetic
from mlguard.detectors import (
    CALIBRATED_DETECTORS,
    TEXTBOOK_RULES,
    FeatureReference,
    encode_column,
    fit_reference,
    score_encoded,
    score_feature,
    textbook_alarm,
)
from mlguard.inject import FAULTS
from mlguard.schema import Schema, infer_schema
from mlguard.validate import breaks_schema, feature_flags, validate_batch

BATCH_SIZES = (500, 1_000, 5_000)
N_ROWS = 60_000
N_CALIBRATION_BATCHES = 200
TARGET_FPR = 0.01
MAX_SEED = 2**31 - 1

#: Faults the demo offers. ``time_travel`` needs time-ordered data, which the demo does not have.
DEMO_FAULTS = tuple(name for name, fault in FAULTS.items() if not fault.needs_time_order)

#: Allowed severity per unit, inclusive.
SEVERITY_RANGES: dict[str, tuple[float, float]] = {
    "fraction_of_rows": (0.0001, 1.0),
    "sigma": (0.0, 5.0),
    "relative_scale": (0.0, 5.0),
    "binary": (1.0, 1.0),
}

_REFERENCE, _CALIBRATION = 0.4, 0.3  # the rest is the evaluation part


@dataclass(frozen=True)
class _Setup:
    """Everything that depends only on the batch size and the seed, never on a tested batch."""

    dataset: Dataset
    evaluation_rows: np.ndarray
    references: dict[str, FeatureReference]
    thresholds: dict[str, dict[str, float]]
    schema: Schema


def _parts(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng([seed, 7]).permutation(n_rows)
    first, second = int(n_rows * _REFERENCE), int(n_rows * (_REFERENCE + _CALIBRATION))
    return order[:first], order[first:second], order[second:]


def _table(seed: int) -> Dataset:
    return make_synthetic(N_ROWS, seed)


@lru_cache(maxsize=8)
def _setup(batch_size: int, seed: int) -> _Setup:
    dataset = _table(seed)
    reference_rows, calibration_rows, evaluation_rows = _parts(N_ROWS, seed)
    reference = dataset.frame.iloc[reference_rows]
    references = {
        name: fit_reference(name, reference[name].to_numpy(), dataset.kind(name))  # type: ignore[arg-type]
        for name in dataset.features
    }
    schema = infer_schema(reference[list(dataset.features)])
    rng = np.random.default_rng([seed, 11, batch_size])
    thresholds: dict[str, dict[str, float]] = {}
    for name, fitted in references.items():
        encoded = encode_column(fitted, dataset.frame[name].to_numpy())
        detectors = CALIBRATED_DETECTORS[fitted.kind]
        scores: dict[str, list[float]] = {detector: [] for detector in detectors}
        for _ in range(N_CALIBRATION_BATCHES):
            rows = rng.choice(calibration_rows, size=batch_size, replace=False)
            batch_scores = score_encoded(fitted, encoded[rows])
            for detector in detectors:
                scores[detector].append(batch_scores[detector])
        thresholds[name] = {
            detector: float(np.nanquantile(values, 1.0 - TARGET_FPR, method="higher"))
            for detector, values in scores.items()
        }
    return _Setup(dataset, evaluation_rows, references, thresholds, schema)


def _check(fault: str, severity: float, batch_size: int, seed: int) -> None:
    if fault not in DEMO_FAULTS:
        raise ValueError(f"unknown fault {fault!r}; choose one of {', '.join(DEMO_FAULTS)}")
    low, high = SEVERITY_RANGES[FAULTS[fault].severity_unit]
    if not (np.isfinite(severity) and low <= severity <= high):
        raise ValueError(f"severity of {fault} must be between {low:g} and {high:g}")
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"batch size must be one of {', '.join(map(str, BATCH_SIZES))}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")


def candidate_features(fault: str) -> tuple[str, ...]:
    """Features of the demo table that a fault can act on (none for faults on whole rows)."""
    target = FAULTS[fault].target
    dataset = _setup(BATCH_SIZES[0], 0).dataset
    if target == "rows":
        return ()
    if target == "numeric":
        return dataset.numeric
    if target == "categorical":
        return dataset.categorical
    return dataset.features


def _rules(name: str, scores: dict[str, float] | None, setup: _Setup) -> dict[str, dict]:
    """Every rule's verdict on one feature; ``None`` scores mean the column is gone."""
    kind = setup.references[name].kind
    verdicts: dict[str, dict] = {}
    for detector in CALIBRATED_DETECTORS["numeric"]:
        if detector not in CALIBRATED_DETECTORS[kind]:
            verdicts[f"calibrated:{detector}"] = {"applies": False}
            continue
        if scores is None:
            verdicts[f"calibrated:{detector}"] = {"applies": True, "alarm": None}
            continue
        score, limit = scores[detector], setup.thresholds[name][detector]
        verdicts[f"calibrated:{detector}"] = {
            "applies": True,
            "alarm": bool(np.isfinite(score) and score > limit),
            "score": _finite(score),
            "limit": limit,
        }
    for rule, (key, _, limit) in TEXTBOOK_RULES.items():
        if scores is None:
            verdicts[f"textbook:{rule}"] = {"applies": True, "alarm": None}
            continue
        alarm = textbook_alarm(rule, scores)
        verdicts[f"textbook:{rule}"] = (
            {"applies": False}
            if alarm is None
            else {"applies": True, "alarm": alarm, "score": _finite(scores[key]), "limit": limit}
        )
    return verdicts


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _verdicts(batch: pd.DataFrame, setup: _Setup) -> dict[str, dict[str, dict]]:
    out = {}
    for name in setup.dataset.features:
        scores = (
            score_feature(setup.references[name], batch[name].to_numpy())
            if name in batch.columns
            else None
        )
        verdicts = _rules(name, scores, setup)
        feature = setup.schema[name]
        verdicts["schema"] = {
            "applies": True,
            "alarm": True
            if name not in batch.columns
            else breaks_schema(feature, feature_flags(feature, batch[name].to_numpy())),
        }
        out[name] = verdicts
    return out


def run_demo(
    fault: str,
    severity: float,
    *,
    batch_size: int = 1_000,
    seed: int = 0,
    feature: str | None = None,
) -> dict[str, Any]:
    """Score one clean batch and the same batch with the fault, with every rule of the benchmark.

    The answer holds rule verdicts, scores, limits and anomaly types only: no row of the table.
    """
    _check(fault, severity, batch_size, seed)
    spec = FAULTS[fault]
    candidates = candidate_features(fault)
    if spec.target == "rows":
        if feature is not None:
            raise ValueError(f"{fault} rewrites whole rows; do not name a feature")
    elif feature is None:
        feature = candidates[int(np.random.default_rng([seed, 13]).integers(len(candidates)))]
    elif feature not in candidates:
        raise ValueError(
            f"{fault} cannot act on {feature!r}; choose one of {', '.join(candidates)}"
        )
    setup = _setup(batch_size, seed)
    rng = np.random.default_rng([seed, 17, batch_size])
    rows = rng.choice(setup.evaluation_rows, size=batch_size, replace=False)
    clean = setup.dataset.frame.iloc[rows][list(setup.dataset.features)].reset_index(drop=True)
    faulty = spec.apply(clean, feature, severity, np.random.default_rng([seed, 19]))
    anomalies = {
        label: sorted({anomaly.type.value for anomaly in validate_batch(batch, setup.schema)})
        for label, batch in (("clean", clean), ("faulty", faulty))
    }
    return {
        "fault": fault,
        "severity": severity,
        "severity_unit": spec.severity_unit,
        "negative_control": spec.negative_control,
        "feature": feature,
        "batch_size": batch_size,
        "seed": seed,
        "target_fpr": TARGET_FPR,
        "n_calibration_batches": N_CALIBRATION_BATCHES,
        "features": list(setup.dataset.features),
        "clean": _verdicts(clean, setup),
        "faulty": _verdicts(faulty, setup),
        "schema_anomaly_types": anomalies,
    }
