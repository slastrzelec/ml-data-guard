"""Benchmark runner: which detector finds which injected fault, and at what false-alarm rate.

Leakage rules (SPEC section 3) as code structure:

* ``make_splits`` produces four disjoint row sets; every later step receives only the split it is
  allowed to see;
* reference statistics are fitted on the reference split, thresholds on clean batches of the
  calibration split, faults are injected into batches of the evaluation split only;
* the downstream model is trained on the reference split and scored on the held-out split, which
  no detector and no threshold ever sees;
* every random draw comes from a generator keyed by (seed, purpose, indices), so the result does
  not depend on the order of execution.

Run: ``python -m mlguard.bench --config benchmarks/configs/synthetic.json --out results.json
--report report.md``
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

import mlguard
from mlguard.datasets import (
    Dataset,
    load_beijing,
    load_covertype,
    load_news,
    make_synthetic,
    make_synthetic_series,
)
from mlguard.detectors import (
    CALIBRATED_DETECTORS,
    TEXTBOOK_RULES,
    FeatureReference,
    category_codes,
    fit_reference,
    score_counts,
    score_feature,
    strict_numeric,
    textbook_alarm,
)
from mlguard.distances import bin_counts
from mlguard.inject import FAULTS
from mlguard.schema import infer_schema
from mlguard.validate import FeatureFlags, breaks_schema, feature_flags

_FRACTIONS = (0.001, 0.005, 0.01, 0.05, 0.2)
DEFAULT_SEVERITIES: dict[str, tuple[float, ...]] = {
    "pinned_value": _FRACTIONS,
    "dropped_feature": (1,),
    "type_change": _FRACTIONS,
    "new_category": _FRACTIONS,
    "out_of_domain_numeric": _FRACTIONS,
    "missing_rate_increase": _FRACTIONS,
    "unit_change": _FRACTIONS,
    "mean_shift": (0.01, 0.05, 0.1, 0.25, 0.5),
    "variance_change": (0.02, 0.05, 0.1, 0.25, 0.5),
    "duplicate_rows": _FRACTIONS,
    "time_travel": (1, 2, 3, 5, 10),
    "tiny_noise": (0.0001,),
}

# Purposes of the random streams; part of every generator key.
_SPLIT, _CALIBRATION, _CLEAN, _FAULT_BATCH, _FAULT_FEATURE, _FAULT_INJECT = 1, 2, 3, 4, 5, 6
_MODEL_BATCH, _BOOTSTRAP = 7, 8

_Z95 = 1.959963984540054
_DETECTORS = CALIBRATED_DETECTORS["numeric"]  # superset of the categorical ones
#: Rules that need no calibration: the fixed textbook thresholds and the inferred schema.
_SCHEMA_RULE = "schema"
_RULES = (*TEXTBOOK_RULES, _SCHEMA_RULE)


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class BenchConfig:
    """Everything that determines a benchmark result."""

    dataset: str
    seed: int = 0
    dataset_rows: int | None = None
    data_path: str | None = None
    batch_sizes: tuple[int, ...] = (1_000, 10_000)
    split: tuple[float, float, float, float] = (0.40, 0.25, 0.25, 0.10)
    target_fpr: float = 0.01
    n_calibration_batches: int = 1_000
    n_clean_batches: int = 1_000
    n_seeds: int = 100
    n_bins: int = 10
    max_categories: int = 100
    smoothing: float = 1e-4
    faults: tuple[str, ...] | None = None
    severities: dict[str, tuple[float, ...]] = field(default_factory=dict)
    model: bool = True
    model_batch_size: int = 5_000
    model_max_iter: int = 100
    #: "static": every batch is compared with the reference split. "rolling" (time-ordered data
    #: only): every batch is compared with the rows just before it.
    reference_mode: str = "static"
    #: Size of the rolling reference, in batches.
    rolling_reference_batches: int = 5

    def __post_init__(self) -> None:
        if len(self.split) != 4 or min(self.split) <= 0 or abs(sum(self.split) - 1) > 1e-9:
            raise ValueError("split must be four positive fractions that sum to 1")
        if not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1")
        if not self.batch_sizes or min(self.batch_sizes) < 1:
            raise ValueError("batch_sizes must be positive")
        if min(self.n_calibration_batches, self.n_clean_batches, self.n_seeds) < 1:
            raise ValueError("the numbers of batches and seeds must be positive")
        if self.n_calibration_batches * self.target_fpr < 1:
            raise ValueError(
                "too few calibration batches to estimate the threshold: "
                "n_calibration_batches * target_fpr must be at least 1"
            )
        if self.reference_mode not in ("static", "rolling"):
            raise ValueError("reference_mode must be 'static' or 'rolling'")
        if self.rolling_reference_batches < 1:
            raise ValueError("rolling_reference_batches must be at least 1")
        unknown = (set(self.faults or ()) | set(self.severities)) - set(FAULTS)
        if unknown:
            raise ValueError(f"unknown faults: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchConfig:
        known = {item.name for item in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        values = dict(raw)
        for key in ("batch_sizes", "split", "faults"):
            if values.get(key) is not None:
                values[key] = tuple(values[key])
        if "severities" in values:
            values["severities"] = {k: tuple(v) for k, v in values["severities"].items()}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    def severities_of(self, fault: str) -> tuple[float, ...]:
        return self.severities.get(fault, DEFAULT_SEVERITIES[fault])


def load_dataset(config: BenchConfig) -> Dataset:
    """The dataset named in the config; synthetic data is generated, public data read from disk."""
    if config.dataset in ("synthetic", "synthetic_series"):
        if config.dataset_rows is None:
            raise ValueError("dataset_rows is required for synthetic datasets")
        maker = make_synthetic if config.dataset == "synthetic" else make_synthetic_series
        return maker(config.dataset_rows, config.seed)
    loaders = {"covertype": load_covertype, "news": load_news, "beijing": load_beijing}
    if config.dataset in loaders:
        if config.data_path is None:
            raise ValueError(f"data_path is required for the {config.dataset} dataset")
        return loaders[config.dataset](config.data_path)
    raise ValueError(f"unknown dataset {config.dataset!r}")


# --------------------------------------------------------------------------- splits and batches


@dataclass(frozen=True)
class Splits:
    """Row positions of the four disjoint splits."""

    reference: np.ndarray
    calibration: np.ndarray
    evaluation: np.ndarray
    held_out: np.ndarray


def _rng(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng([seed, *key])


def make_splits(
    n_rows: int, fractions: tuple[float, float, float, float], *, seed: int, time_ordered: bool
) -> Splits:
    """Disjoint splits that cover every row once.

    Unordered data is shuffled with the seed first. Time-ordered data is cut into contiguous
    blocks in the order reference < calibration < evaluation < held-out.
    """
    order = np.arange(n_rows) if time_ordered else _rng(seed, _SPLIT).permutation(n_rows)
    cuts = [int(n_rows * sum(fractions[: index + 1])) for index in range(3)]
    parts = np.split(order, cuts)
    if min(part.size for part in parts) == 0:
        raise ValueError("the dataset is too small for these split fractions")
    return Splits(*parts)


def draw_batch(
    pool: np.ndarray, batch_size: int, rng: np.random.Generator, *, time_ordered: bool
) -> np.ndarray:
    """Row positions of one batch: a random subsample, or a contiguous window for ordered data."""
    if batch_size > pool.size:
        raise ValueError(f"batch size {batch_size} exceeds the split size {pool.size}")
    if time_ordered:
        start = int(rng.integers(0, pool.size - batch_size + 1))
        return pool[start : start + batch_size]
    return rng.choice(pool, size=batch_size, replace=False)


# --------------------------------------------------------------------------- statistics


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """95% Wilson score interval of a proportion."""
    if trials <= 0:
        return (math.nan, math.nan)
    rate = successes / trials
    denominator = 1 + _Z95**2 / trials
    centre = (rate + _Z95**2 / (2 * trials)) / denominator
    half = _Z95 * math.sqrt(rate * (1 - rate) / trials + _Z95**2 / (4 * trials**2)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == trials else min(1.0, centre + half)
    return (low, high)


def _number(value: float) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), 6)


def _proportion(successes: int, trials: int) -> dict[str, Any] | None:
    if trials == 0:
        return None
    low, high = wilson_interval(successes, trials)
    return {
        "k": int(successes),
        "n": int(trials),
        "rate": _number(successes / trials),
        "low": _number(low),
        "high": _number(high),
    }


def bootstrap_mean_interval(
    values: np.ndarray, rng: np.random.Generator, *, n_resamples: int = 2_000
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    samples = rng.integers(0, values.size, size=(n_resamples, values.size))
    low, high = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


# --------------------------------------------------------------------------- scoring


class _Scorer:
    """Reference statistics of every feature plus a fast path for clean batches."""

    def __init__(self, dataset: Dataset, reference_rows: np.ndarray, config: BenchConfig) -> None:
        self.dataset = dataset
        self.smoothing = config.smoothing
        self.references: dict[str, FeatureReference] = {}
        self._numbers: dict[str, np.ndarray] = {}
        self._codes: dict[str, np.ndarray] = {}
        for feature in dataset.features:
            column = dataset.frame[feature].to_numpy()
            reference = fit_reference(
                feature,
                column[reference_rows],
                dataset.kind(feature),  # type: ignore[arg-type]
                n_bins=config.n_bins,
                max_categories=config.max_categories,
            )
            self.references[feature] = reference
            # Encoding the clean column once is only a speed-up: it uses the frozen reference
            # (edges / vocabulary) and gives the same counts as scoring the raw values.
            if reference.kind == "numeric":
                self._numbers[feature] = strict_numeric(column)
            else:
                self._codes[feature] = category_codes(column, reference.vocabulary)

    def clean(self, feature: str, rows: np.ndarray) -> dict[str, float]:
        reference = self.references[feature]
        if reference.kind == "numeric":
            numbers = self._numbers[feature][rows]
            counts = bin_counts(numbers, reference.edges)
            present = numbers[~np.isnan(numbers)]
            return score_counts(reference, counts, present, smoothing=self.smoothing)
        counts = np.bincount(self._codes[feature][rows], minlength=reference.counts.size)
        return score_counts(reference, counts.astype(np.float64), smoothing=self.smoothing)

    def faulty(self, feature: str, values: np.ndarray, rows: np.ndarray) -> dict[str, float]:
        return score_feature(self.references[feature], values, smoothing=self.smoothing)


class _RollingScorer:
    """Scores a batch against the rows just before it instead of a fixed reference.

    On data with seasons a fixed reference describes the whole year, and one week never looks
    like the whole year. The rows before a batch are clean data from the same season. Statistics
    are fitted on that window only; the batch itself is never part of its own reference.
    """

    def __init__(self, dataset: Dataset, config: BenchConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._columns = {
            feature: (
                strict_numeric(dataset.frame[feature].to_numpy())
                if dataset.kind(feature) == "numeric"
                else dataset.frame[feature].to_numpy(dtype=object)
            )
            for feature in dataset.features
        }
        self._window: tuple[int, int] | None = None
        self._fitted: dict[str, FeatureReference | None] = {}

    def _reference(self, feature: str, rows: np.ndarray) -> FeatureReference | None:
        start, size = int(rows[0]), self.config.rolling_reference_batches * len(rows)
        if start - size < 0:
            raise ValueError("not enough rows before the batch for a rolling reference")
        if (start, size) != self._window:
            self._window, self._fitted = (start, size), {}
        if feature not in self._fitted:
            try:
                self._fitted[feature] = fit_reference(
                    feature,
                    self._columns[feature][start - size : start],
                    self.dataset.kind(feature),  # type: ignore[arg-type]
                    n_bins=self.config.n_bins,
                    max_categories=self.config.max_categories,
                )
            except ValueError:  # the window holds no value of this feature (sensor outage)
                self._fitted[feature] = None
        return self._fitted[feature]

    def clean(self, feature: str, rows: np.ndarray) -> dict[str, float]:
        return self.faulty(feature, self._columns[feature][rows], rows)

    def faulty(self, feature: str, values: np.ndarray, rows: np.ndarray) -> dict[str, float]:
        reference = self._reference(feature, rows)
        if reference is None:
            # nothing to compare with: no score, hence no alarm from any rule
            names = [*CALIBRATED_DETECTORS[self.dataset.kind(feature)], "chi2_pvalue", "ks_pvalue"]  # type: ignore[index]
            return dict.fromkeys(names, math.nan)
        return score_feature(reference, values, smoothing=self.config.smoothing)


class _SchemaCheck:
    """The schema validator as one more rule: "does this feature of this batch break the schema?"

    The schema is inferred from the reference split with the default heuristics and never edited,
    so its false alarms on clean batches are part of what is measured.
    """

    def __init__(self, dataset: Dataset, reference_rows: np.ndarray) -> None:
        reference = dataset.frame.iloc[reference_rows][list(dataset.features)]
        self.schema = infer_schema(reference)
        # Clean data is checked once per column; a clean batch is then a slice of the flags.
        self._flags = {
            feature.name: feature_flags(feature, dataset.frame[feature.name].to_numpy())
            for feature in self.schema.features
        }

    def clean(self, feature: str, rows: np.ndarray) -> bool:
        flags = self._flags[feature]
        sliced = FeatureFlags(flags.present[rows], flags.mismatched[rows], flags.violating[rows])
        return breaks_schema(self.schema[feature], sliced)

    def faulty(self, feature: str, batch: pd.DataFrame) -> bool:
        if feature not in batch.columns:
            return True  # a missing column is a schema error by definition
        return breaks_schema(
            self.schema[feature], feature_flags(self.schema[feature], batch[feature].to_numpy())
        )


def calibrate(
    scorer: _Scorer | _RollingScorer,
    calibration_rows: np.ndarray,
    batch_size: int,
    config: BenchConfig,
) -> dict[str, dict[str, float]]:
    """Thresholds per feature and detector from clean batches of the calibration split only."""
    dataset = scorer.dataset
    collected: dict[str, dict[str, list[float]]] = {
        feature: {name: [] for name in CALIBRATED_DETECTORS[dataset.kind(feature)]}  # type: ignore[index]
        for feature in dataset.features
    }
    for index in range(config.n_calibration_batches):
        rng = _rng(config.seed, _CALIBRATION, batch_size, index)
        rows = draw_batch(calibration_rows, batch_size, rng, time_ordered=dataset.time_ordered)
        for feature, per_detector in collected.items():
            scores = scorer.clean(feature, rows)
            for name, values in per_detector.items():
                values.append(scores[name])
    level = 1.0 - config.target_fpr
    return {
        feature: {
            name: float(np.nanquantile(np.array(values), level, method="higher"))
            for name, values in per_detector.items()
        }
        for feature, per_detector in collected.items()
    }


def _alarms(
    scores: dict[str, float], thresholds: dict[str, float]
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Alarms of one feature in one batch: calibrated arm and textbook arm."""
    calibrated = {name: bool(scores[name] > limit) for name, limit in thresholds.items()}
    textbook = {}
    for rule in TEXTBOOK_RULES:
        alarm = textbook_alarm(rule, scores)
        if alarm is not None:
            textbook[rule] = alarm
    return calibrated, textbook


class _Tally:
    """Counts alarms and opportunities per detector."""

    def __init__(self) -> None:
        self.hits: dict[str, int] = {}
        self.trials: dict[str, int] = {}

    def add(self, alarms: dict[str, bool]) -> None:
        for name, alarm in alarms.items():
            self.hits[name] = self.hits.get(name, 0) + int(alarm)
            self.trials[name] = self.trials.get(name, 0) + 1

    def summary(self, names: tuple[str, ...]) -> dict[str, Any]:
        return {
            name: _proportion(self.hits.get(name, 0), self.trials.get(name, 0)) for name in names
        }


def _any(per_feature: list[dict[str, bool]]) -> dict[str, bool]:
    """Batch-level alarm: a detector alarms when it alarms on any feature it applies to."""
    merged: dict[str, bool] = {}
    for alarms in per_feature:
        for name, alarm in alarms.items():
            merged[name] = merged.get(name, False) or alarm
    return merged


def evaluate_clean(
    scorer: _Scorer | _RollingScorer,
    schema_check: _SchemaCheck,
    evaluation_rows: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    batch_size: int,
    config: BenchConfig,
) -> dict[str, Any]:
    """False-positive rates on clean evaluation batches (the natural-variation control)."""
    dataset = scorer.dataset
    tallies = {key: _Tally() for key in ("cal_feature", "cal_batch", "text_feature", "text_batch")}
    for index in range(config.n_clean_batches):
        rng = _rng(config.seed, _CLEAN, batch_size, index)
        rows = draw_batch(evaluation_rows, batch_size, rng, time_ordered=dataset.time_ordered)
        calibrated_all, textbook_all = [], []
        for feature in dataset.features:
            calibrated, textbook = _alarms(scorer.clean(feature, rows), thresholds[feature])
            textbook[_SCHEMA_RULE] = schema_check.clean(feature, rows)
            tallies["cal_feature"].add(calibrated)
            tallies["text_feature"].add(textbook)
            calibrated_all.append(calibrated)
            textbook_all.append(textbook)
        tallies["cal_batch"].add(_any(calibrated_all))
        tallies["text_batch"].add(_any(textbook_all))
    rules = _RULES
    return {
        "n_batches": config.n_clean_batches,
        "calibrated": {
            "per_feature": tallies["cal_feature"].summary(_DETECTORS),
            "per_batch": tallies["cal_batch"].summary(_DETECTORS),
        },
        "textbook": {
            "per_feature": tallies["text_feature"].summary(rules),
            "per_batch": tallies["text_batch"].summary(rules),
        },
    }


# --------------------------------------------------------------------------- faults


def _fault_names(dataset: Dataset, config: BenchConfig) -> list[str]:
    names = list(config.faults) if config.faults is not None else list(FAULTS)
    return [name for name in names if dataset.time_ordered or not FAULTS[name].needs_time_order]


def _candidates(dataset: Dataset, fault: str) -> tuple[str, ...]:
    target = FAULTS[fault].target
    if target == "numeric":
        return dataset.numeric
    if target == "categorical":
        return dataset.categorical
    if FAULTS[fault].needs_time_order and dataset.group_by is not None:
        # shifting the series name within its own series would change nothing
        return tuple(name for name in dataset.features if name != dataset.group_by)
    return dataset.features


def _injectable(fault: str, values: np.ndarray) -> bool:
    """Whether a fault can change this column of this batch at all.

    A mean shift needs a spread, a unit change needs a non-zero value, several faults need at
    least one value. Without this check a dry week of an hourly ``rain`` feature would be labelled
    faulty although nothing could be injected.
    """
    if fault in ("mean_shift", "variance_change", "unit_change", "out_of_domain_numeric"):
        numbers = strict_numeric(values)
        present = numbers[~np.isnan(numbers)]
        if present.size == 0:
            return False
        if fault in ("mean_shift", "variance_change"):
            return bool(present.std() > 0)
        if fault == "unit_change":
            return bool(np.any(present != 0))
        return True
    if fault in ("type_change", "missing_rate_increase"):
        return any(value is not None and value == value for value in values)
    return True


def _choose_feature(
    dataset: Dataset, fault: str, config: BenchConfig, batch: pd.DataFrame, *key: int
) -> str | None:
    """The faulted feature of one run, drawn among the features the fault can act on.

    The draw depends on the seed and the run only, so detection and model impact use the same
    feature whenever it is injectable in both batches (always, unless a feature is constant or
    empty in one of them).
    """
    if FAULTS[fault].target == "rows":
        return None
    candidates = tuple(
        name for name in _candidates(dataset, fault) if _injectable(fault, batch[name].to_numpy())
    )
    if not candidates:
        raise ValueError(f"{fault} has no applicable feature in this batch of {dataset.name}")
    return candidates[int(_rng(config.seed, _FAULT_FEATURE, *key).integers(len(candidates)))]


def _batch(dataset: Dataset, rows: np.ndarray) -> pd.DataFrame:
    # The target travels with its row, so a fault that rewrites whole rows keeps each row's
    # label; faults on one feature never touch it and detectors never score it.
    return dataset.frame.iloc[rows][[*dataset.features, dataset.target]]


def _inject(
    dataset: Dataset, batch: pd.DataFrame, fault: str, feature: str | None, severity: float,
    config: BenchConfig, *key: int,
) -> pd.DataFrame:  # fmt: skip
    rng = _rng(config.seed, _FAULT_INJECT, *key)
    if FAULTS[fault].needs_time_order and dataset.group_by is not None:
        # several interleaved series: the shift in time happens within each series
        return FAULTS[fault].apply(batch, feature, severity, rng, group_by=dataset.group_by)
    return FAULTS[fault].apply(batch, feature, severity, rng)


def evaluate_faults(
    scorer: _Scorer | _RollingScorer,
    schema_check: _SchemaCheck,
    evaluation_rows: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    batch_size: int,
    config: BenchConfig,
) -> list[dict[str, Any]]:
    """Recall per fault and severity on batches of the evaluation split."""
    dataset = scorer.dataset
    results = []
    for fault_index, fault in enumerate(sorted(FAULTS)):
        if fault not in _fault_names(dataset, config):
            continue
        spec = FAULTS[fault]
        for severity_index, severity in enumerate(config.severities_of(fault)):
            calibrated_tally, textbook_tally = _Tally(), _Tally()
            by_feature: dict[str, _Tally] = {}
            for seed_index in range(config.n_seeds):
                key = (fault_index, severity_index, seed_index)
                rng = _rng(config.seed, _FAULT_BATCH, batch_size, *key)
                rows = draw_batch(
                    evaluation_rows, batch_size, rng, time_ordered=dataset.time_ordered
                )
                batch = _batch(dataset, rows)
                feature = _choose_feature(dataset, fault, config, batch, *key)
                faulty = _inject(dataset, batch, fault, feature, severity, config, *key)
                if not spec.preserves_shape:
                    # a missing column has no distribution to compare; only the schema sees it
                    textbook_tally.add({_SCHEMA_RULE: schema_check.faulty(str(feature), faulty)})
                    continue
                scored = dataset.features if feature is None else (feature,)
                calibrated_all, textbook_all = [], []
                for name in scored:
                    scores = scorer.faulty(name, faulty[name].to_numpy(), rows)
                    calibrated, textbook = _alarms(scores, thresholds[name])
                    textbook[_SCHEMA_RULE] = schema_check.faulty(name, faulty)
                    calibrated_all.append(calibrated)
                    textbook_all.append(textbook)
                calibrated_tally.add(_any(calibrated_all))
                textbook_tally.add(_any(textbook_all))
                if feature is not None:
                    by_feature.setdefault(feature, _Tally()).add(calibrated_all[0])
            results.append(
                {
                    "fault": fault,
                    "severity": severity,
                    "severity_unit": spec.severity_unit,
                    "negative_control": spec.negative_control,
                    "level": "batch" if spec.target == "rows" else "feature",
                    "scorable": spec.preserves_shape,
                    "n": config.n_seeds,
                    "calibrated": calibrated_tally.summary(_DETECTORS),
                    "textbook": textbook_tally.summary(_RULES),
                    # Alarms per faulted feature (calibrated arm): {feature: {detector: [k, n]}}.
                    "by_feature": {
                        name: {
                            detector: [tally.hits[detector], tally.trials[detector]]
                            for detector in sorted(tally.hits)
                        }
                        for name, tally in sorted(by_feature.items())
                    },
                }
            )
    return results


# --------------------------------------------------------------------------- downstream model


class _Model:
    """Gradient boosting trained on the reference split; used only to measure fault impact."""

    def __init__(self, scorer: _Scorer, reference_rows: np.ndarray, config: BenchConfig) -> None:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                'the model-impact part needs scikit-learn: pip install "ml-data-guard[bench]"'
            ) from error
        self.dataset = scorer.dataset
        self.references = scorer.references
        self.estimator = HistGradientBoostingClassifier(
            max_iter=config.model_max_iter, early_stopping=False, random_state=config.seed
        )
        reference = self.dataset.frame.iloc[reference_rows]
        self.estimator.fit(self.encode(reference), reference[self.dataset.target].to_numpy())

    def encode(self, batch: pd.DataFrame) -> np.ndarray:
        """Feature matrix; a missing column, text in a numeric column and unseen or missing
        categories all become NaN, which the model handles natively."""
        columns = []
        for feature in self.dataset.features:
            reference = self.references[feature]
            if feature not in batch.columns:
                columns.append(np.full(len(batch), np.nan))
            elif reference.kind == "numeric":
                columns.append(strict_numeric(batch[feature].to_numpy()))
            else:
                codes = category_codes(batch[feature].to_numpy(), reference.vocabulary)
                known = codes < len(reference.vocabulary)
                columns.append(np.where(known, codes, np.nan).astype(np.float64))
        return np.column_stack(columns)

    def correct(self, batch: pd.DataFrame) -> np.ndarray:
        """Per-row correctness against the labels carried by the batch itself."""
        labels = batch[self.dataset.target].to_numpy()
        return self.estimator.predict(self.encode(batch)) == labels


def evaluate_model_impact(scorer: _Scorer, splits: Splits, config: BenchConfig) -> dict[str, Any]:
    """Change of the model's accuracy when a fault is injected into held-out batches."""
    dataset = scorer.dataset
    model = _Model(scorer, splits.reference, config)
    batch_size = min(config.model_batch_size, splits.held_out.size)
    correct = np.zeros(len(dataset.frame), dtype=bool)
    held_out = dataset.frame.iloc[splits.held_out]
    correct[splits.held_out] = model.correct(held_out)
    results = []
    for fault_index, fault in enumerate(sorted(FAULTS)):
        if fault not in _fault_names(dataset, config):
            continue
        for severity_index, severity in enumerate(config.severities_of(fault)):
            drops = np.empty(config.n_seeds)
            for seed_index in range(config.n_seeds):
                key = (fault_index, severity_index, seed_index)
                rng = _rng(config.seed, _MODEL_BATCH, *key)
                rows = draw_batch(
                    splits.held_out, batch_size, rng, time_ordered=dataset.time_ordered
                )
                batch = _batch(dataset, rows)
                feature = _choose_feature(dataset, fault, config, batch, *key)
                faulty = _inject(dataset, batch, fault, feature, severity, config, *key)
                drops[seed_index] = correct[rows].mean() - model.correct(faulty).mean()
            low, high = bootstrap_mean_interval(
                drops, _rng(config.seed, _BOOTSTRAP, fault_index, severity_index)
            )
            results.append(
                {
                    "fault": fault,
                    "severity": severity,
                    "accuracy_drop": _number(drops.mean()),
                    "low": _number(low),
                    "high": _number(high),
                }
            )
    return {
        "metric": "accuracy",
        "batch_size": batch_size,
        "clean_accuracy": _number(correct[splits.held_out].mean()),
        "faults": results,
    }


# --------------------------------------------------------------------------- run and report


def run_benchmark(
    config: BenchConfig, *, dataset: Dataset | None = None, commit: str | None = None
) -> dict[str, Any]:
    """Run the whole benchmark and return a JSON-serialisable result."""
    dataset = dataset if dataset is not None else load_dataset(config)
    splits = make_splits(
        len(dataset.frame), config.split, seed=config.seed, time_ordered=dataset.time_ordered
    )
    static = _Scorer(dataset, splits.reference, config)
    schema_check = _SchemaCheck(dataset, splits.reference)
    scorer: _Scorer | _RollingScorer = static
    if config.reference_mode == "rolling":
        if not dataset.time_ordered:
            raise ValueError("a rolling reference needs time-ordered data")
        scorer = _RollingScorer(dataset, config)
    runs = []
    for batch_size in config.batch_sizes:
        print(f"[bench] {dataset.name}: batch size {batch_size}", file=sys.stderr, flush=True)
        thresholds = calibrate(scorer, splits.calibration, batch_size, config)
        runs.append(
            {
                "batch_size": batch_size,
                "thresholds": {
                    feature: {name: _number(value) for name, value in per_detector.items()}
                    for feature, per_detector in thresholds.items()
                },
                "clean": evaluate_clean(
                    scorer, schema_check, splits.evaluation, thresholds, batch_size, config
                ),
                "faults": evaluate_faults(
                    scorer, schema_check, splits.evaluation, thresholds, batch_size, config
                ),
            }
        )
    versions = {
        "python": platform.python_version(),
        "ml-data-guard": mlguard.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    model = None
    if config.model:
        print(f"[bench] {dataset.name}: model impact", file=sys.stderr, flush=True)
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
        model = evaluate_model_impact(static, splits, config)
    return {
        "meta": {
            "dataset": dataset.name,
            "source": dataset.source,
            "n_rows": len(dataset.frame),
            "features": {
                "numeric": list(dataset.numeric),
                "categorical": list(dataset.categorical),
            },
            "split_sizes": {
                "reference": int(splits.reference.size),
                "calibration": int(splits.calibration.size),
                "evaluation": int(splits.evaluation.size),
                "held_out": int(splits.held_out.size),
            },
            "config": config.to_dict(),
            "commit": commit,
            "versions": versions,
        },
        "runs": runs,
        "model": model,
    }


def to_json(result: dict[str, Any]) -> str:
    """Canonical text form of a result: stable key order, innermost objects and lists on one
    line (keeps the file small and the diffs readable), trailing newline."""
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    text = re.sub(r"[\[{][^\[\]{}]*[\]}]", lambda match: re.sub(r"\s+", " ", match.group()), text)
    return text + "\n"


def _cell(entry: dict[str, Any] | None, *, interval: bool = False) -> str:
    if entry is None:
        return "n/a"
    if interval:
        return f"{entry['rate']:.3f} [{entry['low']:.3f}, {entry['high']:.3f}]"
    return f"{entry['rate']:.2f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _label(entry: dict[str, Any]) -> str:
    control = " (control)" if entry["negative_control"] else ""
    return f"`{entry['fault']}`{control}"


def render_report(result: dict[str, Any]) -> str:
    """Markdown report of a benchmark result; a pure function of the JSON content."""
    meta, config = result["meta"], result["meta"]["config"]
    sizes = meta["split_sizes"]
    lines = [
        f"# Benchmark report: {meta['dataset']}",
        "",
        f"- Data: {meta['source']}, {meta['n_rows']:,} rows, "
        f"{len(meta['features']['numeric'])} numeric and "
        f"{len(meta['features']['categorical'])} categorical features.",
        f"- Splits (rows): reference {sizes['reference']:,}, calibration {sizes['calibration']:,}, "
        f"evaluation {sizes['evaluation']:,}, held-out {sizes['held_out']:,}.",
        "- Reference: "
        + (
            f"rolling, the {config['rolling_reference_batches']} batches of rows before each batch."
            if config.get("reference_mode") == "rolling"
            else "static, the reference split."
        ),
        f"- Seed {config['seed']}; {config['n_calibration_batches']} calibration batches, "
        f"{config['n_clean_batches']} clean batches, {config['n_seeds']} faulty batches per fault "
        f"and severity; target false-positive rate {config['target_fpr']}.",
        f"- Commit: {meta['commit'] or 'not recorded'}; "
        + ", ".join(f"{name} {version}" for name, version in sorted(meta["versions"].items()))
        + ".",
        "",
        "Rates are followed by 95% Wilson intervals where space allows; every interval is in the",
        f"JSON file. With {config['n_seeds']} faulty batches a recall of 0.50 has an interval of",
        f"about +/- {_Z95 * 0.5 / math.sqrt(config['n_seeds']):.2f}.",
        "",
    ]
    for run in result["runs"]:
        clean = run["clean"]
        lines += [
            f"## Batch size {run['batch_size']:,}",
            "",
            "### False alarms on clean batches",
            "",
        ]
        rows = [
            [
                f"`{name}` (calibrated)",
                _cell(clean["calibrated"]["per_feature"][name], interval=True),
                _cell(clean["calibrated"]["per_batch"][name], interval=True),
            ]
            for name in _DETECTORS
        ] + [
            [
                f"`{rule}` ({'inferred schema' if rule == _SCHEMA_RULE else 'textbook'})",
                _cell(clean["textbook"]["per_feature"][rule], interval=True),
                _cell(clean["textbook"]["per_batch"][rule], interval=True),
            ]
            for rule in _RULES
        ]
        lines += _table(["Rule", "Per feature", "Per batch (any feature)"], rows)
        for arm, names, title in (
            ("calibrated", _DETECTORS, "Recall, calibrated thresholds"),
            ("textbook", _RULES, "Recall, textbook rules and schema check"),
        ):
            lines += [f"### {title}", ""]
            rows = [
                [_label(entry), f"{entry['severity']:g}", entry["level"]]
                + [_cell(entry[arm][name]) for name in names]
                for entry in run["faults"]
                if entry["scorable"] or arm == "textbook"
            ]
            lines += _table(["Fault", "Severity", "Level", *[f"`{name}`" for name in names]], rows)
        unscorable = sorted({entry["fault"] for entry in run["faults"] if not entry["scorable"]})
        if unscorable:
            lines += [
                "Not scorable by a distribution distance (the column is gone; only the schema "
                "check applies): " + ", ".join(f"`{name}`" for name in unscorable) + ".",
                "",
            ]
    if result["model"] is not None:
        lines += _render_model(result)
    lines += [
        "## What this report does not measure",
        "",
        "- Real incidents: every fault is synthetic and follows the documented injector.",
        "- Several faults at once, or a fault that builds up slowly over many batches.",
        "- Calibration on unlimited fresh data: thresholds come from one finite split and are",
        "  applied to another, so the achieved false-positive rate is above the target, and the",
        "  gap grows with the batch size relative to the split size. The achieved rate is shown.",
        "- The model part uses one model family and accuracy only.",
        "",
    ]
    return "\n".join(lines)


def _render_model(result: dict[str, Any]) -> list[str]:
    model = result["model"]
    run = result["runs"][len(result["runs"]) // 2]
    best = {
        (entry["fault"], entry["severity"]): max(
            (item["rate"] for item in entry["calibrated"].values() if item is not None),
            default=None,
        )
        for entry in run["faults"]
    }
    by_schema = {
        (entry["fault"], entry["severity"]): entry["textbook"][_SCHEMA_RULE]
        for entry in run["faults"]
    }
    lines = [
        "## Does the alert matter?",
        "",
        f"Model: gradient boosting trained on the reference split; clean held-out accuracy "
        f"{model['clean_accuracy']:.4f}. Accuracy drop = clean minus faulty on held-out batches of "
        f"{model['batch_size']:,} rows (positive = the fault hurts). Best recall = highest "
        f"calibrated recall of any distance at batch size {run['batch_size']:,}; schema = recall "
        "of the schema check at the same batch size.",
        "",
    ]
    rows, silent, noisy = [], [], []
    for entry in model["faults"]:
        recall = best.get((entry["fault"], entry["severity"]))
        rows.append(
            [
                f"`{entry['fault']}`",
                f"{entry['severity']:g}",
                f"{entry['accuracy_drop']:+.4f} [{entry['low']:+.4f}, {entry['high']:+.4f}]",
                "n/a" if recall is None else f"{recall:.2f}",
                _cell(by_schema.get((entry["fault"], entry["severity"]))),
            ]
        )
        tag = f"`{entry['fault']}` at {entry['severity']:g}"
        if recall is not None and recall < 0.5 and entry["low"] > 0.005:
            silent.append(tag)
        if recall is not None and recall >= 0.9 and abs(entry["accuracy_drop"]) < 0.005:
            noisy.append(tag)
    lines += _table(["Fault", "Severity", "Accuracy drop [95% CI]", "Best recall", "Schema"], rows)
    lines += [
        "Model loses more than 0.005 accuracy but no detector reaches recall 0.5: "
        + (", ".join(silent) or "none")
        + ".",
        "",
        "A detector reaches recall 0.9 but accuracy moves by less than 0.005: "
        + (", ".join(noisy) or "none")
        + ".",
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fault-injection benchmark.")
    parser.add_argument("--config", required=True, type=Path, help="JSON config file")
    parser.add_argument("--out", required=True, type=Path, help="where to write the JSON result")
    parser.add_argument("--report", type=Path, help="where to write the Markdown report")
    parser.add_argument("--commit", default=None, help="commit hash to record in the result")
    arguments = parser.parse_args(argv)
    config = BenchConfig.from_dict(json.loads(arguments.config.read_text(encoding="utf-8")))
    result = run_benchmark(config, commit=arguments.commit)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(to_json(result), encoding="utf-8", newline="\n")
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(render_report(result), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
