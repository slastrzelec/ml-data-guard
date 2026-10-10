"""Per-feature detectors: one score per (feature, batch), always "batch versus reference".

``fit_reference`` looks at the reference split only and freezes everything a score needs: bin
edges or vocabulary, reference counts and, for numeric features, the sorted reference values.
``score_feature`` then maps any batch of that feature to one number per detector. Thresholds are
not part of this module; they are calibrated or fixed by the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from mlguard.distances import (
    FloatArray,
    bin_counts,
    category_counts,
    chi_square_test,
    fit_vocabulary,
    js_divergence,
    kl_divergence,
    ks_test_presorted,
    l1_distance,
    linf_distance,
    normalize,
    psi,
    quantile_edges,
)

FeatureKind = Literal["numeric", "categorical"]

#: Detectors whose threshold is calibrated on clean batches (larger score = more different).
CALIBRATED_DETECTORS: dict[FeatureKind, tuple[str, ...]] = {
    "numeric": ("linf", "l1", "kl", "js", "psi", "chi2", "ks"),
    "categorical": ("linf", "l1", "kl", "js", "psi", "chi2"),
}

#: Fixed rules as used in practice: name -> (score, direction, threshold).
TEXTBOOK_RULES: dict[str, tuple[str, Literal["below", "above"], float]] = {
    "chi2 p<0.05": ("chi2_pvalue", "below", 0.05),
    "ks p<0.05": ("ks_pvalue", "below", 0.05),
    "psi>0.2": ("psi", "above", 0.2),
    "psi>0.1": ("psi", "above", 0.1),
    "linf>0.01": ("linf", "above", 0.01),
}


@dataclass(frozen=True)
class FeatureReference:
    """Everything the detectors keep about one feature of the reference split (no raw rows for
    categorical features; the sorted values of numeric features are kept in memory only)."""

    name: str
    kind: FeatureKind
    counts: FloatArray
    edges: FloatArray | None = None
    vocabulary: tuple[object, ...] | None = None
    sorted_values: FloatArray | None = None


def strict_numeric(values: ArrayLike) -> FloatArray:
    """Float array where everything that is not a number (text, ``None``, booleans) is NaN.

    A number stored as text is *not* parsed: a column that silently changed its type is treated
    as missing, which is what a type-strict consumer of the data would see.
    """
    array = np.asarray(values)
    if array.dtype != object:
        if np.issubdtype(array.dtype, np.number):
            return array.astype(np.float64)
        return np.full(array.shape, np.nan)
    result = np.full(array.shape, np.nan)
    for position, value in enumerate(array):
        is_number = isinstance(value, int | float | np.integer | np.floating)
        if is_number and not isinstance(value, bool | np.bool_):
            result[position] = float(value)
    return result


def fit_reference(
    name: str,
    values: ArrayLike,
    kind: FeatureKind,
    *,
    n_bins: int = 10,
    max_categories: int = 100,
) -> FeatureReference:
    """Freeze the reference description of one feature."""
    if kind == "numeric":
        numbers = strict_numeric(values)
        edges = quantile_edges(numbers, n_bins=n_bins)
        present = np.sort(numbers[~np.isnan(numbers)])
        return FeatureReference(
            name, kind, bin_counts(numbers, edges), edges=edges, sorted_values=present
        )
    if kind == "categorical":
        items = np.asarray(values, dtype=object)
        vocabulary = fit_vocabulary(items, max_categories=max_categories)
        return FeatureReference(
            name, kind, category_counts(items, vocabulary), vocabulary=vocabulary
        )
    raise ValueError(f"kind must be 'numeric' or 'categorical', got {kind!r}")


def category_codes(values: ArrayLike, vocabulary: tuple[object, ...]) -> np.ndarray:
    """Position of every value in ``category_counts``' output: vocabulary index, unseen or missing.

    ``np.bincount(codes, minlength=len(vocabulary) + 2)`` equals ``category_counts(values, ...)``;
    the benchmark encodes a whole column once and then counts batches without a Python loop.
    """
    index = {category: position for position, category in enumerate(vocabulary)}
    unseen, missing = len(vocabulary), len(vocabulary) + 1
    items = np.asarray(values, dtype=object)
    codes = np.empty(items.size, dtype=np.int64)
    for position, value in enumerate(items):
        if value is None or (isinstance(value, float) and value != value):
            codes[position] = missing
        else:
            codes[position] = index.get(value, unseen)
    return codes


def score_counts(
    reference: FeatureReference,
    counts: FloatArray,
    present: FloatArray | None = None,
    *,
    smoothing: float = 1e-4,
) -> dict[str, float]:
    """Detector scores from the batch counts (and, for numeric features, its non-missing values)."""
    p, q = normalize(counts), normalize(reference.counts)
    chi2 = chi_square_test(counts, reference.counts)
    scores = {
        "linf": linf_distance(p, q),
        "l1": l1_distance(p, q),
        "kl": kl_divergence(p, q, smoothing=smoothing),
        "js": js_divergence(p, q),
        "psi": psi(p, q, smoothing=smoothing),
        "chi2": chi2.statistic,
        "chi2_pvalue": chi2.pvalue,
    }
    if reference.kind == "numeric":
        if present is not None and present.size:
            ks = ks_test_presorted(present, reference.sorted_values)
            scores["ks"], scores["ks_pvalue"] = ks.statistic, ks.pvalue
        else:
            scores["ks"], scores["ks_pvalue"] = float("nan"), float("nan")
    return scores


def score_feature(
    reference: FeatureReference, values: ArrayLike, *, smoothing: float = 1e-4
) -> dict[str, float]:
    """All detector scores of one batch of one feature against its reference.

    Keys: ``linf``, ``l1``, ``kl`` (batch || reference), ``js``, ``psi``, ``chi2``,
    ``chi2_pvalue`` and, for numeric features, ``ks`` and ``ks_pvalue`` (NaN when the batch has
    no numeric value left, because KS cannot see missing values).
    """
    if reference.kind == "numeric":
        numbers = strict_numeric(values)
        counts = bin_counts(numbers, reference.edges)
        return score_counts(reference, counts, numbers[~np.isnan(numbers)], smoothing=smoothing)
    counts = category_counts(np.asarray(values, dtype=object), reference.vocabulary)
    return score_counts(reference, counts, smoothing=smoothing)


def textbook_alarm(rule: str, scores: dict[str, float]) -> bool | None:
    """Whether a fixed rule alarms; ``None`` when the rule does not apply to this feature."""
    key, direction, threshold = TEXTBOOK_RULES[rule]
    if key not in scores:
        return None
    value = scores[key]
    return bool(value < threshold) if direction == "below" else bool(value > threshold)
