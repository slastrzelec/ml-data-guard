"""Distances between two distributions of one feature.

Three kinds of input are supported and never mixed implicitly:

* probability vectors (``linf_distance``, ``l1_distance``, ``kl_divergence``, ``js_divergence``,
  ``psi``): two aligned vectors that each sum to one;
* count vectors (``chi_square_test``): the sample sizes matter;
* raw numeric samples (``ks_test``).

``fit_vocabulary`` / ``category_counts`` and ``quantile_edges`` / ``bin_counts`` turn raw values
into aligned count vectors. They are split into a *fit* step that sees the reference data only and a
*count* step that is applied unchanged to any batch, so that nothing about a batch under test can
leak into how it is measured.

Every function is pure, deterministic and raises ``ValueError`` on invalid input instead of
returning a number that looks plausible.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

#: Largest accepted deviation of a probability vector's sum from one.
SUM_TOLERANCE = 1e-9


class StatResult(NamedTuple):
    """Statistic and p-value of a two-sample test."""

    statistic: float
    pvalue: float


# --------------------------------------------------------------------------- validation


def _as_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} must not contain negative values")
    return array


def _as_probabilities(values: ArrayLike, name: str) -> FloatArray:
    array = _as_vector(values, name)
    total = float(array.sum())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to 1, got {total!r}; use normalize() on counts")
    return array


def _as_pair(p: ArrayLike, q: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p_array = _as_probabilities(p, "p")
    q_array = _as_probabilities(q, "q")
    if p_array.shape != q_array.shape:
        raise ValueError(
            f"p and q must have the same length, got {p_array.size} and {q_array.size}"
        )
    return p_array, q_array


def _smooth(p: FloatArray, q: FloatArray, smoothing: float) -> tuple[FloatArray, FloatArray]:
    if not math.isfinite(smoothing) or smoothing < 0:
        raise ValueError(f"smoothing must be a finite non-negative number, got {smoothing!r}")
    if smoothing == 0:
        return p, q
    scale = 1.0 + smoothing * p.size
    return (p + smoothing) / scale, (q + smoothing) / scale


def normalize(counts: ArrayLike) -> FloatArray:
    """Turn a vector of non-negative counts into a probability vector."""
    array = _as_vector(counts, "counts")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("counts must have a positive total")
    return array / total


# --------------------------------------------------------------------------- probability vectors


def linf_distance(p: ArrayLike, q: ArrayLike) -> float:
    """Largest change in the probability of any single value, in [0, 1].

    The L-infinity (Chebyshev) distance of Breck et al., "Data Validation for Machine Learning".
    A threshold reads directly as "no value may gain or lose more than x of the traffic".
    """
    p_array, q_array = _as_pair(p, q)
    return float(np.max(np.abs(p_array - q_array)))


def l1_distance(p: ArrayLike, q: ArrayLike) -> float:
    """Sum of absolute probability changes, in [0, 2] (twice the total variation distance)."""
    p_array, q_array = _as_pair(p, q)
    return float(np.sum(np.abs(p_array - q_array)))


def _kl(p: FloatArray, q: FloatArray) -> float:
    support = p > 0
    if np.any(q[support] == 0):
        return math.inf
    # log(p) - log(q) instead of log(p / q): the ratio overflows for subnormal q.
    value = float(np.sum(p[support] * (np.log(p[support]) - np.log(q[support]))))
    return max(value, 0.0)


def kl_divergence(
    p: ArrayLike, q: ArrayLike, *, smoothing: float = 0.0, base: float = math.e
) -> float:
    """Kullback-Leibler divergence KL(p || q); not symmetric.

    Infinite when ``p`` puts mass where ``q`` has none. ``smoothing`` adds that constant to every
    probability of both vectors and renormalises, which makes the result finite; it is off by
    default so that the caller has to choose it explicitly.
    """
    if not base > 1:
        raise ValueError(f"base must be greater than 1, got {base!r}")
    p_array, q_array = _smooth(*_as_pair(p, q), smoothing)
    return _kl(p_array, q_array) / math.log(base)


def js_divergence(p: ArrayLike, q: ArrayLike, *, base: float = 2.0) -> float:
    """Jensen-Shannon divergence; symmetric, always finite, in [0, 1] for base 2."""
    if not base > 1:
        raise ValueError(f"base must be greater than 1, got {base!r}")
    p_array, q_array = _as_pair(p, q)
    mixture = 0.5 * (p_array + q_array)
    value = 0.5 * _kl(p_array, mixture) + 0.5 * _kl(q_array, mixture)
    return value / math.log(base)


def psi(p: ArrayLike, q: ArrayLike, *, smoothing: float = 0.0) -> float:
    """Population stability index: sum((p - q) * ln(p / q)) = KL(p || q) + KL(q || p).

    Infinite when exactly one of the two vectors has no mass in a bin, unless ``smoothing`` is set.
    """
    p_array, q_array = _smooth(*_as_pair(p, q), smoothing)
    occupied = (p_array > 0) | (q_array > 0)
    p_occupied, q_occupied = p_array[occupied], q_array[occupied]
    if np.any(p_occupied == 0) or np.any(q_occupied == 0):
        return math.inf
    log_ratio = np.log(p_occupied) - np.log(q_occupied)
    value = float(np.sum((p_occupied - q_occupied) * log_ratio))
    return max(value, 0.0)


# --------------------------------------------------------------------------- counts and samples


def chi_square_test(counts_a: ArrayLike, counts_b: ArrayLike) -> StatResult:
    """Pearson chi-square test of homogeneity for two aligned count vectors.

    Both vectors are treated as samples. Categories that are empty in both are dropped; with fewer
    than two occupied categories the two samples cannot differ and the result is (0, 1).
    """
    a = _as_vector(counts_a, "counts_a")
    b = _as_vector(counts_b, "counts_b")
    if a.shape != b.shape:
        raise ValueError(f"count vectors must have the same length, got {a.size} and {b.size}")
    total_a, total_b = float(a.sum()), float(b.sum())
    if total_a <= 0 or total_b <= 0:
        raise ValueError("both count vectors must have a positive total")
    occupied = (a + b) > 0
    a, b = a[occupied], b[occupied]
    if a.size < 2:
        return StatResult(0.0, 1.0)
    column_totals = a + b
    expected_a = column_totals * total_a / (total_a + total_b)
    expected_b = column_totals * total_b / (total_a + total_b)
    statistic = float(
        np.sum((a - expected_a) ** 2 / expected_a + (b - expected_b) ** 2 / expected_b)
    )
    pvalue = float(stats.chi2.sf(statistic, df=a.size - 1))
    return StatResult(statistic, pvalue)


def ks_test(sample_a: ArrayLike, sample_b: ArrayLike) -> StatResult:
    """Two-sample, two-sided Kolmogorov-Smirnov test on raw numeric samples.

    The statistic is exact (largest gap between the two empirical CDFs, ties included). The p-value
    is the asymptotic Kolmogorov approximation with Stephens' small-sample correction; it is
    approximate for small samples and conservative when there are many ties.
    Missing values are rejected: the caller decides whether a missing value is a fault of its own.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    for name, array in (("sample_a", a), ("sample_b", b)):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        if np.any(np.isnan(array)):
            raise ValueError(f"{name} must not contain NaN")
    a, b = np.sort(a), np.sort(b)
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / a.size
    cdf_b = np.searchsorted(b, points, side="right") / b.size
    statistic = float(np.max(np.abs(cdf_a - cdf_b)))
    return StatResult(statistic, _ks_pvalue(statistic, a.size, b.size))


def _ks_pvalue(statistic: float, size_a: int, size_b: int) -> float:
    effective_n = math.sqrt(size_a * size_b / (size_a + size_b))
    scaled = (effective_n + 0.12 + 0.11 / effective_n) * statistic
    return float(np.clip(stats.kstwobign.sf(scaled), 0.0, 1.0))


def ks_test_presorted(sample: ArrayLike, sorted_reference: FloatArray) -> StatResult:
    """Same result as ``ks_test(sample, sorted_reference)`` for a large, already sorted reference.

    Between two neighbouring sample points the sample CDF is flat and the reference CDF only grows,
    so the largest gap is reached at a sample point: either at it or just before it. That needs
    ``O(n log N)`` instead of sorting and scanning all ``N`` reference values for every batch.
    ``sorted_reference`` must be sorted, non-empty and free of NaN; this is not re-checked.
    """
    values = np.asarray(sample, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"sample must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("sample must not be empty")
    if np.any(np.isnan(values)):
        raise ValueError("sample must not contain NaN")
    values = np.sort(values)
    n, size = values.size, sorted_reference.size
    at = np.abs(
        np.searchsorted(values, values, side="right") / n
        - np.searchsorted(sorted_reference, values, side="right") / size
    )
    before = np.abs(
        np.searchsorted(values, values, side="left") / n
        - np.searchsorted(sorted_reference, values, side="left") / size
    )
    statistic = float(max(at.max(), before.max()))
    return StatResult(statistic, _ks_pvalue(statistic, n, size))


# --------------------------------------------------------------------------- reference-only binning


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def fit_vocabulary(
    reference: Iterable[Hashable], *, max_categories: int | None = None
) -> tuple[Hashable, ...]:
    """Categories of a feature, learned from the reference data only.

    Missing values (``None``, NaN) are never part of the vocabulary. With ``max_categories`` only
    the most frequent categories are kept; the order is deterministic (frequency, then type and
    repr).
    """
    if max_categories is not None and max_categories < 1:
        raise ValueError(f"max_categories must be at least 1, got {max_categories!r}")
    frequencies = Counter(value for value in reference if not _is_missing(value))
    if not frequencies:
        raise ValueError("reference must contain at least one non-missing value")
    ordered = sorted(
        frequencies, key=lambda value: (-frequencies[value], type(value).__name__, repr(value))
    )
    return tuple(ordered[:max_categories])


def category_counts(values: Iterable[Hashable], vocabulary: Sequence[Hashable]) -> FloatArray:
    """Count ``values`` per vocabulary entry, plus two trailing buckets: unseen and missing.

    The result has ``len(vocabulary) + 2`` entries for any input, so reference and batch counts are
    always aligned. A category that the reference never saw lands in the *unseen* bucket.
    """
    if len(vocabulary) == 0:
        raise ValueError("vocabulary must not be empty")
    index = {category: position for position, category in enumerate(vocabulary)}
    if len(index) != len(vocabulary):
        raise ValueError("vocabulary must not contain duplicates")
    unseen, missing = len(vocabulary), len(vocabulary) + 1
    counts = np.zeros(len(vocabulary) + 2, dtype=np.float64)
    for value in values:
        if _is_missing(value):
            counts[missing] += 1
        else:
            counts[index.get(value, unseen)] += 1
    return counts


def quantile_edges(reference: ArrayLike, *, n_bins: int = 10) -> FloatArray:
    """Strictly increasing bin edges at the quantiles of the reference data only.

    NaN is ignored, infinities are rejected. Repeated quantiles collapse into one edge, so a value
    that carries at least ``1 / n_bins`` of the reference mass (a sentinel, a zero-inflated feature)
    becomes an edge and gets a bin of its own in ``bin_counts``.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins!r}")
    array = np.asarray(reference, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"reference must be one-dimensional, got shape {array.shape}")
    array = array[~np.isnan(array)]
    if array.size == 0:
        raise ValueError("reference must contain at least one non-missing value")
    if np.any(np.isinf(array)):
        raise ValueError("reference must not contain infinite values")
    levels = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(array, levels))


def bin_counts(values: ArrayLike, edges: ArrayLike) -> FloatArray:
    """Count ``values`` in the bins defined by ``edges``; the last bucket counts NaN.

    With ``k`` edges there are ``2 * k + 1`` value bins: below the first edge, exactly on it,
    between the first and the second, exactly on the second, ..., above the last. Values outside the
    reference range therefore always have a bin, and a point mass on an edge is not diluted by its
    neighbours. The result has ``2 * k + 2`` entries.
    """
    edge_array = np.asarray(edges, dtype=np.float64)
    if edge_array.ndim != 1 or edge_array.size == 0:
        raise ValueError("edges must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(edge_array)) or np.any(np.diff(edge_array) <= 0):
        raise ValueError("edges must be finite and strictly increasing")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {array.shape}")
    missing = np.isnan(array)
    present = array[~missing]
    positions = np.searchsorted(edge_array, present, side="left") + np.searchsorted(
        edge_array, present, side="right"
    )
    n_value_bins = 2 * edge_array.size + 1
    counts = np.zeros(n_value_bins + 1, dtype=np.float64)
    counts[:n_value_bins] = np.bincount(positions, minlength=n_value_bins)
    counts[n_value_bins] = float(missing.sum())
    return counts
