"""Turning raw values into aligned count vectors without looking at the batch under test."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlguard.distances import (
    bin_counts,
    category_counts,
    fit_vocabulary,
    linf_distance,
    normalize,
    quantile_edges,
)

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
maybe_missing = st.one_of(finite, st.just(math.nan), st.just(math.inf), st.just(-math.inf))
categories = st.one_of(st.sampled_from(["a", "b", "c", "d", "e"]), st.integers(0, 3), st.none())


# --------------------------------------------------------------------------- categorical


def test_vocabulary_is_ordered_by_frequency_and_skips_missing_values() -> None:
    reference = ["b", "a", "b", None, "c", "b", "a", math.nan]
    assert fit_vocabulary(reference) == ("b", "a", "c")


def test_vocabulary_can_be_capped_to_the_most_frequent_categories() -> None:
    reference = ["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"]
    assert fit_vocabulary(reference, max_categories=2) == ("a", "b")


def test_vocabulary_order_does_not_depend_on_the_order_of_the_rows() -> None:
    reference = ["x", "y", "z", 1, 2, "x"]
    assert fit_vocabulary(reference) == fit_vocabulary(list(reversed(reference)))


def test_category_counts_put_unseen_and_missing_values_in_their_own_buckets() -> None:
    vocabulary = ("a", "b")
    counts = category_counts(["a", "a", "b", "NEW", "OTHER", None, math.nan, None], vocabulary)
    np.testing.assert_array_equal(counts, [2, 1, 2, 3])


def test_a_new_category_is_visible_to_linf_even_though_the_reference_never_saw_it() -> None:
    reference = ["a"] * 50 + ["b"] * 50
    batch = ["a"] * 40 + ["b"] * 40 + ["c"] * 20
    vocabulary = fit_vocabulary(reference)
    p = normalize(category_counts(reference, vocabulary))
    q = normalize(category_counts(batch, vocabulary))
    assert linf_distance(p, q) == pytest.approx(0.2)


def test_a_number_turned_into_a_string_is_an_unseen_category() -> None:
    vocabulary = fit_vocabulary([1, 2, 1])
    np.testing.assert_array_equal(category_counts(["1", "2", 1], vocabulary), [1, 0, 2, 0])


@given(st.lists(categories, min_size=1, max_size=40), st.lists(categories, max_size=40))
def test_category_counts_are_aligned_and_account_for_every_row(
    reference: list[object], batch: list[object]
) -> None:
    if all(value is None for value in reference):
        reference = [*reference, "a"]
    vocabulary = fit_vocabulary(reference)
    reference_counts = category_counts(reference, vocabulary)
    batch_counts = category_counts(batch, vocabulary)
    assert reference_counts.shape == batch_counts.shape == (len(vocabulary) + 2,)
    assert reference_counts.sum() == len(reference)
    assert batch_counts.sum() == len(batch)
    assert reference_counts[len(vocabulary)] == 0  # the reference has no unseen categories


def test_invalid_vocabulary_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-missing"):
        fit_vocabulary([None, math.nan])
    with pytest.raises(ValueError, match="max_categories"):
        fit_vocabulary(["a"], max_categories=0)
    with pytest.raises(ValueError, match="empty"):
        category_counts(["a"], ())
    with pytest.raises(ValueError, match="duplicates"):
        category_counts(["a"], ("a", "a"))


# --------------------------------------------------------------------------- numeric


def test_quantile_edges_split_the_reference_evenly() -> None:
    reference = np.arange(1000, dtype=float)
    edges = quantile_edges(reference, n_bins=4)
    np.testing.assert_allclose(edges, [249.75, 499.5, 749.25])
    counts = bin_counts(reference, edges)
    # below, on-edge, between, on-edge, between, on-edge, above, missing
    np.testing.assert_array_equal(counts, [250, 0, 250, 0, 250, 0, 250, 0])


def test_quantile_edges_ignore_missing_values() -> None:
    reference = np.arange(1000, dtype=float)
    with_missing = np.concatenate([reference, [math.nan] * 100])
    np.testing.assert_array_equal(quantile_edges(with_missing), quantile_edges(reference))


def test_values_outside_the_reference_range_have_a_bin() -> None:
    edges = quantile_edges(np.arange(100, dtype=float), n_bins=4)
    counts = bin_counts([-1e9, 1e9, math.inf, -math.inf], edges)
    assert counts[0] == 2
    assert counts[-2] == 2
    assert counts.sum() == 4


def test_missing_values_are_counted_in_the_last_bucket() -> None:
    counts = bin_counts([1.0, math.nan, math.nan], [0.0])
    np.testing.assert_array_equal(counts, [0, 0, 1, 2])


def test_a_sentinel_value_in_the_reference_gets_its_own_bin() -> None:
    """Zero-inflated feature: pinning more rows to the sentinel moves mass into one exact bin."""
    rng = np.random.default_rng(0)
    reference = np.concatenate([np.full(300, -1.0), rng.uniform(0, 10, 700)])
    edges = quantile_edges(reference, n_bins=10)
    assert -1.0 in edges
    on_sentinel = 2 * int(np.flatnonzero(edges == -1.0)[0]) + 1
    counts = bin_counts(reference, edges)
    assert counts[on_sentinel] == 300


def test_a_constant_reference_still_separates_lower_equal_and_higher_values() -> None:
    edges = quantile_edges([5.0] * 20)
    np.testing.assert_array_equal(edges, [5.0])
    np.testing.assert_array_equal(bin_counts([4.0, 5.0, 5.0, 6.0, 7.0, 8.0], edges), [1, 2, 3, 0])


def test_a_unit_change_moves_all_mass_to_the_top_bin() -> None:
    reference = np.linspace(1, 100, 1000)
    edges = quantile_edges(reference)
    p = normalize(bin_counts(reference, edges))
    q = normalize(bin_counts(reference * 1000, edges))
    assert q[-2] == 1.0
    assert linf_distance(p, q) == pytest.approx(0.9)


@given(
    st.lists(finite, min_size=1, max_size=60),
    st.lists(maybe_missing, max_size=60),
    st.integers(min_value=2, max_value=12),
)
def test_bin_counts_are_aligned_and_account_for_every_row(
    reference: list[float], batch: list[float], n_bins: int
) -> None:
    edges = quantile_edges(reference, n_bins=n_bins)
    assert 1 <= edges.size <= n_bins - 1
    assert np.all(np.diff(edges) > 0)
    reference_counts = bin_counts(reference, edges)
    batch_counts = bin_counts(batch, edges)
    assert reference_counts.shape == batch_counts.shape == (2 * edges.size + 2,)
    assert reference_counts.sum() == len(reference)
    assert batch_counts.sum() == len(batch)
    assert batch_counts[-1] == sum(1 for value in batch if math.isnan(value))


@given(st.lists(finite, min_size=1, max_size=60), st.lists(finite, min_size=1, max_size=60))
def test_edges_depend_on_the_reference_only(reference: list[float], batch: list[float]) -> None:
    """Leakage rule: the batch under test never influences how it is binned."""
    edges_before = quantile_edges(reference)
    bin_counts(batch, edges_before)
    np.testing.assert_array_equal(quantile_edges(reference), edges_before)
    np.testing.assert_array_equal(quantile_edges(list(reversed(reference))), edges_before)


def test_inputs_are_not_modified() -> None:
    reference = np.array([3.0, 1.0, 2.0, math.nan])
    snapshot = reference.copy()
    edges = quantile_edges(reference)
    bin_counts(reference, edges)
    np.testing.assert_array_equal(reference, snapshot)


def test_invalid_binning_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        quantile_edges([1.0, 2.0], n_bins=1)
    with pytest.raises(ValueError, match="non-missing"):
        quantile_edges([math.nan])
    with pytest.raises(ValueError, match="infinite"):
        quantile_edges([1.0, math.inf])
    with pytest.raises(ValueError, match="one-dimensional"):
        quantile_edges([[1.0, 2.0]])
    with pytest.raises(ValueError, match="edges"):
        bin_counts([1.0], [])
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_counts([1.0], [2.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        bin_counts([1.0], [1.0, math.inf])
    with pytest.raises(ValueError, match="one-dimensional"):
        bin_counts([[1.0]], [0.0])
