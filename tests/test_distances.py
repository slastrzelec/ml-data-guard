"""Distances on probability vectors, counts and samples."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy import stats
from scipy.spatial.distance import jensenshannon

from mlguard.distances import (
    chi_square_test,
    js_divergence,
    kl_divergence,
    ks_test,
    l1_distance,
    linf_distance,
    normalize,
    psi,
)

TOL = 1e-9
SYMMETRIC = [linf_distance, l1_distance, js_divergence, psi]
ALL_VECTOR_DISTANCES = [*SYMMETRIC, kl_divergence]


def _weights(size: int, *, min_value: float = 0.0) -> st.SearchStrategy[list[float]]:
    return st.lists(
        st.floats(min_value=min_value, max_value=1.0, allow_nan=False),
        min_size=size,
        max_size=size,
    ).filter(lambda values: sum(values) > 1e-3)


@st.composite
def probability_pairs(draw: st.DrawFn, *, positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two aligned probability vectors; zeros are allowed unless ``positive``."""
    size = draw(st.integers(min_value=2, max_value=12))
    min_value = 1e-3 if positive else 0.0
    p = normalize(draw(_weights(size, min_value=min_value)))
    q = normalize(draw(_weights(size, min_value=min_value)))
    return p, q


@st.composite
def count_pairs(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    size = draw(st.integers(min_value=2, max_value=10))
    counts = st.lists(st.integers(min_value=0, max_value=500), min_size=size, max_size=size)
    a = np.array(draw(counts.filter(lambda values: sum(values) > 0)), dtype=float)
    b = np.array(draw(counts.filter(lambda values: sum(values) > 0)), dtype=float)
    return a, b


samples = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=1, max_size=60
)


# --------------------------------------------------------------------------- known values


def test_known_values_on_a_two_category_feature() -> None:
    p, q = [0.5, 0.5], [0.9, 0.1]
    assert linf_distance(p, q) == pytest.approx(0.4)
    assert l1_distance(p, q) == pytest.approx(0.8)


def test_linf_reports_the_single_largest_change_not_the_sum() -> None:
    p, q = [0.4, 0.3, 0.2, 0.1], [0.1, 0.4, 0.3, 0.2]
    assert linf_distance(p, q) == pytest.approx(0.3)
    assert l1_distance(p, q) == pytest.approx(0.6)


def test_kl_known_value_and_asymmetry() -> None:
    p, q = [0.5, 0.5], [0.25, 0.75]
    expected_pq = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    expected_qp = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert kl_divergence(p, q) == pytest.approx(expected_pq)
    assert kl_divergence(q, p) == pytest.approx(expected_qp)
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_kl_base_two_is_in_bits() -> None:
    assert kl_divergence([1.0, 0.0], [0.5, 0.5], base=2) == pytest.approx(1.0)


def test_psi_known_value() -> None:
    p, q = [0.5, 0.5], [0.25, 0.75]
    expected = (0.5 - 0.25) * math.log(0.5 / 0.25) + (0.5 - 0.75) * math.log(0.5 / 0.75)
    assert psi(p, q) == pytest.approx(expected)


def test_js_is_one_bit_for_disjoint_distributions() -> None:
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert js_divergence([1.0, 0.0], [0.0, 1.0], base=math.e) == pytest.approx(math.log(2))


def test_disjoint_support_is_infinite_without_smoothing_and_finite_with_it() -> None:
    p, q = [1.0, 0.0], [0.0, 1.0]
    assert kl_divergence(p, q) == math.inf
    assert psi(p, q) == math.inf
    assert math.isfinite(kl_divergence(p, q, smoothing=1e-6))
    assert math.isfinite(psi(p, q, smoothing=1e-6))


def test_kl_is_finite_when_only_q_has_extra_support() -> None:
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert psi([1.0, 0.0], [0.5, 0.5]) == math.inf


def test_bins_empty_in_both_vectors_do_not_change_the_result() -> None:
    p, q = [0.5, 0.5], [0.25, 0.75]
    padded_p, padded_q = [0.5, 0.0, 0.5], [0.25, 0.0, 0.75]
    for distance in ALL_VECTOR_DISTANCES:
        assert distance(padded_p, padded_q) == pytest.approx(distance(p, q))


def test_smoothing_known_value_is_renormalised() -> None:
    # (1, 0) and (0, 1) with smoothing 1 become (2/3, 1/3) and (1/3, 2/3).
    assert kl_divergence([1.0, 0.0], [0.0, 1.0], smoothing=1.0) == pytest.approx(math.log(2) / 3)
    assert psi([1.0, 0.0], [0.0, 1.0], smoothing=1.0) == pytest.approx(2 * math.log(2) / 3)


def test_smoothing_moves_both_vectors_towards_uniform() -> None:
    p, q = [0.9, 0.1], [0.1, 0.9]
    assert kl_divergence(p, q, smoothing=0.5) < kl_divergence(p, q)
    assert psi(p, q, smoothing=0.5) < psi(p, q)


def test_chi_square_known_value() -> None:
    # 2x2 table, every expected count is 15: statistic = 4 * 5**2 / 15.
    result = chi_square_test([10, 20], [20, 10])
    assert result.statistic == pytest.approx(100 / 15)
    assert result.pvalue == pytest.approx(0.009823, abs=1e-6)


def test_chi_square_grows_with_sample_size_for_the_same_proportions() -> None:
    """The reason the paper prefers L-inf: a fixed tiny shift becomes 'significant' at scale."""
    small = chi_square_test([500, 500], [510, 490])
    large = chi_square_test([500_000, 500_000], [510_000, 490_000])
    assert small.pvalue > 0.05
    assert large.pvalue < 1e-6
    assert linf_distance(normalize([500, 500]), normalize([510, 490])) == pytest.approx(
        linf_distance(normalize([500_000, 500_000]), normalize([510_000, 490_000]))
    )


def test_chi_square_with_a_single_occupied_category_cannot_reject() -> None:
    assert chi_square_test([7, 0, 0], [3, 0, 0]) == (0.0, 1.0)


def test_ks_known_values() -> None:
    assert ks_test([1, 2, 3], [4, 5, 6]).statistic == pytest.approx(1.0)
    assert ks_test([1, 2, 3, 4], [3, 4, 5, 6]).statistic == pytest.approx(0.5)
    assert ks_test([1, 2, 3], [1, 2, 3]) == (0.0, 1.0)


def test_ks_handles_ties() -> None:
    # CDFs at 0: 5/5 vs 4/5.
    assert ks_test([0, 0, 0, 0, 0], [0, 0, 0, 0, 1]).statistic == pytest.approx(0.2)


def test_ks_pvalue_is_close_to_the_exact_one_for_moderate_samples() -> None:
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.0, 1.0, 400), rng.normal(0.25, 1.0, 500)
    exact = stats.ks_2samp(a, b, method="exact")
    result = ks_test(a, b)
    assert result.statistic == pytest.approx(exact.statistic)
    assert result.pvalue == pytest.approx(exact.pvalue, rel=0.1)
    assert result.pvalue < 0.01


def test_ks_pvalue_is_roughly_uniform_when_both_samples_share_a_distribution() -> None:
    rng = np.random.default_rng(11)
    pvalues = [ks_test(rng.normal(size=300), rng.normal(size=300)).pvalue for _ in range(400)]
    rejected = np.mean(np.array(pvalues) < 0.05)
    assert 0.02 <= rejected <= 0.08


def test_ks_places_infinite_values_at_the_ends() -> None:
    assert ks_test([-math.inf, 0.0], [0.0, math.inf]).statistic == pytest.approx(0.5)


# --------------------------------------------------------------------------- properties


@given(probability_pairs())
def test_identical_distributions_have_zero_distance(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, _ = pair
    for distance in ALL_VECTOR_DISTANCES:
        assert distance(p, p) == 0.0


@given(probability_pairs())
def test_symmetry(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    for distance in SYMMETRIC:
        forward, backward = distance(p, q), distance(q, p)
        assert forward == backward or forward == pytest.approx(backward, rel=1e-9, abs=TOL)


@given(probability_pairs())
def test_bounds(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    linf, l1 = linf_distance(p, q), l1_distance(p, q)
    assert 0.0 <= linf <= 1.0
    assert 0.0 <= l1 <= 2.0 + TOL
    assert 0.0 <= js_divergence(p, q) <= 1.0 + TOL
    assert kl_divergence(p, q) >= 0.0
    assert psi(p, q) >= 0.0


@given(probability_pairs())
def test_linf_is_at_most_half_of_l1(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Gains and losses of probability balance, so no single change exceeds half of the total."""
    p, q = pair
    assert linf_distance(p, q) <= 0.5 * l1_distance(p, q) + TOL


@given(probability_pairs())
def test_l1_is_at_most_the_number_of_categories_times_linf(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    p, q = pair
    assert l1_distance(p, q) <= p.size * linf_distance(p, q) + TOL


@given(probability_pairs())
def test_pinsker_inequality_links_l1_and_kl(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    assert l1_distance(p, q) ** 2 <= 2.0 * kl_divergence(p, q) + TOL


@given(probability_pairs())
def test_js_in_bits_is_at_most_total_variation(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    assert js_divergence(p, q) <= 0.5 * l1_distance(p, q) + TOL


@given(probability_pairs(positive=True))
def test_psi_is_the_symmetrised_kl(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    assert psi(p, q) == pytest.approx(kl_divergence(p, q) + kl_divergence(q, p), rel=1e-9, abs=TOL)


@given(probability_pairs(), st.data())
def test_relabelling_categories_changes_nothing(
    pair: tuple[np.ndarray, np.ndarray], data: st.DataObject
) -> None:
    p, q = pair
    order = data.draw(st.permutations(range(p.size)))
    for distance in ALL_VECTOR_DISTANCES:
        original, permuted = distance(p, q), distance(p[order], q[order])
        assert original == permuted or original == pytest.approx(permuted, rel=1e-9, abs=TOL)


@given(probability_pairs(positive=True), st.data())
def test_merging_two_categories_never_increases_l1_kl_js_psi(
    pair: tuple[np.ndarray, np.ndarray], data: st.DataObject
) -> None:
    """Data-processing inequality: coarser bins cannot reveal a larger difference."""
    p, q = pair
    assume(p.size >= 3)
    i = data.draw(st.integers(min_value=0, max_value=p.size - 2))

    def merge(vector: np.ndarray) -> np.ndarray:
        merged = np.delete(vector, i + 1)
        merged[i] = vector[i] + vector[i + 1]
        return normalize(merged)

    for distance in (l1_distance, kl_divergence, js_divergence, psi):
        assert distance(merge(p), merge(q)) <= distance(p, q) + TOL


@given(probability_pairs(positive=True))
def test_kl_matches_scipy(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    assert kl_divergence(p, q) == pytest.approx(float(stats.entropy(p, q)), rel=1e-9, abs=TOL)


@given(probability_pairs())
def test_js_matches_scipy(pair: tuple[np.ndarray, np.ndarray]) -> None:
    p, q = pair
    expected = float(jensenshannon(p, q, base=2)) ** 2
    assert js_divergence(p, q) == pytest.approx(expected, rel=1e-6, abs=1e-9)


@given(count_pairs())
def test_chi_square_matches_scipy(pair: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = pair
    occupied = (a + b) > 0
    assume(occupied.sum() >= 2)
    expected = stats.chi2_contingency(np.vstack([a[occupied], b[occupied]]), correction=False)
    result = chi_square_test(a, b)
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-9, abs=TOL)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=1e-9, abs=1e-12)


@given(count_pairs())
def test_chi_square_is_symmetric_and_zero_for_equal_proportions(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    a, b = pair
    assert chi_square_test(a, b).statistic == pytest.approx(
        chi_square_test(b, a).statistic, rel=1e-9, abs=TOL
    )
    same = chi_square_test(a, 3 * a)
    assert same.statistic == pytest.approx(0.0, abs=TOL)
    assert same.pvalue == pytest.approx(1.0)


@given(samples, samples)
def test_ks_properties(a: list[float], b: list[float]) -> None:
    forward, backward = ks_test(a, b), ks_test(b, a)
    assert 0.0 <= forward.statistic <= 1.0
    assert 0.0 <= forward.pvalue <= 1.0
    assert forward == backward
    assert ks_test(a, a) == (0.0, 1.0)


@given(samples, samples)
def test_ks_statistic_matches_scipy(a: list[float], b: list[float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SciPy warns about its own p-value on tiny samples
        expected = stats.ks_2samp(a, b, method="asymp").statistic
    assert ks_test(a, b).statistic == pytest.approx(float(expected), abs=1e-12)


@given(samples, samples, st.floats(min_value=-100, max_value=100), st.floats(0.01, 100))
def test_ks_is_invariant_to_shifting_and_rescaling_both_samples(
    a: list[float], b: list[float], shift: float, scale: float
) -> None:
    def transformed(values: list[float]) -> np.ndarray:
        return np.round(np.asarray(values), 3) * scale + shift

    reference = ks_test(np.round(a, 3), np.round(b, 3)).statistic
    assert ks_test(transformed(a), transformed(b)).statistic == pytest.approx(reference)


@given(_weights(5), st.floats(min_value=0.01, max_value=1000))
def test_normalize_ignores_the_scale_of_the_counts(counts: list[float], factor: float) -> None:
    scaled = normalize(np.asarray(counts) * factor)
    np.testing.assert_allclose(scaled, normalize(counts), rtol=1e-9, atol=1e-12)
    assert scaled.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- invalid input


@pytest.mark.parametrize("distance", ALL_VECTOR_DISTANCES)
@pytest.mark.parametrize(
    ("p", "q"),
    [
        ([0.5, 0.5], [0.5, 0.25, 0.25]),  # different lengths
        ([0.5, 0.6], [0.5, 0.5]),  # does not sum to one
        ([5.0, 5.0], [0.5, 0.5]),  # counts instead of probabilities
        ([1.5, -0.5], [0.5, 0.5]),  # negative probability
        ([math.nan, 1.0], [0.5, 0.5]),
        ([math.inf, 0.0], [0.5, 0.5]),
        ([], []),
        ([[0.5, 0.5]], [[0.5, 0.5]]),  # not one-dimensional
    ],
)
def test_vector_distances_reject_invalid_input(
    distance: Callable[..., float], p: list[float], q: list[float]
) -> None:
    with pytest.raises(ValueError, match=r"must"):
        distance(p, q)


@pytest.mark.parametrize("smoothing", [-1e-9, math.nan, math.inf])
def test_invalid_smoothing_is_rejected(smoothing: float) -> None:
    for distance in (kl_divergence, psi):
        with pytest.raises(ValueError, match="smoothing"):
            distance([0.5, 0.5], [0.5, 0.5], smoothing=smoothing)


@pytest.mark.parametrize("base", [1.0, 0.5, 0.0, -2.0, math.nan])
def test_invalid_logarithm_base_is_rejected(base: float) -> None:
    for distance in (kl_divergence, js_divergence):
        with pytest.raises(ValueError, match="base"):
            distance([0.5, 0.5], [0.5, 0.5], base=base)


@pytest.mark.parametrize("counts", [[0, 0], [-1, 2], [math.nan, 1], [], [[1, 2]]])
def test_normalize_rejects_invalid_counts(counts: list[float]) -> None:
    with pytest.raises(ValueError, match=r"must"):
        normalize(counts)


@pytest.mark.parametrize(
    ("a", "b"),
    [([1, 2], [1, 2, 3]), ([0, 0], [1, 2]), ([1, 2], [0, 0]), ([-1, 2], [1, 2]), ([], [])],
)
def test_chi_square_rejects_invalid_counts(a: list[float], b: list[float]) -> None:
    with pytest.raises(ValueError, match=r"must"):
        chi_square_test(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [([], [1.0]), ([1.0], []), ([1.0, math.nan], [1.0]), ([[1.0]], [1.0])],
)
def test_ks_rejects_empty_missing_and_non_flat_samples(a: list[float], b: list[float]) -> None:
    with pytest.raises(ValueError, match=r"must"):
        ks_test(a, b)
