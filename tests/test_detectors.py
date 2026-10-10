"""Per-feature detectors: reference fitting, scores and fixed rules."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlguard.detectors import (
    CALIBRATED_DETECTORS,
    TEXTBOOK_RULES,
    category_codes,
    fit_reference,
    score_feature,
    strict_numeric,
    textbook_alarm,
)
from mlguard.distances import category_counts, ks_test, ks_test_presorted

tied = st.lists(st.integers(-5, 5).map(float), min_size=1, max_size=40)
categories = st.one_of(st.sampled_from(["a", "b", "c", "d"]), st.integers(0, 2), st.none())


def test_strict_numeric_keeps_numbers_and_blanks_everything_else() -> None:
    mixed = np.array(
        [1, 2.5, "3.5", None, True, np.int64(4), np.float64(5.5), math.nan], dtype=object
    )
    np.testing.assert_array_equal(
        strict_numeric(mixed), [1.0, 2.5, np.nan, np.nan, np.nan, 4.0, 5.5, np.nan]
    )
    np.testing.assert_array_equal(strict_numeric(np.array([1, 2])), [1.0, 2.0])
    assert np.isnan(strict_numeric(np.array(["1", "2"]))).all()
    assert np.isnan(strict_numeric(np.array([True, False]))).all()


@given(tied, tied)
def test_presorted_ks_equals_the_plain_ks(sample: list[float], reference: list[float]) -> None:
    expected = ks_test(sample, reference)
    result = ks_test_presorted(sample, np.sort(np.array(reference)))
    assert result.statistic == pytest.approx(expected.statistic, abs=1e-12)
    assert result.pvalue == pytest.approx(expected.pvalue, abs=1e-12)


def test_presorted_ks_sees_a_gap_just_before_a_sample_point() -> None:
    reference = np.arange(1.0, 11.0)
    assert ks_test_presorted([10.0], reference).statistic == pytest.approx(0.9)
    assert ks_test_presorted([1.0], reference).statistic == pytest.approx(0.9)


@pytest.mark.parametrize("sample", [[], [1.0, math.nan], [[1.0]]])
def test_presorted_ks_rejects_invalid_samples(sample: list[float]) -> None:
    with pytest.raises(ValueError, match="must"):
        ks_test_presorted(sample, np.array([1.0, 2.0]))


@given(st.lists(categories, min_size=1, max_size=40), st.lists(categories, max_size=40))
def test_category_codes_count_like_category_counts(reference: list, batch: list) -> None:
    vocabulary = ("a", "b", 0, 1)
    for values in (reference, batch, [*batch, math.nan, "unseen"]):
        counted = np.bincount(category_codes(values, vocabulary), minlength=len(vocabulary) + 2)
        np.testing.assert_array_equal(counted, category_counts(values, vocabulary))


def test_a_reference_scored_against_itself_is_at_distance_zero() -> None:
    rng = np.random.default_rng(0)
    numeric = rng.normal(size=2_000)
    categorical = rng.choice(np.array(["a", "b", "c"], dtype=object), 2_000)
    for values, kind in ((numeric, "numeric"), (categorical, "categorical")):
        reference = fit_reference("f", values, kind)
        scores = score_feature(reference, values)
        assert set(CALIBRATED_DETECTORS[kind]) <= set(scores)
        for name in CALIBRATED_DETECTORS[kind]:
            assert scores[name] == pytest.approx(0.0, abs=1e-12)
        assert scores["chi2_pvalue"] == pytest.approx(1.0)
    assert "ks" not in score_feature(fit_reference("f", categorical, "categorical"), categorical)


def test_numeric_reference_keeps_edges_counts_and_sorted_values() -> None:
    values = np.array([3.0, 1.0, math.nan, 2.0, 4.0])
    reference = fit_reference("f", values, "numeric", n_bins=2)
    np.testing.assert_array_equal(reference.edges, [2.5])
    np.testing.assert_array_equal(reference.counts, [2, 0, 2, 1])
    np.testing.assert_array_equal(reference.sorted_values, [1.0, 2.0, 3.0, 4.0])


def test_categorical_reference_caps_the_vocabulary() -> None:
    values = ["a"] * 5 + ["b"] * 3 + ["c"]
    reference = fit_reference("f", values, "categorical", max_categories=2)
    assert reference.vocabulary == ("a", "b")
    np.testing.assert_array_equal(reference.counts, [5, 3, 1, 0])


def test_text_in_a_numeric_feature_is_scored_as_missing_and_hidden_from_ks() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=4_000)
    reference = fit_reference("f", values, "numeric")
    batch = values[:1_000].astype(object)
    batch[:200] = [str(value) for value in batch[:200]]
    scores = score_feature(reference, batch)
    assert scores["linf"] == pytest.approx(0.2, abs=0.01)
    assert scores["ks"] < 0.06  # the 800 remaining numbers still follow the reference


def test_a_batch_without_any_number_has_no_ks_score() -> None:
    reference = fit_reference("f", np.arange(100.0), "numeric")
    scores = score_feature(reference, np.array([None, "x", math.nan], dtype=object))
    assert math.isnan(scores["ks"])
    assert math.isnan(scores["ks_pvalue"])
    assert scores["linf"] == pytest.approx(1.0)
    assert textbook_alarm("ks p<0.05", scores) is False


def test_textbook_rules_alarm_in_the_right_direction() -> None:
    quiet = {"chi2_pvalue": 0.5, "ks_pvalue": 0.5, "psi": 0.05, "linf": 0.005}
    loud = {"chi2_pvalue": 0.01, "ks_pvalue": 0.01, "psi": 0.3, "linf": 0.02}
    assert [textbook_alarm(rule, quiet) for rule in TEXTBOOK_RULES] == [False] * 5
    assert [textbook_alarm(rule, loud) for rule in TEXTBOOK_RULES] == [True] * 5
    assert textbook_alarm("psi>0.2", {"psi": 0.15}) is False
    assert textbook_alarm("psi>0.1", {"psi": 0.15}) is True
    assert textbook_alarm("ks p<0.05", {"psi": 0.3}) is None  # no KS for categorical features


def test_unknown_feature_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        fit_reference("f", [1.0, 2.0], "ordinal")  # type: ignore[arg-type]
