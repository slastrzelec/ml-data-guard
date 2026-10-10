"""Fault injectors: purity, determinism, monotone severity and the meaning of every fault."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal, assert_series_equal

from mlguard import inject
from mlguard.distances import bin_counts, linf_distance, normalize, quantile_edges
from mlguard.inject import FAULTS

SPEC_CATALOGUE = {
    "pinned_value", "dropped_feature", "type_change", "new_category", "out_of_domain_numeric",
    "missing_rate_increase", "unit_change", "mean_shift", "variance_change", "duplicate_rows",
    "time_travel", "tiny_noise",
}  # fmt: skip

FEATURES_BY_TARGET = {
    "numeric": ["x", "k"],
    "categorical": ["c", "k"],
    "any": ["x", "k", "c"],
    "rows": [None],
}
CASES = [
    (name, feature)
    for name, fault in FAULTS.items()
    for feature in FEATURES_BY_TARGET[fault.target]
]
FRACTION_CASES = [case for case in CASES if FAULTS[case[0]].severity_unit == "fraction_of_rows"]

# The generic properties run once per (fault, feature) pair, so each of them gets fewer examples.
GENERIC = settings(max_examples=25)
SPECIFIC = settings(max_examples=100)


@st.composite
def batches(draw: st.DrawFn, *, min_rows: int = 3, max_rows: int = 30) -> pd.DataFrame:
    """A small mixed-type batch with missing values and a non-default index.

    Hypothesis chooses the size and a seed; NumPy fills the columns, which is an order of magnitude
    faster than drawing every cell (the price is weaker shrinking of failing examples).
    """
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    x = np.round(rng.normal(0.0, 100.0, size=n), 2)
    x[rng.random(n) < 0.2] = np.nan
    x[rng.random(n) < 0.1] = 0.0
    x[0], x[1] = 0.5, 7.25  # never constant, never all missing
    k = rng.integers(0, 6, size=n)
    k[0], k[1] = 1, 4
    c = rng.choice(np.array(["a", "b", "c", None], dtype=object), size=n)
    c[0], c[1] = "a", "b"
    return pd.DataFrame(
        {
            "x": x,
            "k": k.astype(np.int64),
            "c": c,
            "g": rng.choice(np.array(["s1", "s2"], dtype=object), size=n),
            "t": np.arange(n, dtype=np.int64),
        },
        index=pd.Index(rng.permutation(np.arange(100, 100 + n)), name="row_id"),
    )


def severities(name: str, *, positive: bool = False) -> st.SearchStrategy[float]:
    unit = FAULTS[name].severity_unit
    if unit == "fraction_of_rows":
        return st.floats(min_value=0.01 if positive else 0.0, max_value=1.0)
    if unit == "lead_steps":
        return st.integers(min_value=1 if positive else 0, max_value=6)
    if unit == "binary":
        return st.just(1) if positive else st.sampled_from([0, 1])
    return st.floats(min_value=0.05 if positive else 0.0, max_value=3.0)


def run(
    name: str, batch: pd.DataFrame, feature: str | None, severity: float, seed: int
) -> pd.DataFrame:
    return FAULTS[name].apply(batch, feature, severity, np.random.default_rng(seed))


def _equal(u: object, v: object) -> bool:
    """Two missing values are equal; a number never equals its text form."""
    if inject._is_missing(u) or inject._is_missing(v):
        return inject._is_missing(u) and inject._is_missing(v)
    if isinstance(u, str) or isinstance(v, str):
        return isinstance(u, str) and isinstance(v, str) and u == v
    return bool(u == v)


def same(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Element-wise equality of two columns, see ``_equal``."""
    pairs = zip(left.to_numpy(dtype=object), right.to_numpy(dtype=object), strict=True)
    return np.array([_equal(u, v) for u, v in pairs], dtype=bool)


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, feature: str | None) -> np.ndarray:
    columns = [feature] if feature is not None else list(before.columns)
    unchanged = np.ones(len(before), dtype=bool)
    for column in columns:
        unchanged &= same(before[column], after[column])
    return ~unchanged


# --------------------------------------------------------------------------- catalogue


def test_registry_is_the_catalogue_of_the_spec() -> None:
    assert set(FAULTS) == SPEC_CATALOGUE
    assert [name for name, fault in FAULTS.items() if fault.negative_control] == ["tiny_noise"]
    assert [name for name, fault in FAULTS.items() if fault.needs_time_order] == ["time_travel"]
    assert [name for name, fault in FAULTS.items() if not fault.preserves_shape] == [
        "dropped_feature"
    ]


# --------------------------------------------------------------------------- properties of all


@pytest.mark.parametrize(("name", "feature"), CASES)
@GENERIC
@given(data=st.data())
def test_input_is_not_modified_and_output_is_deterministic(
    name: str, feature: str | None, data: st.DataObject
) -> None:
    batch = data.draw(batches())
    severity = data.draw(severities(name))
    seed = data.draw(st.integers(0, 2**32 - 1))
    snapshot = batch.copy(deep=True)

    first = run(name, batch, feature, severity, seed)
    second = run(name, batch, feature, severity, seed)

    assert_frame_equal(batch, snapshot)
    assert_frame_equal(first, second)
    for column in first.columns:
        if first[column].dtype != object:
            assert not np.shares_memory(first[column].to_numpy(), batch[column].to_numpy())


@pytest.mark.parametrize(("name", "feature"), CASES)
@GENERIC
@given(data=st.data())
def test_zero_severity_returns_an_equal_copy(
    name: str, feature: str | None, data: st.DataObject
) -> None:
    batch = data.draw(batches())
    result = run(name, batch, feature, 0, data.draw(st.integers(0, 2**32 - 1)))
    assert result is not batch
    assert_frame_equal(result, batch)


@pytest.mark.parametrize(("name", "feature"), CASES)
@GENERIC
@given(data=st.data())
def test_positive_severity_changes_the_batch(
    name: str, feature: str | None, data: st.DataObject
) -> None:
    """A batch labelled as faulty must not be silently clean."""
    batch = data.draw(batches())
    severity = data.draw(severities(name, positive=True))
    result = run(name, batch, feature, severity, data.draw(st.integers(0, 2**32 - 1)))
    if name == "dropped_feature":
        assert feature not in result.columns
    elif name in {"tiny_noise", "duplicate_rows", "time_travel"}:
        pass  # may legitimately write a value equal to the old one
    else:
        assert changed_rows(batch, result, feature).any()


@pytest.mark.parametrize(("name", "feature"), CASES)
@GENERIC
@given(data=st.data())
def test_index_columns_and_other_features_are_preserved(
    name: str, feature: str | None, data: st.DataObject
) -> None:
    batch = data.draw(batches())
    severity = data.draw(severities(name))
    result = run(name, batch, feature, severity, data.draw(st.integers(0, 2**32 - 1)))

    assert result.index.equals(batch.index)
    assert result.index.name == batch.index.name
    if FAULTS[name].preserves_shape:
        assert list(result.columns) == list(batch.columns)
        assert result.shape == batch.shape
    else:
        assert [column for column in batch.columns if column != feature or severity == 0] == list(
            result.columns
        )
    if feature is not None:
        others = [column for column in batch.columns if column != feature]
        assert_frame_equal(result[others], batch[others])


@pytest.mark.parametrize(("name", "feature"), FRACTION_CASES)
@GENERIC
@given(data=st.data())
def test_changes_are_nested_in_severity(
    name: str, feature: str | None, data: st.DataObject
) -> None:
    """With one seed, whatever a small severity breaks stays broken the same way at a larger one."""
    batch = data.draw(batches())
    low, high = sorted(data.draw(st.tuples(severities(name), severities(name))))
    seed = data.draw(st.integers(0, 2**32 - 1))
    mild, strong = run(name, batch, feature, low, seed), run(name, batch, feature, high, seed)

    mild_rows, strong_rows = (
        changed_rows(batch, mild, feature),
        changed_rows(batch, strong, feature),
    )
    assert not (mild_rows & ~strong_rows).any()
    assert mild_rows.sum() <= strong_rows.sum()
    columns = [feature] if feature is not None else list(batch.columns)
    for column in columns:
        assert same(mild[column][mild_rows], strong[column][mild_rows]).all()


# --------------------------------------------------------------------------- meaning of each fault


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_pinned_value_sets_the_requested_fraction_to_the_sentinel(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    result = run("pinned_value", batch, "x", severity, seed)
    rows = changed_rows(batch, result, "x")
    assert (result["x"][rows] == -1).all()
    assert rows.sum() == max(1, math.ceil(severity * len(batch) - 1e-9))
    assert result["x"].dtype == np.float64


def test_pinned_value_only_counts_rows_that_are_not_pinned_already() -> None:
    batch = pd.DataFrame({"x": [-1.0, -1.0, 3.0, 4.0, 5.0, 6.0]})
    result = run("pinned_value", batch, "x", 0.5, seed=0)
    assert (result["x"] == -1).sum() == 4  # 2 already pinned + half of the 4 others


def test_pinned_value_on_text_keeps_other_values_untouched() -> None:
    batch = pd.DataFrame({"c": ["a", "b", "c", "d"]})
    result = FAULTS["pinned_value"].apply(
        batch, "c", 1.0, np.random.default_rng(0), sentinel="unknown"
    )
    assert result["c"].tolist() == ["unknown"] * 4
    mixed = run("pinned_value", batch, "c", 0.5, seed=0)
    assert sorted(map(str, mixed["c"].tolist())).count("-1") == 2
    assert mixed["c"].dtype == object  # -1 stays a number, it is not turned into "-1"
    assert -1 in mixed["c"].tolist()


@SPECIFIC
@given(batches(), st.integers(0, 2**32 - 1))
def test_dropped_feature_removes_exactly_one_column(batch: pd.DataFrame, seed: int) -> None:
    result = run("dropped_feature", batch, "x", 1, seed)
    assert list(result.columns) == ["k", "c", "g", "t"]


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_type_change_turns_numbers_into_their_text_form(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    result = run("type_change", batch, "x", severity, seed)
    is_text = np.array([isinstance(value, str) for value in result["x"]])
    present = batch["x"].notna().to_numpy()
    assert is_text.sum() == max(1, math.ceil(severity * present.sum() - 1e-9))
    assert not (is_text & ~present).any()  # missing values are not turned into "nan"
    assert [float(value) for value in result["x"][is_text]] == batch["x"][is_text].tolist()
    assert result["x"][~is_text].tolist() == pytest.approx(
        batch["x"][~is_text].tolist(), nan_ok=True
    )


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_new_category_was_never_in_the_batch(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    for feature, expected in (("c", "__new__"), ("k", int(batch["k"].max()) + 1)):
        result = run("new_category", batch, feature, severity, seed)
        rows = changed_rows(batch, result, feature)
        assert rows.sum() == max(1, math.ceil(severity * len(batch) - 1e-9))
        assert (result[feature][rows] == expected).all()
        assert expected not in batch[feature].tolist()
    assert run("new_category", batch, "k", severity, seed)["k"].dtype == np.int64


def test_new_category_accepts_an_explicit_new_name() -> None:
    batch = pd.DataFrame({"c": ["a", "b", "a", "b"]})
    result = FAULTS["new_category"].apply(batch, "c", 0.5, np.random.default_rng(0), category="z")
    assert sorted(result["c"].tolist()).count("z") == 2


def test_new_category_avoids_a_name_that_is_already_taken() -> None:
    batch = pd.DataFrame({"c": ["__new__", "a", "b"]})
    result = run("new_category", batch, "c", 1.0, seed=0)
    assert result["c"].tolist() == ["__new___"] * 3


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_out_of_domain_values_are_above_the_batch_maximum(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    for feature in ("x", "k"):
        result = run("out_of_domain_numeric", batch, feature, severity, seed)
        rows = changed_rows(batch, result, feature)
        present = batch[feature].notna()
        low, high = batch[feature].min(), batch[feature].max()
        assert rows.sum() == max(1, math.ceil(severity * present.sum() - 1e-9))
        assert (result[feature][rows] > high).all()
        assert (result[feature][rows] <= high + (high - low) + 1).all()
        assert not (rows & ~present.to_numpy()).any()
        assert result[feature].dtype == batch[feature].dtype


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_missing_rate_increase_adds_the_requested_number_of_missing_values(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    for feature in ("x", "k", "c"):
        before = int(batch[feature].isna().sum())
        result = run("missing_rate_increase", batch, feature, severity, seed)
        added = max(1, math.ceil(severity * (len(batch) - before) - 1e-9))
        assert int(result[feature].isna().sum()) == before + added
        kept = result[feature].notna().to_numpy()
        assert same(result[feature][kept], batch[feature][kept]).all()


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_unit_change_multiplies_the_requested_fraction_by_the_factor(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    for feature in ("x", "k"):
        result = run("unit_change", batch, feature, severity, seed)
        rows = changed_rows(batch, result, feature)
        eligible = batch[feature].notna() & (batch[feature] != 0)
        assert rows.sum() == max(1, math.ceil(severity * eligible.sum() - 1e-9))
        np.testing.assert_allclose(result[feature][rows], batch[feature][rows] * 1000)
        assert result[feature].dtype == batch[feature].dtype


@SPECIFIC
@given(batches(), st.floats(0.0, 3.0), st.integers(0, 2**32 - 1))
def test_mean_shift_moves_the_mean_by_severity_sigmas_and_keeps_the_spread(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    result = run("mean_shift", batch, "x", severity, seed)
    std = batch["x"].std(ddof=0)
    assert result["x"].mean() - batch["x"].mean() == pytest.approx(
        severity * std, rel=1e-9, abs=1e-9
    )
    assert result["x"].std(ddof=0) == pytest.approx(std, rel=1e-9, abs=1e-9)
    assert result["x"].isna().equals(batch["x"].isna())


@SPECIFIC
@given(batches(), st.floats(0.0, 3.0), st.integers(0, 2**32 - 1))
def test_variance_change_scales_the_spread_and_keeps_the_mean(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    result = run("variance_change", batch, "x", severity, seed)
    std = batch["x"].std(ddof=0)
    assert result["x"].std(ddof=0) == pytest.approx((1 + severity) * std, rel=1e-9)
    assert result["x"].mean() == pytest.approx(batch["x"].mean(), rel=1e-9, abs=1e-9)


@SPECIFIC
@given(batches(), st.floats(0.01, 3.0), st.integers(0, 2**32 - 1))
def test_shifts_keep_integer_features_integer_by_rounding_to_a_neighbour(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    mean, std = batch["k"].mean(), batch["k"].std(ddof=0)
    exact = {
        "mean_shift": batch["k"] + severity * std,
        "variance_change": mean + (batch["k"] - mean) * (1 + severity),
    }
    for name, target in exact.items():
        result = run(name, batch, "k", severity, seed)
        assert result["k"].dtype == np.int64
        assert ((result["k"] - target).abs() < 1 + 1e-9).all()
        assert not result["k"].equals(batch["k"])


def test_random_rounding_of_integer_features_is_unbiased() -> None:
    batch = pd.DataFrame({"k": np.random.default_rng(0).poisson(4.0, 50_000)})
    std = batch["k"].std(ddof=0)
    shifted = run("mean_shift", batch, "k", 0.01, seed=1)
    assert shifted["k"].mean() - batch["k"].mean() == pytest.approx(0.01 * std, abs=0.004)
    assert (shifted["k"] != batch["k"]).mean() == pytest.approx(0.01 * std, abs=0.004)
    stretched = run("variance_change", batch, "k", 0.25, seed=1)
    assert stretched["k"].std(ddof=0) == pytest.approx(1.25 * std, rel=0.02)
    assert stretched["k"].mean() == pytest.approx(batch["k"].mean(), abs=0.02)


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_duplicate_rows_copies_other_rows_of_the_batch(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    result = run("duplicate_rows", batch, None, severity, seed)
    # "t" is unique per row, so it identifies the source row of every output row.
    sources = result["t"].to_numpy()
    copied = sources != batch["t"].to_numpy()
    assert copied.sum() == max(1, math.ceil(severity * len(batch) - 1e-9))
    assert_frame_equal(result.reset_index(drop=True), batch.iloc[sources].reset_index(drop=True))
    assert [str(dtype) for dtype in result.dtypes] == [str(dtype) for dtype in batch.dtypes]


@SPECIFIC
@given(batches(), st.integers(1, 6), st.integers(0, 2**32 - 1))
def test_time_travel_reads_the_value_from_the_future(
    batch: pd.DataFrame, lead: int, seed: int
) -> None:
    result = run("time_travel", batch, "x", lead, seed)
    n = len(batch)
    expected = batch["x"].to_numpy().copy()
    if lead < n:
        expected[: n - lead] = batch["x"].to_numpy()[lead:]
    np.testing.assert_array_equal(result["x"].to_numpy(), expected)


@SPECIFIC
@given(batches(), st.integers(1, 6), st.integers(0, 2**32 - 1))
def test_time_travel_never_crosses_a_group_boundary(
    batch: pd.DataFrame, lead: int, seed: int
) -> None:
    result = FAULTS["time_travel"].apply(
        batch, "t", lead, np.random.default_rng(seed), group_by="g"
    )
    for _, group in batch.assign(leaked=result["t"]).groupby("g", sort=False):
        own, leaked = group["t"].to_numpy(), group["leaked"].to_numpy()
        expected = own.copy()
        if lead < len(own):
            expected[: len(own) - lead] = own[lead:]
        np.testing.assert_array_equal(leaked, expected)


def test_time_travel_leaves_the_marginal_distribution_almost_unchanged() -> None:
    """Why distribution distances are not expected to see this fault (SPEC section 5)."""
    rng = np.random.default_rng(3)
    batch = pd.DataFrame({"x": rng.normal(size=5000).cumsum()})
    leaked = run("time_travel", batch, "x", 1, seed=0)
    edges = quantile_edges(batch["x"])
    p = normalize(bin_counts(batch["x"], edges))
    q = normalize(bin_counts(leaked["x"], edges))
    assert linf_distance(p, q) < 0.001
    assert not np.array_equal(batch["x"].to_numpy(), leaked["x"].to_numpy())


@SPECIFIC
@given(batches(), st.floats(0.01, 1.0), st.integers(0, 2**32 - 1))
def test_tiny_noise_stays_inside_the_domain_of_the_feature(
    batch: pd.DataFrame, severity: float, seed: int
) -> None:
    for feature in ("k", "c"):
        result = run("tiny_noise", batch, feature, severity, seed)
        rows = changed_rows(batch, result, feature)
        assert rows.sum() <= max(1, math.ceil(severity * len(batch) - 1e-9))
        allowed = batch[feature].to_numpy(dtype=object)
        assert all(any(_equal(value, a) for a in allowed) for value in result[feature])
    jittered = run("tiny_noise", batch, "x", severity, seed)
    assert jittered["x"].dropna().between(batch["x"].min(), batch["x"].max()).all()
    moved = changed_rows(batch, jittered, "x")
    spread = float(np.nanstd(batch["x"]))
    # every new value is a value of the batch moved by a small jitter
    originals = batch["x"].dropna().to_numpy()
    for value in jittered["x"][moved].dropna():
        assert np.abs(originals - value).min() <= spread


def test_tiny_noise_does_not_leave_the_domain_of_a_bounded_feature() -> None:
    rng = np.random.default_rng(2)
    batch = pd.DataFrame({"x": np.where(rng.random(5_000) < 0.3, 0.0, rng.gamma(2.0, 3.0, 5_000))})
    result = run("tiny_noise", batch, "x", 0.2, seed=0)
    assert (result["x"] != batch["x"]).sum() > 500
    assert result["x"].min() >= 0.0
    assert result["x"].max() <= batch["x"].max()


def test_tiny_noise_jitter_is_a_tenth_of_a_standard_deviation() -> None:
    batch = pd.DataFrame({"x": np.random.default_rng(4).normal(0.0, 1.0, 50_000)})
    result = run("tiny_noise", batch, "x", 1.0, seed=0)
    # resampling keeps the spread, a 0.1 sigma jitter adds 0.5% to it
    assert result["x"].std() == pytest.approx(batch["x"].std() * np.sqrt(1.01), rel=0.01)


def test_tiny_noise_at_the_spec_level_touches_one_row_in_ten_thousand() -> None:
    batch = pd.DataFrame({"x": np.random.default_rng(1).normal(size=20_000)})
    result = run("tiny_noise", batch, "x", 1e-4, seed=0)
    assert (result["x"] != batch["x"]).sum() == 2


# --------------------------------------------------------------------------- with the distances


def test_pinning_a_fifth_of_the_rows_is_an_linf_distance_of_a_fifth() -> None:
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"x": rng.uniform(0, 10, 20_000)})
    batch = pd.DataFrame({"x": rng.uniform(0, 10, 5_000)})
    faulty = run("pinned_value", batch, "x", 0.2, seed=0)
    edges = quantile_edges(reference["x"])
    p = normalize(bin_counts(reference["x"], edges))
    assert linf_distance(p, normalize(bin_counts(batch["x"], edges))) < 0.02
    assert linf_distance(p, normalize(bin_counts(faulty["x"], edges))) == pytest.approx(
        0.2, abs=0.02
    )


# --------------------------------------------------------------------------- text dtypes


def test_a_string_column_keeps_its_dtype_while_it_holds_only_text() -> None:
    batch = pd.DataFrame({"c": pd.Series(["a", "b", "c", "d"], dtype="str")})
    assert run("new_category", batch, "c", 0.5, seed=0)["c"].dtype == batch["c"].dtype
    assert run("missing_rate_increase", batch, "c", 0.5, seed=0)["c"].dtype == batch["c"].dtype
    assert run("time_travel", batch, "c", 1, seed=0)["c"].dtype == batch["c"].dtype
    assert run("pinned_value", batch, "c", 0.5, seed=0)["c"].dtype == object


def test_a_categorical_column_becomes_object() -> None:
    batch = pd.DataFrame({"c": pd.Categorical(["a", "b", "a", "b"])})
    result = run("new_category", batch, "c", 0.5, seed=0)
    assert result["c"].dtype == object
    assert (result["c"] == "__new__").sum() == 2
    assert_series_equal(batch["c"], pd.Series(pd.Categorical(["a", "b", "a", "b"]), name="c"))


# --------------------------------------------------------------------------- invalid input


@pytest.mark.parametrize(("name", "feature"), CASES)
def test_invalid_batches_are_rejected(name: str, feature: str | None) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="DataFrame"):
        FAULTS[name].apply({"x": [1.0]}, feature, 0.5, rng)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"empty|at least two rows"):
        FAULTS[name].apply(pd.DataFrame({"x": [], "k": [], "c": []}), feature, 0.5, rng)
    if feature is not None:
        with pytest.raises(ValueError, match="no feature"):
            FAULTS[name].apply(pd.DataFrame({"other": [1.0, 2.0]}), feature, 0.5, rng)
        with pytest.raises(ValueError, match="needs a feature"):
            FAULTS[name].apply(pd.DataFrame({"x": [1.0, 2.0]}), None, 0.5, rng)


@pytest.mark.parametrize(("name", "feature"), CASES)
@pytest.mark.parametrize("severity", [-0.1, math.nan, math.inf, "0.5", None])
def test_invalid_severities_are_rejected(name: str, feature: str | None, severity: object) -> None:
    batch = pd.DataFrame({"x": [1.0, 2.0, 3.0], "k": [1, 2, 3], "c": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="severity"):
        FAULTS[name].apply(batch, feature, severity, np.random.default_rng(0))  # type: ignore[arg-type]


@pytest.mark.parametrize(("name", "feature"), FRACTION_CASES)
def test_a_fraction_above_one_is_rejected(name: str, feature: str | None) -> None:
    batch = pd.DataFrame({"x": [1.0, 2.0, 3.0], "k": [1, 2, 3], "c": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="fraction"):
        FAULTS[name].apply(batch, feature, 1.01, np.random.default_rng(0))


def test_time_travel_needs_a_whole_number_of_steps() -> None:
    batch = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="whole number"):
        run("time_travel", batch, "x", 1.5, seed=0)
    with pytest.raises(ValueError, match="no feature"):
        FAULTS["time_travel"].apply(batch, "x", 1, np.random.default_rng(0), group_by="missing")


@pytest.mark.parametrize(
    "name", ["type_change", "out_of_domain_numeric", "unit_change", "mean_shift", "variance_change"]
)
def test_numeric_faults_reject_text_and_boolean_features(name: str) -> None:
    batch = pd.DataFrame({"c": ["a", "b", "c"], "b": [True, False, True]})
    for feature in ("c", "b"):
        with pytest.raises(ValueError, match="numeric"):
            run(name, batch, feature, 0.5, seed=0)


def test_new_category_rejects_float_features_and_existing_categories() -> None:
    batch = pd.DataFrame({"x": [1.0, 2.0], "c": ["a", "b"]})
    with pytest.raises(ValueError, match="categorical"):
        run("new_category", batch, "x", 0.5, seed=0)
    with pytest.raises(ValueError, match="already occurs"):
        FAULTS["new_category"].apply(batch, "c", 0.5, np.random.default_rng(0), category="a")


@pytest.mark.parametrize("name", ["mean_shift", "variance_change"])
def test_shifts_reject_constant_and_all_missing_features(name: str) -> None:
    batch = pd.DataFrame({"x": [2.0, 2.0, 2.0], "y": [math.nan] * 3})
    with pytest.raises(ValueError, match="constant"):
        run(name, batch, "x", 0.5, seed=0)
    with pytest.raises(ValueError, match="no non-missing"):
        run(name, batch, "y", 0.5, seed=0)


def test_out_of_domain_rejects_an_all_missing_feature_and_handles_a_constant_one() -> None:
    with pytest.raises(ValueError, match="no non-missing"):
        run("out_of_domain_numeric", pd.DataFrame({"x": [math.nan, math.nan]}), "x", 0.5, seed=0)
    result = run("out_of_domain_numeric", pd.DataFrame({"x": [0.0, 0.0, 0.0]}), "x", 1.0, seed=0)
    assert (result["x"] > 0).all()


@pytest.mark.parametrize("factor", [0, 1, math.nan, math.inf])
def test_unit_change_rejects_factors_that_change_nothing(factor: float) -> None:
    batch = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="factor"):
        FAULTS["unit_change"].apply(batch, "x", 0.5, np.random.default_rng(0), factor=factor)


def test_unit_change_with_a_fractional_factor_makes_integers_float() -> None:
    batch = pd.DataFrame({"k": [10, 20, 30, 40]})
    result = FAULTS["unit_change"].apply(batch, "k", 1.0, np.random.default_rng(0), factor=0.5)
    assert result["k"].tolist() == [5.0, 10.0, 15.0, 20.0]
