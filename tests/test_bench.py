"""Benchmark runner: leakage rules, statistics, determinism and sanity of the results."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlguard import bench
from mlguard.bench import (
    BenchConfig,
    bootstrap_mean_interval,
    draw_batch,
    make_splits,
    render_report,
    run_benchmark,
    to_json,
    wilson_interval,
)
from mlguard.datasets import Dataset, make_synthetic, make_synthetic_series
from mlguard.detectors import TEXTBOOK_RULES

FRACTIONS = (0.40, 0.25, 0.25, 0.10)
SMALL = BenchConfig(
    dataset="synthetic",
    dataset_rows=120_000,
    seed=7,
    batch_sizes=(300,),
    n_calibration_batches=400,
    n_clean_batches=400,
    n_seeds=15,
    model_batch_size=1_000,
    model_max_iter=20,
)
TINY = dataclasses.replace(
    SMALL,
    dataset_rows=8_000,
    batch_sizes=(100,),
    n_calibration_batches=100,
    n_clean_batches=20,
    n_seeds=2,
    faults=("pinned_value", "new_category", "duplicate_rows", "dropped_feature"),
    model_batch_size=200,
    model_max_iter=5,
)


@pytest.fixture(scope="module")
def result() -> dict:
    return run_benchmark(SMALL)


def faults_of(result: dict, name: str) -> dict[float, dict]:
    return {
        entry["severity"]: entry for entry in result["runs"][0]["faults"] if entry["fault"] == name
    }


def corrupted(dataset: Dataset, rows: np.ndarray) -> Dataset:
    """Copy of the dataset in which the given rows carry obviously different data."""
    frame = dataset.frame.copy(deep=True)
    for feature in dataset.numeric:
        column = frame[feature].to_numpy(dtype=np.float64, copy=True)
        column[rows] = column[rows] * 1000.0 + 12345.0
        frame[feature] = column
    for feature in dataset.categorical:
        column = frame[feature].to_numpy(dtype=object, copy=True)
        column[rows] = "CORRUPTED"
        frame[feature] = column
    target = frame[dataset.target].to_numpy(copy=True)
    target[rows] = 1 - target[rows]
    frame[dataset.target] = target
    return dataclasses.replace(dataset, frame=frame)


# --------------------------------------------------------------------------- splits and batches


@given(st.integers(40, 5_000), st.integers(0, 2**32 - 1), st.booleans())
def test_splits_are_disjoint_complete_and_sized_as_requested(
    n_rows: int, seed: int, time_ordered: bool
) -> None:
    splits = make_splits(n_rows, FRACTIONS, seed=seed, time_ordered=time_ordered)
    parts = [splits.reference, splits.calibration, splits.evaluation, splits.held_out]
    together = np.concatenate(parts)
    assert together.size == n_rows
    np.testing.assert_array_equal(np.sort(together), np.arange(n_rows))
    for part, fraction in zip(parts, FRACTIONS, strict=True):
        assert abs(part.size - fraction * n_rows) <= 1
    if time_ordered:
        np.testing.assert_array_equal(together, np.arange(n_rows))


def test_unordered_splits_are_shuffled_and_depend_on_the_seed() -> None:
    first = make_splits(1_000, FRACTIONS, seed=1, time_ordered=False)
    again = make_splits(1_000, FRACTIONS, seed=1, time_ordered=False)
    other = make_splits(1_000, FRACTIONS, seed=2, time_ordered=False)
    np.testing.assert_array_equal(first.reference, again.reference)
    assert not np.array_equal(first.reference, other.reference)
    assert not np.array_equal(first.reference, np.arange(first.reference.size))


def test_too_small_datasets_are_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        make_splits(3, FRACTIONS, seed=0, time_ordered=False)


@given(st.integers(1, 200), st.integers(0, 2**32 - 1), st.booleans())
def test_a_batch_is_a_set_of_distinct_rows_of_its_pool(
    batch_size: int, seed: int, time_ordered: bool
) -> None:
    pool = np.arange(1_000, 1_200)
    rows = draw_batch(pool, batch_size, np.random.default_rng(seed), time_ordered=time_ordered)
    assert rows.size == batch_size == np.unique(rows).size
    assert np.isin(rows, pool).all()
    if time_ordered:
        np.testing.assert_array_equal(rows, np.arange(rows[0], rows[0] + batch_size))


def test_a_batch_larger_than_its_pool_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        draw_batch(np.arange(10), 11, np.random.default_rng(0), time_ordered=False)


# --------------------------------------------------------------------------- statistics


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    low, high = wilson_interval(0, 100)
    assert low == 0.0
    assert high == pytest.approx(0.0370, abs=1e-4)
    assert wilson_interval(100, 100)[1] == 1.0
    assert all(np.isnan(wilson_interval(0, 0)))


@given(st.integers(1, 500), st.data())
def test_wilson_interval_contains_the_observed_rate(trials: int, data: st.DataObject) -> None:
    successes = data.draw(st.integers(0, trials))
    low, high = wilson_interval(successes, trials)
    assert 0.0 <= low <= successes / trials <= high <= 1.0


def test_bootstrap_interval_brackets_the_mean_and_is_seeded() -> None:
    values = np.random.default_rng(0).normal(0.3, 1.0, 400)
    low, high = bootstrap_mean_interval(values, np.random.default_rng(1))
    assert low < values.mean() < high
    assert high - low == pytest.approx(4 * values.std() / np.sqrt(400), rel=0.15)
    assert (low, high) == bootstrap_mean_interval(values, np.random.default_rng(1))


# --------------------------------------------------------------------------- configuration


def test_config_round_trips_through_json() -> None:
    config = dataclasses.replace(SMALL, severities={"mean_shift": (0.1, 0.2)})
    assert BenchConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.severities_of("mean_shift") == (0.1, 0.2)
    assert config.severities_of("time_travel") == (1, 2, 3, 5, 10)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": (0.5, 0.5, 0.0, 0.0)}, "split"),
        ({"split": (0.4, 0.3, 0.2)}, "split"),
        ({"target_fpr": 0.0}, "target_fpr"),
        ({"batch_sizes": ()}, "batch_sizes"),
        ({"n_seeds": 0}, "positive"),
        ({"n_calibration_batches": 50}, "too few calibration batches"),
        ({"faults": ("typo",)}, "unknown faults"),
        ({"severities": {"typo": (1,)}}, "unknown faults"),
    ],
)
def test_invalid_configs_are_rejected(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(SMALL, **change)


def test_config_files_with_unknown_keys_or_missing_data_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        BenchConfig.from_dict({"dataset": "synthetic", "batch_size": 10})
    with pytest.raises(ValueError, match="dataset_rows"):
        bench.load_dataset(BenchConfig(dataset="synthetic"))
    with pytest.raises(ValueError, match="data_path"):
        bench.load_dataset(BenchConfig(dataset="covertype"))
    with pytest.raises(ValueError, match="unknown dataset"):
        bench.load_dataset(BenchConfig(dataset="iris"))


def test_every_shipped_config_is_valid() -> None:
    configs = sorted(
        (Path(__file__).resolve().parents[1] / "benchmarks" / "configs").glob("*.json")
    )
    assert len(configs) >= 4
    for path in configs:
        BenchConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- leakage rules


def test_thresholds_do_not_depend_on_evaluation_or_held_out_rows() -> None:
    """Leakage rules 1, 3 and 5: corrupting the rows that must stay unseen changes nothing."""
    config = dataclasses.replace(TINY, model=False, faults=("tiny_noise",))
    dataset = make_synthetic(config.dataset_rows, config.seed)
    splits = make_splits(len(dataset.frame), config.split, seed=config.seed, time_ordered=False)
    unseen = np.concatenate([splits.evaluation, splits.held_out])
    baseline = run_benchmark(config, dataset=dataset)["runs"][0]
    changed = run_benchmark(config, dataset=corrupted(dataset, unseen))["runs"][0]
    assert changed["thresholds"] == baseline["thresholds"]
    assert changed["clean"] != baseline["clean"]  # the corruption is visible where it should be


def test_thresholds_do_depend_on_the_calibration_rows() -> None:
    config = dataclasses.replace(TINY, model=False, faults=("tiny_noise",))
    dataset = make_synthetic(config.dataset_rows, config.seed)
    splits = make_splits(len(dataset.frame), config.split, seed=config.seed, time_ordered=False)
    baseline = run_benchmark(config, dataset=dataset)["runs"][0]
    changed = run_benchmark(config, dataset=corrupted(dataset, splits.calibration))["runs"][0]
    assert changed["thresholds"] != baseline["thresholds"]


def test_reference_statistics_use_the_reference_split_only() -> None:
    dataset = make_synthetic(TINY.dataset_rows, TINY.seed)
    splits = make_splits(len(dataset.frame), TINY.split, seed=TINY.seed, time_ordered=False)
    others = np.concatenate([splits.calibration, splits.evaluation, splits.held_out])
    clean = bench._Scorer(dataset, splits.reference, TINY)
    dirty = bench._Scorer(corrupted(dataset, others), splits.reference, TINY)
    for feature in dataset.features:
        np.testing.assert_array_equal(
            clean.references[feature].counts, dirty.references[feature].counts
        )


def test_the_model_is_trained_on_the_reference_split(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    original = bench._Model

    class Spy(original):  # type: ignore[misc, valid-type]
        def __init__(self, scorer, reference_rows, config) -> None:  # type: ignore[no-untyped-def]
            seen["rows"] = reference_rows
            super().__init__(scorer, reference_rows, config)

    monkeypatch.setattr(bench, "_Model", Spy)
    run_benchmark(TINY)
    splits = make_splits(TINY.dataset_rows, TINY.split, seed=TINY.seed, time_ordered=False)
    np.testing.assert_array_equal(seen["rows"], splits.reference)


def test_clean_fast_path_gives_the_same_scores_as_scoring_raw_values() -> None:
    dataset = make_synthetic(6_000, seed=5)
    splits = make_splits(6_000, FRACTIONS, seed=5, time_ordered=False)
    scorer = bench._Scorer(dataset, splits.reference, TINY)
    rows = splits.evaluation[:500]
    for feature in dataset.features:
        raw = dataset.frame[feature].to_numpy()[rows]
        assert scorer.clean(feature, rows) == pytest.approx(
            scorer.faulty(feature, raw), nan_ok=True
        )


# --------------------------------------------------------------------------- results


def test_same_config_gives_a_byte_identical_result() -> None:
    assert to_json(run_benchmark(TINY)) == to_json(run_benchmark(TINY))


def test_a_different_seed_gives_a_different_result() -> None:
    other = dataclasses.replace(TINY, seed=TINY.seed + 1)
    assert to_json(run_benchmark(TINY)) != to_json(run_benchmark(other))


def test_result_is_plain_json_with_the_full_config(result: dict) -> None:
    text = to_json(result)
    assert json.loads(text) == result
    assert "NaN" not in text
    assert result["meta"]["config"] == SMALL.to_dict()
    assert result["meta"]["split_sizes"] == {
        "reference": 48_000, "calibration": 30_000, "evaluation": 30_000, "held_out": 12_000,
    }  # fmt: skip
    assert {"numpy", "pandas", "scipy", "scikit-learn", "python"} <= set(result["meta"]["versions"])


def test_clean_batches_stay_near_the_calibrated_false_positive_rate(result: dict) -> None:
    """Negative control `natural_daily_variation`: fresh clean batches rarely alarm."""
    per_feature = result["runs"][0]["clean"]["calibrated"]["per_feature"]
    for name, entry in per_feature.items():
        assert entry["n"] >= 2_000
        assert entry["rate"] <= 0.03, name
    per_batch = result["runs"][0]["clean"]["calibrated"]["per_batch"]
    for name, entry in per_batch.items():
        assert entry["rate"] >= per_feature[name]["rate"]


def test_tiny_noise_control_does_not_alarm(result: dict) -> None:
    control = faults_of(result, "tiny_noise")[0.0001]
    assert control["negative_control"]
    for name, entry in control["calibrated"].items():
        assert entry["rate"] <= 2 / 15, name


@pytest.mark.parametrize(
    ("fault", "severity"),
    [
        ("pinned_value", 0.2),
        ("unit_change", 0.2),
        ("new_category", 0.2),
        ("missing_rate_increase", 0.2),
        ("out_of_domain_numeric", 0.2),
        ("mean_shift", 0.5),
    ],
)
def test_strong_faults_are_found_by_linf(result: dict, fault: str, severity: float) -> None:
    assert faults_of(result, fault)[severity]["calibrated"]["linf"]["rate"] >= 0.9


def test_recall_does_not_fall_as_a_fault_gets_stronger(result: dict) -> None:
    entries = faults_of(result, "pinned_value")
    rates = [entries[severity]["calibrated"]["chi2"]["rate"] for severity in sorted(entries)]
    assert rates[-1] == 1.0
    assert rates[0] < rates[-1]
    assert all(later >= earlier - 0.2 for earlier, later in itertools.pairwise(rates))


def test_fault_levels_and_feature_breakdown_are_recorded(result: dict) -> None:
    duplicates = faults_of(result, "duplicate_rows")[0.2]
    assert duplicates["level"] == "batch"
    assert duplicates["by_feature"] == {}
    pinned = faults_of(result, "pinned_value")[0.2]
    assert pinned["level"] == "feature"
    assert sum(counts["linf"][1] for counts in pinned["by_feature"].values()) == SMALL.n_seeds
    assert set(pinned["by_feature"]) <= set(
        result["meta"]["features"]["numeric"] + result["meta"]["features"]["categorical"]
    )
    new_category = faults_of(result, "new_category")[0.2]
    assert new_category["calibrated"]["ks"] is None  # KS does not apply to categorical features
    assert set(new_category["textbook"]) == set(TEXTBOOK_RULES)


def test_a_dropped_feature_is_not_scorable_but_has_a_model_impact(result: dict) -> None:
    dropped = faults_of(result, "dropped_feature")[1]
    assert dropped["scorable"] is False
    assert all(entry is None for entry in dropped["calibrated"].values())
    impact = [entry for entry in result["model"]["faults"] if entry["fault"] == "dropped_feature"]
    assert len(impact) == 1
    assert impact[0]["accuracy_drop"] > 0


def test_model_part_reports_a_sensible_clean_accuracy_and_intervals(result: dict) -> None:
    model = result["model"]
    assert 0.6 < model["clean_accuracy"] < 0.9
    assert len(model["faults"]) == len(result["runs"][0]["faults"])
    for entry in model["faults"]:
        assert entry["low"] <= entry["accuracy_drop"] <= entry["high"]
    strongest = {entry["fault"]: entry for entry in model["faults"] if entry["severity"] == 0.2}
    assert strongest["unit_change"]["accuracy_drop"] > 0.005


def test_duplicated_rows_keep_their_labels_so_the_model_is_not_hurt(result: dict) -> None:
    duplicates = [
        entry for entry in result["model"]["faults"] if entry["fault"] == "duplicate_rows"
    ]
    assert len(duplicates) == 5
    for entry in duplicates:
        assert abs(entry["accuracy_drop"]) < 0.01


def test_time_travel_is_run_on_time_ordered_data_only(result: dict) -> None:
    assert faults_of(result, "time_travel") == {}


def test_time_travel_changes_the_model_but_not_the_marginal_distribution() -> None:
    """The expected result of SPEC section 5, kept as a regression test."""
    config = dataclasses.replace(
        SMALL,
        dataset="synthetic_series",
        dataset_rows=60_000,
        n_calibration_batches=200,
        n_clean_batches=50,
        n_seeds=20,
        faults=("time_travel",),
        severities={"time_travel": (1,)},
    )
    result = run_benchmark(config, dataset=make_synthetic_series(60_000, config.seed))
    entry = result["runs"][0]["faults"][0]
    # Shifting lag_1 or lag_mean_3 hands the model future information: its accuracy moves...
    assert abs(result["model"]["faults"][0]["accuracy_drop"]) > 0.03
    # ...while every distribution detector stays at the level of its false alarms.
    for name, recall in entry["calibrated"].items():
        assert recall["rate"] <= 0.2, name


def test_report_is_a_pure_function_of_the_result(result: dict) -> None:
    report = render_report(result)
    assert report == render_report(json.loads(to_json(result)))
    for heading in (
        "# Benchmark report: synthetic",
        "## Batch size 300",
        "### False alarms on clean batches",
        "### Recall, calibrated thresholds",
        "### Recall, textbook rules",
        "## Does the alert matter?",
        "## What this report does not measure",
    ):
        assert heading in report
    assert "`tiny_noise` (control)" in report
    assert "`dropped_feature`" in report
    assert "nan" not in report.lower()


def test_command_line_writes_result_and_report(tmp_path: Path) -> None:
    config = dataclasses.replace(TINY, model=False).to_dict()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out, report = tmp_path / "out" / "result.json", tmp_path / "out" / "report.md"
    code = bench.main(
        [
            "--config",
            str(config_path),
            "--out",
            str(out),
            "--report",
            str(report),
            "--commit",
            "abc",
        ]
    )
    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["meta"]["commit"] == "abc"
    assert written["model"] is None
    assert "Does the alert matter?" not in report.read_text(encoding="utf-8")
    assert "Commit: abc" in report.read_text(encoding="utf-8")
