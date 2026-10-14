"""The fault-injection demo (SPEC section 17.3): leakage rules, determinism, sensible answers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mlguard import demo
from mlguard.demo import DEMO_FAULTS, candidate_features, run_demo
from mlguard.inject import FAULTS


def test_the_three_parts_of_the_table_are_disjoint_and_complete():
    reference, calibration, evaluation = demo._parts(1_000, seed=5)
    rows = np.concatenate([reference, calibration, evaluation])

    assert len(rows) == len(set(rows.tolist())) == 1_000
    assert len(reference) == 400
    assert len(calibration) == 300


def test_thresholds_and_schema_do_not_depend_on_the_evaluation_part(monkeypatch):
    """Replace every evaluation row with garbage: limits and schema must not move."""
    demo._setup.cache_clear()
    before = demo._setup(500, 3)
    _, _, evaluation = demo._parts(demo.N_ROWS, 3)
    original = demo._table

    def garbage(seed: int):
        dataset = original(seed)
        frame = dataset.frame.copy()
        for name in dataset.numeric:
            frame.loc[evaluation, name] = 1e9
        for name in dataset.categorical:
            frame.loc[evaluation, name] = "garbage"
        return replace(dataset, frame=frame)

    monkeypatch.setattr(demo, "_table", garbage)
    demo._setup.cache_clear()
    after = demo._setup(500, 3)
    demo._setup.cache_clear()

    assert after.thresholds == before.thresholds
    assert after.schema == before.schema
    assert np.array_equal(after.evaluation_rows, before.evaluation_rows)


def test_the_same_request_gives_the_same_answer():
    first = run_demo("mean_shift", 0.1, batch_size=500, seed=4)
    demo._setup.cache_clear()
    second = run_demo("mean_shift", 0.1, batch_size=500, seed=4)

    assert first == second


def test_a_strong_fault_is_found_by_the_distances_and_the_schema():
    result = run_demo("unit_change", 0.2, batch_size=1_000, seed=2)
    verdicts = result["faulty"][result["feature"]]

    assert verdicts["schema"]["alarm"] is True
    assert all(
        verdicts[f"calibrated:{d}"]["alarm"] for d in ("linf", "l1", "kl", "js", "chi2", "ks")
    )


def test_a_legal_shift_is_found_by_the_distances_and_not_by_the_schema():
    result = run_demo("mean_shift", 0.5, batch_size=5_000, seed=0, feature="x_normal")
    verdicts = result["faulty"]["x_normal"]

    assert verdicts["calibrated:ks"]["alarm"] is True
    assert verdicts["schema"]["alarm"] is False


def test_the_negative_control_raises_few_alarms():
    alarms = 0
    for seed in range(5):
        result = run_demo("tiny_noise", 0.0001, batch_size=1_000, seed=seed)
        verdicts = result["faulty"][result["feature"]]
        alarms += sum(
            verdicts[f"calibrated:{d}"].get("alarm") is True for d in ("linf", "l1", "ks")
        )

    assert result["negative_control"] is True
    assert alarms <= 2


def test_a_dropped_column_cannot_be_scored_but_breaks_the_schema():
    result = run_demo("dropped_feature", 1, seed=1, feature="c_small")
    verdicts = result["faulty"]["c_small"]

    assert verdicts["calibrated:linf"] == {"applies": True, "alarm": None}
    assert verdicts["schema"]["alarm"] is True
    assert "MISSING_FEATURE" in result["schema_anomaly_types"]["faulty"]
    assert "MISSING_FEATURE" not in result["schema_anomaly_types"]["clean"]


def test_ks_does_not_apply_to_categorical_features():
    result = run_demo("new_category", 0.05, seed=0, feature="c_large")

    assert result["faulty"]["c_large"]["calibrated:ks"] == {"applies": False}
    assert result["faulty"]["c_large"]["textbook:ks p<0.05"] == {"applies": False}


def test_a_fault_on_rows_has_no_feature():
    result = run_demo("duplicate_rows", 0.2, seed=0)

    assert result["feature"] is None
    assert candidate_features("duplicate_rows") == ()


def test_the_answer_holds_no_rows():
    result = run_demo("new_category", 0.05, seed=0, feature="c_small")
    text = repr(result)

    assert "k01" not in text
    assert "'a'" not in text


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("time_travel", 1), "unknown fault"),
        (("unit_change", 1.5), "between"),
        (("unit_change", float("nan")), "between"),
        (("mean_shift", -0.1), "between"),
        (("dropped_feature", 0.5), "between"),
    ],
)
def test_invalid_requests_are_refused(arguments, message):
    with pytest.raises(ValueError, match=message):
        run_demo(*arguments)


def test_invalid_batch_size_seed_and_feature_are_refused():
    with pytest.raises(ValueError, match="batch size"):
        run_demo("unit_change", 0.1, batch_size=7)
    with pytest.raises(ValueError, match="seed"):
        run_demo("unit_change", 0.1, seed=-1)
    with pytest.raises(ValueError, match="cannot act"):
        run_demo("unit_change", 0.1, feature="c_small")
    with pytest.raises(ValueError, match="whole rows"):
        run_demo("duplicate_rows", 0.1, feature="x_count")


def test_every_offered_fault_runs():
    for name in DEMO_FAULTS:
        low, high = demo.SEVERITY_RANGES[FAULTS[name].severity_unit]
        result = run_demo(name, high if low == high else min(high, 0.2), batch_size=500, seed=0)
        assert set(result["clean"]) == set(result["features"])
