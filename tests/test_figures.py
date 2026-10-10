"""The numbers behind the README figures come straight from the result files (SPEC 16.6).

Only the selection of the numbers is tested. Drawing needs matplotlib, which the tests do not
install.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("make_figures", REPO / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_figures)


def _stored(rate: float) -> dict[str, float]:
    return {"k": 0, "n": 100, "rate": rate, "low": rate - 0.01, "high": rate + 0.01}


def _result() -> dict:
    def fault(name: str, severity: float, linf: float | None) -> dict:
        return {
            "fault": name,
            "severity": severity,
            "calibrated": {"linf": None if linf is None else _stored(linf)},
            "textbook": {"schema": _stored(1.0)},
        }

    return {
        "model": {
            "faults": [
                {"fault": "unit_change", "severity": 0.05, "accuracy_drop": 0.011, "low": 0.009,
                 "high": 0.012},
                {"fault": "unit_change", "severity": 0.01, "accuracy_drop": -0.002, "low": -0.003,
                 "high": -0.001},
            ]
        },
        "runs": [
            {"batch_size": 1000, "clean": {}, "faults": []},
            {
                "batch_size": 10000,
                "clean": {
                    "calibrated": {"per_feature": {"linf": _stored(0.02)},
                                   "per_batch": {"linf": _stored(0.3)}},
                },
                "faults": [
                    fault("unit_change", 0.05, 0.9),
                    fault("unit_change", 0.01, 0.4),
                    fault("mean_shift", 0.5, 0.7),
                    fault("dropped_feature", 1, None),
                ],
            },
        ],
    }  # fmt: skip


def test_the_run_is_chosen_by_batch_size():
    assert make_figures.run_at(_result(), 10000)["batch_size"] == 10000
    with pytest.raises(ValueError, match="no single run"):
        make_figures.run_at(_result(), 500)


def test_clean_rates_are_per_feature():
    run = make_figures.run_at(_result(), 10000)

    assert make_figures.clean_rate(run, "calibrated", "linf") == pytest.approx((0.02, 0.01, 0.03))


def test_a_dose_response_holds_one_fault_in_increasing_severity():
    run = make_figures.run_at(_result(), 10000)

    points = make_figures.dose_response(run, "unit_change", "calibrated", "linf")

    assert [severity for severity, _ in points] == [0.01, 0.05]
    assert [cell[0] for _, cell in points] == [0.4, 0.9]
    assert points[0][1] == pytest.approx((0.4, 0.39, 0.41))


def test_cells_that_were_not_scored_are_left_out_and_not_drawn_as_zero():
    run = make_figures.run_at(_result(), 10000)

    assert make_figures.dose_response(run, "dropped_feature", "calibrated", "linf") == []
    assert len(make_figures.dose_response(run, "dropped_feature", "textbook", "schema")) == 1


def test_an_unknown_fault_is_an_error():
    run = make_figures.run_at(_result(), 10000)

    with pytest.raises(ValueError, match="time_travel"):
        make_figures.dose_response(run, "time_travel", "calibrated", "linf")
    with pytest.raises(ValueError, match="time_travel"):
        make_figures.accuracy_loss(_result(), "time_travel")


def test_accuracy_loss_is_in_percentage_points_and_keeps_its_sign():
    loss = make_figures.accuracy_loss(_result(), "unit_change")

    assert [severity for severity, _ in loss] == [0.01, 0.05]
    assert loss[0][1] == pytest.approx((-0.2, -0.3, -0.1))
    assert loss[1][1] == pytest.approx((1.1, 0.9, 1.2))


def test_every_number_a_figure_needs_exists_in_the_published_results():
    for name, batch_size, _ in make_figures.CLEAN_PANELS:
        run = make_figures.run_at(make_figures.load(name), batch_size)
        for arm, rule, _ in make_figures.CLEAN_RULES:
            assert make_figures.clean_rate(run, arm, rule) is not None
        for rule in make_figures.CALIBRATED:
            assert make_figures.clean_rate(run, "calibrated", rule) is not None
    covertype = make_figures.run_at(make_figures.load("covertype"), 10_000)
    for fault in ("unit_change", "mean_shift"):
        for arm, rule, *_ in make_figures.DOSE_RULES:
            assert len(make_figures.dose_response(covertype, fault, arm, rule)) == 5
    series = make_figures.load("synthetic_series")
    assert len(make_figures.accuracy_loss(series, "time_travel")) == 5


def test_the_figures_named_in_the_readme_are_the_ones_the_script_draws():
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    for name in make_figures.DRAWINGS:
        assert f"figures/{name}" in readme
        assert (REPO / "figures" / name).is_file()


def test_selecting_the_numbers_does_not_need_matplotlib():
    assert "matplotlib" not in sys.modules
