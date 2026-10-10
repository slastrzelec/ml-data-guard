"""The static results explorer (SPEC section 16): faithful to the result files, and inert."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"

_spec = importlib.util.spec_from_file_location(
    "build_explorer", REPO / "tools" / "build_explorer.py"
)
build_explorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_explorer)

PUBLISHED = sorted(path.stem for path in RESULTS.glob("*.json") if path.stem != "smoke")


def _cell(count: int, total: int) -> dict[str, float]:
    return {"k": count, "n": total, "rate": count / total, "low": 0.123456, "high": 0.987654}


def _tiny_result() -> dict:
    calibrated = {key: _cell(1, 100) for key in build_explorer.CALIBRATED}
    textbook = {key: _cell(5, 100) for key in (*build_explorer.TEXTBOOK, "schema")}
    return {
        "meta": {
            "commit": "abc1234",
            "n_rows": 5000,
            "config": {"reference_mode": "static", "target_fpr": 0.01, "data_path": "C:/secret"},
        },
        "model": {
            "batch_size": 500,
            "clean_accuracy": 0.81234567,
            "faults": [
                {
                    "fault": "unit_change",
                    "severity": 0.01,
                    "accuracy_drop": 0.002,
                    "low": 0.001,
                    "high": 0.003,
                },
            ],
        },
        "runs": [
            {
                "batch_size": 1000,
                "thresholds": {"x_secret_feature": {"linf": 0.5}},
                "clean": {
                    "n_batches": 200,
                    "calibrated": {"per_feature": calibrated, "per_batch": calibrated},
                    "textbook": {"per_feature": textbook, "per_batch": textbook},
                },
                "faults": [
                    {
                        "fault": "unit_change",
                        "severity": 0.01,
                        "severity_unit": "fraction_of_rows",
                        "level": "feature",
                        "negative_control": False,
                        "scorable": True,
                        "n": 100,
                        "by_feature": {"x_secret_feature": {"linf": [1, 7]}},
                        "calibrated": dict(calibrated, linf=_cell(40, 100)),
                        "textbook": textbook,
                    },
                    {
                        "fault": "dropped_feature",
                        "severity": 1,
                        "severity_unit": "binary",
                        "level": "feature",
                        "negative_control": False,
                        "scorable": False,
                        "n": 100,
                        "by_feature": {},
                        "calibrated": dict.fromkeys(build_explorer.CALIBRATED),
                        "textbook": dict(
                            dict.fromkeys(build_explorer.TEXTBOOK), schema=_cell(100, 100)
                        ),
                    },
                ],
            }
        ],
    }


def test_a_run_keeps_rates_and_intervals_rounded_to_four_decimals():
    run = build_explorer.compact_run("tiny", "Tiny", _tiny_result())
    size = run["sizes"][0]
    fault = size["faults"][0]

    assert fault["rules"]["calibrated:linf"] == [0.4, 0.1235, 0.9877]
    assert fault["rules"]["textbook:schema"] == [0.05, 0.1235, 0.9877]
    assert size["clean"]["feature"]["calibrated:ks"] == [0.01, 0.1235, 0.9877]
    assert fault["impact"] == [0.002, 0.001, 0.003]
    assert run["model"] == {"clean_accuracy": 0.8123, "batch_size": 500}
    assert (run["commit"], run["rows"], size["n_clean"]) == ("abc1234", 5000, 200)


def test_a_cell_that_was_not_scored_stays_empty_and_is_not_turned_into_zero():
    dropped = build_explorer.compact_run("tiny", "Tiny", _tiny_result())["sizes"][0]["faults"][1]

    assert dropped["rules"]["calibrated:linf"] is None
    assert dropped["rules"]["textbook:psi>0.2"] is None
    assert dropped["rules"]["textbook:schema"] == [1.0, 0.1235, 0.9877]
    assert dropped["impact"] is None


def test_feature_names_thresholds_and_paths_do_not_reach_the_page():
    run = build_explorer.compact_run("tiny", "Tiny", _tiny_result())
    page = build_explorer.render({"runs": [run]}, f"<p>{build_explorer.PLACEHOLDER}</p>")

    assert "x_secret_feature" not in page
    assert "C:/secret" not in page
    assert "thresholds" not in page
    assert "by_feature" not in page


def test_a_fault_without_a_label_is_an_error():
    result = _tiny_result()
    result["runs"][0]["faults"][0]["fault"] = "brand_new_fault"

    with pytest.raises(ValueError, match="brand_new_fault"):
        build_explorer.compact_run("tiny", "Tiny", result)


def test_a_run_without_a_model_is_accepted():
    result = _tiny_result()
    result["model"] = None

    run = build_explorer.compact_run("tiny", "Tiny", result)

    assert run["model"] is None
    assert run["sizes"][0]["faults"][0]["impact"] is None


def test_the_data_cannot_close_its_script_tag():
    page = build_explorer.render(
        {"label": "</script><b>"}, f"<script>{build_explorer.PLACEHOLDER}</script>"
    )

    assert page.count("</script>") == 1
    assert json.loads(page[len("<script>") : -len("</script>")]) == {"label": "</script><b>"}


@pytest.mark.parametrize("template", ["no placeholder", "/*EXPLORER_DATA*/ and /*EXPLORER_DATA*/"])
def test_the_template_must_hold_the_placeholder_exactly_once(template):
    with pytest.raises(ValueError, match="exactly once"):
        build_explorer.render({}, template)


def test_unknown_or_missing_result_files_are_an_error(tmp_path):
    (tmp_path / "covertype.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="do not match"):
        build_explorer.build_data(tmp_path)


def test_every_published_run_is_in_the_explorer():
    assert sorted(build_explorer.RUNS) == PUBLISHED


@pytest.mark.parametrize("name", PUBLISHED)
def test_every_cell_of_a_published_result_is_shown_with_the_same_numbers(name):
    result = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    shown = build_explorer.compact_run(name, build_explorer.RUNS[name], result)
    impact = {(m["fault"], m["severity"]): m for m in result["model"]["faults"]}

    def same(cell, stored):
        if stored is None:
            return cell is None
        return cell == [round(stored[key], 4) for key in ("rate", "low", "high")]

    assert [size["batch_size"] for size in shown["sizes"]] == [
        r["batch_size"] for r in result["runs"]
    ]
    for size, run in zip(shown["sizes"], result["runs"], strict=True):
        for level, key in (("feature", "per_feature"), ("batch", "per_batch")):
            for arm in ("calibrated", "textbook"):
                for rule, stored in run["clean"][arm][key].items():
                    assert same(size["clean"][level][f"{arm}:{rule}"], stored)
        assert len(size["faults"]) == len(run["faults"])
        for fault, entry in zip(size["faults"], run["faults"], strict=True):
            assert (fault["fault"], fault["severity"]) == (entry["fault"], entry["severity"])
            assert fault["level"] == entry["level"]
            assert fault["control"] == entry["negative_control"]
            rules = {
                f"{arm}:{rule}": entry[arm][rule]
                for arm in ("calibrated", "textbook")
                for rule in entry[arm]
            }
            assert set(fault["rules"]) == set(rules)
            assert all(same(fault["rules"][rule], stored) for rule, stored in rules.items())
            model = impact[(entry["fault"], entry["severity"])]
            assert fault["impact"] == [
                round(model[key], 6) for key in ("accuracy_drop", "low", "high")
            ]


def test_the_tracked_page_is_what_the_build_produces():
    tracked = (REPO / "explorer" / "index.html").read_text(encoding="utf-8").replace("\r\n", "\n")

    assert tracked == build_explorer.build_page()
    assert build_explorer.main(["--check"]) == 0


def test_the_page_loads_nothing_and_stores_nothing():
    page = (REPO / "explorer" / "index.html").read_text(encoding="utf-8")

    assert re.findall(r"https?://", page) == []
    assert re.findall(r"<(?:link|img|iframe|object|embed)\b", page) == []
    assert re.findall(r"<script[^>]*\bsrc=", page) == []
    for forbidden in (
        "fetch(", "XMLHttpRequest", "WebSocket", "sendBeacon", "import(", "localStorage",
        "sessionStorage", "indexedDB", "document.cookie", "innerHTML", "outerHTML",
        "insertAdjacentHTML", "document.write", "eval(", "new Function",
    ):  # fmt: skip
        assert forbidden not in page, forbidden
