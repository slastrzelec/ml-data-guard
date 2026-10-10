"""Build the static results explorer (SPEC section 16.1).

Reads the published result files, keeps the aggregate numbers only and writes one self-contained
HTML page. The page computes no statistic: every rate and interval is copied from the result
files, rounded to four decimals.

    python tools/build_explorer.py            # rewrite explorer/index.html
    python tools/build_explorer.py --check    # fail if the tracked page is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
TEMPLATE = REPO / "explorer" / "template.html"
OUTPUT = REPO / "explorer" / "index.html"
PLACEHOLDER = "/*EXPLORER_DATA*/"
DECIMALS = 4

RUNS: dict[str, str] = {
    "covertype": "Covertype",
    "news": "Online News Popularity",
    "beijing_static": "Beijing air quality, static reference",
    "beijing_rolling": "Beijing air quality, rolling reference",
    "synthetic": "Synthetic table (i.i.d.)",
    "synthetic_series": "Synthetic time series",
}

CALIBRATED: dict[str, str] = {
    "linf": "L-inf",
    "l1": "L1",
    "kl": "KL",
    "js": "JS",
    "psi": "PSI",
    "chi2": "chi-square",
    "ks": "KS",
}
TEXTBOOK: dict[str, str] = {
    "chi2 p<0.05": "chi-square p < 0.05",
    "ks p<0.05": "KS p < 0.05",
    "linf>0.01": "L-inf > 0.01",
    "psi>0.1": "PSI > 0.1",
    "psi>0.2": "PSI > 0.2",
}
SCHEMA_RULE = "schema"

FAULTS: dict[str, str] = {
    "unit_change": "Unit change (x1000)",
    "pinned_value": "Value pinned to -1",
    "out_of_domain_numeric": "Out-of-range values",
    "missing_rate_increase": "Missing values",
    "type_change": "Number stored as text",
    "new_category": "New category",
    "dropped_feature": "Dropped column",
    "mean_shift": "Mean shift",
    "variance_change": "Variance change",
    "duplicate_rows": "Duplicated rows",
    "time_travel": "Feature shifted into the future",
    "tiny_noise": "Tiny noise (negative control)",
}


def rules() -> list[dict[str, str]]:
    """Every rule of the benchmark, in display order, with the group it belongs to."""
    listed = [
        {"id": f"calibrated:{key}", "label": label, "group": "calibrated"}
        for key, label in CALIBRATED.items()
    ]
    listed += [
        {"id": f"textbook:{key}", "label": label, "group": "textbook"}
        for key, label in TEXTBOOK.items()
    ]
    listed.append({"id": f"textbook:{SCHEMA_RULE}", "label": "Inferred schema", "group": "schema"})
    return listed


def _rate(cell: dict[str, Any] | None) -> list[float] | None:
    """A stored proportion as [rate, low, high]; None stays None (not scored)."""
    if cell is None:
        return None
    return [round(float(cell[key]), DECIMALS) for key in ("rate", "low", "high")]


def _by_rule(calibrated: dict[str, Any], textbook: dict[str, Any]) -> dict[str, Any]:
    cells = {f"calibrated:{key}": _rate(calibrated[key]) for key in CALIBRATED}
    cells.update({f"textbook:{key}": _rate(textbook[key]) for key in (*TEXTBOOK, SCHEMA_RULE)})
    return cells


def _impact(model: dict[str, Any] | None) -> dict[str, list[float]]:
    if not model:
        return {}
    return {
        f"{entry['fault']}|{entry['severity']:g}": [
            round(float(entry[key]), 6) for key in ("accuracy_drop", "low", "high")
        ]
        for entry in model["faults"]
    }


def compact_run(name: str, label: str, result: dict[str, Any]) -> dict[str, Any]:
    """The part of one result file that the explorer shows."""
    meta, model = result["meta"], result.get("model")
    config = meta["config"]
    impact = _impact(model)
    sizes = []
    for run in result["runs"]:
        clean = run["clean"]
        faults = []
        for entry in run["faults"]:
            if entry["fault"] not in FAULTS:
                raise ValueError(f"{name}: fault {entry['fault']!r} has no label")
            faults.append(
                {
                    "fault": entry["fault"],
                    "severity": entry["severity"],
                    "unit": entry["severity_unit"],
                    "level": entry["level"],
                    "control": bool(entry["negative_control"]),
                    "n": entry["n"],
                    "rules": _by_rule(entry["calibrated"], entry["textbook"]),
                    "impact": impact.get(f"{entry['fault']}|{entry['severity']:g}"),
                }
            )
        sizes.append(
            {
                "batch_size": run["batch_size"],
                "n_clean": clean["n_batches"],
                "clean": {
                    level: _by_rule(clean["calibrated"][key], clean["textbook"][key])
                    for level, key in (("feature", "per_feature"), ("batch", "per_batch"))
                },
                "faults": faults,
            }
        )
    return {
        "id": name,
        "label": label,
        "rows": meta["n_rows"],
        "commit": meta["commit"],
        "reference": config["reference_mode"],
        "target_fpr": config["target_fpr"],
        "model": (
            {"clean_accuracy": round(float(model["clean_accuracy"]), DECIMALS),
             "batch_size": model["batch_size"]}
            if model
            else None
        ),
        "sizes": sizes,
    }  # fmt: skip


def build_data(results_dir: Path = RESULTS) -> dict[str, Any]:
    """Everything the page needs, from every published result file."""
    published = sorted(path.stem for path in results_dir.glob("*.json") if path.stem != "smoke")
    if published != sorted(RUNS):
        raise ValueError(f"result files {published} do not match the known runs {sorted(RUNS)}")
    runs = []
    for name, label in RUNS.items():
        result = json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))
        runs.append(compact_run(name, label, result))
    return {"rules": rules(), "faults": FAULTS, "runs": runs}


def render(data: dict[str, Any], template: str) -> str:
    """Put the data into the template; `</` is escaped so the JSON cannot close its script tag."""
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"the template must contain {PLACEHOLDER} exactly once")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return template.replace(PLACEHOLDER, payload.replace("</", "<\\/"))


def build_page() -> str:
    template = TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n")
    return render(build_data(), template)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)
    page = build_page()
    if args.check:
        tracked = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        tracked = tracked.replace("\r\n", "\n")
        if tracked != page:
            print("explorer/index.html is out of date: run python tools/build_explorer.py")
            return 1
        print("explorer/index.html is up to date")
        return 0
    OUTPUT.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(page.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
