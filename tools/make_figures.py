"""Draw the static figures of the README from the published result files (SPEC section 16.6).

Needs matplotlib, which is not a dependency of the package or of the tests:

    pip install -e ".[figures]"
    python tools/make_figures.py

The functions that pick the numbers import nothing but the standard library and are tested; the
drawing is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
FIGURES = REPO / "figures"

Cell = tuple[float, float, float]

CALIBRATED = ("linf", "l1", "kl", "js", "psi", "chi2", "ks")

# The batch size of each column is the one used in the first table of the README.
CLEAN_PANELS = (
    ("covertype", 10_000, "Covertype"),
    ("news", 1_000, "Online News"),
    ("beijing_static", 10_000, "Beijing air quality"),
    ("synthetic", 100_000, "Synthetic, i.i.d."),
    ("synthetic_series", 10_000, "Synthetic series"),
)
CLEAN_RULES = (
    ("textbook", "chi2 p<0.05", "chi-square p < 0.05"),
    ("textbook", "ks p<0.05", "KS p < 0.05"),
    ("textbook", "linf>0.01", "L-inf > 0.01"),
    ("textbook", "psi>0.2", "PSI > 0.2"),
    ("textbook", "schema", "Inferred schema"),
)
DOSE_RULES = (
    ("calibrated", "linf", "L-inf, calibrated", "#1f6f8b", "o", "-"),
    ("calibrated", "chi2", "chi-square, calibrated", "#6a4c93", "s", "-"),
    ("calibrated", "ks", "KS, calibrated", "#2a9d5c", "^", "-"),
    ("textbook", "psi>0.2", "PSI > 0.2", "#7a7a7a", "D", "--"),
    ("textbook", "schema", "Inferred schema", "#d84315", "*", "-"),
)


def load(name: str, results_dir: Path = RESULTS) -> dict[str, Any]:
    return json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))


def run_at(result: dict[str, Any], batch_size: int) -> dict[str, Any]:
    matches = [run for run in result["runs"] if run["batch_size"] == batch_size]
    if len(matches) != 1:
        raise ValueError(f"no single run with {batch_size} rows per batch")
    return matches[0]


def cell(stored: dict[str, Any] | None) -> Cell | None:
    """A stored proportion as (rate, low, high); a cell that was not scored stays None."""
    if stored is None:
        return None
    return (float(stored["rate"]), float(stored["low"]), float(stored["high"]))


def clean_rate(run: dict[str, Any], arm: str, rule: str) -> Cell | None:
    """Share of clean (feature, batch) pairs a rule flags."""
    return cell(run["clean"][arm]["per_feature"][rule])


def dose_response(run: dict[str, Any], fault: str, arm: str, rule: str) -> list[tuple[float, Cell]]:
    """(severity, recall) of one rule for every severity of a fault, in increasing severity."""
    points = [
        (float(entry["severity"]), cell(entry[arm][rule]))
        for entry in run["faults"]
        if entry["fault"] == fault
    ]
    if not points:
        raise ValueError(f"fault {fault!r} is not in this run")
    return sorted((severity, value) for severity, value in points if value is not None)


def accuracy_loss(result: dict[str, Any], fault: str) -> list[tuple[float, Cell]]:
    """(severity, loss of accuracy with its interval) in percentage points."""
    points = [
        (
            float(entry["severity"]),
            (100 * entry["accuracy_drop"], 100 * entry["low"], 100 * entry["high"]),
        )
        for entry in result["model"]["faults"]
        if entry["fault"] == fault
    ]
    if not points:
        raise ValueError(f"fault {fault!r} has no model result")
    return sorted(points)


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e3e6e8",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _percent(axis: Any, which: str = "x") -> None:
    from matplotlib.ticker import PercentFormatter

    getattr(axis, f"{which}axis").set_major_formatter(PercentFormatter(1.0, decimals=0))


def draw_false_alarms(plt: Any) -> Any:
    rows = [label for _, _, label in CLEAN_RULES]
    rows.insert(4, "7 distances, calibrated to 1%")
    figure, axes = plt.subplots(1, len(CLEAN_PANELS), figsize=(12.5, 3.1), sharey=True)
    for axis, (name, batch_size, title) in zip(axes, CLEAN_PANELS, strict=True):
        run = run_at(load(name), batch_size)
        for arm, rule, label in CLEAN_RULES:
            rate, low, high = clean_rate(run, arm, rule)
            y = rows.index(label)
            color = "#d84315" if rule == "schema" else "#37474f"
            axis.plot([low, high], [y, y], color=color, linewidth=3, alpha=0.35)
            axis.plot(rate, y, "o", color=color, markersize=6)
            axis.annotate(
                f"{100 * rate:.1f}%", (rate, y), xytext=(7 if rate < 0.7 else -7, 0),
                textcoords="offset points", va="center", ha="left" if rate < 0.7 else "right",
                fontsize=8.5, color=color,
            )  # fmt: skip
        rates = [clean_rate(run, "calibrated", rule)[0] for rule in CALIBRATED]
        y = rows.index("7 distances, calibrated to 1%")
        axis.plot(rates, [y] * len(rates), "o", color="#1f6f8b", markersize=5, alpha=0.7)
        axis.annotate(
            f"{100 * min(rates):.1f} - {100 * max(rates):.1f}%", (max(rates), y), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=8.5, color="#1f6f8b",
        )  # fmt: skip
        axis.set_title(f"{title}\n{batch_size:,} rows per batch")
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks([0, 0.5, 1])
        _percent(axis)
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), rows)
    axes[0].invert_yaxis()
    figure.suptitle(
        "False alarms on clean data: share of clean (feature, batch) pairs flagged, 95% intervals",
        x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.08,
    )  # fmt: skip
    return figure


def _dose_panel(axis: Any, run: dict[str, Any], fault: str, percent_of_rows: bool) -> None:
    for arm, rule, label, color, marker, line in DOSE_RULES:
        points = dose_response(run, fault, arm, rule)
        doses = [severity for severity, _ in points]
        axis.fill_between(
            doses, [c[1] for _, c in points], [c[2] for _, c in points], color=color, alpha=0.13,
            linewidth=0,
        )  # fmt: skip
        axis.plot(
            doses, [c[0] for _, c in points], line, marker=marker, color=color, label=label,
            markersize=9 if marker == "*" else 5.5, linewidth=1.8,
        )  # fmt: skip
    axis.set_xscale("log")
    axis.set_xticks(doses)
    axis.set_xticklabels([f"{100 * d:g}%" if percent_of_rows else f"{d:g}" for d in doses])
    axis.minorticks_off()
    axis.set_ylim(-0.04, 1.04)
    _percent(axis, "y")


def draw_schema_and_distances(plt: Any) -> Any:
    run = run_at(load("covertype"), 10_000)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    _dose_panel(left, run, "unit_change", percent_of_rows=True)
    left.set_title("Unit change (x1000): breaks the contract")
    left.set_xlabel("share of rows affected")
    left.set_ylabel("faulty batches flagged (recall)")
    _dose_panel(right, run, "mean_shift", percent_of_rows=False)
    right.set_title("Mean shift: stays inside the contract")
    right.set_xlabel("shift in standard deviations (all rows)")
    left.legend(frameon=False, loc="center left", fontsize=9)
    figure.suptitle(
        "A schema and a distribution distance find different faults\n"
        "Covertype, 10,000 rows per batch, 100 batches per point, 95% intervals",
        x=0.01, ha="left", fontsize=11.5, fontweight="bold",
    )  # fmt: skip
    return figure


def draw_time_travel(plt: Any) -> Any:
    result = load("synthetic_series")
    run = run_at(result, 10_000)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    for index, rule in enumerate(CALIBRATED):
        points = dose_response(run, "time_travel", "calibrated", rule)
        left.plot(
            [s for s, _ in points], [c[0] for _, c in points], "-o", color="#1f6f8b", alpha=0.55,
            markersize=4, linewidth=1.2, label="7 calibrated distances" if index == 0 else None,
        )  # fmt: skip
    points = dose_response(run, "time_travel", "textbook", "schema")
    left.plot(
        [s for s, _ in points], [c[0] for _, c in points], "-*", color="#d84315", markersize=9,
        linewidth=1.8, label="Inferred schema",
    )  # fmt: skip
    steps = [s for s, _ in points]
    left.set_ylim(-0.04, 1.04)
    _percent(left, "y")
    left.set_xticks(steps)
    left.set_title("What the checks see")
    left.set_xlabel("feature shifted forward by this many time steps")
    left.set_ylabel("faulty batches flagged (recall)")
    left.legend(frameon=False, loc="upper right", fontsize=9)

    loss = accuracy_loss(result, "time_travel")
    values = [c[0] for _, c in loss]
    right.axhline(0, color="#37474f", linewidth=1)
    right.errorbar(
        [s for s, _ in loss], values,
        yerr=[[c[0] - c[1] for _, c in loss], [c[2] - c[0] for _, c in loss]],
        fmt="o-", color="#d84315", capsize=4, linewidth=1.8, markersize=6,
    )  # fmt: skip
    for (step, _), value in zip(loss, values, strict=True):
        right.annotate(
            f"{value:+.1f}", (step, value), xytext=(8, -3), textcoords="offset points", fontsize=9,
        )  # fmt: skip
    right.set_xticks(steps)
    right.set_title("What happens to the model")
    right.set_xlabel("feature shifted forward by this many time steps")
    right.set_ylabel("loss of accuracy (points)")
    right.annotate(
        "below zero: the leak makes the model look better", (steps[-1], 0), xytext=(0, -12),
        textcoords="offset points", ha="right", fontsize=8.5, color="#37474f",
    )  # fmt: skip
    figure.suptitle(
        "A feature computed from the future passes every single-table check\n"
        "Synthetic series, 10,000 rows per batch, 95% intervals",
        x=0.01, ha="left", fontsize=11.5, fontweight="bold",
    )  # fmt: skip
    return figure


DRAWINGS = {
    "false_alarms.png": draw_false_alarms,
    "schema_and_distances.png": draw_schema_and_distances,
    "time_travel.png": draw_time_travel,
}


def main() -> int:
    try:
        import matplotlib
    except ImportError:
        print('matplotlib is missing: pip install -e ".[figures]"')
        return 2
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIGURES.mkdir(exist_ok=True)
    for name, draw in DRAWINGS.items():
        figure = draw(plt)
        figure.savefig(FIGURES / name, metadata={"Software": None})
        plt.close(figure)
        print(f"wrote figures/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
