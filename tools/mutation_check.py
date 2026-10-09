"""Targeted mutation checks: re-introduce known bugs and require the test suite to fail.

Each mutant is one textual change applied to a temporary copy of the repository; the original
files are never touched. A mutant is checked against the test files of the module it changes.
This is a fixed, hand-written list, not exhaustive mutation testing.

Usage: python tools/mutation_check.py [--only TEXT] [--skip-baseline]
  --only TEXT       check only the mutants whose name or file contains TEXT
  --skip-baseline   do not run the unmodified suite first
Exit code 0 when every mutant is killed, 1 when a mutant survives or the baseline fails.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COPIED = ("src", "tests", "pyproject.toml", ".gitignore")
DISTANCES = "src/mlguard/distances.py"
INJECT = "src/mlguard/inject.py"
DISTANCE_TESTS = ("tests/test_distances.py", "tests/test_binning.py")
INJECT_TESTS = ("tests/test_inject.py",)


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str
    tests: tuple[str, ...]


MUTANTS = [
    Mutant(
        "linf returns the sum instead of the maximum",
        DISTANCES,
        "return float(np.max(np.abs(p_array - q_array)))",
        "return float(np.sum(np.abs(p_array - q_array)))",
        DISTANCE_TESTS,
    ),
    Mutant(
        "linf drops the absolute value",
        DISTANCES,
        "return float(np.max(np.abs(p_array - q_array)))",
        "return float(np.max(p_array - q_array))",
        DISTANCE_TESTS,
    ),
    Mutant(
        "l1 returns total variation (half of L1)",
        DISTANCES,
        "return float(np.sum(np.abs(p_array - q_array)))",
        "return 0.5 * float(np.sum(np.abs(p_array - q_array)))",
        DISTANCE_TESTS,
    ),
    Mutant(
        "kl swaps numerator and denominator",
        DISTANCES,
        "(np.log(p[support]) - np.log(q[support]))",
        "(np.log(q[support]) - np.log(p[support]))",
        DISTANCE_TESTS,
    ),
    Mutant(
        "kl returns a finite value on disjoint support",
        DISTANCES,
        "    if np.any(q[support] == 0):\n        return math.inf\n",
        "    support = support & (q > 0)\n",
        DISTANCE_TESTS,
    ),
    Mutant(
        "js uses p instead of the mixture",
        DISTANCES,
        "mixture = 0.5 * (p_array + q_array)",
        "mixture = p_array",
        DISTANCE_TESTS,
    ),
    Mutant(
        "js ignores the logarithm base",
        DISTANCES,
        "    return value / math.log(base)\n\n\ndef psi",
        "    return value\n\n\ndef psi",
        DISTANCE_TESTS,
    ),
    Mutant(
        "psi is one-directional",
        DISTANCES,
        "np.sum((p_occupied - q_occupied) * log_ratio)",
        "np.sum(p_occupied * log_ratio)",
        DISTANCE_TESTS,
    ),
    Mutant(
        "psi ignores empty bins instead of returning infinity",
        DISTANCES,
        "    if np.any(p_occupied == 0) or np.any(q_occupied == 0):\n        return math.inf\n",
        "    keep = (p_occupied > 0) & (q_occupied > 0)\n"
        "    p_occupied, q_occupied = p_occupied[keep], q_occupied[keep]\n",
        DISTANCE_TESTS,
    ),
    Mutant(
        "smoothing is not renormalised",
        DISTANCES,
        "scale = 1.0 + smoothing * p.size",
        "scale = 1.0",
        DISTANCE_TESTS,
    ),
    Mutant(
        "probability vectors are not checked to sum to one",
        DISTANCES,
        "if abs(total - 1.0) > SUM_TOLERANCE:",
        "if False:",
        DISTANCE_TESTS,
    ),
    Mutant(
        "negative values are accepted",
        DISTANCES,
        "if np.any(array < 0):",
        "if False:",
        DISTANCE_TESTS,
    ),
    Mutant(
        "chi-square uses the wrong degrees of freedom",
        DISTANCES,
        "df=a.size - 1",
        "df=a.size",
        DISTANCE_TESTS,
    ),
    Mutant(
        "chi-square expected counts ignore the sample sizes",
        DISTANCES,
        "expected_a = column_totals * total_a / (total_a + total_b)",
        "expected_a = column_totals * 0.5",
        DISTANCE_TESTS,
    ),
    Mutant(
        "ks statistic is one-sided",
        DISTANCES,
        "statistic = float(np.max(np.abs(cdf_a - cdf_b)))",
        "statistic = float(np.max(cdf_a - cdf_b))",
        DISTANCE_TESTS,
    ),
    Mutant(
        "ks empirical CDF excludes tied values",
        DISTANCES,
        'cdf_b = np.searchsorted(b, points, side="right") / b.size',
        'cdf_b = np.searchsorted(b, points, side="left") / b.size',
        DISTANCE_TESTS,
    ),
    Mutant(
        "ks p-value ignores the sample sizes",
        DISTANCES,
        "scaled = (effective_n + 0.12 + 0.11 / effective_n) * statistic",
        "scaled = statistic",
        DISTANCE_TESTS,
    ),
    Mutant(
        "unseen categories are counted as missing",
        DISTANCES,
        "counts[index.get(value, unseen)] += 1",
        "counts[index.get(value, missing)] += 1",
        DISTANCE_TESTS,
    ),
    Mutant(
        "missing values are part of the vocabulary",
        DISTANCES,
        "Counter(value for value in reference if not _is_missing(value))",
        "Counter(value for value in reference)",
        DISTANCE_TESTS,
    ),
    Mutant(
        "values on an edge share a bin with their neighbours",
        DISTANCES,
        'np.searchsorted(edge_array, present, side="left") + np.searchsorted(',
        'np.searchsorted(edge_array, present, side="right") + np.searchsorted(',
        DISTANCE_TESTS,
    ),
    Mutant(
        "missing values are dropped from the bin counts",
        DISTANCES,
        "counts[n_value_bins] = float(missing.sum())",
        "counts[n_value_bins] = 0.0",
        DISTANCE_TESTS,
    ),
    Mutant(
        "quantile edges include the minimum and maximum",
        DISTANCES,
        "levels = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]",
        "levels = np.linspace(0.0, 1.0, n_bins + 1)",
        DISTANCE_TESTS,
    ),
    Mutant(
        "injector modifies its input",
        INJECT,
        "    result = batch.copy(deep=True)\n    result[feature] = column",
        "    result = batch\n    result[feature] = column",
        INJECT_TESTS,
    ),
    Mutant(
        "injector ignores the seed",
        INJECT,
        "order = rng.permutation(candidates)",
        "order = np.random.default_rng().permutation(candidates)",
        INJECT_TESTS,
    ),
    Mutant(
        "affected rows are not nested in severity",
        INJECT,
        "return np.sort(order[: _row_count(fraction, candidates.size)])",
        "count = _row_count(fraction, candidates.size)\n"
        "    return np.sort(rng.choice(candidates, size=count, replace=False))",
        INJECT_TESTS,
    ),
    Mutant(
        "a small positive severity can change no row",
        INJECT,
        "return min(n_eligible, max(1, math.ceil(fraction * n_eligible - 1e-9)))",
        "return math.floor(fraction * n_eligible)",
        INJECT_TESTS,
    ),
    Mutant(
        "pinned_value re-pins rows that already hold the sentinel",
        INJECT,
        "rows = _pick(~(series == sentinel).to_numpy(dtype=bool, na_value=False), fraction, rng)",
        "rows = _pick(np.ones(len(series), dtype=bool), fraction, rng)",
        INJECT_TESTS,
    ),
    Mutant(
        "dropped_feature keeps the column",
        INJECT,
        "return result.drop(columns=[feature]) if _non_negative(severity) > 0 else result",
        "_non_negative(severity)\n    return result",
        INJECT_TESTS,
    ),
    Mutant(
        "type_change turns missing values into text",
        INJECT,
        'series = _numeric_column(batch, feature, "type_change")\n'
        "    rows = _pick(series.notna().to_numpy(), _fraction(severity), rng)",
        'series = _numeric_column(batch, feature, "type_change")\n'
        "    rows = _pick(np.ones(len(series), dtype=bool), _fraction(severity), rng)",
        INJECT_TESTS,
    ),
    Mutant(
        "new_category reuses an existing category name",
        INJECT,
        "while not integer_coded and category in existing:",
        "while False:",
        INJECT_TESTS,
    ),
    Mutant(
        "out_of_domain values stay inside the range",
        INJECT,
        "new = high + span * rng.uniform(0.1, 1.0, size=len(values))[rows]",
        "new = low + span * rng.uniform(0.1, 1.0, size=len(values))[rows]",
        INJECT_TESTS,
    ),
    Mutant(
        "unit_change counts zero rows as changed",
        INJECT,
        "rows = _pick(~np.isnan(as_float) & (as_float != 0), _fraction(severity), rng)",
        "rows = _pick(~np.isnan(as_float), _fraction(severity), rng)",
        INJECT_TESTS,
    ),
    Mutant(
        "mean_shift is not measured in standard deviations",
        INJECT,
        "values + sigmas * std if sigmas > 0",
        "values + sigmas if sigmas > 0",
        INJECT_TESTS,
    ),
    Mutant(
        "variance_change also moves the mean",
        INJECT,
        "mean + (values - mean) * (1.0 + growth)",
        "values * (1.0 + growth)",
        INJECT_TESTS,
    ),
    Mutant(
        "time_travel reads the past instead of the future",
        INJECT,
        "source[positions[:-lead]] = positions[lead:]",
        "source[positions[lead:]] = positions[:-lead]",
        INJECT_TESTS,
    ),
    Mutant(
        "time_travel leaks across groups",
        INJECT,
        "groups = [np.sort(positions) for positions in grouped.values()]",
        "groups = [source.copy()]",
        INJECT_TESTS,
    ),
    Mutant(
        "tiny_noise is not tiny",
        INJECT,
        "rng.normal(0.0, 0.1 * std, size=len(values))",
        "rng.normal(0.0, 10.0 * std, size=len(values))",
        INJECT_TESTS,
    ),
    Mutant(
        "duplicate_rows can copy a row onto itself",
        INJECT,
        "rng.integers(1, n_rows, size=n_rows)",
        "rng.integers(0, n_rows, size=n_rows)",
        INJECT_TESTS,
    ),
    Mutant(
        "a number written into a text column is silently turned into text",
        INJECT,
        "if isinstance(original, pd.StringDtype) and only_text:",
        "if isinstance(original, pd.StringDtype):",
        INJECT_TESTS,
    ),
]


def _copy_repo(target: Path) -> None:
    for name in COPIED:
        source = REPO / name
        if source.is_dir():
            shutil.copytree(source, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target / name)


def _environment(workdir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(workdir / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _imports_the_copy(workdir: Path) -> bool:
    location = subprocess.run(
        [sys.executable, "-c", "import mlguard; print(mlguard.__file__)"],
        cwd=workdir, env=_environment(workdir), capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    return Path(location).resolve().is_relative_to(workdir.resolve())


def _tests_pass(workdir: Path, tests: tuple[str, ...] = ()) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests],
        cwd=workdir, env=_environment(workdir), capture_output=True, text=True, check=False,
    )  # fmt: skip
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="substring of a mutant's name or file")
    parser.add_argument("--skip-baseline", action="store_true")
    arguments = parser.parse_args()
    selected = [
        (number, mutant)
        for number, mutant in enumerate(MUTANTS, start=1)
        if arguments.only in mutant.name or arguments.only in mutant.path
    ]
    if not selected:
        print(f"ERROR: no mutant matches {arguments.only!r}")
        return 1

    with tempfile.TemporaryDirectory() as temporary:
        baseline = Path(temporary) / "baseline"
        baseline.mkdir()
        _copy_repo(baseline)
        if not _imports_the_copy(baseline):
            print("ERROR: the tests would import the installed package, not the copy")
            return 1
        if not arguments.skip_baseline and not _tests_pass(baseline):
            print("ERROR: the unmodified test suite fails; fix it before checking mutants")
            return 1

        survivors: list[str] = []
        for number, mutant in selected:
            workdir = Path(temporary) / f"mutant_{number:02d}"
            workdir.mkdir()
            _copy_repo(workdir)
            target = workdir / mutant.path
            text = target.read_text(encoding="utf-8")
            if text.count(mutant.old) != 1:
                print(f"ERROR: mutant {number} ({mutant.name}) does not apply exactly once")
                return 1
            target.write_text(text.replace(mutant.old, mutant.new), encoding="utf-8")
            killed = not _tests_pass(workdir, mutant.tests)
            print(
                f"{'killed  ' if killed else 'SURVIVED'}  {number:02d}  {mutant.name}", flush=True
            )
            if not killed:
                survivors.append(mutant.name)

    print(f"\n{len(selected) - len(survivors)}/{len(selected)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
