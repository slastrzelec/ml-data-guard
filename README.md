# ml-data-guard

Data validation for ML pipelines, benchmarked with injected faults.

> **Status: work in progress (phase 1 of 3).** The benchmark runs end to end and is tested on
> synthetic data. The full runs (synthetic and Covertype) are not published yet, so this README
> makes no claim about which method is better.

The goal is to measure, not to assume: inject known data faults (a feature pinned to `-1`, a dropped
column, a unit change, a distribution shift, a feature computed with future information) and report
which detection methods catch them and how many false alarms they raise. The methods compared are the
`L-inf` and `L1` distances from Breck et al., "Data Validation for Machine Learning" (SysML 2019),
against chi-square, KL, JS, PSI and Kolmogorov-Smirnov. This is an independent implementation and
evaluation, not a copy of TensorFlow Data Validation.

Scope, data-security rules, evaluation-leakage rules and acceptance criteria: [`SPEC.md`](SPEC.md).

## What exists today

- `mlguard.distances`: `L-inf`, `L1`, KL, JS, PSI on probability vectors; chi-square on counts;
  two-sample KS on raw samples; binning whose edges and vocabularies come from the reference data only.
- `mlguard.inject`: twelve fault injectors on pandas DataFrames (pinned value, dropped feature, type
  change, new category, out-of-domain values, missing values, unit change, mean shift, variance
  change, duplicated rows, time travel, and a tiny-noise negative control), each a pure, seeded
  function with a documented severity unit.
- `mlguard.bench`: the benchmark. Reference statistics from the reference split, thresholds
  calibrated on clean batches of a separate split to a 1% false-positive target, faults injected into
  evaluation batches only, and a model trained on the reference split and scored on a held-out split
  to see whether an alarm matters. Fixed "textbook" rules (`p < 0.05`, `PSI > 0.2`) are measured next
  to the calibrated ones. Output: a JSON result and a generated Markdown report.

## Quick start

```bash
git clone https://github.com/slastrzelec/ml-data-guard.git
cd ml-data-guard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m mlguard.bench --config benchmarks/configs/smoke.json --out smoke.json --report smoke.md
```

The last command is a small synthetic run (under a minute) that writes the same kind of report as the
full runs. Configs and how to run them: [`benchmarks/README.md`](benchmarks/README.md).

## Testing

The tests need no network, no API key and no data download. `ruff`, `pytest` with a coverage
threshold and `pip-audit` run on Python 3.11 and 3.12 in GitHub Actions.

**What is verified**

- **Distances:** hand-computed values; agreement with SciPy as an independent oracle (KL, JS,
  chi-square, KS); property-based checks of identity, symmetry, bounds, `L-inf <= L1 / 2`, Pinsker's
  inequality and the data-processing inequality (merging categories never increases a distance).
- **Binning:** every row is counted exactly once; unseen categories, missing values and values
  outside the reference range each have a bucket; edges do not depend on the batch under test.
- **Fault injectors:** for every fault and feature type, property-based checks that the input is
  not modified, the same seed gives the same output, zero severity returns an equal copy, a positive
  severity changes at least one row, and the changes at a small severity are a subset of those at a
  larger one; plus the meaning of each fault (exact fraction of rows, mean moved by exactly
  `severity` standard deviations, time travel reads only future rows and never crosses a group).
- **Evaluation leakage:** the four splits are disjoint and complete; thresholds and reference
  statistics stay identical when the evaluation and held-out rows are replaced by garbage; the model
  is trained on the reference split only.
- **Benchmark results:** the same config gives a byte-identical JSON file; clean batches and the
  tiny-noise control stay near the calibrated false-positive rate; strong faults are found; a time
  shift changes the model's accuracy while the distribution distances stay silent (the expected
  blind spot); duplicated rows keep their labels.
- **Failure handling:** unnormalised, negative, non-finite, empty or misaligned input, unknown
  config keys and damaged data files raise an error instead of returning a plausible number.
- **Security and hygiene:** no socket can be opened during tests; no data, secret or large file is
  tracked; the package imports no `pickle`-like or network module and calls no `eval`/`exec`.

**How the tests were checked:** `python tools/mutation_check.py` re-introduces known bugs one at a
time (for example `L-inf` returning a sum, an injector that ignores the seed, thresholds calibrated on
the evaluation split, a model trained on the held-out split, time-ordered data shuffled before
splitting) and requires the suite to fail for each of them. The list is hand-written and targeted; it
is not exhaustive mutation testing.

**Not covered**

- Schema, single-batch validator and CLI: not implemented yet.
- The Covertype loader on the real file: it is tested on a small hand-made archive of the same
  format; the download script itself is not tested (it is the only code that uses the network).
- Whether the synthetic faults resemble real incidents: they are simplified, documented models.
- pandas nullable (`Int64`, `boolean`) and Arrow-backed dtypes in the injectors.
- Numerical behaviour on vectors with more than a few dozen categories or with probabilities
  below `1e-12`.

## License

MIT - see [`LICENSE`](LICENSE). Datasets and their licences: [`data/README.md`](data/README.md).

## Author

**Sławomir Strzelec** - [Portfolio](https://slastrzelec.github.io/portfolio/) -
[GitHub](https://github.com/slastrzelec) - [LinkedIn](https://www.linkedin.com/in/sławomir-strzelec)
