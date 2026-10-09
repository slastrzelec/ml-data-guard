# ml-data-guard

Data validation for ML pipelines, benchmarked with injected faults.

> **Status: work in progress (phase 1 of 3).** The distance functions and the fault injectors exist
> and are tested; the benchmark that connects them does not.
> There are no benchmark results yet, so this README makes no claim about which method is better.

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

## Quick start

```bash
git clone https://github.com/slastrzelec/ml-data-guard.git
cd ml-data-guard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Testing

The tests need no network, no API key and no data download. `ruff`, `pytest` with a coverage
threshold and `pip-audit` run on Python 3.11 and 3.12 in GitHub Actions.

**What is verified**

- **Distances:** hand-computed values; agreement with SciPy as an independent oracle (KL, JS,
  chi-square); property-based checks of identity, symmetry, bounds, `L-inf <= L1 / 2`, Pinsker's
  inequality and the data-processing inequality (merging categories never increases a distance).
- **Binning:** every row is counted exactly once; unseen categories, missing values and values
  outside the reference range each have a bucket; edges do not depend on the batch under test.
- **Fault injectors:** for every fault and feature type, property-based checks that the input is
  not modified, the same seed gives the same output, zero severity returns an equal copy, a positive
  severity changes at least one row, and the changes at a small severity are a subset of those at a
  larger one; plus the meaning of each fault (exact fraction of rows, mean moved by exactly
  `severity` standard deviations, time travel reads only future rows and never crosses a group).
- **Failure handling:** unnormalised, negative, non-finite, empty or misaligned input raises
  `ValueError` instead of returning a plausible number.
- **Security and hygiene:** no socket can be opened during tests; no data, secret or large file is
  tracked; the package imports no `pickle`-like or network module and calls no `eval`/`exec`.

**How the tests were checked:** `python tools/mutation_check.py` re-introduces known bugs one at a
time (for example `L-inf` returning a sum, KL with swapped arguments, wrong chi-square degrees of
freedom, an injector that modifies its input or ignores the seed, time travel that reads the past)
and requires the suite to fail for each of them.
The list is hand-written and targeted; it is not exhaustive mutation testing.

**Not covered**

- Benchmark, schema, validator and CLI: not implemented yet.
- Whether the synthetic faults resemble real incidents: they are simplified, documented models.
- pandas nullable (`Int64`, `boolean`) and Arrow-backed dtypes in the injectors.
- Real production data and real incidents; streaming validation; non-tabular data.
- Numerical behaviour on vectors with more than a few dozen categories or with probabilities
  below `1e-12`.

## License

MIT - see [`LICENSE`](LICENSE). Datasets and their licences: [`data/README.md`](data/README.md).

## Author

**Sławomir Strzelec** - [Portfolio](https://slastrzelec.github.io/portfolio/) -
[GitHub](https://github.com/slastrzelec) - [LinkedIn](https://www.linkedin.com/in/sławomir-strzelec)
