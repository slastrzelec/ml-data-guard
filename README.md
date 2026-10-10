# ml-data-guard

Data validation for ML pipelines, benchmarked with injected faults.

> **Status: phases 1 and 2 of 3 done.** The benchmark with its first results and the library layer
> (schema, validator, comparison, skew check, model unit test, CLI) are in place. Two more datasets
> and the portfolio page are not done yet.

The goal is to measure, not to assume: inject known data faults (a feature pinned to `-1`, a dropped
column, a unit change, a distribution shift, a feature computed with future information) and report
which detection methods catch them and how many false alarms they raise. The methods compared are the
`L-inf` and `L1` distances from Breck et al., "Data Validation for Machine Learning" (SysML 2019),
against chi-square, KL, JS, PSI and Kolmogorov-Smirnov. This is an independent implementation and
evaluation, not a copy of TensorFlow Data Validation.

Scope, data-security rules, evaluation-leakage rules and acceptance criteria: [`SPEC.md`](SPEC.md).

## Results

Three runs, all reproducible from the stored configs: UCI Covertype (581,012 rows, 12 features), a
synthetic i.i.d. table (4,000,000 rows) and a synthetic time-ordered series (2,000,000 rows). Per
run and batch size: 1,000 calibration batches, 1,000 clean evaluation batches and 100 faulty batches
for each fault and severity. Brackets are 95% intervals. Full tables:
[Covertype](benchmarks/results/covertype.md), [synthetic](benchmarks/results/synthetic.md),
[series](benchmarks/results/synthetic_series.md).

### 1. Fixed "textbook" rules: false alarms on clean data

Share of clean (feature, batch) pairs that raise an alarm. Nothing was injected, so every alarm is false.

| Rule | Covertype, 1,000 rows | Covertype, 10,000 rows | Synthetic, 100,000 rows | Time series, 10,000 rows |
|---|---|---|---|---|
| chi-square `p < 0.05` | 0.055 [0.051, 0.059] | 0.049 [0.045, 0.053] | 0.053 [0.048, 0.059] | 0.556 [0.541, 0.571] |
| KS `p < 0.05` | 0.035 [0.032, 0.039] | 0.037 [0.033, 0.041] | 0.040 [0.035, 0.045] | 0.446 [0.428, 0.464] |
| `L-inf > 0.01` | 0.936 [0.931, 0.940] | 0.014 [0.012, 0.016] | 0.000 [0.000, 0.001] | 0.439 [0.423, 0.454] |
| `PSI > 0.2` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.001] |
| Calibrated, any detector | 0.007 - 0.013 | 0.009 - 0.013 | 0.011 - 0.017 | 0.005 - 0.019 |

- A 5% rate per feature is what `p < 0.05` promises, but a batch has many features: on Covertype
  (12 features) chi-square `p < 0.05` flags **44% of clean batches** at 10,000 rows
  [0.41, 0.47], KS 27%.
- On autocorrelated data the p-values are simply wrong: more than half of all clean features alarm,
  and 98% of clean batches. Calibrating the threshold on clean batches of the same kind brings this
  back to 0.5 - 1.9% per feature.
- A fixed `L-inf` threshold is only meaningful for large batches: `0.01` fires on 94% of clean
  features at 1,000 rows and on almost none at 100,000.
- `PSI > 0.2` never fires on clean data. The price is in the next table.

### 2. Recall at an equal false-alarm budget (Covertype, 10,000 rows per batch)

Every detector is calibrated to a 1% false-positive target on clean batches (achieved: 0.9 - 1.3% per
feature). Recall = share of 100 faulty batches flagged on the faulted feature. Last two columns: the
fixed `PSI > 0.2` rule, and the loss of accuracy of a gradient-boosting model (clean accuracy 0.832)
when the same fault hits held-out data, in percentage points.

| Fault | Rows affected | `L-inf` | `L1` | KL | chi-square | KS | `PSI > 0.2` | Accuracy loss (pp) |
|---|---|---|---|---|---|---|---|---|
| New category | 0.1% | 0.01 | 0.04 | 0.86 | 1.00 | n/a | 0.00 | +0.00 [+0.00, +0.01] |
| New category | 1% | 0.05 | 0.20 | 1.00 | 1.00 | n/a | 0.00 | +0.05 [+0.04, +0.06] |
| Missing values | 0.1% | 0.01 | 0.02 | 0.99 | 1.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.01] |
| Missing values | 20% | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 | 1.00 | +0.98 [+0.73, +1.27] |
| Number stored as text | 20% | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | +0.98 [+0.69, +1.31] |
| Pinned to `-1` | 1% | 0.41 | 0.19 | 0.34 | 0.34 | 0.14 | 0.00 | +0.11 [+0.09, +0.14] |
| Pinned to `-1` | 5% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.18 | +0.70 [+0.57, +0.84] |
| Unit change (x1000) | 1% | 0.40 | 0.13 | 0.24 | 0.22 | 0.09 | 0.00 | +0.21 [+0.18, +0.24] |
| Unit change (x1000) | 5% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | +1.09 [+0.95, +1.23] |
| Out-of-range values | 1% | 0.48 | 0.11 | 0.16 | 0.16 | 0.11 | 0.00 | +0.22 [+0.18, +0.25] |
| Out-of-range values | 5% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | +1.00 [+0.83, +1.17] |
| Mean shift, 0.05 sigma | all | 0.96 | 0.96 | 0.98 | 0.97 | 1.00 | 0.19 | +0.16 [+0.11, +0.20] |
| Tiny noise (control) | 0.01% | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |

With 100 batches per cell a recall of 0.50 has an interval of about +/- 0.10; differences of a few
points between two detectors are not established.

- **No single detector wins.** For a value the reference has never seen (a new category, the first
  missing values), chi-square and KL catch 0.1% of the rows; `L-inf` needs 5%. For mass that moves
  into one place (a sentinel, a unit change, out-of-range values in 1% of the rows) `L-inf` has the
  highest recall, 0.40 - 0.48 against 0.16 - 0.34 for KL and chi-square; for the sentinel the gap
  (0.41 against 0.34) is inside the interval.
- **From 5% of the rows upward every calibrated detector finds every fault of this kind.** The
  differences are in the small faults, where the model also loses little.
- **KS cannot see missing values or a changed type** (recall at most 0.02 at any severity), because it
  compares only the numbers that are left.
- **`PSI > 0.2` is quiet for the wrong reason.** It misses a unit change in 5% of the rows in all 100
  batches, while that fault costs the model 1.09 points of accuracy [0.95, 1.23].
- **Alarms that do not matter exist too.** 0.1% missing values are flagged by chi-square in every
  batch and cost the model nothing measurable.
- A dropped column cannot be scored by any distribution distance. It costs the model 5.5 points
  [4.2, 7.0] and is the job of the schema validator (next phase).

### 3. The blind spot: a feature computed with future information

On the time series, shifting a feature forward in time (`time_travel`) leaves its distribution almost
unchanged. Over five shift lengths and both batch sizes the best detector flags 1 - 5% of the faulty
batches, which is its false-alarm level, while the model's accuracy moves by **-5.1 points
[-6.8, -3.4]** (shift 1: the leak makes the model look better) to **+7.0 points [+4.8, +9.3]**
(shift 10). Comparing distributions cannot find this class of error; it needs a row-by-row
comparison of training and serving features. That check now exists (`mlguard skew`): in a test on
the same fault it flags 1,999 of 2,000 rows. It needs both versions of the same rows, which is why it
is not a column of the tables above.

### How to read these numbers (limits)

- **Sample size matters more than the choice of distance.** On the synthetic table a unit change in
  1% of the rows is found by `L-inf` in 1% of the batches at 1,000 rows, 24% at 10,000 and 99% at
  100,000.
- **Lattice-valued features inflate recall for tiny shifts.** A mean shift of 0.01 sigma on Covertype
  is flagged by KL in 51% of the batches, but that comes from features recorded on a coarse grid
  (the two distance-to-water features: `L-inf` 17 of 17), where any shift moves values off the grid.
  On the other eight numeric features `L-inf` flags 1 of 83. The synthetic zero-inflated feature
  behaves the same way. The per-feature counts are in the JSON files (`by_feature`).
- **The 1% target is not reached exactly.** Thresholds come from one finite split and are applied to
  another; the achieved rate (0.5 - 1.9% per feature) is always reported next to recall.
- **Duplicated rows** are only seen as extra sampling noise: recall stays near the batch-level
  false-alarm rate until 20% of the rows are duplicates, and the model is not affected.
- Faults are synthetic and injected one at a time; one model family, accuracy only; two of the three
  datasets are synthetic. The Online News Popularity and Beijing air-quality runs are not done yet.

## What it does

**Measure** (the benchmark, results above):

- `mlguard.distances`, `mlguard.detectors`: `L-inf`, `L1`, KL, JS, PSI, chi-square and KS per feature,
  with bins and vocabularies taken from the reference data only.
- `mlguard.inject`: twelve seeded fault injectors with documented severity units.
- `mlguard.bench`: thresholds calibrated on a separate split, faults injected into evaluation batches
  only, a model scored on a held-out split, fixed "textbook" rules measured next to the calibrated ones.

**Use** (the library and the `mlguard` command):

- **Schema.** `mlguard infer` reads reference data and writes a small JSON schema: type, share of
  rows that must have a value, numeric range, allowed categories. It holds statistics, never rows;
  category values can be stored hashed or left out.
- **Validation.** `mlguard validate` checks one batch against the schema and explains each finding
  with the observed number, the limit and a suggested schema change. It catches a dropped column, a
  changed type, new categories, out-of-range values and missing values.
- **Distribution shift.** `mlguard compare` measures the distance of a batch to the reference and
  calibrates the limit on the reference for the size of that batch, because the benchmark showed
  that a fixed limit is only right for one batch size.
- **Training/serving skew.** `mlguard skew` joins two logs on a key and compares the same feature of
  the same row. This is the check that finds a feature computed with future information, the blind
  spot of every distribution check above.
- **Model unit test.** `run_model_test` feeds rows that the schema allows, including its edges, to
  your feature or prediction code and returns a single synthetic row that breaks it.

Every command exits with 0 (clean), 1 (anomaly) or 2 (unreadable input), and prints text or JSON,
so it can be a CI step. Cell values are not printed unless asked for.

**Not in scope:** repairing data, streaming validation, text or image data, a web interface.

## Quick start

```bash
git clone https://github.com/slastrzelec/ml-data-guard.git
cd ml-data-guard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

A five-minute tour on synthetic data, with the expected exit code of every command, is in
[`examples/README.md`](examples/README.md). In short:

```bash
python examples/make_example_data.py
mlguard infer examples/data/reference.csv -o examples/data/schema.json --exclude id,target
mlguard validate examples/data/batch_broken.csv --schema examples/data/schema.json
```

```
ERROR   x_normal: OUT_OF_RANGE - 197 of 2000 rows (9.85%) are outside [1.6879, 94.6904]; observed range [19.1831, 85724.5]
        suggestion: widen the range of 'x_normal' to [19.1831, 85724.5], or raise max_out_of_range_fraction to 0.0985
ERROR   x_zero_inflated: MISSING_FEATURE - the column is missing from the batch
        suggestion: restore the column, or mark 'x_zero_inflated' as deprecated
ERROR   c_small: UNEXPECTED_VALUES - 100 of 2000 rows (5.00%) hold 1 distinct value(s) not in the domain
        suggestion: add the new value(s) to the domain of 'c_small', or raise max_unexpected_fraction to 0.05
validate: 3 error(s), 0 warning(s)
```

Reproducing the benchmark: [`benchmarks/README.md`](benchmarks/README.md).

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
- **Schema:** an inferred schema accepts its own reference data; the tolerance for missing values
  accepts fresh samples; JSON round trip; with hashed or omitted domains no category value
  appears in the schema file or in a report.
- **Validator:** every anomaly type is raised by a minimal example and not on clean data. Against
  the injectors it catches six of the twelve faults (dropped column, type change, new category,
  out-of-range values, unit change, missing values); the other six stay inside the schema and
  are asserted to be missed, so the limit is documented by a test.
- **Comparison and skew:** the limit a batch is compared with does not depend on that batch;
  clean batches rarely alarm and shifted ones do; the skew check flags a time shift in 1,999 of
  2,000 rows where the distribution check reports nothing.
- **Model unit test:** generated rows satisfy the schema and reach its edges; a function with a
  hidden assumption fails and the returned row reproduces the failure on its own.
- **Command line:** exit codes 0 / 1 / 2, the shape of the JSON output, and the walk-through in
  `examples/README.md`, which is executed command by command.
- **Failure handling:** unnormalised, negative, non-finite, empty or misaligned input, unknown
  config keys and damaged data files raise an error instead of returning a plausible number.
- **Security and hygiene:** no socket can be opened during tests; no data, secret or large file is
  tracked; the package imports no `pickle`-like or network module and calls no `eval`/`exec`.

**How the tests were checked:** `python tools/mutation_check.py` re-introduces known bugs one at a
time (for example `L-inf` returning a sum, an injector that ignores the seed, thresholds calibrated on
the evaluation split, a silenced anomaly type, category values printed by default, exit code 0
despite anomalies) and requires the suite to fail for each of the 82 mutants. The list is hand-written and targeted; it
is not exhaustive mutation testing.

**Not covered**

- The published numbers themselves: tests cover the code on small synthetic runs; the full runs
  were made once per config on one machine (re-running a config reproduced the JSON file exactly,
  except for the control that was changed between the two runs).
- The Covertype loader on the real file: it is tested on a small hand-made archive of the same
  format; the download script itself is not tested (it is the only code that uses the network).
- Whether the synthetic faults resemble real incidents: they are simplified, documented models.
- pandas nullable (`Int64`, `boolean`) and Arrow-backed dtypes; Parquet input (it needs `pyarrow`,
  which is not installed in CI).
- Large inputs: validation and skew comparison loop over cells in Python and were only run on
  tables of up to 20,000 rows.
- The example GitHub Actions workflow in `examples/` is not executed anywhere.
- Numerical behaviour on vectors with more than a few dozen categories or with probabilities
  below `1e-12`.

## Limitations

- Inferred schemas are loose on purpose and still not safe from false alarms: a heavy-tailed feature
  will sooner or later exceed an inferred range. The anomaly then proposes the new range; accepting
  it is a human decision.
- Hashing a domain keeps values out of files and reports but does not protect short or guessable
  values against a dictionary attack. Use `--domains none` for personal data.
- CSV has no types: a numeric column with one text cell is read as text and reported as a type
  mismatch for every row.
- Calibrating `compare` takes a few seconds and uses half of the reference data for the statistics.

## Roadmap

- [ ] Benchmark on Online News Popularity and Beijing air quality (real time-ordered data).
- [ ] Optional comparison with TFDV and Evidently on the same injected faults.
- [ ] Portfolio page.

## License

MIT - see [`LICENSE`](LICENSE). Datasets and their licences: [`data/README.md`](data/README.md).

## Author

**Sławomir Strzelec** - [Portfolio](https://slastrzelec.github.io/portfolio/) -
[GitHub](https://github.com/slastrzelec) - [LinkedIn](https://www.linkedin.com/in/sławomir-strzelec)
