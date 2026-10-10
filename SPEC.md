# SPEC: ml-data-guard

Status: **APPROVED v0.5 (2026-10-10).** Phases 1 and 2 complete (audits in section 13); phase 3 design in section 15. Scope changes go into this file first.

Data validation for ML pipelines, benchmarked with injected faults.

## 1. Problem and goal

Errors in the data fed to an ML pipeline silently degrade models: a feature pinned to `-1` after a refactor,
a dropped column, a unit change, a feature computed with future information (training/serving skew,
"time travel"). Aggregate model metrics often hide such errors because the model is robust to noise.

Goal: a small Python library + CLI + CI step that (1) infers a schema from reference data,
(2) validates new batches against it, (3) compares batches (training vs serving, batch vs batch),
and (4) **measures how well each detection method works** on a catalogue of injected faults.

The benchmark is the core of the project, not the library. Design is informed by the published
Google paper "Data Validation for Machine Learning" (Breck et al., SysML 2019); this project is an
independent implementation and evaluation, not a copy of TFDV.

## 2. Non-goals

- Not a TFDV / Great Expectations / Evidently replacement; no web platform, no database, no user accounts.
- No claim of production readiness on real company data: faults are synthetic and the datasets are public.
- No LLM or text-embedding features in v1 (tabular data only).
- No automatic repair of data.

## 3. Data and security (binding)

| Topic | Rule |
|---|---|
| Datasets | Only public datasets with a permissive licence; licence and source recorded in `data/README.md`. No personal, client or employer data. |
| Raw data in repo | Not tracked. A download script fetches data once and verifies a SHA-256 checksum. A test fails if data files are tracked. |
| Network | Tests and the benchmark run **offline** (synthetic generators + cached data). A test fails if the code under test opens a socket. |
| Schema/stat snapshots | Contain statistics and bounded categorical domains only (max N distinct values, configurable), never raw rows. String domains can be disabled or hashed because category values can themselves be personal data. |
| Reports | Never print raw rows; anomalies show feature name, statistic and threshold. |
| Deserialisation | No `pickle`/`eval`; schemas are JSON/YAML (safe loader). |
| Secrets | None required. `.env` git-ignored; CI uses no secrets. |
| Supply chain | Pinned dependencies, Dependabot, `pip-audit` in CI. |

### Evaluation leakage rules (the project's own data-leakage section)

1. The schema and all reference statistics are inferred **only** from the reference (training) split.
2. Faults are injected **only** into serving/test batches, never into the reference or calibration split.
3. Detector thresholds are calibrated on a separate **calibration** split of clean batches, never on the
   evaluation batches; every detector is calibrated to the same target false-positive rate.
4. Fixed seeds; the benchmark is deterministic (same seed -> byte-identical results).
5. Downstream model metrics are computed on a held-out split that no detector or threshold has seen.
6. Data without a time order (Covertype, Online News Popularity): rows are shuffled once with a fixed seed
   before splitting, because the files are not stored in random order; batches are disjoint slices.
7. Time-ordered data (Beijing air quality, synthetic series): splits are contiguous in time and ordered
   reference < calibration < evaluation < held-out; no shuffling across the boundaries. Calibration and
   evaluation each cover a full year of the Beijing data, so natural seasonal variation is part of the
   calibrated null and not counted as a detection. With a rolling reference (section 15) a batch is compared
   with the rows just before it, never with itself or with later rows.
8. Bin edges and category vocabularies used by a distance come from the reference split only.

## 4. Components

| Component | Description |
|---|---|
| Schema | Per-feature type, presence (min fraction), value count, numeric range / categorical domain, lifecycle stage. Inferred from reference data without overfitting (documented heuristics), versioned as a file. |
| Single-batch validator | Compares a batch to the schema; every anomaly has a type, a feature, a human-readable description and a suggested schema change. |
| Distances | Pure functions on probability vectors, counts and samples (`distances.py`): `L-inf`, `L1`, chi-square, KL, JS, PSI, KS, plus reference-only binning. |
| Inter-batch comparators | Distribution distance per feature: `L-inf` (max change in category probability) and `L1`, with a threshold meaningful to humans; baselines: chi-square, KL, JS, PSI, KS. Feature skew via key-join of training and serving rows. |
| Model unit test | Synthetic examples generated from the schema (seeded fuzzing) are fed to the user's training/prediction function to expose hidden assumptions (e.g. `log(x)` with `x <= 0`). |
| Fault injectors | Pure functions `(batch, feature, severity, rng) -> faulty batch` on pandas DataFrames; see section 5. |
| Benchmark runner | Runs detectors x faults x severities x seeds, writes JSON + Markdown report. |
| CLI | `mlguard infer`, `mlguard validate`, `mlguard compare`, `mlguard skew`, `mlguard bench`; non-zero exit code on anomalies (CI step). |

## 5. Fault catalogue (v1)

`pinned_value` (slice of rows set to a sentinel, e.g. -1) - `dropped_feature` - `type_change` (number -> string) -
`new_category` - `out_of_domain_numeric` - `missing_rate_increase` - `unit_change` (x1000) -
`mean_shift` (delta in sigma) - `variance_change` - `duplicate_rows` -
`time_travel` (feature computed from information not available at serving time).
`time_travel` is evaluated on time-ordered data only (synthetic series in phase 1, Beijing air quality in
phase 3). Expected and reported as such: marginal-distribution distances are not designed to see it, because
the feature's distribution barely changes; the detector that can is the key-joined feature-skew comparison.
A miss by the distribution distances is a result, not a bug.
Two **negative controls** that must NOT alarm: `tiny_noise` (0.01% of rows resampled from a slightly
different distribution; the jitter is clipped to the observed range, because a value outside the feature's
domain would be a real anomaly and not noise) and `natural_daily_variation`. The second one is not an injector: it is a fresh
clean batch drawn by the benchmark runner from the evaluation split. Bootstrapping an existing batch
instead would add resampling noise on top of sampling noise and inflate the false-positive rate.

Severity has one unit per fault, recorded in the fault registry: a fraction of rows (`pinned_value`,
`type_change`, `new_category`, `out_of_domain_numeric`, `missing_rate_increase`, `unit_change`,
`duplicate_rows`, `tiny_noise`), batch standard deviations (`mean_shift`), relative growth of the standard
deviation (`variance_change`), a number of time steps (`time_travel`), or none (`dropped_feature` is binary
and has a single level, an exception to the five-level rule of section 6). `mean_shift` and `variance_change` keep an integer feature integer by
unbiased random rounding (a count does not become 4.02, which a detector would spot for the wrong reason).
A positive severity always
changes at least one eligible row, and for a fixed seed the rows changed at one severity stay changed in
the same way at every larger severity, so detection curves are monotone by construction of the faults.

## 6. Evaluation methodology

**Data.** Three public tabular datasets (section 11) and two synthetic generators (i.i.d. table, time-ordered
series). Per dataset four disjoint splits: reference 40% / calibration 25% / evaluation 25% / held-out 10%.
Calibration and evaluation have the same size on purpose: batches are random subsamples of a finite pool, and
equal pools give both the same finite-population effect, so the calibrated false-positive rate carries over.
Time-ordered data: contiguous splits, batches are contiguous windows at random start positions.

**Detectors.** One score per feature and batch, always "batch versus reference":

| Feature kind | Representation (fitted on the reference only) | Detectors |
|---|---|---|
| Numeric | 10 quantile bins + exact-value bins + missing bucket | `L-inf`, `L1`, KL, JS, PSI, chi-square on the bins; KS on the raw values |
| Categorical | reference vocabulary (max 100) + unseen bucket + missing bucket | `L-inf`, `L1`, KL, JS, PSI, chi-square |

KL is KL(batch || reference); KL and PSI use additive smoothing `1e-4`. In a numeric feature every value that
is not a number (text, `None`) counts as missing; KS ignores missing values, which is reported as a blind spot.

**Two decision rules ("arms").**

1. *Calibrated:* per feature and detector the threshold is the `1 - alpha` quantile of the scores of clean
   calibration batches, `alpha = 0.01`. Every detector gets the same false-positive budget, so recall is
   comparable.
2. *Textbook:* fixed rules as they are used in practice, no calibration: chi-square `p < 0.05`, KS `p < 0.05`,
   PSI `> 0.2`, PSI `> 0.1`, `L-inf > 0.01`. Reported with their measured false-positive rate.

**Batch sizes.** 1,000 / 10,000 / 100,000 rows on synthetic data (4,000,000 rows, so each pool holds
1,000,000) to show how each rule depends on the sample size; 1,000 / 10,000 on Covertype (pools of about
145,000 rows).

**Finite-pool effect (limitation, measured not hidden).** Thresholds are calibrated on one finite split and
applied to batches of another. Each split deviates from the reference in its own way, which widens the
null distribution by roughly `batch size / pool size`. The achieved false-positive rate is therefore
above the 1% target, more so for large batches, and differs slightly between detectors. The report always
shows the achieved rate with its interval next to recall; recall of two detectors is only comparable when
their achieved false-positive rates are. This is also what happens in production, where thresholds come
from a finite window of past data.

**Runs.** 1,000 calibration batches and 1,000 clean evaluation batches per batch size (the clean batches are
the `natural_daily_variation` control). Per fault x severity (5 levels, section 5) x 100 seeds: one evaluation
batch, one randomly chosen applicable feature, one injected fault.

**Metrics.**

- False-positive rate per feature (share of clean batch-feature pairs that alarm) and per batch (share of
  clean batches where any feature alarms; with a dozen features this is several times the per-feature rate).
- Recall per fault and severity: share of faulty batches where the detector alarms on the faulted feature.
  `duplicate_rows` touches every feature, so its recall is batch-level and must be read against the batch-level
  false-positive rate. `dropped_feature` cannot be scored by a distribution distance at all; it is reported as
  not applicable here and belongs to the schema validator (phase 2).
- Proportions carry 95% Wilson score intervals; mean changes of the model metric carry 95% bootstrap intervals.
- **Does the alert matter?** A gradient-boosting model (scikit-learn, optional dependency of the benchmark) is
  trained once on the reference split. For every fault x severity x seed the same fault is injected into a
  batch of the held-out split and the change of accuracy against the clean batch is recorded. The report
  lists faults that alarm without hurting the model and faults that hurt the model without an alarm.
- Results are written as JSON (config, seeds, package versions, commit) plus a generated Markdown report; the
  report states what is not measured.

**Reproducibility.** Every random draw comes from a generator keyed by (seed, purpose, batch size, indices), so
results do not depend on execution order. The same config on the same pinned environment gives a byte-identical
JSON file (tested on synthetic data). Across machines the model metric may differ in the last digits because of
multithreaded floating-point sums; detector results do not.

## 7. Tests (what is verified, what is not)

| Area | Verified |
|---|---|
| Distances | known vectors; independent oracles (SciPy); identity -> 0; symmetry; `L-inf <= L1/2`; bounds; Pinsker and data-processing inequalities; invalid input raises |
| Schema | inference round-trip (inferred schema accepts its own reference data); serialisation round-trip; no raw rows in snapshot |
| Validator | each anomaly type is raised by a minimal example and not raised on clean data |
| Injectors | property-based (hypothesis): input not mutated, index/columns/other features preserved, zero severity -> equal copy, changes nested in severity, seed -> same output; the meaning of every fault (exact fraction of rows, exact shift, future-only time travel); invalid input raises |
| Benchmark | splits are disjoint and complete; thresholds and reference statistics do not change when evaluation or held-out rows are corrupted; determinism (same seed -> identical JSON); clean batches and `tiny_noise` stay near the calibrated FPR; strong faults are found |
| CLI | exit codes, JSON output schema |
| Security/hygiene | no network, no tracked data/secrets, no `pickle`/`eval` |
| Quality gates | `ruff`, `pytest`, coverage threshold, `pip-audit` on Python 3.11 and 3.12 in GitHub Actions |

Mutation checks (break a distance, silence an anomaly type, calibrate on the evaluation split) must make the suite fail.
They are a fixed list of hand-written mutants applied to a temporary copy of the sources
(`tools/mutation_check.py`, run in CI); this is targeted, not exhaustive, mutation testing.

**Not covered:** real production data and real incidents, streaming/online validation, non-tabular data,
distributed scale.

## 8. Layout

```
ml-data-guard/
  src/mlguard/        distances.py  inject.py  detectors.py  datasets.py  bench.py
                      schema.py  validate.py  compare.py  modeltest.py  cli.py
  examples/           runnable walk-through and a CI workflow to copy
  tests/
  tools/              mutation_check.py, data download script
  benchmarks/         configs, results (small, tracked) and generated report
  data/               README.md only (sources, licences, checksums)
  SPEC.md  README.md  LICENSE  pyproject.toml  requirements-dev.txt (pinned, generated)
  .github/workflows/ci.yml  .github/dependabot.yml
```

## 9. Phases

1. **Core:** distances + baselines, injectors, benchmark on synthetic data and Covertype. Tests and CI from day one.
2. **Library:** schema inference/versioning, single-batch validator, model unit test, CLI, CI example.
3. **Report:** Online News Popularity and Beijing air quality, optional baselines against TFDV / Evidently (optional extra, not in core CI), README and portfolio page with recruiter-style Testing section.

## 10. Acceptance criteria

1. Clean clone: `pip install -e .[dev] && pytest` passes offline with no environment variables.
2. The benchmark (`python -m mlguard.bench`, `mlguard bench` once the CLI exists) reproduces the published
   report from the stored config and seed on the pinned environment.
3. The report contains recall per fault/severity, FPR, and the "alert vs model impact" table for every detector.
4. No data files, secrets or personal data tracked (test enforces it).
5. README and portfolio page state what is measured and what is not; no claim without a test or a result behind it.

## 11. Decisions (closed 2026-10-09)

| Role | Dataset | Size | Licence | Phase |
|---|---|---|---|---|
| Classification | Covertype (UCI, DOI 10.24432/C50K5N) | 581,012 rows, 54 features, no missing values | CC BY 4.0 | 1 |
| Classification (planned as regression, see section 15) | Online News Popularity (UCI, DOI 10.24432/C5NS3V) | 39,644 rows, 58 predictive columns, no missing values | CC BY 4.0 | 3 |
| Time-ordered | Beijing Multi-Site Air Quality (UCI, DOI 10.24432/C5RK5G) | 420,768 hourly rows, 12 stations, 2013-2017, missing values | CC BY 4.0 | 3 |

- None of the three contains records of individual people. CC BY 4.0 requires attribution: `data/README.md`
  and the report cite the source of every dataset.
- Covertype stores two categorical features one-hot encoded (4 wilderness areas, 40 soil types); they are
  decoded back to two categorical columns so that the categorical distances have real domains to work on.
- Rejected: Adult and Bank Marketing (records of individuals), California Housing (no categorical features,
  unclear licence), Bike Sharing (17k rows), NYC TLC trip records (no explicit licence found).
- TFDV / Evidently: optional baselines in phase 3 only, as an optional extra outside the core CI.
- Licence: MIT. Supported Python: 3.11 and 3.12.

## 12. Risks

- **"Yet another TFDV"** - mitigated by making the benchmark the deliverable and by comparing against existing tools.
- Synthetic faults may not resemble real incidents - stated as a limitation, fault catalogue is documented and extensible.
- Threshold calibration can leak evaluation data - prevented by the split rules in section 3 and tested.
- Scope creep into a platform - non-goals above.

## 13. Phase audits

### Phase 1 (2026-10-10, commit 0e99fae plus the results commit)

| Acceptance criterion | State | Evidence |
|---|---|---|
| 1. Clean clone installs and `pytest` passes offline, no environment variables | Met | CI on Python 3.11 and 3.12 (Ubuntu) and a local run on Windows 11: 551 tests. CI installs the pinned `requirements-dev.txt`; `pip install -e .[dev]` with unpinned versions is not exercised by CI. |
| 2. The benchmark reproduces the published report from config and seed | Met on one machine | The three full configs were run twice on the same Windows machine (commits de7200a and 0e99fae). Covertype and the series were identical except for the recorded commit; the synthetic run differed only in the `tiny_noise` entries, whose injector was changed in between. Not checked across machines. |
| 3. Report has recall per fault and severity, FPR, and alert versus model impact | Partly | Recall and FPR are reported for every detector. The impact table shows the best recall of any detector per fault, not one column per detector. |
| 4. No data, secrets or personal data tracked | Met | `tests/test_repo_hygiene.py` in CI; the Covertype archive is git-ignored and checksum-pinned. |
| 5. README states what is measured and what is not | Met for the README | The portfolio page belongs to phase 3. |

Scope of phase 1 delivered: distances and baselines, twelve injectors, benchmark on synthetic data and
Covertype, tests, mutation checks and CI from the first commit.

Deviations from the first approved version, all recorded above: split 40/25/25/10 (section 6), Covertype at
1,000 and 10,000 rows per batch only, `natural_daily_variation` as fresh clean batches instead of an
injector, integer-preserving `mean_shift` / `variance_change`, Wilson intervals for proportions.

Known weaknesses carried forward: achieved false-positive rate 0.5 - 1.9% against a 1% target; recall on
tiny shifts is inflated by lattice-valued features; 100 faulty batches per cell give wide intervals; the
feature-skew comparison that could detect `time_travel` does not exist yet.

### Phase 2 (2026-10-10)

| Acceptance criterion (section 14.7) | State | Evidence |
|---|---|---|
| 1. `infer`, `validate`, `compare`, `skew` work on the example data; the walk-through runs as written | Met | `tests/test_cli.py::test_the_walkthrough_runs_as_written` executes every command of `examples/README.md` and checks its exit code. Checked on Linux; on Windows only through the user's local test run. |
| 2. Validator catches the schema-violating injectors; what it misses is listed | Met | `tests/test_validate.py`: six faults caught, six asserted to be missed. |
| 3. Skew flags `time_travel` where the distribution detectors do not | Met | `tests/test_compare.py::test_skew_finds_time_travel_that_the_distances_miss`. |
| 4. No cell values by default; `domains=none` leaves no value in the schema | Met | Tests in `test_schema.py`, `test_validate.py`, `test_cli.py`. |
| 5. CI green on 3.11 and 3.12; mutation checks cover the new modules | To be confirmed by CI | 25 new mutants (82 in total), all killed locally in partial runs. |

Not delivered or weaker than the design: Parquet input is untested (needs `pyarrow`); cell-wise Python loops
limit validation and skew to moderate table sizes; the phase 1 benchmark was not extended with the skew
check (by design, section 14.3).

## 14. Phase 2 design: the library layer

Phase 1 answered "which check finds what". Phase 2 turns the checks into something a pipeline can call. No
new statistics are invented here; the comparison step reuses the detectors of the benchmark.

### 14.1 Schema (`schema.py`)

A schema is a versioned JSON file with one entry per feature:

| Field | Meaning | How it is inferred from reference data |
|---|---|---|
| `kind` | `numeric` or `categorical` | From the column type only (numbers, not booleans, are numeric). An integer-coded category has to be declared by the caller; the tool does not guess from the number of distinct values. |
| `dtype` | `int`, `float` or `string` | From the column type. |
| `min_presence` | Smallest allowed share of non-missing values | 1.0 if nothing is missing; otherwise twice the missing share of the reference plus one percentage point is tolerated, because later batches are smaller than the reference and scatter more. |
| `minimum`, `maximum` | Numeric range | Observed range widened by 10% of its width on each side, never across zero (`padded`, default), exactly as observed (`observed`), or not at all (`none`). Heavy-tailed features will still exceed an inferred range sooner or later; the anomaly then suggests the new range. |
| `domain` | Allowed categorical values | All observed values if there are at most 100 (configurable); none for free-text features. |
| `max_out_of_range_fraction`, `max_unexpected_fraction` | Tolerated share of violating rows | 0; raised by hand for features with known junk. |
| `stage` | `active` or `deprecated` | `active`. A deprecated feature may be absent and is not validated. |

Columns that are not features (keys, targets, timestamps) are listed once under `ignored` and never validated.

Inference is deliberately loose ("without overfitting"): a schema that rejects the next clean batch is worse
than no schema. Every anomaly therefore carries a suggested schema change, and tightening is a human decision.

Data security: the file holds statistics and bounded domains, never rows. Domains can be stored as
`plain` values, as `hashed` values (SHA-256, truncated) or not at all (`none`). Hashing keeps values out of
the file and out of reports; it does not protect low-entropy values against a dictionary attack, so `none` is
the setting for personal data. Reports never print cell values unless the caller asks for it.

### 14.2 Single-batch validator (`validate.py`)

`validate_batch(batch, schema)` returns a list of anomalies; an empty list means the batch conforms.

| Anomaly | Raised when | Level |
|---|---|---|
| `EMPTY_BATCH` | the batch has no rows | error |
| `MISSING_FEATURE` | an active feature has no column | error |
| `UNEXPECTED_FEATURE` | a column is not in the schema | warning |
| `TYPE_MISMATCH` | a numeric feature holds values that are not numbers (or non-integers in an `int` feature) | error |
| `LOW_PRESENCE` | share of non-missing values is below `min_presence` | error |
| `OUT_OF_RANGE` | share of values outside `[minimum, maximum]` exceeds the tolerance | error |
| `UNEXPECTED_VALUES` | share of values outside the domain exceeds the tolerance | error |

Each anomaly has: feature, type, level, a one-sentence description, the observed statistic, the threshold it
was compared with, and a suggested schema change. Text in a numeric column is a type mismatch and is not
counted again as missing or out of range.

### 14.3 Comparison of two batches (`compare.py`)

- **Distribution shift:** `compare_batches(reference, batch, schema, detector=...)` scores every active feature
  with one detector of phase 1 and raises `DISTRIBUTION_SHIFT` above a threshold. The threshold is either
  given by the caller or calibrated: the reference is split in two halves with a seed, statistics are fitted
  on the first half and the threshold is the `1 - target_fpr` quantile of the scores of clean batches (of the
  size of the batch under test) drawn from the second half. The batch under test never influences the
  threshold. The benchmark showed that a fixed threshold is only right for one batch size, which is why
  calibration is the default.
- **Feature skew:** `feature_skew(training, serving, key=...)` joins the two tables on a key and compares
  the same feature of the same row. Result per feature: joined rows, mismatching rows, share; plus keys
  present on one side only. Floats are compared with a tolerance, everything else exactly, missing equals
  missing. A share above the tolerance (default 0) raises `FEATURE_SKEW`. This is the check for
  `time_travel` and for any training/serving skew, and it needs what distribution checks do not: the two
  versions of the same rows. It is tested on the `time_travel` injector; it is not added to the phase 1
  benchmark, where every per-row change would trivially be found.

### 14.4 Model unit test (`modeltest.py`)

`generate_examples(schema, n_rows, seed)` draws rows that the schema allows, including its edges: range
minimum and maximum, zero when allowed, every domain value, and missing values when `min_presence < 1`.
`run_model_test(function, schema, ...)` feeds them to a user function (training step or prediction) and fails
when the function raises, produces NaN or infinity (NumPy floating-point errors are raised, not ignored), or
returns the wrong number of predictions. On failure it searches for a single row that reproduces the problem
and reports it; the row is synthetic, so it can be shown. The finding is "your code assumes more than your
schema promises": either the code or the schema has to change.

### 14.5 CLI (`cli.py`) and CI step

`mlguard infer | validate | compare | skew | bench`. Input: CSV (Parquet when `pyarrow` is installed).
Output: text for people or `--format json` for machines. Exit codes: 0 no anomaly, 1 at least one error-level
anomaly (warnings too with `--strict`), 2 wrong usage or unreadable input. `examples/` holds a runnable
walk-through on synthetic data and a GitHub Actions workflow to copy.

### 14.6 Tests added in phase 2

| Area | Verified |
|---|---|
| Schema | inferred schema accepts its own reference; the presence margin tolerates fresh samples; JSON round trip; domain bound; hashed and omitted domains leave no value in the file; invalid files are rejected |
| Validator | every anomaly type is raised by a minimal example and not on clean data; which injected faults the validator catches |
| Comparison | calibration never sees the batch under test; clean batches rarely alarm, shifted ones do; skew finds `time_travel` that the distances miss |
| Model unit test | generated rows satisfy the schema and are seeded; a function with a hidden assumption fails with a reproducing row; a robust function passes |
| CLI | exit codes 0 / 1 / 2, JSON output shape, no cell values in the output by default |

### 14.7 Acceptance criteria for phase 2

1. `mlguard infer`, `validate`, `compare`, `skew` work on the example data from a clean install; the walk-through in `examples/` runs as written (tested).
2. The validator catches `dropped_feature`, `type_change`, `new_category`, and range and presence violations of the injectors; this is tested, and what it does not catch is listed.
3. The skew comparison flags `time_travel` in a test where the distribution detectors do not.
4. No command prints cell values by default; a schema with `domains=none` contains no value from the data (tested).
5. CI stays green on Python 3.11 and 3.12; mutation checks cover the new modules.

## 15. Phase 3 design: two more datasets, a rolling reference, the schema as a rule

### 15.1 Datasets

- **Online News Popularity** (39,644 rows). The task is the binary one of the dataset's authors: popular
  means at least 1,400 shares (the median). The first plan said regression; the benchmark measures the
  effect of a fault on classification accuracy, and a second metric for one dataset would have blurred the
  comparison. URL and article age are not features. The channel and weekday indicators are decoded into two
  categorical columns: 44 numeric and 2 categorical features. Batches of 500 and 1,000 rows; with pools of
  about 9,900 rows the finite-pool effect of section 6 is larger here than anywhere else.
- **Beijing Multi-Site Air Quality** (12 stations, hourly, 412,029 rows with a PM2.5 reading). Rows are
  ordered by time and then by station. Target: PM2.5 above 75. Features: ten sensor readings, wind
  direction, station, and two features that describe the past of PM2.5 at the same station (one hour
  earlier; mean of the 24 hours before). `time_travel` shifts a feature in time within each station and is
  never applied to the station column itself.

Both loaders validate the layout of the downloaded file; both checksums are pinned.

### 15.2 Rolling reference

On data with seasons a week never looks like the whole year, so thresholds calibrated against a static
reference become very wide. `reference_mode: rolling` compares every batch with the rows just before it
(default: five batches). Statistics are fitted on that window only. Thresholds are still calibrated on
clean batches of the calibration split, each against its own preceding window. Only for time-ordered data.
A window that holds no value of a feature gives no score and therefore no alarm. Beijing is run in both
modes; whether the rolling reference helps on this data is an open question that the run answers.

### 15.3 The schema check as one more rule

The single-batch validator of phase 2 joins the benchmark as a rule that needs no calibration: the schema is
inferred from the reference split with the default heuristics and is not edited by hand; a feature of a batch
alarms when it has at least one error-level anomaly. Its false alarms on clean batches are measured like those
of every other rule. This makes `dropped_feature` scorable (by this rule only) and lets the report compare
"does the data still obey its contract" with "does the data still look the same".

### 15.4 Faults that cannot be injected into a batch

Real features can be constant or empty within a batch (no rain for a week). A mean shift needs a spread, a
unit change a non-zero value, several faults at least one value. The faulted feature of a run is therefore
drawn among the features on which the fault can act in that batch. On the datasets of phase 1 this changes
nothing (checked: the smoke run is identical); on Beijing it means that such features are faulted less often.

### 15.5 Consequences for published results

The result files gain the schema rule and two config fields, so all published runs are repeated at the final
commit. Unchanged parts must come out identical (the determinism of section 6 is the check).

### 15.6 Acceptance criteria for phase 3

1. Results and reports for Covertype, Online News Popularity, Beijing (static and rolling) and both synthetic
   sets come from one commit and are reproducible from the stored configs.
2. README states the findings with intervals and names what was not measured; no claim without a number in
   the result files.
3. Portfolio page with the recruiter-style Testing section, linked from the portfolio index and the repository.
4. The optional comparison with TFDV / Evidently is either done as an optional extra or explicitly dropped
   with the reason.

### 15.7 Decision: no TFDV / Evidently comparison

Dropped. The question it was meant to answer (does a schema-based tool find what distances miss, and at
what cost) is answered inside the benchmark by the schema rule of section 15.3, under the same splits,
seeds and faults as every other rule. Running external tools would add heavy, fast-moving dependencies
and compare default configurations of tools that are meant to be tuned by hand, which is a weaker
comparison than it looks. The decision can be reopened as an optional extra outside the core CI.

