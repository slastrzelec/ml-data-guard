# SPEC: ml-data-guard

Status: **APPROVED v0.3 (2026-10-10).** Scope changes go into this file first.

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
   reference < calibration < evaluation < held-out; no shuffling across the boundaries. Thresholds are
   calibrated on clean batches that are compared with a reference window at the same time lag as in
   evaluation, so that natural seasonal drift is part of the calibrated null and not counted as a detection.
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
| CLI | `mlguard infer`, `mlguard validate`, `mlguard compare`, `mlguard bench`; non-zero exit code on anomalies (CI step). |

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
                      schema.py  validate.py  compare.py  cli.py   (phase 2)
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
| Regression | Online News Popularity (UCI, DOI 10.24432/C5NS3V) | about 40k rows, 58 features, no missing values | CC BY 4.0 | 3 |
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
