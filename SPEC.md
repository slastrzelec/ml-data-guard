# SPEC: ml-data-guard

Status: **APPROVED v0.2 (2026-10-09).** Scope changes go into this file first.

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
different distribution) and `natural_daily_variation`. The second one is not an injector: it is a fresh
clean batch drawn by the benchmark runner from the evaluation split. Bootstrapping an existing batch
instead would add resampling noise on top of sampling noise and inflate the false-positive rate.

Severity has one unit per fault, recorded in the fault registry: a fraction of rows (`pinned_value`,
`type_change`, `new_category`, `out_of_domain_numeric`, `missing_rate_increase`, `unit_change`,
`duplicate_rows`, `tiny_noise`), batch standard deviations (`mean_shift`), relative growth of the standard
deviation (`variance_change`), a number of time steps (`time_travel`), or none (`dropped_feature` is binary
and has a single level, an exception to the five-level rule of section 6). A positive severity always
changes at least one eligible row, and for a fixed seed the rows changed at one severity stay changed in
the same way at every larger severity, so detection curves are monotone by construction of the faults.

## 6. Evaluation methodology

- Three public tabular datasets (section 11). Per dataset: reference / calibration / evaluation / held-out
  splits, evaluation cut into batches. Online News Popularity is small (about 40k rows): its evaluation batches
  are bootstrap resamples of the evaluation split and its confidence intervals are expected to be wide.
- For every fault x severity (>= 5 levels) x seed (>= 30): one faulty batch; clean batches are negatives.
- Label: "fault injected" (positive) vs "clean" (negative) per batch and per feature where applicable.
- Metrics: **recall by fault and severity**, **false-positive rate on clean batches** (target set at calibration),
  alert precision, detection curve vs severity, bootstrap confidence intervals.
- **Does the alert matter?** For each fault also record the drop of a downstream model's metric on the held-out split.
  Report where a detector alarms but the model is unaffected (noise) and where the model degrades but nothing alarms.
- All results come with the exact commit, seeds and config; the report states what is not measured.

## 7. Tests (what is verified, what is not)

| Area | Verified |
|---|---|
| Distances | known vectors; independent oracles (SciPy); identity -> 0; symmetry; `L-inf <= L1/2`; bounds; Pinsker and data-processing inequalities; invalid input raises |
| Schema | inference round-trip (inferred schema accepts its own reference data); serialisation round-trip; no raw rows in snapshot |
| Validator | each anomaly type is raised by a minimal example and not raised on clean data |
| Injectors | property-based (hypothesis): input not mutated, index/columns/other features preserved, zero severity -> equal copy, changes nested in severity, seed -> same output; the meaning of every fault (exact fraction of rows, exact shift, future-only time travel); invalid input raises |
| Benchmark | determinism (same seed -> identical report); negative controls stay below the calibrated FPR |
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
  src/mlguard/        distances.py  schema.py  validate.py  compare.py  inject.py  bench.py  cli.py
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
2. `mlguard bench` reproduces the published report bit-for-bit from the stored config and seed.
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
