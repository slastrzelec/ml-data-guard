# Walk-through

Everything below runs on synthetic data generated locally; nothing is downloaded. Run the commands
from the repository root after `pip install -e ".[dev]"`. The line after each command is its exit
code (a CI step fails on anything but 0). This file is executed by `tests/test_examples.py`, so the
commands and exit codes are known to be true.

## 1. Data and schema

```bash
python examples/make_example_data.py
# -> exit code 0
mlguard infer examples/data/reference.csv -o examples/data/schema.json --exclude id,target
# -> exit code 0
```

`schema.json` now describes seven features: type, share of rows that must have a value, numeric
range or allowed categories. It contains no row of the data. Edit it by hand to tighten or loosen
an expectation.

## 2. Validate a batch against the schema

```bash
mlguard validate examples/data/batch_ok.csv --schema examples/data/schema.json
# -> exit code 0
mlguard validate examples/data/batch_broken.csv --schema examples/data/schema.json
# -> exit code 1
```

The broken batch has three injected faults and the validator names each one: `x_normal` has 10% of
its rows multiplied by 1000 (`OUT_OF_RANGE`), `c_small` has a category the schema has never seen
(`UNEXPECTED_VALUES`), and the column `x_zero_inflated` is gone (`MISSING_FEATURE`). Every finding
comes with the observed number, the limit and a suggested schema change. Add `--format json` for a
machine-readable result.

## 3. Compare a batch with the reference data

```bash
mlguard compare examples/data/reference.csv examples/data/batch_ok.csv --schema examples/data/schema.json
# -> exit code 0
mlguard compare examples/data/reference.csv examples/data/batch_broken.csv --schema examples/data/schema.json
# -> exit code 1
```

`compare` measures the `L-inf` distance per feature and calibrates the limit on the reference data
for the size of the batch, at a 1% false-alarm target per feature. With seven features roughly one
clean batch in fifteen will still raise an alarm somewhere; that is the price of checking many
features, and it is measured in the benchmark.

## 4. Training/serving skew

```bash
mlguard skew examples/data/training_log.csv examples/data/serving_log.csv --key row_id
# -> exit code 1
```

In the serving log the feature `lag_1` is read one step too late, so it contains the future. Its
distribution is unchanged and no distribution check sees it; joining the two logs on the row key
shows that 2,999 of 3,000 rows differ.

## 5. Model unit test

```bash
python examples/model_unit_test.py
# -> exit code 1
```

The script feeds rows that the schema allows to two feature functions. The one that computes
`log(x_sentinel)` fails, because the schema allows `-1`; the output shows a single generated row
that reproduces the failure.

## As a CI step

[`github-actions/validate-data.yml`](github-actions/validate-data.yml) is a workflow to copy into a
repository that stores a schema and receives data batches.
