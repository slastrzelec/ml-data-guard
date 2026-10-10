# Benchmark report: synthetic_series

- Data: synthetic, 2,000,000 rows, 3 numeric and 1 categorical features.
- Splits (rows): reference 800,000, calibration 500,000, evaluation 500,000, held-out 200,000.
- Reference: static, the reference split.
- Seed 20261010; 1000 calibration batches, 1000 clean batches, 100 faulty batches per fault and severity; target false-positive rate 0.01.
- Commit: f7fd371; ml-data-guard 0.1.0, numpy 2.4.6, pandas 3.0.6, python 3.11.17, scikit-learn 1.9.1, scipy 1.17.1.

Rates are followed by 95% Wilson intervals where space allows; every interval is in the
JSON file. With 100 faulty batches a recall of 0.50 has an interval of
about +/- 0.10.

## Batch size 1,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.009 [0.007, 0.012] | 0.030 [0.021, 0.043] |
| `l1` (calibrated) | 0.008 [0.006, 0.012] | 0.033 [0.024, 0.046] |
| `kl` (calibrated) | 0.008 [0.006, 0.011] | 0.031 [0.022, 0.044] |
| `js` (calibrated) | 0.007 [0.005, 0.011] | 0.029 [0.020, 0.041] |
| `psi` (calibrated) | 0.008 [0.006, 0.011] | 0.030 [0.021, 0.043] |
| `chi2` (calibrated) | 0.009 [0.007, 0.013] | 0.036 [0.026, 0.049] |
| `ks` (calibrated) | 0.005 [0.003, 0.008] | 0.014 [0.008, 0.023] |
| `chi2 p<0.05` (textbook) | 0.544 [0.528, 0.559] | 0.984 [0.974, 0.990] |
| `ks p<0.05` (textbook) | 0.414 [0.396, 0.432] | 0.664 [0.634, 0.693] |
| `psi>0.2` (textbook) | 0.004 [0.003, 0.007] | 0.018 [0.011, 0.028] |
| `psi>0.1` (textbook) | 0.041 [0.035, 0.048] | 0.162 [0.140, 0.186] |
| `linf>0.01` (textbook) | 0.987 [0.983, 0.990] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `mean_shift` | 0.01 | feature | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| `mean_shift` | 0.05 | feature | 0.01 | 0.01 | 0.00 | 0.01 | 0.01 | 0.00 | 0.05 |
| `mean_shift` | 0.1 | feature | 0.13 | 0.12 | 0.12 | 0.13 | 0.13 | 0.15 | 0.20 |
| `mean_shift` | 0.25 | feature | 0.38 | 0.45 | 0.44 | 0.44 | 0.44 | 0.43 | 0.43 |
| `mean_shift` | 0.5 | feature | 0.95 | 0.99 | 0.98 | 0.98 | 0.98 | 0.98 | 0.99 |
| `time_travel` | 1 | feature | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `time_travel` | 2 | feature | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| `time_travel` | 3 | feature | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `time_travel` | 5 | feature | 0.00 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |
| `time_travel` | 10 | feature | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `mean_shift` | 0.01 | feature | 0.46 | 0.45 | 0.00 | 0.01 | 0.99 | 0.00 |
| `mean_shift` | 0.05 | feature | 0.44 | 0.48 | 0.00 | 0.00 | 1.00 | 0.00 |
| `mean_shift` | 0.1 | feature | 0.69 | 0.73 | 0.00 | 0.02 | 1.00 | 0.00 |
| `mean_shift` | 0.25 | feature | 0.97 | 0.98 | 0.03 | 0.17 | 1.00 | 0.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 0.76 | 1.00 | 1.00 | 0.00 |
| `time_travel` | 1 | feature | 0.50 | 0.37 | 0.01 | 0.02 | 0.99 | 0.00 |
| `time_travel` | 2 | feature | 0.54 | 0.38 | 0.00 | 0.02 | 0.99 | 0.00 |
| `time_travel` | 3 | feature | 0.55 | 0.49 | 0.00 | 0.01 | 0.99 | 0.00 |
| `time_travel` | 5 | feature | 0.55 | 0.40 | 0.00 | 0.05 | 1.00 | 0.00 |
| `time_travel` | 10 | feature | 0.56 | 0.42 | 0.00 | 0.03 | 0.98 | 0.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.58 | 0.44 | 0.00 | 0.01 | 0.99 | 0.00 |

## Batch size 10,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.019 [0.016, 0.024] | 0.060 [0.047, 0.076] |
| `l1` (calibrated) | 0.016 [0.012, 0.020] | 0.057 [0.044, 0.073] |
| `kl` (calibrated) | 0.013 [0.010, 0.017] | 0.045 [0.034, 0.060] |
| `js` (calibrated) | 0.013 [0.010, 0.017] | 0.045 [0.034, 0.060] |
| `psi` (calibrated) | 0.013 [0.010, 0.017] | 0.045 [0.034, 0.060] |
| `chi2` (calibrated) | 0.013 [0.010, 0.018] | 0.048 [0.036, 0.063] |
| `ks` (calibrated) | 0.005 [0.003, 0.009] | 0.009 [0.005, 0.017] |
| `chi2 p<0.05` (textbook) | 0.556 [0.541, 0.571] | 0.984 [0.974, 0.990] |
| `ks p<0.05` (textbook) | 0.446 [0.428, 0.464] | 0.734 [0.706, 0.760] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.439 [0.423, 0.454] | 0.955 [0.940, 0.966] |
| `schema` (inferred schema) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `mean_shift` | 0.01 | feature | 0.03 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| `mean_shift` | 0.05 | feature | 0.20 | 0.34 | 0.31 | 0.31 | 0.31 | 0.29 | 0.34 |
| `mean_shift` | 0.1 | feature | 0.57 | 0.59 | 0.62 | 0.62 | 0.62 | 0.62 | 0.65 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `time_travel` | 1 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| `time_travel` | 2 | feature | 0.03 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.05 |
| `time_travel` | 3 | feature | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 |
| `time_travel` | 5 | feature | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 |
| `time_travel` | 10 | feature | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `mean_shift` | 0.01 | feature | 0.30 | 0.35 | 0.00 | 0.00 | 0.20 | 0.00 |
| `mean_shift` | 0.05 | feature | 0.77 | 0.70 | 0.00 | 0.00 | 0.53 | 0.00 |
| `mean_shift` | 0.1 | feature | 0.97 | 0.99 | 0.00 | 0.00 | 0.95 | 0.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 0.00 |
| `time_travel` | 1 | feature | 0.58 | 0.55 | 0.00 | 0.00 | 0.41 | 0.00 |
| `time_travel` | 2 | feature | 0.60 | 0.48 | 0.00 | 0.00 | 0.49 | 0.00 |
| `time_travel` | 3 | feature | 0.51 | 0.43 | 0.00 | 0.00 | 0.39 | 0.00 |
| `time_travel` | 5 | feature | 0.54 | 0.40 | 0.00 | 0.00 | 0.47 | 0.00 |
| `time_travel` | 10 | feature | 0.43 | 0.34 | 0.00 | 0.00 | 0.36 | 0.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.46 | 0.39 | 0.00 | 0.00 | 0.40 | 0.00 |

## Does the alert matter?

Model: gradient boosting trained on the reference split; clean held-out accuracy 0.7939. Accuracy drop = clean minus faulty on held-out batches of 5,000 rows (positive = the fault hurts). Best recall = highest calibrated recall of any distance at batch size 10,000; schema = recall of the schema check at the same batch size.

| Fault | Severity | Accuracy drop [95% CI] | Best recall | Schema |
|---|---|---|---|---|
| `mean_shift` | 0.01 | +0.0001 [-0.0000, +0.0002] | 0.03 | 0.00 |
| `mean_shift` | 0.05 | +0.0002 [+0.0000, +0.0004] | 0.34 | 0.00 |
| `mean_shift` | 0.1 | +0.0004 [+0.0001, +0.0008] | 0.65 | 0.00 |
| `mean_shift` | 0.25 | +0.0052 [+0.0039, +0.0065] | 1.00 | 0.00 |
| `mean_shift` | 0.5 | +0.0173 [+0.0128, +0.0215] | 1.00 | 0.00 |
| `time_travel` | 1 | -0.0509 [-0.0677, -0.0341] | 0.01 | 0.00 |
| `time_travel` | 2 | -0.0007 [-0.0010, -0.0005] | 0.05 | 0.00 |
| `time_travel` | 3 | +0.0187 [+0.0125, +0.0250] | 0.02 | 0.00 |
| `time_travel` | 5 | +0.0352 [+0.0227, +0.0483] | 0.03 | 0.00 |
| `time_travel` | 10 | +0.0699 [+0.0482, +0.0929] | 0.01 | 0.00 |
| `tiny_noise` | 0.0001 | +0.0000 [+0.0000, +0.0000] | 0.03 | 0.00 |

Model loses more than 0.005 accuracy but no detector reaches recall 0.5: `time_travel` at 3, `time_travel` at 5, `time_travel` at 10.

A detector reaches recall 0.9 but accuracy moves by less than 0.005: none.

## What this report does not measure

- Real incidents: every fault is synthetic and follows the documented injector.
- Several faults at once, or a fault that builds up slowly over many batches.
- Calibration on unlimited fresh data: thresholds come from one finite split and are
  applied to another, so the achieved false-positive rate is above the target, and the
  gap grows with the batch size relative to the split size. The achieved rate is shown.
- The model part uses one model family and accuracy only.
