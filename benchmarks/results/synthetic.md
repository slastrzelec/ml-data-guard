# Benchmark report: synthetic

- Data: synthetic, 4,000,000 rows, 5 numeric and 2 categorical features.
- Splits (rows): reference 1,600,000, calibration 1,000,000, evaluation 1,000,000, held-out 400,000.
- Seed 20261010; 1000 calibration batches, 1000 clean batches, 100 faulty batches per fault and severity; target false-positive rate 0.01.
- Commit: de7200a; ml-data-guard 0.1.0, numpy 2.4.6, pandas 3.0.6, python 3.11.17, scikit-learn 1.9.1, scipy 1.17.1.

Rates are followed by 95% Wilson intervals where space allows; every interval is in the
JSON file. With 100 faulty batches a recall of 0.50 has an interval of
about +/- 0.10.

## Batch size 1,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.009 [0.007, 0.011] | 0.059 [0.046, 0.075] |
| `l1` (calibrated) | 0.011 [0.009, 0.014] | 0.077 [0.062, 0.095] |
| `kl` (calibrated) | 0.013 [0.011, 0.016] | 0.090 [0.074, 0.109] |
| `js` (calibrated) | 0.012 [0.010, 0.015] | 0.081 [0.066, 0.100] |
| `psi` (calibrated) | 0.012 [0.010, 0.015] | 0.081 [0.066, 0.100] |
| `chi2` (calibrated) | 0.013 [0.010, 0.016] | 0.087 [0.071, 0.106] |
| `ks` (calibrated) | 0.009 [0.007, 0.012] | 0.045 [0.034, 0.060] |
| `chi2 p<0.05` (textbook) | 0.046 [0.042, 0.051] | 0.289 [0.262, 0.318] |
| `ks p<0.05` (textbook) | 0.035 [0.030, 0.041] | 0.162 [0.140, 0.186] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.932 [0.925, 0.937] | 1.000 [0.996, 1.000] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.07 | 0.10 | 0.11 | 0.10 | 0.10 | 0.10 | 0.08 |
| `duplicate_rows` | 0.005 | batch | 0.04 | 0.08 | 0.07 | 0.07 | 0.07 | 0.09 | 0.05 |
| `duplicate_rows` | 0.01 | batch | 0.04 | 0.08 | 0.10 | 0.10 | 0.10 | 0.11 | 0.08 |
| `duplicate_rows` | 0.05 | batch | 0.13 | 0.08 | 0.13 | 0.10 | 0.10 | 0.10 | 0.08 |
| `duplicate_rows` | 0.2 | batch | 0.22 | 0.27 | 0.40 | 0.37 | 0.37 | 0.39 | 0.12 |
| `mean_shift` | 0.01 | feature | 0.22 | 0.21 | 0.22 | 0.22 | 0.22 | 0.22 | 0.21 |
| `mean_shift` | 0.05 | feature | 0.51 | 0.48 | 0.50 | 0.50 | 0.50 | 0.51 | 0.72 |
| `mean_shift` | 0.1 | feature | 0.50 | 0.54 | 0.60 | 0.60 | 0.60 | 0.59 | 0.74 |
| `mean_shift` | 0.25 | feature | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.02 | 0.01 | 0.04 | 0.03 | 0.03 | 0.93 | 0.01 |
| `missing_rate_increase` | 0.005 | feature | 0.00 | 0.02 | 0.86 | 0.50 | 0.72 | 0.87 | 0.01 |
| `missing_rate_increase` | 0.01 | feature | 0.00 | 0.03 | 0.81 | 0.79 | 0.81 | 0.81 | 0.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `new_category` | 0.001 | feature | 0.04 | 0.02 | 0.03 | 0.02 | 0.02 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 0.01 | 0.00 | 0.96 | 0.56 | 0.64 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 0.01 | 0.00 | 1.00 | 0.86 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| `out_of_domain_numeric` | 0.005 | feature | 0.00 | 0.03 | 0.03 | 0.02 | 0.02 | 0.03 | 0.01 |
| `out_of_domain_numeric` | 0.01 | feature | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 |
| `out_of_domain_numeric` | 0.05 | feature | 0.90 | 0.42 | 0.83 | 0.82 | 0.82 | 0.88 | 0.84 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.00 | 0.02 | 0.00 | 0.01 | 0.01 | 0.50 | 0.02 |
| `pinned_value` | 0.005 | feature | 0.00 | 0.01 | 0.44 | 0.26 | 0.35 | 0.45 | 0.03 |
| `pinned_value` | 0.01 | feature | 0.01 | 0.01 | 0.44 | 0.42 | 0.44 | 0.44 | 0.03 |
| `pinned_value` | 0.05 | feature | 0.96 | 0.63 | 0.91 | 0.88 | 0.88 | 0.92 | 0.82 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.04 | 0.01 |
| `type_change` | 0.001 | feature | 0.00 | 0.02 | 0.02 | 0.02 | 0.02 | 0.79 | 0.00 |
| `type_change` | 0.005 | feature | 0.00 | 0.01 | 0.74 | 0.53 | 0.74 | 0.74 | 0.00 |
| `type_change` | 0.01 | feature | 0.01 | 0.04 | 0.85 | 0.85 | 0.85 | 0.85 | 0.02 |
| `type_change` | 0.05 | feature | 1.00 | 0.63 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `unit_change` | 0.001 | feature | 0.01 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 |
| `unit_change` | 0.005 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.02 |
| `unit_change` | 0.01 | feature | 0.01 | 0.03 | 0.04 | 0.03 | 0.03 | 0.04 | 0.03 |
| `unit_change` | 0.05 | feature | 0.75 | 0.31 | 0.65 | 0.62 | 0.62 | 0.72 | 0.68 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.13 | 0.13 | 0.14 | 0.14 | 0.14 | 0.14 | 0.19 |
| `variance_change` | 0.05 | feature | 0.45 | 0.42 | 0.49 | 0.47 | 0.47 | 0.53 | 0.54 |
| `variance_change` | 0.1 | feature | 0.69 | 0.71 | 0.77 | 0.76 | 0.76 | 0.81 | 0.58 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` |
|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.43 | 0.29 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.005 | batch | 0.24 | 0.20 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.01 | batch | 0.35 | 0.23 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.05 | batch | 0.39 | 0.24 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.2 | batch | 0.75 | 0.34 | 0.00 | 0.00 | 1.00 |
| `mean_shift` | 0.01 | feature | 0.26 | 0.49 | 0.21 | 0.21 | 0.95 |
| `mean_shift` | 0.05 | feature | 0.57 | 0.75 | 0.27 | 0.31 | 0.99 |
| `mean_shift` | 0.1 | feature | 0.69 | 0.82 | 0.36 | 0.37 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.57 | 0.58 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.93 | 0.04 | 0.00 | 0.00 | 0.91 |
| `missing_rate_increase` | 0.005 | feature | 0.87 | 0.04 | 0.00 | 0.00 | 0.91 |
| `missing_rate_increase` | 0.01 | feature | 0.84 | 0.01 | 0.00 | 0.00 | 0.99 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.03 | 0.91 | 0.91 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.07 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.94 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.85 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.89 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.04 | 0.04 | 0.00 | 0.00 | 0.95 |
| `out_of_domain_numeric` | 0.005 | feature | 0.09 | 0.06 | 0.00 | 0.00 | 0.95 |
| `out_of_domain_numeric` | 0.01 | feature | 0.08 | 0.09 | 0.00 | 0.00 | 0.99 |
| `out_of_domain_numeric` | 0.05 | feature | 0.95 | 1.00 | 0.00 | 0.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.55 | 0.12 | 0.00 | 0.00 | 0.90 |
| `pinned_value` | 0.005 | feature | 0.49 | 0.05 | 0.00 | 0.00 | 0.95 |
| `pinned_value` | 0.01 | feature | 0.45 | 0.07 | 0.00 | 0.00 | 0.93 |
| `pinned_value` | 0.05 | feature | 0.96 | 0.98 | 0.47 | 0.47 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.08 | 0.05 | 0.00 | 0.00 | 0.92 |
| `type_change` | 0.001 | feature | 0.81 | 0.01 | 0.00 | 0.00 | 0.98 |
| `type_change` | 0.005 | feature | 0.75 | 0.05 | 0.00 | 0.00 | 0.98 |
| `type_change` | 0.01 | feature | 0.86 | 0.03 | 0.00 | 0.00 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.03 | 0.74 | 0.74 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.01 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.08 | 0.05 | 0.00 | 0.00 | 0.93 |
| `unit_change` | 0.005 | feature | 0.10 | 0.03 | 0.00 | 0.00 | 0.95 |
| `unit_change` | 0.01 | feature | 0.09 | 0.04 | 0.00 | 0.00 | 0.96 |
| `unit_change` | 0.05 | feature | 0.83 | 0.87 | 0.00 | 0.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.72 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.18 | 0.33 | 0.11 | 0.11 | 0.98 |
| `variance_change` | 0.05 | feature | 0.59 | 0.62 | 0.25 | 0.25 | 1.00 |
| `variance_change` | 0.1 | feature | 0.88 | 0.66 | 0.18 | 0.19 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 0.38 | 0.74 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.86 | 1.00 | 1.00 |

Not scorable by a distribution distance (the column is gone; this is a schema check): `dropped_feature`.

## Batch size 10,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.011 [0.009, 0.013] | 0.075 [0.060, 0.093] |
| `l1` (calibrated) | 0.013 [0.010, 0.016] | 0.086 [0.070, 0.105] |
| `kl` (calibrated) | 0.010 [0.008, 0.012] | 0.067 [0.053, 0.084] |
| `js` (calibrated) | 0.010 [0.008, 0.013] | 0.069 [0.055, 0.086] |
| `psi` (calibrated) | 0.010 [0.008, 0.013] | 0.069 [0.055, 0.086] |
| `chi2` (calibrated) | 0.010 [0.008, 0.013] | 0.070 [0.056, 0.088] |
| `ks` (calibrated) | 0.012 [0.009, 0.015] | 0.057 [0.044, 0.073] |
| `chi2 p<0.05` (textbook) | 0.044 [0.039, 0.049] | 0.268 [0.241, 0.296] |
| `ks p<0.05` (textbook) | 0.045 [0.039, 0.051] | 0.205 [0.181, 0.231] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.027 [0.023, 0.031] | 0.175 [0.153, 0.200] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.07 | 0.03 | 0.05 | 0.05 | 0.05 | 0.05 | 0.04 |
| `duplicate_rows` | 0.005 | batch | 0.06 | 0.10 | 0.10 | 0.09 | 0.09 | 0.10 | 0.09 |
| `duplicate_rows` | 0.01 | batch | 0.10 | 0.12 | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 |
| `duplicate_rows` | 0.05 | batch | 0.07 | 0.19 | 0.15 | 0.15 | 0.15 | 0.15 | 0.09 |
| `duplicate_rows` | 0.2 | batch | 0.25 | 0.39 | 0.44 | 0.45 | 0.45 | 0.44 | 0.14 |
| `mean_shift` | 0.01 | feature | 0.41 | 0.37 | 0.43 | 0.43 | 0.43 | 0.43 | 0.60 |
| `mean_shift` | 0.05 | feature | 0.78 | 0.88 | 0.94 | 0.94 | 0.94 | 0.95 | 0.98 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.00 | 0.01 | 0.91 | 0.89 | 0.89 | 0.93 | 0.01 |
| `missing_rate_increase` | 0.005 | feature | 0.02 | 0.10 | 0.94 | 0.94 | 0.94 | 0.94 | 0.00 |
| `missing_rate_increase` | 0.01 | feature | 0.22 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `new_category` | 0.001 | feature | 0.01 | 0.02 | 0.99 | 0.88 | 0.75 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 0.03 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 0.02 | 0.18 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `out_of_domain_numeric` | 0.005 | feature | 0.05 | 0.03 | 0.05 | 0.04 | 0.04 | 0.04 | 0.04 |
| `out_of_domain_numeric` | 0.01 | feature | 0.32 | 0.17 | 0.45 | 0.45 | 0.45 | 0.47 | 0.14 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.00 | 0.00 | 0.49 | 0.45 | 0.41 | 0.50 | 0.02 |
| `pinned_value` | 0.005 | feature | 0.02 | 0.06 | 0.48 | 0.48 | 0.48 | 0.47 | 0.00 |
| `pinned_value` | 0.01 | feature | 0.19 | 0.25 | 0.61 | 0.60 | 0.60 | 0.62 | 0.11 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `type_change` | 0.001 | feature | 0.02 | 0.01 | 0.78 | 0.78 | 0.78 | 0.78 | 0.01 |
| `type_change` | 0.005 | feature | 0.02 | 0.10 | 0.84 | 0.84 | 0.84 | 0.86 | 0.00 |
| `type_change` | 0.01 | feature | 0.38 | 0.29 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `type_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `unit_change` | 0.001 | feature | 0.01 | 0.04 | 0.04 | 0.04 | 0.04 | 0.04 | 0.02 |
| `unit_change` | 0.005 | feature | 0.03 | 0.03 | 0.06 | 0.06 | 0.06 | 0.07 | 0.02 |
| `unit_change` | 0.01 | feature | 0.24 | 0.10 | 0.30 | 0.30 | 0.30 | 0.31 | 0.08 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.45 | 0.39 | 0.49 | 0.49 | 0.49 | 0.49 | 0.59 |
| `variance_change` | 0.05 | feature | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.77 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` |
|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.19 | 0.13 | 0.00 | 0.00 | 0.14 |
| `duplicate_rows` | 0.005 | batch | 0.29 | 0.27 | 0.00 | 0.00 | 0.16 |
| `duplicate_rows` | 0.01 | batch | 0.36 | 0.20 | 0.00 | 0.00 | 0.21 |
| `duplicate_rows` | 0.05 | batch | 0.46 | 0.28 | 0.00 | 0.00 | 0.22 |
| `duplicate_rows` | 0.2 | batch | 0.77 | 0.45 | 0.00 | 0.00 | 0.41 |
| `mean_shift` | 0.01 | feature | 0.48 | 0.67 | 0.21 | 0.21 | 0.42 |
| `mean_shift` | 0.05 | feature | 0.97 | 0.99 | 0.27 | 0.27 | 0.81 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 0.37 | 0.37 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.57 | 0.57 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.93 | 0.03 | 0.00 | 0.00 | 0.01 |
| `missing_rate_increase` | 0.005 | feature | 0.94 | 0.09 | 0.00 | 0.00 | 0.04 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.01 | 0.00 | 0.00 | 0.18 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.03 | 0.91 | 0.91 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.03 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.07 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.12 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.05 | 0.04 | 0.00 | 0.00 | 0.03 |
| `out_of_domain_numeric` | 0.005 | feature | 0.16 | 0.06 | 0.00 | 0.00 | 0.06 |
| `out_of_domain_numeric` | 0.01 | feature | 0.62 | 0.33 | 0.00 | 0.00 | 0.39 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.52 | 0.03 | 0.00 | 0.00 | 0.03 |
| `pinned_value` | 0.005 | feature | 0.50 | 0.05 | 0.00 | 0.00 | 0.03 |
| `pinned_value` | 0.01 | feature | 0.71 | 0.24 | 0.00 | 0.00 | 0.24 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.47 | 0.47 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.04 | 0.05 | 0.00 | 0.00 | 0.02 |
| `type_change` | 0.001 | feature | 0.78 | 0.05 | 0.00 | 0.00 | 0.06 |
| `type_change` | 0.005 | feature | 0.87 | 0.04 | 0.00 | 0.00 | 0.03 |
| `type_change` | 0.01 | feature | 1.00 | 0.06 | 0.00 | 0.00 | 0.16 |
| `type_change` | 0.05 | feature | 1.00 | 0.03 | 0.74 | 0.74 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.08 | 0.05 | 0.00 | 0.00 | 0.00 |
| `unit_change` | 0.005 | feature | 0.18 | 0.06 | 0.00 | 0.00 | 0.04 |
| `unit_change` | 0.01 | feature | 0.51 | 0.28 | 0.00 | 0.00 | 0.33 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.77 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.59 | 0.66 | 0.11 | 0.11 | 0.45 |
| `variance_change` | 0.05 | feature | 1.00 | 0.89 | 0.25 | 0.25 | 0.96 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 0.18 | 0.18 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 0.38 | 0.78 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 |

Not scorable by a distribution distance (the column is gone; this is a schema check): `dropped_feature`.

## Batch size 100,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.011 [0.009, 0.014] | 0.076 [0.061, 0.094] |
| `l1` (calibrated) | 0.017 [0.014, 0.020] | 0.114 [0.096, 0.135] |
| `kl` (calibrated) | 0.015 [0.012, 0.018] | 0.104 [0.087, 0.124] |
| `js` (calibrated) | 0.015 [0.012, 0.018] | 0.103 [0.086, 0.123] |
| `psi` (calibrated) | 0.015 [0.012, 0.018] | 0.103 [0.086, 0.123] |
| `chi2` (calibrated) | 0.014 [0.012, 0.018] | 0.100 [0.083, 0.120] |
| `ks` (calibrated) | 0.012 [0.009, 0.015] | 0.059 [0.046, 0.075] |
| `chi2 p<0.05` (textbook) | 0.053 [0.048, 0.059] | 0.321 [0.293, 0.351] |
| `ks p<0.05` (textbook) | 0.040 [0.035, 0.045] | 0.191 [0.168, 0.217] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.004] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.03 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.02 |
| `duplicate_rows` | 0.005 | batch | 0.15 | 0.15 | 0.14 | 0.14 | 0.14 | 0.14 | 0.12 |
| `duplicate_rows` | 0.01 | batch | 0.05 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.07 |
| `duplicate_rows` | 0.05 | batch | 0.10 | 0.15 | 0.10 | 0.10 | 0.10 | 0.10 | 0.09 |
| `duplicate_rows` | 0.2 | batch | 0.16 | 0.37 | 0.37 | 0.37 | 0.37 | 0.37 | 0.20 |
| `mean_shift` | 0.01 | feature | 0.66 | 0.65 | 0.72 | 0.72 | 0.72 | 0.72 | 0.83 |
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.00 | 0.05 | 0.93 | 0.93 | 0.93 | 0.93 | 0.01 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.77 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.03 |
| `new_category` | 0.001 | feature | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 1.00 | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| `out_of_domain_numeric` | 0.005 | feature | 0.88 | 0.52 | 0.82 | 0.82 | 0.82 | 0.83 | 0.85 |
| `out_of_domain_numeric` | 0.01 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.02 | 0.06 | 0.50 | 0.50 | 0.50 | 0.50 | 0.02 |
| `pinned_value` | 0.005 | feature | 0.90 | 0.56 | 0.85 | 0.85 | 0.85 | 0.85 | 0.61 |
| `pinned_value` | 0.01 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 | 0.10 | 0.01 |
| `type_change` | 0.001 | feature | 0.01 | 0.04 | 0.80 | 0.80 | 0.80 | 0.80 | 0.01 |
| `type_change` | 0.005 | feature | 1.00 | 0.71 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `type_change` | 0.01 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `type_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `unit_change` | 0.001 | feature | 0.03 | 0.04 | 0.06 | 0.06 | 0.06 | 0.06 | 0.04 |
| `unit_change` | 0.005 | feature | 0.79 | 0.45 | 0.74 | 0.74 | 0.74 | 0.74 | 0.69 |
| `unit_change` | 0.01 | feature | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` |
|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.30 | 0.16 | 0.00 | 0.00 | 0.00 |
| `duplicate_rows` | 0.005 | batch | 0.32 | 0.25 | 0.00 | 0.00 | 0.00 |
| `duplicate_rows` | 0.01 | batch | 0.39 | 0.15 | 0.00 | 0.00 | 0.00 |
| `duplicate_rows` | 0.05 | batch | 0.41 | 0.20 | 0.00 | 0.00 | 0.00 |
| `duplicate_rows` | 0.2 | batch | 0.74 | 0.43 | 0.00 | 0.00 | 0.00 |
| `mean_shift` | 0.01 | feature | 0.83 | 0.83 | 0.21 | 0.21 | 0.39 |
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 0.27 | 0.27 | 0.77 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 0.37 | 0.37 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.57 | 0.57 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.93 | 0.03 | 0.00 | 0.00 | 0.00 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.08 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.08 | 0.91 | 0.91 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.11 | 0.08 | 0.00 | 0.00 | 0.00 |
| `out_of_domain_numeric` | 0.005 | feature | 0.94 | 1.00 | 0.00 | 0.00 | 0.00 |
| `out_of_domain_numeric` | 0.01 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 0.26 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.51 | 0.05 | 0.00 | 0.00 | 0.00 |
| `pinned_value` | 0.005 | feature | 0.95 | 0.95 | 0.00 | 0.00 | 0.00 |
| `pinned_value` | 0.01 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 0.05 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.47 | 0.47 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.13 | 0.04 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.001 | feature | 0.80 | 0.04 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.05 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.01 | feature | 1.00 | 0.05 | 0.00 | 0.00 | 0.07 |
| `type_change` | 0.05 | feature | 1.00 | 0.03 | 0.74 | 0.74 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.07 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.11 | 0.05 | 0.00 | 0.00 | 0.00 |
| `unit_change` | 0.005 | feature | 0.83 | 0.85 | 0.00 | 0.00 | 0.00 |
| `unit_change` | 0.01 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 0.14 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.83 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 1.00 | 0.99 | 0.11 | 0.11 | 0.42 |
| `variance_change` | 0.05 | feature | 1.00 | 1.00 | 0.25 | 0.25 | 1.00 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 0.18 | 0.18 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 0.38 | 0.80 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.79 | 1.00 | 1.00 |

Not scorable by a distribution distance (the column is gone; this is a schema check): `dropped_feature`.

## Does the alert matter?

Model: gradient boosting trained on the reference split; clean held-out accuracy 0.7180. Accuracy drop = clean minus faulty on held-out batches of 5,000 rows (positive = the fault hurts). Best recall = highest calibrated recall of any detector at batch size 10,000.

| Fault | Severity | Accuracy drop [95% CI] | Best recall |
|---|---|---|---|
| `dropped_feature` | 1 | +0.0175 [+0.0147, +0.0202] | n/a |
| `duplicate_rows` | 0.001 | +0.0000 [-0.0000, +0.0001] | 0.07 |
| `duplicate_rows` | 0.005 | +0.0000 [-0.0001, +0.0001] | 0.10 |
| `duplicate_rows` | 0.01 | -0.0001 [-0.0003, +0.0001] | 0.12 |
| `duplicate_rows` | 0.05 | +0.0001 [-0.0003, +0.0005] | 0.19 |
| `duplicate_rows` | 0.2 | +0.0003 [-0.0005, +0.0011] | 0.45 |
| `mean_shift` | 0.01 | +0.0009 [+0.0005, +0.0013] | 0.60 |
| `mean_shift` | 0.05 | +0.0011 [+0.0007, +0.0015] | 0.98 |
| `mean_shift` | 0.1 | +0.0011 [+0.0006, +0.0015] | 1.00 |
| `mean_shift` | 0.25 | +0.0044 [+0.0034, +0.0054] | 1.00 |
| `mean_shift` | 0.5 | +0.0115 [+0.0094, +0.0136] | 1.00 |
| `missing_rate_increase` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.93 |
| `missing_rate_increase` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.94 |
| `missing_rate_increase` | 0.01 | +0.0003 [+0.0001, +0.0004] | 1.00 |
| `missing_rate_increase` | 0.05 | +0.0009 [+0.0006, +0.0011] | 1.00 |
| `missing_rate_increase` | 0.2 | +0.0034 [+0.0028, +0.0039] | 1.00 |
| `new_category` | 0.001 | +0.0000 [-0.0000, +0.0000] | 1.00 |
| `new_category` | 0.005 | +0.0000 [-0.0000, +0.0001] | 1.00 |
| `new_category` | 0.01 | +0.0001 [-0.0000, +0.0001] | 1.00 |
| `new_category` | 0.05 | +0.0005 [+0.0004, +0.0007] | 1.00 |
| `new_category` | 0.2 | +0.0020 [+0.0015, +0.0025] | 1.00 |
| `out_of_domain_numeric` | 0.001 | +0.0001 [+0.0001, +0.0002] | 0.01 |
| `out_of_domain_numeric` | 0.005 | +0.0008 [+0.0006, +0.0009] | 0.05 |
| `out_of_domain_numeric` | 0.01 | +0.0011 [+0.0009, +0.0013] | 0.47 |
| `out_of_domain_numeric` | 0.05 | +0.0066 [+0.0061, +0.0071] | 1.00 |
| `out_of_domain_numeric` | 0.2 | +0.0269 [+0.0252, +0.0286] | 1.00 |
| `pinned_value` | 0.001 | +0.0001 [+0.0000, +0.0001] | 0.50 |
| `pinned_value` | 0.005 | +0.0003 [+0.0002, +0.0004] | 0.48 |
| `pinned_value` | 0.01 | +0.0007 [+0.0005, +0.0010] | 0.62 |
| `pinned_value` | 0.05 | +0.0036 [+0.0028, +0.0045] | 1.00 |
| `pinned_value` | 0.2 | +0.0127 [+0.0099, +0.0159] | 1.00 |
| `tiny_noise` | 0.0001 | +0.0000 [-0.0000, +0.0000] | 0.01 |
| `type_change` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.78 |
| `type_change` | 0.005 | +0.0001 [+0.0000, +0.0002] | 0.86 |
| `type_change` | 0.01 | +0.0002 [+0.0001, +0.0003] | 1.00 |
| `type_change` | 0.05 | +0.0012 [+0.0009, +0.0014] | 1.00 |
| `type_change` | 0.2 | +0.0042 [+0.0035, +0.0049] | 1.00 |
| `unit_change` | 0.001 | +0.0002 [+0.0001, +0.0002] | 0.04 |
| `unit_change` | 0.005 | +0.0005 [+0.0004, +0.0006] | 0.07 |
| `unit_change` | 0.01 | +0.0011 [+0.0009, +0.0013] | 0.31 |
| `unit_change` | 0.05 | +0.0053 [+0.0049, +0.0058] | 1.00 |
| `unit_change` | 0.2 | +0.0231 [+0.0215, +0.0249] | 1.00 |
| `variance_change` | 0.02 | +0.0002 [+0.0001, +0.0003] | 0.59 |
| `variance_change` | 0.05 | +0.0004 [+0.0002, +0.0006] | 1.00 |
| `variance_change` | 0.1 | +0.0009 [+0.0006, +0.0013] | 1.00 |
| `variance_change` | 0.25 | +0.0040 [+0.0032, +0.0048] | 1.00 |
| `variance_change` | 0.5 | +0.0071 [+0.0056, +0.0086] | 1.00 |

Model loses more than 0.005 accuracy but no detector reaches recall 0.5: none.

A detector reaches recall 0.9 but accuracy moves by less than 0.005: `mean_shift` at 0.05, `mean_shift` at 0.1, `mean_shift` at 0.25, `missing_rate_increase` at 0.001, `missing_rate_increase` at 0.005, `missing_rate_increase` at 0.01, `missing_rate_increase` at 0.05, `missing_rate_increase` at 0.2, `new_category` at 0.001, `new_category` at 0.005, `new_category` at 0.01, `new_category` at 0.05, `new_category` at 0.2, `pinned_value` at 0.05, `type_change` at 0.01, `type_change` at 0.05, `type_change` at 0.2, `variance_change` at 0.05, `variance_change` at 0.1, `variance_change` at 0.25.

## What this report does not measure

- Real incidents: every fault is synthetic and follows the documented injector.
- Several faults at once, or a fault that builds up slowly over many batches.
- Calibration on unlimited fresh data: thresholds come from one finite split and are
  applied to another, so the achieved false-positive rate is above the target, and the
  gap grows with the batch size relative to the split size. The achieved rate is shown.
- The model part uses one model family and accuracy only.
