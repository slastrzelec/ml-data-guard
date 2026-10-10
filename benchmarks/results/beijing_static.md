# Benchmark report: beijing

- Data: UCI Beijing Multi-Site Air Quality, CC BY 4.0, https://doi.org/10.24432/C5RK5G, 412,029 rows, 12 numeric and 2 categorical features.
- Splits (rows): reference 164,811, calibration 103,007, evaluation 103,008, held-out 41,203.
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
| `linf` (calibrated) | 0.033 [0.030, 0.036] | 0.283 [0.256, 0.312] |
| `l1` (calibrated) | 0.030 [0.027, 0.033] | 0.238 [0.213, 0.265] |
| `kl` (calibrated) | 0.028 [0.025, 0.031] | 0.231 [0.206, 0.258] |
| `js` (calibrated) | 0.030 [0.027, 0.033] | 0.231 [0.206, 0.258] |
| `psi` (calibrated) | 0.027 [0.024, 0.029] | 0.211 [0.187, 0.237] |
| `chi2` (calibrated) | 0.019 [0.017, 0.021] | 0.162 [0.140, 0.186] |
| `ks` (calibrated) | 0.026 [0.023, 0.029] | 0.186 [0.163, 0.211] |
| `chi2 p<0.05` (textbook) | 0.924 [0.920, 0.929] | 1.000 [0.996, 1.000] |
| `ks p<0.05` (textbook) | 0.932 [0.927, 0.936] | 1.000 [0.996, 1.000] |
| `psi>0.2` (textbook) | 0.853 [0.847, 0.858] | 1.000 [0.996, 1.000] |
| `psi>0.1` (textbook) | 0.891 [0.886, 0.896] | 1.000 [0.996, 1.000] |
| `linf>0.01` (textbook) | 0.941 [0.937, 0.945] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.038 [0.035, 0.041] | 0.322 [0.294, 0.352] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.26 | 0.16 | 0.18 | 0.16 | 0.15 | 0.16 | 0.15 |
| `duplicate_rows` | 0.005 | batch | 0.28 | 0.24 | 0.25 | 0.24 | 0.23 | 0.23 | 0.18 |
| `duplicate_rows` | 0.01 | batch | 0.29 | 0.26 | 0.24 | 0.23 | 0.22 | 0.16 | 0.22 |
| `duplicate_rows` | 0.05 | batch | 0.27 | 0.24 | 0.23 | 0.23 | 0.25 | 0.21 | 0.17 |
| `duplicate_rows` | 0.2 | batch | 0.27 | 0.24 | 0.23 | 0.22 | 0.22 | 0.13 | 0.20 |
| `mean_shift` | 0.01 | feature | 0.13 | 0.23 | 0.28 | 0.28 | 0.30 | 0.20 | 0.10 |
| `mean_shift` | 0.05 | feature | 0.09 | 0.19 | 0.26 | 0.26 | 0.27 | 0.21 | 0.06 |
| `mean_shift` | 0.1 | feature | 0.13 | 0.19 | 0.23 | 0.22 | 0.24 | 0.22 | 0.09 |
| `mean_shift` | 0.25 | feature | 0.05 | 0.16 | 0.24 | 0.18 | 0.23 | 0.18 | 0.03 |
| `mean_shift` | 0.5 | feature | 0.09 | 0.24 | 0.28 | 0.27 | 0.28 | 0.19 | 0.14 |
| `missing_rate_increase` | 0.001 | feature | 0.08 | 0.06 | 0.06 | 0.06 | 0.07 | 0.05 | 0.05 |
| `missing_rate_increase` | 0.005 | feature | 0.04 | 0.03 | 0.03 | 0.04 | 0.03 | 0.10 | 0.05 |
| `missing_rate_increase` | 0.01 | feature | 0.04 | 0.02 | 0.03 | 0.02 | 0.02 | 0.07 | 0.02 |
| `missing_rate_increase` | 0.05 | feature | 0.00 | 0.01 | 0.11 | 0.02 | 0.01 | 0.43 | 0.01 |
| `missing_rate_increase` | 0.2 | feature | 0.17 | 0.12 | 0.40 | 0.20 | 0.33 | 0.58 | 0.01 |
| `new_category` | 0.001 | feature | 0.05 | 0.03 | 0.02 | 0.02 | 0.02 | 0.56 | n/a |
| `new_category` | 0.005 | feature | 0.03 | 0.04 | 0.04 | 0.03 | 0.03 | 0.50 | n/a |
| `new_category` | 0.01 | feature | 0.04 | 0.02 | 0.02 | 0.02 | 0.02 | 0.55 | n/a |
| `new_category` | 0.05 | feature | 0.04 | 0.08 | 0.63 | 0.08 | 0.09 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 0.55 | 1.00 | 0.62 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.01 | 0.03 |
| `out_of_domain_numeric` | 0.005 | feature | 0.03 | 0.05 | 0.05 | 0.05 | 0.02 | 0.03 | 0.05 |
| `out_of_domain_numeric` | 0.01 | feature | 0.03 | 0.02 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 |
| `out_of_domain_numeric` | 0.05 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 |
| `out_of_domain_numeric` | 0.2 | feature | 0.08 | 0.05 | 0.05 | 0.03 | 0.02 | 0.03 | 0.05 |
| `pinned_value` | 0.001 | feature | 0.05 | 0.04 | 0.03 | 0.04 | 0.03 | 0.09 | 0.02 |
| `pinned_value` | 0.005 | feature | 0.04 | 0.03 | 0.03 | 0.03 | 0.04 | 0.11 | 0.02 |
| `pinned_value` | 0.01 | feature | 0.04 | 0.03 | 0.03 | 0.05 | 0.02 | 0.07 | 0.05 |
| `pinned_value` | 0.05 | feature | 0.04 | 0.02 | 0.12 | 0.02 | 0.01 | 0.22 | 0.02 |
| `pinned_value` | 0.2 | feature | 0.23 | 0.17 | 0.28 | 0.21 | 0.28 | 0.26 | 0.04 |
| `time_travel` | 1 | feature | 0.05 | 0.06 | 0.06 | 0.07 | 0.05 | 0.03 | 0.04 |
| `time_travel` | 2 | feature | 0.02 | 0.03 | 0.02 | 0.04 | 0.04 | 0.01 | 0.03 |
| `time_travel` | 3 | feature | 0.04 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 |
| `time_travel` | 5 | feature | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 |
| `time_travel` | 10 | feature | 0.10 | 0.07 | 0.07 | 0.06 | 0.04 | 0.04 | 0.08 |
| `tiny_noise` (control) | 0.0001 | feature | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.001 | feature | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |
| `type_change` | 0.005 | feature | 0.02 | 0.00 | 0.00 | 0.01 | 0.02 | 0.00 | 0.01 |
| `type_change` | 0.01 | feature | 0.06 | 0.05 | 0.06 | 0.05 | 0.04 | 0.03 | 0.04 |
| `type_change` | 0.05 | feature | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.28 | 0.01 |
| `type_change` | 0.2 | feature | 0.01 | 0.05 | 0.31 | 0.15 | 0.23 | 0.59 | 0.04 |
| `unit_change` | 0.001 | feature | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| `unit_change` | 0.005 | feature | 0.03 | 0.03 | 0.03 | 0.03 | 0.01 | 0.03 | 0.03 |
| `unit_change` | 0.01 | feature | 0.05 | 0.06 | 0.05 | 0.06 | 0.04 | 0.02 | 0.06 |
| `unit_change` | 0.05 | feature | 0.03 | 0.01 | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 |
| `unit_change` | 0.2 | feature | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.02 | 0.00 |
| `variance_change` | 0.02 | feature | 0.07 | 0.14 | 0.26 | 0.21 | 0.27 | 0.25 | 0.08 |
| `variance_change` | 0.05 | feature | 0.08 | 0.12 | 0.22 | 0.19 | 0.22 | 0.19 | 0.06 |
| `variance_change` | 0.1 | feature | 0.08 | 0.14 | 0.21 | 0.23 | 0.26 | 0.18 | 0.08 |
| `variance_change` | 0.25 | feature | 0.10 | 0.18 | 0.24 | 0.24 | 0.27 | 0.22 | 0.08 |
| `variance_change` | 0.5 | feature | 0.12 | 0.16 | 0.22 | 0.21 | 0.22 | 0.19 | 0.08 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `dropped_feature` | 1 | feature | n/a | n/a | n/a | n/a | n/a | 1.00 |
| `duplicate_rows` | 0.001 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.43 |
| `duplicate_rows` | 0.005 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 |
| `duplicate_rows` | 0.01 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.41 |
| `duplicate_rows` | 0.05 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| `duplicate_rows` | 0.2 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.32 |
| `mean_shift` | 0.01 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.03 |
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 0.01 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.07 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.06 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `missing_rate_increase` | 0.001 | feature | 1.00 | 0.97 | 0.90 | 0.96 | 0.97 | 0.12 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.89 | 0.86 | 0.90 | 0.92 | 0.24 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.93 | 0.88 | 0.91 | 0.94 | 0.21 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 0.81 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.36 | 0.46 | 0.63 | 1.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.44 | 0.51 | 0.60 | 1.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.40 | 0.47 | 0.53 | 1.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 1.00 | 0.95 | 0.90 | 0.99 | 1.00 | 0.10 |
| `out_of_domain_numeric` | 0.005 | feature | 0.99 | 0.92 | 0.89 | 0.91 | 0.99 | 0.16 |
| `out_of_domain_numeric` | 0.01 | feature | 0.99 | 0.93 | 0.91 | 0.93 | 0.99 | 0.19 |
| `out_of_domain_numeric` | 0.05 | feature | 0.94 | 0.94 | 0.84 | 0.91 | 1.00 | 0.26 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.31 |
| `pinned_value` | 0.001 | feature | 1.00 | 0.93 | 0.86 | 0.90 | 0.92 | 0.86 |
| `pinned_value` | 0.005 | feature | 1.00 | 0.88 | 0.83 | 0.88 | 0.93 | 0.84 |
| `pinned_value` | 0.01 | feature | 1.00 | 0.98 | 0.92 | 0.94 | 0.98 | 0.88 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.85 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.86 |
| `time_travel` | 1 | feature | 0.99 | 0.93 | 0.91 | 0.96 | 0.99 | 0.04 |
| `time_travel` | 2 | feature | 1.00 | 0.95 | 0.94 | 0.99 | 1.00 | 0.00 |
| `time_travel` | 3 | feature | 1.00 | 0.92 | 0.92 | 0.99 | 1.00 | 0.07 |
| `time_travel` | 5 | feature | 0.99 | 0.90 | 0.89 | 0.94 | 0.99 | 0.04 |
| `time_travel` | 10 | feature | 1.00 | 0.92 | 0.94 | 0.97 | 1.00 | 0.01 |
| `tiny_noise` (control) | 0.0001 | feature | 0.95 | 0.91 | 0.87 | 0.92 | 0.96 | 0.03 |
| `type_change` | 0.001 | feature | 1.00 | 0.92 | 0.92 | 0.94 | 1.00 | 1.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.91 | 0.93 | 0.96 | 1.00 | 1.00 |
| `type_change` | 0.01 | feature | 1.00 | 0.98 | 0.96 | 0.98 | 1.00 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 1.00 | 0.99 | 0.94 | 0.98 | 1.00 | 1.00 |
| `unit_change` | 0.005 | feature | 1.00 | 0.94 | 0.92 | 0.93 | 1.00 | 1.00 |
| `unit_change` | 0.01 | feature | 1.00 | 0.96 | 0.93 | 0.96 | 1.00 | 1.00 |
| `unit_change` | 0.05 | feature | 0.99 | 0.94 | 0.85 | 0.92 | 0.99 | 1.00 |
| `unit_change` | 0.2 | feature | 0.99 | 0.94 | 0.94 | 0.95 | 0.99 | 1.00 |
| `variance_change` | 0.02 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.10 |
| `variance_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.27 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 |

Not scorable by a distribution distance (the column is gone; only the schema check applies): `dropped_feature`.

## Batch size 10,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.102 [0.097, 0.107] | 0.589 [0.558, 0.619] |
| `l1` (calibrated) | 0.075 [0.071, 0.080] | 0.598 [0.567, 0.628] |
| `kl` (calibrated) | 0.077 [0.072, 0.081] | 0.653 [0.623, 0.682] |
| `js` (calibrated) | 0.088 [0.083, 0.092] | 0.726 [0.698, 0.753] |
| `psi` (calibrated) | 0.090 [0.085, 0.095] | 0.791 [0.765, 0.815] |
| `chi2` (calibrated) | 0.081 [0.077, 0.086] | 0.606 [0.575, 0.636] |
| `ks` (calibrated) | 0.088 [0.083, 0.093] | 0.607 [0.576, 0.637] |
| `chi2 p<0.05` (textbook) | 0.921 [0.917, 0.926] | 1.000 [0.996, 1.000] |
| `ks p<0.05` (textbook) | 0.979 [0.976, 0.981] | 1.000 [0.996, 1.000] |
| `psi>0.2` (textbook) | 0.691 [0.684, 0.699] | 1.000 [0.996, 1.000] |
| `psi>0.1` (textbook) | 0.812 [0.805, 0.818] | 1.000 [0.996, 1.000] |
| `linf>0.01` (textbook) | 0.917 [0.913, 0.922] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.048 [0.044, 0.051] | 0.402 [0.372, 0.433] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.62 | 0.63 | 0.67 | 0.81 | 0.87 | 0.64 | 0.65 |
| `duplicate_rows` | 0.005 | batch | 0.54 | 0.56 | 0.59 | 0.66 | 0.68 | 0.58 | 0.58 |
| `duplicate_rows` | 0.01 | batch | 0.57 | 0.52 | 0.58 | 0.65 | 0.68 | 0.52 | 0.51 |
| `duplicate_rows` | 0.05 | batch | 0.54 | 0.58 | 0.63 | 0.68 | 0.73 | 0.54 | 0.59 |
| `duplicate_rows` | 0.2 | batch | 0.59 | 0.58 | 0.65 | 0.66 | 0.67 | 0.58 | 0.57 |
| `mean_shift` | 0.01 | feature | 0.22 | 0.37 | 0.41 | 0.42 | 0.42 | 0.40 | 0.20 |
| `mean_shift` | 0.05 | feature | 0.24 | 0.34 | 0.39 | 0.38 | 0.42 | 0.41 | 0.16 |
| `mean_shift` | 0.1 | feature | 0.24 | 0.33 | 0.34 | 0.36 | 0.40 | 0.33 | 0.17 |
| `mean_shift` | 0.25 | feature | 0.18 | 0.26 | 0.32 | 0.35 | 0.47 | 0.29 | 0.18 |
| `mean_shift` | 0.5 | feature | 0.32 | 0.50 | 0.53 | 0.61 | 0.72 | 0.43 | 0.35 |
| `missing_rate_increase` | 0.001 | feature | 0.08 | 0.07 | 0.07 | 0.07 | 0.08 | 0.11 | 0.09 |
| `missing_rate_increase` | 0.005 | feature | 0.09 | 0.09 | 0.19 | 0.11 | 0.20 | 0.14 | 0.12 |
| `missing_rate_increase` | 0.01 | feature | 0.06 | 0.08 | 0.18 | 0.19 | 0.19 | 0.15 | 0.07 |
| `missing_rate_increase` | 0.05 | feature | 0.19 | 0.21 | 0.37 | 0.28 | 0.34 | 0.44 | 0.11 |
| `missing_rate_increase` | 0.2 | feature | 0.54 | 0.37 | 0.74 | 0.51 | 0.52 | 0.80 | 0.11 |
| `new_category` | 0.001 | feature | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 | 0.55 | n/a |
| `new_category` | 0.005 | feature | 0.01 | 0.01 | 0.50 | 0.01 | 0.50 | 0.53 | n/a |
| `new_category` | 0.01 | feature | 0.03 | 0.03 | 0.59 | 0.56 | 0.57 | 0.64 | n/a |
| `new_category` | 0.05 | feature | 0.48 | 0.51 | 1.00 | 0.57 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.13 | 0.07 | 0.09 | 0.08 | 0.08 | 0.09 | 0.07 |
| `out_of_domain_numeric` | 0.005 | feature | 0.10 | 0.08 | 0.09 | 0.08 | 0.05 | 0.08 | 0.08 |
| `out_of_domain_numeric` | 0.01 | feature | 0.15 | 0.07 | 0.08 | 0.09 | 0.05 | 0.08 | 0.07 |
| `out_of_domain_numeric` | 0.05 | feature | 0.22 | 0.16 | 0.20 | 0.16 | 0.14 | 0.16 | 0.16 |
| `out_of_domain_numeric` | 0.2 | feature | 0.43 | 0.24 | 0.25 | 0.21 | 0.21 | 0.30 | 0.21 |
| `pinned_value` | 0.001 | feature | 0.13 | 0.09 | 0.08 | 0.09 | 0.08 | 0.17 | 0.10 |
| `pinned_value` | 0.005 | feature | 0.11 | 0.08 | 0.20 | 0.10 | 0.17 | 0.18 | 0.11 |
| `pinned_value` | 0.01 | feature | 0.12 | 0.05 | 0.13 | 0.13 | 0.11 | 0.11 | 0.06 |
| `pinned_value` | 0.05 | feature | 0.18 | 0.14 | 0.28 | 0.20 | 0.28 | 0.26 | 0.07 |
| `pinned_value` | 0.2 | feature | 0.46 | 0.37 | 0.46 | 0.41 | 0.40 | 0.49 | 0.19 |
| `time_travel` | 1 | feature | 0.17 | 0.12 | 0.14 | 0.13 | 0.13 | 0.15 | 0.15 |
| `time_travel` | 2 | feature | 0.13 | 0.13 | 0.11 | 0.12 | 0.12 | 0.11 | 0.14 |
| `time_travel` | 3 | feature | 0.06 | 0.10 | 0.08 | 0.10 | 0.12 | 0.07 | 0.12 |
| `time_travel` | 5 | feature | 0.14 | 0.05 | 0.07 | 0.08 | 0.08 | 0.10 | 0.06 |
| `time_travel` | 10 | feature | 0.12 | 0.06 | 0.08 | 0.08 | 0.09 | 0.10 | 0.04 |
| `tiny_noise` (control) | 0.0001 | feature | 0.11 | 0.06 | 0.08 | 0.08 | 0.08 | 0.08 | 0.07 |
| `type_change` | 0.001 | feature | 0.10 | 0.06 | 0.06 | 0.06 | 0.07 | 0.07 | 0.07 |
| `type_change` | 0.005 | feature | 0.13 | 0.07 | 0.09 | 0.09 | 0.07 | 0.11 | 0.09 |
| `type_change` | 0.01 | feature | 0.07 | 0.07 | 0.06 | 0.08 | 0.09 | 0.07 | 0.06 |
| `type_change` | 0.05 | feature | 0.11 | 0.09 | 0.25 | 0.15 | 0.19 | 0.40 | 0.06 |
| `type_change` | 0.2 | feature | 0.38 | 0.26 | 0.72 | 0.36 | 0.39 | 0.79 | 0.09 |
| `unit_change` | 0.001 | feature | 0.15 | 0.10 | 0.11 | 0.11 | 0.06 | 0.13 | 0.11 |
| `unit_change` | 0.005 | feature | 0.09 | 0.07 | 0.07 | 0.06 | 0.06 | 0.07 | 0.07 |
| `unit_change` | 0.01 | feature | 0.11 | 0.09 | 0.11 | 0.10 | 0.10 | 0.09 | 0.10 |
| `unit_change` | 0.05 | feature | 0.17 | 0.08 | 0.08 | 0.10 | 0.10 | 0.10 | 0.07 |
| `unit_change` | 0.2 | feature | 0.20 | 0.08 | 0.10 | 0.08 | 0.06 | 0.11 | 0.07 |
| `variance_change` | 0.02 | feature | 0.21 | 0.41 | 0.43 | 0.44 | 0.43 | 0.40 | 0.20 |
| `variance_change` | 0.05 | feature | 0.23 | 0.31 | 0.33 | 0.32 | 0.32 | 0.35 | 0.18 |
| `variance_change` | 0.1 | feature | 0.26 | 0.28 | 0.31 | 0.31 | 0.35 | 0.36 | 0.16 |
| `variance_change` | 0.25 | feature | 0.45 | 0.39 | 0.42 | 0.41 | 0.39 | 0.49 | 0.25 |
| `variance_change` | 0.5 | feature | 0.43 | 0.37 | 0.43 | 0.41 | 0.41 | 0.49 | 0.26 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `dropped_feature` | 1 | feature | n/a | n/a | n/a | n/a | n/a | 1.00 |
| `duplicate_rows` | 0.001 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.42 |
| `duplicate_rows` | 0.005 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.39 |
| `duplicate_rows` | 0.01 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 |
| `duplicate_rows` | 0.05 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| `duplicate_rows` | 0.2 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.34 |
| `mean_shift` | 0.01 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.00 |
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.08 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 0.05 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.13 |
| `missing_rate_increase` | 0.001 | feature | 1.00 | 0.99 | 0.67 | 0.87 | 0.96 | 0.09 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.94 | 0.67 | 0.80 | 0.89 | 0.26 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 1.00 | 0.75 | 0.86 | 0.94 | 0.34 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.98 | 0.90 | 0.97 | 1.00 | 0.83 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.13 | 0.33 | 0.49 | 1.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.17 | 0.40 | 0.53 | 1.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.24 | 0.46 | 0.50 | 1.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.99 | 0.99 | 0.79 | 0.92 | 0.99 | 0.63 |
| `out_of_domain_numeric` | 0.005 | feature | 1.00 | 0.99 | 0.79 | 0.85 | 0.99 | 0.65 |
| `out_of_domain_numeric` | 0.01 | feature | 1.00 | 0.98 | 0.83 | 0.93 | 0.99 | 0.58 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 0.64 | 0.83 | 1.00 | 0.66 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.61 |
| `pinned_value` | 0.001 | feature | 1.00 | 0.98 | 0.66 | 0.78 | 0.89 | 0.85 |
| `pinned_value` | 0.005 | feature | 1.00 | 0.98 | 0.62 | 0.79 | 0.92 | 0.84 |
| `pinned_value` | 0.01 | feature | 1.00 | 1.00 | 0.67 | 0.86 | 0.95 | 0.86 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.87 | 0.96 | 1.00 | 0.85 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 |
| `time_travel` | 1 | feature | 0.99 | 0.97 | 0.79 | 0.88 | 0.97 | 0.08 |
| `time_travel` | 2 | feature | 0.99 | 0.98 | 0.76 | 0.92 | 0.98 | 0.05 |
| `time_travel` | 3 | feature | 0.98 | 0.97 | 0.77 | 0.88 | 0.97 | 0.03 |
| `time_travel` | 5 | feature | 0.99 | 0.98 | 0.64 | 0.83 | 0.98 | 0.05 |
| `time_travel` | 10 | feature | 1.00 | 1.00 | 0.75 | 0.90 | 1.00 | 0.07 |
| `tiny_noise` (control) | 0.0001 | feature | 0.93 | 0.98 | 0.69 | 0.75 | 0.93 | 0.03 |
| `type_change` | 0.001 | feature | 1.00 | 0.95 | 0.75 | 0.82 | 0.96 | 1.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.96 | 0.80 | 0.88 | 0.99 | 1.00 |
| `type_change` | 0.01 | feature | 1.00 | 0.99 | 0.84 | 0.94 | 1.00 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 1.00 | 0.99 | 0.79 | 0.90 | 0.99 | 1.00 |
| `unit_change` | 0.005 | feature | 0.97 | 0.95 | 0.75 | 0.84 | 0.95 | 1.00 |
| `unit_change` | 0.01 | feature | 0.98 | 0.96 | 0.74 | 0.84 | 0.97 | 1.00 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 0.76 | 0.90 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.88 | 0.93 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 | 0.18 |
| `variance_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.38 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.61 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.82 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.78 |

Not scorable by a distribution distance (the column is gone; only the schema check applies): `dropped_feature`.

## Does the alert matter?

Model: gradient boosting trained on the reference split; clean held-out accuracy 0.9687. Accuracy drop = clean minus faulty on held-out batches of 5,000 rows (positive = the fault hurts). Best recall = highest calibrated recall of any distance at batch size 10,000; schema = recall of the schema check at the same batch size.

| Fault | Severity | Accuracy drop [95% CI] | Best recall | Schema |
|---|---|---|---|---|
| `dropped_feature` | 1 | +0.0068 [+0.0037, +0.0107] | n/a | 1.00 |
| `duplicate_rows` | 0.001 | -0.0000 [-0.0000, +0.0000] | 0.87 | 0.42 |
| `duplicate_rows` | 0.005 | -0.0000 [-0.0001, -0.0000] | 0.68 | 0.39 |
| `duplicate_rows` | 0.01 | +0.0000 [-0.0000, +0.0001] | 0.68 | 0.40 |
| `duplicate_rows` | 0.05 | +0.0001 [-0.0000, +0.0003] | 0.73 | 0.33 |
| `duplicate_rows` | 0.2 | +0.0003 [-0.0000, +0.0006] | 0.67 | 0.34 |
| `mean_shift` | 0.01 | -0.0001 [-0.0001, +0.0000] | 0.42 | 0.00 |
| `mean_shift` | 0.05 | +0.0000 [-0.0003, +0.0003] | 0.42 | 0.08 |
| `mean_shift` | 0.1 | +0.0008 [+0.0004, +0.0014] | 0.40 | 0.05 |
| `mean_shift` | 0.25 | +0.0021 [+0.0009, +0.0034] | 0.47 | 0.04 |
| `mean_shift` | 0.5 | +0.0097 [+0.0053, +0.0148] | 0.72 | 0.13 |
| `missing_rate_increase` | 0.001 | +0.0000 [+0.0000, +0.0000] | 0.11 | 0.09 |
| `missing_rate_increase` | 0.005 | +0.0000 [-0.0000, +0.0000] | 0.20 | 0.26 |
| `missing_rate_increase` | 0.01 | +0.0001 [+0.0001, +0.0002] | 0.19 | 0.34 |
| `missing_rate_increase` | 0.05 | +0.0003 [+0.0002, +0.0005] | 0.44 | 0.83 |
| `missing_rate_increase` | 0.2 | +0.0008 [+0.0004, +0.0012] | 0.80 | 1.00 |
| `new_category` | 0.001 | +0.0000 [+0.0000, +0.0000] | 0.55 | 1.00 |
| `new_category` | 0.005 | +0.0000 [-0.0000, +0.0000] | 0.53 | 1.00 |
| `new_category` | 0.01 | +0.0000 [-0.0000, +0.0000] | 0.64 | 1.00 |
| `new_category` | 0.05 | +0.0000 [-0.0000, +0.0000] | 1.00 | 1.00 |
| `new_category` | 0.2 | +0.0001 [+0.0001, +0.0002] | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.13 | 0.63 |
| `out_of_domain_numeric` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.10 | 0.65 |
| `out_of_domain_numeric` | 0.01 | +0.0001 [+0.0001, +0.0002] | 0.15 | 0.58 |
| `out_of_domain_numeric` | 0.05 | +0.0008 [+0.0005, +0.0012] | 0.22 | 0.66 |
| `out_of_domain_numeric` | 0.2 | +0.0033 [+0.0021, +0.0046] | 0.43 | 0.61 |
| `pinned_value` | 0.001 | +0.0000 [+0.0000, +0.0000] | 0.17 | 0.85 |
| `pinned_value` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.20 | 0.84 |
| `pinned_value` | 0.01 | +0.0002 [+0.0001, +0.0003] | 0.13 | 0.86 |
| `pinned_value` | 0.05 | +0.0012 [+0.0007, +0.0017] | 0.28 | 0.85 |
| `pinned_value` | 0.2 | +0.0031 [+0.0018, +0.0046] | 0.49 | 0.86 |
| `time_travel` | 1 | -0.0010 [-0.0017, -0.0004] | 0.17 | 0.08 |
| `time_travel` | 2 | +0.0008 [+0.0002, +0.0014] | 0.14 | 0.05 |
| `time_travel` | 3 | +0.0025 [+0.0013, +0.0038] | 0.12 | 0.03 |
| `time_travel` | 5 | +0.0053 [+0.0029, +0.0078] | 0.14 | 0.05 |
| `time_travel` | 10 | +0.0095 [+0.0057, +0.0137] | 0.12 | 0.07 |
| `tiny_noise` | 0.0001 | +0.0000 [+0.0000, +0.0000] | 0.11 | 0.03 |
| `type_change` | 0.001 | -0.0000 [-0.0000, +0.0000] | 0.10 | 1.00 |
| `type_change` | 0.005 | +0.0000 [+0.0000, +0.0000] | 0.13 | 1.00 |
| `type_change` | 0.01 | +0.0000 [-0.0000, +0.0001] | 0.09 | 1.00 |
| `type_change` | 0.05 | +0.0003 [+0.0002, +0.0004] | 0.40 | 1.00 |
| `type_change` | 0.2 | +0.0010 [+0.0006, +0.0015] | 0.79 | 1.00 |
| `unit_change` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.15 | 1.00 |
| `unit_change` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.09 | 1.00 |
| `unit_change` | 0.01 | +0.0001 [+0.0001, +0.0002] | 0.11 | 1.00 |
| `unit_change` | 0.05 | +0.0007 [+0.0004, +0.0010] | 0.17 | 1.00 |
| `unit_change` | 0.2 | +0.0037 [+0.0022, +0.0052] | 0.20 | 1.00 |
| `variance_change` | 0.02 | -0.0000 [-0.0000, +0.0000] | 0.44 | 0.18 |
| `variance_change` | 0.05 | -0.0000 [-0.0001, +0.0001] | 0.35 | 0.38 |
| `variance_change` | 0.1 | -0.0001 [-0.0002, +0.0001] | 0.36 | 0.61 |
| `variance_change` | 0.25 | +0.0001 [-0.0003, +0.0004] | 0.49 | 0.82 |
| `variance_change` | 0.5 | +0.0002 [-0.0002, +0.0006] | 0.49 | 0.78 |

Model loses more than 0.005 accuracy but no detector reaches recall 0.5: `time_travel` at 10.

A detector reaches recall 0.9 but accuracy moves by less than 0.005: `new_category` at 0.05, `new_category` at 0.2.

## What this report does not measure

- Real incidents: every fault is synthetic and follows the documented injector.
- Several faults at once, or a fault that builds up slowly over many batches.
- Calibration on unlimited fresh data: thresholds come from one finite split and are
  applied to another, so the achieved false-positive rate is above the target, and the
  gap grows with the batch size relative to the split size. The achieved rate is shown.
- The model part uses one model family and accuracy only.
