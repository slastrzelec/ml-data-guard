# Benchmark report: beijing

- Data: UCI Beijing Multi-Site Air Quality, CC BY 4.0, https://doi.org/10.24432/C5RK5G, 412,029 rows, 12 numeric and 2 categorical features.
- Splits (rows): reference 164,811, calibration 103,007, evaluation 103,008, held-out 41,203.
- Reference: rolling, the 5 batches of rows before each batch.
- Seed 20261010; 1000 calibration batches, 1000 clean batches, 100 faulty batches per fault and severity; target false-positive rate 0.01.
- Commit: f7fd371; ml-data-guard 0.1.0, numpy 2.4.6, pandas 3.0.6, python 3.11.17, scikit-learn 1.9.1, scipy 1.17.1.

Rates are followed by 95% Wilson intervals where space allows; every interval is in the
JSON file. With 100 faulty batches a recall of 0.50 has an interval of
about +/- 0.10.

## Batch size 1,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.021 [0.018, 0.023] | 0.187 [0.164, 0.212] |
| `l1` (calibrated) | 0.019 [0.017, 0.021] | 0.148 [0.127, 0.171] |
| `kl` (calibrated) | 0.018 [0.016, 0.021] | 0.140 [0.120, 0.163] |
| `js` (calibrated) | 0.018 [0.016, 0.021] | 0.138 [0.118, 0.161] |
| `psi` (calibrated) | 0.020 [0.018, 0.022] | 0.166 [0.144, 0.190] |
| `chi2` (calibrated) | 0.018 [0.016, 0.020] | 0.146 [0.125, 0.169] |
| `ks` (calibrated) | 0.014 [0.012, 0.016] | 0.102 [0.085, 0.122] |
| `chi2 p<0.05` (textbook) | 0.913 [0.908, 0.917] | 1.000 [0.996, 1.000] |
| `ks p<0.05` (textbook) | 0.940 [0.936, 0.944] | 1.000 [0.996, 1.000] |
| `psi>0.2` (textbook) | 0.727 [0.719, 0.734] | 1.000 [0.996, 1.000] |
| `psi>0.1` (textbook) | 0.840 [0.834, 0.846] | 1.000 [0.996, 1.000] |
| `linf>0.01` (textbook) | 0.943 [0.939, 0.947] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.038 [0.035, 0.041] | 0.322 [0.294, 0.352] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.15 | 0.12 | 0.12 | 0.14 | 0.13 | 0.10 | 0.09 |
| `duplicate_rows` | 0.005 | batch | 0.20 | 0.16 | 0.17 | 0.16 | 0.18 | 0.17 | 0.13 |
| `duplicate_rows` | 0.01 | batch | 0.20 | 0.15 | 0.16 | 0.11 | 0.14 | 0.17 | 0.11 |
| `duplicate_rows` | 0.05 | batch | 0.17 | 0.14 | 0.14 | 0.13 | 0.15 | 0.11 | 0.10 |
| `duplicate_rows` | 0.2 | batch | 0.16 | 0.13 | 0.13 | 0.14 | 0.17 | 0.12 | 0.11 |
| `mean_shift` | 0.01 | feature | 0.09 | 0.21 | 0.22 | 0.26 | 0.27 | 0.20 | 0.07 |
| `mean_shift` | 0.05 | feature | 0.08 | 0.21 | 0.25 | 0.27 | 0.30 | 0.21 | 0.08 |
| `mean_shift` | 0.1 | feature | 0.11 | 0.26 | 0.27 | 0.29 | 0.30 | 0.25 | 0.07 |
| `mean_shift` | 0.25 | feature | 0.07 | 0.21 | 0.25 | 0.28 | 0.29 | 0.20 | 0.08 |
| `mean_shift` | 0.5 | feature | 0.14 | 0.33 | 0.32 | 0.39 | 0.41 | 0.29 | 0.19 |
| `missing_rate_increase` | 0.001 | feature | 0.04 | 0.03 | 0.02 | 0.03 | 0.03 | 0.02 | 0.01 |
| `missing_rate_increase` | 0.005 | feature | 0.03 | 0.03 | 0.04 | 0.02 | 0.02 | 0.04 | 0.01 |
| `missing_rate_increase` | 0.01 | feature | 0.02 | 0.03 | 0.03 | 0.04 | 0.02 | 0.03 | 0.03 |
| `missing_rate_increase` | 0.05 | feature | 0.00 | 0.00 | 0.10 | 0.01 | 0.02 | 0.09 | 0.00 |
| `missing_rate_increase` | 0.2 | feature | 0.18 | 0.11 | 0.41 | 0.18 | 0.27 | 0.33 | 0.00 |
| `new_category` | 0.001 | feature | 0.07 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | n/a |
| `new_category` | 0.005 | feature | 0.05 | 0.04 | 0.03 | 0.03 | 0.02 | 0.03 | n/a |
| `new_category` | 0.01 | feature | 0.04 | 0.04 | 0.03 | 0.03 | 0.02 | 0.03 | n/a |
| `new_category` | 0.05 | feature | 0.06 | 0.11 | 0.74 | 0.11 | 0.13 | 0.62 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 0.56 | 1.00 | 0.71 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| `out_of_domain_numeric` | 0.005 | feature | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 |
| `out_of_domain_numeric` | 0.01 | feature | 0.02 | 0.03 | 0.03 | 0.03 | 0.02 | 0.01 | 0.03 |
| `out_of_domain_numeric` | 0.05 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `out_of_domain_numeric` | 0.2 | feature | 0.03 | 0.04 | 0.03 | 0.07 | 0.03 | 0.05 | 0.03 |
| `pinned_value` | 0.001 | feature | 0.03 | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.01 |
| `pinned_value` | 0.005 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.00 | 0.01 |
| `pinned_value` | 0.01 | feature | 0.05 | 0.04 | 0.03 | 0.04 | 0.04 | 0.03 | 0.04 |
| `pinned_value` | 0.05 | feature | 0.01 | 0.02 | 0.14 | 0.02 | 0.02 | 0.10 | 0.01 |
| `pinned_value` | 0.2 | feature | 0.24 | 0.15 | 0.33 | 0.21 | 0.28 | 0.29 | 0.03 |
| `time_travel` | 1 | feature | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| `time_travel` | 2 | feature | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| `time_travel` | 3 | feature | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.02 | 0.01 |
| `time_travel` | 5 | feature | 0.05 | 0.01 | 0.03 | 0.01 | 0.04 | 0.03 | 0.01 |
| `time_travel` | 10 | feature | 0.06 | 0.05 | 0.07 | 0.06 | 0.05 | 0.06 | 0.05 |
| `tiny_noise` (control) | 0.0001 | feature | 0.02 | 0.01 | 0.01 | 0.01 | 0.00 | 0.02 | 0.01 |
| `type_change` | 0.001 | feature | 0.00 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| `type_change` | 0.005 | feature | 0.03 | 0.04 | 0.03 | 0.04 | 0.04 | 0.02 | 0.03 |
| `type_change` | 0.01 | feature | 0.03 | 0.02 | 0.01 | 0.02 | 0.01 | 0.02 | 0.02 |
| `type_change` | 0.05 | feature | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.2 | feature | 0.01 | 0.05 | 0.34 | 0.12 | 0.16 | 0.23 | 0.03 |
| `unit_change` | 0.001 | feature | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 | 0.02 | 0.02 |
| `unit_change` | 0.005 | feature | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 |
| `unit_change` | 0.01 | feature | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| `unit_change` | 0.05 | feature | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `unit_change` | 0.2 | feature | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `variance_change` | 0.02 | feature | 0.10 | 0.25 | 0.29 | 0.28 | 0.32 | 0.28 | 0.07 |
| `variance_change` | 0.05 | feature | 0.09 | 0.17 | 0.24 | 0.20 | 0.28 | 0.21 | 0.07 |
| `variance_change` | 0.1 | feature | 0.09 | 0.24 | 0.30 | 0.30 | 0.35 | 0.27 | 0.08 |
| `variance_change` | 0.25 | feature | 0.12 | 0.23 | 0.29 | 0.29 | 0.32 | 0.26 | 0.09 |
| `variance_change` | 0.5 | feature | 0.08 | 0.22 | 0.27 | 0.29 | 0.28 | 0.25 | 0.05 |

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
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.01 |
| `mean_shift` | 0.1 | feature | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.07 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.06 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `missing_rate_increase` | 0.001 | feature | 0.97 | 0.98 | 0.82 | 0.96 | 0.97 | 0.12 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.95 | 0.75 | 0.85 | 0.93 | 0.24 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.97 | 0.85 | 0.90 | 0.95 | 0.21 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.96 | 0.95 | 1.00 | 1.00 | 0.81 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 0.47 | n/a | 0.32 | 0.43 | 0.68 | 1.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.45 | 0.52 | 0.66 | 1.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.43 | 0.46 | 0.63 | 1.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.98 | 0.92 | 0.79 | 0.87 | 0.97 | 0.10 |
| `out_of_domain_numeric` | 0.005 | feature | 0.98 | 0.96 | 0.74 | 0.89 | 0.97 | 0.16 |
| `out_of_domain_numeric` | 0.01 | feature | 0.97 | 0.95 | 0.78 | 0.87 | 0.96 | 0.19 |
| `out_of_domain_numeric` | 0.05 | feature | 0.99 | 0.93 | 0.72 | 0.84 | 0.99 | 0.26 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.94 | 1.00 | 1.00 | 0.31 |
| `pinned_value` | 0.001 | feature | 0.89 | 0.93 | 0.77 | 0.84 | 0.93 | 0.86 |
| `pinned_value` | 0.005 | feature | 1.00 | 0.90 | 0.70 | 0.82 | 0.91 | 0.84 |
| `pinned_value` | 0.01 | feature | 1.00 | 0.96 | 0.75 | 0.89 | 0.98 | 0.88 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.91 | 0.99 | 1.00 | 0.85 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.86 |
| `time_travel` | 1 | feature | 0.98 | 0.93 | 0.79 | 0.91 | 0.98 | 0.04 |
| `time_travel` | 2 | feature | 0.99 | 0.96 | 0.76 | 0.90 | 0.99 | 0.00 |
| `time_travel` | 3 | feature | 1.00 | 0.96 | 0.75 | 0.93 | 1.00 | 0.07 |
| `time_travel` | 5 | feature | 0.97 | 0.91 | 0.74 | 0.89 | 0.96 | 0.04 |
| `time_travel` | 10 | feature | 0.98 | 0.92 | 0.78 | 0.90 | 0.98 | 0.01 |
| `tiny_noise` (control) | 0.0001 | feature | 0.95 | 0.97 | 0.75 | 0.86 | 0.97 | 0.03 |
| `type_change` | 0.001 | feature | 0.97 | 0.94 | 0.76 | 0.90 | 0.95 | 1.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.93 | 0.76 | 0.86 | 0.98 | 1.00 |
| `type_change` | 0.01 | feature | 1.00 | 0.99 | 0.83 | 0.94 | 1.00 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.97 | 0.93 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 1.00 | 0.99 | 0.85 | 0.96 | 1.00 | 1.00 |
| `unit_change` | 0.005 | feature | 0.99 | 0.96 | 0.85 | 0.95 | 0.99 | 1.00 |
| `unit_change` | 0.01 | feature | 0.99 | 0.96 | 0.85 | 0.92 | 1.00 | 1.00 |
| `unit_change` | 0.05 | feature | 0.97 | 0.97 | 0.80 | 0.88 | 0.98 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 0.96 | 0.92 | 0.96 | 0.99 | 1.00 |
| `variance_change` | 0.02 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.10 |
| `variance_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.27 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 0.40 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 |

Not scorable by a distribution distance (the column is gone; only the schema check applies): `dropped_feature`.

## Batch size 10,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.078 [0.073, 0.082] | 0.485 [0.454, 0.516] |
| `l1` (calibrated) | 0.074 [0.070, 0.079] | 0.485 [0.454, 0.516] |
| `kl` (calibrated) | 0.071 [0.067, 0.075] | 0.459 [0.428, 0.490] |
| `js` (calibrated) | 0.076 [0.072, 0.081] | 0.529 [0.498, 0.560] |
| `psi` (calibrated) | 0.069 [0.065, 0.073] | 0.524 [0.493, 0.555] |
| `chi2` (calibrated) | 0.061 [0.057, 0.065] | 0.396 [0.366, 0.427] |
| `ks` (calibrated) | 0.085 [0.080, 0.090] | 0.480 [0.449, 0.511] |
| `chi2 p<0.05` (textbook) | 0.926 [0.921, 0.930] | 1.000 [0.996, 1.000] |
| `ks p<0.05` (textbook) | 0.986 [0.983, 0.988] | 1.000 [0.996, 1.000] |
| `psi>0.2` (textbook) | 0.654 [0.647, 0.662] | 1.000 [0.996, 1.000] |
| `psi>0.1` (textbook) | 0.787 [0.780, 0.793] | 1.000 [0.996, 1.000] |
| `linf>0.01` (textbook) | 0.915 [0.910, 0.920] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.048 [0.044, 0.051] | 0.402 [0.372, 0.433] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.55 | 0.55 | 0.47 | 0.63 | 0.61 | 0.45 | 0.54 |
| `duplicate_rows` | 0.005 | batch | 0.43 | 0.43 | 0.45 | 0.49 | 0.48 | 0.40 | 0.44 |
| `duplicate_rows` | 0.01 | batch | 0.49 | 0.44 | 0.41 | 0.53 | 0.49 | 0.37 | 0.45 |
| `duplicate_rows` | 0.05 | batch | 0.51 | 0.48 | 0.47 | 0.53 | 0.53 | 0.41 | 0.45 |
| `duplicate_rows` | 0.2 | batch | 0.49 | 0.53 | 0.48 | 0.55 | 0.51 | 0.36 | 0.52 |
| `mean_shift` | 0.01 | feature | 0.22 | 0.45 | 0.47 | 0.46 | 0.52 | 0.44 | 0.22 |
| `mean_shift` | 0.05 | feature | 0.14 | 0.39 | 0.45 | 0.47 | 0.51 | 0.40 | 0.17 |
| `mean_shift` | 0.1 | feature | 0.23 | 0.43 | 0.48 | 0.50 | 0.55 | 0.44 | 0.23 |
| `mean_shift` | 0.25 | feature | 0.19 | 0.42 | 0.55 | 0.65 | 0.70 | 0.44 | 0.20 |
| `mean_shift` | 0.5 | feature | 0.27 | 0.70 | 0.77 | 0.79 | 0.78 | 0.62 | 0.41 |
| `missing_rate_increase` | 0.001 | feature | 0.07 | 0.07 | 0.08 | 0.08 | 0.06 | 0.07 | 0.09 |
| `missing_rate_increase` | 0.005 | feature | 0.08 | 0.09 | 0.20 | 0.12 | 0.19 | 0.16 | 0.11 |
| `missing_rate_increase` | 0.01 | feature | 0.05 | 0.05 | 0.13 | 0.13 | 0.12 | 0.10 | 0.06 |
| `missing_rate_increase` | 0.05 | feature | 0.14 | 0.18 | 0.39 | 0.26 | 0.26 | 0.30 | 0.06 |
| `missing_rate_increase` | 0.2 | feature | 0.29 | 0.45 | 0.86 | 0.67 | 0.73 | 0.78 | 0.08 |
| `new_category` | 0.001 | feature | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | n/a |
| `new_category` | 0.005 | feature | 0.00 | 0.00 | 0.49 | 0.00 | 0.49 | 0.49 | n/a |
| `new_category` | 0.01 | feature | 0.01 | 0.02 | 0.56 | 0.55 | 0.56 | 0.56 | n/a |
| `new_category` | 0.05 | feature | 0.49 | 0.49 | 1.00 | 0.50 | 0.88 | 0.70 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.10 | 0.08 | 0.07 | 0.08 | 0.04 | 0.09 | 0.08 |
| `out_of_domain_numeric` | 0.005 | feature | 0.08 | 0.07 | 0.07 | 0.09 | 0.05 | 0.06 | 0.09 |
| `out_of_domain_numeric` | 0.01 | feature | 0.06 | 0.05 | 0.06 | 0.04 | 0.02 | 0.06 | 0.05 |
| `out_of_domain_numeric` | 0.05 | feature | 0.11 | 0.10 | 0.12 | 0.08 | 0.08 | 0.12 | 0.10 |
| `out_of_domain_numeric` | 0.2 | feature | 0.33 | 0.30 | 0.33 | 0.28 | 0.22 | 0.34 | 0.30 |
| `pinned_value` | 0.001 | feature | 0.10 | 0.07 | 0.05 | 0.05 | 0.04 | 0.05 | 0.10 |
| `pinned_value` | 0.005 | feature | 0.08 | 0.06 | 0.14 | 0.07 | 0.15 | 0.12 | 0.06 |
| `pinned_value` | 0.01 | feature | 0.12 | 0.12 | 0.22 | 0.16 | 0.16 | 0.18 | 0.12 |
| `pinned_value` | 0.05 | feature | 0.18 | 0.17 | 0.27 | 0.19 | 0.21 | 0.20 | 0.06 |
| `pinned_value` | 0.2 | feature | 0.30 | 0.34 | 0.50 | 0.42 | 0.44 | 0.42 | 0.12 |
| `time_travel` | 1 | feature | 0.08 | 0.06 | 0.06 | 0.08 | 0.03 | 0.06 | 0.09 |
| `time_travel` | 2 | feature | 0.08 | 0.10 | 0.09 | 0.07 | 0.08 | 0.07 | 0.10 |
| `time_travel` | 3 | feature | 0.03 | 0.05 | 0.08 | 0.11 | 0.11 | 0.01 | 0.05 |
| `time_travel` | 5 | feature | 0.12 | 0.09 | 0.09 | 0.09 | 0.07 | 0.07 | 0.10 |
| `time_travel` | 10 | feature | 0.06 | 0.07 | 0.07 | 0.08 | 0.09 | 0.05 | 0.08 |
| `tiny_noise` (control) | 0.0001 | feature | 0.09 | 0.06 | 0.08 | 0.06 | 0.05 | 0.09 | 0.07 |
| `type_change` | 0.001 | feature | 0.10 | 0.11 | 0.11 | 0.13 | 0.08 | 0.08 | 0.11 |
| `type_change` | 0.005 | feature | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.05 | 0.04 |
| `type_change` | 0.01 | feature | 0.10 | 0.09 | 0.10 | 0.11 | 0.12 | 0.09 | 0.09 |
| `type_change` | 0.05 | feature | 0.09 | 0.18 | 0.34 | 0.23 | 0.25 | 0.23 | 0.12 |
| `type_change` | 0.2 | feature | 0.13 | 0.31 | 0.81 | 0.50 | 0.59 | 0.62 | 0.07 |
| `unit_change` | 0.001 | feature | 0.12 | 0.10 | 0.11 | 0.08 | 0.07 | 0.11 | 0.10 |
| `unit_change` | 0.005 | feature | 0.08 | 0.05 | 0.04 | 0.06 | 0.03 | 0.03 | 0.05 |
| `unit_change` | 0.01 | feature | 0.09 | 0.05 | 0.05 | 0.04 | 0.05 | 0.05 | 0.08 |
| `unit_change` | 0.05 | feature | 0.09 | 0.08 | 0.10 | 0.10 | 0.09 | 0.09 | 0.08 |
| `unit_change` | 0.2 | feature | 0.14 | 0.13 | 0.11 | 0.07 | 0.04 | 0.12 | 0.11 |
| `variance_change` | 0.02 | feature | 0.19 | 0.50 | 0.53 | 0.55 | 0.57 | 0.50 | 0.21 |
| `variance_change` | 0.05 | feature | 0.22 | 0.38 | 0.44 | 0.44 | 0.47 | 0.44 | 0.17 |
| `variance_change` | 0.1 | feature | 0.24 | 0.41 | 0.50 | 0.51 | 0.57 | 0.48 | 0.18 |
| `variance_change` | 0.25 | feature | 0.28 | 0.47 | 0.52 | 0.53 | 0.53 | 0.51 | 0.18 |
| `variance_change` | 0.5 | feature | 0.33 | 0.49 | 0.55 | 0.57 | 0.58 | 0.53 | 0.25 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `dropped_feature` | 1 | feature | n/a | n/a | n/a | n/a | n/a | 1.00 |
| `duplicate_rows` | 0.001 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.42 |
| `duplicate_rows` | 0.005 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.39 |
| `duplicate_rows` | 0.01 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 |
| `duplicate_rows` | 0.05 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| `duplicate_rows` | 0.2 | batch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.34 |
| `mean_shift` | 0.01 | feature | 1.00 | 1.00 | 0.96 | 0.98 | 1.00 | 0.00 |
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.08 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.05 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.13 |
| `missing_rate_increase` | 0.001 | feature | 1.00 | 1.00 | 0.66 | 0.82 | 0.96 | 0.09 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.96 | 0.70 | 0.77 | 0.89 | 0.26 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 1.00 | 0.67 | 0.83 | 0.94 | 0.34 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.98 | 0.92 | 1.00 | 1.00 | 0.83 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.20 | 0.36 | 0.46 | 1.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.13 | 0.40 | 0.51 | 1.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.25 | 0.46 | 0.47 | 1.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 1.00 | 1.00 | 0.84 | 0.94 | 0.99 | 0.63 |
| `out_of_domain_numeric` | 0.005 | feature | 1.00 | 0.99 | 0.69 | 0.83 | 0.99 | 0.65 |
| `out_of_domain_numeric` | 0.01 | feature | 1.00 | 0.98 | 0.69 | 0.84 | 0.99 | 0.58 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 0.61 | 0.79 | 1.00 | 0.66 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.89 | 1.00 | 1.00 | 0.61 |
| `pinned_value` | 0.001 | feature | 1.00 | 1.00 | 0.60 | 0.77 | 0.91 | 0.85 |
| `pinned_value` | 0.005 | feature | 1.00 | 0.98 | 0.60 | 0.78 | 0.92 | 0.84 |
| `pinned_value` | 0.01 | feature | 1.00 | 1.00 | 0.69 | 0.88 | 0.95 | 0.86 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.91 | 0.95 | 1.00 | 0.85 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.86 |
| `time_travel` | 1 | feature | 0.99 | 0.97 | 0.74 | 0.89 | 0.97 | 0.08 |
| `time_travel` | 2 | feature | 1.00 | 1.00 | 0.71 | 0.87 | 1.00 | 0.05 |
| `time_travel` | 3 | feature | 1.00 | 0.99 | 0.71 | 0.84 | 0.98 | 0.03 |
| `time_travel` | 5 | feature | 1.00 | 0.97 | 0.66 | 0.78 | 0.98 | 0.05 |
| `time_travel` | 10 | feature | 1.00 | 1.00 | 0.74 | 0.87 | 1.00 | 0.07 |
| `tiny_noise` (control) | 0.0001 | feature | 0.93 | 0.97 | 0.64 | 0.75 | 0.93 | 0.03 |
| `type_change` | 0.001 | feature | 1.00 | 0.96 | 0.70 | 0.82 | 0.97 | 1.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.96 | 0.70 | 0.83 | 0.97 | 1.00 |
| `type_change` | 0.01 | feature | 1.00 | 1.00 | 0.77 | 0.93 | 1.00 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 1.00 | 1.00 | 0.76 | 0.87 | 1.00 | 1.00 |
| `unit_change` | 0.005 | feature | 0.99 | 0.97 | 0.69 | 0.81 | 0.96 | 1.00 |
| `unit_change` | 0.01 | feature | 0.99 | 0.97 | 0.65 | 0.78 | 0.98 | 1.00 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 0.70 | 0.80 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.86 | 0.94 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.18 |
| `variance_change` | 0.05 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.38 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 0.97 | 0.99 | 1.00 | 0.61 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.82 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | 0.78 |

Not scorable by a distribution distance (the column is gone; only the schema check applies): `dropped_feature`.

## Does the alert matter?

Model: gradient boosting trained on the reference split; clean held-out accuracy 0.9687. Accuracy drop = clean minus faulty on held-out batches of 5,000 rows (positive = the fault hurts). Best recall = highest calibrated recall of any distance at batch size 10,000; schema = recall of the schema check at the same batch size.

| Fault | Severity | Accuracy drop [95% CI] | Best recall | Schema |
|---|---|---|---|---|
| `dropped_feature` | 1 | +0.0068 [+0.0037, +0.0107] | n/a | 1.00 |
| `duplicate_rows` | 0.001 | -0.0000 [-0.0000, +0.0000] | 0.63 | 0.42 |
| `duplicate_rows` | 0.005 | -0.0000 [-0.0001, -0.0000] | 0.49 | 0.39 |
| `duplicate_rows` | 0.01 | +0.0000 [-0.0000, +0.0001] | 0.53 | 0.40 |
| `duplicate_rows` | 0.05 | +0.0001 [-0.0000, +0.0003] | 0.53 | 0.33 |
| `duplicate_rows` | 0.2 | +0.0003 [-0.0000, +0.0006] | 0.55 | 0.34 |
| `mean_shift` | 0.01 | -0.0001 [-0.0001, +0.0000] | 0.52 | 0.00 |
| `mean_shift` | 0.05 | +0.0000 [-0.0003, +0.0003] | 0.51 | 0.08 |
| `mean_shift` | 0.1 | +0.0008 [+0.0004, +0.0014] | 0.55 | 0.05 |
| `mean_shift` | 0.25 | +0.0021 [+0.0009, +0.0034] | 0.70 | 0.04 |
| `mean_shift` | 0.5 | +0.0097 [+0.0053, +0.0148] | 0.79 | 0.13 |
| `missing_rate_increase` | 0.001 | +0.0000 [+0.0000, +0.0000] | 0.09 | 0.09 |
| `missing_rate_increase` | 0.005 | +0.0000 [-0.0000, +0.0000] | 0.20 | 0.26 |
| `missing_rate_increase` | 0.01 | +0.0001 [+0.0001, +0.0002] | 0.13 | 0.34 |
| `missing_rate_increase` | 0.05 | +0.0003 [+0.0002, +0.0005] | 0.39 | 0.83 |
| `missing_rate_increase` | 0.2 | +0.0008 [+0.0004, +0.0012] | 0.86 | 1.00 |
| `new_category` | 0.001 | +0.0000 [+0.0000, +0.0000] | 0.00 | 1.00 |
| `new_category` | 0.005 | +0.0000 [-0.0000, +0.0000] | 0.49 | 1.00 |
| `new_category` | 0.01 | +0.0000 [-0.0000, +0.0000] | 0.56 | 1.00 |
| `new_category` | 0.05 | +0.0000 [-0.0000, +0.0000] | 1.00 | 1.00 |
| `new_category` | 0.2 | +0.0001 [+0.0001, +0.0002] | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.10 | 0.63 |
| `out_of_domain_numeric` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.09 | 0.65 |
| `out_of_domain_numeric` | 0.01 | +0.0001 [+0.0001, +0.0002] | 0.06 | 0.58 |
| `out_of_domain_numeric` | 0.05 | +0.0008 [+0.0005, +0.0012] | 0.12 | 0.66 |
| `out_of_domain_numeric` | 0.2 | +0.0033 [+0.0021, +0.0046] | 0.34 | 0.61 |
| `pinned_value` | 0.001 | +0.0000 [+0.0000, +0.0000] | 0.10 | 0.85 |
| `pinned_value` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.15 | 0.84 |
| `pinned_value` | 0.01 | +0.0002 [+0.0001, +0.0003] | 0.22 | 0.86 |
| `pinned_value` | 0.05 | +0.0012 [+0.0007, +0.0017] | 0.27 | 0.85 |
| `pinned_value` | 0.2 | +0.0031 [+0.0018, +0.0046] | 0.50 | 0.86 |
| `time_travel` | 1 | -0.0010 [-0.0017, -0.0004] | 0.09 | 0.08 |
| `time_travel` | 2 | +0.0008 [+0.0002, +0.0014] | 0.10 | 0.05 |
| `time_travel` | 3 | +0.0025 [+0.0013, +0.0038] | 0.11 | 0.03 |
| `time_travel` | 5 | +0.0053 [+0.0029, +0.0078] | 0.12 | 0.05 |
| `time_travel` | 10 | +0.0095 [+0.0057, +0.0137] | 0.09 | 0.07 |
| `tiny_noise` | 0.0001 | +0.0000 [+0.0000, +0.0000] | 0.09 | 0.03 |
| `type_change` | 0.001 | -0.0000 [-0.0000, +0.0000] | 0.13 | 1.00 |
| `type_change` | 0.005 | +0.0000 [+0.0000, +0.0000] | 0.06 | 1.00 |
| `type_change` | 0.01 | +0.0000 [-0.0000, +0.0001] | 0.12 | 1.00 |
| `type_change` | 0.05 | +0.0003 [+0.0002, +0.0004] | 0.34 | 1.00 |
| `type_change` | 0.2 | +0.0010 [+0.0006, +0.0015] | 0.81 | 1.00 |
| `unit_change` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.12 | 1.00 |
| `unit_change` | 0.005 | +0.0001 [+0.0000, +0.0001] | 0.08 | 1.00 |
| `unit_change` | 0.01 | +0.0001 [+0.0001, +0.0002] | 0.09 | 1.00 |
| `unit_change` | 0.05 | +0.0007 [+0.0004, +0.0010] | 0.10 | 1.00 |
| `unit_change` | 0.2 | +0.0037 [+0.0022, +0.0052] | 0.14 | 1.00 |
| `variance_change` | 0.02 | -0.0000 [-0.0000, +0.0000] | 0.57 | 0.18 |
| `variance_change` | 0.05 | -0.0000 [-0.0001, +0.0001] | 0.47 | 0.38 |
| `variance_change` | 0.1 | -0.0001 [-0.0002, +0.0001] | 0.57 | 0.61 |
| `variance_change` | 0.25 | +0.0001 [-0.0003, +0.0004] | 0.53 | 0.82 |
| `variance_change` | 0.5 | +0.0002 [-0.0002, +0.0006] | 0.58 | 0.78 |

Model loses more than 0.005 accuracy but no detector reaches recall 0.5: `time_travel` at 10.

A detector reaches recall 0.9 but accuracy moves by less than 0.005: `new_category` at 0.05, `new_category` at 0.2.

## What this report does not measure

- Real incidents: every fault is synthetic and follows the documented injector.
- Several faults at once, or a fault that builds up slowly over many batches.
- Calibration on unlimited fresh data: thresholds come from one finite split and are
  applied to another, so the achieved false-positive rate is above the target, and the
  gap grows with the batch size relative to the split size. The achieved rate is shown.
- The model part uses one model family and accuracy only.
