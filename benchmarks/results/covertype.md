# Benchmark report: covertype

- Data: UCI Covertype, CC BY 4.0, https://doi.org/10.24432/C50K5N, 581,012 rows, 10 numeric and 2 categorical features.
- Splits (rows): reference 232,404, calibration 145,253, evaluation 145,253, held-out 58,102.
- Seed 20261010; 1000 calibration batches, 1000 clean batches, 100 faulty batches per fault and severity; target false-positive rate 0.01.
- Commit: 0e99fae; ml-data-guard 0.1.0, numpy 2.4.6, pandas 3.0.6, python 3.11.17, scikit-learn 1.9.1, scipy 1.17.1.

Rates are followed by 95% Wilson intervals where space allows; every interval is in the
JSON file. With 100 faulty batches a recall of 0.50 has an interval of
about +/- 0.10.

## Batch size 1,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.009 [0.007, 0.011] | 0.105 [0.087, 0.126] |
| `l1` (calibrated) | 0.007 [0.006, 0.009] | 0.081 [0.066, 0.100] |
| `kl` (calibrated) | 0.011 [0.009, 0.013] | 0.122 [0.103, 0.144] |
| `js` (calibrated) | 0.010 [0.009, 0.012] | 0.112 [0.094, 0.133] |
| `psi` (calibrated) | 0.011 [0.009, 0.013] | 0.118 [0.099, 0.139] |
| `chi2` (calibrated) | 0.013 [0.011, 0.015] | 0.135 [0.115, 0.158] |
| `ks` (calibrated) | 0.009 [0.008, 0.011] | 0.088 [0.072, 0.107] |
| `chi2 p<0.05` (textbook) | 0.055 [0.051, 0.059] | 0.472 [0.441, 0.503] |
| `ks p<0.05` (textbook) | 0.035 [0.032, 0.039] | 0.272 [0.245, 0.300] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.936 [0.931, 0.940] | 1.000 [0.996, 1.000] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.14 | 0.16 | 0.16 | 0.14 | 0.18 | 0.11 | 0.11 |
| `duplicate_rows` | 0.005 | batch | 0.13 | 0.08 | 0.16 | 0.16 | 0.16 | 0.16 | 0.12 |
| `duplicate_rows` | 0.01 | batch | 0.16 | 0.12 | 0.17 | 0.17 | 0.16 | 0.18 | 0.08 |
| `duplicate_rows` | 0.05 | batch | 0.21 | 0.17 | 0.23 | 0.21 | 0.23 | 0.22 | 0.12 |
| `duplicate_rows` | 0.2 | batch | 0.37 | 0.60 | 0.73 | 0.71 | 0.70 | 0.74 | 0.36 |
| `mean_shift` | 0.01 | feature | 0.17 | 0.18 | 0.20 | 0.20 | 0.19 | 0.18 | 0.09 |
| `mean_shift` | 0.05 | feature | 0.26 | 0.25 | 0.28 | 0.27 | 0.26 | 0.26 | 0.31 |
| `mean_shift` | 0.1 | feature | 0.73 | 0.62 | 0.70 | 0.71 | 0.71 | 0.60 | 0.83 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.00 | 0.03 | 0.05 | 0.04 | 0.03 | 1.00 | 0.04 |
| `missing_rate_increase` | 0.005 | feature | 0.00 | 0.00 | 0.99 | 0.39 | 0.67 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.01 | feature | 0.02 | 0.01 | 1.00 | 0.98 | 1.00 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.69 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `new_category` | 0.001 | feature | 0.00 | 0.02 | 0.05 | 0.05 | 0.05 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 0.00 | 0.00 | 0.99 | 0.62 | 0.68 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 0.00 | 0.01 | 1.00 | 0.86 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 1.00 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.03 |
| `out_of_domain_numeric` | 0.005 | feature | 0.03 | 0.01 | 0.03 | 0.02 | 0.01 | 0.02 | 0.00 |
| `out_of_domain_numeric` | 0.01 | feature | 0.02 | 0.04 | 0.02 | 0.03 | 0.03 | 0.02 | 0.03 |
| `out_of_domain_numeric` | 0.05 | feature | 0.95 | 0.43 | 0.60 | 0.54 | 0.50 | 0.56 | 0.93 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.00 | 0.00 | 0.03 | 0.02 | 0.01 | 0.24 | 0.01 |
| `pinned_value` | 0.005 | feature | 0.01 | 0.04 | 0.21 | 0.13 | 0.16 | 0.21 | 0.04 |
| `pinned_value` | 0.01 | feature | 0.04 | 0.03 | 0.20 | 0.18 | 0.21 | 0.20 | 0.01 |
| `pinned_value` | 0.05 | feature | 0.98 | 0.50 | 0.77 | 0.71 | 0.67 | 0.70 | 0.84 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.001 | feature | 0.00 | 0.00 | 0.06 | 0.05 | 0.05 | 1.00 | 0.01 |
| `type_change` | 0.005 | feature | 0.00 | 0.02 | 1.00 | 0.33 | 0.63 | 1.00 | 0.01 |
| `type_change` | 0.01 | feature | 0.00 | 0.04 | 1.00 | 0.99 | 1.00 | 1.00 | 0.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.74 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.03 |
| `unit_change` | 0.001 | feature | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 | 0.01 | 0.03 |
| `unit_change` | 0.005 | feature | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| `unit_change` | 0.01 | feature | 0.04 | 0.02 | 0.02 | 0.03 | 0.02 | 0.02 | 0.01 |
| `unit_change` | 0.05 | feature | 0.95 | 0.52 | 0.72 | 0.67 | 0.63 | 0.62 | 0.94 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.17 | 0.21 | 0.21 | 0.21 | 0.21 | 0.22 | 0.19 |
| `variance_change` | 0.05 | feature | 0.32 | 0.27 | 0.30 | 0.31 | 0.31 | 0.29 | 0.28 |
| `variance_change` | 0.1 | feature | 0.81 | 0.74 | 0.77 | 0.78 | 0.76 | 0.69 | 0.54 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` |
|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.55 | 0.24 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.005 | batch | 0.56 | 0.32 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.01 | batch | 0.53 | 0.25 | 0.00 | 0.00 | 1.00 |
| `duplicate_rows` | 0.05 | batch | 0.67 | 0.41 | 0.00 | 0.01 | 1.00 |
| `duplicate_rows` | 0.2 | batch | 1.00 | 0.64 | 0.00 | 0.00 | 1.00 |
| `mean_shift` | 0.01 | feature | 0.22 | 0.17 | 0.07 | 0.07 | 0.98 |
| `mean_shift` | 0.05 | feature | 0.34 | 0.39 | 0.16 | 0.20 | 0.98 |
| `mean_shift` | 0.1 | feature | 0.75 | 0.90 | 0.20 | 0.20 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.40 | 0.86 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 1.00 | 0.06 | 0.00 | 0.00 | 0.93 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.91 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.05 | 0.00 | 0.00 | 0.98 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.88 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.86 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.06 | 0.90 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.04 | 0.05 | 0.00 | 0.00 | 0.96 |
| `out_of_domain_numeric` | 0.005 | feature | 0.09 | 0.05 | 0.00 | 0.00 | 0.94 |
| `out_of_domain_numeric` | 0.01 | feature | 0.08 | 0.03 | 0.00 | 0.00 | 0.99 |
| `out_of_domain_numeric` | 0.05 | feature | 0.81 | 1.00 | 0.00 | 0.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.27 | 0.01 | 0.00 | 0.00 | 0.95 |
| `pinned_value` | 0.005 | feature | 0.23 | 0.06 | 0.00 | 0.00 | 0.96 |
| `pinned_value` | 0.01 | feature | 0.26 | 0.05 | 0.00 | 0.01 | 0.98 |
| `pinned_value` | 0.05 | feature | 0.89 | 1.00 | 0.18 | 0.18 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.06 | 0.01 | 0.00 | 0.00 | 0.96 |
| `type_change` | 0.001 | feature | 1.00 | 0.02 | 0.00 | 0.00 | 0.93 |
| `type_change` | 0.005 | feature | 1.00 | 0.02 | 0.00 | 0.00 | 0.98 |
| `type_change` | 0.01 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.99 |
| `type_change` | 0.05 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.04 | 0.07 | 0.00 | 0.00 | 0.95 |
| `unit_change` | 0.005 | feature | 0.03 | 0.01 | 0.00 | 0.00 | 0.96 |
| `unit_change` | 0.01 | feature | 0.08 | 0.07 | 0.00 | 0.00 | 0.93 |
| `unit_change` | 0.05 | feature | 0.89 | 0.99 | 0.00 | 0.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.23 | 0.21 | 0.15 | 0.17 | 0.97 |
| `variance_change` | 0.05 | feature | 0.38 | 0.35 | 0.10 | 0.24 | 0.98 |
| `variance_change` | 0.1 | feature | 0.83 | 0.72 | 0.15 | 0.21 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 0.26 | 0.76 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |

Not scorable by a distribution distance (the column is gone; this is a schema check): `dropped_feature`.

## Batch size 10,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.011 [0.010, 0.014] | 0.127 [0.108, 0.149] |
| `l1` (calibrated) | 0.011 [0.010, 0.013] | 0.129 [0.110, 0.151] |
| `kl` (calibrated) | 0.011 [0.010, 0.013] | 0.129 [0.110, 0.151] |
| `js` (calibrated) | 0.011 [0.010, 0.013] | 0.133 [0.113, 0.155] |
| `psi` (calibrated) | 0.011 [0.010, 0.013] | 0.131 [0.111, 0.153] |
| `chi2` (calibrated) | 0.013 [0.011, 0.015] | 0.144 [0.124, 0.167] |
| `ks` (calibrated) | 0.009 [0.007, 0.011] | 0.073 [0.058, 0.091] |
| `chi2 p<0.05` (textbook) | 0.049 [0.045, 0.053] | 0.443 [0.412, 0.474] |
| `ks p<0.05` (textbook) | 0.037 [0.033, 0.041] | 0.267 [0.241, 0.295] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.014 [0.012, 0.016] | 0.154 [0.133, 0.178] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.11 | 0.19 | 0.16 | 0.13 | 0.14 | 0.15 | 0.09 |
| `duplicate_rows` | 0.005 | batch | 0.14 | 0.15 | 0.08 | 0.13 | 0.12 | 0.11 | 0.07 |
| `duplicate_rows` | 0.01 | batch | 0.20 | 0.18 | 0.10 | 0.13 | 0.12 | 0.13 | 0.11 |
| `duplicate_rows` | 0.05 | batch | 0.17 | 0.18 | 0.20 | 0.24 | 0.23 | 0.22 | 0.11 |
| `duplicate_rows` | 0.2 | batch | 0.51 | 0.62 | 0.68 | 0.68 | 0.68 | 0.69 | 0.26 |
| `mean_shift` | 0.01 | feature | 0.18 | 0.22 | 0.51 | 0.50 | 0.51 | 0.43 | 0.21 |
| `mean_shift` | 0.05 | feature | 0.96 | 0.96 | 0.98 | 0.98 | 0.98 | 0.97 | 1.00 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.01 | 0.02 | 0.99 | 0.97 | 0.88 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.005 | feature | 0.03 | 0.08 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.01 | feature | 0.58 | 0.18 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `new_category` | 0.001 | feature | 0.01 | 0.04 | 0.86 | 0.69 | 0.67 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 0.01 | 0.07 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 0.05 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.01 | 0.03 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| `out_of_domain_numeric` | 0.005 | feature | 0.09 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| `out_of_domain_numeric` | 0.01 | feature | 0.48 | 0.11 | 0.16 | 0.11 | 0.14 | 0.16 | 0.11 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.00 | 0.01 | 0.23 | 0.17 | 0.15 | 0.22 | 0.00 |
| `pinned_value` | 0.005 | feature | 0.06 | 0.05 | 0.22 | 0.22 | 0.22 | 0.22 | 0.02 |
| `pinned_value` | 0.01 | feature | 0.41 | 0.19 | 0.34 | 0.32 | 0.34 | 0.34 | 0.14 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |
| `type_change` | 0.001 | feature | 0.01 | 0.02 | 1.00 | 0.99 | 0.91 | 1.00 | 0.00 |
| `type_change` | 0.005 | feature | 0.01 | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `type_change` | 0.01 | feature | 0.71 | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `type_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `unit_change` | 0.001 | feature | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.03 | 0.01 |
| `unit_change` | 0.005 | feature | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | 0.04 | 0.01 |
| `unit_change` | 0.01 | feature | 0.40 | 0.13 | 0.24 | 0.22 | 0.24 | 0.22 | 0.09 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.58 | 0.50 | 0.63 | 0.63 | 0.63 | 0.63 | 0.28 |
| `variance_change` | 0.05 | feature | 1.00 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.86 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` |
|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.49 | 0.28 | 0.00 | 0.00 | 0.15 |
| `duplicate_rows` | 0.005 | batch | 0.45 | 0.35 | 0.00 | 0.00 | 0.17 |
| `duplicate_rows` | 0.01 | batch | 0.44 | 0.37 | 0.00 | 0.00 | 0.24 |
| `duplicate_rows` | 0.05 | batch | 0.65 | 0.36 | 0.00 | 0.00 | 0.20 |
| `duplicate_rows` | 0.2 | batch | 0.98 | 0.61 | 0.00 | 0.00 | 0.50 |
| `mean_shift` | 0.01 | feature | 0.53 | 0.32 | 0.07 | 0.07 | 0.17 |
| `mean_shift` | 0.05 | feature | 1.00 | 1.00 | 0.19 | 0.20 | 0.95 |
| `mean_shift` | 0.1 | feature | 1.00 | 1.00 | 0.19 | 0.20 | 1.00 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.28 | 0.63 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 1.00 | 0.02 | 0.00 | 0.00 | 0.02 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.04 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.01 | 0.00 | 0.00 | 0.03 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.02 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.02 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.09 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.12 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.04 | 0.02 | 0.00 | 0.00 | 0.01 |
| `out_of_domain_numeric` | 0.005 | feature | 0.09 | 0.11 | 0.00 | 0.00 | 0.05 |
| `out_of_domain_numeric` | 0.01 | feature | 0.36 | 0.38 | 0.00 | 0.00 | 0.44 |
| `out_of_domain_numeric` | 0.05 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.24 | 0.01 | 0.00 | 0.00 | 0.03 |
| `pinned_value` | 0.005 | feature | 0.30 | 0.11 | 0.00 | 0.00 | 0.06 |
| `pinned_value` | 0.01 | feature | 0.55 | 0.42 | 0.00 | 0.00 | 0.40 |
| `pinned_value` | 0.05 | feature | 1.00 | 1.00 | 0.18 | 0.18 | 1.00 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.04 | 0.02 | 0.00 | 0.00 | 0.01 |
| `type_change` | 0.001 | feature | 1.00 | 0.04 | 0.00 | 0.00 | 0.01 |
| `type_change` | 0.005 | feature | 1.00 | 0.04 | 0.00 | 0.00 | 0.01 |
| `type_change` | 0.01 | feature | 1.00 | 0.06 | 0.00 | 0.00 | 0.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.08 | 0.02 | 0.00 | 0.00 | 0.01 |
| `unit_change` | 0.005 | feature | 0.15 | 0.05 | 0.00 | 0.00 | 0.03 |
| `unit_change` | 0.01 | feature | 0.38 | 0.32 | 0.00 | 0.00 | 0.37 |
| `unit_change` | 0.05 | feature | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.75 | 0.49 | 0.15 | 0.16 | 0.54 |
| `variance_change` | 0.05 | feature | 0.99 | 0.98 | 0.09 | 0.24 | 1.00 |
| `variance_change` | 0.1 | feature | 1.00 | 1.00 | 0.11 | 0.20 | 1.00 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 0.26 | 0.54 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 |

Not scorable by a distribution distance (the column is gone; this is a schema check): `dropped_feature`.

## Does the alert matter?

Model: gradient boosting trained on the reference split; clean held-out accuracy 0.8324. Accuracy drop = clean minus faulty on held-out batches of 5,000 rows (positive = the fault hurts). Best recall = highest calibrated recall of any detector at batch size 10,000.

| Fault | Severity | Accuracy drop [95% CI] | Best recall |
|---|---|---|---|
| `dropped_feature` | 1 | +0.0555 [+0.0422, +0.0703] | n/a |
| `duplicate_rows` | 0.001 | -0.0000 [-0.0000, +0.0000] | 0.19 |
| `duplicate_rows` | 0.005 | -0.0001 [-0.0001, +0.0001] | 0.15 |
| `duplicate_rows` | 0.01 | +0.0000 [-0.0001, +0.0002] | 0.20 |
| `duplicate_rows` | 0.05 | +0.0002 [-0.0001, +0.0005] | 0.24 |
| `duplicate_rows` | 0.2 | -0.0001 [-0.0007, +0.0004] | 0.69 |
| `mean_shift` | 0.01 | +0.0001 [+0.0000, +0.0003] | 0.51 |
| `mean_shift` | 0.05 | +0.0016 [+0.0011, +0.0020] | 1.00 |
| `mean_shift` | 0.1 | +0.0048 [+0.0036, +0.0060] | 1.00 |
| `mean_shift` | 0.25 | +0.0220 [+0.0172, +0.0266] | 1.00 |
| `mean_shift` | 0.5 | +0.0367 [+0.0276, +0.0463] | 1.00 |
| `missing_rate_increase` | 0.001 | +0.0000 [+0.0000, +0.0001] | 1.00 |
| `missing_rate_increase` | 0.005 | +0.0002 [+0.0001, +0.0003] | 1.00 |
| `missing_rate_increase` | 0.01 | +0.0006 [+0.0004, +0.0008] | 1.00 |
| `missing_rate_increase` | 0.05 | +0.0018 [+0.0013, +0.0024] | 1.00 |
| `missing_rate_increase` | 0.2 | +0.0098 [+0.0073, +0.0127] | 1.00 |
| `new_category` | 0.001 | +0.0000 [+0.0000, +0.0001] | 1.00 |
| `new_category` | 0.005 | +0.0003 [+0.0002, +0.0003] | 1.00 |
| `new_category` | 0.01 | +0.0005 [+0.0004, +0.0006] | 1.00 |
| `new_category` | 0.05 | +0.0023 [+0.0021, +0.0025] | 1.00 |
| `new_category` | 0.2 | +0.0102 [+0.0096, +0.0108] | 1.00 |
| `out_of_domain_numeric` | 0.001 | +0.0002 [+0.0002, +0.0003] | 0.03 |
| `out_of_domain_numeric` | 0.005 | +0.0009 [+0.0008, +0.0011] | 0.09 |
| `out_of_domain_numeric` | 0.01 | +0.0022 [+0.0018, +0.0025] | 0.48 |
| `out_of_domain_numeric` | 0.05 | +0.0100 [+0.0083, +0.0117] | 1.00 |
| `out_of_domain_numeric` | 0.2 | +0.0454 [+0.0381, +0.0518] | 1.00 |
| `pinned_value` | 0.001 | +0.0001 [+0.0001, +0.0002] | 0.23 |
| `pinned_value` | 0.005 | +0.0006 [+0.0005, +0.0008] | 0.22 |
| `pinned_value` | 0.01 | +0.0011 [+0.0009, +0.0014] | 0.41 |
| `pinned_value` | 0.05 | +0.0070 [+0.0057, +0.0084] | 1.00 |
| `pinned_value` | 0.2 | +0.0246 [+0.0198, +0.0301] | 1.00 |
| `tiny_noise` | 0.0001 | +0.0000 [+0.0000, +0.0000] | 0.01 |
| `type_change` | 0.001 | +0.0000 [+0.0000, +0.0001] | 1.00 |
| `type_change` | 0.005 | +0.0003 [+0.0002, +0.0004] | 1.00 |
| `type_change` | 0.01 | +0.0004 [+0.0003, +0.0006] | 1.00 |
| `type_change` | 0.05 | +0.0037 [+0.0028, +0.0046] | 1.00 |
| `type_change` | 0.2 | +0.0098 [+0.0069, +0.0131] | 1.00 |
| `unit_change` | 0.001 | +0.0002 [+0.0002, +0.0003] | 0.03 |
| `unit_change` | 0.005 | +0.0011 [+0.0009, +0.0013] | 0.05 |
| `unit_change` | 0.01 | +0.0021 [+0.0018, +0.0024] | 0.40 |
| `unit_change` | 0.05 | +0.0109 [+0.0095, +0.0123] | 1.00 |
| `unit_change` | 0.2 | +0.0412 [+0.0350, +0.0473] | 1.00 |
| `variance_change` | 0.02 | +0.0001 [+0.0000, +0.0003] | 0.63 |
| `variance_change` | 0.05 | +0.0012 [+0.0008, +0.0015] | 1.00 |
| `variance_change` | 0.1 | +0.0040 [+0.0029, +0.0051] | 1.00 |
| `variance_change` | 0.25 | +0.0137 [+0.0107, +0.0170] | 1.00 |
| `variance_change` | 0.5 | +0.0257 [+0.0207, +0.0313] | 1.00 |

Model loses more than 0.005 accuracy but no detector reaches recall 0.5: none.

A detector reaches recall 0.9 but accuracy moves by less than 0.005: `mean_shift` at 0.05, `mean_shift` at 0.1, `missing_rate_increase` at 0.001, `missing_rate_increase` at 0.005, `missing_rate_increase` at 0.01, `missing_rate_increase` at 0.05, `new_category` at 0.001, `new_category` at 0.005, `new_category` at 0.01, `new_category` at 0.05, `type_change` at 0.001, `type_change` at 0.005, `type_change` at 0.01, `type_change` at 0.05, `variance_change` at 0.05, `variance_change` at 0.1.

## What this report does not measure

- Real incidents: every fault is synthetic and follows the documented injector.
- Several faults at once, or a fault that builds up slowly over many batches.
- Calibration on unlimited fresh data: thresholds come from one finite split and are
  applied to another, so the achieved false-positive rate is above the target, and the
  gap grows with the batch size relative to the split size. The achieved rate is shown.
- The model part uses one model family and accuracy only.
