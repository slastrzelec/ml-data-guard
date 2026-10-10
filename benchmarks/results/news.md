# Benchmark report: news

- Data: UCI Online News Popularity, CC BY 4.0, https://doi.org/10.24432/C5NS3V, 39,644 rows, 44 numeric and 2 categorical features.
- Splits (rows): reference 15,857, calibration 9,911, evaluation 9,911, held-out 3,965.
- Reference: static, the reference split.
- Seed 20261010; 1000 calibration batches, 1000 clean batches, 100 faulty batches per fault and severity; target false-positive rate 0.01.
- Commit: f7fd371; ml-data-guard 0.1.0, numpy 2.4.6, pandas 3.0.6, python 3.11.17, scikit-learn 1.9.1, scipy 1.17.1.

Rates are followed by 95% Wilson intervals where space allows; every interval is in the
JSON file. With 100 faulty batches a recall of 0.50 has an interval of
about +/- 0.10.

## Batch size 500

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.010 [0.009, 0.011] | 0.328 [0.300, 0.358] |
| `l1` (calibrated) | 0.011 [0.010, 0.012] | 0.376 [0.347, 0.406] |
| `kl` (calibrated) | 0.011 [0.010, 0.012] | 0.402 [0.372, 0.433] |
| `js` (calibrated) | 0.011 [0.010, 0.012] | 0.389 [0.359, 0.420] |
| `psi` (calibrated) | 0.011 [0.010, 0.012] | 0.384 [0.354, 0.415] |
| `chi2` (calibrated) | 0.013 [0.012, 0.014] | 0.455 [0.424, 0.486] |
| `ks` (calibrated) | 0.010 [0.009, 0.011] | 0.301 [0.273, 0.330] |
| `chi2 p<0.05` (textbook) | 0.042 [0.041, 0.044] | 0.842 [0.818, 0.863] |
| `ks p<0.05` (textbook) | 0.033 [0.031, 0.035] | 0.677 [0.647, 0.705] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.000] | 0.004 [0.002, 0.010] |
| `linf>0.01` (textbook) | 0.983 [0.982, 0.984] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.002 [0.002, 0.003] | 0.101 [0.084, 0.121] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.36 | 0.38 | 0.38 | 0.37 | 0.39 | 0.42 | 0.32 |
| `duplicate_rows` | 0.005 | batch | 0.30 | 0.35 | 0.39 | 0.37 | 0.36 | 0.44 | 0.25 |
| `duplicate_rows` | 0.01 | batch | 0.36 | 0.39 | 0.37 | 0.34 | 0.39 | 0.46 | 0.35 |
| `duplicate_rows` | 0.05 | batch | 0.42 | 0.52 | 0.54 | 0.51 | 0.52 | 0.64 | 0.36 |
| `duplicate_rows` | 0.2 | batch | 0.83 | 0.90 | 0.94 | 0.91 | 0.92 | 0.97 | 0.75 |
| `mean_shift` | 0.01 | feature | 0.66 | 0.71 | 0.77 | 0.77 | 0.77 | 0.73 | 0.65 |
| `mean_shift` | 0.05 | feature | 0.77 | 0.83 | 0.88 | 0.89 | 0.89 | 0.85 | 0.86 |
| `mean_shift` | 0.1 | feature | 0.72 | 0.76 | 0.84 | 0.86 | 0.86 | 0.79 | 0.82 |
| `mean_shift` | 0.25 | feature | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.01 | 0.01 | 0.04 | 0.02 | 0.02 | 0.96 | 0.00 |
| `missing_rate_increase` | 0.005 | feature | 0.00 | 0.01 | 0.90 | 0.21 | 0.32 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.01 | feature | 0.01 | 0.02 | 1.00 | 0.56 | 0.94 | 1.00 | 0.02 |
| `missing_rate_increase` | 0.05 | feature | 0.63 | 0.26 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `new_category` | 0.001 | feature | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 0.02 | 0.01 | 1.00 | 0.15 | 0.53 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 0.00 | 0.01 | 1.00 | 0.81 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 0.02 | 0.33 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.02 | 0.01 | 0.03 | 0.02 | 0.01 | 0.13 | 0.02 |
| `out_of_domain_numeric` | 0.005 | feature | 0.01 | 0.00 | 0.06 | 0.00 | 0.02 | 0.08 | 0.00 |
| `out_of_domain_numeric` | 0.01 | feature | 0.01 | 0.02 | 0.11 | 0.11 | 0.10 | 0.11 | 0.00 |
| `out_of_domain_numeric` | 0.05 | feature | 0.44 | 0.14 | 0.34 | 0.31 | 0.30 | 0.37 | 0.24 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 | 0.28 | 0.02 |
| `pinned_value` | 0.005 | feature | 0.02 | 0.02 | 0.21 | 0.04 | 0.07 | 0.29 | 0.00 |
| `pinned_value` | 0.01 | feature | 0.00 | 0.00 | 0.34 | 0.18 | 0.28 | 0.34 | 0.00 |
| `pinned_value` | 0.05 | feature | 0.34 | 0.17 | 0.42 | 0.41 | 0.39 | 0.42 | 0.10 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.02 | 0.01 | 0.03 | 0.01 | 0.01 | 0.17 | 0.00 |
| `type_change` | 0.001 | feature | 0.01 | 0.01 | 0.07 | 0.03 | 0.03 | 0.99 | 0.00 |
| `type_change` | 0.005 | feature | 0.04 | 0.02 | 0.89 | 0.14 | 0.30 | 1.00 | 0.02 |
| `type_change` | 0.01 | feature | 0.00 | 0.02 | 1.00 | 0.59 | 0.88 | 1.00 | 0.00 |
| `type_change` | 0.05 | feature | 0.66 | 0.32 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.04 |
| `unit_change` | 0.001 | feature | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.06 | 0.02 |
| `unit_change` | 0.005 | feature | 0.01 | 0.00 | 0.13 | 0.04 | 0.05 | 0.17 | 0.02 |
| `unit_change` | 0.01 | feature | 0.00 | 0.01 | 0.14 | 0.12 | 0.14 | 0.13 | 0.03 |
| `unit_change` | 0.05 | feature | 0.29 | 0.15 | 0.34 | 0.29 | 0.27 | 0.36 | 0.14 |
| `unit_change` | 0.2 | feature | 0.96 | 0.95 | 0.98 | 0.98 | 0.98 | 0.98 | 1.00 |
| `variance_change` | 0.02 | feature | 0.68 | 0.72 | 0.77 | 0.77 | 0.77 | 0.76 | 0.66 |
| `variance_change` | 0.05 | feature | 0.67 | 0.72 | 0.75 | 0.76 | 0.76 | 0.73 | 0.69 |
| `variance_change` | 0.1 | feature | 0.71 | 0.74 | 0.77 | 0.77 | 0.78 | 0.76 | 0.67 |
| `variance_change` | 0.25 | feature | 0.97 | 0.92 | 0.97 | 0.97 | 0.97 | 0.98 | 0.92 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `dropped_feature` | 1 | feature | n/a | n/a | n/a | n/a | n/a | 1.00 |
| `duplicate_rows` | 0.001 | batch | 0.87 | 0.74 | 0.00 | 0.01 | 1.00 | 0.14 |
| `duplicate_rows` | 0.005 | batch | 0.85 | 0.71 | 0.00 | 0.01 | 1.00 | 0.06 |
| `duplicate_rows` | 0.01 | batch | 0.86 | 0.74 | 0.00 | 0.00 | 1.00 | 0.10 |
| `duplicate_rows` | 0.05 | batch | 0.95 | 0.78 | 0.00 | 0.01 | 1.00 | 0.09 |
| `duplicate_rows` | 0.2 | batch | 1.00 | 0.97 | 0.00 | 0.10 | 1.00 | 0.07 |
| `mean_shift` | 0.01 | feature | 0.75 | 0.70 | 0.72 | 0.75 | 1.00 | 0.06 |
| `mean_shift` | 0.05 | feature | 0.85 | 0.90 | 0.78 | 0.84 | 1.00 | 0.10 |
| `mean_shift` | 0.1 | feature | 0.83 | 0.93 | 0.71 | 0.77 | 1.00 | 0.07 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.82 | 0.96 | 1.00 | 0.04 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.39 |
| `missing_rate_increase` | 0.001 | feature | 1.00 | 0.05 | 0.00 | 0.00 | 0.97 | 1.00 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.99 | 1.00 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.06 | 0.00 | 0.02 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.02 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.01 | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.99 | 1.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 1.00 | 1.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.99 | 1.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.23 | 0.08 | 0.00 | 0.00 | 0.99 | 0.65 |
| `out_of_domain_numeric` | 0.005 | feature | 0.09 | 0.04 | 0.00 | 0.00 | 0.97 | 0.71 |
| `out_of_domain_numeric` | 0.01 | feature | 0.14 | 0.07 | 0.00 | 0.00 | 1.00 | 0.75 |
| `out_of_domain_numeric` | 0.05 | feature | 0.60 | 0.46 | 0.10 | 0.10 | 1.00 | 0.75 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 | 0.78 |
| `pinned_value` | 0.001 | feature | 0.29 | 0.02 | 0.00 | 0.00 | 1.00 | 0.84 |
| `pinned_value` | 0.005 | feature | 0.29 | 0.03 | 0.00 | 0.00 | 0.99 | 0.84 |
| `pinned_value` | 0.01 | feature | 0.35 | 0.07 | 0.00 | 0.00 | 1.00 | 0.87 |
| `pinned_value` | 0.05 | feature | 0.63 | 0.23 | 0.28 | 0.28 | 0.99 | 0.84 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 0.96 | 0.99 | 1.00 | 0.85 |
| `tiny_noise` (control) | 0.0001 | feature | 0.19 | 0.02 | 0.00 | 0.00 | 0.96 | 0.01 |
| `type_change` | 0.001 | feature | 1.00 | 0.00 | 0.00 | 0.00 | 0.98 | 1.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.06 | 0.00 | 0.00 | 0.97 | 1.00 |
| `type_change` | 0.01 | feature | 1.00 | 0.01 | 0.00 | 0.03 | 1.00 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.11 | 0.05 | 0.00 | 0.00 | 0.99 | 1.00 |
| `unit_change` | 0.005 | feature | 0.20 | 0.05 | 0.00 | 0.00 | 1.00 | 1.00 |
| `unit_change` | 0.01 | feature | 0.15 | 0.06 | 0.00 | 0.00 | 0.99 | 1.00 |
| `unit_change` | 0.05 | feature | 0.47 | 0.21 | 0.15 | 0.15 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 0.99 | 1.00 | 0.69 | 0.83 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.78 | 0.74 | 0.75 | 0.77 | 1.00 | 0.79 |
| `variance_change` | 0.05 | feature | 0.75 | 0.73 | 0.69 | 0.73 | 1.00 | 0.72 |
| `variance_change` | 0.1 | feature | 0.77 | 0.70 | 0.71 | 0.72 | 1.00 | 0.81 |
| `variance_change` | 0.25 | feature | 0.98 | 0.95 | 0.73 | 0.83 | 1.00 | 0.85 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 0.88 |

Not scorable by a distribution distance (the column is gone; only the schema check applies): `dropped_feature`.

## Batch size 1,000

### False alarms on clean batches

| Rule | Per feature | Per batch (any feature) |
|---|---|---|
| `linf` (calibrated) | 0.013 [0.012, 0.014] | 0.391 [0.361, 0.422] |
| `l1` (calibrated) | 0.014 [0.013, 0.015] | 0.456 [0.425, 0.487] |
| `kl` (calibrated) | 0.013 [0.012, 0.014] | 0.444 [0.413, 0.475] |
| `js` (calibrated) | 0.013 [0.012, 0.014] | 0.433 [0.403, 0.464] |
| `psi` (calibrated) | 0.013 [0.012, 0.014] | 0.436 [0.406, 0.467] |
| `chi2` (calibrated) | 0.015 [0.014, 0.016] | 0.485 [0.454, 0.516] |
| `ks` (calibrated) | 0.012 [0.011, 0.013] | 0.353 [0.324, 0.383] |
| `chi2 p<0.05` (textbook) | 0.041 [0.039, 0.043] | 0.828 [0.803, 0.850] |
| `ks p<0.05` (textbook) | 0.033 [0.032, 0.035] | 0.687 [0.658, 0.715] |
| `psi>0.2` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `psi>0.1` (textbook) | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| `linf>0.01` (textbook) | 0.934 [0.931, 0.936] | 1.000 [0.996, 1.000] |
| `schema` (inferred schema) | 0.004 [0.004, 0.005] | 0.192 [0.169, 0.218] |

### Recall, calibrated thresholds

| Fault | Severity | Level | `linf` | `l1` | `kl` | `js` | `psi` | `chi2` | `ks` |
|---|---|---|---|---|---|---|---|---|---|
| `duplicate_rows` | 0.001 | batch | 0.34 | 0.35 | 0.38 | 0.36 | 0.40 | 0.47 | 0.31 |
| `duplicate_rows` | 0.005 | batch | 0.33 | 0.41 | 0.42 | 0.45 | 0.47 | 0.49 | 0.32 |
| `duplicate_rows` | 0.01 | batch | 0.46 | 0.47 | 0.55 | 0.49 | 0.50 | 0.47 | 0.45 |
| `duplicate_rows` | 0.05 | batch | 0.55 | 0.58 | 0.62 | 0.64 | 0.65 | 0.64 | 0.51 |
| `duplicate_rows` | 0.2 | batch | 0.72 | 0.90 | 0.96 | 0.93 | 0.95 | 0.99 | 0.71 |
| `mean_shift` | 0.01 | feature | 0.70 | 0.73 | 0.77 | 0.80 | 0.80 | 0.77 | 0.70 |
| `mean_shift` | 0.05 | feature | 0.83 | 0.84 | 0.92 | 0.92 | 0.92 | 0.90 | 0.89 |
| `mean_shift` | 0.1 | feature | 0.84 | 0.93 | 0.94 | 0.94 | 0.94 | 0.92 | 0.96 |
| `mean_shift` | 0.25 | feature | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.001 | feature | 0.02 | 0.00 | 0.05 | 0.04 | 0.03 | 0.78 | 0.01 |
| `missing_rate_increase` | 0.005 | feature | 0.00 | 0.02 | 1.00 | 0.56 | 0.87 | 1.00 | 0.02 |
| `missing_rate_increase` | 0.01 | feature | 0.04 | 0.08 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.02 |
| `new_category` | 0.001 | feature | 0.01 | 0.02 | 0.06 | 0.05 | 0.04 | 1.00 | n/a |
| `new_category` | 0.005 | feature | 0.01 | 0.02 | 1.00 | 0.68 | 1.00 | 1.00 | n/a |
| `new_category` | 0.01 | feature | 0.00 | 0.03 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.05 | feature | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `new_category` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `out_of_domain_numeric` | 0.001 | feature | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.01 |
| `out_of_domain_numeric` | 0.005 | feature | 0.02 | 0.01 | 0.08 | 0.06 | 0.08 | 0.08 | 0.01 |
| `out_of_domain_numeric` | 0.01 | feature | 0.01 | 0.03 | 0.10 | 0.09 | 0.10 | 0.10 | 0.02 |
| `out_of_domain_numeric` | 0.05 | feature | 0.90 | 0.50 | 0.77 | 0.73 | 0.73 | 0.82 | 0.86 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `pinned_value` | 0.001 | feature | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.21 | 0.01 |
| `pinned_value` | 0.005 | feature | 0.01 | 0.02 | 0.29 | 0.12 | 0.21 | 0.29 | 0.02 |
| `pinned_value` | 0.01 | feature | 0.01 | 0.00 | 0.36 | 0.35 | 0.35 | 0.35 | 0.01 |
| `pinned_value` | 0.05 | feature | 0.85 | 0.46 | 0.75 | 0.73 | 0.72 | 0.74 | 0.71 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `tiny_noise` (control) | 0.0001 | feature | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.10 | 0.02 |
| `type_change` | 0.001 | feature | 0.03 | 0.02 | 0.06 | 0.06 | 0.06 | 0.80 | 0.02 |
| `type_change` | 0.005 | feature | 0.00 | 0.01 | 1.00 | 0.51 | 0.86 | 1.00 | 0.00 |
| `type_change` | 0.01 | feature | 0.02 | 0.05 | 1.00 | 0.99 | 1.00 | 1.00 | 0.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 | 0.01 |
| `type_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| `unit_change` | 0.001 | feature | 0.01 | 0.02 | 0.00 | 0.01 | 0.01 | 0.05 | 0.01 |
| `unit_change` | 0.005 | feature | 0.00 | 0.01 | 0.14 | 0.12 | 0.14 | 0.14 | 0.01 |
| `unit_change` | 0.01 | feature | 0.00 | 0.00 | 0.13 | 0.13 | 0.13 | 0.14 | 0.00 |
| `unit_change` | 0.05 | feature | 0.73 | 0.42 | 0.74 | 0.69 | 0.68 | 0.78 | 0.67 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.71 | 0.74 | 0.78 | 0.79 | 0.79 | 0.78 | 0.69 |
| `variance_change` | 0.05 | feature | 0.74 | 0.75 | 0.77 | 0.78 | 0.78 | 0.77 | 0.75 |
| `variance_change` | 0.1 | feature | 0.77 | 0.77 | 0.84 | 0.86 | 0.87 | 0.82 | 0.77 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### Recall, textbook rules and schema check

| Fault | Severity | Level | `chi2 p<0.05` | `ks p<0.05` | `psi>0.2` | `psi>0.1` | `linf>0.01` | `schema` |
|---|---|---|---|---|---|---|---|---|
| `dropped_feature` | 1 | feature | n/a | n/a | n/a | n/a | n/a | 1.00 |
| `duplicate_rows` | 0.001 | batch | 0.86 | 0.71 | 0.00 | 0.00 | 1.00 | 0.17 |
| `duplicate_rows` | 0.005 | batch | 0.84 | 0.68 | 0.00 | 0.00 | 1.00 | 0.21 |
| `duplicate_rows` | 0.01 | batch | 0.84 | 0.72 | 0.00 | 0.00 | 1.00 | 0.14 |
| `duplicate_rows` | 0.05 | batch | 0.96 | 0.78 | 0.00 | 0.00 | 1.00 | 0.15 |
| `duplicate_rows` | 0.2 | batch | 1.00 | 0.93 | 0.00 | 0.00 | 1.00 | 0.19 |
| `mean_shift` | 0.01 | feature | 0.78 | 0.76 | 0.70 | 0.74 | 0.99 | 0.06 |
| `mean_shift` | 0.05 | feature | 0.92 | 0.94 | 0.77 | 0.80 | 1.00 | 0.12 |
| `mean_shift` | 0.1 | feature | 0.94 | 1.00 | 0.70 | 0.76 | 1.00 | 0.06 |
| `mean_shift` | 0.25 | feature | 1.00 | 1.00 | 0.80 | 0.94 | 1.00 | 0.04 |
| `mean_shift` | 0.5 | feature | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.40 |
| `missing_rate_increase` | 0.001 | feature | 0.90 | 0.02 | 0.00 | 0.00 | 0.94 | 1.00 |
| `missing_rate_increase` | 0.005 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.98 | 1.00 |
| `missing_rate_increase` | 0.01 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.93 | 1.00 |
| `missing_rate_increase` | 0.05 | feature | 1.00 | 0.04 | 1.00 | 1.00 | 1.00 | 1.00 |
| `missing_rate_increase` | 0.2 | feature | 1.00 | 0.02 | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.001 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.92 | 1.00 |
| `new_category` | 0.005 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.93 | 1.00 |
| `new_category` | 0.01 | feature | 1.00 | n/a | 0.00 | 0.00 | 0.89 | 1.00 |
| `new_category` | 0.05 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `new_category` | 0.2 | feature | 1.00 | n/a | 1.00 | 1.00 | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | feature | 0.19 | 0.02 | 0.00 | 0.00 | 0.95 | 0.73 |
| `out_of_domain_numeric` | 0.005 | feature | 0.12 | 0.02 | 0.00 | 0.00 | 0.95 | 0.75 |
| `out_of_domain_numeric` | 0.01 | feature | 0.11 | 0.07 | 0.00 | 0.00 | 0.93 | 0.81 |
| `out_of_domain_numeric` | 0.05 | feature | 0.91 | 1.00 | 0.10 | 0.10 | 1.00 | 0.83 |
| `out_of_domain_numeric` | 0.2 | feature | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.82 |
| `pinned_value` | 0.001 | feature | 0.30 | 0.02 | 0.00 | 0.00 | 0.95 | 0.84 |
| `pinned_value` | 0.005 | feature | 0.30 | 0.03 | 0.00 | 0.00 | 0.95 | 0.84 |
| `pinned_value` | 0.01 | feature | 0.40 | 0.02 | 0.00 | 0.00 | 0.97 | 0.87 |
| `pinned_value` | 0.05 | feature | 0.84 | 0.95 | 0.28 | 0.28 | 1.00 | 0.84 |
| `pinned_value` | 0.2 | feature | 1.00 | 1.00 | 0.98 | 0.99 | 1.00 | 0.85 |
| `tiny_noise` (control) | 0.0001 | feature | 0.13 | 0.04 | 0.00 | 0.00 | 0.96 | 0.00 |
| `type_change` | 0.001 | feature | 0.88 | 0.03 | 0.00 | 0.00 | 0.96 | 1.00 |
| `type_change` | 0.005 | feature | 1.00 | 0.03 | 0.00 | 0.00 | 0.91 | 1.00 |
| `type_change` | 0.01 | feature | 1.00 | 0.02 | 0.00 | 0.00 | 0.92 | 1.00 |
| `type_change` | 0.05 | feature | 1.00 | 0.01 | 1.00 | 1.00 | 1.00 | 1.00 |
| `type_change` | 0.2 | feature | 1.00 | 0.02 | 1.00 | 1.00 | 1.00 | 1.00 |
| `unit_change` | 0.001 | feature | 0.09 | 0.04 | 0.00 | 0.00 | 0.95 | 0.99 |
| `unit_change` | 0.005 | feature | 0.14 | 0.01 | 0.00 | 0.00 | 0.94 | 1.00 |
| `unit_change` | 0.01 | feature | 0.17 | 0.02 | 0.00 | 0.00 | 0.91 | 1.00 |
| `unit_change` | 0.05 | feature | 0.84 | 0.81 | 0.15 | 0.15 | 1.00 | 1.00 |
| `unit_change` | 0.2 | feature | 1.00 | 1.00 | 0.62 | 0.81 | 1.00 | 1.00 |
| `variance_change` | 0.02 | feature | 0.78 | 0.74 | 0.75 | 0.75 | 1.00 | 0.81 |
| `variance_change` | 0.05 | feature | 0.78 | 0.81 | 0.69 | 0.74 | 1.00 | 0.73 |
| `variance_change` | 0.1 | feature | 0.83 | 0.82 | 0.71 | 0.71 | 1.00 | 0.82 |
| `variance_change` | 0.25 | feature | 1.00 | 1.00 | 0.72 | 0.79 | 1.00 | 0.84 |
| `variance_change` | 0.5 | feature | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 0.90 |

Not scorable by a distribution distance (the column is gone; only the schema check applies): `dropped_feature`.

## Does the alert matter?

Model: gradient boosting trained on the reference split; clean held-out accuracy 0.6749. Accuracy drop = clean minus faulty on held-out batches of 2,000 rows (positive = the fault hurts). Best recall = highest calibrated recall of any distance at batch size 1,000; schema = recall of the schema check at the same batch size.

| Fault | Severity | Accuracy drop [95% CI] | Best recall | Schema |
|---|---|---|---|---|
| `dropped_feature` | 1 | +0.0020 [+0.0012, +0.0029] | n/a | 1.00 |
| `duplicate_rows` | 0.001 | +0.0000 [-0.0001, +0.0001] | 0.47 | 0.17 |
| `duplicate_rows` | 0.005 | +0.0001 [-0.0001, +0.0003] | 0.49 | 0.21 |
| `duplicate_rows` | 0.01 | +0.0001 [-0.0002, +0.0004] | 0.55 | 0.14 |
| `duplicate_rows` | 0.05 | -0.0003 [-0.0010, +0.0004] | 0.65 | 0.15 |
| `duplicate_rows` | 0.2 | +0.0008 [-0.0003, +0.0019] | 0.99 | 0.19 |
| `mean_shift` | 0.01 | -0.0001 [-0.0005, +0.0002] | 0.80 | 0.06 |
| `mean_shift` | 0.05 | +0.0008 [+0.0004, +0.0011] | 0.92 | 0.12 |
| `mean_shift` | 0.1 | +0.0011 [+0.0005, +0.0018] | 0.96 | 0.06 |
| `mean_shift` | 0.25 | +0.0005 [-0.0002, +0.0012] | 1.00 | 0.04 |
| `mean_shift` | 0.5 | +0.0014 [+0.0006, +0.0022] | 1.00 | 0.40 |
| `missing_rate_increase` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.78 | 1.00 |
| `missing_rate_increase` | 0.005 | +0.0000 [-0.0000, +0.0001] | 1.00 | 1.00 |
| `missing_rate_increase` | 0.01 | -0.0000 [-0.0001, +0.0001] | 1.00 | 1.00 |
| `missing_rate_increase` | 0.05 | +0.0001 [-0.0001, +0.0002] | 1.00 | 1.00 |
| `missing_rate_increase` | 0.2 | +0.0003 [-0.0001, +0.0006] | 1.00 | 1.00 |
| `new_category` | 0.001 | +0.0000 [+0.0000, +0.0001] | 1.00 | 1.00 |
| `new_category` | 0.005 | +0.0001 [-0.0000, +0.0002] | 1.00 | 1.00 |
| `new_category` | 0.01 | +0.0001 [+0.0000, +0.0003] | 1.00 | 1.00 |
| `new_category` | 0.05 | +0.0006 [+0.0003, +0.0008] | 1.00 | 1.00 |
| `new_category` | 0.2 | +0.0019 [+0.0015, +0.0024] | 1.00 | 1.00 |
| `out_of_domain_numeric` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.10 | 0.73 |
| `out_of_domain_numeric` | 0.005 | -0.0000 [-0.0001, +0.0001] | 0.08 | 0.75 |
| `out_of_domain_numeric` | 0.01 | +0.0000 [-0.0001, +0.0001] | 0.10 | 0.81 |
| `out_of_domain_numeric` | 0.05 | +0.0003 [-0.0000, +0.0006] | 0.90 | 0.83 |
| `out_of_domain_numeric` | 0.2 | +0.0009 [+0.0004, +0.0015] | 1.00 | 0.82 |
| `pinned_value` | 0.001 | -0.0000 [-0.0000, +0.0000] | 0.21 | 0.84 |
| `pinned_value` | 0.005 | +0.0000 [-0.0000, +0.0001] | 0.29 | 0.84 |
| `pinned_value` | 0.01 | +0.0000 [-0.0001, +0.0001] | 0.36 | 0.87 |
| `pinned_value` | 0.05 | +0.0003 [+0.0001, +0.0006] | 0.85 | 0.84 |
| `pinned_value` | 0.2 | +0.0009 [+0.0005, +0.0014] | 1.00 | 0.85 |
| `tiny_noise` | 0.0001 | +0.0000 [+0.0000, +0.0001] | 0.10 | 0.00 |
| `type_change` | 0.001 | +0.0000 [-0.0000, +0.0000] | 0.80 | 1.00 |
| `type_change` | 0.005 | +0.0000 [-0.0000, +0.0001] | 1.00 | 1.00 |
| `type_change` | 0.01 | +0.0000 [-0.0001, +0.0001] | 1.00 | 1.00 |
| `type_change` | 0.05 | -0.0000 [-0.0002, +0.0001] | 1.00 | 1.00 |
| `type_change` | 0.2 | +0.0004 [+0.0001, +0.0008] | 1.00 | 1.00 |
| `unit_change` | 0.001 | +0.0000 [-0.0000, +0.0001] | 0.05 | 0.99 |
| `unit_change` | 0.005 | +0.0001 [-0.0000, +0.0001] | 0.14 | 1.00 |
| `unit_change` | 0.01 | +0.0000 [-0.0001, +0.0001] | 0.14 | 1.00 |
| `unit_change` | 0.05 | +0.0001 [-0.0001, +0.0003] | 0.78 | 1.00 |
| `unit_change` | 0.2 | +0.0008 [+0.0004, +0.0013] | 1.00 | 1.00 |
| `variance_change` | 0.02 | +0.0000 [-0.0002, +0.0003] | 0.79 | 0.81 |
| `variance_change` | 0.05 | +0.0001 [-0.0001, +0.0004] | 0.78 | 0.73 |
| `variance_change` | 0.1 | -0.0002 [-0.0005, +0.0001] | 0.87 | 0.82 |
| `variance_change` | 0.25 | -0.0000 [-0.0004, +0.0004] | 1.00 | 0.84 |
| `variance_change` | 0.5 | +0.0009 [+0.0003, +0.0015] | 1.00 | 0.90 |

Model loses more than 0.005 accuracy but no detector reaches recall 0.5: none.

A detector reaches recall 0.9 but accuracy moves by less than 0.005: `duplicate_rows` at 0.2, `mean_shift` at 0.05, `mean_shift` at 0.1, `mean_shift` at 0.25, `mean_shift` at 0.5, `missing_rate_increase` at 0.005, `missing_rate_increase` at 0.01, `missing_rate_increase` at 0.05, `missing_rate_increase` at 0.2, `new_category` at 0.001, `new_category` at 0.005, `new_category` at 0.01, `new_category` at 0.05, `new_category` at 0.2, `out_of_domain_numeric` at 0.05, `out_of_domain_numeric` at 0.2, `pinned_value` at 0.2, `type_change` at 0.005, `type_change` at 0.01, `type_change` at 0.05, `type_change` at 0.2, `unit_change` at 0.2, `variance_change` at 0.25, `variance_change` at 0.5.

## What this report does not measure

- Real incidents: every fault is synthetic and follows the documented injector.
- Several faults at once, or a fault that builds up slowly over many batches.
- Calibration on unlimited fresh data: thresholds come from one finite split and are
  applied to another, so the achieved false-positive rate is above the target, and the
  gap grows with the batch size relative to the split size. The achieved rate is shown.
- The model part uses one model family and accuracy only.
