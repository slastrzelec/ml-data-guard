# Benchmarks

| Path | Content |
|---|---|
| `configs/smoke.json` | Small synthetic run (under a minute) to check that everything works. |
| `configs/synthetic.json` | I.i.d. synthetic table, batch sizes 1,000 / 10,000 / 100,000. |
| `configs/synthetic_series.json` | Time-ordered synthetic series for the `time_travel` fault. |
| `configs/covertype.json` | UCI Covertype; needs `python tools/download_data.py covertype` first. |
| `results/` | JSON results and generated Markdown reports of the runs that were actually made. |

Run one config (from the repository root):

```bash
python -m mlguard.bench --config benchmarks/configs/smoke.json --out benchmarks/results/smoke.json --report benchmarks/results/smoke.md
```

A result file records the full config, the seed, the package versions and, when `--commit` is passed,
the commit. The methodology and its limits are in `SPEC.md`, section 6.
