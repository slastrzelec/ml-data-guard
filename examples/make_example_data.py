"""Write the small synthetic tables used by the walk-through in examples/README.md.

Usage: python examples/make_example_data.py [output directory, default examples/data]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mlguard.datasets import make_synthetic, make_synthetic_series
from mlguard.inject import FAULTS


def main(argv: list[str]) -> int:
    target = Path(argv[1] if len(argv) > 1 else "examples/data")
    target.mkdir(parents=True, exist_ok=True)

    table = make_synthetic(22_000, seed=11).frame
    table.insert(0, "id", np.arange(len(table)))
    reference, batch = table.iloc[:20_000], table.iloc[20_000:]
    reference.to_csv(target / "reference.csv", index=False)
    batch.to_csv(target / "batch_ok.csv", index=False)

    rng = np.random.default_rng(5)
    broken = FAULTS["unit_change"].apply(batch, "x_normal", 0.10, rng)
    broken = FAULTS["new_category"].apply(broken, "c_small", 0.05, rng)
    broken = FAULTS["dropped_feature"].apply(broken, "x_zero_inflated", 1, rng)
    broken.to_csv(target / "batch_broken.csv", index=False)

    series = make_synthetic_series(3_000, seed=11).frame
    series.insert(0, "row_id", np.arange(len(series)))
    series.to_csv(target / "training_log.csv", index=False)
    # The serving pipeline reads lag_1 one step too late: it sees the future.
    FAULTS["time_travel"].apply(series, "lag_1", 1, rng).to_csv(
        target / "serving_log.csv", index=False
    )
    print(f"wrote 5 tables to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
