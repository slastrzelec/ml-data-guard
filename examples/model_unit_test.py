"""Model unit test: feed rows that the schema allows to a function with a hidden assumption.

Usage: python examples/model_unit_test.py [schema file, default examples/data/schema.json]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from mlguard.modeltest import run_model_test
from mlguard.schema import Schema


def careless_features(batch: pd.DataFrame) -> np.ndarray:
    """Assumes that x_sentinel is positive. The schema says it can be -1."""
    return np.log(batch["x_sentinel"].to_numpy())


def careful_features(batch: pd.DataFrame) -> np.ndarray:
    """Treats the sentinel and missing values explicitly."""
    values = batch["x_sentinel"].to_numpy(dtype=float)
    return np.where(np.isnan(values) | (values <= 0), 0.0, np.log1p(np.abs(values)))


def main(argv: list[str]) -> int:
    schema = Schema.load(argv[1] if len(argv) > 1 else "examples/data/schema.json")
    failed = False
    for function in (careful_features, careless_features):
        result = run_model_test(function, schema, n_rows=200, seed=0)
        print(f"{function.__name__}: {'passed' if result else 'FAILED - ' + result.problem}")
        if not result:
            failed = True
            print(f"  reproducing row: x_sentinel = {result.failing_example['x_sentinel']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
