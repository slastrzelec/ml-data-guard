"""Command line: exit codes, output formats, and no cell values in the output by default."""

from __future__ import annotations

import itertools
import json
import runpy
import shlex
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlguard import __version__, cli
from mlguard.datasets import make_synthetic, make_synthetic_series
from mlguard.inject import FAULTS
from mlguard.schema import Schema

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("cli")
    table = make_synthetic(12_000, seed=17).frame
    reference, batch = table.iloc[:10_000], table.iloc[10_000:]
    rng = np.random.default_rng(0)
    broken = FAULTS["new_category"].apply(batch, "c_small", 0.1, rng)
    broken = FAULTS["unit_change"].apply(broken, "x_normal", 0.1, rng)
    series = make_synthetic_series(1_000, seed=17).frame
    series.insert(0, "row_id", np.arange(1_000))
    paths = {
        name: folder / f"{name}.csv"
        for name in ("reference", "ok", "broken", "training", "serving")
    }
    reference.to_csv(paths["reference"], index=False)
    batch.to_csv(paths["ok"], index=False)
    broken.to_csv(paths["broken"], index=False)
    series.to_csv(paths["training"], index=False)
    FAULTS["time_travel"].apply(series, "lag_1", 2, rng).to_csv(paths["serving"], index=False)
    paths["schema"] = folder / "schema.json"
    assert (
        cli.main(
            ["infer", str(paths["reference"]), "-o", str(paths["schema"]), "--exclude", "target"]
        )
        == 0
    )
    return paths


def run(capsys: pytest.CaptureFixture[str], *arguments: object) -> tuple[int, str, str]:
    code = cli.main([str(argument) for argument in arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_infer_writes_a_loadable_schema(files: dict[str, Path], tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    schema = Schema.load(files["schema"])
    assert len(schema.features) == 7
    assert schema.ignored == ("target",)
    target = tmp_path / "hashed.json"
    code, out, _ = run(
        capsys, "infer", files["reference"], "-o", target, "--exclude", "target",
        "--domains", "hashed", "--ranges", "none", "--categorical", "x_count",
        "--max-domain-size", 50,
    )  # fmt: skip
    assert code == 0
    assert "8 feature" not in out
    assert "7 feature(s)" in out
    hashed = Schema.load(target)
    assert hashed["c_small"].domain_storage == "hashed"
    assert hashed["x_count"].kind == "categorical"
    assert hashed["x_normal"].maximum is None


def test_validate_exit_codes_and_text_output(files: dict[str, Path], capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = run(capsys, "validate", files["ok"], "--schema", files["schema"])
    assert code == 0
    assert "validate: 0 error(s), 0 warning(s)" in out
    code, out, _ = run(capsys, "validate", files["broken"], "--schema", files["schema"])
    assert code == 1
    assert "ERROR   c_small: UNEXPECTED_VALUES" in out
    assert "ERROR   x_normal: OUT_OF_RANGE" in out
    assert "suggestion:" in out
    assert "validate: 2 error(s), 0 warning(s)" in out
    assert "__new__" not in out  # the unexpected category itself is not printed
    code, out, _ = run(
        capsys, "validate", files["broken"], "--schema", files["schema"], "--show-values"
    )
    assert code == 1
    assert "__new__" in out


def test_warnings_fail_only_in_strict_mode(files: dict[str, Path], tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    extra = tmp_path / "extra.csv"
    pd.read_csv(files["ok"]).assign(surprise=1).to_csv(extra, index=False)
    code, out, _ = run(capsys, "validate", extra, "--schema", files["schema"])
    assert code == 0
    assert "WARNING surprise: UNEXPECTED_FEATURE" in out
    assert "0 error(s), 1 warning(s)" in out
    code, _, _ = run(capsys, "validate", extra, "--schema", files["schema"], "--strict")
    assert code == 1


def test_json_output_is_one_document_with_a_fixed_shape(files: dict[str, Path], capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = run(
        capsys, "validate", files["broken"], "--schema", files["schema"], "--format", "json"
    )
    document = json.loads(out)
    assert code == 1
    assert set(document) == {"command", "status", "n_rows", "schema_version", "anomalies"}
    assert (document["command"], document["status"], document["n_rows"]) == (
        "validate",
        "anomalies",
        2_000,
    )
    assert {entry["type"] for entry in document["anomalies"]} == {
        "UNEXPECTED_VALUES",
        "OUT_OF_RANGE",
    }
    for entry in document["anomalies"]:
        assert set(entry) == {
            "type", "feature", "level", "description", "statistic", "threshold", "suggestion",
        }  # fmt: skip
    code, out, _ = run(
        capsys, "validate", files["ok"], "--schema", files["schema"], "--format", "json"
    )
    assert code == 0
    assert json.loads(out)["status"] == "ok"
    assert json.loads(out)["anomalies"] == []


def test_compare_exit_codes(files: dict[str, Path], capsys) -> None:  # type: ignore[no-untyped-def]
    common = ("--schema", files["schema"], "--n-batches", 200, "--seed", 2)
    code, out, _ = run(capsys, "compare", files["reference"], files["ok"], *common)
    assert (code, out.strip().splitlines()[-1]) == (0, "compare: 0 error(s), 0 warning(s)")
    code, out, _ = run(
        capsys, "compare", files["reference"], files["broken"], *common, "--format", "json"
    )
    document = json.loads(out)
    assert code == 1
    assert set(document) == {
        "command",
        "status",
        "n_reference_rows",
        "n_rows",
        "detector",
        "anomalies",
    }
    assert {entry["feature"] for entry in document["anomalies"]} == {"c_small", "x_normal"}
    code, out, _ = run(
        capsys, "compare", files["reference"], files["ok"], "--schema", files["schema"],
        "--threshold", 0.0, "--detector", "l1",
    )  # fmt: skip
    assert code == 1
    assert "l1 distance" in out
    assert "(fixed)" in out


def test_skew_exit_codes(files: dict[str, Path], capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = run(capsys, "skew", files["training"], files["training"], "--key", "row_id")
    assert (code, out.strip()) == (0, "skew: 0 error(s), 0 warning(s)")
    code, out, _ = run(
        capsys, "skew", files["training"], files["serving"], "--key", "row_id", "--format", "json"
    )
    document = json.loads(out)
    assert code == 1
    assert document["n_joined"] == 1_000
    assert document["mismatches"]["lag_1"] == 998
    assert [entry["feature"] for entry in document["anomalies"]] == ["lag_1"]
    code, _, _ = run(
        capsys, "skew", files["training"], files["serving"], "--key", "row_id",
        "--features", "lag_mean_3, unrelated",
    )  # fmt: skip
    assert code == 0
    code, _, _ = run(
        capsys, "skew", files["training"], files["serving"], "--key", "row_id", "--max-share", 1.0
    )
    assert code == 0


def test_unreadable_input_is_exit_code_two(files: dict[str, Path], tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, out, err = run(capsys, "validate", tmp_path / "missing.csv", "--schema", files["schema"])
    assert (code, out) == (2, "")
    assert "error: file not found" in err
    text = tmp_path / "data.txt"
    text.write_text("a,b\n1,2\n", encoding="utf-8")
    code, _, err = run(capsys, "validate", text, "--schema", files["schema"])
    assert code == 2
    assert "unsupported file type" in err
    broken = tmp_path / "schema.json"
    broken.write_text("{", encoding="utf-8")
    code, _, err = run(capsys, "validate", files["ok"], "--schema", broken)
    assert code == 2
    assert "not valid JSON" in err
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    code, _, err = run(capsys, "validate", empty, "--schema", files["schema"])
    assert code == 2
    code, _, err = run(capsys, "skew", files["training"], files["serving"], "--key", "nope")
    assert code == 2
    assert "no key column" in err
    code, _, err = run(
        capsys,
        "compare",
        files["reference"],
        files["ok"],
        "--schema",
        files["schema"],
        "--detector",
        "x",
    )
    assert code == 2
    assert "unknown detector" in err


def test_wrong_usage_is_exit_code_two(capsys) -> None:  # type: ignore[no-untyped-def]
    for arguments in ([], ["validate"], ["unknown-command"], ["validate", "x.csv"]):
        with pytest.raises(SystemExit) as raised:
            cli.main(arguments)
        assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        cli.main(["--version"])
    assert raised.value.code == 0
    assert f"mlguard {__version__}" in capsys.readouterr().out


def test_bench_is_reachable_through_the_command_line(tmp_path: Path) -> None:
    config = {
        "dataset": "synthetic", "dataset_rows": 6000, "batch_sizes": [100], "model": False,
        "n_calibration_batches": 100, "n_clean_batches": 10, "n_seeds": 2, "faults": ["tiny_noise"],
    }  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "result.json"
    assert cli.main(["bench", "--config", str(tmp_path / "config.json"), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["dataset"] == "synthetic"


def test_parquet_needs_pyarrow_or_works(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises((ValueError, OSError)):
        cli.read_table(path)


# --------------------------------------------------------------------------- the walk-through


def walkthrough_steps() -> list[tuple[str, int]]:
    """(command, expected exit code) pairs from the fenced blocks of examples/README.md."""
    lines = (REPO / "examples" / "README.md").read_text(encoding="utf-8").splitlines()
    steps = []
    for line, following in itertools.pairwise(lines):
        if following.startswith("# -> exit code "):
            steps.append((line, int(following.removeprefix("# -> exit code "))))
    return steps


def test_the_walkthrough_runs_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    steps = walkthrough_steps()
    assert len(steps) == 8
    shutil.copytree(REPO / "examples", tmp_path / "examples", ignore=shutil.ignore_patterns("data"))
    monkeypatch.chdir(tmp_path)
    for command, expected in steps:
        words = shlex.split(command)
        if words[0] == "mlguard":
            code = cli.main(words[1:])
        else:
            assert words[0] == "python"
            monkeypatch.setattr("sys.argv", words[1:])
            with pytest.raises(SystemExit) as raised:
                runpy.run_path(words[1], run_name="__main__")
            code = raised.value.code
        assert code == expected, command
    output = capsys.readouterr().out
    assert "x_normal: OUT_OF_RANGE" in output
    assert "x_zero_inflated: MISSING_FEATURE" in output
    assert "c_small: UNEXPECTED_VALUES" in output
    assert "lag_1: FEATURE_SKEW - 2999 of 3000" in output
    assert "careless_features: FAILED" in output
    assert "careful_features: passed" in output
