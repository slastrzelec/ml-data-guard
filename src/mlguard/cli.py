"""Command line: ``mlguard infer | validate | compare | skew | bench``.

Exit codes: 0 no anomaly, 1 at least one error-level anomaly (warnings too with ``--strict``),
2 wrong usage or unreadable input. With ``--format json`` the result goes to standard output as
one JSON document, so the command can be a step of a CI pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from mlguard import __version__, bench
from mlguard.compare import compare_batches, feature_skew, skew_anomalies
from mlguard.schema import DOMAIN_STORAGE, RANGE_MODES, Schema, infer_schema
from mlguard.validate import Anomaly, has_errors, validate_batch

EXIT_OK, EXIT_ANOMALIES, EXIT_USAGE = 0, 1, 2


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV file, or a Parquet file when ``pyarrow`` is installed."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise ValueError("reading Parquet needs the pyarrow package") from error
    raise ValueError(f"unsupported file type {suffix!r}: use .csv or .parquet")


def _names(text: str | None) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()] if text else []


def _report(
    command: str, anomalies: list[Anomaly], details: dict[str, Any], arguments: argparse.Namespace
) -> int:
    failed = has_errors(anomalies, strict=getattr(arguments, "strict", False))
    if arguments.format == "json":
        document = {
            "command": command,
            "status": "anomalies" if failed else "ok",
            **details,
            "anomalies": [anomaly.to_dict() for anomaly in anomalies],
        }
        print(json.dumps(document, indent=2, ensure_ascii=False))
    else:
        for anomaly in anomalies:
            where = anomaly.feature if anomaly.feature is not None else "(batch)"
            print(
                f"{anomaly.level.upper():7} {where}: {anomaly.type.value} - {anomaly.description}"
            )
            if anomaly.suggestion:
                print(f"        suggestion: {anomaly.suggestion}")
        errors = sum(anomaly.level == "error" for anomaly in anomalies)
        print(f"{command}: {errors} error(s), {len(anomalies) - errors} warning(s)")
    return EXIT_ANOMALIES if failed else EXIT_OK


def _infer(arguments: argparse.Namespace) -> int:
    schema = infer_schema(
        read_table(arguments.data),
        categorical=_names(arguments.categorical),
        exclude=_names(arguments.exclude),
        numeric_ranges=arguments.ranges,
        max_domain_size=arguments.max_domain_size,
        domains=arguments.domains,
    )
    schema.save(arguments.output)
    print(f"infer: wrote {arguments.output} with {len(schema.features)} feature(s)")
    return EXIT_OK


def _validate(arguments: argparse.Namespace) -> int:
    batch = read_table(arguments.data)
    schema = Schema.load(arguments.schema)
    anomalies = validate_batch(batch, schema, show_values=arguments.show_values)
    details = {"n_rows": len(batch), "schema_version": schema.version}
    return _report("validate", anomalies, details, arguments)


def _compare(arguments: argparse.Namespace) -> int:
    reference, batch = read_table(arguments.reference), read_table(arguments.batch)
    anomalies = compare_batches(
        reference,
        batch,
        Schema.load(arguments.schema),
        detector=arguments.detector,
        threshold=arguments.threshold,
        target_fpr=arguments.target_fpr,
        n_batches=arguments.n_batches,
        seed=arguments.seed,
    )
    details = {
        "n_reference_rows": len(reference),
        "n_rows": len(batch),
        "detector": arguments.detector,
    }
    return _report("compare", anomalies, details, arguments)


def _skew(arguments: argparse.Namespace) -> int:
    report = feature_skew(
        read_table(arguments.training),
        read_table(arguments.serving),
        key=arguments.key,
        features=_names(arguments.features) or None,
        rtol=arguments.rtol,
    )
    details = {
        "n_joined": report.n_joined,
        "n_training_only": report.n_training_only,
        "n_serving_only": report.n_serving_only,
        "mismatches": report.mismatches,
    }
    return _report(
        "skew", skew_anomalies(report, max_share=arguments.max_share), details, arguments
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlguard", description="Data validation for ML pipelines."
    )
    parser.add_argument("--version", action="version", version=f"mlguard {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def output_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--format", choices=("text", "json"), default="text")

    infer = commands.add_parser("infer", help="infer a schema from reference data")
    infer.add_argument("data")
    infer.add_argument("-o", "--output", required=True, help="schema file to write")
    infer.add_argument("--categorical", help="comma-separated numeric columns that are codes")
    infer.add_argument("--exclude", help="comma-separated columns that are not features")
    infer.add_argument("--ranges", choices=RANGE_MODES, default="padded")
    infer.add_argument("--domains", choices=DOMAIN_STORAGE, default="plain")
    infer.add_argument("--max-domain-size", type=int, default=100)
    infer.set_defaults(run=_infer)

    validate = commands.add_parser("validate", help="check one batch against a schema")
    validate.add_argument("data")
    validate.add_argument("--schema", required=True)
    validate.add_argument("--strict", action="store_true", help="warnings fail the command too")
    validate.add_argument(
        "--show-values", action="store_true", help="print unexpected category values"
    )
    output_options(validate)
    validate.set_defaults(run=_validate)

    compare = commands.add_parser("compare", help="detect distribution shift against a reference")
    compare.add_argument("reference")
    compare.add_argument("batch")
    compare.add_argument("--schema", required=True)
    compare.add_argument("--detector", default="linf")
    compare.add_argument("--threshold", type=float, help="fixed limit instead of calibration")
    compare.add_argument("--target-fpr", type=float, default=0.01)
    compare.add_argument("--n-batches", type=int, default=500)
    compare.add_argument("--seed", type=int, default=0)
    output_options(compare)
    compare.set_defaults(run=_compare)

    skew = commands.add_parser("skew", help="compare training and serving features row by row")
    skew.add_argument("training")
    skew.add_argument("serving")
    skew.add_argument("--key", required=True, help="column that identifies a row in both tables")
    skew.add_argument("--features", help="comma-separated features (default: all shared columns)")
    skew.add_argument("--max-share", type=float, default=0.0)
    skew.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance for numbers")
    output_options(skew)
    skew.set_defaults(run=_skew)

    commands.add_parser("bench", help="run the fault-injection benchmark", add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments_list = sys.argv[1:] if argv is None else list(argv)
    if arguments_list[:1] == ["bench"]:
        return bench.main(arguments_list[1:])
    arguments = _parser().parse_args(arguments_list)
    try:
        return arguments.run(arguments)
    except (
        ValueError,
        KeyError,
        OSError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
