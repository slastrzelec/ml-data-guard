"""Security and hygiene rules from SPEC section 3, enforced as tests."""

from __future__ import annotations

import ast
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src" / "mlguard"

FORBIDDEN_SUFFIXES = {
    ".csv", ".tsv", ".parquet", ".feather", ".arrow", ".h5", ".hdf5", ".npy", ".npz",
    ".pkl", ".pickle", ".joblib", ".zip", ".gz", ".bz2", ".xz", ".7z", ".tar",
    ".xls", ".xlsx", ".sqlite", ".db", ".pem", ".key",
}  # fmt: skip
MAX_TRACKED_FILE_BYTES = 500_000

UNSAFE_MODULES = {"pickle", "cPickle", "dill", "cloudpickle", "joblib", "marshal", "shelve"}
NETWORK_MODULES = {
    "socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp", "ftplib",
}  # fmt: skip
UNSAFE_CALLS = {"eval", "exec", "compile", "__import__"}


def _tracked_files() -> list[Path]:
    git = shutil.which("git")
    if git is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout: tracked files cannot be listed")
    output = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    return [REPO / name for name in output.split("\0") if name]


def _source_trees() -> list[tuple[Path, ast.AST]]:
    files = sorted(SOURCE.rglob("*.py"))
    assert files, "no source files found"
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in files]


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_no_data_files_are_tracked() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in _tracked_files()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    assert offenders == []


def test_data_directory_tracks_only_its_readme() -> None:
    tracked = {
        str(path.relative_to(REPO).as_posix())
        for path in _tracked_files()
        if path.relative_to(REPO).parts[0] == "data"
    }
    assert tracked <= {"data/README.md"}


def test_no_env_files_are_tracked() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in _tracked_files()
        if path.name == ".env" or path.name.startswith(".env.")
    ]
    assert offenders == []


def test_no_large_files_are_tracked() -> None:
    offenders = [
        str(path.relative_to(REPO))
        for path in _tracked_files()
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES
    ]
    assert offenders == []


def test_gitignore_excludes_secrets_and_data() -> None:
    rules = set((REPO / ".gitignore").read_text(encoding="utf-8").split())
    assert {".env", ".env.*", "data/*", "!data/README.md", "*.csv", "*.parquet", "*.pkl"} <= rules


def test_source_does_not_use_unsafe_deserialisation() -> None:
    for path, tree in _source_trees():
        assert not (_imported_modules(tree) & UNSAFE_MODULES), path


def test_source_does_not_evaluate_code() -> None:
    for path, tree in _source_trees():
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (calls & UNSAFE_CALLS), path


def test_source_does_not_load_pickled_arrays() -> None:
    for path, tree in _source_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "allow_pickle":
                assert isinstance(node.value, ast.Constant), path
                assert node.value.value is False, path


def test_source_does_not_import_network_modules() -> None:
    for path, tree in _source_trees():
        assert not (_imported_modules(tree) & NETWORK_MODULES), path


def test_opening_a_socket_fails_inside_the_test_suite() -> None:
    with pytest.raises(RuntimeError, match="offline"):
        socket.socket()
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("example.org", 443))
