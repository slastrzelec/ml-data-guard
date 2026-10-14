"""Release, GitHub Action and PyPI metadata (SPEC section 16.3).

The files are YAML, but the checks below only need lines and a few patterns, so no YAML parser
is added to the test dependencies.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import mlguard

REPO = Path(__file__).resolve().parents[1]
ACTION = (REPO / "action.yml").read_text(encoding="utf-8")
RELEASE = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CI = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
EXAMPLE = (REPO / "examples" / "github-actions" / "validate-data.yml").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def _version() -> str:
    return PYPROJECT["project"]["version"]


def test_package_module_action_and_example_name_the_same_version():
    action_pin = re.search(r"default: ml-data-guard==(\S+)", ACTION)[1]
    example_tag = re.search(r"uses: slastrzelec/ml-data-guard@v(\S+)", EXAMPLE)[1]

    assert {_version(), mlguard.__version__, action_pin, example_tag} == {_version()}


def test_inputs_reach_shell_commands_only_through_environment_variables():
    """`${{ inputs.x }}` inside a `run:` script would let an input inject shell code."""
    in_script = False
    for line in ACTION.splitlines():
        stripped = line.strip()
        if stripped.startswith("run:"):
            in_script = True
            assert "${{" not in stripped, line
            continue
        if in_script and re.match(r"^\s*-?\s*[A-Za-z_-]+:", line):
            in_script = False
        if "${{ inputs." in line:
            key = re.match(r"^\s*(?:-\s*)?([A-Za-z_-]+):", line)
            assert key is not None, line
            assert key[1] != "run", line
            assert not in_script, line


def test_the_action_runs_validate_and_optionally_compare():
    assert 'mlguard validate "$MLGUARD_BATCH" --schema "$MLGUARD_SCHEMA"' in ACTION
    assert "if: ${{ inputs.reference != '' }}" in ACTION
    assert "using: composite" in ACTION


def test_only_the_publish_job_can_request_an_identity_token():
    publish = RELEASE.split("\n  publish:\n", 1)

    assert len(publish) == 2
    assert "id-token: write" not in publish[0]
    assert publish[1].count("id-token: write") == 1
    assert "environment:" in publish[1]
    assert "name: pypi" in publish[1]
    assert "permissions:\n  contents: read" in RELEASE


def test_the_release_stores_no_credential():
    lowered = RELEASE.lower()

    assert "secrets." not in lowered
    assert "password" not in lowered
    assert "api-token" not in lowered
    assert "api_token" not in lowered


def test_a_release_is_checked_before_it_is_published():
    build = RELEASE.split("\n  publish:\n", 1)[0]

    assert "The tag matches the package version" in build
    assert "twine check --strict" in build
    assert "smoke/bin/mlguard --version" in build
    assert "needs: build" in RELEASE
    assert 'tags: ["v*"]' in RELEASE


def test_every_external_action_is_pinned_to_a_version():
    for text in (ACTION, RELEASE, CI, EXAMPLE):
        for ref in re.findall(r"uses:\s*(\S+)", text):
            if ref.startswith("./"):
                continue
            assert re.search(r"@(v\d+(\.\d+)*|release/v\d+)$", ref), ref


def test_ci_runs_the_action_on_a_clean_and_a_broken_batch():
    job = CI.split("\n  action:\n", 1)[1].split("\n  mutation:\n", 1)[0]

    assert job.count("uses: ./") == 2
    assert "batch: examples/data/batch_ok.csv" in job
    assert "batch: examples/data/batch_broken.csv" in job
    assert "continue-on-error: true" in job
    assert "steps.broken.outcome != 'failure'" in job


def _pypi_readme() -> str:
    hook = PYPROJECT["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    text = "".join((REPO / part["path"]).read_text(encoding="utf-8") for part in hook["fragments"])
    for substitution in hook["substitutions"]:
        text = re.sub(substitution["pattern"], substitution["replacement"], text)
    return text


def test_the_readme_on_pypi_has_no_relative_link_or_image():
    targets = re.findall(r"\]\(([^)]+)\)", _pypi_readme())

    assert targets
    assert [target for target in targets if not target.startswith("https://")] == []


def test_images_on_pypi_point_to_raw_files_and_links_to_the_repository():
    text = _pypi_readme()
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    raw = "https://raw.githubusercontent.com/slastrzelec/ml-data-guard/main/"

    assert images
    assert all(
        image.startswith(raw) and (REPO / image.removeprefix(raw)).is_file() for image in images
    )
    assert "(https://github.com/slastrzelec/ml-data-guard/blob/main/SPEC.md)" in text


def test_the_published_files_hold_the_package_and_its_documents_only():
    sdist = PYPROJECT["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    wheel = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]

    assert wheel == ["src/mlguard"]
    assert not any(name.startswith(("data", "benchmarks", "tests", ".env")) for name in sdist)


def test_the_container_runs_the_service_as_a_non_root_user():
    root = (REPO / "Dockerfile").read_text(encoding="utf-8")

    assert "USER app" in root
    assert 'CMD ["python", "-m", "mlguard.api"]' in root
    assert 'pip install ".[api]"' in root


def test_the_container_image_holds_no_data_or_secret():
    ignored = (REPO / ".dockerignore").read_text(encoding="utf-8").split()

    for entry in (".git", "data", ".env", ".env.*", "*.csv", "tests", "benchmarks"):
        assert entry in ignored
