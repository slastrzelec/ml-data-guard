"""The web service (SPEC section 17): limits, data security, consistency with the command line."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mlguard import __version__
from mlguard.api import service as api_module
from mlguard.api.service import (
    MAX_COLUMNS,
    MAX_FILE_BYTES,
    MAX_ROWS,
    create_app,
)
from mlguard.cli import main

SENTINEL = "zz-private-value-7f3a"
STATIC = Path(api_module.__file__).parent / "static"
WIDE = ",".join(f"c{i}" for i in range(MAX_COLUMNS + 1)).encode() + b"\n" + b"1\n"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as session:
        yield session


@pytest.fixture(scope="module")
def quiet_client() -> TestClient:
    """A client that receives the 500 answer instead of the server's exception.

    Clients are module-scoped because the test transport opens a local socket pair for its event
    loop, which the offline guard of ``conftest.py`` refuses inside a test.
    """
    with TestClient(create_app(), raise_server_exceptions=False) as session:
        yield session


def _reference(n_rows: int = 3_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "x": rng.normal(50.0, 10.0, n_rows),
            "count": rng.poisson(4.0, n_rows),
            "c": rng.choice([SENTINEL, "b", "c"], n_rows),
        }
    )


def _broken(reference: pd.DataFrame) -> pd.DataFrame:
    batch = reference.sample(600, random_state=1).reset_index(drop=True)
    batch.loc[:59, "x"] *= 1000.0
    batch.loc[:29, "c"] = SENTINEL + "-new"
    return batch.drop(columns=["count"])


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _upload(name: str, content: bytes) -> tuple[str, bytes, str]:
    return (name, content, "text/csv")


def _schema(client: TestClient, reference: pd.DataFrame, **form: str) -> str:
    response = client.post(
        "/api/v1/infer",
        files={"data": _upload("reference.csv", _csv(reference))},
        data={"exclude": "id", **form},
    )
    assert response.status_code == 200, response.text
    return json.dumps(response.json()["schema"])


# ------------------------------------------------------------------ basics


def test_health_reports_the_package_version(client):
    assert client.get("/api/v1/health").json() == {"status": "ok", "version": __version__}


def test_the_page_and_its_files_are_served(client):
    page = client.get("/")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "/static/app.js" in page.text
    for name in ("app.js", "style.css"):
        assert client.get(f"/static/{name}").status_code == 200


def test_security_headers_are_set_on_every_response(client):
    for path in ("/", "/api/v1/health", "/static/app.js", "/no-such-page"):
        headers = client.get(path).headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["cache-control"] == "no-store"
        csp = headers["content-security-policy"]
        assert "default-src 'none'" in csp
        assert "script-src 'self';" in csp
        assert "cdn" not in csp


def test_only_the_api_documentation_may_load_swagger_from_a_cdn(client):
    docs = client.get("/docs")

    assert docs.status_code == 200
    assert "https://cdn.jsdelivr.net" in docs.headers["content-security-policy"]


def test_no_cookie_and_no_cross_origin_access(client):
    response = client.get("/api/v1/health", headers={"Origin": "https://example.com"})

    assert "set-cookie" not in response.headers
    assert "access-control-allow-origin" not in response.headers


# ------------------------------------------------------------------ uploads: results


def test_infer_validate_finds_the_injected_faults(client):
    reference = _reference()
    schema = _schema(client, reference)

    clean = client.post(
        "/api/v1/validate",
        files={"data": _upload("batch.csv", _csv(reference.sample(500, random_state=2)))},
        data={"schema": schema},
    ).json()
    broken = client.post(
        "/api/v1/validate",
        files={"data": _upload("batch.csv", _csv(_broken(reference)))},
        data={"schema": schema},
    ).json()

    assert clean["status"] == "ok"
    assert broken["status"] == "anomalies"
    assert {anomaly["type"] for anomaly in broken["anomalies"]} == {
        "OUT_OF_RANGE",
        "UNEXPECTED_VALUES",
        "MISSING_FEATURE",
    }


def test_the_api_and_the_command_line_give_the_same_anomalies(client, tmp_path, capsys):
    reference = _reference()
    schema = _schema(client, reference)
    (tmp_path / "schema.json").write_text(schema, encoding="utf-8")
    _broken(reference).to_csv(tmp_path / "batch.csv", index=False)

    api = client.post(
        "/api/v1/validate",
        files={"data": _upload("batch.csv", (tmp_path / "batch.csv").read_bytes())},
        data={"schema": schema},
    ).json()
    code = main(
        ["validate", str(tmp_path / "batch.csv"), "--schema", str(tmp_path / "schema.json"),
         "--format", "json"]
    )  # fmt: skip
    cli = json.loads(capsys.readouterr().out)

    assert code == 1
    assert api["anomalies"] == cli["anomalies"]
    assert api["status"] == cli["status"]


def test_compare_flags_a_shifted_batch_and_not_a_clean_one(client):
    reference = _reference(6_000)
    schema = _schema(client, reference)
    clean = reference.sample(500, random_state=3)
    shifted = clean.copy()
    shifted["x"] += 8.0

    def compare(batch: pd.DataFrame) -> dict:
        return client.post(
            "/api/v1/compare",
            files={
                "reference": _upload("reference.csv", _csv(reference)),
                "data": _upload("batch.csv", _csv(batch)),
            },
            data={"schema": schema, "n_batches": "200"},
        ).json()

    shifted_result = compare(shifted)

    assert compare(clean)["status"] == "ok"
    assert shifted_result["status"] == "anomalies"
    assert [a["feature"] for a in shifted_result["anomalies"]] == ["x"]


# ------------------------------------------------------------------ uploads: data security


def test_no_category_value_appears_in_any_response_by_default(client):
    reference = _reference()
    schema = _schema(client, reference)
    broken = _csv(_broken(reference))
    responses = [
        client.post(
            "/api/v1/infer",
            files={"data": _upload("r.csv", _csv(reference))},
            data={"exclude": "id"},
        ),
        client.post(
            "/api/v1/validate", files={"data": _upload("b.csv", broken)}, data={"schema": schema}
        ),
        client.post(
            "/api/v1/compare",
            files={
                "reference": _upload("r.csv", _csv(reference)),
                "data": _upload("b.csv", broken),
            },
            data={"schema": schema, "n_batches": "100"},
        ),
        client.post(
            "/api/v1/infer",
            files={"data": _upload("r.csv", _csv(reference))},
            data={"exclude": "id", "domains": "none"},
        ),
    ]

    for response in responses:
        assert response.status_code == 200, response.text
        assert SENTINEL not in response.text


def test_plain_domains_cannot_be_requested_for_uploads(client):
    response = client.post(
        "/api/v1/infer",
        files={"data": _upload("r.csv", _csv(_reference()))},
        data={"domains": "plain"},
    )

    assert response.status_code == 422
    assert SENTINEL not in response.text


def test_values_are_shown_only_when_asked_for(client):
    reference = _reference()
    schema = _schema(client, reference)
    broken = _csv(_broken(reference))

    shown = client.post(
        "/api/v1/validate",
        files={"data": _upload("b.csv", broken)},
        data={"schema": schema, "show_values": "true"},
    )

    assert SENTINEL + "-new" in shown.text


def test_requests_write_nothing_to_disk(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    reference = _reference()
    schema = _schema(client, reference)
    client.post(
        "/api/v1/validate",
        files={"data": _upload("b.csv", _csv(_broken(reference)))},
        data={"schema": schema},
    )
    client.post("/api/v1/demo", json={"fault": "unit_change", "severity": 0.05})

    assert list(tmp_path.iterdir()) == []


def test_the_request_log_holds_no_query_string_file_name_or_client(client, caplog):
    caplog.set_level("INFO", logger="mlguard.api")
    client.post(
        "/api/v1/infer?token=secret-query",
        files={"data": _upload("secret-name.csv", _csv(_reference()))},
        data={"exclude": "id"},
    )

    text = caplog.text
    assert "POST /api/v1/infer 200" in text
    assert "secret-query" not in text
    assert "secret-name" not in text
    assert "testclient" not in text
    assert SENTINEL not in text


# ------------------------------------------------------------------ uploads: limits and bad input


@pytest.mark.parametrize(
    ("name", "content", "status", "message"),
    [
        ("data.xlsx", b"a,b\n1,2\n", 415, "only .csv"),
        ("data.csv", b"PK\x03\x04rest-of-a-zip", 415, "not a text CSV"),
        ("data.csv", b"\x1f\x8b\x08\x00gzip", 415, "not a text CSV"),
        ("data.csv", b"a,b\x00\n1,2\n", 415, "not a text CSV"),
        ("data.csv", b"   \n", 422, "empty"),
        ("data.csv", b"a,b\n", 422, "no rows"),
        ("data.csv", "a\né\n".encode("latin-1"), 422, "UTF-8"),
        ("data.csv", b'a,b\n"1,2\n', 422, "could not be read"),
        ("data.csv", b"x\n" + b"1\n" * (MAX_ROWS + 1), 413, "rows"),
        ("data.csv", WIDE, 413, "columns"),
        ("data.csv", b"x\n" + b"123456789\n" * (MAX_FILE_BYTES // 10 + 1), 413, "larger than"),
    ],
    # Short ids: Windows refuses environment variables longer than 32,767 characters, and pytest
    # stores the id of the running test in one.
    ids=["xlsx", "zip", "gzip", "nul-byte", "blank", "header-only", "latin-1", "bad-quote",
         "too-many-rows", "too-many-columns", "too-large"],
)  # fmt: skip
def test_bad_or_oversized_uploads_get_a_short_error(client, name, content, status, message):
    response = client.post("/api/v1/infer", files={"data": (name, content, "text/csv")})

    assert response.status_code == status
    assert message in response.json()["error"]
    assert "Traceback" not in response.text


def test_a_request_body_over_the_limit_is_refused_before_it_is_read(client):
    response = client.post(
        "/api/v1/infer", content=b"x", headers={"content-length": str(50 * 1024 * 1024)}
    )

    assert response.status_code == 413
    assert response.json() == {"error": "request too large"}


@pytest.mark.parametrize(
    ("schema", "message"),
    [("not json", "not a valid"), ("{}", "not a valid"), ("[1, 2]", "not a valid"),
     ("x" * (300 * 1024), "larger than")],
    ids=["not-json", "empty-object", "list", "too-large"],
)  # fmt: skip
def test_a_broken_schema_is_refused(client, schema, message):
    response = client.post(
        "/api/v1/validate",
        files={"data": _upload("b.csv", b"x\n1\n")},
        data={"schema": schema},
    )

    assert response.status_code in (413, 422)
    assert message in response.json()["error"]


def test_missing_fields_are_named(client):
    response = client.post("/api/v1/validate", files={"data": _upload("b.csv", b"x\n1\n")})

    assert response.status_code == 422
    assert response.json() == {"error": "invalid or missing field(s): schema"}


@pytest.mark.parametrize(
    "form",
    [{"detector": "nope"}, {"n_batches": "5000"}, {"n_batches": "10"}, {"target_fpr": "1.5"}],
)
def test_compare_refuses_unknown_detectors_and_unbounded_work(client, form):
    reference = _reference()
    schema = _schema(client, reference)
    response = client.post(
        "/api/v1/compare",
        files={
            "reference": _upload("r.csv", _csv(reference)),
            "data": _upload("b.csv", _csv(reference)),
        },
        data={"schema": schema, **form},
    )

    assert response.status_code == 422


def test_compare_explains_a_batch_too_large_for_the_reference(client):
    reference = _reference(400)
    schema = _schema(client, reference)
    response = client.post(
        "/api/v1/compare",
        files={
            "reference": _upload("r.csv", _csv(reference)),
            "data": _upload("b.csv", _csv(reference)),
        },
        data={"schema": schema},
    )

    assert response.status_code == 422
    assert "batch size" in response.json()["error"]


def test_an_unexpected_error_is_reported_without_details(quiet_client, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(api_module, "run_demo", broken)
    response = quiet_client.post("/api/v1/demo", json={"fault": "unit_change", "severity": 0.1})

    assert response.status_code == 500
    assert response.json() == {"error": "internal error"}
    assert SENTINEL not in response.text
    assert SENTINEL not in caplog.text


# ------------------------------------------------------------------ demo


def test_demo_options_list_every_fault_with_its_unit(client):
    options = client.get("/api/v1/demo/options").json()
    names = {fault["name"] for fault in options["faults"]}

    assert "time_travel" not in names
    assert {"unit_change", "tiny_noise", "duplicate_rows", "dropped_feature"} <= names
    assert options["batch_sizes"] == [500, 1000, 5000]


def test_the_demo_endpoint_answers_with_rule_verdicts(client):
    result = client.post(
        "/api/v1/demo", json={"fault": "unit_change", "severity": 0.2, "seed": 1}
    ).json()

    assert result["faulty"][result["feature"]]["schema"]["alarm"] is True
    assert result["faulty"][result["feature"]]["calibrated:linf"]["alarm"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"fault": "time_travel", "severity": 1},
        {"fault": "unit_change", "severity": 2.0},
        {"fault": "unit_change", "severity": 0.1, "batch_size": 123},
        {"fault": "unit_change", "severity": 0.1, "seed": -1},
        {"fault": "unit_change", "severity": 0.1, "feature": "c_small"},
        {"fault": "duplicate_rows", "severity": 0.1, "feature": "x_count"},
        {"fault": "unit_change"},
    ],
)
def test_the_demo_refuses_invalid_requests(client, body):
    response = client.post("/api/v1/demo", json=body)

    assert response.status_code == 422
    assert "error" in response.json()


# ------------------------------------------------------------------ front end


def test_the_page_loads_nothing_from_elsewhere_and_writes_no_markup_from_data():
    for name in ("index.html", "app.js", "style.css"):
        text = (STATIC / name).read_text(encoding="utf-8")
        for forbidden in (
            "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
            "new Function", "localStorage", "sessionStorage", "indexedDB", "document.cookie",
            "XMLHttpRequest", "WebSocket", "sendBeacon", "@import",
        ):  # fmt: skip
            assert forbidden not in text, (name, forbidden)
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in page
    assert "style=" not in page
    for tag in re.findall(r"<(?:script|link)\b[^>]*>", page):
        assert re.search(r'(?:src|href)="/static/', tag), tag
    for link in re.findall(r"<a\b[^>]*>", page):
        assert 'href="/' in link or 'target="_blank" rel="noopener noreferrer"' in link, link
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "http://" not in script
    assert "https://" not in script


def test_the_page_states_how_uploads_are_handled():
    page = (STATIC / "index.html").read_text(encoding="utf-8")

    assert "processed in memory and not stored" in page
    assert "do not upload personal data" in page


def test_importing_the_package_does_not_import_the_web_code():
    import subprocess
    import sys

    code = "import sys, mlguard, mlguard.cli, mlguard.demo; print('fastapi' in sys.modules)"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONWARNINGS": "error"},
    ).stdout  # fmt: skip
    assert out.strip() == "False"


def test_infer_names_columns_that_do_not_exist(client):
    response = client.post(
        "/api/v1/infer",
        files={"data": _upload("r.csv", _csv(_reference()))},
        data={"exclude": "id, no_such_column"},
    )

    assert response.status_code == 422
    assert "no_such_column" in response.json()["error"]


def test_the_runner_switches_off_the_access_log(monkeypatch):
    from mlguard.api import __main__ as runner

    calls = {}
    monkeypatch.setattr(runner.uvicorn, "run", lambda target, **options: calls.update(options))
    monkeypatch.setenv("PORT", "8123")
    runner.main()

    assert calls["access_log"] is False
    assert calls["proxy_headers"] is False
    assert calls["port"] == 8123
