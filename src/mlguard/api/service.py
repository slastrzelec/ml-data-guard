"""FastAPI application: a stateless check service plus a one-page front end.

Data security (SPEC section 17.2): uploads are read into memory with a size limit, parsed, checked
and dropped at the end of the request. Nothing is written to disk or logged; the request log has
method, path, status and duration only. Responses hold no cell value unless explicitly asked for,
and schemas inferred from uploads store category values hashed or not at all.
"""

from __future__ import annotations

import io
import json
import logging
import time
from importlib.resources import files
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mlguard import __version__
from mlguard.compare import compare_batches
from mlguard.demo import BATCH_SIZES, DEMO_FAULTS, SEVERITY_RANGES, candidate_features, run_demo
from mlguard.detectors import CALIBRATED_DETECTORS
from mlguard.inject import FAULTS
from mlguard.schema import Schema, infer_schema
from mlguard.validate import has_errors, validate_batch

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLUMNS = 200
MAX_SCHEMA_BYTES = 256 * 1024
MAX_REQUEST_BYTES = 3 * MAX_FILE_BYTES + MAX_SCHEMA_BYTES + 64 * 1024
MAX_CALIBRATION_BATCHES = 200
UPLOAD_DOMAINS = ("hashed", "none")

_BINARY_SIGNATURES = (b"PK\x03\x04", b"\x1f\x8b", b"\x89PNG", b"%PDF", b"\x80\x04", b"\x80\x05")
_STATIC = files("mlguard.api") / "static"

log = logging.getLogger("mlguard.api")

CSP_PAGE = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; form-action 'self'; base-uri 'none'; "
    "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
)
# FastAPI's interactive documentation loads Swagger UI from jsDelivr; only those two pages get it.
CSP_DOCS = (
    "default-src 'none'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
    "connect-src 'self'; base-uri 'none'; "
    "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
)


class DemoRequest(BaseModel):
    fault: str
    severity: float
    batch_size: int = 1_000
    seed: int = Field(0, ge=0)
    feature: str | None = None


def _problem(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def read_csv_upload(upload: UploadFile, data: bytes, label: str) -> pd.DataFrame:
    """Parse one uploaded CSV within the limits; every failure is a 4xx with a short message."""
    name = (upload.filename or "").lower()
    if not name.endswith(".csv"):
        raise HTTPException(415, f"{label}: only .csv files are accepted")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{label}: the file is larger than {MAX_FILE_BYTES // 1024} KB")
    if not data.strip():
        raise HTTPException(422, f"{label}: the file is empty")
    if data.startswith(_BINARY_SIGNATURES) or b"\x00" in data[:4096]:
        raise HTTPException(415, f"{label}: this is not a text CSV file")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422, f"{label}: the file is not UTF-8 text") from None
    try:
        header = pd.read_csv(io.StringIO(text), nrows=0)
        if len(header.columns) > MAX_COLUMNS:
            raise HTTPException(413, f"{label}: more than {MAX_COLUMNS} columns")
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1, low_memory=False)
    except HTTPException:
        raise
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raise HTTPException(422, f"{label}: the file could not be read as CSV") from None
    if len(frame) > MAX_ROWS:
        raise HTTPException(413, f"{label}: more than {MAX_ROWS:,} rows")
    if frame.empty:
        raise HTTPException(422, f"{label}: the file has a header but no rows")
    return frame


async def _read(upload: UploadFile, label: str) -> pd.DataFrame:
    data = await upload.read(MAX_FILE_BYTES + 1)
    await upload.close()
    return read_csv_upload(upload, data, label)


def _schema(text: str) -> Schema:
    if len(text.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise HTTPException(413, "schema: larger than 256 KB")
    try:
        return Schema.from_dict(json.loads(text))
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, "schema: not a valid ml-data-guard schema") from None


def _names(text: str | None) -> list[str]:
    return [name.strip() for name in text.split(",") if name.strip()] if text else []


def _report(command: str, anomalies: list, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": command,
        "status": "anomalies" if has_errors(anomalies) else "ok",
        **details,
        "anomalies": [anomaly.to_dict() for anomaly in anomalies],
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="ml-data-guard",
        version=__version__,
        description="Schema inference, batch validation and distribution-shift checks for tabular "
        "ML data, plus a fault-injection demo. Uploads are processed in memory and not stored.",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        started = time.perf_counter()
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_REQUEST_BYTES):
            response = _problem(413, "request too large")
        else:
            response = await call_next(request)
        docs = request.url.path in ("/docs", "/docs/oauth2-redirect")
        response.headers["Content-Security-Policy"] = CSP_DOCS if docs else CSP_PAGE
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        # Method, path (never the query string), status and duration: nothing about the client.
        log.info(
            "%s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, error: HTTPException) -> JSONResponse:
        return _problem(error.status_code, str(error.detail))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = sorted(
            {".".join(str(part) for part in item["loc"][1:]) for item in error.errors()}
        )
        return _problem(422, "invalid or missing field(s): " + ", ".join(fields))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, error: Exception) -> JSONResponse:
        log.error("unexpected %s", type(error).__name__)
        return _problem(500, "internal error")

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/v1/demo/options")
    def demo_options() -> dict[str, Any]:
        """Faults, their severity units and ranges, features and batch sizes the demo accepts."""
        return {
            "batch_sizes": list(BATCH_SIZES),
            "faults": [
                {
                    "name": name,
                    "unit": FAULTS[name].severity_unit,
                    "range": list(SEVERITY_RANGES[FAULTS[name].severity_unit]),
                    "negative_control": FAULTS[name].negative_control,
                    "features": list(candidate_features(name)),
                }
                for name in DEMO_FAULTS
            ],
        }

    @app.post("/api/v1/demo")
    def demo(request: DemoRequest) -> dict[str, Any]:
        """Inject a fault into a synthetic batch and report every rule's verdict."""
        try:
            return run_demo(
                request.fault,
                request.severity,
                batch_size=request.batch_size,
                seed=request.seed,
                feature=request.feature,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.post("/api/v1/infer")
    async def infer(
        data: Annotated[UploadFile, File(description="reference data, CSV")],
        exclude: Annotated[str | None, Form(description="comma-separated columns")] = None,
        categorical: Annotated[
            str | None, Form(description="numeric columns that are codes")
        ] = None,
        domains: Annotated[Literal["hashed", "none"], Form()] = "hashed",
    ) -> dict[str, Any]:
        """Infer a schema. Category values are stored hashed (default) or not at all."""
        frame = await _read(data, "data")
        try:
            schema = infer_schema(
                frame, exclude=_names(exclude), categorical=_names(categorical), domains=domains
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return {"n_rows": len(frame), "schema": schema.to_dict()}

    @app.post("/api/v1/validate")
    async def validate(
        data: Annotated[UploadFile, File(description="batch to check, CSV")],
        schema_text: Annotated[
            str, Form(alias="schema", description="schema JSON from /infer or `mlguard infer`")
        ],
        show_values: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        """Check a batch against a schema; the same result as `mlguard validate --format json`."""
        parsed = _schema(schema_text)
        frame = await _read(data, "data")
        anomalies = validate_batch(frame, parsed, show_values=show_values)
        return _report(
            "validate", anomalies, {"n_rows": len(frame), "schema_version": parsed.version}
        )

    @app.post("/api/v1/compare")
    async def compare(
        reference: Annotated[UploadFile, File(description="reference data, CSV")],
        data: Annotated[UploadFile, File(description="batch to compare, CSV")],
        schema_text: Annotated[str, Form(alias="schema", description="schema JSON")],
        detector: Annotated[str, Form()] = "linf",
        target_fpr: Annotated[float, Form(gt=0, lt=1)] = 0.01,
        n_batches: Annotated[int, Form(ge=100, le=MAX_CALIBRATION_BATCHES)] = 200,
    ) -> dict[str, Any]:
        """Distribution shift of a batch against reference data, with calibrated limits."""
        if detector not in CALIBRATED_DETECTORS["numeric"]:
            raise HTTPException(422, f"unknown detector {detector!r}")
        parsed = _schema(schema_text)
        reference_frame = await _read(reference, "reference")
        frame = await _read(data, "data")
        try:
            anomalies = compare_batches(
                reference_frame,
                frame,
                parsed,
                detector=detector,
                target_fpr=target_fpr,
                n_batches=n_batches,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        details = {"n_reference_rows": len(reference_frame), "n_rows": len(frame)}
        return _report("compare", anomalies, {**details, "detector": detector})

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(str(_STATIC / "index.html"), media_type="text/html")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    return app


app = create_app()
