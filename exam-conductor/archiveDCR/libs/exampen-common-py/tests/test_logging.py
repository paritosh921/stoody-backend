"""Tests for exampen_common.logging — structured output, middleware, context vars.

Test IDs: U-COMMON-LOG-01 through U-COMMON-LOG-06
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from exampen_common.logging import (
    RequestIdMiddleware,
    configure_logging,
    correlation_id_var,
    get_logger,
    request_id_var,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_log_output(logger: logging.Logger, msg: str) -> dict:
    """Emit one log record through *logger* and return the parsed JSON."""
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    # Re-use the same formatter the module installs
    from exampen_common.logging import _ExamPenFormatter

    handler.setFormatter(
        _ExamPenFormatter(
            fmt="%(asctime)s %(level)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    try:
        logger.info(msg)
        raw = buf.getvalue().strip()
        return json.loads(raw)
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# U-COMMON-LOG-01: Output is valid JSON
# ---------------------------------------------------------------------------


def test_log_output_is_json():
    log = get_logger("test.json_format")
    record = _capture_log_output(log, "hello world")
    assert isinstance(record, dict)
    assert record["message"] == "hello world"


# ---------------------------------------------------------------------------
# U-COMMON-LOG-02: Service name present
# ---------------------------------------------------------------------------


def test_log_contains_service_name():
    log = get_logger("test.svc_name")
    record = _capture_log_output(log, "check")
    assert "service" in record


# ---------------------------------------------------------------------------
# U-COMMON-LOG-03: Log level present
# ---------------------------------------------------------------------------


def test_log_contains_level():
    log = get_logger("test.level")
    record = _capture_log_output(log, "check")
    assert record["level"] == "INFO"


# ---------------------------------------------------------------------------
# U-COMMON-LOG-04: Request ID propagated
# ---------------------------------------------------------------------------


def test_request_id_in_log():
    token = request_id_var.set("req-123")
    try:
        log = get_logger("test.rid")
        record = _capture_log_output(log, "with rid")
        assert record.get("request_id") == "req-123"
    finally:
        request_id_var.reset(token)


# ---------------------------------------------------------------------------
# U-COMMON-LOG-05: Correlation ID propagated
# ---------------------------------------------------------------------------


def test_correlation_id_in_log():
    token = correlation_id_var.set("corr-456")
    try:
        log = get_logger("test.cid")
        record = _capture_log_output(log, "with cid")
        assert record.get("correlation_id") == "corr-456"
    finally:
        correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# U-COMMON-LOG-06: RequestIdMiddleware sets headers
# ---------------------------------------------------------------------------


def test_request_id_middleware_sets_headers():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def homepage(request):
        return JSONResponse({"rid": request_id_var.get("")})

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(RequestIdMiddleware)

    client = TestClient(app)

    # Without incoming header — middleware generates one
    resp = client.get("/")
    assert "X-Request-Id" in resp.headers
    assert "X-Correlation-Id" in resp.headers
    body = resp.json()
    assert body["rid"] != ""

    # With incoming header — middleware echoes it
    resp2 = client.get("/", headers={"X-Request-Id": "custom-rid"})
    assert resp2.headers["X-Request-Id"] == "custom-rid"
