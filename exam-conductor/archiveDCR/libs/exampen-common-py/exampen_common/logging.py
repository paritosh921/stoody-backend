"""Structured JSON logging with request/correlation ID propagation.

Provides:
- JSON-formatted structured logger
- Request ID middleware for FastAPI
- Correlation ID support
- Environment-driven log level
"""

from __future__ import annotations

import logging
import os
import uuid
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger import jsonlogger
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
_SERVICE_NAME: str = os.getenv("SERVICE_NAME", "exampen")

# ---------------------------------------------------------------------------
# Custom formatter
# ---------------------------------------------------------------------------


class _ExamPenFormatter(jsonlogger.JsonFormatter):
    """Adds service name, request ID, and correlation ID to every record."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["service"] = _SERVICE_NAME
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        rid = request_id_var.get("")
        if rid:
            log_record["request_id"] = rid

        cid = correlation_id_var.get("")
        if cid:
            log_record["correlation_id"] = cid


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_configured = False


def configure_logging(
    level: str = _LOG_LEVEL,
    service_name: str = _SERVICE_NAME,
) -> None:
    """Configure the root logger with JSON output.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _configured, _SERVICE_NAME
    if _configured:
        return
    _SERVICE_NAME = service_name

    handler = logging.StreamHandler()
    formatter = _ExamPenFormatter(
        fmt="%(asctime)s %(level)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the given *name*.

    Automatically calls :func:`configure_logging` on first use.
    """
    configure_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# FastAPI / Starlette middleware
# ---------------------------------------------------------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Injects ``X-Request-Id`` and ``X-Correlation-Id`` into context vars.

    If the incoming request does not carry these headers, new UUIDs are
    generated.  The response always echoes both headers back.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        rid = request.headers.get("X-Request-Id", uuid.uuid4().hex)
        cid = request.headers.get(
            "X-Correlation-Id",
            request.headers.get("X-Request-Id", uuid.uuid4().hex),
        )

        request_id_var.set(rid)
        correlation_id_var.set(cid)

        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        response.headers["X-Correlation-Id"] = cid
        return response
