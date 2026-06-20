"""Access control helpers for Prometheus metrics endpoints."""

from __future__ import annotations

import hmac
from collections.abc import Mapping


def is_metrics_request_authorized(
    headers: Mapping[str, str],
    *,
    access_token: str,
    debug_mode: bool,
) -> bool:
    expected = access_token.strip()
    if not expected:
        return bool(debug_mode)

    bearer = _bearer_token(headers.get("authorization", ""))
    header_token = headers.get("x-metrics-token", "").strip()

    return hmac.compare_digest(bearer, expected) or hmac.compare_digest(header_token, expected)


def _bearer_token(value: str) -> str:
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return ""
    return token.strip()
