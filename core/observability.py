"""
Prometheus metrics helpers for SkillBot backend observability.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


AUTH_LOGIN_TOTAL = Counter(
    "skillbot_auth_login_total",
    "Total login attempts split by user type and outcome.",
    ["user_type", "outcome"],
)

CHAT_REQUEST_TOTAL = Counter(
    "skillbot_chat_requests_total",
    "Total chat requests split by mode, status, and cache result.",
    ["mode", "status", "cache_hit"],
)

CHAT_REQUEST_DURATION_SECONDS = Histogram(
    "skillbot_chat_request_duration_seconds",
    "Chat request duration in seconds.",
    ["mode", "status"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

OCR_JOB_TOTAL = Counter(
    "skillbot_ocr_jobs_total",
    "Total OCR jobs by kind and status.",
    ["job_type", "status"],
)

OCR_JOB_DURATION_SECONDS = Histogram(
    "skillbot_ocr_job_duration_seconds",
    "OCR job duration in seconds.",
    ["job_type", "status"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)

WEBSOCKET_CONNECTIONS = Gauge(
    "skillbot_websocket_connections",
    "Current websocket connections by channel.",
    ["channel"],
)

DEPENDENCY_HEALTH = Gauge(
    "skillbot_dependency_health",
    "Dependency health status where 1=healthy and 0=unhealthy.",
    ["dependency"],
)


def _safe(value: str | None, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    value = str(value).strip()
    return value or fallback


def record_auth_login(user_type: str, success: bool) -> None:
    AUTH_LOGIN_TOTAL.labels(
        user_type=_safe(user_type),
        outcome="success" if success else "failure",
    ).inc()


def observe_chat_request(mode: str, status: str, cache_hit: bool, duration_seconds: float) -> None:
    normalized_mode = _safe(mode, "general")
    normalized_status = _safe(status)
    cache_label = "true" if cache_hit else "false"

    CHAT_REQUEST_TOTAL.labels(
        mode=normalized_mode,
        status=normalized_status,
        cache_hit=cache_label,
    ).inc()
    CHAT_REQUEST_DURATION_SECONDS.labels(
        mode=normalized_mode,
        status=normalized_status,
    ).observe(max(duration_seconds, 0.0))


def observe_ocr_job(job_type: str, status: str, duration_seconds: float) -> None:
    normalized_job_type = _safe(job_type)
    normalized_status = _safe(status)

    OCR_JOB_TOTAL.labels(
        job_type=normalized_job_type,
        status=normalized_status,
    ).inc()
    OCR_JOB_DURATION_SECONDS.labels(
        job_type=normalized_job_type,
        status=normalized_status,
    ).observe(max(duration_seconds, 0.0))


def track_websocket_connection(channel: str, delta: int) -> None:
    WEBSOCKET_CONNECTIONS.labels(channel=_safe(channel)).inc(delta)


def set_dependency_health(dependency: str, healthy: bool) -> None:
    DEPENDENCY_HEALTH.labels(dependency=_safe(dependency)).set(1 if healthy else 0)

