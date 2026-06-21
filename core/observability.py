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

UPLOAD_SECURITY_TOTAL = Counter(
    "skillbot_upload_security_total",
    "Upload security decisions by policy and outcome.",
    ["policy_id", "outcome"],
)

UPLOAD_SECURITY_REJECTIONS_TOTAL = Counter(
    "skillbot_upload_security_rejections_total",
    "Upload security rejections by policy and reason.",
    ["policy_id", "reason"],
)

UPLOAD_SECURITY_SCAN_DURATION_SECONDS = Histogram(
    "skillbot_upload_security_scan_duration_seconds",
    "Upload malware scan duration in seconds.",
    ["policy_id", "status"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

UPLOAD_SECURITY_ALERT_ACTIVE = Gauge(
    "skillbot_upload_security_alert_active",
    "Upload security alert state where 1=active and 0=inactive.",
    ["alert_type"],
)

UPLOAD_STORAGE_BYTES = Gauge(
    "skillbot_upload_storage_bytes",
    "Private upload storage bytes by prefix.",
    ["prefix"],
)

UPLOAD_FRESHCLAM_AGE_SECONDS = Gauge(
    "skillbot_upload_freshclam_age_seconds",
    "Age in seconds of local ClamAV signature metadata.",
)

UPLOAD_POLICY_LIMIT = Gauge(
    "skillbot_upload_policy_limit",
    "Effective upload policy numeric limits where labels identify the policy and field.",
    ["policy_id", "field"],
)

UPLOAD_POLICY_INFO = Gauge(
    "skillbot_upload_policy_info",
    "Effective upload policy metadata; value is always 1.",
    ["policy_id", "policy_kind", "allowed_extensions", "allowed_mime_types", "allowed_magic_types"],
)

UPLOAD_ROUTE_POLICY_INFO = Gauge(
    "skillbot_upload_route_policy_info",
    "Upload route to policy mapping; value is always 1.",
    ["method", "path_template", "policy_id", "owner_note"],
)

UPLOAD_RUNTIME_CONFIG = Gauge(
    "skillbot_upload_runtime_config",
    "Safe effective upload runtime config values where field identifies the setting.",
    ["field"],
)

UPLOAD_DEPLOY_VALIDATION = Gauge(
    "skillbot_upload_deploy_validation",
    "Last upload-security deployment validation state by safe field.",
    ["field"],
)

UPLOAD_DEPLOY_VALIDATION_CHECK = Gauge(
    "skillbot_upload_deploy_validation_check",
    "Last upload-security deployment validation check result; value is always 1.",
    ["check", "status"],
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


def record_upload_security_decision(policy_id: str, outcome: str) -> None:
    UPLOAD_SECURITY_TOTAL.labels(
        policy_id=_safe(policy_id),
        outcome=_safe(outcome),
    ).inc()


def record_upload_security_rejection(policy_id: str, reason: str) -> None:
    UPLOAD_SECURITY_REJECTIONS_TOTAL.labels(
        policy_id=_safe(policy_id),
        reason=_safe(reason),
    ).inc()


def observe_upload_scan_latency(policy_id: str, status: str, duration_seconds: float) -> None:
    UPLOAD_SECURITY_SCAN_DURATION_SECONDS.labels(
        policy_id=_safe(policy_id),
        status=_safe(status),
    ).observe(max(duration_seconds, 0.0))


def set_upload_security_alert(alert_type: str, active: bool = True) -> None:
    UPLOAD_SECURITY_ALERT_ACTIVE.labels(alert_type=_safe(alert_type)).set(1 if active else 0)


def set_upload_storage_usage(prefix: str, bytes_used: int) -> None:
    UPLOAD_STORAGE_BYTES.labels(prefix=_safe(prefix)).set(max(int(bytes_used), 0))


def set_upload_freshclam_age_seconds(age_seconds: float | int | None) -> None:
    if age_seconds is None:
        return
    UPLOAD_FRESHCLAM_AGE_SECONDS.set(max(float(age_seconds), 0.0))


def set_upload_security_config_metric(metric: str, labels: dict[str, str], value: float) -> None:
    safe_labels = {key: _safe(val) for key, val in labels.items()}
    if metric == "policy_limit":
        UPLOAD_POLICY_LIMIT.labels(**safe_labels).set(value)
    elif metric == "policy_info":
        UPLOAD_POLICY_INFO.labels(**safe_labels).set(value)
    elif metric == "route_policy":
        UPLOAD_ROUTE_POLICY_INFO.labels(**safe_labels).set(value)
    elif metric == "runtime_config":
        UPLOAD_RUNTIME_CONFIG.labels(**safe_labels).set(value)
    elif metric == "deploy_validation":
        UPLOAD_DEPLOY_VALIDATION.labels(**safe_labels).set(value)
    elif metric == "deploy_validation_check":
        UPLOAD_DEPLOY_VALIDATION_CHECK.labels(**safe_labels).set(value)


def clear_upload_deploy_validation_check_metrics() -> None:
    UPLOAD_DEPLOY_VALIDATION_CHECK.clear()


def track_websocket_connection(channel: str, delta: int) -> None:
    WEBSOCKET_CONNECTIONS.labels(channel=_safe(channel)).inc(delta)


def set_dependency_health(dependency: str, healthy: bool) -> None:
    DEPENDENCY_HEALTH.labels(dependency=_safe(dependency)).set(1 if healthy else 0)
