"""Safe Prometheus export for upload-security runtime configuration."""

from __future__ import annotations

from typing import Any

import config_async as settings

from .policies import DEFAULT_UPLOAD_POLICIES, get_upload_policy
from .routes import UPLOAD_ROUTE_POLICY_MAP

_NUMERIC_POLICY_FIELDS = (
    "max_size_bytes",
    "max_total_size_bytes",
    "max_files",
    "max_pdf_pages",
    "max_image_pixels",
    "max_archive_entries",
    "max_archive_depth",
    "max_archive_uncompressed_bytes",
    "max_rows",
    "max_columns",
    "max_sessions",
    "max_frames_per_session",
    "max_frames_per_batch",
    "max_frame_json_bytes",
    "max_payload_base64_bytes",
    "max_decoded_payload_bytes",
    "max_total_chunks",
    "max_pages",
    "max_strokes_per_page",
)


def build_upload_security_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for policy_id in sorted(DEFAULT_UPLOAD_POLICIES):
        policy = get_upload_policy(policy_id)
        rows.append(
            {
                "metric": "policy_info",
                "labels": {
                    "policy_id": policy.policy_id,
                    "policy_kind": policy.policy_kind,
                    "allowed_extensions": ",".join(policy.allowed_extensions or ()),
                    "allowed_mime_types": ",".join(policy.allowed_mime_types or ()),
                    "allowed_magic_types": ",".join(policy.allowed_magic_types or ()),
                },
                "value": 1.0,
            }
        )
        for field_name in _NUMERIC_POLICY_FIELDS:
            value = getattr(policy, field_name, None)
            if value is not None:
                rows.append(
                    {
                        "metric": "policy_limit",
                        "labels": {"policy_id": policy.policy_id, "field": field_name},
                        "value": float(value),
                    }
                )

    for route in UPLOAD_ROUTE_POLICY_MAP:
        rows.append(
            {
                "metric": "route_policy",
                "labels": {
                    "method": route.method,
                    "path_template": route.path_template,
                    "policy_id": route.policy_id,
                    "owner_note": route.owner_note,
                },
                "value": 1.0,
            }
        )

    runtime_config = {
        "upload_security_enabled": bool(settings.UPLOAD_SECURITY_ENABLED),
        "upload_av_enabled": bool(settings.UPLOAD_AV_ENABLED),
        "upload_scan_required": bool(settings.UPLOAD_SCAN_REQUIRED),
        "upload_av_fail_closed": bool(settings.UPLOAD_AV_FAIL_CLOSED),
        "upload_enable_public_static_mount": bool(settings.UPLOAD_ENABLE_PUBLIC_STATIC_MOUNT),
        "upload_allow_public_local_fallback": bool(settings.UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK),
        "upload_max_request_body_mb": float(settings.UPLOAD_MAX_REQUEST_BODY_MB),
        "upload_scanner_timeout_seconds": float(settings.UPLOAD_SCANNER_TIMEOUT_SECONDS),
        "upload_freshclam_max_age_hours": float(settings.UPLOAD_FRESHCLAM_MAX_AGE_HOURS),
    }
    for field, value in runtime_config.items():
        rows.append(
            {
                "metric": "runtime_config",
                "labels": {"field": field},
                "value": float(value),
            }
        )

    return rows
