import json
import time

import config_async
from core.upload_security.metrics_exporter import build_upload_security_metric_rows


def test_upload_security_metric_rows_include_policy_limits():
    rows = build_upload_security_metric_rows()

    policy_rows = [row for row in rows if row["metric"] == "policy_limit"]
    assert any(
        row["labels"]["policy_id"] == "pdf_document"
        and row["labels"]["field"] == "max_size_bytes"
        and row["value"] > 0
        for row in policy_rows
    )


def test_upload_security_metric_rows_include_route_mappings():
    rows = build_upload_security_metric_rows()

    route_rows = [row for row in rows if row["metric"] == "route_policy"]
    assert any(
        row["labels"]["policy_id"] == "hub_raw_data_batch"
        and row["labels"]["method"] == "POST"
        for row in route_rows
    )


def test_upload_security_metric_rows_do_not_export_secrets():
    rows = build_upload_security_metric_rows()
    rendered = repr(rows).lower()

    forbidden_fragments = [
        "secret",
        "password",
        "token",
        "jwt",
        "mongodb_uri",
        "api_key",
        "openai",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in rendered


def test_upload_security_metric_rows_include_deploy_validation_status(tmp_path, monkeypatch):
    status_file = tmp_path / "deploy-validation.json"
    status_file.write_text(
        json.dumps(
            {
                "generated_at_epoch": time.time(),
                "result": {
                    "ok": True,
                    "passed_checks": ["UPLOAD_SCAN_REQUIRED"],
                    "failed_checks": [],
                    "details": {"UPLOAD_SCAN_REQUIRED": "true"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_async, "UPLOAD_DEPLOY_VALIDATION_STATUS_FILE", status_file)

    rows = build_upload_security_metric_rows()

    assert any(
        row["metric"] == "deploy_validation"
        and row["labels"]["field"] == "ok"
        and row["value"] == 1.0
        for row in rows
    )
    assert any(
        row["metric"] == "deploy_validation_check"
        and row["labels"] == {"check": "UPLOAD_SCAN_REQUIRED", "status": "passed"}
        for row in rows
    )
