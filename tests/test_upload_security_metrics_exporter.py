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
