from core.ai_usage.metrics_exporter import (
    build_exampen_ai_usage_metric_rows_from_aggregates,
    build_general_ai_usage_metric_rows_from_aggregates,
    public_identity_ref,
)
from core import observability


def test_general_ai_usage_rows_keep_user_refs_hashed_and_separate_from_exampen():
    rows = build_general_ai_usage_metric_rows_from_aggregates(
        period="today",
        breakdown=[
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "stage": "stoody_book",
                "status": "success",
                "input_tokens": 30,
                "output_tokens": 70,
                "total_tokens": 100,
                "calls": 2,
            }
        ],
        top_users=[{"user_id": "raw-user-123", "total_tokens": 100}],
        top_tenants=[{"tenant_id": "TENANT-1", "total_tokens": 100}],
    )

    assert {
        "metric": "general_tokens",
        "labels": {
            "period": "today",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "stage": "stoody_book",
            "status": "success",
            "token_type": "total",
        },
        "value": 100.0,
    } in rows
    assert {
        "metric": "general_calls",
        "labels": {
            "period": "today",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "stage": "stoody_book",
            "status": "success",
        },
        "value": 2.0,
    } in rows
    assert any(row["metric"] == "general_top_user_tokens" for row in rows)
    rendered = repr(rows)
    assert "raw-user-123" not in rendered
    assert public_identity_ref("raw-user-123") in rendered


def test_exampen_ai_usage_rows_use_exampen_namespace_and_caller_breakdown():
    rows = build_exampen_ai_usage_metric_rows_from_aggregates(
        period="7d",
        breakdown=[
            {
                "caller": "dcr_ai",
                "model": "gpt-4o",
                "input_tokens": 20,
                "output_tokens": 40,
                "cache_read_tokens": 5,
                "cache_creation_tokens": 7,
                "total_tokens": 60,
                "calls": 3,
            }
        ],
    )

    assert {
        "metric": "exampen_tokens",
        "labels": {
            "period": "7d",
            "caller": "dcr_ai",
            "model": "gpt-4o",
            "token_type": "total",
        },
        "value": 60.0,
    } in rows
    assert {
        "metric": "exampen_calls",
        "labels": {
            "period": "7d",
            "caller": "dcr_ai",
            "model": "gpt-4o",
        },
        "value": 3.0,
    } in rows


def test_ai_usage_gauges_can_be_cleared_between_scrapes():
    observability.set_ai_usage_metric(
        "general_top_user_tokens",
        {"period": "today", "rank": "1", "user_ref": "user_abc"},
        50.0,
    )
    assert _has_sample(
        observability.AI_USAGE_TOP_USER_TOKENS,
        {"period": "today", "rank": "1", "user_ref": "user_abc"},
    )

    observability.clear_ai_usage_metrics()

    assert not _has_sample(
        observability.AI_USAGE_TOP_USER_TOKENS,
        {"period": "today", "rank": "1", "user_ref": "user_abc"},
    )


def _has_sample(metric, labels):
    for family in metric.collect():
        for sample in family.samples:
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                return True
    return False
