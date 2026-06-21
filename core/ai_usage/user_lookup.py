"""Lookup helpers for privacy-safe AI usage user references."""

from __future__ import annotations

from typing import Any, Iterable

from .metrics_exporter import public_identity_ref


EMPTY_SUMMARY = {
    "calls": 0,
    "total_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "models": {},
    "providers": {},
    "stages": {},
}


def build_user_ref_lookup_response(
    user_ref: str,
    *,
    events: Iterable[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matching_events = [
        event for event in events
        if public_identity_ref(event.get("user_id"), prefix="user") == user_ref
    ]
    if not matching_events:
        return {
            "found": False,
            "user_ref": user_ref,
            "matches": [],
            "summary": dict(EMPTY_SUMMARY),
        }

    user_ids = sorted({str(event.get("user_id")) for event in matching_events if event.get("user_id")})
    summary = _summarize_events(matching_events)
    return {
        "found": True,
        "user_ref": user_ref,
        "matches": [
            {
                "user_id": user_id,
                "profile": profiles.get(user_id, {}),
            }
            for user_id in user_ids
        ],
        "summary": summary,
    }


def token_count_for_event(event: dict[str, Any]) -> int:
    input_tokens = _int(event.get("actual_input_tokens"))
    output_tokens = _int(event.get("actual_output_tokens"))
    actual_total = input_tokens + output_tokens
    if actual_total > 0:
        return actual_total
    return _int(event.get("estimated_total_tokens"))


def _summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "calls": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "models": {},
        "providers": {},
        "stages": {},
    }
    for event in events:
        input_tokens = _int(event.get("actual_input_tokens"))
        output_tokens = _int(event.get("actual_output_tokens"))
        total_tokens = token_count_for_event(event)
        summary["calls"] += 1
        summary["total_tokens"] += total_tokens
        summary["input_tokens"] += input_tokens
        summary["output_tokens"] += output_tokens
        _add_breakdown(summary["models"], event.get("model"), total_tokens)
        _add_breakdown(summary["providers"], event.get("provider"), total_tokens)
        _add_breakdown(summary["stages"], event.get("stage"), total_tokens)
    return summary


def _add_breakdown(target: dict[str, dict[str, int]], key: Any, total_tokens: int) -> None:
    label = str(key or "unknown").strip() or "unknown"
    item = target.setdefault(label, {"calls": 0, "total_tokens": 0})
    item["calls"] += 1
    item["total_tokens"] += total_tokens


def _int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0
