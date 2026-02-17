"""
Tenant feature flag helpers.

Centralizes default feature values and path-to-feature gating rules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_TENANT_FEATURES: Dict[str, bool] = {
    "smartboard": True,
    "online_class": False,
    "ai_chat": True,
    "stoody_pen": False,
    "exam_mode": True,
    "tutor_panel": True,
    "analytics_dashboard": True,
    "document_management": True,
    "video_lessons": False,
    "question_bank": True,
    "leaderboard": True,
    "student_monitoring": True,
}

# Prefix-based API gating. If a prefix matches, the feature must be enabled.
FEATURE_PATH_PREFIXES: Dict[str, tuple[str, ...]] = {
    "ai_chat": (
        "/api/v1/chat",
        "/api/v1/debugger",
        "/api/debugger",
    ),
    "exam_mode": (
        "/api/v1/mcq",
        "/api/mcq",
    ),
    "video_lessons": (
        "/api/v1/video",
    ),
    "online_class": (
        "/api/v1/classroom",
        "/api/v1/meeting",
        "/api/v1/online-class",
        "/api/v1/sessions",
    ),
    "smartboard": (
        "/api/v1/smartboard",
        "/api/v1/smartboard-sessions",
        "/api/v1/smartboard/token",
    ),
    "stoody_pen": (
        "/api/v1/strokes",
        "/api/v1/dashboard",
        "/api/v1/hub",
        "/api/v1/notes",
        "/api/v1/ocr",
        "/api/v1/question-attempts",
    ),
    "document_management": (
        "/api/v1/pdf",
    ),
    "question_bank": (
        "/api/v1/questions",
    ),
    "leaderboard": (
        "/api/v1/admin/test-attempts",
    ),
    "student_monitoring": (
        "/api/v1/admin/monitoring",
        "/api/v1/admin/students",
    ),
    "tutor_panel": (
        "/api/v1/tutor",
    ),
}


def merge_tenant_features(raw_features: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    merged = dict(DEFAULT_TENANT_FEATURES)
    if isinstance(raw_features, dict):
        for key, value in raw_features.items():
            if key in merged:
                merged[key] = bool(value)
    return merged


def is_feature_enabled(raw_features: Optional[Dict[str, Any]], key: str) -> bool:
    return bool(merge_tenant_features(raw_features).get(key, True))


def required_feature_for_path(path: str) -> Optional[str]:
    for feature_key, prefixes in FEATURE_PATH_PREFIXES.items():
        for prefix in prefixes:
            if path.startswith(prefix):
                return feature_key
    return None

