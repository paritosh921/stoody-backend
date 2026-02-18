"""
Tenant feature helpers.

Feature system v2:
- Canonical feature catalog with categories (core / advanced / max)
- Cumulative tier resolution with optional per-tenant overrides
- Backward compatibility with legacy enabled_features format
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

FEATURE_TIER_CORE = "core"
FEATURE_TIER_ADVANCED = "advanced"
FEATURE_TIER_MAX = "max"
FEATURE_TIER_ORDER: Tuple[str, ...] = (
    FEATURE_TIER_CORE,
    FEATURE_TIER_ADVANCED,
    FEATURE_TIER_MAX,
)

FEATURE_CATEGORY_CORE = "core"
FEATURE_CATEGORY_ADVANCED = "advanced"
FEATURE_CATEGORY_MAX = "max"

STATUS_ACTIVE = "active"
STATUS_DUMMY = "dummy"

FEATURE_CATALOG: List[Dict[str, Any]] = [
    {
        "key": "admin_monitoring",
        "label": "Admin Monitoring",
        "description": "Monitoring views for institution admin and tutor oversight",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["tenant_admin", "tutor"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_ADMIN_MONITORING",
    },
    {
        "key": "admin_student_db",
        "label": "Admin Student DB",
        "description": "Institution admin student management workflows",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["tenant_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_ADMIN_STUDENT_DB",
    },
    {
        "key": "admin_tutor_db",
        "label": "Admin Tutor DB",
        "description": "Institution admin tutor management workflows",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["tenant_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_ADMIN_TUTOR_DB",
    },
    {
        "key": "admin_content_management",
        "label": "Admin Content",
        "description": "Admin content/document management",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["tenant_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_ADMIN_CONTENT",
    },
    {
        "key": "student_learning_mode",
        "label": "Student Learning",
        "description": "Student learning/notes mode",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["student"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_STUDENT_LEARNING",
    },
    {
        "key": "student_exam_mode",
        "label": "Student Exam",
        "description": "Student exam/practice mode",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["student"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_STUDENT_EXAM",
    },
    {
        "key": "tutor_portal_access",
        "label": "Tutor Portal",
        "description": "Tutor login and tutor portal access",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["tenant_admin", "tutor"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_TUTOR_PORTAL",
    },
    {
        "key": "superadmin_inbox",
        "label": "Super Admin Inbox",
        "description": "Tenant admin messaging with super-admin",
        "category": FEATURE_CATEGORY_CORE,
        "audience": ["tenant_admin", "super_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "CORE_SUPERADMIN_INBOX",
    },
    {
        "key": "admin_question_bank",
        "label": "Question Bank",
        "description": "Admin paper builder/question bank",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["tenant_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_ADMIN_QUESTION_BANK",
    },
    {
        "key": "admin_leaderboard",
        "label": "Admin Leaderboard",
        "description": "Leaderboard controls for institution admin",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["tenant_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_ADMIN_LEADERBOARD",
    },
    {
        "key": "tutor_documents",
        "label": "Tutor Documents",
        "description": "Tutor document/content access",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["tutor"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_TUTOR_DOCUMENTS",
    },
    {
        "key": "tutor_leaderboard",
        "label": "Tutor Leaderboard",
        "description": "Tutor leaderboard access",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["tutor"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_TUTOR_LEADERBOARD",
    },
    {
        "key": "tutor_online_class",
        "label": "Tutor Online Class",
        "description": "Live online class/session management",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["tutor", "student"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_TUTOR_ONLINE_CLASS",
    },
    {
        "key": "student_video_lessons",
        "label": "Student Videos",
        "description": "Student video lessons module",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["student"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_STUDENT_VIDEOS",
    },
    {
        "key": "student_leaderboard_view",
        "label": "Student Leaderboard",
        "description": "Student-facing leaderboard visibility",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["student"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_STUDENT_LEADERBOARD",
    },
    {
        "key": "admin_registration_status_tools",
        "label": "Registration Status Tools",
        "description": "Registration status and support flows for tenant admins",
        "category": FEATURE_CATEGORY_ADVANCED,
        "audience": ["tenant_admin"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "ADV_ADMIN_REGISTRATION",
    },
    {
        "key": "student_ai_mentor",
        "label": "AI Mentor",
        "description": "AI mentor/debugger flows for students",
        "category": FEATURE_CATEGORY_MAX,
        "audience": ["student"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "MAX_STUDENT_AI_MENTOR",
    },
    {
        "key": "tutor_analytics",
        "label": "Tutor Analytics",
        "description": "Tutor analytics dashboards",
        "category": FEATURE_CATEGORY_MAX,
        "audience": ["tutor"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "MAX_TUTOR_ANALYTICS",
    },
    {
        "key": "stoody_pen_capture",
        "label": "Stoody Pen Capture",
        "description": "Stoody pen data capture and processing",
        "category": FEATURE_CATEGORY_MAX,
        "audience": ["student", "tutor"],
        "status": STATUS_ACTIVE,
        "default_enabled": True,
        "billing_code": "MAX_STOODY_PEN",
    },
    {
        "key": "smartboard_core_dummy",
        "label": "Smartboard Core (Dummy)",
        "description": "Reserved smartboard entitlement (not enforced yet)",
        "category": FEATURE_CATEGORY_MAX,
        "audience": ["student", "tutor"],
        "status": STATUS_DUMMY,
        "default_enabled": False,
        "billing_code": "MAX_SMARTBOARD_CORE",
    },
    {
        "key": "smartboard_live_session_dummy",
        "label": "Smartboard Live Session (Dummy)",
        "description": "Reserved live-session entitlement (not enforced yet)",
        "category": FEATURE_CATEGORY_MAX,
        "audience": ["student", "tutor"],
        "status": STATUS_DUMMY,
        "default_enabled": False,
        "billing_code": "MAX_SMARTBOARD_LIVE",
    },
    {
        "key": "smartboard_token_dummy",
        "label": "Smartboard Token (Dummy)",
        "description": "Reserved smartboard token entitlement (not enforced yet)",
        "category": FEATURE_CATEGORY_MAX,
        "audience": ["student", "tutor"],
        "status": STATUS_DUMMY,
        "default_enabled": False,
        "billing_code": "MAX_SMARTBOARD_TOKEN",
    },
]

FEATURE_CATALOG_BY_KEY: Dict[str, Dict[str, Any]] = {
    item["key"]: item for item in FEATURE_CATALOG
}

FEATURE_KEYS_BY_CATEGORY: Dict[str, List[str]] = {
    FEATURE_CATEGORY_CORE: [item["key"] for item in FEATURE_CATALOG if item["category"] == FEATURE_CATEGORY_CORE],
    FEATURE_CATEGORY_ADVANCED: [item["key"] for item in FEATURE_CATALOG if item["category"] == FEATURE_CATEGORY_ADVANCED],
    FEATURE_CATEGORY_MAX: [item["key"] for item in FEATURE_CATALOG if item["category"] == FEATURE_CATEGORY_MAX],
}

# Legacy (v1) feature schema retained for backward compatibility.
LEGACY_DEFAULT_TENANT_FEATURES: Dict[str, bool] = {
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

LEGACY_TO_V2_MAP: Dict[str, Tuple[str, ...]] = {
    "smartboard": ("student_learning_mode", "smartboard_core_dummy"),
    "online_class": ("tutor_online_class",),
    "ai_chat": ("student_ai_mentor",),
    "stoody_pen": ("stoody_pen_capture",),
    "exam_mode": ("student_exam_mode",),
    "tutor_panel": ("tutor_portal_access", "admin_tutor_db"),
    "analytics_dashboard": ("tutor_analytics",),
    "document_management": ("admin_content_management", "tutor_documents"),
    "video_lessons": ("student_video_lessons",),
    "question_bank": ("admin_question_bank",),
    "leaderboard": ("admin_leaderboard", "tutor_leaderboard", "student_leaderboard_view"),
    "student_monitoring": ("admin_monitoring", "admin_student_db"),
}

V2_TO_PRIMARY_LEGACY_MAP: Dict[str, str] = {
    "student_learning_mode": "smartboard",
    "tutor_online_class": "online_class",
    "student_ai_mentor": "ai_chat",
    "stoody_pen_capture": "stoody_pen",
    "student_exam_mode": "exam_mode",
    "tutor_portal_access": "tutor_panel",
    "admin_tutor_db": "tutor_panel",
    "tutor_analytics": "analytics_dashboard",
    "admin_content_management": "document_management",
    "tutor_documents": "document_management",
    "student_video_lessons": "video_lessons",
    "admin_question_bank": "question_bank",
    "admin_leaderboard": "leaderboard",
    "tutor_leaderboard": "leaderboard",
    "student_leaderboard_view": "leaderboard",
    "admin_monitoring": "student_monitoring",
    "admin_student_db": "student_monitoring",
}

FEATURE_PATH_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "student_ai_mentor": (
        "/api/v1/chat",
        "/api/v1/debugger",
        "/api/debugger",
    ),
    "student_exam_mode": (
        "/api/v1/mcq",
        "/api/mcq",
    ),
    "student_video_lessons": (
        "/api/v1/video",
    ),
    "tutor_online_class": (
        "/api/v1/classroom",
        "/api/v1/meeting",
        "/api/v1/online-class",
        "/api/v1/sessions",
    ),
    "stoody_pen_capture": (
        "/api/v1/strokes",
        "/api/v1/dashboard",
        "/api/v1/hub",
        "/api/v1/notes",
        "/api/v1/ocr",
        "/api/v1/question-attempts",
    ),
    "smartboard_core_dummy": (
        "/api/v1/smartboard",
        "/api/v1/smartboard-sessions",
    ),
    "smartboard_token_dummy": (
        "/api/v1/smartboard/token",
    ),
    "admin_question_bank": (
        "/api/v1/questions",
        "/api/v1/paper-builder",
    ),
    "admin_monitoring": (
        "/api/v1/admin/monitoring",
        "/api/v1/admin/students",
    ),
    "tutor_portal_access": (
        "/api/v1/tutor",
    ),
}

ROLE_AWARE_PATH_PREFIXES: Dict[str, Dict[str, str]] = {
    "/api/v1/admin/students": {
        "admin": "admin_student_db",
        "tutor": "admin_monitoring",
    },
    "/api/v1/tutor": {
        "admin": "admin_tutor_db",
        "tutor": "tutor_portal_access",
    },
    "/api/v1/video": {
        "admin": "admin_content_management",
        "tutor": "tutor_documents",
        "student": "student_video_lessons",
    },
    "/api/v1/pdf": {
        "admin": "admin_content_management",
        "tutor": "tutor_documents",
        "student": "student_learning_mode",
    },
    "/api/v1/admin/test-attempts": {
        "admin": "admin_leaderboard",
        "tutor": "tutor_leaderboard",
        "student": "student_leaderboard_view",
    },
}


def normalize_tier(raw_tier: Optional[str]) -> str:
    if not raw_tier:
        return FEATURE_TIER_CORE
    tier = str(raw_tier).strip().lower()
    if tier in FEATURE_TIER_ORDER:
        return tier
    return FEATURE_TIER_CORE


def _tier_index(tier: str) -> int:
    try:
        return FEATURE_TIER_ORDER.index(normalize_tier(tier))
    except ValueError:
        return 0


def get_feature_catalog() -> List[Dict[str, Any]]:
    return [dict(item) for item in FEATURE_CATALOG]


def compute_effective_features(
    tier: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    normalized_tier = normalize_tier(tier)
    effective = {key: False for key in FEATURE_CATALOG_BY_KEY.keys()}

    tier_rank = _tier_index(normalized_tier)
    for category in (FEATURE_CATEGORY_CORE, FEATURE_CATEGORY_ADVANCED, FEATURE_CATEGORY_MAX):
        category_rank = _tier_index(category)
        if category_rank <= tier_rank:
            for key in FEATURE_KEYS_BY_CATEGORY[category]:
                default_enabled = bool(FEATURE_CATALOG_BY_KEY[key].get("default_enabled", True))
                effective[key] = default_enabled

    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in effective:
                effective[key] = bool(value)

    return effective


def infer_tier_from_effective(effective: Optional[Dict[str, Any]]) -> str:
    if not isinstance(effective, dict):
        return FEATURE_TIER_CORE
    if any(bool(effective.get(key)) for key in FEATURE_KEYS_BY_CATEGORY[FEATURE_CATEGORY_MAX]):
        return FEATURE_TIER_MAX
    if any(bool(effective.get(key)) for key in FEATURE_KEYS_BY_CATEGORY[FEATURE_CATEGORY_ADVANCED]):
        return FEATURE_TIER_ADVANCED
    return FEATURE_TIER_CORE


def _legacy_to_v2_overrides(raw_legacy: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    overrides: Dict[str, bool] = {}
    if not isinstance(raw_legacy, dict):
        return overrides
    for legacy_key, mapped_keys in LEGACY_TO_V2_MAP.items():
        if legacy_key not in raw_legacy:
            continue
        value = bool(raw_legacy.get(legacy_key))
        for mapped_key in mapped_keys:
            if mapped_key in FEATURE_CATALOG_BY_KEY:
                overrides[mapped_key] = value
    return overrides


def infer_tier_from_legacy(raw_legacy: Optional[Dict[str, Any]]) -> str:
    overrides = _legacy_to_v2_overrides(raw_legacy)
    return infer_tier_from_effective(overrides)


def build_enabled_features_v2(
    raw_v2: Optional[Dict[str, Any]] = None,
    raw_legacy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(raw_v2, dict):
        tier = normalize_tier(raw_v2.get("tier"))
        overrides = {
            key: bool(value)
            for key, value in (raw_v2.get("overrides") or {}).items()
            if key in FEATURE_CATALOG_BY_KEY
        }
        if isinstance(raw_v2.get("effective"), dict):
            effective = {
                key: bool(value)
                for key, value in raw_v2["effective"].items()
                if key in FEATURE_CATALOG_BY_KEY
            }
            if len(effective) < len(FEATURE_CATALOG_BY_KEY):
                resolved = compute_effective_features(tier, overrides)
                resolved.update(effective)
                effective = resolved
        else:
            effective = compute_effective_features(tier, overrides)
        return {
            "version": 2,
            "tier": tier,
            "overrides": overrides,
            "effective": effective,
        }

    legacy_source = raw_legacy if isinstance(raw_legacy, dict) else LEGACY_DEFAULT_TENANT_FEATURES
    inferred_tier = infer_tier_from_legacy(legacy_source)
    inferred_overrides = _legacy_to_v2_overrides(legacy_source)
    return {
        "version": 2,
        "tier": inferred_tier,
        "overrides": inferred_overrides,
        "effective": compute_effective_features(inferred_tier, inferred_overrides),
    }


def merge_tenant_features(
    raw_features: Optional[Dict[str, Any]],
    raw_features_v2: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    return build_enabled_features_v2(raw_features_v2, raw_features).get("effective", {})


def is_feature_enabled(
    raw_features: Optional[Dict[str, Any]],
    key: str,
    raw_features_v2: Optional[Dict[str, Any]] = None,
) -> bool:
    feature_key = key
    if key not in FEATURE_CATALOG_BY_KEY and key in LEGACY_TO_V2_MAP:
        feature_key = LEGACY_TO_V2_MAP[key][0]
    merged = merge_tenant_features(raw_features, raw_features_v2)
    if feature_key in merged:
        return bool(merged.get(feature_key))
    return True


def export_legacy_features(effective_features: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    exported = dict(LEGACY_DEFAULT_TENANT_FEATURES)
    if not isinstance(effective_features, dict):
        return exported
    for v2_key, value in effective_features.items():
        legacy_key = V2_TO_PRIMARY_LEGACY_MAP.get(v2_key)
        if not legacy_key:
            continue
        exported[legacy_key] = bool(value)
    return exported


def required_feature_for_path(path: str, user_type: Optional[str] = None) -> Optional[str]:
    role = (user_type or "").strip().lower()

    for prefix, role_map in ROLE_AWARE_PATH_PREFIXES.items():
        if path.startswith(prefix):
            if role and role in role_map:
                return role_map[role]
            if "admin" in role_map:
                return role_map["admin"]
            return next(iter(role_map.values()), None)

    for feature_key, prefixes in FEATURE_PATH_PREFIXES.items():
        for prefix in prefixes:
            if path.startswith(prefix):
                return feature_key
    return None
