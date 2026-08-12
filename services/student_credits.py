"""Durable, tenant-scoped student contribution credits.

Credits observe successful canvas and student answer-copy writes. They never
mutate those source records. The append-only ledger is balance authority;
jobs and judgments make asynchronous evaluation recoverable and idempotent.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bson import ObjectId
from PIL import Image, ImageDraw, ImageStat
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from utils.s3_storage import download_private_object

logger = logging.getLogger(__name__)

POLICY_COLLECTION = "student_credit_policies"
JOB_COLLECTION = "student_credit_jobs"
JUDGMENT_COLLECTION = "student_credit_judgments"
LEDGER_COLLECTION = "student_credit_ledger"
LOCK_COLLECTION = "student_credit_locks"

SOURCE_STROKE = "stroke_page"
SOURCE_PHOTO = "notebook_photo"
TERMINAL_JOB_STATES = {"completed", "failed"}
RECONCILIATION_MAX_DISPATCHES = 200
POLICY_TRANSITION_LOCK_SECONDS = 60
MISSING_COMPLETION_LOOKUPS = 6
POLICY_LOCK_KEY_PREFIX = "credit-policy-transition"
MISSING_COMPLETION_FAILURE_REASON = "source completion evidence could not be loaded"

DEFAULT_TIERS = [
    {"id": "seed", "name": "Seed", "min_credits": 0, "accent": "#22A06B", "icon": "sprout"},
    {"id": "scribe", "name": "Scribe", "min_credits": 100, "accent": "#2563EB", "icon": "pen-tool"},
    {"id": "pathfinder", "name": "Pathfinder", "min_credits": 500, "accent": "#7C3AED", "icon": "compass"},
    {"id": "beacon", "name": "Beacon", "min_credits": 1500, "accent": "#D97706", "icon": "lamp"},
    {"id": "luminary", "name": "Luminary", "min_credits": 4000, "accent": "#DB2777", "icon": "sparkles"},
]

DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "semantic_judge_enabled": True,
    "stroke_acceptance_threshold": 0.70,
    "image_acceptance_threshold": 0.68,
    "max_randomness_score": 0.35,
    "min_strokes": 4,
    "min_points": 24,
    "min_path_length_mm": 80.0,
    "min_page_coverage": 0.004,
    "min_image_width": 900,
    "min_image_height": 1100,
    "min_written_coverage": 0.002,
    "max_written_coverage": 0.80,
    "min_blur_variance": 55.0,
    "min_ink_density": 0.008,
    "max_ink_density": 0.42,
    "max_skew_angle": 30.0,
    "max_perspective_distortion": 0.25,
    "max_glare_ratio": 0.35,
    "max_overexposure_ratio": 0.70,
    "max_edge_clipping_ratio": 0.20,
    "stroke_mm_per_credit_unit": 250.0,
    "stroke_credits_per_unit": 1,
    "image_credits_per_page": 1,
    "max_stroke_credits_per_page": 5,
    "max_image_credits_per_submission": 10,
    "daily_credit_cap": 100,
    "max_attempts": 5,
    "lease_seconds": 300,
    "tiers": DEFAULT_TIERS,
}

V2_AWARD_POLICY: Dict[str, Any] = {
    "stroke_mm_per_credit_unit": 250.0,
    "stroke_credits_per_unit": 1,
    "image_credits_per_page": 1,
    "max_stroke_credits_per_page": 5,
    "max_image_credits_per_submission": 10,
    "daily_credit_cap": 100,
    "tiers": DEFAULT_TIERS,
}


class CreditPolicyValidationError(ValueError):
    """Raised when a credit-policy payload violates semantic invariants."""


class CreditPolicyConflictError(RuntimeError):
    """Raised when a policy transition cannot proceed safely."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _latest_time(*values: Any) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for raw in values:
        value = _as_utc(raw)
        if value is None:
            continue
        if latest is None or value > latest:
            latest = value
    return latest


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _object_id(value: Any) -> Any:
    try:
        return ObjectId(str(value))
    except Exception:
        return value


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalise_policy(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    if raw:
        policy.update({key: value for key, value in raw.items() if key in DEFAULT_POLICY})
    policy["tiers"] = sorted(
        list(policy.get("tiers") or DEFAULT_TIERS),
        key=lambda item: int(item.get("min_credits") or 0),
    )
    return policy


def _normalise_number(value: Any, *, field: str, cast_float: bool = False) -> float | int:
    try:
        return float(value) if cast_float else int(value)
    except Exception as exc:
        raise CreditPolicyValidationError(f"{field} must be numeric") from exc


def _normalise_tier(tier: Any) -> Dict[str, Any]:
    if not isinstance(tier, dict):
        raise CreditPolicyValidationError("tier entry must be an object")

    tier_id = _clean_text(tier.get("id"))
    name = _clean_text(tier.get("name"))
    icon = _clean_text(tier.get("icon"))
    accent = _clean_text(tier.get("accent"))
    min_credits = _normalise_number(tier.get("min_credits"), field="min_credits")

    if not tier_id or not name:
        raise CreditPolicyValidationError("tier id and name must be non-empty")
    if not icon or not accent:
        raise CreditPolicyValidationError("tier icon and accent are required")
    if min_credits < 0:
        raise CreditPolicyValidationError("tier minimum credits must be non-negative")
    return {"id": tier_id, "name": name, "min_credits": min_credits, "accent": accent, "icon": icon}


def _validate_policy_semantics(policy: Dict[str, Any]) -> None:
    tiers_value = policy.get("tiers")
    if not isinstance(tiers_value, list) or not tiers_value:
        raise CreditPolicyValidationError("tiers must be a non-empty list")

    tiers = [_normalise_tier(tier) for tier in tiers_value]
    ordered = sorted(tiers, key=lambda item: int(item["min_credits"]))
    if ordered[0]["min_credits"] != 0:
        raise CreditPolicyValidationError("first tier minimum credits must be 0")

    ids = [item["id"].casefold() for item in ordered]
    names = [item["name"].casefold() for item in ordered]
    if len(set(ids)) != len(ids):
        raise CreditPolicyValidationError("duplicate tier identifiers are not allowed")
    if len(set(names)) != len(names):
        raise CreditPolicyValidationError("duplicate tier names are not allowed")

    for current, next_item in zip(ordered, ordered[1:]):
        if int(next_item["min_credits"]) <= int(current["min_credits"]):
            raise CreditPolicyValidationError("tier minimum credits must be strictly increasing")

    if float(policy["min_written_coverage"]) > float(policy["max_written_coverage"]):
        raise CreditPolicyValidationError("min_written_coverage must be <= max_written_coverage")
    if float(policy["min_ink_density"]) > float(policy["max_ink_density"]):
        raise CreditPolicyValidationError("min_ink_density must be <= max_ink_density")

    stroke_credits_per_unit = _normalise_number(policy["stroke_credits_per_unit"], field="stroke_credits_per_unit")
    max_stroke_credits_per_page = _normalise_number(
        policy["max_stroke_credits_per_page"], field="max_stroke_credits_per_page"
    )
    image_credits_per_page = _normalise_number(policy["image_credits_per_page"], field="image_credits_per_page")
    max_image_credits_per_submission = _normalise_number(
        policy["max_image_credits_per_submission"], field="max_image_credits_per_submission"
    )
    if stroke_credits_per_unit > max_stroke_credits_per_page:
        raise CreditPolicyValidationError("stroke_credits_per_unit must be <= max_stroke_credits_per_page")
    if image_credits_per_page > max_image_credits_per_submission:
        raise CreditPolicyValidationError("image_credits_per_page must be <= max_image_credits_per_submission")


def _validate_tier_input_order(tiers: Any) -> None:
    if not isinstance(tiers, list) or not tiers:
        raise CreditPolicyValidationError("tiers must be a non-empty list")
    minimums = [int(_normalise_tier(tier)["min_credits"]) for tier in tiers]
    if any(next_value <= current for current, next_value in zip(minimums, minimums[1:])):
        raise CreditPolicyValidationError("tier minimum credits must be supplied in strictly increasing order")


def _is_v2_preset(policy: Dict[str, Any]) -> bool:
    snapshot = _normalise_policy(policy)
    return all(snapshot.get(key) == value for key, value in V2_AWARD_POLICY.items())


def _policy_transition_lock_key(db_name: str) -> str:
    return f"{POLICY_LOCK_KEY_PREFIX}:{db_name}"


def _missing_completion_reason() -> str:
    return MISSING_COMPLETION_FAILURE_REASON


async def _acquire_named_lock(db: Any, lock_key: str, token: str, *, lease_seconds: int) -> bool:
    now = utc_now()
    expires = now + timedelta(seconds=max(1, int(lease_seconds)))
    try:
        locked = await db[LOCK_COLLECTION].find_one_and_update(
            {
                "_id": lock_key,
                "$or": [
                    {"lease_expires_at": {"$lte": now}},
                    {"lease_expires_at": None},
                    {"lease_expires_at": {"$exists": False}},
                    {"lease_token": token},
                ],
            },
            {"$set": {"lease_token": token, "lease_expires_at": expires, "updated_at": now}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return bool(locked and locked.get("lease_token") == token)
    except DuplicateKeyError:
        return False


async def _release_named_lock(db: Any, lock_key: str, token: str) -> None:
    await db[LOCK_COLLECTION].update_one(
        {"_id": lock_key, "lease_token": token},
        {"$set": {"lease_token": None, "lease_expires_at": utc_now(), "updated_at": utc_now()}},
    )


def _policy_transition_lock_token() -> str:
    return f"policy-transition:{uuid.uuid4().hex}"


async def _student_has_positive_daily_credit_over_cap(
    db: Any,
    *,
    cap: int,
) -> bool:
    now = utc_now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = await db[LEDGER_COLLECTION].aggregate(
        [
            {"$match": {"created_at": {"$gte": start}, "delta": {"$gt": 0}}},
            {"$group": {"_id": "$student_record_id", "total": {"$sum": "$delta"}}},
            {"$match": {"total": {"$gt": cap}}},
            {"$limit": 1},
        ]
    ).to_list(length=1)
    return bool(rows)


async def ensure_credit_indexes(db: Any) -> None:
    """Create all credit-domain indexes idempotently in one tenant database."""
    await db[POLICY_COLLECTION].create_index([("_id", ASCENDING)], unique=True)
    await db[JOB_COLLECTION].create_index(
        [("source_type", ASCENDING), ("source_id", ASCENDING), ("source_version", ASCENDING)],
        unique=True,
        name="uniq_credit_source_version",
    )
    await db[JOB_COLLECTION].create_index(
        [("status", ASCENDING), ("next_attempt_at", ASCENDING), ("lease_expires_at", ASCENDING)],
        name="idx_credit_job_dispatch",
    )
    await db[JUDGMENT_COLLECTION].create_index(
        [("source_type", ASCENDING), ("source_id", ASCENDING), ("source_version", ASCENDING)],
        unique=True,
        name="uniq_credit_judgment_source",
    )
    await db[JUDGMENT_COLLECTION].create_index(
        [("student_record_id", ASCENDING), ("decided_at", DESCENDING)],
        name="idx_credit_judgment_student",
    )
    await db[LEDGER_COLLECTION].create_index(
        [("judgment_key", ASCENDING)], unique=True, name="uniq_credit_ledger_judgment"
    )
    await db[LEDGER_COLLECTION].create_index(
        [("student_record_id", ASCENDING), ("created_at", DESCENDING)],
        name="idx_credit_ledger_student",
    )
    await db[LEDGER_COLLECTION].create_index(
        [("group_key", ASCENDING), ("created_at", ASCENDING)],
        name="idx_credit_ledger_group",
    )


async def get_credit_policy(db: Any, *, admin_id: str = "") -> Dict[str, Any]:
    """Return the active policy without creating durable state on read paths."""
    await ensure_credit_indexes(db)
    existing = await db[POLICY_COLLECTION].find_one({"_id": "current"})
    if existing:
        result = _normalise_policy(existing)
        _validate_policy_semantics(result)
        result.update(
            {
                "version": int(existing.get("version") or 1),
                "earning_started_at": existing.get("earning_started_at"),
                "updated_at": existing.get("updated_at"),
            }
        )
        return result

    return _normalise_policy(DEFAULT_POLICY) | {
        "version": 1,
        "earning_started_at": None,
        "updated_at": None,
    }


async def initialize_credit_policy(db: Any, *, admin_id: str = "") -> Dict[str, Any]:
    """Create the tenant policy at an explicit write boundary if it is absent."""
    existing = await db[POLICY_COLLECTION].find_one({"_id": "current"})
    if existing:
        return await get_credit_policy(db, admin_id=admin_id)

    now = utc_now()
    document = {
        "_id": "current",
        **DEFAULT_POLICY,
        "version": 1,
        "admin_id": admin_id,
        "earning_started_at": now,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await db[POLICY_COLLECTION].insert_one(document)
    except DuplicateKeyError:
        return await get_credit_policy(db, admin_id=admin_id)
    return _normalise_policy(document) | {
        "version": 1,
        "earning_started_at": now,
        "updated_at": now,
    }


async def update_credit_policy(
    db: Any,
    changes: Dict[str, Any],
    *,
    admin_id: str,
    db_name: str = "",
) -> Dict[str, Any]:
    lock_key = _policy_transition_lock_key(_clean_text(db_name) or str(admin_id))
    lock_token = _policy_transition_lock_token()
    for _ in range(8):
        if await _acquire_named_lock(db, lock_key, lock_token, lease_seconds=POLICY_TRANSITION_LOCK_SECONDS):
            break
        await asyncio.sleep(0.05)
    else:
        raise CreditPolicyConflictError("policy transition lock is busy")
    try:
        current = await initialize_credit_policy(db, admin_id=admin_id)
        clean = {key: value for key, value in changes.items() if key in DEFAULT_POLICY and value is not None}
        if "tiers" in clean:
            _validate_tier_input_order(clean["tiers"])
        next_policy = _normalise_policy(current | clean)
        _validate_policy_semantics(next_policy)
        now = utc_now()
        update_fields = {key: next_policy[key] for key in DEFAULT_POLICY}
        updated = await db[POLICY_COLLECTION].find_one_and_update(
            {"_id": "current"},
            {
                "$set": {**update_fields, "admin_id": admin_id, "updated_at": now},
                "$inc": {"version": 1},
                "$setOnInsert": {"created_at": now, "earning_started_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    finally:
        await _release_named_lock(db, lock_key, lock_token)
    return _normalise_policy(updated) | {
        "version": int(updated.get("version") or 1),
        "earning_started_at": updated.get("earning_started_at"),
        "updated_at": updated.get("updated_at"),
    }


async def activate_v2_credit_policy(db: Any, *, admin_id: str, db_name: str) -> Tuple[Dict[str, Any], bool]:
    lock_key = _policy_transition_lock_key(db_name)
    lock_token = _policy_transition_lock_token()
    if not await _acquire_named_lock(db, lock_key, lock_token, lease_seconds=POLICY_TRANSITION_LOCK_SECONDS):
        raise CreditPolicyConflictError("policy transition lock is busy")

    try:
        await ensure_credit_indexes(db)
        policy = await initialize_credit_policy(db, admin_id=admin_id)
        if _is_v2_preset(policy):
            return policy, False

        non_terminal = await db[JOB_COLLECTION].count_documents(
            {"status": {"$in": ["pending", "processing", "retry"]}}
        )
        if non_terminal:
            raise CreditPolicyConflictError("cannot activate v2 while credit jobs are not terminal")

        if await _student_has_positive_daily_credit_over_cap(db, cap=int(V2_AWARD_POLICY["daily_credit_cap"])):
            raise CreditPolicyConflictError("cannot activate v2 while a student exceeded daily credit cap today")

        activated_policy = _normalise_policy(policy | V2_AWARD_POLICY)
        _validate_policy_semantics(activated_policy)
        update_payload = {key: activated_policy[key] for key in DEFAULT_POLICY}
        now = utc_now()
        updated = await db[POLICY_COLLECTION].find_one_and_update(
            {"_id": "current"},
            {
                "$set": {**update_payload, "admin_id": admin_id, "updated_at": now},
                "$inc": {"version": 1},
                "$setOnInsert": {"created_at": now, "earning_started_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return _normalise_policy(updated) | {
            "version": int(updated.get("version") or 1),
            "earning_started_at": updated.get("earning_started_at"),
            "updated_at": updated.get("updated_at"),
        }, True
    finally:
        await _release_named_lock(db, lock_key, lock_token)


async def _acquire_missing_completion_lookup_slot(
    db: Any,
    job_id: str,
) -> int:
    row = await db[JOB_COLLECTION].find_one_and_update(
        {"job_id": job_id, "status": {"$in": ["pending", "retry"]}},
        {"$inc": {"missing_completion_lookups": 1}},
        return_document=ReturnDocument.AFTER,
    )
    if not row:
        return 0
    return int(row.get("missing_completion_lookups") or 0)


async def _finalize_missing_completion_failure(
    db: Any,
    *,
    job_id: str,
    missing_completion_lookups: int,
    now: datetime,
) -> None:
    await db[JOB_COLLECTION].update_one(
        {
            "job_id": job_id,
            "status": {"$in": ["pending", "retry"]},
            "missing_completion_lookups": {"$gte": MISSING_COMPLETION_LOOKUPS},
        },
        {
            "$set": {
                "status": "failed",
                "decision": "missing_completion",
                "award_delta": 0,
                "completed_at": now,
                "updated_at": now,
                "missing_completion_lookups": int(missing_completion_lookups),
                "last_error": _missing_completion_reason(),
                "lease_token": None,
                "lease_expires_at": None,
            }
        },
    )


async def resolve_student_record(db: Any, identity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve claims/server source identity to the durable student record."""
    values = {
        _clean_text(identity.get(key))
        for key in ("student_id", "user_id", "id", "sub", "username", "user_name")
        if _clean_text(identity.get(key))
    }
    source_user_id = _clean_text(identity.get("source_user_id"))
    if source_user_id:
        values.add(source_user_id)
    ors: List[Dict[str, Any]] = []
    for value in values:
        ors.extend(
            [
                {"student_id": value},
                {"username": value},
                {"username_lower": value.lower()},
            ]
        )
        if ObjectId.is_valid(value):
            ors.append({"_id": ObjectId(value)})
    if not ors:
        return None
    return await db["students"].find_one({"$or": ors})


async def _resolve_student_by_value(db: Any, identifier: Any) -> Optional[Dict[str, Any]]:
    value = _clean_text(identifier)
    if not value:
        return None
    lowered = value.lower()
    ors: List[Dict[str, Any]] = [
        {"student_id": value},
        {"student_id": lowered},
        {"username": value},
        {"username_lower": lowered},
        {"user_id": value},
    ]
    if ObjectId.is_valid(value):
        ors.append({"_id": ObjectId(value)})
    return await db["students"].find_one({"$or": ors})


def student_identity(student: Dict[str, Any]) -> Dict[str, str]:
    return {
        "student_record_id": str(student.get("_id") or ""),
        "student_id": _clean_text(student.get("student_id")),
        "student_username": _clean_text(student.get("username")),
    }


def stroke_source_descriptor(page: Dict[str, Any]) -> Tuple[str, str, str]:
    user_id = _clean_text(page.get("user_id"))
    copy_id = _clean_text(page.get("copy_id"))
    book_type = _clean_text(page.get("book_type")).upper()
    page_number = int(page.get("page_number") or 0)
    source_id = f"canvas:{user_id}:{copy_id}:{book_type}:{page_number}"
    group_key = f"stroke:{user_id}:{copy_id}:{book_type}:{page_number}"
    return source_id, str(int(page.get("version") or 1)), group_key


def photo_source_descriptor(submission: Dict[str, Any]) -> Tuple[str, str, str]:
    submission_id = _clean_text(submission.get("submission_id"))
    if not submission_id:
        raise ValueError("submission_id is required for photo credit source")
    student_id = _clean_text(submission.get("student_id"))
    source_id = f"photo:{submission_id}"
    group_key = f"photo:{student_id or submission.get('_id') or submission_id}:{submission_id}"
    return source_id, "1", group_key


def _photo_source_ref_from_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "submission_id": _clean_text(submission.get("submission_id")),
        "exam_id": _clean_text(submission.get("exam_id")),
        "student_id": _clean_text(submission.get("student_id")),
        "source": _clean_text(submission.get("source")),
        "submission_channel": _clean_text(submission.get("submission_channel")),
        "page_count": int(submission.get("page_count") or 0),
    }


async def enqueue_credit_job(
    db: Any,
    *,
    db_name: str,
    admin_id: str,
    student: Dict[str, Any],
    source_type: str,
    source_id: str,
    source_version: str,
    group_key: str,
    source_ref: Dict[str, Any],
    source_completed_at: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Idempotently create a durable job after the primary write succeeds."""
    now = utc_now()
    source_completed = _as_utc(source_completed_at)
    lock_key = _policy_transition_lock_key(db_name)
    lock_token = _policy_transition_lock_token()

    for _ in range(10):
        if await _acquire_named_lock(db, lock_key, lock_token, lease_seconds=POLICY_TRANSITION_LOCK_SECONDS):
            break
        await asyncio.sleep(0.05)
    else:
        raise CreditPolicyConflictError("policy transition lock is busy")

    try:
        policy = await initialize_credit_policy(db, admin_id=admin_id)
        cutoff = _as_utc(policy.get("earning_started_at"))
        if not policy.get("enabled", True) or (
            cutoff is not None and source_completed is not None and source_completed < cutoff
        ):
            return None

        document = {
            "job_id": f"credit-{uuid.uuid4().hex}",
            "db_name": db_name,
            "admin_id": admin_id,
            **student_identity(student),
            "source_type": source_type,
            "source_id": source_id,
            "source_version": str(source_version),
            "group_key": group_key,
            "source_ref": dict(source_ref),
            "source_completed_at": source_completed,
            "policy_version": int(policy.get("version") or 1),
            "policy_snapshot": {key: policy.get(key) for key in DEFAULT_POLICY},
            "status": "pending",
            "attempts": 0,
            "next_attempt_at": now,
            "lease_token": None,
            "lease_expires_at": None,
            "created_at": now,
            "updated_at": now,
        }
        source_key = {
            "source_type": source_type,
            "source_id": source_id,
            "source_version": str(source_version),
        }
        await db[JOB_COLLECTION].update_one(source_key, {"$setOnInsert": document}, upsert=True)
        return await db[JOB_COLLECTION].find_one(source_key)
    finally:
        await _release_named_lock(db, lock_key, lock_token)


def _eligible_strokes(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for stroke in page.get("strokes") or []:
        stroke_id = _clean_text(stroke.get("id"))
        if not stroke_id or stroke_id in seen:
            continue
        if stroke.get("processingVersion") != "ble-canonical-v1":
            continue
        if stroke.get("sourceMode") not in {"live", "offlineReplay"}:
            continue
        points = stroke.get("points") or []
        valid_points = [point for point in points if isinstance(point, list) and len(point) >= 6]
        if len(valid_points) < 2:
            continue
        seen.add(stroke_id)
        result.append({**stroke, "points": valid_points})
    return result


def compute_stroke_metrics(page: Dict[str, Any]) -> Dict[str, Any]:
    strokes = _eligible_strokes(page)
    point_count = 0
    path_length = 0.0
    xs: List[float] = []
    ys: List[float] = []
    durations: List[float] = []
    signatures: set[Tuple[int, int, int, int]] = set()
    repeated = 0
    for stroke in strokes:
        points = stroke["points"]
        point_count += len(points)
        first_t = float(points[0][2])
        last_t = float(points[-1][2])
        durations.append(max(0.0, last_t - first_t))
        for index, point in enumerate(points):
            x, y = float(point[0]), float(point[1])
            xs.append(x)
            ys.append(y)
            if index:
                previous = points[index - 1]
                path_length += math.hypot(x - float(previous[0]), y - float(previous[1]))
        signature = (
            round(float(points[0][0])), round(float(points[0][1])),
            round(float(points[-1][0])), round(float(points[-1][1])),
        )
        if signature in signatures:
            repeated += 1
        signatures.add(signature)
    width = max(xs) - min(xs) if xs else 0.0
    height = max(ys) - min(ys) if ys else 0.0
    coverage = _clamp((width * height) / (210.0 * 297.0))
    repeat_ratio = repeated / max(len(strokes), 1)
    complexity = _clamp(point_count / max(len(strokes) * 12.0, 1.0))
    spread_score = _clamp(coverage / 0.10)
    length_score = _clamp(path_length / 1200.0)
    deterministic_score = _clamp(
        0.30 * complexity + 0.35 * spread_score + 0.35 * length_score - 0.45 * repeat_ratio
    )
    return {
        "stroke_count": len(strokes),
        "point_count": point_count,
        "path_length_mm": round(path_length, 3),
        "duration_ms": round(sum(durations), 3),
        "coverage": round(coverage, 6),
        "repeat_ratio": round(repeat_ratio, 6),
        "deterministic_score": round(deterministic_score, 6),
    }


def render_stroke_page(page: Dict[str, Any], *, width: int = 1024, height: int = 1448) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    scale_x, scale_y = width / 210.0, height / 297.0
    for stroke in _eligible_strokes(page):
        points = [(float(p[0]) * scale_x, float(p[1]) * scale_y) for p in stroke["points"]]
        if len(points) >= 2:
            draw.line(points, fill="#111827", width=3, joint="curve")
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def compute_image_metrics(data: bytes) -> Dict[str, Any]:
    image = Image.open(io.BytesIO(data))
    grayscale = image.convert("L")
    width, height = image.size
    stat = ImageStat.Stat(grayscale)
    mean = float(stat.mean[0])
    contrast = float(stat.stddev[0])
    resized = grayscale.resize((256, 256))
    pixels = list(resized.getdata())
    total_pixels = max(len(pixels), 1)

    written_coverage_ratio = sum(1 for value in pixels if value < 190) / total_pixels
    bright_pixel_ratio = sum(1 for value in pixels if value > 245) / total_pixels
    saturated_pixel_ratio = sum(1 for value in pixels if value > 248) / total_pixels
    # White paper is expected and is not itself glare. Treat bright pixels as
    # glare only when they are anomalous against a darker scene, and treat
    # saturation as overexposure only when useful dark writing has disappeared.
    glare_ratio = bright_pixel_ratio * _clamp((230.0 - mean) / 50.0)
    overexposure_ratio = saturated_pixel_ratio * _clamp(
        (0.02 - written_coverage_ratio) / 0.02
    )
    ink_density = written_coverage_ratio
    edge_clipping_ratio = 0.0
    skew_angle = 0.0
    perspective_distortion = 0.0
    blur_variance = 0.0

    try:
        import cv2
        import numpy as np

        array = np.array(resized)
        blur_variance = float(cv2.Laplacian(array, cv2.CV_64F).var())
        _, binary = cv2.threshold(array, 220, 255, cv2.THRESH_BINARY_INV)

        points = np.column_stack(np.where(binary > 0))
        if points.size >= 20:
            xy = points.astype(np.float32)
            xy[:, [0, 1]] = xy[:, [1, 0]]
            cov = np.cov(xy, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            principal = eigenvectors[:, int(np.argmax(eigenvalues))]
            angle = abs(math.degrees(math.atan2(principal[1], principal[0]))) % 90.0
            skew_angle = min(angle, 90.0 - angle)

        edges = cv2.Canny(array, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest, True)
            quadrilateral = cv2.approxPolyDP(largest, 0.02 * perimeter, True)
            if len(quadrilateral) == 4 and cv2.contourArea(quadrilateral) >= 0.25 * array.size:
                corners = quadrilateral.reshape(4, 2).astype(np.float32)
                sums = corners.sum(axis=1)
                diffs = np.diff(corners, axis=1).reshape(-1)
                top_left = corners[int(np.argmin(sums))]
                bottom_right = corners[int(np.argmax(sums))]
                top_right = corners[int(np.argmin(diffs))]
                bottom_left = corners[int(np.argmax(diffs))]
                top = float(np.linalg.norm(top_right - top_left))
                bottom = float(np.linalg.norm(bottom_right - bottom_left))
                left = float(np.linalg.norm(bottom_left - top_left))
                right = float(np.linalg.norm(bottom_right - top_right))
                width_imbalance = abs(top - bottom) / max(top, bottom, 1.0)
                height_imbalance = abs(left - right) / max(left, right, 1.0)
                perspective_distortion = _clamp(max(width_imbalance, height_imbalance))

        border = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        edge_clipping_ratio = float(np.count_nonzero(border)) / max(int(border.size), 1)
    except Exception:
        blur_variance = contrast * contrast

    exposure_score = _clamp(1.0 - abs(mean - 210.0) / 150.0)
    contrast_score = _clamp(contrast / 64.0)
    blur_score = _clamp(blur_variance / 220.0)
    ink_score = _clamp(ink_density / 0.08) * _clamp((0.5 - ink_density) / 0.2)
    dimension_score = _clamp(min(width / 1200.0, height / 1600.0))
    written_score = _clamp(written_coverage_ratio / 0.10)
    skew_score = _clamp(1.0 - (skew_angle / 90.0))
    perspective_score = _clamp(1.0 - perspective_distortion)
    glare_score = _clamp(1.0 - glare_ratio * 2.2)
    overexposure_score = _clamp(1.0 - overexposure_ratio * 1.5)
    edge_score = _clamp(1.0 - edge_clipping_ratio * 4.0)
    deterministic_score = _clamp(
        0.15 * dimension_score + 0.18 * blur_score + 0.12 * exposure_score
        + 0.08 * contrast_score + 0.10 * ink_score + 0.12 * written_score
        + 0.07 * skew_score + 0.06 * perspective_score + 0.04 * glare_score
        + 0.04 * overexposure_score + 0.04 * edge_score
    )
    return {
        "width": width,
        "height": height,
        "blur_variance": round(blur_variance, 3),
        "brightness": round(mean, 3),
        "contrast": round(contrast, 3),
        "written_coverage_ratio": round(written_coverage_ratio, 6),
        "skew_angle": round(skew_angle, 6),
        "perspective_distortion": round(perspective_distortion, 6),
        "glare_ratio": round(glare_ratio, 6),
        "overexposure_ratio": round(overexposure_ratio, 6),
        "bright_pixel_ratio": round(bright_pixel_ratio, 6),
        "saturated_pixel_ratio": round(saturated_pixel_ratio, 6),
        "edge_clipping_ratio": round(edge_clipping_ratio, 6),
        "ink_density": round(ink_density, 6),
        "deterministic_score": round(deterministic_score, 6),
    }


def _hard_gate_reasons(metrics: Dict[str, Any], policy: Dict[str, Any], source_type: str) -> List[str]:
    reasons: List[str] = []
    if source_type == SOURCE_STROKE:
        checks = (
            (metrics["stroke_count"] < int(policy["min_strokes"]), "too_few_strokes"),
            (metrics["point_count"] < int(policy["min_points"]), "too_few_points"),
            (metrics["path_length_mm"] < float(policy["min_path_length_mm"]), "insufficient_writing_length"),
            (metrics["coverage"] < float(policy["min_page_coverage"]), "insufficient_page_coverage"),
            (metrics["repeat_ratio"] > 0.65, "repeated_trace_pattern"),
        )
    else:
        checks = (
            (metrics["width"] < int(policy["min_image_width"]), "image_width_too_low"),
            (metrics["height"] < int(policy["min_image_height"]), "image_height_too_low"),
            (
                metrics.get("written_coverage_ratio", 0.0) < float(policy["min_written_coverage"]),
                "insufficient_written_coverage",
            ),
            (
                metrics.get("written_coverage_ratio", 0.0) > float(policy["max_written_coverage"]),
                "excessive_written_coverage",
            ),
            (metrics["blur_variance"] < float(policy["min_blur_variance"]), "image_blurred"),
            (metrics["ink_density"] < float(policy["min_ink_density"]), "page_blank_or_too_faint"),
            (metrics["ink_density"] > float(policy["max_ink_density"]), "page_overfilled_or_occluded"),
            (
                metrics.get("skew_angle", 0.0) > float(policy["max_skew_angle"]),
                "image_skew_too_high",
            ),
            (
                metrics.get("perspective_distortion", 0.0) > float(policy["max_perspective_distortion"]),
                "perspective_distortion_too_high",
            ),
            (
                metrics.get("glare_ratio", 0.0) > float(policy["max_glare_ratio"]),
                "image_glare_excessive",
            ),
            (
                metrics.get("overexposure_ratio", 0.0) > float(policy["max_overexposure_ratio"]),
                "image_overexposed",
            ),
            (
                metrics.get("edge_clipping_ratio", 0.0) > float(policy["max_edge_clipping_ratio"]),
                "image_edge_clipping",
            ),
        )
    return [reason for failed, reason in checks if failed]


async def _semantic_judgment(db: Any, image_bytes: bytes, source_type: str) -> Dict[str, Any]:
    from llm_gate import LLMGate

    prompt = (
        "Judge only contribution quality, not academic correctness. Determine whether this "
        "is a genuinely written, reasonably legible learning page rather than blank content, "
        "random lines, repeated loops, doodles, scribbles, or an unusable photograph. Return "
        "strict JSON with has_handwriting (boolean), legible (boolean), quality_score (0..1), "
        "randomness_score (0..1), and reason_codes (array of short snake_case strings)."
    )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    responses_input = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}", "detail": "high"},
        ],
    }]
    response_schema = {
        "type": "object",
        "properties": {
            "has_handwriting": {"type": "boolean"},
            "legible": {"type": "boolean"},
            "quality_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "randomness_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "maxLength": 80},
                "maxItems": 8,
            },
        },
        "required": [
            "has_handwriting",
            "legible",
            "quality_score",
            "randomness_score",
            "reason_codes",
        ],
        "additionalProperties": False,
    }
    gate = LLMGate(db)
    response = await gate.call(
        model_id=os.getenv("CREDITS_QUALITY_MODEL", "gpt-4o-mini"),
        prompt="",
        caller_id="credits_quality_judge",
        responses_input=responses_input,
        json_schema=response_schema,
        max_output_tokens=300,
        temperature=0.0,
        metadata={"credits_source_type": source_type, "prompt_version": "credits-quality-v1"},
    )
    if response.completion_status != "completed":
        raise RuntimeError(f"credits judge incomplete: {response.incomplete_reason}")
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("credits judge returned a non-object")
    return {
        "has_handwriting": bool(parsed.get("has_handwriting")),
        "legible": bool(parsed.get("legible")),
        "quality_score": _clamp(float(parsed.get("quality_score", 0.0))),
        "randomness_score": _clamp(float(parsed.get("randomness_score", 1.0))),
        "reason_codes": [str(item)[:80] for item in (parsed.get("reason_codes") or [])[:8]],
    }


async def _judge_stroke_source(db: Any, job: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    source_ref = job.get("source_ref") if isinstance(job.get("source_ref"), dict) else {}
    source_id = _clean_text(job.get("source_id"))
    query = _canvas_source_lookup_query(source_ref, source_id)
    if not query:
        raise RuntimeError("credit source canvas page lookup is unavailable")
    page = await db["canvas_pages"].find_one(query)
    if not page:
        raise RuntimeError("credit source canvas page is unavailable")
    metrics = compute_stroke_metrics(page)
    reasons = _hard_gate_reasons(metrics, policy, SOURCE_STROKE)
    semantic: Dict[str, Any] = {}
    if not reasons and policy.get("semantic_judge_enabled", True):
        semantic = await _semantic_judgment(db, render_stroke_page(page), SOURCE_STROKE)
        if not semantic["has_handwriting"]:
            reasons.append("handwriting_not_detected")
        if not semantic["legible"]:
            reasons.append("writing_not_legible")
        if semantic["randomness_score"] > float(policy["max_randomness_score"]):
            reasons.append("random_or_scribbled_content")
    semantic_score = float(semantic.get("quality_score", metrics["deterministic_score"]))
    score = min(float(metrics["deterministic_score"]), semantic_score)
    if score < float(policy["stroke_acceptance_threshold"]):
        reasons.append("stroke_quality_below_threshold")
    units = math.ceil(float(metrics["path_length_mm"]) / float(policy["stroke_mm_per_credit_unit"]))
    target = min(
        int(policy["max_stroke_credits_per_page"]),
        units * int(policy["stroke_credits_per_unit"]),
    )
    return {
        "decision": "rejected" if reasons else "accepted",
        "quality_score": round(score, 6),
        "target_credits": target if not reasons else 0,
        "reason_codes": sorted(set(reasons)),
        "metrics": metrics,
        "semantic": semantic,
    }


async def _judge_photo_source(db: Any, job: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    submission_id = _clean_text(job.get("source_ref", {}).get("submission_id"))
    pages = await db["evalpen_answer_pages"].find(
        {"submission_id": submission_id, "raw_image_ref": {"$nin": [None, ""]}},
        {"page_number": 1, "raw_image_ref": 1, "content_type": 1},
    ).sort("page_number", 1).to_list(length=50)
    if not pages:
        raise RuntimeError("credit source answer-copy pages are unavailable")
    page_results: List[Dict[str, Any]] = []
    accepted_pages = 0
    for page in pages:
        raw_ref = _clean_text(page.get("raw_image_ref"))
        data = await download_private_object(
            raw_ref,
            allowed_key_prefix="private/exampen/student-answer-copies",
        )
        metrics = compute_image_metrics(data)
        reasons = _hard_gate_reasons(metrics, policy, SOURCE_PHOTO)
        semantic: Dict[str, Any] = {}
        if not reasons and policy.get("semantic_judge_enabled", True):
            image = Image.open(io.BytesIO(data)).convert("RGB")
            output = io.BytesIO()
            image.thumbnail((1400, 1800))
            image.save(output, format="PNG", optimize=True)
            semantic = await _semantic_judgment(db, output.getvalue(), SOURCE_PHOTO)
            if not semantic["has_handwriting"]:
                reasons.append("handwriting_not_detected")
            if not semantic["legible"]:
                reasons.append("writing_not_legible")
            if semantic["randomness_score"] > float(policy["max_randomness_score"]):
                reasons.append("random_or_scribbled_content")
        semantic_score = float(semantic.get("quality_score", metrics["deterministic_score"]))
        score = min(float(metrics["deterministic_score"]), semantic_score)
        if score < float(policy["image_acceptance_threshold"]):
            reasons.append("image_quality_below_threshold")
        accepted = not reasons
        accepted_pages += int(accepted)
        page_results.append({
            "page_number": int(page.get("page_number") or 0),
            "accepted": accepted,
            "quality_score": round(score, 6),
            "reason_codes": sorted(set(reasons)),
            "metrics": metrics,
            "semantic": semantic,
        })
    target = min(
        int(policy["max_image_credits_per_submission"]),
        accepted_pages * int(policy["image_credits_per_page"]),
    )
    all_pages_accepted = accepted_pages == len(page_results) and bool(page_results)
    all_reasons = sorted({reason for page in page_results for reason in page["reason_codes"]})
    if not all_pages_accepted:
        all_reasons = sorted(set(all_reasons) | {"photo_submission_requires_all_pages"})
        target = 0
    return {
        "decision": "accepted" if all_pages_accepted else "rejected",
        "quality_score": round(sum(page["quality_score"] for page in page_results) / len(page_results), 6),
        "target_credits": target,
        "reason_codes": all_reasons if not all_pages_accepted else [],
        "metrics": {"page_count": len(page_results), "accepted_pages": accepted_pages, "pages": page_results},
        "semantic": {},
    }


async def judge_credit_source(db: Any, job: Dict[str, Any]) -> Dict[str, Any]:
    policy = _normalise_policy(job.get("policy_snapshot"))
    if job.get("source_type") == SOURCE_STROKE:
        return await _judge_stroke_source(db, job, policy)
    if job.get("source_type") == SOURCE_PHOTO:
        return await _judge_photo_source(db, job, policy)
    raise ValueError("unsupported credit source type")


async def _acquire_group_lock(db: Any, job: Dict[str, Any], token: str) -> bool:
    now = utc_now()
    expires = now + timedelta(seconds=int(_normalise_policy(job.get("policy_snapshot"))["lease_seconds"]))
    try:
        locked = await db[LOCK_COLLECTION].find_one_and_update(
            {
                "_id": job["group_key"],
                "$or": [
                    {"lease_expires_at": {"$lte": now}},
                    {"lease_expires_at": None},
                    {"lease_expires_at": {"$exists": False}},
                    {"lease_token": token},
                ],
            },
            {"$set": {"lease_token": token, "lease_expires_at": expires, "updated_at": now}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return bool(locked and locked.get("lease_token") == token)
    except DuplicateKeyError:
        return False


async def _release_group_lock(db: Any, group_key: str, token: str) -> None:
    await db[LOCK_COLLECTION].update_one(
        {"_id": group_key, "lease_token": token},
        {"$set": {"lease_token": None, "lease_expires_at": utc_now(), "updated_at": utc_now()}},
    )


def _student_day_lock_key(student_record_id: str, at: datetime) -> str:
    return f"student-day:{student_record_id}:{at.date().isoformat()}"


async def _acquire_student_day_lock(
    db: Any,
    day_lock_key: str,
    token: str,
    *,
    lease_seconds: int,
    now: Optional[datetime] = None,
) -> bool:
    if not day_lock_key or not token:
        return False
    current = _as_utc(now) or utc_now()
    lease = current + timedelta(seconds=max(0, int(lease_seconds)))
    try:
        locked = await db[LOCK_COLLECTION].find_one_and_update(
            {
                "_id": day_lock_key,
                "$or": [
                    {"lease_expires_at": {"$lte": current}},
                    {"lease_expires_at": None},
                    {"lease_expires_at": {"$exists": False}},
                    {"lease_token": token},
                ],
            },
            {"$set": {"lease_token": token, "lease_expires_at": lease, "updated_at": current}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return bool(locked and locked.get("lease_token") == token)
    except DuplicateKeyError:
        return False


async def _release_student_day_lock(db: Any, day_lock_key: str, token: str) -> None:
    await db[LOCK_COLLECTION].update_one(
        {"_id": day_lock_key, "lease_token": token},
        {"$set": {"lease_token": None, "lease_expires_at": utc_now(), "updated_at": utc_now()}},
    )


async def _daily_awarded(db: Any, student_record_id: str, now: datetime) -> int:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = await db[LEDGER_COLLECTION].aggregate(
        [
            {"$match": {"student_record_id": student_record_id, "created_at": {"$gte": start}, "delta": {"$gt": 0}}},
            {"$group": {"_id": None, "total": {"$sum": "$delta"}}},
        ]
    ).to_list(length=1)
    return int(rows[0].get("total") or 0) if rows else 0


async def _commit_judgment_and_ledger(db: Any, job: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    token = _clean_text(job.get("lease_token"))
    if not token or not await _acquire_group_lock(db, job, token):
        raise RuntimeError("credit source group is busy")
    policy = _normalise_policy(job.get("policy_snapshot"))
    now = utc_now()
    group_key = str(job["group_key"])
    student_record_id = str(job["student_record_id"])
    student_day_lock_key = _student_day_lock_key(student_record_id, now)
    student_day_locked = False
    judgment_key = f"{job['source_type']}:{job['source_id']}:{job['source_version']}"
    try:
        student_day_locked = await _acquire_student_day_lock(
            db,
            student_day_lock_key,
            token,
            lease_seconds=int(policy["lease_seconds"]),
            now=now,
        )
        if not student_day_locked:
            raise RuntimeError("student daily credit lock is busy")

        existing = await db[JUDGMENT_COLLECTION].find_one({"judgment_key": judgment_key})
        if not existing:
            judgment = {
                "judgment_key": judgment_key,
                "job_id": job["job_id"],
                "admin_id": job.get("admin_id"),
                "student_record_id": job["student_record_id"],
                "student_id": job.get("student_id"),
                "student_username": job.get("student_username"),
                "source_type": job["source_type"],
                "source_id": job["source_id"],
                "source_version": job["source_version"],
                "group_key": job["group_key"],
                "policy_version": job["policy_version"],
                "decision": result["decision"],
                "quality_score": result["quality_score"],
                "target_credits": int(result["target_credits"]),
                "reason_codes": result.get("reason_codes") or [],
                "metrics": result.get("metrics") or {},
                "semantic": result.get("semantic") or {},
                "award_delta": None,
                "decided_at": now,
                "created_at": now,
            }
            try:
                await db[JUDGMENT_COLLECTION].insert_one(judgment)
                existing = judgment
            except DuplicateKeyError:
                existing = await db[JUDGMENT_COLLECTION].find_one({"judgment_key": judgment_key})

        ledger = await db[LEDGER_COLLECTION].find_one({"judgment_key": judgment_key})
        if not ledger:
            prior_rows = await db[LEDGER_COLLECTION].aggregate(
                [
                    {"$match": {"group_key": job["group_key"], "delta": {"$gt": 0}}},
                    {"$group": {"_id": None, "total": {"$sum": "$delta"}}},
                ]
            ).to_list(length=1)
            prior = int(prior_rows[0].get("total") or 0) if prior_rows else 0
            target = int(existing.get("target_credits") or 0) if existing.get("decision") == "accepted" else 0
            delta = max(0, target - prior)
            awarded_today = await _daily_awarded(db, job["student_record_id"], now)
            delta = min(delta, max(0, int(policy["daily_credit_cap"]) - awarded_today))
            ledger_doc = {
                "entry_id": f"ledger-{uuid.uuid4().hex}",
                "judgment_key": judgment_key,
                "job_id": job["job_id"],
                "admin_id": job.get("admin_id"),
                "student_record_id": job["student_record_id"],
                "student_id": job.get("student_id"),
                "student_username": job.get("student_username"),
                "source_type": job["source_type"],
                "source_id": job["source_id"],
                "source_version": job["source_version"],
                "group_key": job["group_key"],
                "entry_type": "earned_award",
                "delta": delta,
                "created_at": now,
            }
            try:
                await db[LEDGER_COLLECTION].insert_one(ledger_doc)
                ledger = ledger_doc
            except DuplicateKeyError:
                ledger = await db[LEDGER_COLLECTION].find_one({"judgment_key": judgment_key})
        await db[JUDGMENT_COLLECTION].update_one(
            {"judgment_key": judgment_key, "award_delta": None},
            {"$set": {"award_delta": int((ledger or {}).get("delta") or 0), "ledger_committed_at": now}},
        )
        return existing | {"award_delta": int((ledger or {}).get("delta") or 0)}
    finally:
        if student_day_locked:
            await _release_student_day_lock(db, student_day_lock_key, token)
        await _release_group_lock(db, group_key, token)


async def process_credit_job(db: Any, job_id: str, *, execution_token: str) -> Dict[str, Any]:
    now = utc_now()
    candidate = await db[JOB_COLLECTION].find_one({"job_id": job_id})
    if not candidate:
        raise RuntimeError("credit job not found")
    policy = _normalise_policy(candidate.get("policy_snapshot"))
    max_attempts = int(policy["max_attempts"])
    lease_expires = now + timedelta(seconds=int(policy["lease_seconds"]))
    job = await db[JOB_COLLECTION].find_one_and_update(
        {
            "job_id": job_id,
            "status": {"$in": ["pending", "retry", "processing"]},
            "attempts": {"$lt": max_attempts},
            "$or": [
                {"status": {"$in": ["pending", "retry"]}},
                {"lease_expires_at": {"$lte": now}},
                {"lease_token": execution_token},
            ],
        },
        {
            "$set": {
                "status": "processing", "lease_token": execution_token,
                "lease_expires_at": lease_expires, "updated_at": now,
            },
            "$inc": {"attempts": 1},
        },
        return_document=ReturnDocument.AFTER,
    )
    if not job:
        existing = await db[JOB_COLLECTION].find_one({"job_id": job_id})
        return {"job_id": job_id, "status": (existing or {}).get("status", "missing"), "claimed": False}
    try:
        result = await judge_credit_source(db, job)
        judgment = await _commit_judgment_and_ledger(db, job, result)
        completed = await db[JOB_COLLECTION].find_one_and_update(
            {"job_id": job_id, "lease_token": execution_token, "status": "processing"},
            {
                "$set": {
                    "status": "completed", "decision": judgment.get("decision"),
                    "award_delta": int(judgment.get("award_delta") or 0),
                    "completed_at": utc_now(), "updated_at": utc_now(),
                    "lease_token": None, "lease_expires_at": None, "last_error": None,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        if not completed:
            raise RuntimeError("credit job lease changed before completion")
        return {"job_id": job_id, "status": "completed", "decision": judgment.get("decision"), "award_delta": judgment.get("award_delta", 0)}
    except Exception as exc:
        logger.exception("Student credit job failed: %s", job_id)
        attempts = int(job.get("attempts") or 1)
        terminal = attempts >= max_attempts
        delay = min(3600, 30 * (2 ** max(0, attempts - 1)))
        await db[JOB_COLLECTION].update_one(
            {"job_id": job_id, "lease_token": execution_token},
            {
                "$set": {
                    "status": "failed" if terminal else "retry",
                    "last_error": str(exc)[:500],
                    "next_attempt_at": utc_now() + timedelta(seconds=delay),
                    "lease_token": None, "lease_expires_at": None, "updated_at": utc_now(),
                }
            },
        )
        return {"job_id": job_id, "status": "failed" if terminal else "retry", "terminal": terminal, "error": str(exc)}


def _is_canvas_student_source(source_ref: Dict[str, Any], source_id: str) -> bool:
    source_value = _clean_text(source_ref.get("source") or "")
    if not source_value or source_value in {"camera", "touch", "tutor", "teacher"}:
        return False

    pen_mac = _clean_text(source_ref.get("pen_mac") or source_ref.get("penMac"))
    if not pen_mac or pen_mac.lower() == "canvas":
        return False
    return True


def _canvas_source_lookup_query(source_ref: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    query: Dict[str, Any] = {}

    user_id = _clean_text(source_ref.get("user_id"))
    if user_id:
        query["user_id"] = user_id
    book_type = _clean_text(source_ref.get("book_type"))
    if book_type:
        book_type = book_type.upper()
    if book_type:
        query["book_type"] = book_type
    page_number_raw = source_ref.get("page_number")
    if page_number_raw is not None:
        try:
            query["page_number"] = int(page_number_raw)
        except Exception:
            pass

    if not query and source_id.startswith("canvas:"):
        parts = source_id.split(":", 4)
        if len(parts) >= 5:
            query.update(
                {
                "user_id": parts[1],
                "copy_id": parts[2] if parts[2] else {"$exists": False},
                "book_type": str(parts[3]).upper(),
                "page_number": int(parts[4]),
            }
            )
        elif len(parts) >= 4:
            query.update({"user_id": parts[1], "copy_id": {"$exists": False}, "book_type": parts[2]})

    copy_id = _clean_text(source_ref.get("copy_id"))
    if copy_id:
        query["copy_id"] = copy_id
    elif source_id.startswith("canvas:"):
        parsed_copy_id = source_id.split(":", 4)[2] if len(source_id.split(":", 4)) >= 3 else ""
        if parsed_copy_id:
            query["copy_id"] = parsed_copy_id
        else:
            query["copy_id"] = {"$in": [None, ""]}

    return query


async def _lookup_canvas_completion(
    db: Any,
    source_ref: Dict[str, Any],
    source_id: str,
) -> Optional[datetime]:
    query = _canvas_source_lookup_query(source_ref, source_id)
    if not query:
        return None
    page = await db["canvas_pages"].find_one(
        query,
        projection={"last_modified": 1, "created_at": 1, "updated_at": 1},
    )
    if not page:
        return None
    return _latest_time(page.get("source_completed_at"), page.get("last_modified"), page.get("updated_at"), page.get("created_at"))


async def _lookup_photo_completion(
    db: Any,
    source_ref: Dict[str, Any],
    source_id: str,
) -> Optional[datetime]:
    submission_id = _clean_text(source_ref.get("submission_id"))
    if not submission_id and source_id.startswith("photo:"):
        submission_id = _clean_text(source_id.split(":", 1)[1])
    if not submission_id:
        return None
    submission = await db["evalpen_submissions"].find_one(
        {"submission_id": submission_id},
        projection={"submitted_at": 1, "created_at": 1, "updated_at": 1},
    )
    if not submission:
        return None
    return _latest_time(submission.get("submitted_at"), submission.get("updated_at"), submission.get("created_at"))


async def _source_completion_from_job(db: Any, job: Dict[str, Any]) -> Optional[datetime]:
    source_type = str(job.get("source_type") or "")
    source_ref = job.get("source_ref") if isinstance(job.get("source_ref"), dict) else {}
    source_id = _clean_text(job.get("source_id"))
    if source_type == SOURCE_STROKE:
        return await _lookup_canvas_completion(db, source_ref, source_id)
    if source_type == SOURCE_PHOTO:
        return await _lookup_photo_completion(db, source_ref, source_id)
    return None


async def _reconcile_canvas_sources(
    db: Any,
    *,
    db_name: str,
    policy_snapshot: Dict[str, Any],
    limit: int,
) -> int:
    created = 0
    if limit <= 0:
        return 0

    cursor = db["canvas_pages"].aggregate(
        [
            {
                "$match": {
                    "source": {"$exists": True, "$nin": ["", "camera", "touch", "tutor", "teacher"]},
                    "strokes": {
                        "$elemMatch": {
                            "processingVersion": "ble-canonical-v1",
                            "sourceMode": {"$in": ["live", "offlineReplay"]},
                        }
                    },
                    "pen_mac": {
                        "$exists": True,
                        "$type": "string",
                        "$ne": "",
                        "$not": {"$regex": "^canvas$", "$options": "i"},
                    },
                }
            },
            {
                "$addFields": {
                    "_credit_source_id": {
                        "$concat": [
                            "canvas:",
                            {"$toString": "$user_id"},
                            ":",
                            {"$ifNull": [{"$toString": "$copy_id"}, ""]},
                            ":",
                            {"$toUpper": "$book_type"},
                            ":",
                            {"$toString": "$page_number"},
                        ]
                    },
                    "_credit_source_version": {"$toString": {"$ifNull": ["$version", 1]}},
                }
            },
            {
                "$lookup": {
                    "from": JOB_COLLECTION,
                    "let": {
                        "candidate_source_id": "$_credit_source_id",
                        "candidate_source_version": "$_credit_source_version",
                    },
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$source_type", SOURCE_STROKE]},
                                        {"$eq": ["$source_id", "$$candidate_source_id"]},
                                        {"$eq": ["$source_version", "$$candidate_source_version"]},
                                    ]
                                }
                            }
                        },
                        {"$limit": 1},
                    ],
                    "as": "_credit_jobs",
                }
            },
            {"$match": {"_credit_jobs.0": {"$exists": False}}},
            {"$sort": {"last_modified": -1, "_id": 1}},
            {"$limit": limit},
            {
                "$project": {
                    "_id": 0,
                    "user_id": 1,
                    "copy_id": 1,
                    "book_type": 1,
                    "page_number": 1,
                    "version": 1,
                    "source": 1,
                    "pen_mac": 1,
                    "admin_id": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "last_modified": 1,
                }
            },
        ]
    )

    cutoff = policy_snapshot.get("earning_started_at")
    cutoff_dt = _as_utc(cutoff) if isinstance(cutoff, datetime) else None

    async for page in cursor:
        source_id, source_version, group_key = stroke_source_descriptor(page)
        source_ref = {
            "user_id": page.get("user_id"),
            "copy_id": page.get("copy_id"),
            "book_type": page.get("book_type"),
            "page_number": page.get("page_number"),
            "source": page.get("source"),
            "pen_mac": page.get("pen_mac"),
        }
        if not _is_canvas_student_source(source_ref, source_id):
            continue
        student = await _resolve_student_by_value(db, page.get("user_id"))
        if not student:
            continue

        completion = _latest_time(page.get("last_modified"), page.get("updated_at"), page.get("created_at"))
        if cutoff_dt is not None and completion is not None and completion < cutoff_dt:
            continue
        try:
            await enqueue_credit_job(
                db,
                db_name=db_name,
                admin_id=str(page.get("admin_id") or student.get("admin_id") or ""),
                student=student,
                source_type=SOURCE_STROKE,
                source_id=source_id,
                source_version=source_version,
                group_key=group_key,
                source_ref=source_ref,
                source_completed_at=completion,
            )
            created += 1
        except Exception:
            logger.warning("Canvas credit repair enqueue failed for source=%s", source_id)
    return created


async def _reconcile_photo_sources(
    db: Any,
    *,
    db_name: str,
    policy_snapshot: Dict[str, Any],
    limit: int,
) -> int:
    created = 0
    if limit <= 0:
        return 0

    cursor = db["evalpen_submissions"].aggregate(
        [
            {"$match": {"source": "student_web", "submission_id": {"$exists": True, "$ne": None}}},
            {
                "$lookup": {
                    "from": "exampen_student_copy_uploads",
                    "localField": "submission_id",
                    "foreignField": "submission_id",
                    "as": "_student_uploads",
                }
            },
            {
                "$addFields": {
                    "submission_channel": {
                        "$ifNull": [
                            {"$arrayElemAt": ["$_student_uploads.submission_channel", 0]},
                            "student_web",
                        ]
                    },
                    "_credit_source_id": {"$concat": ["photo:", "$submission_id"]},
                }
            },
            {
                "$lookup": {
                    "from": JOB_COLLECTION,
                    "let": {"candidate_source_id": "$_credit_source_id"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$eq": ["$source_type", SOURCE_PHOTO]},
                                        {"$eq": ["$source_id", "$$candidate_source_id"]},
                                        {"$eq": ["$source_version", "1"]},
                                    ]
                                }
                            }
                        },
                        {"$limit": 1},
                    ],
                    "as": "_credit_jobs",
                }
            },
            {"$match": {"_credit_jobs.0": {"$exists": False}}},
            {"$sort": {"submitted_at": -1, "_id": 1}},
            {"$limit": limit},
            {
                "$project": {
                    "_id": 0,
                    "submission_id": 1,
                    "student_id": 1,
                    "admin_id": 1,
                    "exam_id": 1,
                    "source": 1,
                    "submission_channel": 1,
                    "created_at": 1,
                    "submitted_at": 1,
                    "updated_at": 1,
                }
            },
        ]
    )

    cutoff = policy_snapshot.get("earning_started_at")
    cutoff_dt = _as_utc(cutoff) if isinstance(cutoff, datetime) else None

    async for submission in cursor:
        source_id, source_version, group_key = photo_source_descriptor(submission)
        student = await _resolve_student_by_value(db, submission.get("student_id"))
        if not student:
            continue

        completion = _latest_time(
            submission.get("submitted_at"),
            submission.get("updated_at"),
            submission.get("created_at"),
        )
        if cutoff_dt is not None and completion is not None and completion < cutoff_dt:
            continue
        try:
            await enqueue_credit_job(
                db,
                db_name=db_name,
                admin_id=str(submission.get("admin_id") or student.get("admin_id") or ""),
                student=student,
                source_type=SOURCE_PHOTO,
                source_id=source_id,
                source_version=source_version,
                group_key=group_key,
                source_ref=_photo_source_ref_from_submission(submission),
                source_completed_at=completion,
            )
            created += 1
        except Exception:
            logger.warning("Photo credit repair enqueue failed for source=%s", source_id)
    return created


async def reconcile_credit_jobs(db: Any, *, db_name: str, dispatch: bool = True) -> Dict[str, int]:
    """Recover stale leases and dispatch durable due jobs for one tenant."""
    await ensure_credit_indexes(db)
    now = utc_now()
    recovered = await db[JOB_COLLECTION].update_many(
        {"status": "processing", "lease_expires_at": {"$lte": now}},
        {"$set": {"status": "retry", "lease_token": None, "next_attempt_at": now, "updated_at": now}},
    )
    due = await db[JOB_COLLECTION].find(
        {"status": {"$in": ["pending", "retry"]}, "next_attempt_at": {"$lte": now}},
        {"job_id": 1},
    ).sort("next_attempt_at", 1).limit(RECONCILIATION_MAX_DISPATCHES).to_list(length=RECONCILIATION_MAX_DISPATCHES)

    dispatched = 0
    repaired = 0
    if dispatch:
        from celery_app import process_student_credit_job

        policy_snapshot = await get_credit_policy(db, admin_id=db_name)
        if not policy_snapshot.get("enabled", True):
            return {
                "stale_recovered": int(recovered.modified_count),
                "dispatched": 0,
                "repaired": 0,
            }
        policy_cutoff = policy_snapshot.get("earning_started_at")
        cutoff_dt = _as_utc(policy_cutoff) if isinstance(policy_cutoff, datetime) else None

        for row in due:
            job = await db[JOB_COLLECTION].find_one({"job_id": row["job_id"]})
            if not job:
                continue

            completion = _as_utc(job.get("source_completed_at"))
            if completion is None:
                completion = await _source_completion_from_job(db, job)
                if completion is not None:
                    await db[JOB_COLLECTION].update_one({"job_id": row["job_id"]}, {"$set": {"source_completed_at": completion}})

            if completion is None:
                lookup_count = await _acquire_missing_completion_lookup_slot(
                    db,
                    row["job_id"],
                )
                if lookup_count == 0:
                    continue
                if lookup_count >= MISSING_COMPLETION_LOOKUPS:
                    await _finalize_missing_completion_failure(
                        db,
                        job_id=row["job_id"],
                        missing_completion_lookups=lookup_count,
                        now=now,
                    )
                    continue
                await db[JOB_COLLECTION].update_one(
                    {"job_id": row["job_id"]},
                    {"$set": {"missing_completion_lookups": lookup_count, "next_attempt_at": now, "updated_at": now}},
                )
                continue

            if cutoff_dt is not None and completion < cutoff_dt:
                await db[JOB_COLLECTION].update_one(
                    {"job_id": row["job_id"]},
                    {
                        "$set": {
                            "status": "failed",
                            "decision": "ineligible_by_cutoff",
                            "award_delta": 0,
                            "completed_at": now,
                            "updated_at": now,
                            "last_error": "source completion is before policy earning_started_at",
                            "lease_token": None,
                            "lease_expires_at": None,
                        }
                    },
                )
                continue

            process_student_credit_job.delay(db_name, row["job_id"])
            dispatched += 1

        remaining = max(0, RECONCILIATION_MAX_DISPATCHES - dispatched)
        repaired_canvas = await _reconcile_canvas_sources(
            db,
            db_name=db_name,
            policy_snapshot=policy_snapshot,
            limit=min(remaining, 100),
        )
        repaired += repaired_canvas

        remaining = max(0, RECONCILIATION_MAX_DISPATCHES - dispatched - repaired_canvas)
        repaired += await _reconcile_photo_sources(
            db,
            db_name=db_name,
            policy_snapshot=policy_snapshot,
            limit=min(remaining, 100),
        )

    return {
        "stale_recovered": int(recovered.modified_count),
        "dispatched": int(dispatched),
        "repaired": int(repaired),
    }


def tier_for_credits(total: int, tiers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(tiers, key=lambda item: int(item.get("min_credits") or 0))
    current = ordered[0]
    next_tier: Optional[Dict[str, Any]] = None
    for index, tier in enumerate(ordered):
        if total >= int(tier.get("min_credits") or 0):
            current = tier
            next_tier = ordered[index + 1] if index + 1 < len(ordered) else None
    start = int(current.get("min_credits") or 0)
    end = int((next_tier or current).get("min_credits") or start)
    progress = 1.0 if next_tier is None else _clamp((total - start) / max(end - start, 1))
    return {"current": current, "next": next_tier, "progress": round(progress, 4)}


async def get_student_credit_summary(db: Any, student: Dict[str, Any], *, recent_limit: int = 10) -> Dict[str, Any]:
    identity = student_identity(student)
    sid = identity["student_record_id"]
    ledger_rows = await db[LEDGER_COLLECTION].aggregate(
        [{"$match": {"student_record_id": sid}}, {"$group": {"_id": "$source_type", "credits": {"$sum": "$delta"}, "entries": {"$sum": 1}}}]
    ).to_list(length=10)
    total = sum(int(row.get("credits") or 0) for row in ledger_rows)
    judgment_rows = await db[JUDGMENT_COLLECTION].aggregate(
        [{"$match": {"student_record_id": sid}}, {"$group": {"_id": "$decision", "count": {"$sum": 1}}}]
    ).to_list(length=10)
    counts = {str(row.get("_id")): int(row.get("count") or 0) for row in judgment_rows}
    pending = await db[JOB_COLLECTION].count_documents({"student_record_id": sid, "status": {"$in": ["pending", "processing", "retry"]}})
    recent = await db[JUDGMENT_COLLECTION].find(
        {"student_record_id": sid},
        {"_id": 0, "source_type": 1, "decision": 1, "quality_score": 1, "award_delta": 1, "reason_codes": 1, "decided_at": 1},
    ).sort("decided_at", -1).limit(recent_limit).to_list(length=recent_limit)
    policy = await get_credit_policy(db, admin_id=_clean_text(student.get("admin_id")))
    return {
        "total_credits": total,
        "tier": tier_for_credits(total, policy["tiers"]),
        "policy_version": int(policy.get("version") or 1),
        "tiers": list(policy.get("tiers") or []),
        "stats": {"accepted": counts.get("accepted", 0), "rejected": counts.get("rejected", 0), "pending": pending},
        "by_source": {row["_id"]: {"credits": int(row.get("credits") or 0), "entries": int(row.get("entries") or 0)} for row in ledger_rows},
        "recent": recent,
    }


async def get_credit_leaderboard(
    db: Any,
    *,
    allowed_student_record_ids: Optional[Iterable[str]] = None,
    viewer_student_record_id: str = "",
    private_peer_labels: bool = False,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    match: Dict[str, Any] = {"delta": {"$gte": 0}}
    if allowed_student_record_ids is not None:
        allowed = list({str(value) for value in allowed_student_record_ids})
        if not allowed:
            return []
        match["student_record_id"] = {"$in": allowed}
    rows = await db[LEDGER_COLLECTION].aggregate(
        [
            {"$match": match},
            {"$group": {"_id": "$student_record_id", "total_credits": {"$sum": "$delta"}, "first_awarded_at": {"$min": "$created_at"}}},
            {
                "$lookup": {
                    "from": JUDGMENT_COLLECTION,
                    "localField": "_id",
                    "foreignField": "student_record_id",
                    "as": "judgments",
                }
            },
            {
                "$addFields": {
                    "accepted_count": {
                        "$size": {
                            "$filter": {
                                "input": "$judgments",
                                "as": "judgment",
                                "cond": {"$eq": ["$$judgment.decision", "accepted"]},
                            }
                        }
                    }
                }
            },
            {"$sort": {"total_credits": -1, "accepted_count": -1, "first_awarded_at": 1, "_id": 1}},
            {"$project": {"judgments": 0}},
            {"$limit": max(1, min(limit, 100))},
        ]
    ).to_list(length=max(1, min(limit, 100)))
    ids = [_object_id(row["_id"]) for row in rows]
    students = await db["students"].find(
        {"_id": {"$in": ids}},
        {"first_name": 1, "last_name": 1, "name": 1, "student_id": 1, "username": 1},
    ).to_list(length=len(ids)) if ids else []
    by_id = {str(student.get("_id")): student for student in students}
    policy = await get_credit_policy(db)
    result: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sid = str(row["_id"])
        student = by_id.get(sid, {})
        first = _clean_text(student.get("first_name"))
        last = _clean_text(student.get("last_name"))
        full = _clean_text(student.get("name")) or " ".join(value for value in (first, last) if value) or _clean_text(student.get("username")) or "Student"
        own = sid == viewer_student_record_id
        label = full
        if private_peer_labels and not own:
            label = first or full.split(" ")[0]
            if last:
                label += f" {last[0].upper()}."
        total = int(row.get("total_credits") or 0)
        result.append({
            "rank": index, "student_record_id": sid if own or not private_peer_labels else None,
            "student_id": _clean_text(student.get("student_id")) if own or not private_peer_labels else None,
            "display_name": label, "is_self": own, "total_credits": total,
            "tier": tier_for_credits(total, policy["tiers"])["current"],
        })
    return result
