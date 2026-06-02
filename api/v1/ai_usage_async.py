"""AI usage analytics and limit override endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager


router = APIRouter()


class UsageLimitPatch(BaseModel):
    daily_token_limit: Optional[int] = Field(None, ge=0)
    monthly_token_limit: Optional[int] = Field(None, ge=0)
    daily_page_limit: Optional[int] = Field(None, ge=0)
    monthly_page_limit: Optional[int] = Field(None, ge=0)
    daily_call_limit: Optional[int] = Field(None, ge=0)
    monthly_call_limit: Optional[int] = Field(None, ge=0)
    max_tokens_per_region: Optional[int] = Field(None, ge=0)
    enabled: Optional[bool] = None


def _is_admin(user: Dict[str, Any]) -> bool:
    return user.get("user_type") in {"admin", "b2c_admin"}


def _user_id(user: Dict[str, Any]) -> str:
    return str(user.get("user_id") or user.get("_id") or "")


async def _collection(db: DatabaseManager, user: Dict[str, Any], name: str):
    if user.get("user_type") == "b2c_admin":
        return await db.get_b2c_collection(name)
    context_db = await db.get_context_db()
    return context_db[name] if context_db is not None else None


def _date_filter(from_value: Optional[datetime], to_value: Optional[datetime]) -> Dict[str, Any]:
    created_filter: Dict[str, Any] = {}
    if from_value:
        created_filter["$gte"] = from_value
    if to_value:
        created_filter["$lte"] = to_value
    return {"created_at": created_filter} if created_filter else {}


async def _events(
    db: DatabaseManager,
    user: Dict[str, Any],
    query: Dict[str, Any],
    *,
    limit: int = 200,
    sort_desc: bool = True,
) -> List[Dict[str, Any]]:
    collection = await _collection(db, user, "ai_usage_events")
    if collection is None:
        return []
    cursor = collection.find(query, {"_id": 0, "input_units": 1, "event_id": 1, "user_id": 1, "tenant_id": 1, "document_id": 1, "region_id": 1, "region_scope": 1, "stage": 1, "provider": 1, "model": 1, "estimated_total_tokens": 1, "estimated_page_units": 1, "estimated_call_units": 1, "actual_input_tokens": 1, "actual_output_tokens": 1, "usage_source": 1, "status": 1, "error": 1, "latency_ms": 1, "created_at": 1})
    cursor = cursor.sort("created_at", -1 if sort_desc else 1).limit(limit)
    return await cursor.to_list(length=limit)


@router.get("/me")
async def get_my_ai_usage(
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_id = _user_id(current_user)
    from_value = from_ or datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    to_value = to or (from_value + timedelta(days=1) - timedelta(microseconds=1))
    return await _summary(db, current_user, {"user_id": user_id, **_date_filter(from_value, to_value)}, from_value, to_value)


@router.get("/users/{user_id}")
async def get_user_ai_usage(
    user_id: str,
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    if not _is_admin(current_user) and user_id != _user_id(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this user's AI usage")
    return await _summary(db, current_user, {"user_id": user_id, **_date_filter(from_, to)}, from_, to)


@router.get("/documents/{document_id}")
async def get_document_ai_usage(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    query: Dict[str, Any] = {"document_id": document_id}
    if not _is_admin(current_user):
        query["user_id"] = _user_id(current_user)
    return {"document_id": document_id, "events": await _events(db, current_user, query, limit=500)}


@router.get("/events")
async def list_ai_usage_events(
    document_id: Optional[str] = None,
    region_id: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    query: Dict[str, Any] = {}
    if document_id:
        query["document_id"] = document_id
    if region_id:
        query["region_id"] = region_id
    if stage:
        query["stage"] = stage
    if not _is_admin(current_user):
        query["user_id"] = _user_id(current_user)
    return {"events": await _events(db, current_user, query, limit=limit)}


@router.get("/summary")
async def get_ai_usage_summary(
    from_: Optional[datetime] = Query(None, alias="from"),
    to: Optional[datetime] = None,
    group_by: str = "user",
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    group_field = {
        "user": "$user_id",
        "model": "$model",
        "stage": "$stage",
        "provider": "$provider",
    }.get(group_by, "$user_id")
    collection = await _collection(db, current_user, "ai_usage_events")
    if collection is None:
        return {"group_by": group_by, "items": []}
    match = _date_filter(from_, to)
    pipeline = [
        {"$match": match} if match else {"$match": {}},
        {
            "$group": {
                "_id": group_field,
                "calls": {"$sum": 1},
                "estimated_tokens": {"$sum": {"$ifNull": ["$estimated_total_tokens", 0]}},
                "estimated_page_units": {"$sum": {"$ifNull": ["$estimated_page_units", 0]}},
                "estimated_call_units": {"$sum": {"$ifNull": ["$estimated_call_units", 1]}},
                "actual_input_tokens": {"$sum": {"$ifNull": ["$actual_input_tokens", 0]}},
                "actual_output_tokens": {"$sum": {"$ifNull": ["$actual_output_tokens", 0]}},
            }
        },
        {"$sort": {"estimated_tokens": -1}},
    ]
    return {"group_by": group_by, "items": await collection.aggregate(pipeline).to_list(length=500)}


@router.patch("/users/{user_id}/limits")
async def patch_user_limits(
    user_id: str,
    payload: UsageLimitPatch,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return await _patch_limits(db, current_user, "user", user_id, payload)


@router.patch("/tenants/{tenant_id}/limits")
async def patch_tenant_limits(
    tenant_id: str,
    payload: UsageLimitPatch,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return await _patch_limits(db, current_user, "tenant", tenant_id, payload)


async def _patch_limits(
    db: DatabaseManager,
    user: Dict[str, Any],
    scope: str,
    subject_id: str,
    payload: UsageLimitPatch,
) -> Dict[str, Any]:
    collection = await _collection(db, user, "ai_usage_limits")
    if collection is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable")
    updates = payload.dict(exclude_unset=True)
    updates.update({"scope": scope, "subject_id": subject_id, "updated_at": datetime.utcnow()})
    await collection.update_one(
        {"scope": scope, "subject_id": subject_id},
        {"$set": updates, "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"success": True, "scope": scope, "subject_id": subject_id, "limits": updates}


async def _summary(
    db: DatabaseManager,
    user: Dict[str, Any],
    query: Dict[str, Any],
    from_value: Optional[datetime],
    to_value: Optional[datetime],
) -> Dict[str, Any]:
    events = await _events(db, user, query, limit=1000, sort_desc=False)
    tokens_used = 0
    page_units_used = 0
    call_units_used = 0
    calls: Dict[str, int] = {}
    models: Dict[str, Dict[str, int]] = {}
    for event in events:
        token_count = (
            int(event.get("actual_input_tokens") or 0)
            + int(event.get("actual_output_tokens") or 0)
            or int(event.get("estimated_total_tokens") or 0)
        )
        tokens_used += token_count
        page_units = int(event.get("estimated_page_units") or (event.get("input_units") or {}).get("page_count") or 0)
        call_units = int(event.get("estimated_call_units") or 1)
        page_units_used += page_units
        call_units_used += call_units
        stage = event.get("stage") or "unknown"
        model = event.get("model") or "unknown"
        calls[stage] = calls.get(stage, 0) + 1
        models.setdefault(model, {"calls": 0, "tokens": 0, "page_units": 0})
        models[model]["calls"] += 1
        models[model]["tokens"] += token_count
        models[model]["page_units"] += page_units
    return {
        "from": from_value,
        "to": to_value,
        "tokens_used": tokens_used,
        "page_units_used": page_units_used,
        "call_units_used": call_units_used,
        "calls": calls,
        "models": models,
        "events": events,
    }
