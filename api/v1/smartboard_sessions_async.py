"""
Smartboard board-session API.

This stores the teacher's written smartboard pages so they can be opened later
from the teacher web portal. Pairing sessions remain Redis-backed in
smartboard_pair_async.py; live classroom sessions remain in smartboard_async.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from pymongo import ReturnDocument

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/smartboard-sessions", tags=["Smartboard Sessions"])

COLLECTION_NAME = "smartboard_board_sessions"


class CreateSessionRequest(BaseModel):
    name: str
    sessionData: Optional[Dict[str, Any]] = None
    session_data: Optional[Any] = None
    thumbnail: Optional[str] = None
    pairSessionId: Optional[str] = None
    pair_session_id: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None
    sessionData: Optional[Dict[str, Any]] = None
    session_data: Optional[Any] = None
    thumbnail: Optional[str] = None
    pairSessionId: Optional[str] = None
    pair_session_id: Optional[str] = None


class SessionSummary(BaseModel):
    id: str
    name: str
    thumbnail: Optional[str] = None
    pageCount: int
    pairSessionId: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime


class SessionDetail(BaseModel):
    id: str
    name: str
    thumbnail: Optional[str] = None
    sessionData: Dict[str, Any] = Field(default_factory=dict)
    pairSessionId: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    total: int


async def get_db(request: Request):
    return request.app.state.db


async def get_tutor_id_from_token(request: Request) -> str:
    from core.auth import AuthManager

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    token = auth_header.split(" ", 1)[1]
    auth_manager: AuthManager = request.app.state.auth
    try:
        payload = auth_manager.decode_access_token(token)
        tutor_id = payload.get("tutor_id") or payload.get("user_id") or payload.get("sub")
        if not tutor_id:
            raise HTTPException(status_code=401, detail="Token does not contain tutor information")
        return str(tutor_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Token decode failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


def _coerce_session_data(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"pages": [], "currentPageIndex": 0}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="sessionData must be valid JSON") from exc
        return _coerce_session_data(parsed)
    if isinstance(value, dict):
        pages = value.get("pages")
        if pages is None:
            value = {**value, "pages": []}
        elif not isinstance(pages, list):
            raise HTTPException(status_code=422, detail="sessionData.pages must be a list")
        if not isinstance(value.get("currentPageIndex", 0), int):
            value = {**value, "currentPageIndex": 0}
        return value
    raise HTTPException(status_code=422, detail="sessionData must be an object")


def _session_data_from_payload(payload: CreateSessionRequest | UpdateSessionRequest) -> Optional[Dict[str, Any]]:
    if payload.sessionData is not None:
        return _coerce_session_data(payload.sessionData)
    if payload.session_data is not None:
        return _coerce_session_data(payload.session_data)
    return None


def _page_count(session_data: Dict[str, Any]) -> int:
    pages = session_data.get("pages")
    return len(pages) if isinstance(pages, list) else 0


def _serialize_summary(doc: dict) -> SessionSummary:
    session_data = _coerce_session_data(doc.get("session_data"))
    return SessionSummary(
        id=str(doc["_id"]),
        name=doc.get("name", "Untitled"),
        thumbnail=doc.get("thumbnail"),
        pageCount=int(doc.get("page_count") or _page_count(session_data)),
        pairSessionId=doc.get("pair_session_id"),
        createdAt=doc.get("created_at", datetime.utcnow()),
        updatedAt=doc.get("updated_at", datetime.utcnow()),
    )


def _serialize_detail(doc: dict) -> SessionDetail:
    session_data = _coerce_session_data(doc.get("session_data"))
    return SessionDetail(
        id=str(doc["_id"]),
        name=doc.get("name", "Untitled"),
        thumbnail=doc.get("thumbnail"),
        sessionData=session_data,
        pairSessionId=doc.get("pair_session_id"),
        createdAt=doc.get("created_at", datetime.utcnow()),
        updatedAt=doc.get("updated_at", datetime.utcnow()),
    )


async def _get_collection(request: Request, db):
    tenant_db = await db.get_tenant_db_from_request(request)
    if tenant_db is None:
        tenant_db = await db.get_master_db()
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return tenant_db[COLLECTION_NAME]


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db),
):
    tutor_id = await get_tutor_id_from_token(request)
    collection = await _get_collection(request, db)
    query = {"tutor_id": tutor_id}

    try:
        total = await collection.count_documents(query)
        cursor = (
            collection.find(query, {"session_data.pages.strokes": 0, "session_data.pages.cells": 0})
            .sort("updated_at", -1)
            .skip(offset)
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return SessionListResponse(sessions=[_serialize_summary(doc) for doc in docs], total=total)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list smartboard board sessions: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list sessions") from exc


@router.post("", response_model=SessionDetail)
async def create_session(
    request: Request,
    payload: CreateSessionRequest,
    db=Depends(get_db),
):
    tutor_id = await get_tutor_id_from_token(request)
    collection = await _get_collection(request, db)
    session_data = _session_data_from_payload(payload)
    if session_data is None:
        raise HTTPException(status_code=422, detail="sessionData is required")

    now = datetime.utcnow()
    doc = {
        "tutor_id": tutor_id,
        "pair_session_id": payload.pairSessionId or payload.pair_session_id,
        "name": payload.name,
        "thumbnail": payload.thumbnail,
        "session_data": session_data,
        "page_count": _page_count(session_data),
        "created_at": now,
        "updated_at": now,
    }

    try:
        result = await collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        logger.info("Created smartboard board session '%s' for tutor %s", payload.name, tutor_id)
        return _serialize_detail(doc)
    except Exception as exc:
        logger.error("Failed to create smartboard board session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create session") from exc


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    request: Request,
    session_id: str,
    db=Depends(get_db),
):
    tutor_id = await get_tutor_id_from_token(request)
    collection = await _get_collection(request, db)

    try:
        obj_id = ObjectId(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid session ID") from exc

    doc = await collection.find_one({"_id": obj_id, "tutor_id": tutor_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_detail(doc)


@router.put("/{session_id}", response_model=SessionDetail)
async def update_session(
    request: Request,
    session_id: str,
    payload: UpdateSessionRequest,
    db=Depends(get_db),
):
    tutor_id = await get_tutor_id_from_token(request)
    collection = await _get_collection(request, db)

    try:
        obj_id = ObjectId(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid session ID") from exc

    update_doc: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    if payload.name is not None:
        update_doc["name"] = payload.name
    if payload.thumbnail is not None:
        update_doc["thumbnail"] = payload.thumbnail
    pair_session_id = payload.pairSessionId or payload.pair_session_id
    if pair_session_id is not None:
        update_doc["pair_session_id"] = pair_session_id

    session_data = _session_data_from_payload(payload)
    if session_data is not None:
        update_doc["session_data"] = session_data
        update_doc["page_count"] = _page_count(session_data)

    try:
        doc = await collection.find_one_and_update(
            {"_id": obj_id, "tutor_id": tutor_id},
            {"$set": update_doc},
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        logger.info("Updated smartboard board session %s for tutor %s", session_id, tutor_id)
        return _serialize_detail(doc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update smartboard board session: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update session") from exc


@router.delete("/{session_id}")
async def delete_session(
    request: Request,
    session_id: str,
    db=Depends(get_db),
):
    tutor_id = await get_tutor_id_from_token(request)
    collection = await _get_collection(request, db)

    try:
        obj_id = ObjectId(session_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid session ID") from exc

    result = await collection.delete_one({"_id": obj_id, "tutor_id": tutor_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info("Deleted smartboard board session %s for tutor %s", session_id, tutor_id)
    return {"success": True, "message": "Session deleted"}
