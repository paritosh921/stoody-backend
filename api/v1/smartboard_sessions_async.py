"""
Smartboard Sessions API

This module handles saving, loading, and managing smartboard canvas sessions
for tutors. Sessions are stored in MongoDB under the tutor's account.

Collection: smartboard_sessions
Structure:
{
    _id: ObjectId,
    tutor_id: str,
    name: str,
    thumbnail: str (base64, optional),
    pages: [{
        id: str,
        strokes: [...],
        background: str
    }],
    current_page_index: int,
    created_at: datetime,
    updated_at: datetime
}
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/smartboard-sessions", tags=["Smartboard Sessions"])


# ----- Pydantic Models -----

class StrokePoint(BaseModel):
    x: float
    y: float
    pressure: Optional[float] = None


class Stroke(BaseModel):
    id: Optional[str] = None
    points: List[StrokePoint] = Field(default_factory=list)
    color: Optional[str] = "#000000"
    strokeWidth: Optional[float] = 2.0
    timestamp: Optional[float] = None
    tool: Optional[str] = None
    shapeType: Optional[str] = None
    startPoint: Optional[StrokePoint] = None
    endPoint: Optional[StrokePoint] = None


class Page(BaseModel):
    id: str
    strokes: List[Stroke] = Field(default_factory=list)
    background: str = "plain"


class SessionData(BaseModel):
    pages: List[Page] = Field(default_factory=list)
    currentPageIndex: int = 0


class CreateSessionRequest(BaseModel):
    name: str
    sessionData: SessionData
    thumbnail: Optional[str] = None  # Base64 encoded thumbnail


class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None
    sessionData: Optional[SessionData] = None
    thumbnail: Optional[str] = None


class SessionSummary(BaseModel):
    id: str
    name: str
    thumbnail: Optional[str] = None
    pageCount: int
    createdAt: datetime
    updatedAt: datetime


class SessionDetail(BaseModel):
    id: str
    name: str
    thumbnail: Optional[str] = None
    sessionData: SessionData
    createdAt: datetime
    updatedAt: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    total: int


# ----- Helper Functions -----

async def get_db(request: Request):
    """Get database manager from app state."""
    return request.app.state.db


async def get_tutor_id_from_token(request: Request) -> str:
    """
    Extract tutor_id from the main_app_token in Authorization header.
    The smartboard PWA uses main_app_token for this API.
    """
    from core.auth import AuthManager

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")

    token = auth_header.split(" ", 1)[1]

    # Decode the JWT to get tutor_id
    auth_manager: AuthManager = request.app.state.auth
    try:
        payload = auth_manager.decode_access_token(token)
        tutor_id = payload.get("tutor_id") or payload.get("user_id") or payload.get("sub")
        if not tutor_id:
            raise HTTPException(status_code=401, detail="Token does not contain tutor information")
        return tutor_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token decode failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def serialize_session_summary(doc: dict) -> SessionSummary:
    """Convert MongoDB document to SessionSummary."""
    pages = doc.get("pages", [])
    return SessionSummary(
        id=str(doc["_id"]),
        name=doc.get("name", "Untitled"),
        thumbnail=doc.get("thumbnail"),
        pageCount=len(pages),
        createdAt=doc.get("created_at", datetime.utcnow()),
        updatedAt=doc.get("updated_at", datetime.utcnow()),
    )


def serialize_session_detail(doc: dict) -> SessionDetail:
    """Convert MongoDB document to SessionDetail."""
    pages_data = doc.get("pages", [])
    pages = []
    for p in pages_data:
        strokes = []
        for s in p.get("strokes", []):
            points = [StrokePoint(**pt) for pt in s.get("points", [])]
            stroke = Stroke(
                id=s.get("id"),
                points=points,
                color=s.get("color", "#000000"),
                strokeWidth=s.get("strokeWidth", 2.0),
                timestamp=s.get("timestamp"),
                tool=s.get("tool"),
                shapeType=s.get("shapeType"),
                startPoint=StrokePoint(**s["startPoint"]) if s.get("startPoint") else None,
                endPoint=StrokePoint(**s["endPoint"]) if s.get("endPoint") else None,
            )
            strokes.append(stroke)
        pages.append(Page(
            id=p.get("id", str(uuid4())),
            strokes=strokes,
            background=p.get("background", "plain")
        ))

    session_data = SessionData(
        pages=pages,
        currentPageIndex=doc.get("current_page_index", 0)
    )

    return SessionDetail(
        id=str(doc["_id"]),
        name=doc.get("name", "Untitled"),
        thumbnail=doc.get("thumbnail"),
        sessionData=session_data,
        createdAt=doc.get("created_at", datetime.utcnow()),
        updatedAt=doc.get("updated_at", datetime.utcnow()),
    )


# ----- API Endpoints -----

@router.get("", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    db=Depends(get_db)
):
    """
    List all sessions for the authenticated tutor.
    Returns session summaries (without full stroke data) for performance.
    """
    tutor_id = await get_tutor_id_from_token(request)

    try:
        # Get tenant database (sessions are per-tenant)
        tenant_db = await db.get_tenant_db_from_request(request)
        if tenant_db is None:
            # Fallback to master DB if no tenant context
            tenant_db = await db.get_master_db()

        if tenant_db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        collection = tenant_db["smartboard_sessions"]

        # Query sessions for this tutor
        query = {"tutor_id": tutor_id}

        # Get total count
        total = await collection.count_documents(query)

        # Get sessions with pagination, sorted by updated_at descending
        cursor = collection.find(
            query,
            # Exclude full pages data for list view, only get metadata and page count
            {"pages.strokes": 0}
        ).sort("updated_at", -1).skip(offset).limit(limit)

        sessions_docs = await cursor.to_list(length=limit)

        sessions = [serialize_session_summary(doc) for doc in sessions_docs]

        return SessionListResponse(sessions=sessions, total=total)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


@router.post("", response_model=SessionDetail)
async def create_session(
    request: Request,
    payload: CreateSessionRequest,
    db=Depends(get_db)
):
    """
    Create a new session for the authenticated tutor.
    """
    tutor_id = await get_tutor_id_from_token(request)

    try:
        tenant_db = await db.get_tenant_db_from_request(request)
        if tenant_db is None:
            tenant_db = await db.get_master_db()

        if tenant_db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        collection = tenant_db["smartboard_sessions"]

        now = datetime.utcnow()

        # Convert Pydantic models to dicts
        pages_data = []
        for page in payload.sessionData.pages:
            strokes_data = []
            for stroke in page.strokes:
                stroke_dict = stroke.model_dump()
                # Convert StrokePoint models to dicts
                stroke_dict["points"] = [pt.model_dump() for pt in stroke.points]
                if stroke.startPoint:
                    stroke_dict["startPoint"] = stroke.startPoint.model_dump()
                if stroke.endPoint:
                    stroke_dict["endPoint"] = stroke.endPoint.model_dump()
                strokes_data.append(stroke_dict)
            pages_data.append({
                "id": page.id,
                "strokes": strokes_data,
                "background": page.background
            })

        doc = {
            "tutor_id": tutor_id,
            "name": payload.name,
            "thumbnail": payload.thumbnail,
            "pages": pages_data,
            "current_page_index": payload.sessionData.currentPageIndex,
            "created_at": now,
            "updated_at": now,
        }

        result = await collection.insert_one(doc)
        doc["_id"] = result.inserted_id

        logger.info(f"Created session '{payload.name}' for tutor {tutor_id}")

        return serialize_session_detail(doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    request: Request,
    session_id: str,
    db=Depends(get_db)
):
    """
    Get a specific session by ID.
    Only returns the session if it belongs to the authenticated tutor.
    """
    tutor_id = await get_tutor_id_from_token(request)

    try:
        tenant_db = await db.get_tenant_db_from_request(request)
        if tenant_db is None:
            tenant_db = await db.get_master_db()

        if tenant_db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        collection = tenant_db["smartboard_sessions"]

        try:
            obj_id = ObjectId(session_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        doc = await collection.find_one({
            "_id": obj_id,
            "tutor_id": tutor_id
        })

        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")

        return serialize_session_detail(doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")


@router.put("/{session_id}", response_model=SessionDetail)
async def update_session(
    request: Request,
    session_id: str,
    payload: UpdateSessionRequest,
    db=Depends(get_db)
):
    """
    Update an existing session.
    Only updates the session if it belongs to the authenticated tutor.
    """
    tutor_id = await get_tutor_id_from_token(request)

    try:
        tenant_db = await db.get_tenant_db_from_request(request)
        if tenant_db is None:
            tenant_db = await db.get_master_db()

        if tenant_db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        collection = tenant_db["smartboard_sessions"]

        try:
            obj_id = ObjectId(session_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Build update document
        update_doc = {"updated_at": datetime.utcnow()}

        if payload.name is not None:
            update_doc["name"] = payload.name

        if payload.thumbnail is not None:
            update_doc["thumbnail"] = payload.thumbnail

        if payload.sessionData is not None:
            # Convert Pydantic models to dicts
            pages_data = []
            for page in payload.sessionData.pages:
                strokes_data = []
                for stroke in page.strokes:
                    stroke_dict = stroke.model_dump()
                    stroke_dict["points"] = [pt.model_dump() for pt in stroke.points]
                    if stroke.startPoint:
                        stroke_dict["startPoint"] = stroke.startPoint.model_dump()
                    if stroke.endPoint:
                        stroke_dict["endPoint"] = stroke.endPoint.model_dump()
                    strokes_data.append(stroke_dict)
                pages_data.append({
                    "id": page.id,
                    "strokes": strokes_data,
                    "background": page.background
                })
            update_doc["pages"] = pages_data
            update_doc["current_page_index"] = payload.sessionData.currentPageIndex

        result = await collection.find_one_and_update(
            {"_id": obj_id, "tutor_id": tutor_id},
            {"$set": update_doc},
            return_document=True
        )

        if not result:
            raise HTTPException(status_code=404, detail="Session not found")

        logger.info(f"Updated session {session_id} for tutor {tutor_id}")

        return serialize_session_detail(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.delete("/{session_id}")
async def delete_session(
    request: Request,
    session_id: str,
    db=Depends(get_db)
):
    """
    Delete a session.
    Only deletes the session if it belongs to the authenticated tutor.
    """
    tutor_id = await get_tutor_id_from_token(request)

    try:
        tenant_db = await db.get_tenant_db_from_request(request)
        if tenant_db is None:
            tenant_db = await db.get_master_db()

        if tenant_db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        collection = tenant_db["smartboard_sessions"]

        try:
            obj_id = ObjectId(session_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        result = await collection.delete_one({
            "_id": obj_id,
            "tutor_id": tutor_id
        })

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")

        logger.info(f"Deleted session {session_id} for tutor {tutor_id}")

        return {"success": True, "message": "Session deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")
