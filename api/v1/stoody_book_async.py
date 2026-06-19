"""
Stoody Book API.

Authenticated student workspace for PDF-backed study sessions. Sessions are
soft-deleted; uploaded PDFs remain user-owned records and can be reused by later
sessions.
"""

import hashlib
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import Binary, ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from config_async import settings
from core.database import DatabaseManager
from services.async_openai_service import AsyncOpenAIService

logger = logging.getLogger(__name__)

router = APIRouter()

SESSIONS_COLLECTION = "stoody_book_sessions"
PDFS_COLLECTION = "stoody_book_pdfs"
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_TOTAL_PDF_BYTES = 100 * 1024 * 1024
MAX_PROMPT_TEXT_CHARS = 60000
MAX_MESSAGES_PER_SESSION = 80


class SessionCreateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)


class SessionUpdateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)
    active_pdf_id: Optional[str] = Field(default=None, max_length=64)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


def _now() -> datetime:
    return datetime.utcnow()


def _clean_label(value: Optional[str], fallback: str = "New session") -> str:
    label = (value or "").strip()
    return label[:120] if label else fallback


def _object_id(value: str, label: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found",
        )


def _user_scope(current_user: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(current_user.get("user_id") or current_user.get("id") or "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user is missing a user id",
        )

    return {
        "user_id": user_id,
        "tenant_id": current_user.get("tenant_id"),
        "db_name": current_user.get("db_name"),
        "admin_id": str(current_user.get("admin_id")) if current_user.get("admin_id") else None,
        "user_type": current_user.get("user_type"),
        "is_b2c": bool(current_user.get("is_b2c") or current_user.get("user_type") == "b2c_user"),
    }


async def _workspace_db(db: DatabaseManager, current_user: Dict[str, Any]):
    scope = _user_scope(current_user)
    if scope["is_b2c"]:
        mongo_db = await db.get_b2c_db()
    else:
        db_name = scope.get("db_name")
        if not db_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant database is not available for this user",
            )
        mongo_db = await db.get_tenant_db(str(db_name))

    if mongo_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stoody Book storage is not available",
        )
    return mongo_db


def _owner_filter(current_user: Dict[str, Any]) -> Dict[str, Any]:
    scope = _user_scope(current_user)
    return {
        "user_id": scope["user_id"],
        "db_name": scope.get("db_name"),
        "is_b2c": scope.get("is_b2c", False),
    }


def _serialize_pdf(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not doc:
        return None
    return {
        "id": str(doc["_id"]),
        "filename": doc.get("filename") or "PDF",
        "size": int(doc.get("size") or 0),
        "pages": int(doc.get("pages") or 0),
        "text_chars": len(doc.get("text") or ""),
        "sha256": doc.get("sha256"),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
    }


def _serialize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    created_at = message.get("created_at")
    return {
        "role": message.get("role"),
        "content": message.get("content") or "",
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }


async def _pdf_map(mongo_db, pdf_ids: List[ObjectId]) -> Dict[ObjectId, Dict[str, Any]]:
    if not pdf_ids:
        return {}
    docs = await mongo_db[PDFS_COLLECTION].find(
        {"_id": {"$in": pdf_ids}},
        {"content": 0, "text": 0},
    ).to_list(length=len(pdf_ids))
    return {doc["_id"]: doc for doc in docs}


async def _get_session(mongo_db, current_user: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    session_oid = _object_id(session_id, "Session")
    doc = await mongo_db[SESSIONS_COLLECTION].find_one({
        "_id": session_oid,
        **_owner_filter(current_user),
        "deleted_at": None,
    })
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return doc


async def _storage_used_bytes(mongo_db, current_user: Dict[str, Any]) -> int:
    pipeline = [
        {"$match": {**_owner_filter(current_user), "deleted_at": None}},
        {"$group": {"_id": None, "total": {"$sum": "$size"}}},
    ]
    result = await mongo_db[PDFS_COLLECTION].aggregate(pipeline).to_list(length=1)
    return int((result[0] or {}).get("total") or 0) if result else 0


def _extract_pdf_text(pdf_bytes: bytes) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception:
        logger.exception("pypdf is not available for Stoody Book PDF extraction")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF reading is not available",
        )

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_texts: List[str] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_texts.append(f"Page {index}: {' '.join(text.split())}")
            if sum(len(item) for item in page_texts) >= MAX_PROMPT_TEXT_CHARS:
                break
        return {
            "pages": len(reader.pages),
            "text": "\n\n".join(page_texts)[:MAX_PROMPT_TEXT_CHARS],
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to parse uploaded Stoody Book PDF")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read this PDF",
        )


def _build_system_prompt(pdf_doc: Dict[str, Any]) -> str:
    return "\n\n".join([
        "You are Stoody Book. Answer the student's questions using the uploaded PDF content.",
        "If the answer is not supported by the PDF, say that clearly and ask for the relevant page or section.",
        "For homework-style questions, teach the reasoning before giving a final answer.",
        f"PDF: {pdf_doc.get('filename') or 'PDF'}",
        f"Extracted PDF text:\n{pdf_doc.get('text') or ''}",
    ])


async def _serialize_session(mongo_db, session: Dict[str, Any], include_messages: bool = False) -> Dict[str, Any]:
    raw_pdf_ids = [pid for pid in session.get("pdf_ids", []) if isinstance(pid, ObjectId)]
    pdfs_by_id = await _pdf_map(mongo_db, raw_pdf_ids)
    pdfs = [_serialize_pdf(pdfs_by_id[pid]) for pid in raw_pdf_ids if pid in pdfs_by_id]
    active_pdf_id = session.get("active_pdf_id")
    active_pdf = _serialize_pdf(pdfs_by_id.get(active_pdf_id)) if isinstance(active_pdf_id, ObjectId) else None

    data = {
        "id": str(session["_id"]),
        "label": session.get("label") or "New session",
        "active_pdf_id": str(active_pdf_id) if isinstance(active_pdf_id, ObjectId) else None,
        "active_pdf": active_pdf,
        "pdfs": [pdf for pdf in pdfs if pdf],
        "message_count": len(session.get("messages") or []),
        "created_at": session.get("created_at").isoformat() if session.get("created_at") else None,
        "updated_at": session.get("updated_at").isoformat() if session.get("updated_at") else None,
    }
    if include_messages:
        data["messages"] = [_serialize_message(message) for message in session.get("messages", [])]
    return data


@router.get("/sessions")
async def list_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    sessions = await mongo_db[SESSIONS_COLLECTION].find(
        {**_owner_filter(current_user), "deleted_at": None},
    ).sort("updated_at", -1).to_list(length=100)
    storage_used = await _storage_used_bytes(mongo_db, current_user)
    return {
        "success": True,
        "data": {
            "sessions": [await _serialize_session(mongo_db, session) for session in sessions],
            "storage_used_bytes": storage_used,
            "storage_limit_bytes": MAX_TOTAL_PDF_BYTES,
            "max_pdf_bytes": MAX_PDF_BYTES,
        },
    }


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    now = _now()
    scope = _user_scope(current_user)
    session_doc = {
        **_owner_filter(current_user),
        "tenant_id": scope.get("tenant_id"),
        "admin_id": scope.get("admin_id"),
        "user_type": scope.get("user_type"),
        "label": _clean_label(payload.label),
        "active_pdf_id": None,
        "pdf_ids": [],
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    result = await mongo_db[SESSIONS_COLLECTION].insert_one(session_doc)
    session_doc["_id"] = result.inserted_id
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, session_doc, True)}}


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, session, True)}}


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    payload: SessionUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    update: Dict[str, Any] = {"updated_at": _now()}

    if payload.label is not None:
        update["label"] = _clean_label(payload.label)

    if payload.active_pdf_id is not None:
        pdf_oid = _object_id(payload.active_pdf_id, "PDF")
        if pdf_oid not in session.get("pdf_ids", []):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF is not attached to this session")
        update["active_pdf_id"] = pdf_oid

    await mongo_db[SESSIONS_COLLECTION].update_one({"_id": session["_id"]}, {"$set": update})
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, refreshed, True)}}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {"$set": {"deleted_at": _now(), "updated_at": _now()}},
    )
    return {"success": True}


@router.post("/sessions/{session_id}/pdfs", status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    session_id: str,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    filename = (file.filename or "uploaded.pdf").strip() or "uploaded.pdf"
    if not filename.lower().endswith(".pdf") or file.content_type not in {"application/pdf", "application/octet-stream", "", None}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed")

    content = await file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PDF must be 10 MB or smaller")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This file is not a valid PDF")

    storage_used = await _storage_used_bytes(mongo_db, current_user)
    digest = hashlib.sha256(content).hexdigest()
    pdf_doc = await mongo_db[PDFS_COLLECTION].find_one({**_owner_filter(current_user), "sha256": digest, "deleted_at": None})

    if not pdf_doc:
        if storage_used + len(content) > MAX_TOTAL_PDF_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Total PDF storage limit is 100 MB")
        extracted = _extract_pdf_text(content)
        now = _now()
        scope = _user_scope(current_user)
        pdf_doc = {
            **_owner_filter(current_user),
            "tenant_id": scope.get("tenant_id"),
            "admin_id": scope.get("admin_id"),
            "filename": filename[:220],
            "content_type": "application/pdf",
            "size": len(content),
            "sha256": digest,
            "pages": extracted["pages"],
            "text": extracted["text"],
            "content": Binary(content),
            "created_at": now,
            "updated_at": now,
            "deleted_at": None,
        }
        result = await mongo_db[PDFS_COLLECTION].insert_one(pdf_doc)
        pdf_doc["_id"] = result.inserted_id

    now = _now()
    update_doc: Dict[str, Any] = {
        "$set": {
            "active_pdf_id": pdf_doc["_id"],
            "updated_at": now,
        },
        "$addToSet": {"pdf_ids": pdf_doc["_id"]},
    }
    if not session.get("pdf_ids") and _clean_label(session.get("label")) == "New session":
        update_doc["$set"]["label"] = filename[:120]
    await mongo_db[SESSIONS_COLLECTION].update_one({"_id": session["_id"]}, update_doc)
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {
        "success": True,
        "data": {
            "pdf": _serialize_pdf(pdf_doc),
            "session": await _serialize_session(mongo_db, refreshed, True),
            "storage_used_bytes": await _storage_used_bytes(mongo_db, current_user),
            "storage_limit_bytes": MAX_TOTAL_PDF_BYTES,
        },
    }


@router.get("/pdfs/{pdf_id}/content")
async def get_pdf_content(
    pdf_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    pdf_oid = _object_id(pdf_id, "PDF")
    pdf_doc = await mongo_db[PDFS_COLLECTION].find_one({
        "_id": pdf_oid,
        **_owner_filter(current_user),
        "deleted_at": None,
    })
    if not pdf_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found")
    filename = (pdf_doc.get("filename") or "stoody-book.pdf").replace('"', "")
    return Response(
        bytes(pdf_doc.get("content") or b""),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.post("/sessions/{session_id}/chat")
async def chat_with_pdf(
    session_id: str,
    payload: ChatRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    del request
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    active_pdf_id = session.get("active_pdf_id")
    if not isinstance(active_pdf_id, ObjectId):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a PDF before asking")

    pdf_doc = await mongo_db[PDFS_COLLECTION].find_one({
        "_id": active_pdf_id,
        **_owner_filter(current_user),
        "deleted_at": None,
    }, {"content": 0})
    if not pdf_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found")
    if not pdf_doc.get("text"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No selectable text found in this PDF")

    user_message = {
        "role": "user",
        "content": payload.message.strip(),
        "created_at": _now(),
    }
    history = [
        {"role": item.get("role"), "content": item.get("content")}
        for item in (session.get("messages") or [])[-10:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    messages = [{"role": "system", "content": _build_system_prompt(pdf_doc)}]
    messages.extend(history)
    messages.append({"role": "user", "content": payload.message.strip()})

    openai_service = AsyncOpenAIService()
    ai_response = await openai_service.chat_completion_async(
        messages=messages,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
    )
    if not ai_response.get("success"):
        logger.error("Stoody Book chat failed: %s", ai_response.get("error"))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stoody Book could not answer right now")

    assistant_message = {
        "role": "assistant",
        "content": ai_response.get("response") or "",
        "created_at": _now(),
    }
    next_messages = (session.get("messages") or []) + [user_message, assistant_message]
    next_messages = next_messages[-MAX_MESSAGES_PER_SESSION:]
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {"$set": {"messages": next_messages, "updated_at": _now()}},
    )
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {
        "success": True,
        "data": {
            "response": assistant_message["content"],
            "model": ai_response.get("model"),
            "usage": ai_response.get("usage"),
            "session": await _serialize_session(mongo_db, refreshed, True),
        },
    }
