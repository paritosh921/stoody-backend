"""
Stoody Book API.

Authenticated student workspace for PDF-backed study sessions. Sessions are
soft-deleted; uploaded PDFs remain user-owned records and can be reused by later
sessions.
"""

import hashlib
import io
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

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
MAX_PAGE_TEXT_CHARS = 12000
MAX_SEARCH_RESULTS = 12
MAX_CITATIONS = 3
TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
LEGACY_PAGE_RE = re.compile(r"(?:^|\n\n)Page\s+(\d+):\s*", re.IGNORECASE)


class SessionCreateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)


class SessionUpdateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)
    active_pdf_id: Optional[str] = Field(default=None, max_length=64)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class AnnotationCreateRequest(BaseModel):
    pdf_id: str = Field(..., min_length=1, max_length=64)
    page: int = Field(..., ge=1)
    quote: str = Field(..., min_length=1, max_length=500)
    note: Optional[str] = Field(default=None, max_length=1200)
    color: Optional[str] = Field(default="yellow", max_length=24)


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
    data = {
        "role": message.get("role"),
        "content": message.get("content") or "",
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }
    citations = message.get("citations")
    if isinstance(citations, list):
        data["citations"] = citations
    return data


def _serialize_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    created_at = annotation.get("created_at")
    return {
        "id": annotation.get("id"),
        "pdf_id": str(annotation.get("pdf_id")) if annotation.get("pdf_id") else None,
        "page": int(annotation.get("page") or 1),
        "quote": annotation.get("quote") or "",
        "note": annotation.get("note") or "",
        "color": annotation.get("color") or "yellow",
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


async def _get_pdf(mongo_db, current_user: Dict[str, Any], pdf_id: str, include_content: bool = False) -> Dict[str, Any]:
    pdf_oid = _object_id(pdf_id, "PDF")
    projection = None if include_content else {"content": 0}
    doc = await mongo_db[PDFS_COLLECTION].find_one({
        "_id": pdf_oid,
        **_owner_filter(current_user),
        "deleted_at": None,
    }, projection)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found")
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
        stored_pages: List[Dict[str, Any]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                normalized = " ".join(text.split())
                stored_pages.append({"page": index, "text": normalized[:MAX_PAGE_TEXT_CHARS]})
                page_texts.append(f"Page {index}: {normalized}")
            if sum(len(item) for item in page_texts) >= MAX_PROMPT_TEXT_CHARS:
                break
        return {
            "pages": len(reader.pages),
            "text": "\n\n".join(page_texts)[:MAX_PROMPT_TEXT_CHARS],
            "page_texts": stored_pages,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to parse uploaded Stoody Book PDF")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read this PDF",
        )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value or "")}


def _pdf_pages(pdf_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    stored_pages = pdf_doc.get("page_texts")
    if isinstance(stored_pages, list):
        for item in stored_pages:
            if not isinstance(item, dict):
                continue
            try:
                page_number = int(item.get("page") or 0)
            except Exception:
                continue
            text = _normalize_text(item.get("text"))
            if page_number >= 1 and text:
                pages.append({"page": page_number, "text": text[:MAX_PAGE_TEXT_CHARS]})
        if pages:
            return pages

    text = str(pdf_doc.get("text") or "")
    matches = list(LEGACY_PAGE_RE.finditer(text))
    if not matches:
        normalized = _normalize_text(text)
        return [{"page": 1, "text": normalized[:MAX_PAGE_TEXT_CHARS]}] if normalized else []

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = _normalize_text(text[start:end])
        if page_text:
            pages.append({"page": int(match.group(1)), "text": page_text[:MAX_PAGE_TEXT_CHARS]})
    return pages


def _snippet_around(text: str, query_tokens: set[str], width: int = 280) -> str:
    lowered = text.lower()
    positions = [
        lowered.find(token)
        for token in query_tokens
        if token and lowered.find(token) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - width // 3)
    end = min(len(text), start + width)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet


def _search_pdf_pages(pages: Sequence[Dict[str, Any]], query: str, limit: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
    query_text = _normalize_text(query)
    if not query_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search query is required")

    query_tokens = _tokens(query_text)
    if not query_tokens:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search query is too short")

    results: List[Dict[str, Any]] = []
    for page in pages:
        page_text = _normalize_text(page.get("text"))
        if not page_text:
            continue
        page_tokens = _tokens(page_text)
        overlap = query_tokens & page_tokens
        phrase_bonus = 2 if query_text.lower() in page_text.lower() else 0
        score = len(overlap) + phrase_bonus
        if score <= 0:
            continue
        results.append({
            "page": int(page.get("page") or 1),
            "score": score,
            "snippet": _snippet_around(page_text, query_tokens),
        })

    results.sort(key=lambda item: (-item["score"], item["page"]))
    return results[: max(1, min(limit, MAX_SEARCH_RESULTS))]


def _select_grounding_pages(question: str, pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        return _search_pdf_pages(pages, question, limit=5)
    except HTTPException:
        return []


def _suggest_citations(question: str, answer: str, pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_tokens = _tokens(f"{question} {answer}")
    if not query_tokens:
        return []

    ranked: List[Dict[str, Any]] = []
    for page in pages:
        page_text = _normalize_text(page.get("text"))
        if not page_text:
            continue
        overlap = query_tokens & _tokens(page_text)
        if not overlap:
            continue
        ranked.append({
            "page": int(page.get("page") or 1),
            "score": len(overlap),
            "quote": _snippet_around(page_text, overlap, width=220),
        })
    ranked.sort(key=lambda item: (-item["score"], item["page"]))
    return [{"page": item["page"], "quote": item["quote"]} for item in ranked[:MAX_CITATIONS]]


def _validate_annotation_payload(page_text: str, page: int, quote: str, note: Optional[str]) -> Dict[str, Any]:
    clean_quote = _normalize_text(quote)
    clean_note = _normalize_text(note)
    if not clean_quote:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Annotation quote is required")
    if clean_quote.lower() not in _normalize_text(page_text).lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Annotation quote was not found on page {page}",
        )
    return {
        "quote": clean_quote[:500],
        "note": clean_note[:1200],
    }


def _build_system_prompt(pdf_doc: Dict[str, Any], grounding_pages: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    pages = grounding_pages or []
    grounding = "\n\n".join([
        f"Page {item.get('page')}: {item.get('snippet') or item.get('text') or ''}"
        for item in pages
    ])
    return "\n\n".join([
        "You are Stoody Book. Answer the student's questions using the uploaded PDF content.",
        "If the answer is not supported by the PDF, say that clearly and ask for the relevant page or section.",
        "For homework-style questions, teach the reasoning before giving a final answer.",
        "When useful, mention page numbers from the supplied context.",
        f"PDF: {pdf_doc.get('filename') or 'PDF'}",
        f"Most relevant pages:\n{grounding}" if grounding else "Most relevant pages: none found from the question.",
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
        "annotations": [_serialize_annotation(annotation) for annotation in session.get("annotations", [])],
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
        "annotations": [],
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
            "page_texts": extracted["page_texts"],
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
    pdf_doc = await _get_pdf(mongo_db, current_user, pdf_id, include_content=True)
    filename = (pdf_doc.get("filename") or "stoody-book.pdf").replace('"', "")
    return Response(
        bytes(pdf_doc.get("content") or b""),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@router.get("/pdfs/{pdf_id}/pages")
async def read_pdf_pages(
    pdf_id: str,
    start: int = 1,
    end: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    pdf_doc = await _get_pdf(mongo_db, current_user, pdf_id)
    pages = _pdf_pages(pdf_doc)
    start_page = max(1, start)
    end_page = max(start_page, end) if end else start_page
    selected = [
        {"page": item["page"], "text": item["text"]}
        for item in pages
        if start_page <= int(item["page"]) <= end_page
    ]
    return {"success": True, "data": {"pages": selected}}


@router.get("/pdfs/{pdf_id}/search")
async def search_pdf(
    pdf_id: str,
    q: str,
    limit: int = 8,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    pdf_doc = await _get_pdf(mongo_db, current_user, pdf_id)
    results = _search_pdf_pages(_pdf_pages(pdf_doc), q, limit=limit)
    return {"success": True, "data": {"results": results}}


@router.post("/sessions/{session_id}/annotations", status_code=status.HTTP_201_CREATED)
async def create_annotation(
    session_id: str,
    payload: AnnotationCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    pdf_oid = _object_id(payload.pdf_id, "PDF")
    if pdf_oid not in session.get("pdf_ids", []):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF is not attached to this session")

    pdf_doc = await _get_pdf(mongo_db, current_user, payload.pdf_id)
    pages = _pdf_pages(pdf_doc)
    page_doc = next((item for item in pages if int(item["page"]) == payload.page), None)
    if not page_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")
    validated = _validate_annotation_payload(page_doc["text"], payload.page, payload.quote, payload.note)
    annotation = {
        "id": uuid.uuid4().hex,
        "pdf_id": pdf_oid,
        "page": payload.page,
        "quote": validated["quote"],
        "note": validated["note"],
        "color": (payload.color or "yellow")[:24],
        "created_at": _now(),
    }
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {"$push": {"annotations": annotation}, "$set": {"updated_at": _now()}},
    )
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, refreshed, True)}}


@router.delete("/sessions/{session_id}/annotations/{annotation_id}")
async def delete_annotation(
    session_id: str,
    annotation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {"$pull": {"annotations": {"id": annotation_id}}, "$set": {"updated_at": _now()}},
    )
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, refreshed, True)}}


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

    pdf_doc = await _get_pdf(mongo_db, current_user, str(active_pdf_id))
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
    pdf_pages = _pdf_pages(pdf_doc)
    grounding_pages = _select_grounding_pages(payload.message, pdf_pages)
    messages = [{"role": "system", "content": _build_system_prompt(pdf_doc, grounding_pages)}]
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
        "citations": _suggest_citations(payload.message, ai_response.get("response") or "", pdf_pages),
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
