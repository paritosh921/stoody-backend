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
from datetime import datetime, timedelta
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
MAX_OPEN_CHECKS = 10
TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
LEGACY_PAGE_RE = re.compile(r"(?:^|\n\n)Page\s+(\d+):\s*", re.IGNORECASE)
REVIEW_INTERVALS = {1: 1, 2: 3, 3: 7, 4: 14, 5: 30}


class SessionCreateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)


class SessionUpdateRequest(BaseModel):
    label: Optional[str] = Field(default=None, max_length=120)
    active_pdf_id: Optional[str] = Field(default=None, max_length=64)
    view_state: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    mode: Optional[str] = Field(default=None, max_length=24)


class AnnotationCreateRequest(BaseModel):
    pdf_id: str = Field(..., min_length=1, max_length=64)
    page: int = Field(..., ge=1)
    quote: str = Field(..., min_length=1, max_length=500)
    note: Optional[str] = Field(default=None, max_length=1200)
    color: Optional[str] = Field(default="yellow", max_length=24)


class LearningModeRequest(BaseModel):
    mode: str = Field(..., max_length=24)


class StudyCheckRequest(BaseModel):
    concept: Optional[str] = Field(default=None, max_length=120)


class LearningEventRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=120)
    outcome: str = Field(..., max_length=24)
    check_id: Optional[str] = Field(default=None, max_length=64)
    page: Optional[int] = Field(default=None, ge=1)
    quote: Optional[str] = Field(default=None, max_length=500)
    prompt: Optional[str] = Field(default=None, max_length=500)


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


def _iso(value: datetime) -> str:
    return value.isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def _build_view_state(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    pdf_id = str(raw.get("pdf_id") or "").strip()
    if pdf_id:
        _object_id(pdf_id, "PDF")
    try:
        page = max(1, int(raw.get("page") or 1))
    except Exception:
        page = 1
    try:
        zoom = float(raw.get("zoom") or 1.05)
    except Exception:
        zoom = 1.05
    focused = raw.get("focused_quote") if isinstance(raw.get("focused_quote"), dict) else None
    focused_quote = None
    if focused:
        try:
            focused_page = max(1, int(focused.get("page") or page))
        except Exception:
            focused_page = page
        quote = _normalize_text(focused.get("quote"))[:500]
        if quote:
            focused_quote = {"page": focused_page, "quote": quote}
    state = {
        "pdf_id": pdf_id or None,
        "page": page,
        "zoom": max(0.65, min(2.0, round(zoom, 2))),
    }
    if focused_quote:
        state["focused_quote"] = focused_quote
    return state


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


def _normalize_concept(value: Any) -> str:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(value or ""))]
    return " ".join(tokens[:8])


def _clean_learner_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = state if isinstance(state, dict) else {}
    mode = raw.get("mode") if raw.get("mode") in {"answer", "learning"} else "answer"
    concepts = raw.get("concepts") if isinstance(raw.get("concepts"), list) else []
    open_checks = raw.get("open_checks") if isinstance(raw.get("open_checks"), list) else []
    return {
        "mode": mode,
        "concepts": [item for item in concepts if isinstance(item, dict) and item.get("normalized")],
        "open_checks": [item for item in open_checks if isinstance(item, dict) and item.get("id")][-MAX_OPEN_CHECKS:],
    }


def _record_learning_event(
    learner_state: Optional[Dict[str, Any]],
    event: Dict[str, Any],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current = now or _now()
    state = _clean_learner_state(learner_state)
    label = _normalize_text(event.get("concept"))[:120]
    normalized = _normalize_concept(label)
    if not normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Concept is required")

    outcome = str(event.get("outcome") or "").strip().lower()
    if outcome not in {"correct", "partial", "incorrect", "skipped"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported learning outcome")

    concepts = state["concepts"]
    concept = next((item for item in concepts if item.get("normalized") == normalized), None)
    if not concept:
        concept = {
            "label": label,
            "normalized": normalized,
            "box": 1,
            "checks": [],
            "first_seen_at": _iso(current),
        }
        concepts.append(concept)

    box = int(concept.get("box") or 1)
    if outcome == "correct":
        box = min(5, box + 1)
    elif outcome in {"incorrect", "skipped"}:
        box = 1
    concept["box"] = box
    concept["label"] = label
    concept["page"] = event.get("page") or concept.get("page")
    concept["quote"] = _normalize_text(event.get("quote"))[:500] or concept.get("quote") or ""
    concept["last_seen_at"] = _iso(current)
    concept["due_at"] = _iso(current + timedelta(days=REVIEW_INTERVALS.get(box, 1)))
    checks = concept.get("checks") if isinstance(concept.get("checks"), list) else []
    checks.append({
        "id": str(event.get("check_id") or uuid.uuid4().hex),
        "outcome": outcome,
        "prompt": _normalize_text(event.get("prompt"))[:500],
        "created_at": _iso(current),
    })
    concept["checks"] = checks[-20:]
    check_id = str(event.get("check_id") or "")
    if check_id:
        state["open_checks"] = [
            check for check in state.get("open_checks", [])
            if check.get("id") != check_id
        ]
    state["concepts"] = concepts
    state["mode"] = state.get("mode") or "learning"
    return state


def _compute_due_reviews(learner_state: Optional[Dict[str, Any]], now: Optional[datetime] = None, limit: int = 5) -> List[Dict[str, Any]]:
    current = now or _now()
    state = _clean_learner_state(learner_state)
    due: List[Dict[str, Any]] = []
    for concept in state.get("concepts", []):
        due_at = _parse_iso(concept.get("due_at"))
        if not due_at or due_at > current:
            continue
        due.append({
            "label": concept.get("label") or concept.get("normalized") or "Concept",
            "page": int(concept.get("page") or 1),
            "quote": concept.get("quote") or "",
            "due_at": concept.get("due_at"),
        })
    due.sort(key=lambda item: item.get("due_at") or "")
    return due[: max(1, min(limit, 12))]


def _build_study_check(pages: Sequence[Dict[str, Any]], concept: Optional[str] = None) -> Dict[str, Any]:
    concept_text = _normalize_text(concept)[:120]
    selected = None
    if concept_text:
        matches = _search_pdf_pages(pages, concept_text, limit=1)
        selected = matches[0] if matches else None
    if not selected and pages:
        page = pages[0]
        selected = {
            "page": int(page.get("page") or 1),
            "snippet": _snippet_around(_normalize_text(page.get("text")), _tokens(_normalize_text(page.get("text"))) or set(), width=220),
        }
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No readable PDF text is available")
    label = concept_text or "this passage"
    return {
        "id": uuid.uuid4().hex,
        "concept": label,
        "page": int(selected.get("page") or 1),
        "quote": selected.get("snippet") or "",
        "prompt": f"Before I explain: what do you already think about {label}?",
        "created_at": _iso(_now()),
    }


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


def _learner_state_summary(learner_state: Optional[Dict[str, Any]]) -> str:
    state = _clean_learner_state(learner_state)
    concepts = state.get("concepts", [])[-6:]
    reviews = _compute_due_reviews(state, limit=3)
    lines = [f"Mode: {state.get('mode')}"]
    if concepts:
        lines.append("Covered concepts: " + ", ".join(str(item.get("label") or item.get("normalized")) for item in concepts))
    if reviews:
        lines.append("Due reviews: " + ", ".join(str(item.get("label")) for item in reviews))
    return "\n".join(lines)


def _build_system_prompt(
    pdf_doc: Dict[str, Any],
    grounding_pages: Optional[Sequence[Dict[str, Any]]] = None,
    learner_state: Optional[Dict[str, Any]] = None,
) -> str:
    pages = grounding_pages or []
    state = _clean_learner_state(learner_state)
    grounding = "\n\n".join([
        f"Page {item.get('page')}: {item.get('snippet') or item.get('text') or ''}"
        for item in pages
    ])
    learning_guidance = (
        "Learning Mode is on. Prefer active recall: ask one short prediction or retrieval check before a full explanation when the question is conceptual. "
        "Resolve any prior check if the student appears to be answering it. Avoid clutter; cite the best page anchor."
        if state.get("mode") == "learning"
        else "Answer Mode is on. Answer directly, but keep claims grounded in the PDF."
    )
    return "\n\n".join([
        "You are Stoody Book. Answer the student's questions using the uploaded PDF content.",
        "If the answer is not supported by the PDF, say that clearly and ask for the relevant page or section.",
        "For homework-style questions, teach the reasoning before giving a final answer.",
        "When useful, mention page numbers from the supplied context.",
        learning_guidance,
        f"Learner state:\n{_learner_state_summary(state)}",
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
        "learner_state": _clean_learner_state(session.get("learner_state")),
        "due_reviews": _compute_due_reviews(session.get("learner_state")),
        "view_state": _build_view_state(session.get("view_state")),
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
        "learner_state": {"mode": "answer", "concepts": [], "open_checks": []},
        "view_state": None,
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
        current_view = _build_view_state(session.get("view_state")) or {}
        current_view["pdf_id"] = payload.active_pdf_id
        update["view_state"] = _build_view_state(current_view)

    if payload.view_state is not None:
        view_state = _build_view_state(payload.view_state)
        if view_state and view_state.get("pdf_id"):
            pdf_oid = _object_id(str(view_state["pdf_id"]), "PDF")
            if pdf_oid not in session.get("pdf_ids", []):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF is not attached to this session")
        update["view_state"] = view_state

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
            "view_state": {"pdf_id": str(pdf_doc["_id"]), "page": 1, "zoom": 1.05},
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


@router.patch("/sessions/{session_id}/learning-mode")
async def set_learning_mode(
    session_id: str,
    payload: LearningModeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mode = payload.mode.strip().lower()
    if mode not in {"answer", "learning"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Learning mode must be answer or learning")
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    learner_state = _clean_learner_state(session.get("learner_state"))
    learner_state["mode"] = mode
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {"$set": {"learner_state": learner_state, "updated_at": _now()}},
    )
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, refreshed, True)}}


@router.post("/sessions/{session_id}/study-check", status_code=status.HTTP_201_CREATED)
async def create_study_check(
    session_id: str,
    payload: StudyCheckRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    active_pdf_id = session.get("active_pdf_id")
    if not isinstance(active_pdf_id, ObjectId):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload a PDF before starting a study check")
    pdf_doc = await _get_pdf(mongo_db, current_user, str(active_pdf_id))
    check = _build_study_check(_pdf_pages(pdf_doc), payload.concept)
    learner_state = _clean_learner_state(session.get("learner_state"))
    learner_state["mode"] = "learning"
    learner_state["open_checks"] = (learner_state.get("open_checks") or []) + [check]
    learner_state["open_checks"] = learner_state["open_checks"][-MAX_OPEN_CHECKS:]
    assistant_message = {
        "role": "assistant",
        "content": check["prompt"],
        "citations": [{"page": check["page"], "quote": check["quote"]}],
        "learning_check_id": check["id"],
        "created_at": _now(),
    }
    next_messages = (session.get("messages") or []) + [assistant_message]
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {
            "$set": {
                "learner_state": learner_state,
                "messages": next_messages[-MAX_MESSAGES_PER_SESSION:],
                "updated_at": _now(),
            },
        },
    )
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"check": check, "session": await _serialize_session(mongo_db, refreshed, True)}}


@router.post("/sessions/{session_id}/learning-events", status_code=status.HTTP_201_CREATED)
async def record_learning_event(
    session_id: str,
    payload: LearningEventRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    learner_state = _record_learning_event(session.get("learner_state"), payload.model_dump())
    await mongo_db[SESSIONS_COLLECTION].update_one(
        {"_id": session["_id"]},
        {"$set": {"learner_state": learner_state, "updated_at": _now()}},
    )
    refreshed = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"session": await _serialize_session(mongo_db, refreshed, True)}}


@router.get("/sessions/{session_id}/reviews")
async def list_due_reviews(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    mongo_db = await _workspace_db(db, current_user)
    session = await _get_session(mongo_db, current_user, session_id)
    return {"success": True, "data": {"reviews": _compute_due_reviews(session.get("learner_state"))}}


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
    learner_state = _clean_learner_state(session.get("learner_state"))
    if payload.mode in {"answer", "learning"}:
        learner_state["mode"] = payload.mode
    messages = [{"role": "system", "content": _build_system_prompt(pdf_doc, grounding_pages, learner_state)}]
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
        {"$set": {"messages": next_messages, "learner_state": learner_state, "updated_at": _now()}},
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
