"""Resolve and render canonical Practice evidence from synchronized strokes."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from bson import ObjectId

from core.database import DatabaseManager
from core.user_identity import canvas_user_id_variants
from services.student_credits import render_stroke_page


class PracticeStrokeEvidenceError(RuntimeError):
    """The referenced Practice answer is incomplete or cannot be verified."""


@dataclass(frozen=True)
class ResolvedPracticeEvidence:
    data_urls: List[str]
    receipt: Dict[str, Any]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _user_id_variants(current_user: Dict[str, Any]) -> List[Any]:
    values: List[Any] = list(canvas_user_id_variants(current_user))
    for raw in (
        current_user.get("user_id"),
        current_user.get("_id"),
        current_user.get("student_id"),
    ):
        if raw is None:
            continue
        values.extend([raw, str(raw)])
        try:
            if ObjectId.is_valid(str(raw)):
                values.append(ObjectId(str(raw)))
        except Exception:
            pass
    result: List[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = f"{type(value).__name__}:{value}"
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


async def _canvas_collection(current_user: Dict[str, Any], db: DatabaseManager):
    is_b2c = current_user.get("is_b2c", False) or current_user.get("user_type") == "b2c_user"
    if is_b2c:
        b2c_db = await db.get_b2c_db()
        if b2c_db is None:
            raise PracticeStrokeEvidenceError("The student canvas database is unavailable.")
        return b2c_db["canvas_pages"]
    db_name = current_user.get("db_name")
    if not db_name:
        raise PracticeStrokeEvidenceError("Tenant context is required to verify Practice writing.")
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise PracticeStrokeEvidenceError("The tenant canvas database is unavailable.")
    return tenant_db["canvas_pages"]


def _normalise_page_refs(refs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    default_book = str(refs.get("bookType") or "").upper()
    virtual_pages = refs.get("virtualPages") or []
    pages: List[Dict[str, Any]] = []
    for raw in virtual_pages:
        page = _as_mapping(raw)
        ordinal = int(page.get("ordinal") or 0)
        physical = page.get("physicalPageNo")
        if physical is None:
            book_type, page_number = "VIRTUAL", ordinal
        else:
            book_type = str(page.get("bookType") or default_book).upper()
            page_number = int(physical)
        if not book_type or page_number < 0:
            continue
        raw_intervals = page.get("timeIntervals") or page.get("visitIntervals") or []
        time_intervals = [
            {
                "start_ts": _as_mapping(interval).get("startTs"),
                "end_ts": _as_mapping(interval).get("endTs"),
            }
            for interval in raw_intervals
            if _as_mapping(interval)
        ]
        if not time_intervals:
            time_intervals = [{"start_ts": page.get("startTs"), "end_ts": page.get("endTs")}]
        pages.append(
            {
                "book_type": book_type,
                "page_number": page_number,
                "ordinal": ordinal or None,
                "start_ts": page.get("startTs"),
                "end_ts": page.get("endTs"),
                "time_intervals": time_intervals,
            }
        )

    if pages:
        return pages

    intervals = [
        {
            "start_ts": _as_mapping(interval).get("startTs"),
            "end_ts": _as_mapping(interval).get("endTs"),
        }
        for interval in (refs.get("timeIntervals") or [])
        if _as_mapping(interval)
    ]
    for page_number in refs.get("activePages") or []:
        first_interval = intervals[0] if intervals else {}
        pages.append(
            {
                "book_type": default_book,
                "page_number": int(page_number),
                "ordinal": None,
                "start_ts": first_interval.get("start_ts"),
                "end_ts": first_interval.get("end_ts"),
                "time_intervals": intervals,
            }
        )
    return pages


def _stroke_matches_scope(
    stroke: Mapping[str, Any],
    *,
    practice_session_id: str,
    question_id: str,
    ordinal: Optional[int],
    start_ts: Any,
    end_ts: Any,
    time_intervals: Optional[List[Mapping[str, Any]]] = None,
) -> bool:
    stroke_session = stroke.get("practiceSessionId")
    stroke_question = stroke.get("questionId")
    stroke_ordinal = stroke.get("virtualPageOrdinal")
    has_scope = stroke_session is not None or stroke_question is not None
    if has_scope:
        if str(stroke_session or "") != practice_session_id:
            return False
        if str(stroke_question or "") != question_id:
            return False
        if ordinal is not None and int(stroke_ordinal or 0) != ordinal:
            return False
        return True

    # Legacy scoped pages may predate the ownership fields. Accept only a
    # stroke whose wall-clock activity falls inside the submitted page window.
    timestamp = stroke.get("startedAt", stroke.get("timestamp"))
    if not isinstance(timestamp, (int, float)):
        return False
    intervals = list(time_intervals or [])
    if not intervals:
        intervals = [{"start_ts": start_ts, "end_ts": end_ts}]
    for interval in intervals:
        interval_start = interval.get("start_ts")
        if not isinstance(interval_start, (int, float)):
            continue
        interval_end = interval.get("end_ts")
        upper = interval_end if isinstance(interval_end, (int, float)) else float("inf")
        if float(interval_start) <= float(timestamp) <= float(upper):
            return True
    return False


async def resolve_practice_stroke_evidence(
    *,
    current_user: Dict[str, Any],
    db: DatabaseManager,
    refs: Any,
    payload_question_id: str,
) -> ResolvedPracticeEvidence:
    ref_map = _as_mapping(refs)
    copy_id = str(ref_map.get("copyId") or "").strip()
    practice_session_id = str(ref_map.get("practiceSessionId") or "").strip()
    question_id = str(ref_map.get("questionId") or "").strip()
    if not copy_id or not practice_session_id or not question_id:
        raise PracticeStrokeEvidenceError("Practice page ownership references are incomplete.")
    if question_id != str(payload_question_id):
        raise PracticeStrokeEvidenceError("Practice page references belong to a different question.")

    page_refs = _normalise_page_refs(ref_map)
    if not page_refs:
        raise PracticeStrokeEvidenceError("No synchronized Practice pages were referenced.")

    collection = await _canvas_collection(current_user, db)
    identities = {
        (page["book_type"], page["page_number"])
        for page in page_refs
    }
    query = {
        "user_id": {"$in": _user_id_variants(current_user)},
        "copy_id": copy_id,
        "$or": [
            {"book_type": book_type, "page_number": page_number}
            for book_type, page_number in identities
        ],
    }
    docs = await collection.find(query).sort("last_modified", 1).to_list(length=100)
    docs_by_identity: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for doc in docs:
        key = (str(doc.get("book_type") or "").upper(), int(doc.get("page_number") or 0))
        docs_by_identity.setdefault(key, []).append(doc)

    data_urls: List[str] = []
    page_receipts: List[Dict[str, Any]] = []
    for page_ref in page_refs:
        key = (page_ref["book_type"], page_ref["page_number"])
        scoped: Dict[str, Mapping[str, Any]] = {}
        for doc in docs_by_identity.get(key, []):
            for raw_stroke in doc.get("strokes") or []:
                if not isinstance(raw_stroke, Mapping):
                    continue
                if raw_stroke.get("processingVersion") != "ble-canonical-v1":
                    continue
                if raw_stroke.get("sourceMode") not in {"live", "offlineReplay", "touch"}:
                    continue
                if not _stroke_matches_scope(
                    raw_stroke,
                    practice_session_id=practice_session_id,
                    question_id=question_id,
                    ordinal=page_ref["ordinal"],
                    start_ts=page_ref["start_ts"],
                    end_ts=page_ref["end_ts"],
                    time_intervals=page_ref.get("time_intervals"),
                ):
                    continue
                stroke_id = str(raw_stroke.get("id") or "").strip()
                if stroke_id:
                    scoped[stroke_id] = raw_stroke

        strokes = list(scoped.values())
        if not strokes:
            raise PracticeStrokeEvidenceError(
                f"Synchronized writing for {key[0]} page {key[1]} is missing or belongs to another question."
            )

        canonical_json = json.dumps(strokes, sort_keys=True, separators=(",", ":"), default=str)
        content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        png = render_stroke_page({"strokes": strokes})
        data_urls.append(f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}")
        page_receipts.append(
            {
                "book_type": key[0],
                "page_number": key[1],
                "virtual_page_ordinal": page_ref["ordinal"],
                "stroke_count": len(strokes),
                "point_count": sum(len(stroke.get("points") or []) for stroke in strokes),
                "content_sha256": content_hash,
            }
        )

    receipt_payload = {
        "version": "practice-canonical-evidence-v1",
        "copy_id": copy_id,
        "practice_session_id": practice_session_id,
        "question_id": question_id,
        "pages": page_receipts,
    }
    receipt_payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResolvedPracticeEvidence(data_urls=data_urls, receipt=receipt_payload)
