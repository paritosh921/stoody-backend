"""Immutable paper snapshots for conducted ExamPen sessions.

The content-management document is an authoring object.  A conducted exam
needs an immutable version of that paper so that later changes, retries, or a
second sitting cannot change the questions that were used to mark a student.

This module owns the bridge between a finalized ``documents`` record and the
session-scoped ExamPen metadata consumed by DCR/PCR engines.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

PAPER_VERSIONS_COLLECTION = "exampen_paper_versions"
PAPER_QUESTIONS_COLLECTION = "exampen_paper_questions"


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _source_question_id(question: Dict[str, Any]) -> str:
    """Return the stable source id used to trace a snapshot question."""
    for key in ("question_id", "id", "_id"):
        value = _as_text(question.get(key))
        if value:
            return value
    return ""


def _question_text(question: Dict[str, Any]) -> str:
    for key in ("question_text", "text", "question", "content"):
        value = _as_text(question.get(key))
        if value:
            return value
    return ""


def _question_marks(question: Dict[str, Any]) -> Optional[float]:
    for key in ("max_marks", "marks", "points", "total_points"):
        raw = question.get(key)
        if raw in (None, "") or isinstance(raw, bool):
            continue
        try:
            marks = float(raw)
        except (TypeError, ValueError):
            continue
        if marks > 0:
            return marks
    return None


def _question_reference_solution(question: Dict[str, Any]) -> str:
    for key in (
        "reference_solution",
        "solution",
        "answer",
        "correct_answer",
        "correctAnswer",
        "final_answer_text",
    ):
        value = _as_text(question.get(key))
        if value:
            return value

    metadata = question.get("metadata")
    if isinstance(metadata, dict):
        for key in ("reference_solution", "solution", "answer"):
            value = _as_text(metadata.get(key))
            if value:
                return value
    return ""


def _question_rubric(question: Dict[str, Any]) -> str:
    for key in ("rubric", "marking_scheme", "marking_criteria", "explanation"):
        value = _as_text(question.get(key))
        if value:
            return value

    metadata = question.get("metadata")
    if isinstance(metadata, dict):
        for key in ("rubric", "marking_scheme", "marking_criteria"):
            value = _as_text(metadata.get(key))
            if value:
                return value
    return ""


def _answer_mapping_rank(mapping: Dict[str, Any]) -> tuple:
    """Choose the strongest reviewed worked-answer mapping deterministically."""
    source = _as_text(mapping.get("source")).lower()
    strategy = _as_text(mapping.get("mapping_strategy")).lower()
    source_rank = 0
    if source == "manual_answer_segmentation":
        source_rank = 40
    elif source == "answer_sheet_full_ocr":
        source_rank = 30
    elif source in {"answer_sheet", "uploaded_answer_sheet", "upload"}:
        source_rank = 25
    elif source == "ai_generated" or strategy == "ai_generated_solution":
        source_rank = 10
    review_status = _as_text(mapping.get("review_status")).lower()
    review_rank = 10 if review_status in {"accepted", "trusted"} else 0
    if review_status == "rejected":
        review_rank -= 20
    try:
        confidence_rank = float(mapping.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence_rank = 0.0
    return (source_rank, review_rank, confidence_rank)


def _approved_pcr_mapping(mapping: Dict[str, Any]) -> bool:
    """Whether a reviewed mapping is safe to freeze into a PCR marking key."""
    return (
        _as_text(mapping.get("review_status")).lower() in {"accepted", "trusted"}
        and not bool(mapping.get("manual_review_required"))
        and bool(_as_text(mapping.get("final_answer_text")) or _as_text(mapping.get("answer_text")))
    )


async def materialize_pcr_marking_plan(
    tenant_db: Any,
    *,
    document_id: str,
    questions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Make a self-contained PCR marking plan from reviewed authoring data.

    Worked-answer mappings deliberately remain separate from editable OCR
    questions while content is being reviewed.  A finalized paper version
    cannot depend on that mutable lookup, so only explicitly accepted/trusted
    mappings are copied into the in-memory questions passed to the immutable
    snapshot.  The original authoring questions are never mutated.
    """
    mappings = await tenant_db["answer_question_mappings"].find(
        {"document_id": document_id}
    ).to_list(length=10000)
    approved_by_question: Dict[str, Dict[str, Any]] = {}
    for mapping in mappings:
        if not _approved_pcr_mapping(mapping):
            continue
        question_id = _as_text(
            mapping.get("question_id") or mapping.get("question_region_id")
        )
        if not question_id:
            continue
        current = approved_by_question.get(question_id)
        if current is None or _answer_mapping_rank(mapping) > _answer_mapping_rank(current):
            approved_by_question[question_id] = mapping

    materialized: List[Dict[str, Any]] = []
    direct_solution_count = 0
    mapped_solution_count = 0
    for question in questions:
        merged = copy.deepcopy(question)
        direct_solution = (
            merged.get("reference_solution")
            or merged.get("solution")
            or merged.get("answer")
            or merged.get("correct_answer")
            or merged.get("correctAnswer")
            or merged.get("final_answer_text")
        )
        if _as_text(direct_solution):
            direct_solution_count += 1
            materialized.append(merged)
            continue

        question_id = _source_question_id(merged)
        mapping = approved_by_question.get(question_id)
        if mapping is not None:
            reference_solution = _as_text(
                mapping.get("final_answer_text") or mapping.get("answer_text")
            )
            if reference_solution:
                merged["reference_solution"] = reference_solution
                merged["marking_plan_source"] = "approved_answer_mapping"
                merged["marking_plan_mapping_id"] = _as_text(mapping.get("mapping_id"))
                mapped_solution_count += 1
        materialized.append(merged)

    return materialized, {
        "approved_mapping_candidates": len(approved_by_question),
        "questions_using_direct_solution": direct_solution_count,
        "questions_using_approved_mapping": mapped_solution_count,
    }


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _content_hash(document: Dict[str, Any], questions: Iterable[Dict[str, Any]]) -> str:
    """Hash only fields that determine the exam paper and marking outcome."""
    normalized_questions: List[Dict[str, Any]] = []
    for position, question in enumerate(questions, start=1):
        normalized_questions.append(
            {
                "position": position,
                "source_question_id": _source_question_id(question),
                "question_text": _question_text(question),
                "question_type": _as_text(question.get("question_type")),
                "subject": _as_text(question.get("subject")) or _as_text(document.get("subject")),
                "marks": _question_marks(question),
                "rubric": _question_rubric(question),
                "reference_solution": _question_reference_solution(question),
                "expects_diagram": bool(question.get("has_diagram") or question.get("expects_diagram")),
            }
        )
    payload = {
        "document_id": _as_text(document.get("document_id")),
        "exam_mode": _as_text(document.get("exam_mode")),
        "title": _as_text(document.get("title")),
        "subject": _as_text(document.get("subject")),
        "questions": normalized_questions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_pcr_questions(questions: Iterable[Dict[str, Any]]) -> List[str]:
    """Return human-readable readiness errors for a PCR marking paper.

    OCR extraction is not a marking plan.  Each question must be identifiable,
    have text and marks, and contain either an approved reference solution or
    an explicit rubric before the paper becomes immutable.
    """
    errors: List[str] = []
    for position, question in enumerate(questions, start=1):
        label = _source_question_id(question) or str(position)
        if not _source_question_id(question):
            errors.append(f"Q {label}: missing stable question id")
        if not _question_text(question):
            errors.append(f"Q {label}: missing question text")
        if _question_marks(question) is None:
            errors.append(f"Q {label}: assign marks greater than zero")
        if not _question_reference_solution(question) and not _question_rubric(question):
            errors.append(
                f"Q {label}: add an approved reference solution or marking rubric"
            )
    return errors


async def ensure_indexes(tenant_db: Any) -> None:
    """Create idempotent indexes for immutable paper snapshots."""
    await tenant_db[PAPER_VERSIONS_COLLECTION].create_index(
        "paper_version_id", unique=True, name="uniq_paper_version_id"
    )
    await tenant_db[PAPER_VERSIONS_COLLECTION].create_index(
        "document_id", name="idx_paper_document_id"
    )
    await tenant_db[PAPER_QUESTIONS_COLLECTION].create_index(
        [("paper_version_id", 1), ("source_question_id", 1)],
        unique=True,
        name="uniq_paper_question",
    )


async def create_paper_snapshot(
    tenant_db: Any,
    document: Dict[str, Any],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Persist one immutable version of a reviewed document.

    The operation is idempotent for the same immutable content hash.  It is
    intentionally performed before a document is marked finalized, so a
    partial snapshot failure cannot leave a supposedly frozen paper without a
    usable marking snapshot.
    """
    document_id = _as_text(document.get("document_id"))
    if not document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot snapshot a paper without document_id",
        )

    content_hash = _content_hash(document, questions)
    paper_version_id = f"paper-{document_id}-{content_hash[:16]}"
    versions = tenant_db[PAPER_VERSIONS_COLLECTION]
    paper_questions = tenant_db[PAPER_QUESTIONS_COLLECTION]
    await ensure_indexes(tenant_db)

    existing = await versions.find_one({"paper_version_id": paper_version_id})
    if existing is not None:
        return existing

    now = datetime.now(timezone.utc)
    question_docs: List[Dict[str, Any]] = []
    for position, question in enumerate(questions, start=1):
        source_question_id = _source_question_id(question)
        clean_question = copy.deepcopy(question)
        clean_question.pop("_id", None)
        # Some older authoring records rely on Mongo's _id rather than a
        # separate ``id`` field.  Preserve that identity explicitly inside
        # the immutable payload so downstream DCR/PCR adapters never see an
        # anonymous question after _id is removed.
        if source_question_id and not clean_question.get("id") and not clean_question.get("question_id"):
            clean_question["id"] = source_question_id
        question_docs.append(
            {
                "paper_version_id": paper_version_id,
                "document_id": document_id,
                "source_question_id": source_question_id,
                "position": position,
                "question": clean_question,
                "created_at": now,
            }
        )

    inserted_question_ids: List[Any] = []
    try:
        if question_docs:
            result = await paper_questions.insert_many(question_docs, ordered=True)
            inserted_question_ids = list(result.inserted_ids)

        version_doc = {
            "paper_version_id": paper_version_id,
            "document_id": document_id,
            "exam_mode": _as_text(document.get("exam_mode")),
            "title": _as_text(document.get("title")),
            "subject": _as_text(document.get("subject")),
            "admin_id": _as_text(document.get("admin_id")) or None,
            "teacher_ids": [str(item) for item in (document.get("teacher_ids") or [])],
            "content_hash": content_hash,
            "question_count": len(question_docs),
            "created_at": now,
            "source_document_finalized_at": document.get("exam_finalized_at"),
        }
        await versions.insert_one(version_doc)
        return version_doc
    except Exception:
        if inserted_question_ids:
            await paper_questions.delete_many({"_id": {"$in": inserted_question_ids}})
        raise


async def load_or_create_paper_snapshot(
    tenant_db: Any,
    document: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a final paper snapshot, migrating legacy finalized papers once."""
    version_id = _as_text(document.get("exam_paper_version_id"))
    versions = tenant_db[PAPER_VERSIONS_COLLECTION]
    paper_questions = tenant_db[PAPER_QUESTIONS_COLLECTION]

    version = await versions.find_one({"paper_version_id": version_id}) if version_id else None
    if version is None:
        document_id = _as_text(document.get("document_id"))
        cursor = tenant_db["questions"].find({"document_id": document_id})
        source_questions = await cursor.to_list(length=10000)
        if not source_questions:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Finalized paper has no source questions to snapshot",
            )
        snapshot_source_questions = source_questions
        marking_plan_summary: Optional[Dict[str, int]] = None
        if document.get("exam_mode") == "pcr":
            # Legacy finalized documents may predate immutable snapshots but
            # already have reviewed mappings.  Apply the same safe merge used
            # at finalization rather than falsely declaring those papers
            # unusable or, worse, snapshotting unreviewed OCR output.
            snapshot_source_questions, marking_plan_summary = await materialize_pcr_marking_plan(
                tenant_db,
                document_id=document_id,
                questions=source_questions,
            )
            errors = validate_pcr_questions(snapshot_source_questions)
            if errors:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "message": "PCR paper is missing marking information",
                        "errors": errors[:50],
                    },
                )
        version = await create_paper_snapshot(tenant_db, document, snapshot_source_questions)
        snapshot_update: Dict[str, Any] = {
            "exam_paper_version_id": version["paper_version_id"],
            "exam_content_hash": version["content_hash"],
            "exam_snapshot_backfilled_at": datetime.now(timezone.utc),
        }
        if marking_plan_summary is not None:
            snapshot_update["exam_snapshot_marking_plan"] = marking_plan_summary
        await tenant_db["documents"].update_one(
            {"document_id": document_id},
            {"$set": snapshot_update},
        )

    cursor = paper_questions.find({"paper_version_id": version["paper_version_id"]}).sort("position", 1)
    snapshot_questions = await cursor.to_list(length=10000)
    if not snapshot_questions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Paper snapshot contains no questions",
        )
    return version, snapshot_questions


def session_question_id(exam_id: str, source_question_id: str) -> str:
    """Return a globally unique metadata id for one session question."""
    return f"{exam_id}::{source_question_id}"


async def snapshot_paper_to_session(
    tenant_db: Any,
    *,
    exam_id: str,
    paper_version: Dict[str, Any],
    snapshot_questions: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Materialize a paper version as immutable metadata for one session.

    ``evalpen_questions.question_id`` is globally unique in the existing PCR
    repository.  Session-scoping it here avoids overwriting metadata when the
    same paper is conducted more than once.
    """
    exam_type = _as_text(paper_version.get("exam_mode"))
    raw_questions = [dict(item.get("question") or {}) for item in snapshot_questions]

    if exam_type == "pcr":
        from api.v1._exampen_imports import load_exampen

        adapter = load_exampen("pcr.metadata_adapter")
        storage = load_exampen("pcr.storage")
        repo = storage.QuestionRepository(tenant_db)
        solutions = storage.SolutionRepository(tenant_db)
        await repo.ensure_indexes()
        await solutions.ensure_indexes()

        docs: List[Dict[str, Any]] = []
        for snapshot_question, raw_question in zip(snapshot_questions, raw_questions):
            source_id = _as_text(snapshot_question.get("source_question_id"))
            pcr_doc = adapter.adapt_question_to_pcr(
                raw_question,
                exam_id=exam_id,
                default_subject=paper_version.get("subject"),
            )
            pcr_doc["question_id"] = session_question_id(exam_id, source_id)
            pcr_doc["source_question_id"] = source_id
            # The immutable snapshot position is the authoritative Q-number
            # for a conducted sitting.  PCR response markers use Q1/Q2/etc.,
            # while the session metadata uses canonical UUID-based IDs.
            # Keeping both prevents the old synthetic ``exam_Q1`` mapping
            # from breaking real session evaluation.
            pcr_doc["question_number"] = int(snapshot_question.get("position") or 0) or None
            pcr_doc["paper_version_id"] = paper_version["paper_version_id"]
            pcr_doc["immutable"] = True
            pcr_doc["question_text"] = _question_text(raw_question)
            pcr_doc["reference_solution"] = _question_reference_solution(raw_question) or None
            pcr_doc["rubric"] = _question_rubric(raw_question) or pcr_doc.get("rubric")
            marks = _question_marks(raw_question)
            if marks is not None:
                pcr_doc["max_marks"] = marks
            docs.append(pcr_doc)

        inserted, updated = await repo.upsert_questions_bulk(docs)

        for doc in docs:
            reference_solution = _as_text(doc.get("reference_solution"))
            if not reference_solution:
                continue
            latest = await solutions.get_latest_solution(doc["question_id"])
            if latest and latest.get("reference_solution") == reference_solution and latest.get("solution_source") == "teacher":
                continue
            await solutions.upsert_solution(
                {
                    "question_id": doc["question_id"],
                    "reference_solution": reference_solution,
                    "solution_source": "teacher",
                    "model_used": None,
                    "paper_version_id": paper_version["paper_version_id"],
                    "created_at": datetime.now(timezone.utc),
                }
            )
        return {"questions_inserted": inserted, "questions_updated": updated}

    if exam_type == "dcr":
        from api.v1.tutor_async import sync_dcr_answer_keys

        result = await sync_dcr_answer_keys(
            tenant_db=tenant_db,
            questions=raw_questions,
            exam_id=exam_id,
            exam_doc={
                "document_id": paper_version.get("document_id"),
                "paper_version_id": paper_version["paper_version_id"],
            },
        )
        return {"answer_keys_upserted": int((result or {}).get("upserted", 0))}

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Unsupported paper exam mode: {exam_type or 'missing'}",
    )


async def delete_session_snapshot(tenant_db: Any, exam_id: str, exam_type: str) -> None:
    """Best-effort cleanup when a session cannot be created."""
    if exam_type == "pcr":
        await tenant_db["evalpen_questions"].delete_many({"exam_id": exam_id})
        await tenant_db["evalpen_solutions"].delete_many({"question_id": {"$regex": f"^{exam_id}::"}})
    elif exam_type == "dcr":
        await tenant_db["exampen_answer_keys"].delete_many({"exam_id": exam_id})
