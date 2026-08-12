"""Immutable paper snapshots for conducted ExamPen sessions.

The content-management document is an authoring object.  A conducted exam
needs an immutable version of that paper so that later changes, retries, or a
second sitting cannot change the questions that were used to mark a student.

This module owns the bridge between a finalized ``documents`` record and the
session-scoped ExamPen metadata consumed by DCR/PCR engines.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import logging
import os
import uuid
from datetime import date, datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import HTTPException, status
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from config_async import settings
from core.upload_security.storage import safe_storage_segment
from services.answer_mapping_contract import normalize_answer_label
from services.question_marking_contract import (
    normalize_question_penalty,
    parse_question_penalty,
)
from utils.s3_storage import (
    PrivateObjectStorageError,
    download_file,
    download_private_object,
    upload_private_object,
)

logger = logging.getLogger(__name__)

PAPER_VERSIONS_COLLECTION = "exampen_paper_versions"
PAPER_QUESTIONS_COLLECTION = "exampen_paper_questions"
PAPER_ASSETS_COLLECTION = "exampen_paper_assets"
PAPER_LAYOUT_SCHEMA_VERSION = 1
CANONICAL_PAPER_ASSET_PREFIX = "private/exampen/paper-assets"


class CanonicalPaperAssetError(RuntimeError):
    """The immutable paper asset is missing, altered, or not durable.

    Retrying an AI worker cannot restore an asset that was never materialized,
    so the durable PCR job scheduler must treat this as an operator-repair
    condition rather than a transient provider failure.
    """

    retryable = False


def _marking_policy_module() -> Any:
    """Load PCR policy helpers without making the hyphenated package a normal import."""

    return importlib.import_module("exam-conductor.pcr.marking_policy")


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


def _asset_reference(asset: Dict[str, Any], *, role: str) -> Dict[str, Any]:
    """Return the only asset fields a paper snapshot is allowed to consume."""

    return {
        "asset_id": _as_text(asset.get("asset_id")),
        "storage_uri": _as_text(asset.get("storage_uri")),
        "sha256": _as_text(asset.get("sha256")),
        "size_bytes": int(asset.get("size_bytes") or 0),
        "filename": _as_text(asset.get("filename")),
        "content_type": _as_text(asset.get("content_type")) or "application/pdf",
        "role": role,
    }


def _paper_asset_id(sha256: str) -> str:
    return f"paper-asset-sha256-{sha256.lower()}"


def _paper_asset_tenant_segment(tenant_db: Any) -> str:
    return safe_storage_segment(getattr(tenant_db, "name", None), fallback="tenant")


def _authoring_asset_path(source_path: str) -> Path:
    """Resolve a legacy authoring path without permitting arbitrary reads."""

    backend_root = Path(__file__).resolve().parents[1]
    candidate = Path(source_path)
    if not candidate.is_absolute():
        candidate = backend_root / candidate
    candidate = candidate.resolve(strict=False)
    roots = [
        Path(settings.UPLOAD_PRIVATE_LOCAL_DIR).resolve(strict=False),
        (backend_root / "uploads").resolve(strict=False),
    ]
    if not any(root == candidate or root in candidate.parents for root in roots):
        raise CanonicalPaperAssetError("Paper source is outside approved upload storage")
    return candidate


async def _read_authoring_asset(source_path: str, *, expected_sha256: str) -> bytes:
    """Read a scanned authoring upload once, before it becomes an exam asset."""

    path = _as_text(source_path)
    if not path:
        raise CanonicalPaperAssetError("Paper source path is missing")
    if path.startswith("s3://"):
        data = await download_file(path)
    else:
        candidate = _authoring_asset_path(path)
        if not candidate.is_file():
            raise CanonicalPaperAssetError("Paper source file is unavailable")
        data = await asyncio.to_thread(candidate.read_bytes)
    if not data:
        raise CanonicalPaperAssetError("Paper source file is empty or unavailable")
    actual_sha256 = hashlib.sha256(data).hexdigest()
    expected = _as_text(expected_sha256).lower()
    if expected and actual_sha256 != expected:
        raise CanonicalPaperAssetError("Paper source failed integrity verification")
    return data


async def load_canonical_paper_asset(asset: Dict[str, Any]) -> bytes:
    """Load a snapshot-pinned paper asset from private object storage only."""

    storage_uri = _as_text(asset.get("storage_uri"))
    expected_sha256 = _as_text(asset.get("sha256")).lower()
    if not storage_uri or not expected_sha256:
        raise CanonicalPaperAssetError("Immutable paper asset manifest is incomplete")
    try:
        data = await download_private_object(
            storage_uri,
            allowed_key_prefix=CANONICAL_PAPER_ASSET_PREFIX,
            max_bytes=50 * 1024 * 1024,
        )
    except PrivateObjectStorageError as exc:
        raise CanonicalPaperAssetError("Immutable paper asset could not be loaded") from exc
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise CanonicalPaperAssetError("Immutable paper asset failed integrity verification")
    return data


async def _materialize_paper_asset(
    tenant_db: Any,
    *,
    source_path: str,
    sha256: str,
    filename: str,
    role: str,
) -> Dict[str, Any]:
    """Copy a verified authoring PDF to content-addressed private object storage."""

    expected_sha256 = _as_text(sha256).lower()
    if not expected_sha256:
        raise CanonicalPaperAssetError(f"{role.replace('_', ' ').capitalize()} SHA-256 is missing")
    assets = tenant_db[PAPER_ASSETS_COLLECTION]
    asset_id = _paper_asset_id(expected_sha256)
    existing = await assets.find_one({"asset_id": asset_id})
    if existing and _as_text(existing.get("sha256")).lower() == expected_sha256:
        # A manifest is not trusted merely because it exists. Verify that the
        # pinned object can still be read before reusing it for a new session.
        await load_canonical_paper_asset(existing)
        return _asset_reference(existing, role=role)

    data = await _read_authoring_asset(source_path, expected_sha256=expected_sha256)
    tenant = _paper_asset_tenant_segment(tenant_db)
    object_key = f"{CANONICAL_PAPER_ASSET_PREFIX}/{tenant}/{expected_sha256}.pdf"
    try:
        storage_uri = await upload_private_object(
            data,
            object_key=object_key,
            content_type="application/pdf",
            metadata={
                "asset_id": asset_id,
                "sha256": expected_sha256,
                "role": role,
                "source": "exampen_paper_snapshot",
            },
        )
    except PrivateObjectStorageError as exc:
        raise CanonicalPaperAssetError("Could not store immutable paper asset") from exc

    asset = {
        "asset_id": asset_id,
        "storage_uri": storage_uri,
        "sha256": expected_sha256,
        "size_bytes": len(data),
        "filename": safe_storage_segment(filename, fallback=f"{role}.pdf"),
        "content_type": "application/pdf",
        "created_at": datetime.now(timezone.utc),
        "source": "exampen_paper_snapshot",
    }
    await assets.update_one(
        {"asset_id": asset_id},
        {"$setOnInsert": asset},
        upsert=True,
    )
    stored = await assets.find_one({"asset_id": asset_id})
    if not stored:
        raise CanonicalPaperAssetError("Immutable paper asset manifest was not persisted")
    return _asset_reference(stored, role=role)


async def materialize_paper_assets(tenant_db: Any, document: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Freeze durable question/solution PDFs before a PCR paper is finalized."""

    await ensure_indexes(tenant_db)
    question_paper = await _materialize_paper_asset(
        tenant_db,
        source_path=_as_text(document.get("file_path")),
        sha256=_as_text(document.get("sha256")),
        filename=_as_text(document.get("filename")) or "question-paper.pdf",
        role="question_paper",
    )
    assets: Dict[str, Dict[str, Any]] = {"question_paper": question_paper}
    if _as_text(document.get("answer_sheet_path")):
        assets["teacher_solution"] = await _materialize_paper_asset(
            tenant_db,
            source_path=_as_text(document.get("answer_sheet_path")),
            sha256=_as_text(document.get("answer_sheet_sha256")),
            filename=_as_text(document.get("answer_sheet_filename")) or "teacher-solution.pdf",
            role="teacher_solution",
        )
    return assets


async def migrate_legacy_paper_snapshot_assets(
    tenant_db: Any,
    document: Dict[str, Any],
    *,
    paper_version_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One-time, hash-verified migration for a finalized legacy PCR snapshot.

    This is intentionally an explicit operator action. A grading worker must
    never reconstruct an allegedly immutable paper from an arbitrary mutable
    path while evaluating a student submission.
    """

    document_id = _as_text(document.get("document_id"))
    if not document_id:
        raise CanonicalPaperAssetError("Cannot migrate a paper without document_id")
    versions = tenant_db[PAPER_VERSIONS_COLLECTION]
    query: Dict[str, Any] = {"document_id": document_id}
    if paper_version_id:
        query["paper_version_id"] = _as_text(paper_version_id)
    version = await versions.find_one(query)
    if not version:
        raise CanonicalPaperAssetError("Finalized paper snapshot was not found")
    existing_assets = dict(version.get("paper_assets") or {})
    if existing_assets.get("question_paper"):
        await load_canonical_paper_asset(dict(existing_assets["question_paper"]))
        return version

    # A historical migration is safe only when the currently reachable source
    # still has the exact bytes that the finalized snapshot recorded. Never
    # silently bind an old exam to a later-edited authoring document.
    context = dict(version.get("paper_context") or {})
    expected_question_sha = _as_text(context.get("question_paper_sha256")).lower()
    actual_question_sha = _as_text(document.get("sha256")).lower()
    if not expected_question_sha or expected_question_sha != actual_question_sha:
        raise CanonicalPaperAssetError(
            "Legacy paper migration cannot prove the question-paper SHA-256"
        )
    expected_solution_sha = _as_text(context.get("teacher_solution_sha256")).lower()
    if expected_solution_sha:
        actual_solution_sha = _as_text(document.get("answer_sheet_sha256")).lower()
        if expected_solution_sha != actual_solution_sha:
            raise CanonicalPaperAssetError(
                "Legacy paper migration cannot prove the teacher-solution SHA-256"
            )

    assets = await materialize_paper_assets(tenant_db, document)
    context["question_paper_asset_id"] = assets["question_paper"]["asset_id"]
    if assets.get("teacher_solution"):
        context["teacher_solution_asset_id"] = assets["teacher_solution"]["asset_id"]
    now = datetime.now(timezone.utc)
    updated = await versions.find_one_and_update(
        {
            "paper_version_id": version["paper_version_id"],
            "$or": [
                {"paper_assets.question_paper": {"$exists": False}},
                {"paper_assets.question_paper": None},
            ],
        },
        {
            "$set": {
                "paper_assets": assets,
                "paper_context": context,
                "paper_assets_migrated_at": now,
                "paper_assets_migration": "content-addressed-private-s3-v1",
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated:
        return updated
    current = await versions.find_one({"paper_version_id": version["paper_version_id"]})
    if current and dict(current.get("paper_assets") or {}).get("question_paper"):
        await load_canonical_paper_asset(dict(current["paper_assets"]["question_paper"]))
        return current
    raise CanonicalPaperAssetError("Paper asset migration lost ownership")


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


def _is_objective_question(question: Dict[str, Any]) -> bool:
    return _as_text(question.get("question_type")).lower() in {
        "mcq",
        "objective",
        "integer",
    }


def _objective_option_labels(question: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    enhanced_options = question.get("enhanced_options")
    if isinstance(enhanced_options, list):
        for index, option in enumerate(enhanced_options):
            if not isinstance(option, dict):
                continue
            content = _as_text(
                option.get("content") or option.get("text") or option.get("value")
            )
            if not content:
                continue
            label = normalize_answer_label(
                option.get("label") or option.get("key") or option.get("id")
            ) or chr(ord("A") + index)
            if label not in labels:
                labels.append(label)
    options = question.get("options")
    if isinstance(options, list):
        for index, option in enumerate(options):
            content = (
                _as_text(
                    option.get("content")
                    or option.get("text")
                    or option.get("value")
                )
                if isinstance(option, dict)
                else _as_text(option)
            )
            if content:
                label = chr(ord("A") + index)
                if label not in labels:
                    labels.append(label)
    return labels


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
        raw_value = question.get(key)
        # Structured criteria are a list of records, not a legacy free-text
        # rubric.  Do not turn their Python representation into prompt text.
        if isinstance(raw_value, (list, dict)):
            continue
        value = _as_text(raw_value)
        if value:
            return value

    metadata = question.get("metadata")
    if isinstance(metadata, dict):
        for key in ("rubric", "marking_scheme", "marking_criteria"):
            raw_value = metadata.get(key)
            if isinstance(raw_value, (list, dict)):
                continue
            value = _as_text(raw_value)
            if value:
                return value
    return ""


def _question_marking_criteria(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract structured criteria while preserving legacy free-text rubrics."""

    raw_value = question.get("marking_criteria")
    if raw_value is None and isinstance(question.get("metadata"), dict):
        raw_value = question["metadata"].get("marking_criteria")
    if raw_value is None:
        return []
    # Older content can use ``marking_criteria`` as a text alias for rubric.
    # It becomes structured only once the value is an actual list/JSON list.
    if isinstance(raw_value, str) and not raw_value.lstrip().startswith("["):
        return []
    try:
        return _marking_policy_module().normalize_marking_criteria(
            raw_value,
            assign_missing_ids=False,
        )
    except ValueError:
        return []


def _raw_question_assessment_units(question: Dict[str, Any]) -> Any:
    raw_value = question.get("assessment_units")
    if raw_value is None and isinstance(question.get("metadata"), dict):
        raw_value = question["metadata"].get("assessment_units")
    return raw_value


def _question_assessment_units(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_value = _raw_question_assessment_units(question)
    if raw_value is None:
        return []
    try:
        return _marking_policy_module().normalize_assessment_units(
            raw_value,
            assign_missing_ids=False,
        )
    except ValueError:
        return []


def _question_method_policy(question: Dict[str, Any]) -> Dict[str, Any]:
    """Return the explicit method contract, defaulting to any valid method."""

    raw_value = question.get("method_policy")
    if raw_value is None and isinstance(question.get("metadata"), dict):
        raw_value = question["metadata"].get("method_policy")
    return _marking_policy_module().normalize_method_policy(raw_value)


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


def _content_hash(
    document: Dict[str, Any],
    questions: Iterable[Dict[str, Any]],
    question_layout: Optional[List[Dict[str, Any]]] = None,
    paper_context: Optional[Dict[str, Any]] = None,
) -> str:
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
                "marking_criteria": _question_marking_criteria(question),
                "assessment_units": _question_assessment_units(question),
                "options": copy.deepcopy(question.get("options") or []),
                "enhanced_options": copy.deepcopy(
                    question.get("enhanced_options") or []
                ),
                "correct_answer": normalize_answer_label(
                    question.get("correct_answer")
                    or question.get("correctAnswer")
                ),
                "penalty_marks": normalize_question_penalty(
                    question.get("penalty", question.get("penalty_marks")),
                    question_type=question.get("question_type"),
                    document_question_type=document.get("question_type"),
                ),
                "expects_diagram": bool(question.get("has_diagram") or question.get("expects_diagram")),
            }
        )
    payload = {
        "document_id": _as_text(document.get("document_id")),
        "exam_mode": _as_text(document.get("exam_mode")),
        "title": _as_text(document.get("title")),
        "subject": _as_text(document.get("subject")),
        "pcr_marking_policy": document.get("pcr_marking_policy"),
        "question_paper_sha256": _as_text(document.get("sha256")) or None,
        "teacher_solution_sha256": _as_text(document.get("answer_sheet_sha256")) or None,
        "questions": normalized_questions,
        "question_layout": question_layout or [],
        "paper_context": paper_context or {},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _question_print_order(question: Dict[str, Any], fallback: int) -> int:
    for key in ("question_number", "extraction_order"):
        raw_value = question.get(key)
        if raw_value in (None, "") or isinstance(raw_value, bool):
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return fallback


def _regions_from_reviewed_ocr_anchors(
    document: Dict[str, Any],
    questions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build non-overlapping page bands from trusted OCR question anchors.

    Manual boxes are authoritative when present.  For papers whose OCR quality
    gate explicitly says manual segmentation is unnecessary, the persisted
    question anchors are the reviewed layout evidence and must remain usable at
    finalization.  The fallback is deliberately fail-closed: every immutable
    question must have exactly one numbered anchor on the same source page.
    """
    if document.get("ocr_manual_segmentation_recommended") is not False:
        return [], [
            "No reviewed question regions are saved. Segment every printed question before finalizing."
        ]

    layout_summary = document.get("ocr_layout_summary")
    raw_pages = layout_summary.get("pages") if isinstance(layout_summary, dict) else None
    if not isinstance(raw_pages, list) or not raw_pages:
        return [], [
            "Question OCR has no reviewed page anchors. Reprocess OCR or segment the paper manually."
        ]

    anchors_by_number: Dict[int, Dict[str, Any]] = {}
    errors: List[str] = []
    for page_index, page in enumerate(raw_pages, start=1):
        if not isinstance(page, dict):
            errors.append(f"OCR layout page {page_index}: invalid page metadata")
            continue
        try:
            page_number = int(page.get("page") or page_index)
        except (TypeError, ValueError):
            errors.append(f"OCR layout page {page_index}: invalid page number")
            continue
        raw_anchors = page.get("question_anchors") or []
        if not isinstance(raw_anchors, list):
            errors.append(f"OCR layout page {page_number}: invalid question anchors")
            continue
        for anchor_index, anchor in enumerate(raw_anchors, start=1):
            if not isinstance(anchor, dict):
                errors.append(f"OCR layout page {page_number}, anchor {anchor_index}: invalid anchor")
                continue
            try:
                number = int(str(anchor.get("number") or "").strip())
                y = float(anchor.get("y"))
            except (TypeError, ValueError):
                errors.append(f"OCR layout page {page_number}, anchor {anchor_index}: invalid number or position")
                continue
            if number < 1 or y < 0:
                errors.append(f"OCR layout page {page_number}, anchor {anchor_index}: invalid number or position")
                continue
            if number in anchors_by_number:
                errors.append(f"OCR layout: duplicate question anchor {number}")
                continue
            anchors_by_number[number] = {
                "number": number,
                "page_number": page_number,
                "y": y,
                "page_height": page.get("page_height") or page.get("height") or page.get("height_points"),
            }

    question_by_number: Dict[int, Dict[str, Any]] = {}
    for position, question in enumerate(questions, start=1):
        number = _question_print_order(question, position)
        if number in question_by_number:
            errors.append(f"Question {number}: duplicate printed order")
            continue
        question_by_number[number] = question

    expected_numbers = set(question_by_number)
    anchor_numbers = set(anchors_by_number)
    for number in sorted(expected_numbers - anchor_numbers):
        errors.append(f"Question {number}: no reviewed OCR page anchor")
    for number in sorted(anchor_numbers - expected_numbers):
        errors.append(f"OCR anchor {number}: has no reviewed question record")
    if errors:
        return [], errors

    anchors_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for number in sorted(expected_numbers):
        question = question_by_number[number]
        anchor = anchors_by_number[number]
        try:
            question_page = int(question.get("page_number") or 0)
        except (TypeError, ValueError):
            question_page = 0
        if question_page > 0 and question_page != anchor["page_number"]:
            errors.append(
                f"Question {number}: OCR question page {question_page} conflicts with anchor page {anchor['page_number']}"
            )
        anchors_by_page.setdefault(anchor["page_number"], []).append(anchor)
    if errors:
        return [], errors

    regions: List[Dict[str, Any]] = []
    for page_number, page_anchors in sorted(anchors_by_page.items()):
        page_anchors.sort(key=lambda item: (item["y"], item["number"]))
        raw_height = next((item.get("page_height") for item in page_anchors if item.get("page_height")), None)
        try:
            page_height = float(raw_height) if raw_height is not None else 0.0
        except (TypeError, ValueError):
            page_height = 0.0
        last_y = page_anchors[-1]["y"]
        if page_height <= last_y:
            gaps = [
                right["y"] - left["y"]
                for left, right in zip(page_anchors, page_anchors[1:])
                if right["y"] > left["y"]
            ]
            trailing_span = sorted(gaps)[len(gaps) // 2] if gaps else max(last_y, 1.0)
            page_height = max(last_y + trailing_span, last_y * 1.05, 1.0)

        anchor_percent = [min(100.0, max(0.0, item["y"] / page_height * 100.0)) for item in page_anchors]
        boundaries = [0.0]
        boundaries.extend(
            (left + right) / 2.0
            for left, right in zip(anchor_percent, anchor_percent[1:])
        )
        boundaries.append(100.0)

        for index, anchor in enumerate(page_anchors):
            number = anchor["number"]
            question_id = _source_question_id(question_by_number[number])
            top = boundaries[index]
            bottom = boundaries[index + 1]
            regions.append(
                {
                    "id": question_id,
                    "pageNumber": page_number,
                    "x": 0.0,
                    "y": round(top, 6),
                    "width": 100.0,
                    "height": round(bottom - top, 6),
                    "order": number,
                    "label": f"Q{number}",
                    "ocrStatus": "completed",
                    "manualReviewRequired": False,
                    "layoutSource": "reviewed_ocr_anchor",
                }
            )
    return regions, []


def build_question_layout(
    document: Dict[str, Any],
    questions: Iterable[Dict[str, Any]],
    regions_document: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate and normalize the reviewed question-to-page layout.

    A question list alone cannot tell the answer mapper where a printed
    question belongs.  New PCR papers therefore freeze one non-overlapping,
    page-scoped source region for every immutable question.
    """
    question_list = list(questions)
    errors: List[str] = []
    question_ids: List[str] = []
    seen_question_ids: set[str] = set()
    for position, question in enumerate(question_list, start=1):
        question_id = _source_question_id(question)
        if not question_id:
            continue
        if question_id in seen_question_ids:
            errors.append(f"Q {position}: duplicate stable question id {question_id}")
        seen_question_ids.add(question_id)
        question_ids.append(question_id)

    raw_regions = list((regions_document or {}).get("regions") or [])
    if not raw_regions:
        raw_regions, anchor_errors = _regions_from_reviewed_ocr_anchors(document, question_list)
        if anchor_errors:
            return [], anchor_errors

    excluded_pages = {
        int(page)
        for page in ((regions_document or {}).get("excluded_pages") or [])
        if isinstance(page, int) and page > 0
    }
    try:
        page_count = int(document.get("pages_count") or 0)
    except (TypeError, ValueError):
        page_count = 0

    normalized_by_id: Dict[str, Dict[str, Any]] = {}
    seen_orders: set[int] = set()
    for index, region in enumerate(raw_regions, start=1):
        region_id = _as_text(region.get("id"))
        if not region_id:
            errors.append(f"Region {index}: missing stable region id")
            continue
        if region_id in normalized_by_id:
            errors.append(f"Region {region_id}: duplicate region id")
            continue
        try:
            page_number = int(region.get("pageNumber"))
            order = int(region.get("order"))
            x = float(region.get("x"))
            y = float(region.get("y"))
            width = float(region.get("width"))
            height = float(region.get("height"))
        except (TypeError, ValueError):
            errors.append(f"Region {region_id}: invalid page, order, or bounding box")
            continue

        if page_number < 1 or (page_count and page_number > page_count):
            errors.append(f"Region {region_id}: page {page_number} is outside the source PDF")
        if page_number in excluded_pages:
            errors.append(f"Region {region_id}: points to excluded page {page_number}")
        if order < 1:
            errors.append(f"Region {region_id}: order must be greater than zero")
        elif order in seen_orders:
            errors.append(f"Region {region_id}: duplicate question order {order}")
        seen_orders.add(order)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            errors.append(f"Region {region_id}: bounding box must have positive in-page dimensions")
        if x + width > 100.0001 or y + height > 100.0001:
            errors.append(f"Region {region_id}: bounding box extends beyond the page")
        if _as_text(region.get("ocrStatus")).lower() != "completed":
            errors.append(f"Region {region_id}: OCR/review is not complete")
        if bool(region.get("manualReviewRequired")):
            errors.append(f"Region {region_id}: resolve its extraction review warning")

        normalized_by_id[region_id] = {
            "schema_version": PAPER_LAYOUT_SCHEMA_VERSION,
            "source_question_id": region_id,
            "source_region_id": region_id,
            "question_number": order,
            "page_number": page_number,
            "bbox_percent": {
                "x": round(x, 6),
                "y": round(y, 6),
                "width": round(width, 6),
                "height": round(height, 6),
            },
            "label": _as_text(region.get("label")) or f"Q{order}",
            "has_sub_questions": bool(region.get("hasSubQuestions")),
            "layout_source": _as_text(region.get("layoutSource")) or "manual_region",
        }

    question_id_set = set(question_ids)
    region_id_set = set(normalized_by_id)
    for question_id in sorted(question_id_set - region_id_set):
        errors.append(f"Question {question_id}: no saved source-page region")
    for region_id in sorted(region_id_set - question_id_set):
        errors.append(f"Region {region_id}: has no reviewed question record")

    expected_orders = list(range(1, len(question_list) + 1))
    actual_orders = sorted(
        item["question_number"]
        for region_id, item in normalized_by_id.items()
        if region_id in question_id_set
    )
    if actual_orders != expected_orders:
        errors.append(
            "Question region order must be unique and contiguous from 1 to "
            f"{len(question_list)}"
        )

    # Any material overlap makes page evidence ownership ambiguous.  Tiny
    # edge overlaps from drag rounding are tolerated (2% of the smaller box).
    comparable = [
        item for key, item in normalized_by_id.items() if key in question_id_set
    ]
    for left_index, left in enumerate(comparable):
        for right in comparable[left_index + 1 :]:
            if left["page_number"] != right["page_number"]:
                continue
            a = left["bbox_percent"]
            b = right["bbox_percent"]
            overlap_width = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
            overlap_height = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
            overlap_area = overlap_width * overlap_height
            smaller_area = min(a["width"] * a["height"], b["width"] * b["height"])
            if smaller_area > 0 and overlap_area / smaller_area > 0.02:
                errors.append(
                    f"Question regions {left['source_region_id']} and "
                    f"{right['source_region_id']} overlap on page {left['page_number']}"
                )

    layout = sorted(
        comparable,
        key=lambda item: (item["question_number"], item["page_number"]),
    )
    return layout, errors


def full_document_visual_contract(document: Dict[str, Any]) -> Dict[str, Any]:
    """Return the immutable full-document grading capability for one paper.

    Semantic ownership for camera/PDF copies belongs to the multimodal grader,
    which receives the original paper, teacher solution, frozen catalog, and
    complete student copy together. OCR regions remain useful review metadata
    but are not a prerequisite for that path.
    """

    enabled = os.getenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true").strip().lower()
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    model_id = (
        os.getenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "gpt-5.1").strip()
    )
    blockers: List[str] = []
    if enabled in {"0", "false", "no", "off"}:
        blockers.append("Full-document visual grading is disabled")
    if provider and provider != "openai":
        blockers.append("The configured AI provider does not support the canonical PDF grading path")
    if not model_id.lower().startswith(("gpt-5", "gpt-4.1", "gpt-4o")):
        blockers.append("The configured grading model does not support the canonical visual contract")
    if not _as_text(document.get("file_path")):
        blockers.append("The immutable question-paper asset is unavailable")

    return {
        # V2 freezes the global-to-local evidence-graph contract. Existing V1
        # paper versions remain valid and continue on their already locked
        # single-call grading contract; newly finalized papers use V2.
        "version": "canonical-full-document-visual-v2",
        "mode": "full_document_visual",
        "ready": not blockers,
        "model_id": model_id,
        "prompt_cache_scope": _as_text(document.get("document_id")),
        "question_paper_sha256": _as_text(document.get("sha256")) or None,
        "teacher_solution_sha256": _as_text(document.get("answer_sheet_sha256")) or None,
        "has_teacher_solution_asset": bool(_as_text(document.get("answer_sheet_path"))),
        "requires_question_regions": False,
        "blockers": blockers,
    }


def resolve_question_layout_for_finalization(
    document: Dict[str, Any],
    questions: Iterable[Dict[str, Any]],
    regions_document: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve the paper evidence strategy without letting OCR gate vision.

    Reviewed regions stay authoritative when available. If they are missing or
    ambiguous, a ready canonical full-document visual contract makes those
    layout errors advisory because the primary grader reads the complete source
    documents directly. Deployments that disable the visual path retain the
    strict region requirement for their legacy mapper.
    """

    question_list = list(questions)
    layout, layout_errors = build_question_layout(
        document,
        question_list,
        regions_document,
    )
    visual_contract = full_document_visual_contract(document)
    if not layout_errors:
        return {
            "ready": True,
            "strategy": "verified_question_regions",
            "question_layout": layout,
            "paper_context": {
                **visual_contract,
                "ready": True,
                "mode": "verified_question_regions",
                "requires_question_regions": True,
                "blockers": [],
            },
            "warnings": [],
            "errors": [],
        }
    if visual_contract["ready"]:
        return {
            "ready": True,
            "strategy": "full_document_visual",
            "question_layout": [],
            "paper_context": visual_contract,
            "warnings": layout_errors,
            "errors": [],
        }
    return {
        "ready": False,
        "strategy": "unavailable",
        "question_layout": [],
        "paper_context": visual_contract,
        "warnings": [],
        "errors": [*layout_errors, *visual_contract["blockers"]],
    }


def validate_pcr_questions(
    questions: Iterable[Dict[str, Any]],
    *,
    marking_policy: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return human-readable readiness errors for a PCR marking paper.

    OCR extraction is not a marking plan.  Each question must be identifiable,
    have text and marks, and contain either an approved reference solution or
    an explicit rubric before the paper becomes immutable.
    """
    policy_module = _marking_policy_module()
    policy = policy_module.normalize_marking_policy(marking_policy)
    uses_structured_criteria = policy_module.is_structured_rubric_policy(policy)
    errors: List[str] = []
    for position, question in enumerate(questions, start=1):
        label = _source_question_id(question) or str(position)
        if not _source_question_id(question):
            errors.append(f"Q {label}: missing stable question id")
        if not _question_text(question):
            errors.append(f"Q {label}: missing question text")
        if _question_marks(question) is None:
            errors.append(f"Q {label}: assign marks greater than zero")
        extraction_metadata = question.get("extraction_metadata")
        if isinstance(extraction_metadata, dict) and extraction_metadata.get(
            "manual_review_required"
        ):
            errors.append(f"Q {label}: resolve the OCR/layout review warning")
        if _is_objective_question(question):
            option_labels = _objective_option_labels(question)
            correct_answer = normalize_answer_label(
                question.get("correct_answer") or question.get("correctAnswer")
            )
            if len(option_labels) < 2:
                errors.append(f"Q {label}: add at least two objective answer options")
            if not correct_answer:
                errors.append(f"Q {label}: select the correct objective answer")
            elif option_labels and correct_answer not in option_labels:
                errors.append(
                    f"Q {label}: correct answer {correct_answer} is not one of the saved options"
                )
            try:
                parse_question_penalty(
                    question.get("penalty", question.get("penalty_marks")),
                )
            except ValueError as exc:
                errors.append(f"Q {label}: invalid negative marking: {exc}")
            # Objective PCR does not need an AI rubric. The camera grader only
            # transcribes a selected option; the server applies the frozen key.
            continue

        # A legacy paper still needs free-text marking material. A structured
        # PCR paper instead has a complete teacher-authored criterion plan,
        # so its optional worked solution/notes are helpful context rather
        # than a second mandatory source of marks.
        if (
            not uses_structured_criteria
            and not _question_reference_solution(question)
            and not _question_rubric(question)
        ):
            errors.append(
                f"Q {label}: add an approved reference solution or marking rubric"
            )
        if uses_structured_criteria:
            raw_assessment_units = _raw_question_assessment_units(question)
            try:
                assessment_units = policy_module.normalize_assessment_units(
                    raw_assessment_units,
                    assign_missing_ids=False,
                )
            except ValueError as exc:
                errors.append(f"Q {label}: invalid assessment units: {exc}")
                continue
            if assessment_units:
                unit_errors = policy_module.validate_assessment_units(
                    assessment_units,
                    _question_marks(question),
                    require_reference_solution=True,
                )
                errors.extend(f"Q {label}: {error}" for error in unit_errors)
                projected_criteria = policy_module.flatten_assessment_unit_criteria(
                    assessment_units
                )
                saved_criteria = _question_marking_criteria(question)
                if policy_module.snapshot_criteria(projected_criteria) != policy_module.snapshot_criteria(saved_criteria):
                    errors.append(
                        f"Q {label}: assessment-unit criteria projection is out of sync; save or regenerate the marking plan"
                    )
                continue
            try:
                method_policy = _question_method_policy(question)
            except ValueError as exc:
                errors.append(f"Q {label}: {exc}")
                method_policy = policy_module.default_method_policy()
            criteria = _question_marking_criteria(question)
            criterion_errors = policy_module.validate_marking_criteria(
                criteria,
                _question_marks(question),
                # Teachers may use broader multi-mark criteria for essays,
                # speeches, language work, and other holistic responses. The
                # grader still validates every award against the frozen range.
                require_atomic=False,
            )
            errors.extend(f"Q {label}: {error}" for error in criterion_errors)
    return errors


async def ensure_indexes(tenant_db: Any) -> None:
    """Create idempotent indexes for immutable paper snapshots."""
    await tenant_db[PAPER_ASSETS_COLLECTION].create_index(
        "asset_id", unique=True, name="uniq_paper_asset"
    )
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
    *,
    question_layout: Optional[List[Dict[str, Any]]] = None,
    paper_context: Optional[Dict[str, Any]] = None,
    paper_assets: Optional[Dict[str, Dict[str, Any]]] = None,
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

    policy_module = _marking_policy_module()
    marking_policy = policy_module.normalize_marking_policy(
        document.get("pcr_marking_policy")
    )
    content_hash = _content_hash(
        {**document, "pcr_marking_policy": marking_policy},
        questions,
        question_layout,
        paper_context,
    )
    paper_version_id = f"paper-{document_id}-{content_hash[:16]}"
    versions = tenant_db[PAPER_VERSIONS_COLLECTION]
    paper_questions = tenant_db[PAPER_QUESTIONS_COLLECTION]
    await ensure_indexes(tenant_db)

    now = datetime.now(timezone.utc)
    reservation_token = uuid.uuid4().hex
    layout_by_question = {
        _as_text(item.get("source_question_id")): copy.deepcopy(item)
        for item in (question_layout or [])
        if _as_text(item.get("source_question_id"))
    }
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
        clean_question["penalty"] = normalize_question_penalty(
            clean_question.get("penalty", clean_question.get("penalty_marks")),
            question_type=clean_question.get("question_type"),
            document_question_type=document.get("question_type"),
        )
        if policy_module.is_structured_rubric_policy(marking_policy):
            # Snapshot exactly the validated criterion rows.  This prevents a
            # later authoring edit from changing a conducted sitting.
            clean_question["marking_criteria"] = policy_module.snapshot_criteria(
                _question_marking_criteria(clean_question)
            )
            clean_question["method_policy"] = policy_module.snapshot_method_policy(
                _question_method_policy(clean_question)
            )
            assessment_units = _question_assessment_units(clean_question)
            if assessment_units:
                clean_question["assessment_units"] = (
                    policy_module.snapshot_assessment_units(assessment_units)
                )
        question_docs.append(
            {
                "paper_version_id": paper_version_id,
                "document_id": document_id,
                "source_question_id": source_question_id,
                "position": position,
                "question": clean_question,
                "layout": layout_by_question.get(source_question_id),
                "created_at": now,
            }
        )

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
        "layout_schema_version": PAPER_LAYOUT_SCHEMA_VERSION if question_layout else None,
        "question_layout": copy.deepcopy(question_layout or []),
        "layout_status": (
            "verified"
            if question_layout
            else (
                "full_document_visual"
                if (paper_context or {}).get("mode") == "full_document_visual"
                else "legacy_unverified"
            )
        ),
        "paper_context": copy.deepcopy(paper_context or {}),
        "paper_assets": copy.deepcopy(paper_assets or {}),
        "pcr_marking_policy": copy.deepcopy(marking_policy),
        "created_at": now,
        "source_document_finalized_at": document.get("exam_finalized_at"),
        "snapshot_status": "building",
        "snapshot_reservation_token": reservation_token,
        "snapshot_reservation_expires_at": now + timedelta(minutes=10),
    }

    existing = await versions.find_one({"paper_version_id": paper_version_id})
    if existing is not None and existing.get("snapshot_status") != "building":
        return existing
    if existing is not None:
        expires_at = existing.get("snapshot_reservation_expires_at")
        if isinstance(expires_at, datetime):
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at > now:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="The immutable paper snapshot is already being created",
                )
        reclaimed = await versions.update_one(
            {
                "paper_version_id": paper_version_id,
                "snapshot_status": "building",
                "snapshot_reservation_token": existing.get("snapshot_reservation_token"),
            },
            {
                "$set": {
                    "snapshot_reservation_token": reservation_token,
                    "snapshot_reservation_expires_at": now + timedelta(minutes=10),
                }
            },
        )
        if reclaimed.matched_count != 1:
            current = await versions.find_one({"paper_version_id": paper_version_id})
            if current is not None and current.get("snapshot_status") != "building":
                return current
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The immutable paper snapshot changed while it was being reclaimed",
            )
        await paper_questions.delete_many({"paper_version_id": paper_version_id})
    else:
        try:
            await versions.insert_one(version_doc)
        except DuplicateKeyError:
            current = await versions.find_one({"paper_version_id": paper_version_id})
            if current is not None and current.get("snapshot_status") != "building":
                return current
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The immutable paper snapshot is already being created",
            )

    try:
        if question_docs:
            await paper_questions.insert_many(question_docs, ordered=True)
        ready_at = datetime.now(timezone.utc)
        committed = await versions.update_one(
            {
                "paper_version_id": paper_version_id,
                "snapshot_status": "building",
                "snapshot_reservation_token": reservation_token,
            },
            {
                "$set": {"snapshot_status": "ready", "snapshot_ready_at": ready_at},
                "$unset": {
                    "snapshot_reservation_token": "",
                    "snapshot_reservation_expires_at": "",
                },
            },
        )
        if committed.matched_count != 1:
            raise RuntimeError("Lost immutable paper snapshot ownership before commit")
        stored = await versions.find_one({"paper_version_id": paper_version_id})
        return stored or {**version_doc, "snapshot_status": "ready", "snapshot_ready_at": ready_at}
    except Exception:
        await paper_questions.delete_many({"paper_version_id": paper_version_id})
        await versions.delete_one(
            {
                "paper_version_id": paper_version_id,
                "snapshot_status": "building",
                "snapshot_reservation_token": reservation_token,
            }
        )
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
            errors = validate_pcr_questions(
                snapshot_source_questions,
                marking_policy=document.get("pcr_marking_policy"),
            )
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

        policy_module = _marking_policy_module()
        marking_policy = policy_module.normalize_marking_policy(
            paper_version.get("pcr_marking_policy")
        )
        docs: List[Dict[str, Any]] = []
        for snapshot_question, raw_question in zip(snapshot_questions, raw_questions):
            source_id = _as_text(snapshot_question.get("source_question_id"))
            question_layout = copy.deepcopy(snapshot_question.get("layout") or {})
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
            pcr_doc["question_layout"] = question_layout or None
            pcr_doc["source_page_number"] = question_layout.get("page_number")
            pcr_doc["source_region_id"] = question_layout.get("source_region_id")
            pcr_doc["source_bbox_percent"] = question_layout.get("bbox_percent")
            pcr_doc["marking_policy"] = copy.deepcopy(marking_policy)
            pcr_doc["marking_criteria"] = _question_marking_criteria(raw_question)
            pcr_doc["method_policy"] = _question_method_policy(raw_question)
            pcr_doc["assessment_units"] = _question_assessment_units(raw_question)
            if policy_module.is_structured_rubric_policy(marking_policy):
                pcr_doc["evaluation_mode"] = policy_module.STRUCTURED_RUBRIC_MODE
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
