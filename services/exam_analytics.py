"""Canonical published ExamPen attempts for teacher analytics.

Analytics must never infer a final score from mutable OCR/evaluation rows.  The
student results BFF exposes an exam only after the submission is published, and
PCR marks are released from an integrity-checked publication snapshot.  This
module applies the same contract for teacher dashboards and leaderboards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from services.exampen_submission_readiness import validate_publication_snapshot


@dataclass(frozen=True)
class PublishedExamAttempt:
    """One final, published exam result owned by one roster student."""

    student_key: str
    exam_id: str
    percentage: float
    total_score: float
    max_score: float
    subject: str = ""
    published_at: Optional[datetime] = None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _marks(value: Any) -> float:
    return max(0.0, _number(value))


def _as_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _newer_submission(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    candidate_at = _as_datetime(candidate.get("published_at"))
    current_at = _as_datetime(current.get("published_at"))
    if candidate_at and current_at:
        return candidate_at > current_at
    if candidate_at:
        return True
    if current_at:
        return False
    return _text(candidate.get("submission_id")) > _text(current.get("submission_id"))


def _student_alias_map(students: Iterable[Mapping[str, Any]]) -> Dict[str, str]:
    """Map both legacy account ids and roster ids to the analytics student key."""

    aliases: Dict[str, str] = {}
    for student in students:
        student_key = _text(student.get("_id"))
        if not student_key:
            continue
        for identity in (student_key, _text(student.get("student_id"))):
            if identity:
                aliases[identity] = student_key
    return aliases


def _latest_by_question(
    documents: Iterable[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    for index, document in enumerate(documents):
        question_key = _text(document.get("question_id")) or _text(document.get("_id"))
        if not question_key:
            question_key = f"row-{index}"
        current = latest.get(question_key)
        if current is None:
            latest[question_key] = document
            continue
        candidate_at = _as_datetime(
            document.get("updated_at") or document.get("created_at")
        )
        current_at = _as_datetime(
            current.get("updated_at") or current.get("created_at")
        )
        if candidate_at and (not current_at or candidate_at > current_at):
            latest[question_key] = document
    return latest


async def load_published_exam_attempts(
    tenant_db: Any,
    students: Iterable[Mapping[str, Any]],
) -> List[PublishedExamAttempt]:
    """Load final ExamPen percentages for a tutor-visible student roster.

    One latest published submission is counted per student and exam.  A PCR
    exam is withheld entirely when its immutable snapshot is absent or fails
    validation, matching the student results contract.  DCR rows are accepted
    only when the same student/exam has a published submission.
    """

    alias_map = _student_alias_map(students)
    if not alias_map:
        return []

    submissions = (
        await tenant_db["evalpen_submissions"]
        .find(
            {
                "student_id": {"$in": list(alias_map)},
                "publication_status": "published",
            },
            {
                "_id": 0,
                "submission_id": 1,
                "exam_id": 1,
                "student_id": 1,
                "published_at": 1,
                "publication_snapshot": 1,
            },
        )
        .to_list(length=5000)
    )

    latest_submissions: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for submission in submissions:
        student_key = alias_map.get(_text(submission.get("student_id")))
        exam_id = _text(submission.get("exam_id"))
        submission_id = _text(submission.get("submission_id"))
        if not student_key or not exam_id or not submission_id:
            continue
        key = (student_key, exam_id)
        current = latest_submissions.get(key)
        if current is None or _newer_submission(submission, current):
            latest_submissions[key] = submission

    if not latest_submissions:
        return []

    exam_ids = list(dict.fromkeys(exam_id for _, exam_id in latest_submissions))
    question_docs = (
        await tenant_db["evalpen_questions"]
        .find(
            {"exam_id": {"$in": exam_ids}},
            {"_id": 0, "exam_id": 1, "subject": 1},
        )
        .to_list(length=max(len(exam_ids) * 1000, 5000))
    )
    pcr_exam_ids = {
        _text(question.get("exam_id"))
        for question in question_docs
        if _text(question.get("exam_id"))
    }
    question_subjects: Dict[str, str] = {}
    for question in question_docs:
        exam_id = _text(question.get("exam_id"))
        subject = _text(question.get("subject"))
        if exam_id and subject:
            question_subjects.setdefault(exam_id, subject)

    exam_docs = (
        await tenant_db["exampen_exams"]
        .find(
            {"exam_id": {"$in": exam_ids}},
            {"_id": 0, "exam_id": 1, "subject": 1, "prepared_document_id": 1},
        )
        .to_list(length=5000)
    )
    exam_doc_map = {_text(document.get("exam_id")): document for document in exam_docs}
    document_ids = [
        _text(document.get("prepared_document_id"))
        for document in exam_docs
        if _text(document.get("prepared_document_id"))
    ]
    prepared_docs = (
        await tenant_db["documents"]
        .find(
            {"document_id": {"$in": document_ids}},
            {"_id": 0, "document_id": 1, "subject": 1},
        )
        .to_list(length=5000)
        if document_ids
        else []
    )
    prepared_doc_map = {
        _text(document.get("document_id")): document for document in prepared_docs
    }

    dcr_documents = (
        await tenant_db["exampen_dcr_results"]
        .find(
            {
                "exam_id": {"$in": exam_ids},
                "student_id": {"$in": list(alias_map)},
            },
            {
                "question_id": 1,
                "exam_id": 1,
                "student_id": 1,
                "score": 1,
                "max_score": 1,
                "created_at": 1,
                "updated_at": 1,
            },
        )
        .to_list(length=10000)
    )
    dcr_by_attempt: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for document in dcr_documents:
        student_key = alias_map.get(_text(document.get("student_id")))
        exam_id = _text(document.get("exam_id"))
        key = (student_key or "", exam_id)
        if student_key and key in latest_submissions:
            dcr_by_attempt.setdefault(key, []).append(document)

    attempts: List[PublishedExamAttempt] = []
    for (student_key, exam_id), submission in latest_submissions.items():
        pcr_total = 0.0
        pcr_max = 0.0
        if exam_id in pcr_exam_ids:
            snapshot = submission.get("publication_snapshot")
            if not validate_publication_snapshot(
                snapshot,
                submission_id=_text(submission.get("submission_id")),
                exam_id=exam_id,
            ):
                # Same fail-closed integrity rule as the student result BFF.
                continue
            pcr_total = _number(snapshot.get("total_score"))
            pcr_max = _marks(snapshot.get("total_max_score"))

        dcr_total = 0.0
        dcr_max = 0.0
        for result in _latest_by_question(
            dcr_by_attempt.get((student_key, exam_id), [])
        ).values():
            dcr_total += _number(result.get("score"))
            dcr_max += _marks(result.get("max_score"))

        total_score = pcr_total + dcr_total
        total_max = pcr_max + dcr_max
        if total_max <= 0:
            continue

        exam_document = exam_doc_map.get(exam_id, {})
        prepared_document = prepared_doc_map.get(
            _text(exam_document.get("prepared_document_id")), {}
        )
        subject = (
            _text(exam_document.get("subject"))
            or _text(prepared_document.get("subject"))
            or question_subjects.get(exam_id, "")
        )
        published_at = _as_datetime(submission.get("published_at"))
        if not published_at and isinstance(
            submission.get("publication_snapshot"), Mapping
        ):
            published_at = _as_datetime(
                submission["publication_snapshot"].get("published_at")
            )

        attempts.append(
            PublishedExamAttempt(
                student_key=student_key,
                exam_id=exam_id,
                percentage=round(total_score / total_max * 100.0, 2),
                total_score=round(total_score, 2),
                max_score=round(total_max, 2),
                subject=subject,
                published_at=published_at,
            )
        )

    return attempts
