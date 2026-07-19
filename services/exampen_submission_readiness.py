"""Authoritative ExamPen PCR publication-readiness checks.

The teacher queue, publish endpoint, and any future bulk-publish path must use
this module.  A missing response row is never interpreted as a blank answer;
only an explicit, evidence-backed ``not_attempted`` row is terminal.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List


TERMINAL_RESPONSE_STATUSES = {
    "evaluated",
    "evaluated_with_warnings",
    "evaluated_teacher_reviewed",
    "not_attempted",
}


def _blocker(code: str, message: str, **details: Any) -> Dict[str, Any]:
    return {"code": code, "message": message, **details}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _has_unresolved_blocking_flag(response: Dict[str, Any]) -> bool:
    for flag in response.get("flags") or []:
        if not isinstance(flag, dict) or flag.get("severity") != "blocking":
            continue
        resolution = flag.get("resolution")
        if not isinstance(resolution, dict) or not resolution.get("resolved"):
            return True
    return False


def _absence_is_proven(response: Dict[str, Any]) -> bool:
    assignment = response.get("question_assignment")
    proof = assignment.get("absence_proof") if isinstance(assignment, dict) else None
    return bool(
        response.get("is_missing_response")
        and response.get("answer_state") == "not_attempted"
        and response.get("absence_proven") is True
        and isinstance(proof, dict)
        and proof.get("verified") is True
        and not str(response.get("detected_text") or "").strip()
        and not (response.get("source_pages") or [])
    )


async def assess_submission_readiness(
    tenant_db: Any,
    submission_id: str,
    *,
    _preloaded: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a fail-closed, audit-friendly readiness report."""

    submission = (
        _preloaded.get("submission")
        if _preloaded is not None
        else await tenant_db["evalpen_submissions"].find_one(
            {"submission_id": submission_id}
        )
    )
    if submission is None:
        return {
            "submission_id": submission_id,
            "ready": False,
            "blockers": [_blocker("submission_not_found", "Submission was not found")],
            "counts": {},
        }

    exam_id = str(submission.get("exam_id") or "")
    blockers: List[Dict[str, Any]] = []

    if _preloaded is not None:
        job = _preloaded.get("job")
    else:
        job_docs = (
            await tenant_db["exampen_processing_jobs"]
            .find({"submission_id": submission_id})
            .sort([("created_at", -1), ("updated_at", -1)])
            .to_list(length=1)
        )
        job = job_docs[0] if job_docs else None
    processing_status = str(job.get("status") or "") if job else "missing"
    if processing_status != "completed":
        blockers.append(
            _blocker(
                "processing_not_completed",
                "Answer-copy processing has not completed successfully",
                processing_status=processing_status,
            )
        )

    segmentation_status = str(submission.get("segmentation_status") or "")
    if segmentation_status != "complete":
        blockers.append(
            _blocker(
                "segmentation_not_completed",
                "Answer-copy segmentation has not completed",
                segmentation_status=segmentation_status,
            )
        )

    questions = (
        list(_preloaded.get("questions") or [])
        if _preloaded is not None
        else await tenant_db["evalpen_questions"]
        .find({"exam_id": exam_id})
        .sort([("question_number", 1), ("question_id", 1)])
        .to_list(length=1000)
    )
    if not questions:
        blockers.append(
            _blocker("paper_catalog_missing", "The immutable paper question catalog is empty")
        )

    exam = (
        _preloaded.get("exam")
        if _preloaded is not None
        else await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    )
    expected_paper_version = str((exam or {}).get("paper_version_id") or "")
    if expected_paper_version:
        mismatched = [
            str(question.get("question_id") or "")
            for question in questions
            if str(question.get("paper_version_id") or "") != expected_paper_version
        ]
        if mismatched:
            blockers.append(
                _blocker(
                    "paper_version_mismatch",
                    "Question catalog does not match the conducted paper version",
                    question_ids=mismatched,
                )
            )

    responses = (
        list(_preloaded.get("responses") or [])
        if _preloaded is not None
        else await tenant_db["evalpen_detected_responses"].find(
            {
                "submission_id": submission_id,
                "superseded_at": {"$exists": False},
            }
        ).to_list(length=5000)
    )
    response_ids = [str(response.get("response_id") or "") for response in responses]
    evaluations = (
        list(_preloaded.get("evaluations") or [])
        if _preloaded is not None
        else (
            await tenant_db["evalpen_evaluations"]
            .find({"response_id": {"$in": response_ids}})
            .to_list(length=5000)
            if response_ids
            else []
        )
    )
    evaluations_by_response: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        evaluations_by_response[str(evaluation.get("response_id") or "")].append(evaluation)

    catalog_by_id = {
        str(question.get("question_id") or ""): question
        for question in questions
        if str(question.get("question_id") or "")
    }
    responses_by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    responses_by_evidence_atom: Dict[str, List[str]] = defaultdict(list)
    unassigned_response_ids: List[str] = []
    unresolved_flag_ids: List[str] = []
    unknown_question_response_ids: List[str] = []

    for response in responses:
        response_id = str(response.get("response_id") or "")
        question_id = str(response.get("question_id") or "")
        if not question_id:
            unassigned_response_ids.append(response_id)
        elif question_id not in catalog_by_id:
            unknown_question_response_ids.append(response_id)
        else:
            responses_by_question[question_id].append(response)
        if _has_unresolved_blocking_flag(response):
            unresolved_flag_ids.append(response_id)
        for raw_atom_id in response.get("evidence_atom_ids") or []:
            atom_id = str(raw_atom_id or "").strip()
            if atom_id:
                responses_by_evidence_atom[atom_id].append(response_id)

    if unassigned_response_ids:
        blockers.append(
            _blocker(
                "unassigned_evidence",
                "Some answer-copy evidence is not assigned to a paper question",
                response_ids=unassigned_response_ids,
            )
        )
    if unknown_question_response_ids:
        blockers.append(
            _blocker(
                "unknown_question_assignment",
                "Some responses refer to questions outside the immutable paper",
                response_ids=unknown_question_response_ids,
            )
        )
    if unresolved_flag_ids:
        blockers.append(
            _blocker(
                "unresolved_blocking_flags",
                "Some responses still have unresolved blocking flags",
                response_ids=unresolved_flag_ids,
            )
        )
    duplicate_atoms = {
        atom_id: sorted(set(owner_ids))
        for atom_id, owner_ids in responses_by_evidence_atom.items()
        if len(set(owner_ids)) > 1
    }
    if duplicate_atoms:
        blockers.append(
            _blocker(
                "duplicate_evidence_ownership",
                "The same answer-copy evidence is owned by more than one response",
                evidence_atoms=duplicate_atoms,
            )
        )

    for question_id, question in catalog_by_id.items():
        owned = responses_by_question.get(question_id, [])
        if not owned:
            blockers.append(
                _blocker(
                    "question_state_missing",
                    "A paper question has no verified answer state",
                    question_id=question_id,
                    question_number=question.get("question_number"),
                )
            )
            continue
        if len(owned) != 1:
            blockers.append(
                _blocker(
                    "duplicate_question_responses",
                    "A paper question has more than one active response",
                    question_id=question_id,
                    response_ids=[str(item.get("response_id") or "") for item in owned],
                )
            )
            continue

        response = owned[0]
        response_id = str(response.get("response_id") or "")
        response_status = str(response.get("eval_status") or "").lower()
        assignment = response.get("question_assignment")
        if response.get("manual_review_required") or (
            isinstance(assignment, dict)
            and assignment.get("manual_review_required")
        ):
            blockers.append(
                _blocker(
                    "response_assignment_requires_review",
                    "Question ownership or answer evidence still requires teacher review",
                    question_id=question_id,
                    response_id=response_id,
                    reason=response.get("manual_review_reason")
                    or (
                        assignment.get("reason")
                        if isinstance(assignment, dict)
                        else None
                    ),
                )
            )
        if response.get("is_missing_response") and not _absence_is_proven(response):
            blockers.append(
                _blocker(
                    "absence_not_proven",
                    "A not-attempted zero lacks verified document-coverage evidence",
                    question_id=question_id,
                    response_id=response_id,
                )
            )
        if response_status not in TERMINAL_RESPONSE_STATUSES:
            blockers.append(
                _blocker(
                    "response_not_terminal",
                    "A question response is not in a terminal evaluated state",
                    question_id=question_id,
                    response_id=response_id,
                    eval_status=response_status,
                )
            )

        response_evaluations = evaluations_by_response.get(response_id, [])
        if len(response_evaluations) != 1:
            blockers.append(
                _blocker(
                    "evaluation_cardinality_invalid",
                    "A question must have exactly one active evaluation",
                    question_id=question_id,
                    response_id=response_id,
                    evaluation_count=len(response_evaluations),
                )
            )
            continue

        evaluation = response_evaluations[0]
        if str(evaluation.get("question_id") or "") != question_id:
            blockers.append(
                _blocker(
                    "evaluation_question_mismatch",
                    "Evaluation ownership does not match response ownership",
                    question_id=question_id,
                    response_id=response_id,
                )
            )
        if evaluation.get("manual_review_required"):
            blockers.append(
                _blocker(
                    "evaluation_requires_review",
                    "An evaluation still requires teacher review",
                    question_id=question_id,
                    response_id=response_id,
                )
            )

        expected_max = _number(question.get("max_marks"))
        actual_max = _number(evaluation.get("max_score"))
        total_score = _number(evaluation.get("total_score"))
        if expected_max is None or expected_max <= 0:
            blockers.append(
                _blocker("invalid_question_max", "Question maximum marks are invalid", question_id=question_id)
            )
        elif actual_max is None or abs(actual_max - expected_max) > 0.01:
            blockers.append(
                _blocker(
                    "evaluation_max_mismatch",
                    "Evaluation maximum does not match the immutable question",
                    question_id=question_id,
                    expected_max=expected_max,
                    actual_max=actual_max,
                )
            )
        if total_score is None or total_score < 0 or (
            expected_max is not None and total_score > expected_max + 0.01
        ):
            blockers.append(
                _blocker(
                    "evaluation_score_invalid",
                    "Evaluation score is missing or outside the immutable bounds",
                    question_id=question_id,
                    total_score=total_score,
                )
            )
        if response.get("is_missing_response") and total_score not in {0, 0.0}:
            blockers.append(
                _blocker(
                    "not_attempted_nonzero",
                    "A proven not-attempted response must have zero marks",
                    question_id=question_id,
                    total_score=total_score,
                )
            )

    return {
        "submission_id": submission_id,
        "exam_id": exam_id,
        "ready": not blockers,
        "blockers": blockers,
        "counts": {
            "question_count": len(catalog_by_id),
            "response_count": len(responses),
            "evaluation_count": len(evaluations),
            "blocker_count": len(blockers),
        },
        "processing_status": processing_status,
        "segmentation_status": segmentation_status,
        "paper_version_id": expected_paper_version or None,
    }


async def assess_submissions_readiness(
    tenant_db: Any,
    submission_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Assess an exam queue with a bounded set of batch database reads.

    This keeps teacher queue latency roughly constant as student count grows;
    it replaces the former seven-query-per-student loop.
    """
    ordered_ids = list(dict.fromkeys(str(item) for item in submission_ids if str(item)))
    if not ordered_ids:
        return {}

    submissions = await tenant_db["evalpen_submissions"].find(
        {"submission_id": {"$in": ordered_ids}}
    ).to_list(length=len(ordered_ids))
    submissions_by_id = {
        str(item.get("submission_id") or ""): item for item in submissions
    }
    exam_ids = sorted(
        {
            str(item.get("exam_id") or "")
            for item in submissions
            if str(item.get("exam_id") or "")
        }
    )

    jobs = await tenant_db["exampen_processing_jobs"].find(
        {"submission_id": {"$in": ordered_ids}}
    ).sort([("created_at", -1), ("updated_at", -1)]).to_list(
        length=max(len(ordered_ids) * 5, len(ordered_ids))
    )
    jobs_by_submission: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        jobs_by_submission.setdefault(str(job.get("submission_id") or ""), job)

    questions = (
        await tenant_db["evalpen_questions"].find(
            {"exam_id": {"$in": exam_ids}}
        ).sort([("question_number", 1), ("question_id", 1)]).to_list(
            length=max(len(exam_ids) * 1000, 1000)
        )
        if exam_ids
        else []
    )
    questions_by_exam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_exam[str(question.get("exam_id") or "")].append(question)

    exams = (
        await tenant_db["exampen_exams"].find(
            {"exam_id": {"$in": exam_ids}}
        ).to_list(length=len(exam_ids))
        if exam_ids
        else []
    )
    exams_by_id = {str(item.get("exam_id") or ""): item for item in exams}

    responses = await tenant_db["evalpen_detected_responses"].find(
        {
            "submission_id": {"$in": ordered_ids},
            "superseded_at": {"$exists": False},
        }
    ).to_list(length=max(len(ordered_ids) * 1000, 5000))
    responses_by_submission: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    response_to_submission: Dict[str, str] = {}
    response_ids: List[str] = []
    for response in responses:
        submission_id = str(response.get("submission_id") or "")
        response_id = str(response.get("response_id") or "")
        responses_by_submission[submission_id].append(response)
        if response_id:
            response_ids.append(response_id)
            response_to_submission[response_id] = submission_id

    evaluations = (
        await tenant_db["evalpen_evaluations"].find(
            {"response_id": {"$in": response_ids}}
        ).to_list(length=max(len(response_ids) * 2, 5000))
        if response_ids
        else []
    )
    evaluations_by_submission: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        owner = response_to_submission.get(str(evaluation.get("response_id") or ""))
        if owner:
            evaluations_by_submission[owner].append(evaluation)

    reports: Dict[str, Dict[str, Any]] = {}
    for submission_id in ordered_ids:
        submission = submissions_by_id.get(submission_id)
        exam_id = str((submission or {}).get("exam_id") or "")
        reports[submission_id] = await assess_submission_readiness(
            tenant_db,
            submission_id,
            _preloaded={
                "submission": submission,
                "job": jobs_by_submission.get(submission_id),
                "questions": questions_by_exam.get(exam_id, []),
                "exam": exams_by_id.get(exam_id),
                "responses": responses_by_submission.get(submission_id, []),
                "evaluations": evaluations_by_submission.get(submission_id, []),
            },
        )
    return reports


def readiness_message(report: Dict[str, Any], *, limit: int = 3) -> str:
    blockers = report.get("blockers") or []
    messages = [str(item.get("message") or item.get("code") or "Not ready") for item in blockers[:limit]]
    suffix = f" (+{len(blockers) - limit} more)" if len(blockers) > limit else ""
    return "; ".join(messages) + suffix


def validate_publication_snapshot(
    snapshot: Any,
    *,
    submission_id: str,
    exam_id: str,
) -> bool:
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("snapshot_version") != 1:
        return False
    if str(snapshot.get("submission_id") or "") != submission_id:
        return False
    if str(snapshot.get("exam_id") or "") != exam_id:
        return False
    rows = snapshot.get("score_rows")
    if not isinstance(rows, list) or not rows:
        return False
    question_ids = [
        str(row.get("question_id") or "")
        for row in rows
        if isinstance(row, dict)
    ]
    if len(question_ids) != len(rows) or any(not value for value in question_ids):
        return False
    if len(question_ids) != len(set(question_ids)):
        return False
    row_total = sum(_number(row.get("score")) or 0.0 for row in rows)
    row_max = sum(_number(row.get("max_score")) or 0.0 for row in rows)
    snapshot_total = _number(snapshot.get("total_score"))
    snapshot_max = _number(snapshot.get("total_max_score"))
    if snapshot_total is None or abs(row_total - snapshot_total) > 0.01:
        return False
    if snapshot_max is None or abs(row_max - snapshot_max) > 0.01:
        return False
    expected_hash = str(snapshot.get("snapshot_hash") or "")
    core = {key: value for key, value in snapshot.items() if key != "snapshot_hash"}
    actual_hash = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return bool(expected_hash) and expected_hash == actual_hash


async def build_publication_snapshot(
    tenant_db: Any,
    submission_id: str,
    *,
    actor_id: str,
) -> Dict[str, Any]:
    """Materialize the exact immutable score rows released to the student."""

    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": submission_id}
    )
    if submission is None:
        raise ValueError("Submission not found")
    exam_id = str(submission.get("exam_id") or "")
    questions = (
        await tenant_db["evalpen_questions"]
        .find({"exam_id": exam_id})
        .sort([("question_number", 1), ("question_id", 1)])
        .to_list(length=1000)
    )
    responses = await tenant_db["evalpen_detected_responses"].find(
        {"submission_id": submission_id, "superseded_at": {"$exists": False}}
    ).to_list(length=5000)
    by_question = {
        str(response.get("question_id") or ""): response
        for response in responses
        if str(response.get("question_id") or "")
    }
    response_ids = [str(response.get("response_id") or "") for response in responses]
    evaluations = await tenant_db["evalpen_evaluations"].find(
        {"response_id": {"$in": response_ids}}
    ).to_list(length=5000)
    by_response = {
        str(evaluation.get("response_id") or ""): evaluation
        for evaluation in evaluations
    }

    score_rows: List[Dict[str, Any]] = []
    total_score = 0.0
    total_max = 0.0
    for question in questions:
        question_id = str(question.get("question_id") or "")
        response = by_question[question_id]
        response_id = str(response.get("response_id") or "")
        evaluation = by_response[response_id]
        score = float(evaluation.get("total_score") or 0.0)
        maximum = float(question.get("max_marks") or 0.0)
        total_score += score
        total_max += maximum
        score_rows.append(
            {
                "question_id": question_id,
                "question_number": question.get("question_number"),
                "response_id": response_id,
                "evaluation_id": str(evaluation.get("evaluation_id") or ""),
                "answer_state": response.get("answer_state") or "detected",
                "score": round(score, 2),
                "max_score": round(maximum, 2),
                "model_used": evaluation.get("model_used"),
                "teacher_reviewed": bool(evaluation.get("teacher_reviewed")),
                "overall_feedback": evaluation.get("overall_feedback"),
                "criterion_marks": evaluation.get("criterion_marks"),
                "step_marks": evaluation.get("step_marks"),
                "reference_solution": evaluation.get("reference_solution"),
                "teacher_feedback": evaluation.get("teacher_feedback")
                or evaluation.get("teacher_note"),
            }
        )

    published_at = datetime.now(timezone.utc)
    snapshot_core = {
        "snapshot_version": 1,
        "submission_id": submission_id,
        "exam_id": exam_id,
        "student_id": str(submission.get("student_id") or ""),
        "paper_version_id": str((await tenant_db["exampen_exams"].find_one(
            {"exam_id": exam_id}, {"paper_version_id": 1}
        ) or {}).get("paper_version_id") or ""),
        "score_rows": score_rows,
        "total_score": round(total_score, 2),
        "total_max_score": round(total_max, 2),
        "published_by": actor_id,
        "published_at": published_at.isoformat(),
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(snapshot_core, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {**snapshot_core, "snapshot_hash": snapshot_hash, "published_at_dt": published_at}
