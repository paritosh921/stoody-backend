"""Teacher BFF — aggregation endpoints for the teacher UI.

Routes are mounted at ``/api/v1/exampen/teacher``.

Since everything runs in one process (monolith), the BFF calls storage
repos directly instead of making HTTP calls to backing services.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from exampen.dcr.core.auth_bridge import (
    ExamPenUser,
    get_exampen_user,
    require_exampen_role,
)
from exampen.dcr.storage.exam_repo import ExamRepo
from exampen.dcr.storage.score_event_store import ScoreEventStore
from exampen.dcr.storage.objection_repo import ObjectionRepo
from exampen.dcr.storage.analytics_cache_repo import AnalyticsCacheRepo
from exampen.dcr.storage.assignment_repo import AssignmentRepo
from exampen.dcr.storage.plagiarism_repo import PlagiarismRepo

logger = logging.getLogger(__name__)
router = APIRouter()

_EVALUATOR_ROLES = frozenset({
    "teacher", "evaluator", "hod", "principal", "super_admin",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/dashboard")
async def teacher_dashboard(
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """Aggregated teacher dashboard: recent exams, pending objections, flags.

    Returns a single payload combining data from multiple repos to avoid
    N+1 round trips from the frontend.
    """
    db = await _get_tenant_db(request, user)
    exam_repo = ExamRepo(db)
    obj_repo = ObjectionRepo(db)
    plag_repo = PlagiarismRepo(db)

    # Recent exams (last 10)
    exams = await exam_repo.list_exams(user.tenant_id, limit=10)

    # Pending objections across all exams
    obj_coll = db["exampen_objections"]
    pending_objs = await obj_coll.find(
        {
            "tenant_id": user.tenant_id,
            "state": {"$in": ["filed", "assigned"]},
        }
    ).sort("created_at", -1).limit(20).to_list(length=20)

    # Unreviewed plagiarism flags
    plag_coll = db["exampen_plagiarism_flags"]
    pending_flags = await plag_coll.find(
        {
            "tenant_id": user.tenant_id,
            "verdict": None,
        }
    ).sort("created_at", -1).limit(20).to_list(length=20)

    return {
        "recent_exams": exams,
        "pending_objections": pending_objs,
        "pending_objection_count": len(pending_objs),
        "unreviewed_flags": pending_flags,
        "unreviewed_flag_count": len(pending_flags),
    }


@router.get("/exams/{exam_id}/summary")
async def exam_summary(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """Single-exam summary: config + scores overview + objection count + assignment list.

    Aggregated from multiple repos for the exam detail page.
    """
    db = await _get_tenant_db(request, user)
    exam_repo = ExamRepo(db)
    score_store = ScoreEventStore(db)
    obj_repo = ObjectionRepo(db)
    assign_repo = AssignmentRepo(db)
    analytics = AnalyticsCacheRepo(db)

    # Exam config
    exam = await exam_repo.get_by_id(exam_id, user.tenant_id)
    if exam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")

    # Score overview (per-student totals)
    overview = await score_store.get_exam_overview(exam_id, user.tenant_id)

    # Objections for this exam
    objections = await obj_repo.list_by_exam(exam_id, user.tenant_id, limit=50)

    # Staff assignments
    assignments = await assign_repo.list_by_exam(exam_id, user.tenant_id)

    # Class stats (cached or fallback)
    class_stats = await analytics.get_class_stats(exam_id, user.tenant_id)

    return {
        "exam": exam,
        "score_overview": overview,
        "objections": objections,
        "objection_count": len(objections),
        "assignments": assignments,
        "class_stats": class_stats,
    }


@router.get("/exams/{exam_id}/students/{student_id}")
async def student_detail(
    exam_id: str,
    student_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """Detailed view of a student in an exam: scores + event history + objections.

    Aggregated for the individual student review page.
    """
    db = await _get_tenant_db(request, user)
    score_store = ScoreEventStore(db)
    obj_repo = ObjectionRepo(db)

    # Current scores
    scores = await score_store.get_current_scores(exam_id, student_id, user.tenant_id)
    if not scores:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No scores found for student")

    total = sum(r.get("score", 0.0) for r in scores)

    # Event history
    events = await score_store.get_event_history(exam_id, student_id, user.tenant_id)

    # Objections filed by this student for this exam
    obj_coll = db["exampen_objections"]
    student_objs = await obj_coll.find(
        {
            "exam_id": exam_id,
            "student_id": student_id,
            "tenant_id": user.tenant_id,
        }
    ).sort("created_at", -1).to_list(length=50)

    return {
        "exam_id": exam_id,
        "student_id": student_id,
        "total_score": round(total, 2),
        "scores": scores,
        "event_history": events,
        "objections": student_objs,
    }


@router.get("/my-assignments")
async def my_assignments(
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """List exams where the current user is assigned as invigilator or evaluator."""
    db = await _get_tenant_db(request, user)
    coll = db["exampen_assignments"]

    assignments = await coll.find(
        {"user_id": user.user_id, "tenant_id": user.tenant_id}
    ).to_list(length=100)

    # Enrich with exam titles
    exam_repo = ExamRepo(db)
    result = []
    for a in assignments:
        exam = await exam_repo.get_by_id(a["exam_id"], user.tenant_id)
        result.append({
            "exam_id": a["exam_id"],
            "roles": a.get("roles", []),
            "exam_title": exam.get("title", "") if exam else "",
            "exam_state": exam.get("state", "") if exam else "",
        })

    return {"assignments": result}
