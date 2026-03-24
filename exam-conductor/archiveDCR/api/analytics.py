"""Analytics endpoints — leaderboard, class stats, question analysis, student performance.

Routes are mounted at ``/api/v1/exampen/analytics``.
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
from exampen.dcr.storage.analytics_cache_repo import AnalyticsCacheRepo
from exampen.dcr.storage.score_event_store import ScoreEventStore

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVALUATOR_ROLES = frozenset({
    "teacher", "evaluator", "hod", "principal", "super_admin",
})


async def _get_tenant_db(request: Request, user: ExamPenUser):
    db = await request.app.state.db.get_tenant_db(user.tenant_id)
    if db is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
    return db


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/exams/{exam_id}/leaderboard")
async def get_leaderboard(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Return the leaderboard for an exam.

    Students see a limited view (their own rank + surrounding context).
    Evaluators see the full leaderboard.
    """
    db = await _get_tenant_db(request, user)
    cache = AnalyticsCacheRepo(db)
    entries = await cache.get_leaderboard(exam_id, user.tenant_id)

    # If the cache is empty, compute on-the-fly from score store
    if not entries:
        store = ScoreEventStore(db)
        overview = await store.get_exam_overview(exam_id, user.tenant_id)
        # Sort by total_score descending, assign ranks
        overview.sort(key=lambda r: r.get("total_score", 0), reverse=True)
        entries = []
        for rank, row in enumerate(overview, start=1):
            entries.append({
                "student_id": row["student_id"],
                "total_score": row["total_score"],
                "question_count": row.get("question_count", 0),
                "rank": rank,
            })

    is_evaluator = bool(_EVALUATOR_ROLES.intersection(user.exampen_roles))
    if not is_evaluator:
        # Students see only their own entry + neighbours
        own_idx = next(
            (i for i, e in enumerate(entries) if e.get("student_id") == user.user_id),
            None,
        )
        if own_idx is not None:
            start = max(0, own_idx - 2)
            end = min(len(entries), own_idx + 3)
            entries = entries[start:end]
        else:
            entries = []

    return {"exam_id": exam_id, "entries": entries}


@router.get("/exams/{exam_id}/class-stats")
async def get_class_stats(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """Return cached class statistics for an exam (mean, median, pass rate, etc.).

    Falls back to basic aggregation from the score store if no cache exists.
    """
    db = await _get_tenant_db(request, user)
    cache = AnalyticsCacheRepo(db)
    stats = await cache.get_class_stats(exam_id, user.tenant_id)

    if stats is not None:
        return stats

    # Fallback: compute basic stats from score store
    store = ScoreEventStore(db)
    overview = await store.get_exam_overview(exam_id, user.tenant_id)
    if not overview:
        return {"exam_id": exam_id, "student_count": 0}

    scores = [r.get("total_score", 0.0) for r in overview]
    scores.sort()
    n = len(scores)
    total = sum(scores)
    mean = total / n if n else 0
    median = scores[n // 2] if n else 0

    return {
        "exam_id": exam_id,
        "student_count": n,
        "mean_score": round(mean, 2),
        "median_score": round(median, 2),
        "min_score": scores[0] if scores else 0,
        "max_score": scores[-1] if scores else 0,
    }


@router.get("/exams/{exam_id}/questions")
async def get_question_analysis(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("evaluator", "hod", "principal", "super_admin")
    ),
) -> dict[str, Any]:
    """Per-question analysis: average score, attempt count, difficulty index.

    Aggregated from the current scores collection.
    """
    db = await _get_tenant_db(request, user)
    coll = db["exampen_score_current"]

    pipeline = [
        {"$match": {"exam_id": exam_id, "tenant_id": user.tenant_id}},
        {
            "$group": {
                "_id": "$question_id",
                "avg_score": {"$avg": "$score"},
                "max_marks": {"$max": "$max_marks"},
                "attempt_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    results = await coll.aggregate(pipeline).to_list(length=500)

    questions = []
    for r in results:
        q_id = r["_id"]
        if q_id == "__exam__":
            continue
        avg = r.get("avg_score", 0)
        max_m = r.get("max_marks")
        difficulty = round(avg / max_m, 3) if max_m and max_m > 0 else None
        questions.append({
            "question_id": q_id,
            "avg_score": round(avg, 2),
            "max_marks": max_m,
            "attempt_count": r.get("attempt_count", 0),
            "difficulty_index": difficulty,
        })

    return {"exam_id": exam_id, "questions": questions}


@router.get("/students/{student_id}/performance")
async def get_student_performance(
    student_id: str,
    request: Request,
    user: ExamPenUser = Depends(get_exampen_user),
) -> dict[str, Any]:
    """Cross-exam performance summary for a student.

    Students may only view their own performance.
    """
    is_evaluator = bool(_EVALUATOR_ROLES.intersection(user.exampen_roles))
    if not is_evaluator and user.user_id != student_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Students may only view their own performance",
        )

    db = await _get_tenant_db(request, user)
    coll = db["exampen_score_current"]

    pipeline = [
        {"$match": {
            "student_id": student_id,
            "tenant_id": user.tenant_id,
            "question_id": {"$ne": "__exam__"},
        }},
        {
            "$group": {
                "_id": "$exam_id",
                "total_score": {"$sum": "$score"},
                "question_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    results = await coll.aggregate(pipeline).to_list(length=500)

    exams = [
        {
            "exam_id": r["_id"],
            "total_score": round(r["total_score"], 2),
            "question_count": r["question_count"],
        }
        for r in results
    ]
    return {"student_id": student_id, "exams": exams}
