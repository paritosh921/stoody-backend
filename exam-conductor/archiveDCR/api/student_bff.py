"""Student BFF — aggregation endpoints for the student UI.

Routes are mounted at ``/api/v1/exampen/student``.

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
from exampen.dcr.storage.chat_repo import ChatRepo

logger = logging.getLogger(__name__)
router = APIRouter()


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
async def student_dashboard(
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("student")
    ),
) -> dict[str, Any]:
    """Aggregated student dashboard: upcoming exams, recent results, objection status.

    Returns a single payload combining data the student portal needs on load.
    """
    db = await _get_tenant_db(request, user)
    exam_repo = ExamRepo(db)
    score_store = ScoreEventStore(db)
    obj_repo = ObjectionRepo(db)

    # Upcoming exams (armed or created)
    upcoming = await exam_repo.list_exams(
        user.tenant_id,
        filters={"state": {"$in": ["created", "armed"]}},
        limit=10,
    )

    # Completed exams with published/locked scores
    published = await exam_repo.list_exams(
        user.tenant_id,
        filters={"state": {"$in": ["published", "locked"]}},
        limit=10,
    )

    # Scores summary across published exams
    score_coll = db["exampen_score_current"]
    score_pipeline = [
        {
            "$match": {
                "student_id": user.user_id,
                "tenant_id": user.tenant_id,
                "question_id": {"$ne": "__exam__"},
            }
        },
        {
            "$group": {
                "_id": "$exam_id",
                "total_score": {"$sum": "$score"},
                "question_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": -1}},
        {"$limit": 10},
    ]
    score_summary = await score_coll.aggregate(score_pipeline).to_list(length=10)

    # Student's objections
    obj_coll = db["exampen_objections"]
    my_objections = await obj_coll.find(
        {
            "student_id": user.user_id,
            "tenant_id": user.tenant_id,
        }
    ).sort("created_at", -1).limit(10).to_list(length=10)

    return {
        "upcoming_exams": upcoming,
        "published_exams": published,
        "score_summary": score_summary,
        "my_objections": my_objections,
    }


@router.get("/exams/{exam_id}")
async def exam_result(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("student")
    ),
) -> dict[str, Any]:
    """Detailed exam result for the student: scores + rank context.

    Only available when scores are published or locked.
    """
    db = await _get_tenant_db(request, user)
    exam_repo = ExamRepo(db)
    score_store = ScoreEventStore(db)
    analytics = AnalyticsCacheRepo(db)

    # Verify exam exists and scores are visible
    exam = await exam_repo.get_by_id(exam_id, user.tenant_id)
    if exam is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")

    visible_states = {"published", "objection_window", "locked"}
    if exam.get("state") not in visible_states:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Scores are not yet published for this exam",
        )

    # Student's scores
    scores = await score_store.get_current_scores(
        exam_id, user.user_id, user.tenant_id,
    )
    total = sum(r.get("score", 0.0) for r in scores)

    # Leaderboard context (limited to surrounding ranks)
    leaderboard = await analytics.get_leaderboard(exam_id, user.tenant_id)
    if not leaderboard:
        # Compute on-the-fly
        overview = await score_store.get_exam_overview(exam_id, user.tenant_id)
        overview.sort(key=lambda r: r.get("total_score", 0), reverse=True)
        leaderboard = [
            {"student_id": r["student_id"], "total_score": r["total_score"], "rank": i + 1}
            for i, r in enumerate(overview)
        ]

    # Find student's rank
    own_rank = None
    own_idx = None
    for i, entry in enumerate(leaderboard):
        if entry.get("student_id") == user.user_id:
            own_rank = entry.get("rank", i + 1)
            own_idx = i
            break

    # Surrounding context (2 above, 2 below)
    rank_context = []
    if own_idx is not None:
        start = max(0, own_idx - 2)
        end = min(len(leaderboard), own_idx + 3)
        rank_context = leaderboard[start:end]

    # Class stats
    class_stats = await analytics.get_class_stats(exam_id, user.tenant_id)

    return {
        "exam_id": exam_id,
        "exam_title": exam.get("title", ""),
        "student_id": user.user_id,
        "total_score": round(total, 2),
        "scores": scores,
        "rank": own_rank,
        "rank_context": rank_context,
        "class_stats": class_stats,
    }


@router.get("/exams/{exam_id}/objections")
async def my_exam_objections(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("student")
    ),
) -> dict[str, Any]:
    """List the student's own objections for an exam."""
    db = await _get_tenant_db(request, user)
    obj_coll = db["exampen_objections"]

    items = await obj_coll.find(
        {
            "exam_id": exam_id,
            "student_id": user.user_id,
            "tenant_id": user.tenant_id,
        }
    ).sort("created_at", -1).to_list(length=50)

    return {"exam_id": exam_id, "objections": items}


@router.get("/exams/{exam_id}/chat")
async def my_exam_chat(
    exam_id: str,
    request: Request,
    user: ExamPenUser = Depends(
        require_exampen_role("student")
    ),
) -> dict[str, Any]:
    """List chat threads the student has with teachers for an exam.

    Returns all messages where the student is a party.
    """
    db = await _get_tenant_db(request, user)
    chat_coll = db["exampen_chat_messages"]

    messages = await chat_coll.find(
        {
            "exam_id": exam_id,
            "student_id": user.user_id,
            "tenant_id": user.tenant_id,
        }
    ).sort("created_at", 1).to_list(length=1000)

    # Group by teacher_id
    threads: dict[str, list[dict]] = {}
    for m in messages:
        tid = m.get("teacher_id", "unknown")
        threads.setdefault(tid, []).append(m)

    return {
        "exam_id": exam_id,
        "threads": [
            {"teacher_id": tid, "messages": msgs}
            for tid, msgs in threads.items()
        ],
    }
