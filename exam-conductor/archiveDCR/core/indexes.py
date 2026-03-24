"""MongoDB index creation for all ExamPen collections.

Call ``ensure_exampen_indexes(db)`` once per tenant database (typically at
first access) to create the necessary indexes.  All indexes are created
with ``background=True`` so they don't block normal operations.

The function is idempotent — ``create_index`` is a no-op when the index
already exists with identical spec.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger(__name__)

# Type alias for index specs: (collection_name, keys, unique, sparse, name)
_IndexSpec = Tuple[str, List[Tuple[str, int]], bool, bool, str]

# ---------------------------------------------------------------------------
# Index catalog — one entry per index across all 16 ExamPen collections
# ---------------------------------------------------------------------------

_EXAMPEN_INDEXES: List[_IndexSpec] = [
    # --- exampen_exams ---
    ("exampen_exams", [("tenant_id", ASCENDING), ("state", ASCENDING)],
     False, False, "idx_ep_exams_tenant_state"),
    ("exampen_exams", [("exam_id", ASCENDING)],
     True, False, "uniq_ep_exams_exam_id"),
    ("exampen_exams", [("tenant_id", ASCENDING), ("created_at", ASCENDING)],
     False, False, "idx_ep_exams_tenant_created"),

    # --- exampen_bindings ---
    ("exampen_bindings", [("exam_id", ASCENDING), ("pen_mac", ASCENDING)],
     True, False, "uniq_ep_bindings_exam_pen"),
    ("exampen_bindings", [("exam_id", ASCENDING), ("student_id", ASCENDING)],
     True, False, "uniq_ep_bindings_exam_student"),

    # --- exampen_strokes_raw ---
    ("exampen_strokes_raw",
     [("exam_id", ASCENDING), ("pen_mac", ASCENDING), ("chunk_index", ASCENDING)],
     True, False, "uniq_ep_strokes_raw_chunk"),
    ("exampen_strokes_raw", [("exam_id", ASCENDING), ("uploaded_at", ASCENDING)],
     False, False, "idx_ep_strokes_raw_exam_time"),

    # --- exampen_strokes_processed ---
    ("exampen_strokes_processed",
     [("idempotency_key", ASCENDING), ("stroke_id", ASCENDING)],
     True, False, "uniq_ep_strokes_proc_idem"),
    ("exampen_strokes_processed",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("page_number", ASCENDING)],
     False, False, "idx_ep_strokes_proc_exam_student_page"),

    # --- exampen_pages ---
    ("exampen_pages",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("page_number", ASCENDING)],
     True, False, "uniq_ep_pages_exam_student_page"),

    # --- exampen_score_events ---
    ("exampen_score_events",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("created_at", ASCENDING)],
     False, False, "idx_ep_score_events_exam_student_time"),
    ("exampen_score_events", [("event_id", ASCENDING)],
     True, False, "uniq_ep_score_events_event_id"),

    # --- exampen_score_current ---
    ("exampen_score_current",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("question_id", ASCENDING)],
     True, False, "uniq_ep_score_current_exam_student_q"),

    # --- exampen_assignments ---
    ("exampen_assignments",
     [("exam_id", ASCENDING), ("user_id", ASCENDING), ("role", ASCENDING)],
     True, False, "uniq_ep_assignments_exam_user_role"),
    ("exampen_assignments", [("user_id", ASCENDING), ("is_active", ASCENDING)],
     False, False, "idx_ep_assignments_user_active"),

    # --- exampen_objections ---
    ("exampen_objections", [("exam_id", ASCENDING), ("student_id", ASCENDING)],
     False, False, "idx_ep_objections_exam_student"),
    ("exampen_objections", [("objection_id", ASCENDING)],
     True, False, "uniq_ep_objections_objection_id"),
    ("exampen_objections", [("state", ASCENDING), ("created_at", ASCENDING)],
     False, False, "idx_ep_objections_state_time"),

    # --- exampen_plagiarism ---
    ("exampen_plagiarism",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("question_id", ASCENDING)],
     True, False, "uniq_ep_plagiarism_exam_student_q"),

    # --- exampen_ai_results ---
    ("exampen_ai_results",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("page_number", ASCENDING)],
     False, False, "idx_ep_ai_results_exam_student_page"),
    ("exampen_ai_results", [("result_id", ASCENDING)],
     True, False, "uniq_ep_ai_results_result_id"),

    # --- exampen_rubrics ---
    ("exampen_rubrics", [("exam_id", ASCENDING), ("question_id", ASCENDING)],
     True, False, "uniq_ep_rubrics_exam_question"),

    # --- exampen_questions ---
    ("exampen_questions", [("exam_id", ASCENDING), ("question_number", ASCENDING)],
     True, False, "uniq_ep_questions_exam_qnum"),

    # --- exampen_analytics ---
    ("exampen_analytics",
     [("exam_id", ASCENDING), ("metric_type", ASCENDING)],
     False, False, "idx_ep_analytics_exam_metric"),

    # --- exampen_chat_messages ---
    ("exampen_chat_messages",
     [("exam_id", ASCENDING), ("student_id", ASCENDING), ("created_at", ASCENDING)],
     False, False, "idx_ep_chat_exam_student_time"),

    # --- exampen_notifications ---
    ("exampen_notifications",
     [("tenant_id", ASCENDING), ("user_id", ASCENDING), ("created_at", ASCENDING)],
     False, False, "idx_ep_notif_tenant_user_time"),
    ("exampen_notifications", [("is_read", ASCENDING), ("created_at", ASCENDING)],
     False, False, "idx_ep_notif_unread_time"),

    # --- exampen_copy_uploads ---
    ("exampen_copy_uploads",
     [("exam_id", ASCENDING), ("student_id", ASCENDING)],
     True, False, "uniq_ep_copy_uploads_exam_student"),
]


async def ensure_exampen_indexes(
    db: AsyncIOMotorDatabase,
    *,
    _already_indexed: Optional[set] = None,
) -> None:
    """Create all ExamPen indexes on *db* (idempotent).

    Parameters
    ----------
    db:
        A Motor ``AsyncIOMotorDatabase`` instance (tenant database).
    _already_indexed:
        Optional set used to track which databases have already been
        indexed within this process lifetime.  Pass the same ``set``
        across calls to avoid redundant work.
    """
    if _already_indexed is not None:
        db_name = db.name
        if db_name in _already_indexed:
            return
        _already_indexed.add(db_name)

    created = 0
    skipped = 0

    for collection_name, keys, unique, sparse, name in _EXAMPEN_INDEXES:
        try:
            kwargs: Dict[str, Any] = {"name": name, "background": True}
            if unique:
                kwargs["unique"] = True
            if sparse:
                kwargs["sparse"] = True

            await db[collection_name].create_index(keys, **kwargs)
            created += 1
        except Exception as e:
            # Index may already exist with different options — log and continue
            logger.warning(
                "Could not create index %s on %s.%s: %s",
                name,
                db.name,
                collection_name,
                e,
            )
            skipped += 1

    logger.info(
        "ExamPen indexes on %s: %d created/verified, %d skipped",
        db.name,
        created,
        skipped,
    )
