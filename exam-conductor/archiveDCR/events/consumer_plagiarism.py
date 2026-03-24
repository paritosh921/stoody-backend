"""NATS consumer: plagiarism detection pipeline.

Subscribes to ``EXAMPEN.plagiarism.check``, runs similarity analysis
across student answers for the same question, writes flags to
``plagiarism_repo``, and publishes ``EXAMPEN.plagiarism.result``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from uuid import uuid4

from ..domain.structural_similarity import compute_structural_similarity
from ..domain.text_similarity import compute_tfidf_similarity
from ..storage.ai_result_repo import AIResultRepo
from ..storage.plagiarism_repo import PlagiarismRepo
from . import subjects
from .publishers import _envelope, _safe_publish

logger = logging.getLogger(__name__)

DURABLE = "exampen-plagiarism"
QUEUE_GROUP = "exampen-plagiarism-workers"

# Thresholds for flagging
STRUCTURAL_THRESHOLD = 0.80
TFIDF_THRESHOLD = 0.85
COMBINED_THRESHOLD = 0.82


def _compute_combined_similarity(text_a: str, text_b: str) -> Dict[str, float]:
    """Run both similarity algorithms and return scores."""
    structural = compute_structural_similarity(text_a, text_b)
    tfidf = compute_tfidf_similarity(text_a, text_b)
    combined = (structural * 0.5) + (tfidf * 0.5)
    return {
        "structural": round(structural, 4),
        "tfidf": round(tfidf, 4),
        "combined": round(combined, 4),
    }


async def plagiarism_handler(
    payload: Dict[str, Any],
    nats: Any,
    db_manager: Any,
) -> None:
    """Run pairwise plagiarism analysis for an exam question.

    Steps:
        1. Fetch all AI results for the exam
        2. Group recognized text by question
        3. Run pairwise similarity for each question
        4. Flag pairs exceeding the threshold
        5. Persist flags and publish result
    """
    event_id = payload.get("event_id", "unknown")
    exam_id = payload.get("exam_id", "")
    tenant_id = payload.get("tenant_id", "")
    question_ids = payload.get("question_ids", [])  # empty = all

    logger.info(
        "Processing plagiarism.check event_id=%s exam=%s questions=%s",
        event_id, exam_id, question_ids or "all",
    )

    db = await db_manager.get_tenant_db(tenant_id)
    ai_repo = AIResultRepo(db)
    plag_repo = PlagiarismRepo(db)

    # Collect all student answers by question_id
    # Note: in production, a dedicated collection query would be more efficient
    answers_by_q: Dict[str, List[Dict[str, str]]] = {}

    # Fetch a broad set of AI results for this exam
    # (simplified: iterate all students from a lookup; in reality use an index)
    all_results = await ai_repo._coll.find(
        {"exam_id": exam_id, "tenant_id": tenant_id}
    ).to_list(length=5000)

    for doc in all_results:
        student_id = doc.get("student_id", "")
        for qr in doc.get("question_results", []):
            qid = qr.get("question_id", "")
            if question_ids and qid not in question_ids:
                continue
            text = qr.get("recognized_text", "")
            if text:
                answers_by_q.setdefault(qid, []).append({
                    "student_id": student_id,
                    "text": text,
                })

    # Run pairwise comparisons
    flags: List[Dict[str, Any]] = []
    for qid, answers in answers_by_q.items():
        n = len(answers)
        for i in range(n):
            for j in range(i + 1, n):
                sim = _compute_combined_similarity(
                    answers[i]["text"], answers[j]["text"]
                )
                if sim["combined"] >= COMBINED_THRESHOLD:
                    flags.append({
                        "exam_id": exam_id,
                        "question_id": qid,
                        "student_a": answers[i]["student_id"],
                        "student_b": answers[j]["student_id"],
                        "similarity_structural": sim["structural"],
                        "similarity_tfidf": sim["tfidf"],
                        "similarity_combined": sim["combined"],
                        "flag_type": "automated",
                    })

    # Persist flags
    if flags:
        await plag_repo.bulk_insert(flags, tenant_id)

    # Publish result
    result_event = _envelope("plagiarism.result", 1, {
        "exam_id": exam_id,
        "tenant_id": tenant_id,
        "flags_count": len(flags),
        "questions_checked": len(answers_by_q),
    })
    await _safe_publish(nats, subjects.PLAGIARISM_RESULT, result_event)

    logger.info(
        "Plagiarism check complete event_id=%s flags=%d questions=%d",
        event_id, len(flags), len(answers_by_q),
    )


async def register(nats: Any, db_manager: Any) -> None:
    """Subscribe to EXAMPEN.plagiarism.check with durable consumer."""
    async def _handler(payload: Dict[str, Any]) -> None:
        await plagiarism_handler(payload, nats, db_manager)

    await nats.subscribe(
        subjects.PLAGIARISM_CHECK,
        _handler,
        queue_group=QUEUE_GROUP,
        durable=DURABLE,
    )
    logger.info("Registered plagiarism_consumer")
