"""Generate AI results, scores, objections, plagiarism flags, and chat."""

from __future__ import annotations

import random
import uuid

from .constants import OBJECTION_STATES, SCORE_FSM_STATES
from .helpers import iso_now


def gen_ai_results(
    rng: random.Random,
    student: dict,
    exam: dict,
) -> list[dict]:
    """Generate AI recognition results per question."""
    results = []
    for q in exam["questions"]:
        confidence = round(rng.uniform(0.65, 0.98), 3)
        step_results = []
        for s, mark in enumerate(q["step_marks"]):
            step_results.append({
                "step": s + 1,
                "max_marks": mark,
                "ai_score": rng.randint(0, mark),
                "confidence": round(rng.uniform(0.60, 0.99), 3),
                "recognized_text": f"Step {s+1} recognized text",
            })
        results.append({
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "student_id": student["id"],
            "exam_id": exam["id"],
            "question_number": q["question_number"],
            "confidence": confidence,
            "recognized_text": f"[Simulated HWR output for Q{q['question_number']}]",
            "is_diagram": rng.random() < 0.1,
            "model_version": "hwr-v0.3.1",
            "step_results": step_results,
            "total_ai_score": sum(sr["ai_score"] for sr in step_results),
            "max_marks": q["max_marks"],
            "processed_at": iso_now(-5),
        })
    return results


def gen_scores(
    rng: random.Random,
    ai_results: list[dict],
    exam: dict,
    student_idx: int,
) -> list[dict]:
    """Generate scores from AI results, some with teacher overrides."""
    override_reasons = [
        "Partial credit for correct approach",
        "Handwriting unclear but answer correct",
        "AI missed diagram annotation",
        "Step marks adjusted after review",
    ]
    scores = []
    for ar in ai_results:
        has_override = rng.random() < 0.20
        state_idx = rng.randint(1, 3) if has_override else 0
        state = SCORE_FSM_STATES[state_idx]

        teacher_score = None
        override_reason = None
        if has_override:
            teacher_score = rng.randint(0, ar["max_marks"])
            override_reason = rng.choice(override_reasons)

        final = teacher_score if teacher_score is not None else ar["total_ai_score"]
        scores.append({
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "student_id": ar["student_id"],
            "exam_id": ar["exam_id"],
            "question_number": ar["question_number"],
            "ai_score": ar["total_ai_score"],
            "teacher_score": teacher_score,
            "final_score": final,
            "max_marks": ar["max_marks"],
            "state": state,
            "override_reason": override_reason,
            "ai_result_id": ar["id"],
            "created_at": iso_now(-4),
            "updated_at": iso_now(-2) if has_override else iso_now(-4),
        })
    return scores


def gen_objections(rng: random.Random, scores: list[dict]) -> list[dict]:
    """Generate 5 objections in various FSM states."""
    reasons = [
        "AI did not recognize my diagram correctly",
        "I believe step 2 deserves partial credit",
        "The handwriting recognition missed key terms",
        "My answer matches the rubric criteria for full marks",
        "The scoring seems inconsistent with other students",
    ]
    resolutions = [
        "Score adjusted after re-evaluation",
        "Original AI score confirmed after manual review",
        "Partial credit awarded for correct approach",
    ]
    eligible = [s for s in scores if s["state"] in ("ai_draft", "teacher_reviewed")]
    if len(eligible) < 5:
        eligible = scores[:5]
    selected = rng.sample(eligible, min(5, len(eligible)))

    objections = []
    for i, score in enumerate(selected):
        state = OBJECTION_STATES[i % len(OBJECTION_STATES)]
        obj = {
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "score_id": score["id"],
            "student_id": score["student_id"],
            "exam_id": score["exam_id"],
            "question_number": score["question_number"],
            "reason": rng.choice(reasons),
            "state": state,
            "filed_at": iso_now(-3),
            "assigned_to": None,
            "resolution": None,
            "resolved_at": None,
        }
        if state in ("assigned", "reviewing", "resolved"):
            obj["assigned_to"] = f"tut_000{rng.randint(1,5)}"
        if state == "resolved":
            obj["resolution"] = rng.choice(resolutions)
            obj["resolved_at"] = iso_now(-1)
        objections.append(obj)

    if len(objections) >= 4:
        for i, st in enumerate(["filed", "reviewing", "resolved", "resolved"]):
            if i < len(objections):
                objections[i]["state"] = st
    return objections


def gen_plagiarism_flags(
    rng: random.Random,
    students: list[dict],
    exam: dict,
) -> list[dict]:
    """Generate 2 plagiarism flags with composite scores above threshold."""
    flags = []
    if len(students) < 4:
        return flags
    for i in range(2):
        s1 = students[i * 2]
        s2 = students[i * 2 + 1]
        tfidf = round(rng.uniform(0.86, 0.97), 3)
        structural = round(rng.uniform(0.80, 0.95), 3)
        composite = round(0.6 * tfidf + 0.4 * structural, 3)
        flags.append({
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "exam_id": exam["id"],
            "question_number": rng.randint(1, len(exam["questions"])),
            "student_a_id": s1["id"],
            "student_b_id": s2["id"],
            "tfidf_cosine_similarity": tfidf,
            "structural_similarity": structural,
            "composite_score": composite,
            "threshold": 0.85,
            "flagged": composite > 0.85,
            "teacher_verdict": None,
            "verdict_reason": None,
            "detected_at": iso_now(-2),
        })
    return flags


def gen_chat_messages(
    rng: random.Random,
    students: list[dict],
    tutors: list[dict],
    exam: dict,
) -> list[dict]:
    """Generate sample teacher-student chat threads."""
    messages = []
    for thread_idx in range(min(3, len(students))):
        student = students[thread_idx]
        tutor = tutors[thread_idx % len(tutors)]
        thread_id = str(uuid.UUID(int=rng.getrandbits(128)))
        msg_count = rng.randint(2, 4)
        templates = [
            (student["id"], "student",
             f"Sir, I have a doubt about Q{rng.randint(1,5)} scoring."),
            (tutor["id"], "tutor",
             "Sure, let me check. Which step are you referring to?"),
            (student["id"], "student",
             "Step 2 - I used the correct formula but got partial marks."),
            (tutor["id"], "tutor",
             "I'll review and update if needed. Thanks for raising this."),
        ]
        for m in range(msg_count):
            sender_id, role, text = templates[m % len(templates)]
            messages.append({
                "id": str(uuid.UUID(int=rng.getrandbits(128))),
                "thread_id": thread_id,
                "exam_id": exam["id"],
                "sender_id": sender_id,
                "sender_role": role,
                "text": text,
                "created_at": iso_now(-2, offset_hours=m),
                "read_at": iso_now(-1) if m < msg_count - 1 else None,
            })
    return messages
