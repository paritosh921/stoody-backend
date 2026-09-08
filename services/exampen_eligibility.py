"""Current student eligibility for PDF/camera collections, without DB backfills.

The legacy `roster` wire field remains for older clients and pen sessions. For
camera PCR exams it is a read-time projection, never an admission snapshot.
"""
from typing import Any, Dict, Iterable

from bson import ObjectId


def uses_live_students(exam: Dict[str, Any]) -> bool:
    return exam.get("exam_type") == "pcr" and exam.get("capture_mode") == "camera"


def owner_values(owner: Any) -> list:
    values = [owner, str(owner)] if owner is not None else []
    if owner is not None and ObjectId.is_valid(str(owner)):
        values.append(ObjectId(str(owner)))
    return list(dict.fromkeys(values))


class ExamEligibility:
    """Request-local cache: no stale cross-request enrolment cache."""

    def __init__(self, tenant_db: Any):
        self.db = tenant_db

    async def resolve_many(self, exams: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
        exams = list(exams)
        ids = list({str(e.get("prepared_document_id")) for e in exams
                    if uses_live_students(e) and e.get("prepared_document_id")})
        papers = await self.db["documents"].find(
            {"document_id": {"$in": ids}},
            {"document_id": 1, "admin_id": 1, "standard": 1, "section": 1},
        ).to_list(length=None) if ids else []
        by_id = {str(p["document_id"]): p for p in papers}
        class_students = {}
        result = []
        for exam in exams:
            if not uses_live_students(exam):
                result.append(dict(exam))
                continue
            paper = by_id.get(str(exam.get("prepared_document_id"))) or {}
            owner = exam.get("admin_id")
            grade = str(paper.get("standard") or "").strip()
            section = str(paper.get("section") or "").strip()
            # Missing or inconsistent ownership/class never falls back to the
            # saved roster or opens admission to the entire tenant.
            if owner is None or str(owner) != str(paper.get("admin_id")) or not grade:
                result.append({**exam, "roster": [], "student_selection_mode": "live_class"})
                continue
            key = (str(owner), grade, section)
            if key not in class_students:
                query = {"admin_id": {"$in": owner_values(owner)}, "grade": grade, "is_active": True}
                if section:
                    query["section"] = section
                students = await self.db["students"].find(
                    query, {"student_id": 1},
                ).to_list(length=None)
                class_students[key] = list(dict.fromkeys(
                    str(s["student_id"]).strip() for s in students if s.get("student_id")
                ))
            result.append({**exam, "roster": class_students[key], "student_selection_mode": "live_class"})
        return result

    async def resolve(self, exam: Dict[str, Any]) -> Dict[str, Any]:
        return (await self.resolve_many([exam]))[0]


async def resolve_exam_students(tenant_db: Any, exam: Dict[str, Any]) -> Dict[str, Any]:
    return await ExamEligibility(tenant_db).resolve(exam)


async def eligible_exam_ids_for_students(tenant_db: Any, student_ids: list[str]) -> list[str]:
    """Teacher discovery uses the same live membership as detail/upload checks."""
    if not student_ids:
        return []
    exams = await tenant_db["exampen_exams"].find(
        {"exam_type": "pcr", "capture_mode": "camera"},
        {"exam_id": 1, "exam_type": 1, "capture_mode": 1, "admin_id": 1, "prepared_document_id": 1},
    ).to_list(length=None)
    allowed = set(student_ids)
    return [e["exam_id"] for e in await ExamEligibility(tenant_db).resolve_many(exams)
            if e.get("exam_id") and allowed.intersection(e.get("roster") or [])]


async def student_exam_visibility(tenant_db: Any, student_id: str) -> Dict[str, Any]:
    """Filter before pagination; include historical copies for download access."""
    student = await tenant_db["students"].find_one({"student_id": student_id})
    paper_ids = []
    if student and student.get("is_active") is True and student.get("admin_id") is not None and student.get("grade"):
        papers = await tenant_db["documents"].find({
            "admin_id": {"$in": owner_values(student["admin_id"])},
            "standard": str(student["grade"]),
            "$or": [{"section": {"$in": [None, ""]}}, {"section": student.get("section")}],
        }, {"document_id": 1}).to_list(length=None)
        paper_ids = [p["document_id"] for p in papers if p.get("document_id")]
    history = await tenant_db["evalpen_submissions"].find(
        {"student_id": student_id}, {"exam_id": 1},
    ).to_list(length=None)
    return {"$or": [
        {"capture_mode": "camera", "prepared_document_id": {"$in": paper_ids},
         "admin_id": {"$in": owner_values((student or {}).get("admin_id"))}},
        {"capture_mode": {"$ne": "camera"}, "roster": student_id},
        {"exam_id": {"$in": [h["exam_id"] for h in history if h.get("exam_id")]}},
    ]}
