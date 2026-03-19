"""
Shared tutor scoping utilities for student visibility.

Centralises the logic for determining which students are visible
to a given tutor based on:
  1. Explicit assignment  (tutor.assigned_student_ids)
  2. Teacher mapping      (student.teacher_ids contains tutor_id)
  3. Teaching assignments  (student grade/section matches tutor config)

Every query includes an admin_id filter for multi-tenant data isolation.
"""

from typing import Dict, Any, Optional, List
import logging

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_tutor_class_criteria(
    tutor_doc: Dict[str, Any],
    admin_oid: Any,
) -> Optional[Dict[str, Any]]:
    """
    Build a MongoDB query filter to find students whose grade/section
    match the tutor's teaching assignments.

    Resolution order (first non-empty wins):
      1. ``teaching_assignments`` – per-assignment (standard, sections) pairs.
      2. ``class_teacher_of``     – single (standard, section) pair.
      3. Flat ``standards`` / ``sections`` – legacy / aggregate fields.

    Returns a query dict (always includes ``admin_id``) or ``None``
    if no class-based criteria exist on the tutor document.
    """
    if admin_oid is None:
        return None

    or_conditions: List[Dict[str, Any]] = []
    seen_pairs: set = set()  # (grade, frozenset(sections))

    # 1) Detailed teaching_assignments — most precise
    teaching_assignments = tutor_doc.get("teaching_assignments") or []
    for assignment in teaching_assignments:
        standard = assignment.get("standard")
        if not standard:
            continue
        assignment_sections = [s for s in (assignment.get("sections") or []) if s]
        pair_key = (standard, frozenset(assignment_sections))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        condition: Dict[str, Any] = {"grade": standard}
        if assignment_sections:
            condition["section"] = {"$in": assignment_sections}
        or_conditions.append(condition)

    # 2) class_teacher_of  (e.g. {"standard": "11", "section": "A"})
    class_teacher_of = tutor_doc.get("class_teacher_of") or {}
    ct_standard = class_teacher_of.get("standard")
    if ct_standard:
        ct_sections = (
            [class_teacher_of["section"]]
            if class_teacher_of.get("section")
            else []
        )
        ct_key = (ct_standard, frozenset(ct_sections))
        if ct_key not in seen_pairs:
            seen_pairs.add(ct_key)
            ct_condition: Dict[str, Any] = {"grade": ct_standard}
            if ct_sections:
                ct_condition["section"] = ct_sections[0]
            or_conditions.append(ct_condition)

    # 3) Fallback to flat standards/sections when nothing above matched
    if not or_conditions:
        standards = [s for s in (tutor_doc.get("standards") or []) if s]
        sections = [s for s in (tutor_doc.get("sections") or []) if s]
        if standards:
            condition = {"grade": {"$in": standards}}
            if sections:
                condition["section"] = {"$in": sections}
            or_conditions.append(condition)

    if not or_conditions:
        return None

    # Wrap in $or (or single condition) with admin_id
    if len(or_conditions) == 1:
        query = or_conditions[0].copy()
    else:
        query = {"$or": or_conditions}

    query["admin_id"] = admin_oid
    return query


# ---------------------------------------------------------------------------
# Full scoped-student retrieval
# ---------------------------------------------------------------------------

async def get_tutor_scoped_students(
    tutor_id: str,
    admin_oid: Any,
    db: Any,
    projection: Optional[Dict[str, Any]] = None,
    tutor_doc: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return the **deduplicated** list of student documents visible to a tutor.

    Combines three sources (all scoped by ``admin_oid``):
      1. Students explicitly assigned  (``tutor.assigned_student_ids``)
      2. Students mapped via ``student.teacher_ids``
      3. Students matching the tutor's teaching assignments (grade/section)

    Parameters
    ----------
    tutor_id : str
        The tutor's business ID (e.g. ``TUT25030001``).
    admin_oid : ObjectId | None
        The admin's ``ObjectId`` for tenant isolation.
    db : DatabaseManager
        Async database manager instance.
    projection : dict, optional
        MongoDB projection to apply to all student queries.
    tutor_doc : dict, optional
        Pre-fetched tutor document.  If ``None`` the document is fetched
        from the ``tutors`` collection using *tutor_id*.
    """
    if admin_oid is None:
        return []

    if tutor_doc is None:
        tutor_doc = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if not tutor_doc:
        return []

    students_union: List[Dict[str, Any]] = []

    # 1) Students explicitly assigned by student_id
    assigned_student_ids = tutor_doc.get("assigned_student_ids") or []
    if assigned_student_ids:
        assigned = await db.mongo_find(
            "students",
            {"student_id": {"$in": assigned_student_ids}, "admin_id": admin_oid},
            projection=projection,
        )
        students_union.extend(assigned)

    # 2) Students mapped via teacher_ids on the student document
    teacher_mapped = await db.mongo_find(
        "students",
        {"teacher_ids": {"$in": [tutor_id]}, "admin_id": admin_oid},
        projection=projection,
    )
    students_union.extend(teacher_mapped)

    # 3) Students matching the tutor's class/section assignments
    class_query = build_tutor_class_criteria(tutor_doc, admin_oid)
    if class_query:
        class_students = await db.mongo_find(
            "students",
            class_query,
            projection=projection,
        )
        students_union.extend(class_students)

    # Deduplicate by MongoDB _id
    seen: set = set()
    result: List[Dict[str, Any]] = []
    for s in students_union:
        sid = str(s.get("_id"))
        if sid not in seen:
            seen.add(sid)
            result.append(s)

    return result
