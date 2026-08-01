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

def _standard_condition(standard: str) -> Dict[str, Any]:
    return {
        "$or": [
            {"grade": standard},
            {"standard": standard},
        ]
    }


def _subject_values(source: Any) -> List[str]:
    if isinstance(source, list):
        return [str(value) for value in source if value]
    if source:
        return [str(source)]
    return []


def _subject_condition(subjects: List[str]) -> Optional[Dict[str, Any]]:
    if not subjects:
        return None
    return {
        "$or": [
            {"subjects": {"$in": subjects}},
            {"subject": {"$in": subjects}},
        ]
    }


def _combine_conditions(conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


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
    seen_pairs: set = set()  # (grade, frozenset(sections), frozenset(subjects))

    # 1) Detailed teaching_assignments — most precise
    teaching_assignments = tutor_doc.get("teaching_assignments") or []
    for assignment in teaching_assignments:
        standard = assignment.get("standard")
        if not standard:
            continue
        assignment_sections = [s for s in (assignment.get("sections") or []) if s]
        assignment_subjects = _subject_values(assignment.get("subject")) + _subject_values(
            assignment.get("subjects")
        )
        pair_key = (standard, frozenset(assignment_sections), frozenset(assignment_subjects))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        conditions: List[Dict[str, Any]] = [_standard_condition(standard)]
        if assignment_sections:
            conditions.append({"section": {"$in": assignment_sections}})
        subject_condition = _subject_condition(assignment_subjects)
        if subject_condition:
            conditions.append(subject_condition)
        or_conditions.append(_combine_conditions(conditions))

    # 2) class_teacher_of  (e.g. {"standard": "11", "section": "A"})
    class_teacher_of = tutor_doc.get("class_teacher_of") or {}
    ct_standard = class_teacher_of.get("standard")
    if ct_standard:
        ct_sections = (
            [class_teacher_of["section"]]
            if class_teacher_of.get("section")
            else []
        )
        ct_key = (ct_standard, frozenset(ct_sections), frozenset())
        if ct_key not in seen_pairs:
            seen_pairs.add(ct_key)
            ct_conditions: List[Dict[str, Any]] = [_standard_condition(ct_standard)]
            if ct_sections:
                ct_conditions.append({"section": ct_sections[0]})
            or_conditions.append(_combine_conditions(ct_conditions))

    # 3) Fallback to flat standards/sections when nothing above matched
    if not or_conditions:
        standards = [s for s in (tutor_doc.get("standards") or []) if s]
        sections = [s for s in (tutor_doc.get("sections") or []) if s]
        if standards:
            conditions = [
                {
                    "$or": [
                        {"grade": {"$in": standards}},
                        {"standard": {"$in": standards}},
                    ]
                }
            ]
            if sections:
                conditions.append({"section": {"$in": sections}})
            subject_condition = _subject_condition(_subject_values(tutor_doc.get("subjects")))
            if subject_condition:
                conditions.append(subject_condition)
            or_conditions.append(_combine_conditions(conditions))

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


# ---------------------------------------------------------------------------
# Finalized paper visibility
# ---------------------------------------------------------------------------

def _normalise_scope_value(value: Any, *, standard: bool = False) -> str:
    """Return a stable comparison value for academic ownership fields."""
    normalised = " ".join(str(value or "").strip().lower().split())
    if standard:
        for prefix in ("class ", "grade ", "standard "):
            if normalised.startswith(prefix):
                normalised = normalised[len(prefix):].strip()
                break
    return normalised


def _document_matches_assignment(
    document: Dict[str, Any],
    *,
    standard: Any,
    sections: Any = None,
    subjects: Any = None,
) -> bool:
    """Match an unassigned institute paper to one teaching assignment.

    Class is mandatory for implicit sharing. Subject restrictions are enforced
    when the assignment contains subjects, while a paper without a section is
    treated as class-wide. This keeps the fallback useful for admin-created
    papers without turning a missing ``teacher_ids`` field into tenant-wide
    access.
    """
    assignment_standard = _normalise_scope_value(standard, standard=True)
    document_standard = _normalise_scope_value(
        document.get("standard") or document.get("grade"),
        standard=True,
    )
    if not assignment_standard or document_standard != assignment_standard:
        return False

    assignment_subjects = {
        _normalise_scope_value(value)
        for value in _subject_values(subjects)
        if _normalise_scope_value(value)
    }
    document_subject = _normalise_scope_value(document.get("subject"))
    if assignment_subjects and (
        not document_subject or document_subject not in assignment_subjects
    ):
        return False

    assignment_sections = {
        _normalise_scope_value(value)
        for value in _subject_values(sections)
        if _normalise_scope_value(value)
    }
    document_section = _normalise_scope_value(document.get("section"))
    if document_section and assignment_sections and document_section not in assignment_sections:
        return False

    return True


def tutor_can_access_document(
    tutor_doc: Dict[str, Any],
    document: Dict[str, Any],
    *,
    tutor_id: str,
    actor_ids: Optional[List[str]] = None,
) -> bool:
    """Return whether a tutor may use a finalized paper.

    Resolution order is deliberate:
      1. Papers explicitly assigned to or owned by the tutor are visible.
      2. Papers explicitly assigned to another teacher are not shared.
      3. Unassigned institute papers are visible only through a matching
         class/subject/section teaching assignment.
    """
    identities = {
        str(value)
        for value in ([tutor_id] + list(actor_ids or []))
        if value
    }
    teacher_ids = {
        str(value)
        for value in (document.get("teacher_ids") or [])
        if value
    }
    owner_ids = {
        str(value)
        for value in (
            document.get("uploaded_by"),
            document.get("created_by"),
            document.get("created_by_tutor_id"),
        )
        if value
    }

    if identities.intersection(teacher_ids) or identities.intersection(owner_ids):
        return True
    if teacher_ids:
        return False

    assignments = tutor_doc.get("teaching_assignments") or []
    assignment_with_class = False
    for assignment in assignments:
        standard = assignment.get("standard") or assignment.get("grade")
        if not standard:
            continue
        assignment_with_class = True
        subjects = _subject_values(assignment.get("subject")) + _subject_values(
            assignment.get("subjects")
        )
        if _document_matches_assignment(
            document,
            standard=standard,
            sections=assignment.get("sections") or assignment.get("section"),
            subjects=subjects,
        ):
            return True
    if assignment_with_class:
        return False

    class_teacher_of = tutor_doc.get("class_teacher_of") or {}
    if class_teacher_of.get("standard") or class_teacher_of.get("grade"):
        return _document_matches_assignment(
            document,
            standard=class_teacher_of.get("standard") or class_teacher_of.get("grade"),
            sections=class_teacher_of.get("section"),
        )

    standards = tutor_doc.get("standards") or []
    for standard in standards:
        if _document_matches_assignment(
            document,
            standard=standard,
            sections=tutor_doc.get("sections"),
            subjects=tutor_doc.get("subjects"),
        ):
            return True
    return False
