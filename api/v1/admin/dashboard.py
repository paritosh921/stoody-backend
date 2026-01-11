import logging
from typing import Dict, Any
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_database, get_cache
from .dependencies import require_admin
from .models import DashboardStats
from .utils import is_b2c_admin

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/dashboard/stats", response_model=DashboardStats)
@limiter.limit("30/minute")
async def get_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get admin dashboard statistics"""
    try:
        is_b2c = is_b2c_admin(current_user)
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        
        cache_key = f"dashboard_stats_{'b2c' if is_b2c else str(admin_id)}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return DashboardStats(**cached_stats)

        if is_b2c:
            all_users = await db.b2c_find("users", {})
            total_students = len(all_users)
            valid_students = len([s for s in all_users if s.get("is_active", True)])
            active_students = len([s for s in all_users if s.get("last_login") is not None])
            
            practice_sets = await db.b2c_find("documents", {"document_type": "Practice Sets"})
            test_series = await db.b2c_find("documents", {"document_type": "Test Series"})
            chapter_notes = await db.b2c_find("documents", {"document_type": "Chapter Notes"})
        else:
            admin_students = await db.mongo_find("students", {"admin_id": admin_id})
            total_students = len(admin_students)
            valid_students = len([s for s in admin_students if s.get("is_active", True)])
            active_students = len([s for s in admin_students if s.get("last_login") is not None])

            items = await db.mongo_find("documents", {"admin_id": admin_id}, projection={"document_type": 1})
            practice_sets = [x for x in items if x.get("document_type") == "Practice Sets"]
            test_series = [x for x in items if x.get("document_type") == "Test Series"]
            chapter_notes = [x for x in items if x.get("document_type") == "Chapter Notes"]

        stats_data = {
            "total_students": total_students,
            "valid_students": valid_students,
            "active_students": active_students,
            "practice_sets_count": len(practice_sets),
            "test_series_count": len(test_series),
            "chapter_notes_count": len(chapter_notes)
        }

        await cache.set(cache_key, stats_data, 300, "admin")
        return DashboardStats(**stats_data)

    except Exception as e:
        logger.error(f"Dashboard stats error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get dashboard statistics")

@router.get("/dashboard/school-stats")
@limiter.limit("30/minute")
async def get_school_dashboard_stats(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """
    Get school-level dashboard statistics with:
    1. Class/Division level breakdown
    2. Teacher summary (teachers per subject per class, documents uploaded)
    """
    try:
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        cache_key = f"school_dashboard_stats_{admin_id}"
        cached_stats = await cache.get(cache_key, "admin")
        if cached_stats:
            return cached_stats

        admin_students = await db.mongo_find("students", {"admin_id": admin_id})
        admin_tutors = await db.mongo_find("tutors", {"created_by": str(admin_id)})
        admin_documents = await db.mongo_find("documents", {"admin_id": admin_id})

        # ===== CLASS/DIVISION LEVEL BREAKDOWN =====
        class_division_stats = {}
        for student in admin_students:
            grade = student.get("grade", "Unknown")
            section = student.get("section", "Unknown")
            is_active = student.get("is_active", True)
            has_logged_in = student.get("last_login") is not None

            key = f"{grade}"
            if key not in class_division_stats:
                class_division_stats[key] = {
                    "grade": grade,
                    "sections": {},
                    "total_students": 0,
                    "active_students": 0,
                    "logged_in_students": 0
                }

            class_division_stats[key]["total_students"] += 1
            if is_active:
                class_division_stats[key]["active_students"] += 1
            if has_logged_in:
                class_division_stats[key]["logged_in_students"] += 1

            if section not in class_division_stats[key]["sections"]:
                class_division_stats[key]["sections"][section] = {
                    "section": section,
                    "total_students": 0,
                    "active_students": 0,
                    "logged_in_students": 0,
                    "subjects": {}
                }

            class_division_stats[key]["sections"][section]["total_students"] += 1
            if is_active:
                class_division_stats[key]["sections"][section]["active_students"] += 1
            if has_logged_in:
                class_division_stats[key]["sections"][section]["logged_in_students"] += 1

            student_subjects = student.get("subjects", []) or []
            for subj in student_subjects:
                if subj not in class_division_stats[key]["sections"][section]["subjects"]:
                    class_division_stats[key]["sections"][section]["subjects"][subj] = 0
                class_division_stats[key]["sections"][section]["subjects"][subj] += 1

        class_division_list = []
        for grade_key, grade_data in class_division_stats.items():
            sections_list = []
            for section_key, section_data in grade_data["sections"].items():
                sections_list.append({
                    "section": section_data["section"],
                    "total_students": section_data["total_students"],
                    "active_students": section_data["active_students"],
                    "logged_in_students": section_data["logged_in_students"],
                    "subjects": section_data["subjects"]
                })
            sections_list.sort(key=lambda x: x["section"])
            class_division_list.append({
                "grade": grade_data["grade"],
                "total_students": grade_data["total_students"],
                "active_students": grade_data["active_students"],
                "logged_in_students": grade_data["logged_in_students"],
                "sections": sections_list
            })
        class_division_list.sort(key=lambda x: str(x["grade"]))

        # ===== TEACHER SUMMARY =====
        teacher_summary = []
        for tutor in admin_tutors:
            tutor_id = tutor.get("tutor_id")
            tutor_name = tutor.get("name", tutor.get("username", "Unknown"))
            tutor_subjects = tutor.get("subjects", []) or []
            tutor_standards = tutor.get("standards", []) or []
            tutor_sections = tutor.get("sections", []) or []

            tutor_docs = [d for d in admin_documents if tutor_id in (d.get("teacher_ids", []) or [])]
            notes_count = len([d for d in tutor_docs if d.get("document_type") == "Chapter Notes"])
            tests_count = len([d for d in tutor_docs if d.get("document_type") == "Test Series"])
            practice_count = len([d for d in tutor_docs if d.get("document_type") == "Practice Sets"])

            assigned_ids = tutor.get("assigned_student_ids", []) or []

            teacher_summary.append({
                "tutor_id": tutor_id,
                "name": tutor_name,
                "email": tutor.get("email"),
                "subjects": tutor_subjects,
                "standards": tutor_standards,
                "sections": tutor_sections,
                "is_active": tutor.get("is_active", True),
                "assigned_students_count": len(assigned_ids),
                "documents_uploaded": {
                    "notes": notes_count,
                    "tests": tests_count,
                    "practice": practice_count,
                    "total": notes_count + tests_count + practice_count
                }
            })
        teacher_summary.sort(key=lambda x: x["name"].lower())

        # ===== DOCUMENT STATS BY CLASS =====
        docs_by_class = {}
        for doc in admin_documents:
            standard = doc.get("standard", "Unknown")
            doc_type = doc.get("document_type", "Unknown")

            if standard not in docs_by_class:
                docs_by_class[standard] = {
                    "grade": standard,
                    "Chapter Notes": 0,
                    "Test Series": 0,
                    "Practice Sets": 0
                }
            if doc_type in docs_by_class[standard]:
                docs_by_class[standard][doc_type] += 1

        docs_by_class_list = list(docs_by_class.values())
        docs_by_class_list.sort(key=lambda x: str(x["grade"]))

        result = {
            "success": True,
            "class_division_stats": class_division_list,
            "teacher_summary": teacher_summary,
            "documents_by_class": docs_by_class_list,
            "totals": {
                "total_classes": len(class_division_list),
                "total_teachers": len(teacher_summary),
                "total_students": len(admin_students),
                "total_documents": len(admin_documents)
            }
        }

        await cache.set(cache_key, result, 300, "admin")
        return result

    except Exception as e:
        logger.error(f"School dashboard stats error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get school dashboard statistics")
