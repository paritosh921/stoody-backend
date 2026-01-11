"""
Student-facing PDF endpoints.
"""

import logging
from typing import Any, Dict, Optional

from bson import ObjectId as BsonObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_database
from api.v1.pdf_schemas import DocumentListResponse, DocumentMetadata
from api.v1.student_async import require_student_or_admin
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/student/practice-sets")
@limiter.limit("30/minute")
async def get_student_practice_sets(
    request: Request,
    plan_type: Optional[str] = Query(None, description="Filter by course plan type"),
    subject: Optional[str] = Query(None, description="Filter by subject"),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get practice sets available for the current student based on their profile."""
    try:
        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        if is_b2c:
            b2c_user = await db.b2c_find_one("users", {"_id": BsonObjectId(current_user["user_id"])})

            if not b2c_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="B2C user profile not found"
                )

            if not b2c_user.get("onboarding_complete"):
                return {
                    "success": True,
                    "data": {
                        "practice_sets": [],
                        "total": 0,
                        "onboarding_required": True
                    }
                }

            user_exam_type = b2c_user.get("exam_type")
            user_class_level = b2c_user.get("class_level")
            user_standard = b2c_user.get("standard")
            user_subjects = b2c_user.get("subjects", [])
            user_plan_types = b2c_user.get("plan_types", [])

            b2c_admin = await db.b2c_find_one("admins", {}, {"_id": 1})
            b2c_admin_id = b2c_admin["_id"] if b2c_admin else None

            filter_query = {
                "document_type": "Practice Sets",
                "ocr_status": "completed",
                "is_active": {"$ne": False}
            }

            if b2c_admin_id:
                try:
                    filter_query["admin_id"] = BsonObjectId(b2c_admin_id)
                except Exception:
                    filter_query["admin_id"] = b2c_admin_id

            if plan_type:
                filter_query["course_plan"] = plan_type
            elif user_plan_types:
                filter_query["course_plan"] = {"$in": user_plan_types}
            elif user_exam_type:
                filter_query["course_plan"] = user_exam_type

            if subject:
                filter_query["subject"] = subject
            elif user_subjects:
                filter_query["subject"] = {"$in": user_subjects}

            if user_standard:
                filter_query["standard"] = user_standard

            logger.info("B2C user %s practice sets query: %s", current_user["user_id"], filter_query)

            practice_sets = await db.b2c_find(
                "documents",
                filter_query,
                sort=[("uploaded_at", -1)]
            )

            logger.info("B2C practice sets found: %s", len(practice_sets))

            practice_sets_list = []
            user_id = current_user["user_id"]

            for doc in practice_sets:
                doc_id = doc.get("document_id") or str(doc.get("_id"))

                sessions = await db.b2c_find(
                    "practice_sessions",
                    {
                        "student_id": user_id,
                        "document_id": doc_id
                    },
                    sort=[("started_at", -1)],
                    limit=10
                )

                has_attempted = len(sessions) > 0
                completed = any(session.get("is_completed", False) for session in sessions)

                practice_sets_list.append({
                    "document_id": doc_id,
                    "title": doc.get("title"),
                    "subject": doc.get("subject"),
                    "difficulty": doc.get("difficulty"),
                    "course_plan": doc.get("course_plan"),
                    "standard": doc.get("standard"),
                    "extracted_questions_count": doc.get("extracted_questions_count", 0),
                    "completed": completed,
                    "attempted": has_attempted,
                    "session_count": len(sessions)
                })

            return {
                "success": True,
                "data": {
                    "practice_sets": practice_sets_list,
                    "total": len(practice_sets_list)
                }
            }

        if user_type == "admin":
            filter_query = {"document_type": "Practice Sets"}

            if plan_type:
                filter_query["course_plan"] = plan_type

            if subject:
                filter_query["subject"] = subject

            practice_sets = await db.mongo_find(
                "documents",
                filter_query,
                sort=[("uploaded_at", -1)]
            )

            practice_sets_list = []
            user_id = current_user["user_id"]

            for doc in practice_sets:
                doc_id = doc["document_id"]

                sessions = await db.mongo_find(
                    "practice_sessions",
                    {
                        "student_id": user_id,
                        "document_id": doc_id
                    },
                    sort=[("started_at", -1)],
                    limit=10
                )

                has_attempted = len(sessions) > 0
                completed = any(session.get("is_completed", False) for session in sessions)

                practice_sets_list.append({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "subject": doc["subject"],
                    "difficulty": doc["difficulty"],
                    "course_plan": doc.get("course_plan"),
                    "standard": doc.get("standard"),
                    "extracted_questions_count": doc.get("extracted_questions_count", 0),
                    "completed": completed,
                    "attempted": has_attempted,
                    "session_count": len(sessions)
                })

            return {
                "success": True,
                "data": {
                    "practice_sets": practice_sets_list,
                    "total": len(practice_sets_list)
                }
            }

        student_profile = await db.mongo_find_one("students", {"_id": BsonObjectId(current_user["user_id"])})

        if not student_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found"
            )

        student_grade = student_profile.get("grade")
        student_subjects = student_profile.get("subjects", [])
        student_plan_types = student_profile.get("plan_types", [])
        student_section = student_profile.get("section")
        student_teacher_ids = student_profile.get("teacher_ids", [])

        filter_query = {
            "document_type": "Practice Sets",
            "ocr_status": "completed",
            "is_active": {"$ne": False}
        }

        admin_id = current_user.get("admin_id")
        if admin_id:
            try:
                admin_oid = BsonObjectId(admin_id)
                admin_filter = {"$in": [admin_oid, admin_id]}
            except Exception:
                admin_filter = admin_id
            filter_query["admin_id"] = admin_filter

        if plan_type:
            filter_query["course_plan"] = plan_type
        elif student_plan_types:
            filter_query["course_plan"] = {"$in": student_plan_types}

        if subject:
            filter_query["subject"] = subject
        elif student_subjects:
            filter_query["subject"] = {"$in": student_subjects}

        and_conditions = []

        if student_grade:
            filter_query["standard"] = student_grade

        if student_section:
            and_conditions.append({
                "$or": [
                    {"section": student_section},
                    {"section": None},
                    {"section": {"$exists": False}}
                ]
            })

        if student_teacher_ids:
            and_conditions.append({
                "$or": [
                    {"teacher_ids": {"$in": student_teacher_ids}},
                    {"teacher_ids": []},
                    {"teacher_ids": None},
                    {"teacher_ids": {"$exists": False}}
                ]
            })

        if and_conditions:
            and_conditions.insert(0, filter_query)
            filter_query = {"$and": and_conditions}

        logger.info(
            "Student profile - Grade: %s, Subjects: %s, Plan Types: %s, Section: %s, Teacher IDs: %s",
            student_grade,
            student_subjects,
            student_plan_types,
            student_section,
            student_teacher_ids
        )
        logger.info("Practice sets filter query: %s", filter_query)

        practice_sets = await db.mongo_find(
            "documents",
            filter_query,
            sort=[("uploaded_at", -1)]
        )

        logger.info("Found %s practice sets matching filter", len(practice_sets))

        practice_sets_list = []
        user_id = current_user["user_id"]

        for doc in practice_sets:
            doc_id = doc["document_id"]

            sessions = await db.mongo_find(
                "practice_sessions",
                {
                    "student_id": user_id,
                    "document_id": doc_id
                },
                sort=[("started_at", -1)],
                limit=10
            )

            has_attempted = len(sessions) > 0
            completed = any(session.get("is_completed", False) for session in sessions)

            latest_session = None
            if sessions:
                latest = sessions[0]
                accuracy_rate = 0.0
                if latest.get("questions_attempted", 0) > 0:
                    accuracy_rate = (latest.get("correct_answers", 0) / latest["questions_attempted"]) * 100

                latest_session = {
                    "questions_attempted": latest.get("questions_attempted", 0),
                    "correct_answers": latest.get("correct_answers", 0),
                    "accuracy_rate": round(accuracy_rate, 1),
                    "started_at": latest.get("started_at").isoformat() if latest.get("started_at") else None,
                    "is_completed": latest.get("is_completed", False)
                }

            practice_sets_list.append({
                "document_id": doc_id,
                "title": doc["title"],
                "subject": doc["subject"],
                "difficulty": doc["difficulty"],
                "course_plan": doc.get("course_plan"),
                "standard": doc.get("standard"),
                "extracted_questions_count": doc.get("extracted_questions_count", 0),
                "completed": completed,
                "attempted": has_attempted,
                "session_count": len(sessions),
                "latest_session": latest_session
            })

        return {
            "success": True,
            "data": {
                "practice_sets": practice_sets_list,
                "total": len(practice_sets_list)
            }
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Get student practice sets error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get practice sets"
        )


@router.get("/student/available-options")
@limiter.limit("30/minute")
async def get_student_available_options(
    request: Request,
    document_type: Optional[str] = Query(None, description="Document type (Practice Sets or Test Series)"),
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get available course plans, subjects, and other options for the student based on admin's uploaded content."""
    try:
        admin_id = current_user.get("admin_id") if current_user.get("user_type") == "student" else current_user.get("user_id")

        try:
            admin_oid = BsonObjectId(admin_id)
            admin_filter = {"$in": [admin_oid, admin_id]}
        except Exception:
            admin_filter = admin_id

        filter_query = {"admin_id": admin_filter, "is_active": {"$ne": False}}
        if document_type:
            filter_query["document_type"] = document_type

        documents = await db.mongo_find("documents", filter_query)

        course_plans = set()
        subjects = set()
        standards = set()

        for doc in documents:
            if doc.get("course_plan"):
                course_plans.add(doc["course_plan"])
            if doc.get("subject"):
                subjects.add(doc["subject"])
            if doc.get("standard"):
                standards.add(doc["standard"])

        return {
            "success": True,
            "data": {
                "course_plans": sorted(list(course_plans)),
                "subjects": sorted(list(subjects)),
                "standards": sorted(list(standards))
            }
        }

    except Exception as exc:
        logger.error("Get available options error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available options"
        )

    try:
        filter_query = {}
        if document_type:
            filter_query["document_type"] = document_type

        total = len(await db.mongo_find("documents", filter_query))

        skip = (page - 1) * limit
        documents = await db.mongo_find(
            "documents",
            filter_query,
            skip=skip,
            limit=limit,
            sort=[("uploaded_at", -1)]
        )

        from pathlib import Path
        document_list = []
        for doc in documents:
            file_path = Path(doc["file_path"])
            file_exists = file_path.exists()

            document_list.append(DocumentMetadata(
                document_id=doc["document_id"],
                title=doc["title"],
                document_type=doc["document_type"],
                subject=doc["subject"],
                difficulty=doc["difficulty"],
                course_plan=doc.get("course_plan"),
                standard=doc.get("standard"),
                file_path=doc["file_path"],
                filename=doc["filename"],
                uploaded_by=doc["uploaded_by"],
                uploaded_at=doc["uploaded_at"],
                ocr_status=doc["ocr_status"],
                ocr_job_id=doc.get("ocr_job_id"),
                extracted_questions_count=doc.get("extracted_questions_count", 0),
                extracted_images_count=doc.get("extracted_images_count", 0),
                file_exists=file_exists
            ))

        return DocumentListResponse(
            documents=document_list,
            total=total,
            page=page,
            limit=limit
        )

    except Exception as exc:
        logger.error("Get documents error: %s", str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )
