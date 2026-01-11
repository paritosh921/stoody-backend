import logging
from typing import Dict, Any, List
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Depends, status

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/structure", tags=["Learning"])
async def get_course_structure(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get the course structure for the logged-in user

    Behavior:
    - Admin users (viewing as student): Returns ALL Chapter Notes documents
    - Student users: Returns only documents matching their profile (exact match)
    - B2C users: Returns documents from B2C database matching their plan (JEE/NEET + class)
    """
    try:
        # Get admin_id for data isolation
        try:
            from api.v1.questions_async import get_admin_id_from_user
            admin_id = get_admin_id_from_user(current_user)
            admin_id = ObjectId(admin_id)
        except Exception:
            # Fallback if helper fails or returns invalid ID
            admin_id = None
            if current_user.get("user_type") == "admin":
                 admin_id = ObjectId(current_user.get("user_id")) # If admin, use own ID

        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        # Handle B2C users - query from B2C database
        if is_b2c:
            # Get B2C user profile from B2C database
            b2c_user = await db.b2c_find_one("users", {"_id": ObjectId(current_user["user_id"])})
            
            if not b2c_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="B2C user profile not found"
                )
            
            # Check if onboarding is complete
            if not b2c_user.get("onboarding_complete"):
                # Return empty structure for users who haven't completed onboarding
                return {
                    "success": True,
                    "data": {
                        "standards": [],
                        "subjects": {},
                        "dashboard_stats": {},
                        "onboarding_required": True
                    }
                }
            
            # Get user's plan details
            user_exam_type = b2c_user.get("exam_type")  # JEE or NEET
            user_class_level = b2c_user.get("class_level")  # 9, 10, 11, 12, or Dropper
            user_standard = b2c_user.get("standard")  # Mapped class (Dropper = 12)
            user_subjects = b2c_user.get("subjects", [])  # Auto-set based on exam
            user_plan_types = b2c_user.get("plan_types", [])  # [JEE] or [NEET]
            
            # Get B2C admin ID for content filtering
            b2c_admin = await db.b2c_find_one("admins", {}, {"_id": 1})
            b2c_admin_id = b2c_admin["_id"] if b2c_admin else None
            
            if not b2c_admin_id:
                logger.warning("No B2C admin found - B2C user will see no content")
                return {
                    "success": True,
                    "data": {
                        "standards": [],
                        "subjects": {},
                        "dashboard_stats": {}
                    }
                }
            
            # Query B2C documents collection - ONLY Chapter Notes for learning structure
            query = {
                "document_type": "Chapter Notes",
                "is_active": {"$ne": False}
            }
            
            # Filter by admin
            try:
                query["admin_id"] = ObjectId(b2c_admin_id)
            except:
                query["admin_id"] = b2c_admin_id
            
            # Filter by course plan (JEE/NEET)
            if user_plan_types:
                query["course_plan"] = {"$in": user_plan_types}
            elif user_exam_type:
                query["course_plan"] = user_exam_type
            
            # Filter by standard (class)
            if user_standard:
                query["standard"] = user_standard
            
            # Filter by subjects
            if user_subjects:
                query["subject"] = {"$in": user_subjects}
            
            logger.info(f"B2C user {current_user['user_id']} query: {query}")
            
            # Get documents from B2C database
            documents = await db.b2c_find("documents", query)
            logger.info(f"B2C documents found: {len(documents)}")
            
        elif user_type == "admin":
            # Admin viewing student panel - show Chapter Notes from their organization
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id
            }
            documents = await db.mongo_find("documents", query)
        else:
            # Regular B2B student login - filter by profile using EXACT match
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Get student's access parameters - these come from admin settings
            student_grade = student.get("grade")  # Exact value from settings (e.g., "12th Pass")
            student_plan_types = student.get("plan_types", [])
            student_subjects = student.get("subjects", [])

            if not student_grade or not student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Student profile incomplete. Please contact admin to set grade and subjects."
                )

            # Query documents with EXACT match on standard
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id,
                "standard": student_grade,
                "subject": {"$in": student_subjects},
                "is_active": {"$ne": False}
            }

            if student_plan_types:
                query["course_plan"] = {"$in": student_plan_types}
            
            documents = await db.mongo_find("documents", query)

        # Organize by standard and subject
        standards_set = set()
        subjects_by_standard: Dict[str, set] = {}
        
        # Organize by subject for dashboard
        dashboard_stats: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for doc in documents:
            std = doc.get("standard")
            subj = doc.get("subject")

            if std and subj:
                standards_set.add(std)
                if std not in subjects_by_standard:
                    subjects_by_standard[std] = set()
                subjects_by_standard[std].add(subj)
            
            # Add to dashboard stats
            if subj:
                if subj not in dashboard_stats:
                    dashboard_stats[subj] = {"chapters": []}
                
                dashboard_stats[subj]["chapters"].append({
                    "id": str(doc["_id"]),
                    "title": doc.get("title"),
                    "subject": subj,
                    "standard": std,
                    "document_type": doc.get("document_type"),
                    "course_plan": doc.get("course_plan")
                })

        # Convert sets to sorted lists
        standards = sorted(list(standards_set))
        subjects_dict = {
            std: sorted(list(subjects))
            for std, subjects in subjects_by_standard.items()
        }

        return {
            "success": True,
            "data": {
                "standards": standards,
                "subjects": subjects_dict,
                "dashboard_stats": dashboard_stats
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get course structure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve course structure: {str(e)}"
        )
