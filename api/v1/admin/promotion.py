import logging
from typing import Dict, Any, List
from bson import ObjectId
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException, Depends, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from api.v1.auth_async import get_database
from .dependencies import require_admin
from .models import SessionPromotionRequest, SessionPromotionResponse

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.post("/session/promote", response_model=SessionPromotionResponse)
@limiter.limit("5/minute")
async def promote_students(
    request: Request,
    promotion_data: SessionPromotionRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Promote students to next session"""
    try:
        admin_id = ObjectId(current_user.get("admin_id", current_user["user_id"]))
        
        # 1. Fetch students
        query = {"admin_id": admin_id, "is_active": True}
        
        # Apply filters
        if promotion_data.student_ids:
            # If specific students requested, override other filters
            query["student_id"] = {"$in": promotion_data.student_ids}
        else:
            if promotion_data.grade_filter:
                query["grade"] = {"$in": promotion_data.grade_filter}
            if promotion_data.section_filter:
                query["section"] = {"$in": promotion_data.section_filter}

        students = await db.mongo_find("students", query)
        
        promoted_count = 0
        skipped_count = 0
        details = []
        updates = []

        # 2. Process each student
        for student in students:
            student_id = student.get("student_id")
            current_grade = student.get("grade")
            current_section = student.get("section")
            
            # Determine new grade
            new_grade = promotion_data.grade_mappings.get(current_grade)
            
            # Determine new section (default to current, or check if overridden)
            new_section = current_section
            if promotion_data.section_updates and student_id in promotion_data.section_updates:
                new_section = promotion_data.section_updates[student_id]
            
            if new_grade:
                # Add to updates list
                updates.append({
                    "filter": {"_id": student["_id"]},
                    "update": {
                        "$set": {
                            "grade": new_grade,
                            "section": new_section,
                            "last_session": promotion_data.new_session, # Track history
                            "updated_at": datetime.utcnow()
                        },
                        "$push": {
                            "promotion_history": {
                                "from_grade": current_grade,
                                "to_grade": new_grade,
                                "from_section": current_section,
                                "to_section": new_section,
                                "session": promotion_data.new_session,
                                "date": datetime.utcnow()
                            }
                        }
                    }
                })
                promoted_count += 1
                details.append({
                    "student_id": student_id,
                    "student_name": student.get("full_name"),
                    "old_grade": current_grade, 
                    "new_grade": new_grade,
                    "status": "promoted"
                })
            else:
                skipped_count += 1
                details.append({
                    "student_id": student_id,
                    "student_name": student.get("full_name"),
                    "old_grade": current_grade, 
                    "new_grade": None,
                    "status": "skipped",
                    "reason": "No mapping found for grade"
                })

        # 3. Apply changes if not preview
        if not promotion_data.preview_only:
            for item in updates:
                await db.mongo_update_one("students", item["filter"], item["update"])
            
            # Deactivate old content if requested
            if promotion_data.deactivate_old_content:
                # Logic to mark old assignments as inactive
                # This depends on how content is linked to session
                pass

        return SessionPromotionResponse(
            success=True,
            message=f"Processed {len(students)} students",
            new_session=promotion_data.new_session,
            students_promoted=promoted_count,
            students_skipped=skipped_count,
            content_deactivated=0, # placeholder
            details=details
        )

    except Exception as e:
        logger.error(f"Session promotion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote session: {str(e)}"
        )
