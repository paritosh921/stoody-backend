"""
Async Settings API for SkillBot
School settings management endpoints (Classes, Sections, Subjects, School Info)
"""

import logging
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.database import DatabaseManager
from core.cache import CacheManager
from api.v1.auth_async import get_current_user
from config_async import settings, MONGODB_URL, DISABLE_MONGODB

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# Helper function to get database
async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


# Helper function to get cache
async def get_cache(request: Request) -> CacheManager:
    return request.app.state.cache


# Pydantic models for Settings
class SchoolInfo(BaseModel):
    """School basic information"""
    school_name: Optional[str] = None
    school_logo: Optional[str] = None  # Base64 encoded or URL
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    address: Optional[str] = None
    website: Optional[str] = None


class SchoolSettingsRequest(BaseModel):
    """Request model for updating school settings"""
    school_info: Optional[SchoolInfo] = None
    classes: Optional[List[str]] = Field(None, description="List of class/grade names")
    sections: Optional[List[str]] = Field(None, description="List of section names (master list)")
    class_sections: Optional[Dict[str, List[str]]] = Field(None, description="Class to sections mapping - which sections are enabled for each class")
    subjects: Optional[List[str]] = Field(None, description="List of subject names")
    plan_types: Optional[List[str]] = Field(None, description="List of plan types (e.g., CBSE, JEE)")
    streams: Optional[List[str]] = Field(None, description="List of streams (e.g., Science, Arts)")


class SchoolSettingsResponse(BaseModel):
    """Response model for school settings"""
    admin_id: str
    school_info: SchoolInfo
    classes: List[str]
    sections: List[str]  # Master list of all sections
    class_sections: Optional[Dict[str, List[str]]] = None  # Class to sections mapping
    subjects: List[str]
    plan_types: List[str]
    streams: List[str]
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


# Default settings for new admins
DEFAULT_SETTINGS = {
    "school_info": {
        "school_name": "",
        "school_logo": "",
        "contact_email": "",
        "contact_phone": "",
        "address": "",
        "website": ""
    },
    "classes": ["9", "10", "11", "12"],
    "sections": ["A", "B", "C", "D", "E", "F"],
    "class_sections": {
        "9": ["A", "B", "C", "D", "E", "F"],
        "10": ["A", "B", "C", "D", "E", "F"],
        "11": ["A", "B", "C", "D", "E", "F"],
        "12": ["A", "B", "C", "D", "E", "F"]
    },
    "subjects": ["Physics", "Chemistry", "Mathematics", "Biology", "English", "Hindi"],
    "plan_types": ["CBSE", "JEE", "NEET", "CUET"],
    "streams": ["Science", "Commerce", "Arts", "Other"]
}


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin access (regular or B2C)"""
    if current_user.get("user_type") not in ["admin", "b2c_admin"]:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user

def is_b2c_admin(current_user: Dict[str, Any]) -> bool:
    """Check if the current user is a B2C admin"""
    return current_user.get("user_type") == "b2c_admin"


@router.get("/settings", response_model=SchoolSettingsResponse)
@limiter.limit("60/minute")
async def get_school_settings(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get school settings for the current admin"""
    try:
        admin_id = current_user.get("user_id") or current_user.get("sub")
        if not admin_id:
            raise HTTPException(status_code=401, detail="Invalid user session")
        
        # Determine if B2C admin
        is_b2c = is_b2c_admin(current_user)
        
        # Try to get from cache first
        cache_key = f"school_settings:{'b2c' if is_b2c else admin_id}"
        cached_settings = await cache.get(cache_key)
        if cached_settings:
            return cached_settings
        
        # Get settings from appropriate database
        if is_b2c:
            settings_doc = await db.b2c_find_one(
                "school_settings",
                {"admin_id": admin_id}
            )
        else:
            settings_doc = await db.mongo_find_one(
                "school_settings",
                {"admin_id": admin_id}
            )
        
        if not settings_doc:
            # Return default settings for new admins
            default_response = {
                "admin_id": admin_id,
                **DEFAULT_SETTINGS,
                "school_info": SchoolInfo(**DEFAULT_SETTINGS["school_info"]).dict(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            # Cache default settings
            await cache.set(cache_key, default_response, ttl=3600)
            return default_response
        
        # Convert MongoDB document to response
        response = {
            "admin_id": admin_id,
            "school_info": SchoolInfo(**settings_doc.get("school_info", DEFAULT_SETTINGS["school_info"])).dict(),
            "classes": settings_doc.get("classes", DEFAULT_SETTINGS["classes"]),
            "sections": settings_doc.get("sections", DEFAULT_SETTINGS["sections"]),
            "class_sections": settings_doc.get("class_sections", DEFAULT_SETTINGS["class_sections"]),
            "subjects": settings_doc.get("subjects", DEFAULT_SETTINGS["subjects"]),
            "plan_types": settings_doc.get("plan_types", DEFAULT_SETTINGS["plan_types"]),
            "streams": settings_doc.get("streams", DEFAULT_SETTINGS["streams"]),
            "created_at": settings_doc.get("created_at"),
            "updated_at": settings_doc.get("updated_at")
        }
        
        # Cache for 1 hour
        await cache.set(cache_key, response, ttl=3600)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting school settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get school settings")


@router.put("/settings", response_model=SchoolSettingsResponse)
@limiter.limit("30/minute")
async def update_school_settings(
    request: Request,
    settings_data: SchoolSettingsRequest,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Update school settings for the current admin"""
    try:
        admin_id = current_user.get("user_id") or current_user.get("sub")
        if not admin_id:
            raise HTTPException(status_code=401, detail="Invalid user session")
        
        # Get existing settings
        existing_settings = await db.mongo_find_one(
            "school_settings",
            {"admin_id": admin_id}
        )
        
        # Build update document
        update_doc = {
            "admin_id": admin_id,
            "updated_at": datetime.utcnow()
        }
        
        if existing_settings:
            update_doc["created_at"] = existing_settings.get("created_at", datetime.utcnow())
        else:
            update_doc["created_at"] = datetime.utcnow()
        
        # Update school info if provided
        if settings_data.school_info:
            existing_info = existing_settings.get("school_info", DEFAULT_SETTINGS["school_info"]) if existing_settings else DEFAULT_SETTINGS["school_info"]
            update_doc["school_info"] = {
                **existing_info,
                **{k: v for k, v in settings_data.school_info.dict().items() if v is not None}
            }
        else:
            update_doc["school_info"] = existing_settings.get("school_info", DEFAULT_SETTINGS["school_info"]) if existing_settings else DEFAULT_SETTINGS["school_info"]
        
        # Update lists if provided
        # Update lists if provided
        update_doc["classes"] = settings_data.classes if settings_data.classes is not None else (existing_settings.get("classes", DEFAULT_SETTINGS["classes"]) if existing_settings else DEFAULT_SETTINGS["classes"])
        
        # Get master sections list
        master_sections = settings_data.sections if settings_data.sections is not None else (existing_settings.get("sections", DEFAULT_SETTINGS["sections"]) if existing_settings else DEFAULT_SETTINGS["sections"])
        update_doc["sections"] = master_sections
        
        # Get class sections: Start with existing (or defaults) to preserve untouched classes
        raw_class_sections = existing_settings.get("class_sections", DEFAULT_SETTINGS["class_sections"]) if existing_settings else DEFAULT_SETTINGS["class_sections"]
        
        # Update with new data if provided
        if settings_data.class_sections is not None:
            # We merge the new settings into the existing dictionary
            # This ensures that if the frontend only sends changed classes (or misses some), we don't lose the others
            if raw_class_sections is None:
                raw_class_sections = {}
            
            # Create a copy to modify
            raw_class_sections = dict(raw_class_sections)
            raw_class_sections.update(settings_data.class_sections)
        
        # Sanitize class_sections: Ensure all sections in the matrix actually exist in the master sections list
        sanitized_class_sections = {}
        if raw_class_sections:
            for cls, sections in raw_class_sections.items():
                if sections:
                    # Only keep sections that are in the master list
                    valid_sections = [s for s in sections if s in master_sections]
                    sanitized_class_sections[cls] = valid_sections
                else:
                    sanitized_class_sections[cls] = []
        
        update_doc["class_sections"] = sanitized_class_sections
        update_doc["subjects"] = settings_data.subjects if settings_data.subjects is not None else (existing_settings.get("subjects", DEFAULT_SETTINGS["subjects"]) if existing_settings else DEFAULT_SETTINGS["subjects"])
        update_doc["plan_types"] = settings_data.plan_types if settings_data.plan_types is not None else (existing_settings.get("plan_types", DEFAULT_SETTINGS["plan_types"]) if existing_settings else DEFAULT_SETTINGS["plan_types"])
        update_doc["streams"] = settings_data.streams if settings_data.streams is not None else (existing_settings.get("streams", DEFAULT_SETTINGS["streams"]) if existing_settings else DEFAULT_SETTINGS["streams"])
        
        # Upsert settings
        collection = await db.get_mongo_collection("school_settings")
        if collection is not None:
            await collection.update_one(
                {"admin_id": admin_id},
                {"$set": update_doc},
                upsert=True
            )
        
        # Invalidate cache
        cache_key = f"school_settings:{admin_id}"
        await cache.delete(cache_key)
        
        # Return updated settings
        response = {
            "admin_id": admin_id,
            "school_info": SchoolInfo(**update_doc["school_info"]).dict(),
            "classes": update_doc["classes"],
            "sections": update_doc["sections"],
            "class_sections": update_doc["class_sections"],
            "subjects": update_doc["subjects"],
            "plan_types": update_doc["plan_types"],
            "streams": update_doc["streams"],
            "created_at": update_doc["created_at"],
            "updated_at": update_doc["updated_at"]
        }
        
        logger.info(f"School settings updated for admin {admin_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating school settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update school settings")


@router.post("/settings/logo")
@limiter.limit("10/minute")
async def upload_school_logo(
    request: Request,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Upload school logo and save as base64"""
    try:
        admin_id = current_user.get("user_id") or current_user.get("sub")
        if not admin_id:
            raise HTTPException(status_code=401, detail="Invalid user session")
        
        # Validate file type
        allowed_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, GIF, and WebP are allowed.")
        
        # Read file content
        content = await file.read()
        
        # Check file size (max 3MB)
        if len(content) > 3 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 3MB.")
        
        # Convert to base64
        base64_logo = f"data:{file.content_type};base64,{base64.b64encode(content).decode('utf-8')}"
        
        # Update settings with new logo
        collection = await db.get_mongo_collection("school_settings")
        if collection is not None:
            await collection.update_one(
                {"admin_id": admin_id},
                {
                    "$set": {
                        "school_info.school_logo": base64_logo,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "admin_id": admin_id,
                        "created_at": datetime.utcnow(),
                        "school_info.school_name": "",
                        "school_info.contact_email": "",
                        "school_info.contact_phone": "",
                        "school_info.address": "",
                        "school_info.website": "",
                        "classes": DEFAULT_SETTINGS["classes"],
                        "sections": DEFAULT_SETTINGS["sections"],
                        "subjects": DEFAULT_SETTINGS["subjects"],
                        "plan_types": DEFAULT_SETTINGS["plan_types"],
                        "streams": DEFAULT_SETTINGS["streams"]
                    }
                },
                upsert=True
            )
        
        # Invalidate cache
        cache_key = f"school_settings:{admin_id}"
        await cache.delete(cache_key)
        
        return {
            "message": "Logo uploaded successfully",
            "logo": base64_logo
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading logo: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload logo")


@router.delete("/settings/logo")
@limiter.limit("10/minute")
async def delete_school_logo(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Delete school logo"""
    try:
        admin_id = current_user.get("user_id") or current_user.get("sub")
        if not admin_id:
            raise HTTPException(status_code=401, detail="Invalid user session")
        
        # Update settings to remove logo
        collection = await db.get_mongo_collection("school_settings")
        if collection is not None:
            await collection.update_one(
                {"admin_id": admin_id},
                {
                    "$set": {
                        "school_info.school_logo": "",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Invalidate cache
        cache_key = f"school_settings:{admin_id}"
        await cache.delete(cache_key)
        
        return {"message": "Logo deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting logo: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete logo")


# Public endpoint to get settings by admin_id (for student-facing features)
@router.get("/settings/public/{admin_id}")
@limiter.limit("120/minute")
async def get_public_school_settings(
    request: Request,
    admin_id: str,
    db: DatabaseManager = Depends(get_database),
    cache: CacheManager = Depends(get_cache)
):
    """Get public school settings (for student-facing features)"""
    try:
        # Try to get from cache first
        cache_key = f"school_settings_public:{admin_id}"
        cached_settings = await cache.get(cache_key)
        if cached_settings:
            return cached_settings
        
        # Get settings from database
        settings_doc = await db.mongo_find_one(
            "school_settings",
            {"admin_id": admin_id}
        )
        
        if not settings_doc:
            # Return default settings
            response = {
                "school_name": "",
                "school_logo": "",
                "classes": DEFAULT_SETTINGS["classes"],
                "sections": DEFAULT_SETTINGS["sections"],
                "class_sections": DEFAULT_SETTINGS["class_sections"],
                "subjects": DEFAULT_SETTINGS["subjects"],
                "plan_types": DEFAULT_SETTINGS["plan_types"],
                "streams": DEFAULT_SETTINGS["streams"]
            }
        else:
            school_info = settings_doc.get("school_info", {})
            response = {
                "school_name": school_info.get("school_name", ""),
                "school_logo": school_info.get("school_logo", ""),
                "classes": settings_doc.get("classes", DEFAULT_SETTINGS["classes"]),
                "sections": settings_doc.get("sections", DEFAULT_SETTINGS["sections"]),
                "class_sections": settings_doc.get("class_sections", DEFAULT_SETTINGS["class_sections"]),
                "subjects": settings_doc.get("subjects", DEFAULT_SETTINGS["subjects"]),
                "plan_types": settings_doc.get("plan_types", DEFAULT_SETTINGS["plan_types"]),
                "streams": settings_doc.get("streams", DEFAULT_SETTINGS["streams"])
            }
        
        # Cache for 10 minutes
        await cache.set(cache_key, response, ttl=600)
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting public school settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get school settings")
