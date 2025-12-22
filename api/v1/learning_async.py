"""
Learning Mode API - Async Routes
Provides endpoints for accessing Chapter Notes from document management
Integrated with student access control and activity tracking
Serves Admin-uploaded study materials (Document Type: "Chapter Notes")
Designed for 1000+ concurrent users with streaming and caching
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import aiofiles
import aiofiles.os

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from utils.s3_storage import download_file as s3_download_file

logger = logging.getLogger(__name__)

router = APIRouter()

# IMPORTANT: Grade/Standard matching is EXACT
# Both student.grade and document.standard should come from the same admin settings
# so they will match exactly (e.g., "12th Pass" == "12th Pass")
# No normalization or fuzzy matching is needed or desired


def grades_match(student_grade: str, doc_standard: str) -> bool:
    """
    Check if student grade matches document standard.
    Uses flexible matching to handle various grade formats.
    
    Args:
        student_grade: Student's grade from profile (e.g., "12", "12th", "Class 12")
        doc_standard: Document's standard field (e.g., "12", "12th Pass")
    
    Returns:
        True if grades match, False otherwise
    """
    if not student_grade or not doc_standard:
        return False
    
    # Exact match first
    if student_grade == doc_standard:
        return True
    
    # Normalize both values for comparison
    def normalize_grade(grade: str) -> str:
        """Normalize grade to just the number"""
        if not grade:
            return ""
        grade = str(grade).lower().strip()
        # Remove common suffixes
        for suffix in ["th", "st", "nd", "rd", " pass", " class", "class "]:
            grade = grade.replace(suffix, "")
        return grade.strip()
    
    normalized_student = normalize_grade(student_grade)
    normalized_doc = normalize_grade(doc_standard)
    
    return normalized_student == normalized_doc


class LearningDocument(BaseModel):
    """Document information for learning mode"""
    document_id: str
    title: str
    subject: str
    standard: str
    course_plan: str
    document_type: str
    difficulty: Optional[str] = None
    file_path: str
    ocr_status: str
    created_at: Optional[datetime] = None


class CourseStructure(BaseModel):
    """Course structure response"""
    standards: List[str]
    subjects: Dict[str, List[str]]


@router.get("/health", tags=["Learning"])
async def health_check(db: DatabaseManager = Depends(get_database)):
    """Health check endpoint"""
    try:
        # Check if documents collection exists
        docs = await db.mongo_find("documents", {"document_type": "Chapter Notes"}, limit=1)
        doc_count = len(docs) if docs else 0
        return {
            "success": True,
            "message": "Learning Mode API is operational",
            "has_documents": doc_count > 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "success": False,
            "message": f"Health check failed: {str(e)}"
        }


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
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_id = ObjectId(admin_id)
        except Exception:
            pass

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



@router.get("/chapters/{standard}/{subject}", tags=["Learning"])
async def get_chapters(
    standard: str,
    subject: str,
    course_plan: str = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all Chapter Notes documents for a specific standard and subject

    Behavior:
    - Admin users: Returns ALL documents for the requested standard/subject
    - Student users: Returns only documents matching their profile (with access verification)

    Args:
        standard: Standard/Grade (e.g., "11", "12")
        subject: Subject name (e.g., "Physics", "Chemistry", "Mathematics")
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_id = ObjectId(admin_id)
        except Exception:
            pass

        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"

        # Handle B2C users - query from B2C database
        if is_b2c:
            logger.info(f"B2C user {current_user['user_id']} fetching chapters for {standard}/{subject}")
            
            # Get B2C user profile
            b2c_user = await db.b2c_find_one("users", {"_id": ObjectId(current_user["user_id"])})
            if not b2c_user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="B2C user profile not found"
                )
            
            # Get user's standard - either from profile or map from class_level
            user_standard = b2c_user.get("standard")
            if not user_standard:
                # Map class_level like "Class 11" to "11"
                class_level = b2c_user.get("class_level", "")
                if "11" in str(class_level):
                    user_standard = "11"
                elif "12" in str(class_level):
                    user_standard = "12"
            
            logger.info(f"B2C user standard: {user_standard}, class_level: {b2c_user.get('class_level')}")
            
            # Build B2C query with plan-based filtering
            b2c_query = {
                "document_type": "Chapter Notes",
                "is_active": {"$ne": False}
            }
            
            # Use user's standard from profile (not the parameter which might be "Not Set")
            if user_standard:
                b2c_query["standard"] = user_standard
            elif standard and standard != "Not Set":
                b2c_query["standard"] = standard
            
            # Filter by subject
            if subject:
                b2c_query["subject"] = subject
            
            # Filter by user's plan (exam_type) - documents use 'course_plan' field
            if b2c_user.get("exam_type"):
                b2c_query["course_plan"] = {"$in": [b2c_user.get("exam_type")]}

            
            logger.info(f"B2C chapters query: {b2c_query}")
            
            # Get documents from B2C database
            documents = await db.b2c_find("documents", b2c_query, sort=[("title", 1)])
            logger.info(f"B2C chapters found: {len(documents)} for standard={user_standard}, subject={subject}")

            
            # Convert to response format
            documents_list = []
            for doc in documents:
                documents_list.append({
                    "document_id": doc.get("document_id") or str(doc["_id"]),
                    "title": doc.get("title"),
                    "subject": doc.get("subject"),
                    "standard": doc.get("standard"),
                    "course_plan": doc.get("course_plan"),
                    "document_type": doc.get("document_type"),
                    "difficulty": doc.get("difficulty"),
                    "file_path": doc.get("file_path"),
                    "ocr_status": doc.get("ocr_status"),
                    "created_at": doc.get("created_at")
                })
            
            return {
                "success": True,
                "data": {
                    "standard": standard,
                    "subject": subject,
                    "documents": documents_list,
                    "total": len(documents_list)
                }
            }

        # Build query based on user type - always filter by admin_id
        if user_type == "admin":
            # Admin viewing student panel - show documents from their organization for this standard/subject
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id,
                "standard": standard,
                "subject": subject
            }
        else:
            # Actual student login - verify access and filter by profile
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Verify student has access to this standard and subject
            # Use 'grade' field instead of 'standard' - with flexible matching
            student_grade = student.get("grade")
            student_plan_types = student.get("plan_types", [])
            student_subjects = student.get("subjects", [])

            # Use flexible grade matching
            if not grades_match(student_grade, standard):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. You are enrolled in grade {student_grade}, not {standard}."
                )

            if subject not in student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. You are not enrolled in {subject}."
                )

            # Query with access control
            query = {
                "document_type": "Chapter Notes",
                "admin_id": admin_id,  # Only show documents from student's admin
                "standard": standard,
                "subject": subject,
                # is_active: {$ne: False} matches True, None, or missing field (default active)
                "is_active": {"$ne": False}
            }

            # If course_plan is provided in the request, use it (must be in student's plan_types)
            if course_plan:
                if course_plan not in student_plan_types:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied. You are not enrolled in {course_plan} plan."
                    )
                query["course_plan"] = course_plan
            # Otherwise, if student has plan_types, filter by all of them
            elif student_plan_types:
                query["course_plan"] = {"$in": student_plan_types}

        # Get Chapter Notes documents based on query
        documents = await db.mongo_find("documents", query, sort=[("title", 1)])


        # Convert MongoDB documents to response format
        documents_list = []
        for doc in documents:
            documents_list.append({
                "document_id": str(doc["_id"]),  # MongoDB uses _id as primary key
                "title": doc.get("title"),
                "subject": doc.get("subject"),
                "standard": doc.get("standard"),
                "course_plan": doc.get("course_plan"),
                "document_type": doc.get("document_type"),
                "difficulty": doc.get("difficulty"),
                "file_path": doc.get("file_path"),
                "ocr_status": doc.get("ocr_status"),
                "created_at": doc.get("created_at")
            })

        return {
            "success": True,
            "data": {
                "standard": standard,
                "subject": subject,
                "documents": documents_list,
                "total": len(documents_list)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chapters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chapters: {str(e)}"
        )


@router.get("/document/{document_id}", tags=["Learning"])
async def get_document_metadata(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get document metadata including PDF URL
    Returns document details with secure PDF URL
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_id = ObjectId(admin_id)
        except Exception:
            pass

        # Get document from database (document_id is MongoDB's _id as string, filtered by admin_id)
        document = await db.mongo_find_one("documents", {"_id": ObjectId(document_id), "admin_id": admin_id})

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        # Verify document type is Chapter Notes
        if document.get("document_type") != "Chapter Notes":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This document is not a Chapter Notes document"
            )

        # Access control based on user type
        user_type = current_user.get("user_type", "student")

        if user_type == "student":
            # For actual student login, verify access permissions
            student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student profile not found"
                )

            # Verify student has access to this document
            student_grade = student.get("grade")
            student_plan_types = student.get("plan_types", [])
            student_subjects = student.get("subjects", [])

            doc_standard = document.get("standard")
            doc_course_plan = document.get("course_plan")
            doc_subject = document.get("subject")

            # Check if student's grade matches document standard (using flexible matching)
            if not grades_match(student_grade, doc_standard):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. This document is not for your grade level."
                )

            # Check if student is enrolled in this subject
            if doc_subject not in student_subjects:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied. You are not enrolled in this subject."
                )

            # Check if document's course plan matches student's plan types
            if student_plan_types and doc_course_plan not in student_plan_types:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. This document is for {doc_course_plan} plan."
                )

        # Return document metadata with PDF URL
        # The PDF URL endpoint will also perform access control
        pdf_url = f"/api/learning/pdf/{document_id}"

        return {
            "success": True,
            "data": {
                "document_id": document_id,
                "title": document.get("title"),
                "subject": document.get("subject"),
                "standard": document.get("standard"),
                "course_plan": document.get("course_plan"),
                "pdf_url": pdf_url
            }
        }

        return {"success": True, "data": chapter}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}"
        )


@router.get("/pdf/{document_id}", tags=["Learning"])
async def get_chapter_pdf(
    document_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Serve the PDF file for a specific chapter with streaming support
    Optimized for 1000+ concurrent users with caching and range requests

    Args:
        document_id: Document ID from documents collection
        request: FastAPI request object for range support
    """
    try:
        # Get admin_id for data isolation
        from api.v1.questions_async import get_admin_id_from_user
        admin_id = get_admin_id_from_user(current_user)
        try:
            admin_id = ObjectId(admin_id)
        except Exception:
            pass

        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        logger.info(f"PDF request: document_id={document_id}, user_type={user_type}, is_b2c={is_b2c}")

        # B2C users - query from B2C database
        if is_b2c:
            # Try to find document by _id first, then by document_id
            try:
                document = await db.b2c_find_one("documents", {"_id": ObjectId(document_id)})
            except:
                document = None
            
            if not document:
                document = await db.b2c_find_one("documents", {"document_id": document_id})
            
            if not document:
                logger.error(f"B2C document not found: {document_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {document_id}"
                )
            
            logger.info(f"B2C document found: {document.get('title')}, file_path: {document.get('file_path')}")
            
            # B2C users have access to all documents in their database
            # No additional access control needed - documents are already filtered by admin
        else:
            # Regular B2B flow - query main database
            document = await db.mongo_find_one("documents", {"_id": ObjectId(document_id), "admin_id": admin_id})

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {document_id}"
                )

            # Verify document type is Chapter Notes
            if document.get("document_type") != "Chapter Notes":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This document is not a Chapter Notes document"
                )

            # Access control for B2B students
            if user_type == "student":
                # For actual student login, verify access permissions
                student = await db.mongo_find_one("students", {"_id": ObjectId(current_user["user_id"])})

                if not student:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Student profile not found"
                    )

                # Verify student has access to this document
                student_grade = student.get("grade")
                student_plan_types = student.get("plan_types", [])
                student_subjects = student.get("subjects", [])

                doc_standard = document.get("standard")
                doc_course_plan = document.get("course_plan")
                doc_subject = document.get("subject")

                # Check if student's grade matches document standard (using flexible matching)
                if not grades_match(student_grade, doc_standard):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied. This document is not for your grade level."
                    )

                # Check if student is enrolled in this subject
                if doc_subject not in student_subjects:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied. You are not enrolled in this subject."
                    )

                # Check if document's course plan matches student's plan types (if student has plan types)
                if student_plan_types and doc_course_plan not in student_plan_types:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied. This document is for {doc_course_plan} plan."
                    )
            # else: Admin users can access all documents


        # Get PDF path - handle S3 vs local storage
        stored_path = document.get("file_path", "")
        
        # Check if this is an S3 path
        if stored_path.startswith("s3://"):
            logger.info(f"Fetching PDF from S3: {stored_path}")
            
            # Download from S3
            file_data = await s3_download_file(stored_path)
            
            if not file_data:
                logger.error(f"Failed to download PDF from S3: {stored_path}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"PDF file not found in S3: {document_id}"
                )
            
            file_size = len(file_data)
            
            # Production-ready headers for caching and performance
            headers = {
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=31536000, immutable",
                "Content-Type": "application/pdf",
                "Content-Disposition": f'inline; filename="{document.get("title", "chapter")}.pdf"',
                "X-Content-Type-Options": "nosniff",
                "Content-Length": str(file_size),
            }
            
            # Check for range request (for seeking in PDF)
            range_header = request.headers.get("range")
            
            if range_header:
                # Parse range header
                range_match = range_header.replace("bytes=", "").split("-")
                start = int(range_match[0]) if range_match[0] else 0
                end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
                
                if start >= file_size or end >= file_size or start > end:
                    raise HTTPException(status_code=416, detail="Requested range not satisfiable")
                
                chunk_data = file_data[start:end + 1]
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                headers["Content-Length"] = str(len(chunk_data))
                
                from fastapi.responses import Response
                return Response(
                    content=chunk_data,
                    status_code=206,
                    headers=headers,
                    media_type="application/pdf"
                )
            
            # Return full file
            from fastapi.responses import Response
            
            # Log viewing activity in background
            try:
                await db.mongo_insert_one("student_activity_log", {
                    "student_id": ObjectId(current_user["user_id"]),
                    "action": "chapter_viewed",
                    "timestamp": datetime.utcnow(),
                    "metadata": {
                        "document_id": document_id,
                        "title": document.get("title"),
                        "subject": document.get("subject"),
                        "standard": document.get("standard"),
                        "course_plan": document.get("course_plan")
                    }
                })
            except Exception as log_error:
                logger.error(f"Failed to log chapter view activity: {str(log_error)}")
            
            return Response(
                content=file_data,
                headers=headers,
                media_type="application/pdf"
            )
        
        # Local file handling
        # Resolve relative paths against backend directory
        import os
        backend_dir = Path(os.getcwd())
        
        if stored_path.startswith("uploads/") or stored_path.startswith("uploads\\"):
            pdf_path = backend_dir / stored_path.replace("\\", "/")
        else:
            pdf_path = Path(stored_path)
        
        logger.info(f"Resolved PDF path: {pdf_path}, exists: {pdf_path.exists()}")

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found for document: {document_id}"
            )

        # Get file stats for headers
        file_stat = await aiofiles.os.stat(pdf_path)
        file_size = file_stat.st_size

        # Production-ready headers for caching and performance
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=31536000, immutable",  # 1 year cache
            "Content-Type": "application/pdf",
            "Content-Disposition": f'inline; filename="{document.get("title", "chapter")}.pdf"',
            "X-Content-Type-Options": "nosniff",
        }

        # Check for range request (for seeking in PDF)
        range_header = request.headers.get("range")

        if range_header:
            # Parse range header (e.g., "bytes=0-1023")
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1

            # Validate range
            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(
                    status_code=416,
                    detail="Requested range not satisfiable"
                )

            chunk_size = end - start + 1

            # Stream partial content
            async def stream_range():
                async with aiofiles.open(pdf_path, mode='rb') as f:
                    await f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        read_size = min(65536, remaining)  # 64KB chunks
                        data = await f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(chunk_size)

            return StreamingResponse(
                stream_range(),
                status_code=206,
                headers=headers,
                media_type="application/pdf"
            )

        # Full file streaming (for initial load)
        async def stream_full():
            async with aiofiles.open(pdf_path, mode='rb') as f:
                while chunk := await f.read(65536):  # 64KB chunks
                    yield chunk

        headers["Content-Length"] = str(file_size)

        # Log viewing activity in background
        try:
            await db.mongo_insert_one("student_activity_log", {
                "student_id": ObjectId(current_user["user_id"]),
                "action": "chapter_viewed",
                "timestamp": datetime.utcnow(),
                "metadata": {
                    "document_id": document_id,
                    "title": document.get("title"),
                    "subject": document.get("subject"),
                    "standard": document.get("standard"),
                    "course_plan": document.get("course_plan")
                }
            })
        except Exception as log_error:
            logger.error(f"Failed to log chapter view activity: {str(log_error)}")

        return StreamingResponse(
            stream_full(),
            headers=headers,
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve PDF file: {str(e)}"
        )
