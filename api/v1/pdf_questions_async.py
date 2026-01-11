"""
Question endpoints for PDF documents.
"""

import base64
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId as BsonObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_database
from api.v1.pdf_dependencies import require_admin
from api.v1.student_async import require_student_or_admin
from core.database import DatabaseManager
from services.pdf_image_service import save_image_to_disk

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get("/documents/{document_id}/questions")
@limiter.limit("60/minute")
async def get_document_questions(
    request: Request,
    document_id: str,
    current_user: Dict[str, Any] = Depends(require_student_or_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Get all questions extracted from a specific document"""
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]
        
        # Verify document exists in appropriate database
        if is_b2c:
            document = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        # Verify the user has access to this document
        # For students, check if document belongs to their admin
        if current_user.get("user_type") == "student":
            # Normalize types for comparison
            student_admin_id = str(current_user.get("admin_id")) if current_user.get("admin_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            # In development mode, allow cross-admin access to simplify testing
            from config_async import DEBUG_MODE as _DEBUG_MODE
            if student_admin_id != document_admin_id_str:
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing student {current_user.get('user_id')} with admin_id={student_admin_id} "
                        f"to access document owned by admin_id={document_admin_id_str}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )

            # Students can only access completed OCR documents (unless in DEBUG_MODE)
            if document.get("ocr_status") != "completed":
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing access to document {document_id} with ocr_status={document.get('ocr_status')}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="This document is not yet available"
                    )
        elif user_type not in ["b2c_admin", "b2c_user"]:
            # For regular admins, verify they own the document (type-safe)
            admin_id = str(current_user.get("user_id")) if current_user.get("user_id") is not None else None
            document_admin_id = document.get("admin_id")
            document_admin_id_str = str(document_admin_id) if document_admin_id is not None else None

            from config_async import DEBUG_MODE as _DEBUG_MODE
            if admin_id != document_admin_id_str:
                if _DEBUG_MODE:
                    logger.warning(
                        f"DEBUG_MODE: allowing admin {admin_id} to access document owned by admin_id={document_admin_id_str}"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="You don't have access to this document"
                    )
        # B2C admins can access all B2C documents (no admin_id check needed)

        # Get questions for this document from appropriate database
        if is_b2c:
            questions = await db.b2c_find("questions", {"document_id": document_id})
        else:
            questions = await db.mongo_find("questions", {"document_id": document_id})

        # Convert ObjectId to string for JSON serialization and map field names
        serialized_questions = []
        for q in questions:
            # Auto-clean orphaned images from the question
            from utils.image_validator import clean_question_images
            cleaned_q, removed_count = await clean_question_images(q, db, is_b2c)

            # If orphaned images were found and removed, update the database
            if removed_count > 0:
                if is_b2c:
                    await db.b2c_update_one(
                        "questions",
                        {"id": q.get("id")},
                        {"$set": {
                            "images": cleaned_q.get("images", []),
                            "question_figures": cleaned_q.get("question_figures", []),
                            "auto_cleaned_at": datetime.utcnow()
                        }}
                    )
                else:
                    await db.mongo_update_one(
                        "questions",
                        {"id": q.get("id")},
                        {"$set": {
                            "images": cleaned_q.get("images", []),
                            "question_figures": cleaned_q.get("question_figures", []),
                            "auto_cleaned_at": datetime.utcnow()
                        }}
                    )
                logger.info(f"Auto-cleaned {removed_count} orphaned images from question {q.get('id')} during retrieval")

            question_dict = {}
            for key, value in cleaned_q.items():
                if isinstance(value, BsonObjectId):
                    question_dict[key] = str(value)
                elif isinstance(value, datetime):
                    question_dict[key] = value.isoformat()
                else:
                    question_dict[key] = value

            # Map backend field names to frontend expected names
            if "text" in question_dict:
                question_dict["question_text"] = question_dict["text"]

            # === ENHANCED: Load base64 image data for question_figures ===
            enriched_figures = []
            for fig_ref in question_dict.get("question_figures", []) or []:
                try:
                    fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                    base64_data = None
                    
                    # First check if base64Data is already embedded in the figure reference
                    if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                        base64_data = fig_ref["base64Data"]
                    else:
                        # Try to get base64Data from images collection
                        if is_b2c:
                            img_doc = await db.b2c_find_one("images", {"_id": fig_id})
                        else:
                            img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                        
                        if img_doc:
                            # Check if base64Data is stored in the document
                            if img_doc.get("base64Data"):
                                base64_data = img_doc["base64Data"]
                            # If not, try to read from file_path and convert to base64
                            elif img_doc.get("file_path"):
                                import os
                                import base64
                                file_path = img_doc["file_path"]
                                if os.path.exists(file_path):
                                    try:
                                        with open(file_path, "rb") as f:
                                            image_bytes = f.read()
                                            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                            content_type = img_doc.get("content_type", "image/jpeg")
                                            if not content_type.startswith("image/"):
                                                content_type = "image/jpeg"
                                            base64_data = f"data:{content_type};base64,{base64_encoded}"
                                    except Exception as file_err:
                                        logger.error(f"Failed to read image file {file_path}: {file_err}")
                    
                    enriched_figures.append({
                        "id": fig_id,
                        "url": f"/api/v1/images/{fig_id}",
                        "base64Data": base64_data,
                        "description": (fig_ref.get("description", "") if isinstance(fig_ref, dict) else ""),
                        "type": "diagram"
                    })
                except Exception as fig_err:
                    logger.error(f"Error processing figure: {fig_err}")
            
            question_dict["question_figures"] = enriched_figures
            
            # === ENHANCED: Load base64 image data for option images ===
            enriched_images = []
            for img_ref in question_dict.get("images", []) or []:
                try:
                    img_id = img_ref.get("id") if isinstance(img_ref, dict) else img_ref
                    base64_data = None
                    
                    # First check if base64Data is already embedded
                    if isinstance(img_ref, dict) and img_ref.get("base64Data"):
                        base64_data = img_ref["base64Data"]
                    else:
                        # Try to get from images collection
                        if is_b2c:
                            img_doc = await db.b2c_find_one("images", {"_id": img_id})
                        else:
                            img_doc = await db.mongo_find_one("images", {"_id": img_id})
                        
                        if img_doc:
                            if img_doc.get("base64Data"):
                                base64_data = img_doc["base64Data"]
                            elif img_doc.get("file_path"):
                                import os
                                import base64
                                file_path = img_doc["file_path"]
                                if os.path.exists(file_path):
                                    try:
                                        with open(file_path, "rb") as f:
                                            image_bytes = f.read()
                                            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
                                            content_type = img_doc.get("content_type", "image/jpeg")
                                            if not content_type.startswith("image/"):
                                                content_type = "image/jpeg"
                                            base64_data = f"data:{content_type};base64,{base64_encoded}"
                                    except Exception as file_err:
                                        logger.error(f"Failed to read option image file {file_path}: {file_err}")
                    
                    enriched_images.append({
                        "id": img_id,
                        "url": f"/api/v1/images/{img_id}",
                        "base64Data": base64_data,
                        "description": (img_ref.get("description", "") if isinstance(img_ref, dict) else ""),
                        "type": img_ref.get("type", "option") if isinstance(img_ref, dict) else "option"
                    })
                except Exception as img_err:
                    logger.error(f"Error processing image: {img_err}")
            
            question_dict["images"] = enriched_images

            serialized_questions.append(question_dict)

        return {
            "document_id": document_id,
            "document_title": document["title"],
            "questions_count": len(serialized_questions),
            "questions": serialized_questions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document questions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document questions"
        )



@router.post("/questions")
@limiter.limit("30/minute")
async def create_question(
    request: Request,
    question_id: str = Form(...),
    question_text: str = Form(...),
    correct_answer: str = Form(...),
    subject: str = Form(...),
    difficulty: str = Form(...),
    document_type: str = Form(...),
    course_plan: str = Form(...),
    standard: str = Form(...),
    question_type: str = Form(default="mcq"),  # mcq or integer
    document_id: Optional[str] = Form(None),
    options_data: str = Form(default="[]"),  # JSON string of options metadata (optional for integer type)
    question_image: Optional[UploadFile] = File(None),
    option_images: List[UploadFile] = File(default=[]),
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Create a new question with optional image uploads"""
    try:
        import uuid
        import json

        # Generate unique question ID
        full_question_id = f"QST{question_id}"

        # Parse options metadata
        options_metadata = json.loads(options_data) if options_data else []

        # Prepare question document
        question_doc = {
            "id": full_question_id,
            "text": question_text,  # Standard field name used by MCQ service
            "question_text": question_text,  # Alias for compatibility
            "question_type": question_type,  # Store question type (mcq or integer)
            "options": [],  # Will be populated below (empty for integer type)
            "correct_answer": correct_answer,
            "subject": subject,
            "difficulty": difficulty,
            "document_type": document_type,
            "course_plan": course_plan,
            "standard": standard,
            "document_id": document_id,
            "created_by": current_user.get("user_id"),
            "created_at": datetime.utcnow(),
            "images": [],
            "question_figures": []
        }

        # Handle question image if provided
        if question_image and question_image.filename:
            logger.info(f"Uploading question image: {question_image.filename}")
            image_data = await question_image.read()

            # Convert to base64 for save_image_to_disk function and storage
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Save to disk (split_composite=False for manually uploaded images)
            image_results = await save_image_to_disk(
                image_base64=image_base64,
                image_id=f"{full_question_id}_question",
                pdf_filename=document_id or full_question_id,
                db=db,
                user_id=current_user.get("user_id"),
                split_composite=False
            )

            # Add to question_figures with base64 data for frontend display
            for img_result in image_results:
                question_doc["question_figures"].append({
                    "id": img_result["id"],
                    "filename": img_result["filename"],
                    "path": img_result["path"],
                    "base64Data": image_base64,
                    "description": "",
                    "type": "diagram",
                    "metadata": {
                        "source": "manual_upload",
                        "uploadedAt": datetime.utcnow().isoformat()
                    }
                })

        # Process options with images
        option_image_index = 0
        for i, opt_meta in enumerate(options_metadata):
            if opt_meta.get("type") == "text":
                question_doc["options"].append(opt_meta.get("content", ""))
            elif opt_meta.get("type") == "image":
                # Get the corresponding image file
                if option_image_index < len(option_images):
                    opt_image = option_images[option_image_index]
                    option_image_index += 1

                    if opt_image and opt_image.filename:
                        logger.info(f"Uploading option {i} image: {opt_image.filename}")
                        image_data = await opt_image.read()

                        # Convert to base64 for save_image_to_disk function and storage
                        image_base64 = base64.b64encode(image_data).decode('utf-8')

                        # Save to disk (split_composite=False for manually uploaded images)
                        image_results = await save_image_to_disk(
                            image_base64=image_base64,
                            image_id=f"{full_question_id}_option_{i}",
                            pdf_filename=document_id or full_question_id,
                            db=db,
                            user_id=current_user.get("user_id"),
                            split_composite=False
                        )

                        # Add to images array with base64 data for frontend display
                        for img_result in image_results:
                            question_doc["images"].append({
                                "id": img_result["id"],
                                "filename": img_result["filename"],
                                "path": img_result["path"],
                                "base64Data": image_base64,
                                "description": f"Option {chr(65 + i)}",
                                "type": "option",
                                "option_index": i,
                                "metadata": {
                                    "source": "manual_upload",
                                    "uploadedAt": datetime.utcnow().isoformat()
                                }
                            })

                        # Store image reference in options
                        question_doc["options"].append(f"[IMAGE:{img_result['id']}]")
                    else:
                        question_doc["options"].append("[Image option]")
                else:
                    question_doc["options"].append("[Image option]")

        # Insert question into MongoDB
        await db.mongo_insert_one("questions", question_doc)

        # Also add to ChromaDB for searchability and MCQ retrieval
        try:
            chromadb_metadata = {
                "document_id": document_id or full_question_id,
                "document_type": document_type,
                "course_plan": course_plan,
                "standard": standard,
                "subject": subject,
                "difficulty": difficulty,
                "source": "manual_creation",
                "created_by": current_user.get("user_id"),
                "created_at": datetime.utcnow().isoformat()
            }

            await db.chroma_add(
                [full_question_id],
                [question_text],
                [chromadb_metadata]
            )
            logger.info(f"Added question {full_question_id} to ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to add question to ChromaDB: {str(e)}")
            # Don't fail the request if ChromaDB insertion fails

        logger.info(f"Created question {full_question_id} with {len(question_doc['question_figures'])} question images and {len(question_doc['images'])} option images")

        return {
            "message": "Question created successfully",
            "question_id": full_question_id
        }

    except Exception as e:
        logger.error(f"Create question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create question: {str(e)}"
        )


@router.put("/questions/{question_id}")
@limiter.limit("30/minute")
async def update_question(
    request: Request,
    question_id: str,
    question_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Update a question"""
    try:
        logger.info(f"📝 Update question request received for question_id={question_id}")
        logger.info(f"   Update data keys: {list(question_data.keys())}")
        logger.info(f"   User: {current_user.get('user_id')}")

        # Check if B2C admin
        user_type = current_user.get("user_type")
        is_b2c = user_type == "b2c_admin"

        # Get existing question from appropriate database
        if is_b2c:
            existing_question = await db.b2c_find_one("questions", {"id": question_id})
        else:
            existing_question = await db.mongo_find_one("questions", {"id": question_id})
        if not existing_question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Update fields
        update_data = {}
        if "text" in question_data:
            update_data["text"] = question_data["text"]
        if "options" in question_data:
            update_data["options"] = question_data["options"]
        if "correct_answer" in question_data:
            update_data["correct_answer"] = question_data["correct_answer"]
        if "subject" in question_data:
            update_data["subject"] = question_data["subject"]
        if "difficulty" in question_data:
            update_data["difficulty"] = question_data["difficulty"]
        if "document_type" in question_data:
            update_data["document_type"] = question_data["document_type"]
        # Helper to process and save new images
        async def process_new_images(images_list, id_prefix):
            processed_images = []
            for i, img in enumerate(images_list):
                # Check if this is a new image upload (has base64Data)
                if img.get("base64Data"):
                    try:
                        logger.info(f"Processing new image upload for question {question_id}")
                        # Generate a unique ID if the current one is temporary or missing
                        img_id = img.get("id")
                        if not img_id or img_id.startswith("img_") or "temp" in img_id:
                            img_id = f"{question_id}_{id_prefix}_{i}_{int(datetime.utcnow().timestamp())}"
                        
                        # Save to disk
                        saved_results = await save_image_to_disk(
                            image_base64=img["base64Data"],
                            image_id=img_id,
                            pdf_filename=existing_question.get("document_id") or existing_question.get("pdf_source") or question_id,
                            db=db,
                            user_id=current_user.get("user_id"),
                            split_composite=False, # Don't split manual uploads
                            is_b2c=is_b2c
                        )
                        
                        # Add saved images to the list
                        for saved_img in saved_results:
                            # Preserve description and type from the frontend object
                            saved_img["description"] = img.get("description", "")
                            saved_img["type"] = img.get("type", "diagram")
                            # IMPORTANT: Include base64Data so frontend can display it
                            saved_img["base64Data"] = img["base64Data"]
                            processed_images.append(saved_img)
                            
                    except Exception as e:
                        logger.error(f"Failed to save new image: {str(e)}")
                        # If save fails, we might want to skip it or let validation fail
                        # For now, we'll skip adding it to processed_images
                else:
                    # Existing image (no base64Data), keep as is
                    processed_images.append(img)
            return processed_images

        if "images" in question_data:
            # Process any new images first
            question_data["images"] = await process_new_images(question_data["images"], "opt")

            # Validate images before updating
            from utils.image_validator import validate_images_list
            valid_images, invalid_image_ids = await validate_images_list(question_data["images"], db, is_b2c)

            if invalid_image_ids:
                logger.warning(f"Question {question_id} update attempted with {len(invalid_image_ids)} invalid images. These will be filtered out: {invalid_image_ids}")

            update_data["images"] = valid_images

        # Support question_figures (diagram images) - separate from option images
        if "question_figures" in question_data:
            # Process any new images first
            question_data["question_figures"] = await process_new_images(question_data["question_figures"], "fig")

            # Validate question figures before updating
            from utils.image_validator import validate_images_list
            valid_figures, invalid_figure_ids = await validate_images_list(question_data["question_figures"], db, is_b2c)

            if invalid_figure_ids:
                logger.warning(f"Question {question_id} update attempted with {len(invalid_figure_ids)} invalid question figures. These will be filtered out: {invalid_figure_ids}")

            update_data["question_figures"] = valid_figures

        # Support enhanced_options (options with images/metadata)
        if "enhanced_options" in question_data:
            update_data["enhanced_options"] = question_data["enhanced_options"]

        if "points" in question_data:
            update_data["points"] = question_data["points"]
        if "penalty" in question_data:
            # Validate penalty max 50
            penalty = question_data["penalty"]
            if penalty > 50:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Penalty cannot exceed 50 points"
                )
            update_data["penalty"] = penalty

        # Add updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        update_data["updated_by"] = current_user.get("user_id")

        # Update in MongoDB (use appropriate database based on user type)
        if is_b2c:
            success = await db.b2c_update_one(
                "questions",
                {"id": question_id},
                {"$set": update_data}
            )
        else:
            success = await db.mongo_update_one(
                "questions",
                {"id": question_id},
                {"$set": update_data}
            )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No changes were made or question not found"
            )

        # Update in ChromaDB with proper metadata (CRITICAL for categorization)
        try:
            # Get updated question data from appropriate database
            if is_b2c:
                updated_question = await db.b2c_find_one("questions", {"id": question_id})
            else:
                updated_question = await db.mongo_find_one("questions", {"id": question_id})

            # Build updated ChromaDB metadata with all fields
            chromadb_metadata = {
                "document_id": updated_question.get("document_id", question_id),
                "document_type": updated_question.get("document_type", "Chapter Notes"),  # CRITICAL!
                "subject": updated_question.get("subject", "General"),
                "difficulty": updated_question.get("difficulty", "medium"),
                "hasImages": len(updated_question.get("images", [])) > 0 or len(updated_question.get("question_figures", [])) > 0,
                "imageCount": len(updated_question.get("images", [])) + len(updated_question.get("question_figures", [])),
                "source": "manual_edit",
                "updated_by": current_user.get("user_id"),
                "updated_at": datetime.utcnow().isoformat()
            }

            # Update ChromaDB (delete and re-add with updated metadata)
            await db.chroma_delete(ids=[question_id])
            await db.chroma_add(
                [question_id],
                [updated_question.get("text", "")],
                [chromadb_metadata]
            )
            logger.info(f"Updated question {question_id} in ChromaDB with document_type={chromadb_metadata['document_type']}")
        except Exception as e:
            logger.warning(f"Failed to update ChromaDB: {str(e)}")
        # Don't fail the request if ChromaDB update fails

        # If points were updated, recalculate document's total_points
        if "points" in update_data:
            # Use document_id consistently (not pdf_source)
            document_id = existing_question.get("document_id") or existing_question.get("pdf_source")
            if document_id:
                if is_b2c:
                    document = await db.b2c_find_one("documents", {"document_id": document_id})
                else:
                    document = await db.mongo_find_one("documents", {"document_id": document_id})
                if document and document.get("document_type") == "Test Series":
                    # Get all questions for this document using document_id
                    if is_b2c:
                        all_questions = await db.b2c_find("questions", {"document_id": document_id})
                    else:
                        all_questions = await db.mongo_find("questions", {"document_id": document_id})

                    # Fallback to pdf_source if document_id didn't find any
                    if not all_questions:
                        if is_b2c:
                            all_questions = await db.b2c_find("questions", {"pdf_source": document_id})
                        else:
                            all_questions = await db.mongo_find("questions", {"pdf_source": document_id})

                    total_points = sum(q.get("points", 1.0) for q in all_questions)

                    # Update document's total_points
                    if is_b2c:
                        await db.b2c_update_one(
                            "documents",
                            {"document_id": document_id},
                            {"$set": {"total_points": total_points}}
                        )
                    else:
                        await db.mongo_update_one(
                            "documents",
                            {"document_id": document_id},
                            {"$set": {"total_points": total_points}}
                        )
                    logger.info(f"Updated document {document_id} total_points to {total_points}")

        return {
            "message": "Question updated successfully",
            "question_id": question_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update question: {str(e)}"
        )


@router.delete("/questions/{question_id}")
@limiter.limit("30/minute")
async def delete_question(
    request: Request,
    question_id: str,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database)
):
    """Delete a question and all its associated images and metadata"""
    try:
        # Check if B2C admin or B2C user
        user_type = current_user.get("user_type")
        is_b2c = user_type in ["b2c_admin", "b2c_user"]

        # Get the question first
        if is_b2c:
            question = await db.b2c_find_one("questions", {"id": question_id})
        else:
            question = await db.mongo_find_one("questions", {"id": question_id})

        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Delete associated images
        deleted_images_count = 0
        if question.get("images"):
            for image in question["images"]:
                image_id = image.get("id")
                if image_id:
                    # Delete from database
                    if is_b2c:
                        result = await db.b2c_delete_one("images", {"_id": image_id})
                    else:
                        result = await db.mongo_delete_one("images", {"_id": image_id})
                    
                    if result:
                        deleted_images_count += 1

                    # Delete file from disk
                    try:
                        file_path = image.get("path")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete image file {image_id}: {str(e)}")

        # Delete question figures
        if question.get("question_figures"):
            for figure in question["question_figures"]:
                figure_id = figure.get("id")
                if figure_id:
                    # Delete from database
                    if is_b2c:
                        result = await db.b2c_delete_one("images", {"_id": figure_id})
                    else:
                        result = await db.mongo_delete_one("images", {"_id": figure_id})
                    
                    if result:
                        deleted_images_count += 1

                    # Delete file from disk
                    try:
                        file_path = figure.get("path")
                        if file_path and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete figure file {figure_id}: {str(e)}")

        # Delete the question from MongoDB
        if is_b2c:
            result = await db.b2c_delete_one("questions", {"id": question_id})
        else:
            result = await db.mongo_delete_one("questions", {"id": question_id})

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question {question_id} not found"
            )

        # Also delete from ChromaDB if it exists there
        try:
            await db.chroma_delete(ids=[question_id])
            logger.info(f"Deleted question {question_id} from ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to delete from ChromaDB (may not exist there): {str(e)}")

        logger.info(f"Deleted question {question_id} and {deleted_images_count} associated images")

        return {
            "message": "Question deleted successfully",
            "question_id": question_id,
            "deleted_images": deleted_images_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete question error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete question: {str(e)}"
        )


