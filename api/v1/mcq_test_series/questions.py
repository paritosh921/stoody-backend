import logging
from typing import Dict, Any, List
from bson import ObjectId

from fastapi import APIRouter, HTTPException, Depends, status

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from services.question_service import QuestionService

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/test-series/{document_id}/questions")
async def get_test_series_questions(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
):
    """
    Get all questions from a specific Test Series document
    Works with ChromaDB data when MongoDB documents are missing
    """
    try:
        user_type = current_user.get("user_type", "student")
        is_b2c = current_user.get("is_b2c", False) or user_type == "b2c_user"
        
        # Handle B2C users - query from B2C database
        if is_b2c:
            logger.info(f"B2C user {current_user['user_id']} fetching test series {document_id}")
            
            # Use B2C helpers if needed, but here logic is direct
            # Try to get document from B2C database
            document = await db.b2c_find_one("documents", {"document_id": document_id})
            
            if not document:
                # Also try by _id
                try:
                    document = await db.b2c_find_one("documents", {"_id": ObjectId(document_id)})
                except:
                    pass
            
            if not document:
                logger.warning(f"B2C document {document_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Test series document not found: {document_id}"
                )
            
            # Verify document type
            if document.get("document_type") != "Test Series":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This document is not a Test Series"
                )
            
            # Check if document is active
            if document.get("is_active") == False:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This test series is not currently available"
                )
            
            # Get questions from B2C database
            mongo_questions = await db.b2c_find(
                "questions",
                {"document_id": document_id},
                sort=[("metadata.page", 1)]
            )
            
            # If no questions found by document_id, try by pdf_source (filename)
            if not mongo_questions:
                mongo_questions = await db.b2c_find(
                    "questions",
                    {"pdf_source": document.get("filename")},
                    sort=[("metadata.page", 1)]
                )
            
            logger.info(f"B2C test series {document_id}: found {len(mongo_questions)} questions")
            
            questions_with_images = []
            if mongo_questions:
                from datetime import datetime
                def to_jsonable(value):
                    if isinstance(value, datetime):
                        return value.isoformat()
                    try:
                        if isinstance(value, ObjectId):
                            return str(value)
                    except Exception:
                        pass
                    if isinstance(value, list):
                        return [to_jsonable(v) for v in value]
                    if isinstance(value, dict):
                        return {k: to_jsonable(v) for k, v in value.items()}
                    return value
                
                for q in mongo_questions:
                    payload = {
                        "id": str(q.get("id") or q.get("_id")),
                        "text": q.get("text", q.get("question_text", "")),
                        "question_text": q.get("question_text", q.get("text", "")),
                        "subject": q.get("subject", document.get("subject")),
                        "difficulty": q.get("difficulty", "medium"),
                        "document_type": "Test Series",
                        "document_id": document_id,
                        "pdf_source": q.get("pdf_source", document.get("filename")),
                        "images": q.get("images", []),
                        "question_figures": q.get("question_figures", []),
                        "options": q.get("options", []),
                        "enhanced_options": q.get("enhanced_options", []),
                        "correct_answer": q.get("correct_answer"),
                        "metadata": q.get("metadata", {}),
                        "points": q.get("points", 1),
                        "penalty": q.get("penalty", 0),
                        "created_at": q.get("created_at"),
                        "extracted_at": q.get("extracted_at"),
                    }
                    
                    # Enrich figures with base64 from B2C database
                    try:
                        figures: List[Dict[str, Any]] = []
                        for fig_ref in (q.get("question_figures", []) or []):
                            fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                            base64_data = None
                            if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                                base64_data = fig_ref["base64Data"]
                            else:
                                img_doc = await db.b2c_find_one("images", {"_id": fig_id})
                                if img_doc:
                                    if img_doc.get("base64Data"):
                                        base64_data = img_doc["base64Data"]
                                    elif img_doc.get("file_path"):
                                        import os, base64
                                        fp = img_doc["file_path"]
                                        if os.path.exists(fp):
                                            with open(fp, "rb") as f:
                                                enc = base64.b64encode(f.read()).decode("utf-8")
                                                ct = img_doc.get("content_type", "image/jpeg")
                                                if not ct.startswith("image/"):
                                                    ct = "image/jpeg"
                                                base64_data = f"data:{ct};base64,{enc}"
                            figures.append({
                                "id": fig_id,
                                "url": f"/api/v1/images/{fig_id}",
                                "base64Data": base64_data,
                                "contentType": "image/jpeg",
                                "filename": (fig_ref.get("filename") if isinstance(fig_ref, dict) else str(fig_id)),
                                "type": "diagram"
                            })
                        payload["questionFigures"] = figures
                    except Exception:
                        payload["questionFigures"] = []
                    
                    # Inline base64 for image-type enhanced options
                    try:
                        eos = payload.get("enhanced_options") or []
                        for i, opt in enumerate(list(eos)):
                            if isinstance(opt, dict) and opt.get("type") == "image":
                                content = opt.get("content")
                                if isinstance(content, str) and content and not content.startswith("data:image"):
                                    img_doc = await db.b2c_find_one("images", {"_id": content})
                                    if img_doc:
                                        b64 = img_doc.get("base64Data")
                                        if not b64 and img_doc.get("file_path"):
                                            import os, base64
                                            fp = img_doc["file_path"]
                                            if os.path.exists(fp):
                                                with open(fp, "rb") as f:
                                                    enc = base64.b64encode(f.read()).decode("utf-8")
                                                    ct = img_doc.get("content_type", "image/jpeg")
                                                    if not ct.startswith("image/"):
                                                        ct = "image/jpeg"
                                                    b64 = f"data:{ct};base64,{enc}"
                                        if b64:
                                            payload["enhanced_options"][i]["content"] = b64
                    except Exception:
                        pass
                    
                    questions_with_images.append(to_jsonable(payload))
            
            return {
                "success": True,
                "data": {
                    "document_id": document.get("document_id") or str(document.get("_id")),
                    "title": document.get("title"),
                    "subject": document.get("subject"),
                    "total_points": document.get("total_points", 0),
                    "total_minutes": document.get("total_minutes", 0),
                    "questions": questions_with_images,
                    "total": len(questions_with_images)
                }
            }

        # Regular B2B flow - Get admin_id for data isolation
        try:
            from api.v1.questions_async import get_admin_id_from_user
            admin_id = get_admin_id_from_user(current_user)
        except ImportError:
            admin_id = current_user.get("user_id") if current_user.get("user_type") == "admin" else None

        try:
            admin_oid = ObjectId(admin_id)
            admin_filter = {"$in": [admin_oid, admin_id]}
        except Exception:
            admin_filter = admin_id

        # Try to get document from MongoDB first (filtered by admin_id)
        document = await db.mongo_find_one("documents", {"document_id": document_id, "admin_id": admin_filter})

        if document:
            # Verify document type
            if document.get("document_type") != "Test Series":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This document is not a Test Series"
                )

            # For students, verify document is active (None/missing = active by default)
            if current_user.get("user_type") == "student" and document.get("is_active") == False:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="This test series is not currently available"
                )

            # 1) Preferred: read directly from Mongo 'questions' by document_id (populated during OCR)
            mongo_questions = await db.mongo_find(
                "questions",
                {"document_id": document_id},
                sort=[("metadata.page", 1)]
            )


            questions_with_images = []
            if mongo_questions:
                # Normalize to JSON-serializable payloads expected by frontend
                from datetime import datetime
                def to_jsonable(value):
                    if isinstance(value, datetime):
                        return value.isoformat()
                    try:
                        if isinstance(value, ObjectId):
                            return str(value)
                    except Exception:
                        pass
                    if isinstance(value, list):
                        return [to_jsonable(v) for v in value]
                    if isinstance(value, dict):
                        return {k: to_jsonable(v) for k, v in value.items()}
                    return value

                for q in mongo_questions:
                    payload = {
                        "id": str(q.get("id") or q.get("_id")),
                        "text": q.get("text", q.get("question_text", "")),
                        "question_text": q.get("question_text", q.get("text", "")),
                        "subject": q.get("subject", document.get("subject")),
                        "difficulty": q.get("difficulty", "medium"),
                        "document_type": "Test Series",
                        "document_id": document_id,
                        "pdf_source": q.get("pdf_source", document.get("filename")),
                        "images": q.get("images", []),
                        "question_figures": q.get("question_figures", []),
                        "options": q.get("options", []),
                        "enhanced_options": q.get("enhanced_options", []),
                        "correct_answer": q.get("correct_answer"),
                        "metadata": q.get("metadata", {}),
                        "points": q.get("points", 1),
                        "penalty": q.get("penalty", 0),
                        "created_at": q.get("created_at"),
                        "extracted_at": q.get("extracted_at"),
                    }

                    # Enrich figures with base64 (for UI that displays diagrams)
                    try:
                        figures: List[Dict[str, Any]] = []
                        for fig_ref in (q.get("question_figures", []) or []):
                            fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref
                            base64_data = None
                            if isinstance(fig_ref, dict) and fig_ref.get("base64Data"):
                                base64_data = fig_ref["base64Data"]
                            else:
                                img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                                if img_doc:
                                    if img_doc.get("base64Data"):
                                        base64_data = img_doc["base64Data"]
                                    elif img_doc.get("file_path"):
                                        import os, base64
                                        fp = img_doc["file_path"]
                                        if os.path.exists(fp):
                                            with open(fp, "rb") as f:
                                                enc = base64.b64encode(f.read()).decode("utf-8")
                                                ct = img_doc.get("content_type", "image/jpeg")
                                                if not ct.startswith("image/"):
                                                    ct = "image/jpeg"
                                                base64_data = f"data:{ct};base64,{enc}"
                            figures.append({
                                "id": fig_id,
                                "url": f"/api/v1/images/{fig_id}",
                                "base64Data": base64_data,
                                "contentType": "image/jpeg",
                                "filename": (fig_ref.get("filename") if isinstance(fig_ref, dict) else str(fig_id)),
                                "type": "diagram"
                            })
                        payload["questionFigures"] = figures
                    except Exception:
                        payload["questionFigures"] = []

                    # Inline base64 for image-type enhanced options when content is an image id
                    try:
                        eos = payload.get("enhanced_options") or []
                        for i, opt in enumerate(list(eos)):
                            if isinstance(opt, dict) and opt.get("type") == "image":
                                content = opt.get("content")
                                if isinstance(content, str) and content and not content.startswith("data:image"):
                                    img_doc = await db.mongo_find_one("images", {"_id": content})
                                    if img_doc:
                                        b64 = img_doc.get("base64Data")
                                        if not b64 and img_doc.get("file_path"):
                                            import os, base64
                                            fp = img_doc["file_path"]
                                            if os.path.exists(fp):
                                                with open(fp, "rb") as f:
                                                    enc = base64.b64encode(f.read()).decode("utf-8")
                                                    ct = img_doc.get("content_type", "image/jpeg")
                                                    if not ct.startswith("image/"):
                                                        ct = "image/jpeg"
                                                    b64 = f"data:{ct};base64,{enc}"
                                        if b64:
                                            payload["enhanced_options"][i]["content"] = b64
                    except Exception:
                        pass
                    questions_with_images.append(to_jsonable(payload))
            else:
                # 2) Fallback: use Chroma via QuestionService
                question_service = QuestionService(admin_id)
                questions = question_service.search_questions(
                    query=None,
                    document_type="Test Series",
                    limit=1000
                )

                if document_id in ("legacy_all", "all", "ALL"):
                    questions_with_images = questions
                else:
                    questions_with_images = [q for q in questions if q.get('pdfSource', '') == document_id or q.get('document_id', '') == document_id]

            return {
                "success": True,
                "data": {
                    "document_id": document.get("document_id"),
                    "title": document.get("title"),
                    "subject": document.get("subject"),
                    "total_points": document.get("total_points", 0),
                    "total_minutes": document.get("total_minutes", 0),
                    "questions": questions_with_images,
                    "total": len(questions_with_images)
                }
            }
        else:
            # Fallback: Get questions directly from admin's collection
            logger.info(f"Document {document_id} not found in MongoDB, searching admin's collection")

            # Get questions from admin's collection
            question_service = QuestionService(admin_id)
            questions = question_service.search_questions(
                query=None,
                document_type="Test Series",
                limit=1000
            )

            # Filter by document_id (questions is already a list of dicts)
            questions_with_images = [q for q in questions if q.get('pdfSource', '') == document_id or q.get('document_id', '') == document_id]

            if not questions_with_images:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No questions found for document: {document_id}"
                )

            # Use first question's metadata for document info
            first_question = questions_with_images[0]
            doc_title = first_question.get("metadata", {}).get("document_title", f"Test Series {document_id}")
            doc_subject = first_question.get("subject", "Unknown")

            return {
                "success": True,
                "data": {
                    "document_id": document_id,
                    "title": doc_title,
                    "subject": doc_subject,
                    "total_points": len(questions_with_images) * 4,  # Estimate 4 points per question
                    "total_minutes": len(questions_with_images) * 2,  # Estimate 2 minutes per question
                    "questions": questions_with_images,
                    "total": len(questions_with_images)
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test series questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve questions: {str(e)}"
        )
