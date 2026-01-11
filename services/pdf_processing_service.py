"""
PDF OCR processing pipeline.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from core.cache import CacheManager
from core.database import DatabaseManager
from api.v1.pdf_schemas import PDFProcessingResult
from services.async_mistral_ocr_service import call_mistral_ocr
from services.pdf_image_service import save_image_to_disk
from services.pdf_question_extraction import extract_questions_from_ocr

logger = logging.getLogger(__name__)

async def run_document_ocr_pipeline(
    document: Dict[str, Any],
    pdf_base64: str,
    job_id: str,
    processing_result: Dict[str, Any],
    current_user: Dict[str, Any],
    db: DatabaseManager,
    cache: CacheManager
) -> PDFProcessingResult:
    """Run the full OCR extraction pipeline for a stored document."""
    document_id = document["document_id"]
    is_b2c = current_user.get("user_type") == "b2c_admin"
    try:
        logger.info(f"Calling Mistral OCR API for job {job_id}")
        ocr_result = await call_mistral_ocr(pdf_base64)

        processing_result["progress"] = 60
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        logger.info(f"Extracting questions from OCR result for job {job_id}")
        extracted_questions = extract_questions_from_ocr(
            ocr_result,
            document.get("subject", "General"),
            document.get("difficulty", "medium")
        )

        document_type = document.get("document_type", "Chapter Notes")
        logger.info(f"Processing extracted images for job {job_id}, document type: {document_type}")

        all_images: List[Dict[str, Any]] = []
        image_base64_map: Dict[str, Dict[str, Any]] = {}

        for page in ocr_result.get("pages", []):
            for img in page.get("images", []):
                if img.get("image_base64"):
                    saved_images = await save_image_to_disk(
                        img["image_base64"],
                        img["id"],
                        document["filename"],
                        db,
                        current_user.get("user_id"),
                        split_composite=True,
                        is_b2c=is_b2c
                    )
                    if saved_images:
                        all_images.extend(saved_images)
                        for saved_img in saved_images:
                            image_base64_map[img["id"]] = {
                                "image_base64": img.get("image_base64", ""),
                                "top_left_x": img.get("top_left_x", 0),
                                "top_left_y": img.get("top_left_y", 0),
                                "bottom_right_x": img.get("bottom_right_x", 0),
                                "bottom_right_y": img.get("bottom_right_y", 0),
                                "page": page.get("index", 0)
                            }
                            if saved_img["id"] != img["id"]:
                                image_base64_map[saved_img["id"]] = image_base64_map[img["id"]]

        logger.info(f"Saved {len(all_images)} images to disk and database")
        logger.info(f"Image base64 map contains {len(image_base64_map)} entries")

        processing_result["progress"] = 80
        processing_result["extracted_questions"] = len(extracted_questions)
        processing_result["extracted_images"] = len(all_images)
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        logger.info(f"Storing {len(extracted_questions)} questions for {document_type}")

        for question in extracted_questions:
            if document_type in ["Practice Sets", "Test Series"]:
                page_index = question.metadata.get('page', 0)
                image_refs = question.metadata.get('image_refs', [])
                question_image_refs = question.metadata.get('question_image_refs', [])
                page_images: List[Dict[str, Any]] = []
                question_figures: List[Dict[str, Any]] = []

                logger.info(
                    f"Question {question.id} references {len(image_refs)} total images "
                    f"({len(question_image_refs)} question figures)"
                )

                if image_refs:
                    for page in ocr_result.get("pages", []):
                        if page.get("index") == page_index:
                            for mistral_img in page.get("images", []):
                                mistral_img_id = mistral_img.get('id')
                                base_img_id = mistral_img_id.split('.')[0] if '.' in mistral_img_id else mistral_img_id

                                is_referenced = any(
                                    base_img_id in ref or mistral_img_id in ref
                                    for ref in image_refs
                                )

                                if not is_referenced:
                                    logger.debug(f"Skipping non-referenced image {mistral_img_id}")
                                    continue

                                logger.info(f"Including {mistral_img_id} - referenced in question")

                                # Find saved images - check for both exact match and split variants
                                # If image was split, the IDs become img-X-A, img-X-B, etc.
                                matching_saved_images = [
                                    img for img in all_images 
                                    if img['id'] == base_img_id or img['id'].startswith(f"{base_img_id}-")
                                ]
                                
                                img_base64_data = image_base64_map.get(mistral_img_id) or image_base64_map.get(base_img_id, {})

                                if matching_saved_images and img_base64_data:
                                    is_question_figure = any(
                                        base_img_id in ref or mistral_img_id in ref
                                        for ref in question_image_refs
                                    )

                                    is_image_based_mcq = question.metadata.get("is_image_based_mcq", False)
                                    if is_image_based_mcq and not is_question_figure:
                                        is_question_figure = False
                                        logger.info(f"Treating {mistral_img_id} as option image for image-based MCQ")
                                    
                                    # For question figures, use the first image (or unsplit original)
                                    # For option images (image-based MCQ), include all split parts
                                    if is_question_figure:
                                        # For question diagrams, prefer the original (unsplit) image
                                        saved_img = next(
                                            (img for img in matching_saved_images if img.get('is_original', False)),
                                            next(
                                                (img for img in matching_saved_images if img['id'] == base_img_id),
                                                matching_saved_images[0]  # Fall back to first part
                                            )
                                        )
                                        image_obj = {
                                            'id': saved_img['id'],
                                            'filename': saved_img['filename'],
                                            'path': saved_img['path'],
                                            'base64Data': img_base64_data.get('image_base64', ''),
                                            'description': '',
                                            'type': 'diagram',
                                            'bbox': {
                                                'top_left_x': img_base64_data.get('top_left_x', 0),
                                                'top_left_y': img_base64_data.get('top_left_y', 0),
                                                'bottom_right_x': img_base64_data.get('bottom_right_x', 0),
                                                'bottom_right_y': img_base64_data.get('bottom_right_y', 0)
                                            },
                                            'metadata': {
                                                'source': 'mistral_ocr',
                                                'page': page_index,
                                                'extractedAt': datetime.utcnow().isoformat()
                                            }
                                        }
                                        question_figures.append(image_obj)
                                        logger.info(f"✅ Added question figure: {saved_img['id']}")
                                    else:
                                        # For option images, prefer split parts over original
                                        # Filter to only use split parts if available
                                        split_images = [img for img in matching_saved_images if not img.get('is_original', True)]
                                        images_to_use = split_images if split_images else matching_saved_images
                                        
                                        for saved_img in images_to_use:
                                            # Get base64 data for this specific split part if available
                                            split_base64_data = image_base64_map.get(saved_img['id'], img_base64_data)
                                            image_obj = {
                                                'id': saved_img['id'],
                                                'filename': saved_img['filename'],
                                                'path': saved_img['path'],
                                                'base64Data': split_base64_data.get('image_base64', ''),
                                                'description': '',
                                                'type': 'diagram',
                                                'bbox': {
                                                    'top_left_x': split_base64_data.get('top_left_x', 0),
                                                    'top_left_y': split_base64_data.get('top_left_y', 0),
                                                    'bottom_right_x': split_base64_data.get('bottom_right_x', 0),
                                                    'bottom_right_y': split_base64_data.get('bottom_right_y', 0)
                                                },
                                                'metadata': {
                                                    'source': 'mistral_ocr',
                                                    'page': page_index,
                                                    'extractedAt': datetime.utcnow().isoformat()
                                                }
                                            }
                                            page_images.append(image_obj)
                                        logger.info(f"✅ Added {len(images_to_use)} option images from {base_img_id}")
                                else:
                                    # Log why the image wasn't matched
                                    if not matching_saved_images:
                                        logger.warning(f"⚠️ Image {base_img_id} not found in all_images (available: {[img['id'] for img in all_images[:10]]}...)")
                                    if not img_base64_data:
                                        logger.warning(f"⚠️ Image {mistral_img_id} not found in image_base64_map (keys: {list(image_base64_map.keys())[:10]}...)")

                logger.info(
                    f"Associated {len(question_figures)} question figures and "
                    f"{len(page_images)} option images with question {question.id}"
                )

                enhanced_options = []
                is_image_based_mcq = question.metadata.get("is_image_based_mcq", False)

                if is_image_based_mcq and page_images:
                    logger.info(
                        f"Creating image-based MCQ options: {len(page_images)} images for question {question.id}"
                    )
                    for idx, img in enumerate(page_images):
                        option_label = chr(65 + idx)
                        enhanced_options.append({
                            'id': f"{question.id}_opt_{idx}",
                            'type': 'image',
                            'content': img.get('base64Data', ''),
                            'label': option_label,
                            'description': img.get('description', ''),
                            'image_id': img.get('id', ''),
                            'metadata': img.get('metadata', {})
                        })
                    logger.info(f"Created {len(enhanced_options)} image-based MCQ options")
                else:
                    logger.info(f"Non image-based MCQ: building text options for question {question.id}")
                    for idx, option_text in enumerate(question.options):
                        option_label = chr(65 + idx)  # A, B, C, D, etc.
                        enhanced_options.append({
                            'id': f"{question.id}_opt_{idx}",
                            'type': 'text',
                            'content': option_text,
                            'label': option_label,
                            'description': ''
                        })

                question_doc = {
                    "id": question.id,
                    "text": question.text,
                    "subject": document.get("subject", "General"),
                    "difficulty": document.get("difficulty", "medium"),
                    "document_type": document_type,
                    "extracted_at": datetime.utcnow(),
                    "pdf_source": document["filename"],
                    "document_id": document_id,
                    "images": page_images,
                    "question_figures": question_figures,
                    "options": question.options,
                    "enhanced_options": enhanced_options,
                    "correct_answer": question.correct_answer,
                    "is_image_based_mcq": question.metadata.get("is_image_based_mcq", False),
                    "metadata": question.metadata,
                    "points": question.points if hasattr(question, 'points') else 1.0,
                    "penalty": question.penalty if hasattr(question, 'penalty') else 0.0,
                    "created_by": current_user.get("user_id"),
                    "created_at": datetime.utcnow()
                }
            else:
                logger.info(f"Using simple extraction for {document_type} - no image association")
                enhanced_options = []
                for idx, option_text in enumerate(question.options):
                    option_label = chr(65 + idx)  # A, B, C, D, etc.
                    enhanced_options.append({
                        'id': f"{question.id}_opt_{idx}",
                        'type': 'text',
                        'content': option_text,
                        'label': option_label,
                        'description': ''
                    })

                question_doc = {
                    "id": question.id,
                    "text": question.text,
                    "subject": document.get("subject", "General"),
                    "difficulty": document.get("difficulty", "medium"),
                    "document_type": document_type,
                    "extracted_at": datetime.utcnow(),
                    "pdf_source": document["filename"],
                    "document_id": document_id,
                    "images": [],
                    "question_figures": [],
                    "options": question.options,
                    "enhanced_options": enhanced_options,
                    "correct_answer": question.correct_answer,
                    "metadata": question.metadata,
                    "points": question.points if hasattr(question, 'points') else 1.0,
                    "penalty": question.penalty if hasattr(question, 'penalty') else 0.0,
                    "created_by": current_user.get("user_id"),
                    "created_at": datetime.utcnow()
                }

            # Save question to appropriate database (B2C or regular)
            if is_b2c:
                await db.b2c_insert_one("questions", question_doc)
            else:
                await db.mongo_insert_one("questions", question_doc)

            # Store richer metadata so other services can reconstruct the question fully
            import json as _json_for_full
            chromadb_metadata = {
                "document_id": document_id,
                "document_type": document_type,
                "subject": document.get("subject", "General"),
                "difficulty": document.get("difficulty", "medium"),
                # Align with legacy readers that expect pdfSource
                "pdfSource": document_id,
                # Include serialized full data for robust reconstruction paths
                "fullData": _json_for_full.dumps(question_doc, default=str),
                "page": question.metadata.get("page", 0)
                if isinstance(question.metadata.get("page", 0), (int, float)) else 0
            }

            await db.chroma_add(
                [question.id],
                [question.text],
                [chromadb_metadata]
            )

        if is_b2c:
            document_fresh = await db.b2c_find_one("documents", {"document_id": document_id})
        else:
            document_fresh = await db.mongo_find_one("documents", {"document_id": document_id})
        total_calculated_points = sum(
            q.points if hasattr(q, 'points') and q.points else 1.0
            for q in extracted_questions
        )

        update_data = {
            "ocr_status": "completed",
            "extracted_questions_count": len(extracted_questions),
            "extracted_images_count": len(all_images),
            "ocr_completed_at": datetime.utcnow()
        }

        if document_fresh and document_fresh.get("document_type") == "Test Series":
            existing_total = document_fresh.get("total_points")
            if existing_total is None or existing_total == 0:
                update_data["total_points"] = total_calculated_points
                logger.info(f"Auto-calculated total_points for {document_id}: {total_calculated_points}")

        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": update_data}
            )

        processing_result["status"] = "completed"
        processing_result["progress"] = 100
        processing_result["pages"] = ocr_result.get("pages", [])
        await cache.set(f"pdf_job:{job_id}", processing_result, 3600, "admin")

        logger.info(f"OCR processing completed for document {document_id}")
        return PDFProcessingResult(**processing_result)
    except Exception as exc:
        logger.error(f"OCR pipeline failed for document {document_id}: {exc}", exc_info=True)
        # Check if B2C admin for error update
        if is_b2c:
            await db.b2c_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "error"}}
            )
        else:
            await db.mongo_update_one(
                "documents",
                {"document_id": document_id},
                {"$set": {"ocr_status": "error"}}
            )

        error_result = {
            "job_id": job_id,
            "status": "error",
            "progress": 100,
            "error": str(exc),
            "timestamp": datetime.utcnow()
        }
        await cache.set(f"pdf_job:{job_id}", error_result, 3600, "admin")
        raise

