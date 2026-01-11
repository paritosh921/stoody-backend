"""
Shared MCQ question helpers.
"""

import logging
from typing import Any, Dict, List

from bson import ObjectId

from core.database import DatabaseManager

logger = logging.getLogger(__name__)


async def get_mcq_questions_with_images(
    db: DatabaseManager,
    where_filter: dict,
    limit: int = 10000
) -> List[Dict[str, Any]]:
    """Helper function to fetch MCQ questions with images from MongoDB."""
    question_docs = await db.mongo_find("questions", where_filter, sort=[("page_number", 1)], limit=limit)

    if not question_docs:
        return []

    questions = []
    for question_doc in question_docs:
        if question_doc and (question_doc.get("options") or question_doc.get("enhanced_options")):
            question_id = question_doc.get("id")

            question_figures_with_urls = []
            for fig_idx, fig_ref in enumerate(question_doc.get("question_figures", [])):
                try:
                    fig_id = fig_ref.get("id") if isinstance(fig_ref, dict) else fig_ref

                    base64_data = None
                    if isinstance(fig_ref, dict) and "base64Data" in fig_ref and fig_ref["base64Data"]:
                        base64_data = fig_ref["base64Data"]
                        logger.debug(
                            "Question %s figure %s: Using embedded base64Data (%s chars)",
                            question_id,
                            fig_idx,
                            len(base64_data)
                        )

                    img_doc = None
                    if not base64_data:
                        img_doc = await db.mongo_find_one("images", {"_id": fig_id})
                        if img_doc:
                            if "base64Data" in img_doc and img_doc["base64Data"]:
                                base64_data = img_doc["base64Data"]
                                logger.debug(
                                    "Question %s figure %s: Fetched base64Data from images collection",
                                    question_id,
                                    fig_idx
                                )
                            elif img_doc.get("file_path"):
                                import os
                                import base64 as b64
                                file_path = img_doc["file_path"]

                                if file_path.startswith("s3://"):
                                    try:
                                        from utils.s3_storage import download_file as s3_download
                                        import asyncio

                                        loop = asyncio.get_event_loop()
                                        if loop.is_running():
                                            image_bytes = await s3_download(file_path)
                                        else:
                                            image_bytes = loop.run_until_complete(s3_download(file_path))

                                        if image_bytes:
                                            base64_encoded = b64.b64encode(image_bytes).decode('utf-8')
                                            content_type = img_doc.get("content_type", "image/jpeg")
                                            if not content_type.startswith("image/"):
                                                content_type = "image/jpeg"
                                            base64_data = f"data:{content_type};base64,{base64_encoded}"
                                            logger.info("? Loaded MCQ figure %s from S3: %s bytes", fig_id, len(base64_data))
                                        else:
                                            logger.warning("?? Failed to download MCQ figure from S3: %s", file_path)
                                    except Exception as s3_err:
                                        logger.error("? S3 download error for %s: %s", file_path, s3_err)
                                else:
                                    if file_path.startswith("uploads/") or file_path.startswith("uploads\\"):
                                        file_path = os.path.join(os.getcwd(), file_path.replace("\\", "/"))

                                    if os.path.exists(file_path):
                                        try:
                                            with open(file_path, "rb") as file_handle:
                                                image_bytes = file_handle.read()
                                                base64_encoded = b64.b64encode(image_bytes).decode('utf-8')
                                                content_type = img_doc.get("content_type", "image/jpeg")
                                                if not content_type.startswith("image/"):
                                                    content_type = "image/jpeg"
                                                base64_data = f"data:{content_type};base64,{base64_encoded}"
                                                logger.info("? Loaded MCQ figure %s from file: %s bytes", fig_id, len(base64_data))
                                        except Exception as file_err:
                                            logger.error("? Failed to read MCQ figure file %s: %s", file_path, file_err)
                                    else:
                                        logger.warning("?? MCQ figure file not found: %s", file_path)
                            else:
                                logger.warning(
                                    "Question %s figure %s: Image doc found but no base64Data or file_path",
                                    question_id,
                                    fig_idx
                                )

                    figure_data = {
                        "id": fig_id,
                        "url": f"/api/v1/images/{fig_id}",
                        "contentType": (img_doc.get("content_type", "image/jpeg") if img_doc else "image/jpeg"),
                        "filename": (img_doc.get("original_filename", fig_id) if img_doc else fig_id),
                        "base64Data": base64_data,
                        "description": (fig_ref.get("description", "") if isinstance(fig_ref, dict) else "")
                    }

                    question_figures_with_urls.append(figure_data)

                    if not base64_data:
                        logger.warning(
                            "Question %s figure %s: No base64Data available, frontend will try URL",
                            question_id,
                            fig_idx
                        )

                except Exception as exc:
                    logger.error(
                        "Error processing figure %s for question %s: %s",
                        fig_idx,
                        question_id,
                        str(exc)
                    )
                    question_figures_with_urls.append({
                        "id": f"error_{fig_idx}",
                        "url": "",
                        "contentType": "image/jpeg",
                        "filename": "image_error",
                        "base64Data": None,
                        "description": "Image loading error - please report to admin"
                    })

            enhanced_options = question_doc.get("enhanced_options", [])

            for opt in enhanced_options:
                if opt.get('type') == 'image':
                    content_preview = opt.get('content', '')[:50] if opt.get('content') else 'NO CONTENT'
                    logger.info(
                        "Question %s option %s: type=image, content_length=%s, preview=%s",
                        question_id,
                        opt.get('label'),
                        len(opt.get('content', '')),
                        content_preview
                    )

            question_data = {
                "id": question_id,
                "text": question_doc.get("text", ""),
                "subject": question_doc.get("subject", ""),
                "difficulty": question_doc.get("difficulty", "medium"),
                "questionType": question_doc.get("question_type", "mcq"),
                "options": question_doc.get("options", []),
                "enhancedOptions": enhanced_options,
                "questionFigures": question_figures_with_urls,
                "correctAnswer": question_doc.get("correct_answer", ""),
                "metadata": question_doc.get("metadata", {})
            }

            questions.append(question_data)

    return questions
