"""
Image validation utilities for SkillBot
Validates image existence across database and filesystem
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


async def validate_image_exists(image_id: str, db) -> bool:
    """
    Check if an image exists in both database and filesystem

    Args:
        image_id: Image ID to validate
        db: DatabaseManager instance

    Returns:
        True if image exists in database and on disk, False otherwise
    """
    try:
        # Check database
        image_data = await db.mongo_find_one("images", {"_id": image_id})
        if not image_data:
            logger.debug(f"Image {image_id} not found in database")
            return False

        # Check filesystem
        from utils.path_utils import get_absolute_path

        stored_path = image_data.get("file_path")
        if not stored_path:
            logger.debug(f"Image {image_id} has no file_path in database")
            return False

        # Normalize path separators
        stored_path_str = str(stored_path).replace("\\", "/")

        # If the path contains 'uploads/', use that portion
        uploads_idx = stored_path_str.lower().find("uploads/")
        if uploads_idx != -1:
            relative_from_uploads = stored_path_str[uploads_idx:]
            file_path = get_absolute_path(relative_from_uploads)
        else:
            if Path(stored_path_str).is_absolute():
                file_path = Path(stored_path_str)
            else:
                file_path = get_absolute_path(stored_path_str)

        exists = file_path.exists()
        if not exists:
            logger.debug(f"Image {image_id} file not found at {file_path}")

        return exists

    except Exception as e:
        logger.error(f"Error validating image {image_id}: {str(e)}")
        return False


async def validate_images_list(image_refs: List[Dict[str, Any]], db) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate a list of image references and separate valid from invalid

    Args:
        image_refs: List of image reference dictionaries with 'id' field
        db: DatabaseManager instance

    Returns:
        Tuple of (valid_images, invalid_image_ids)
    """
    valid_images = []
    invalid_image_ids = []

    for img_ref in image_refs:
        image_id = img_ref.get("id")
        if not image_id:
            logger.warning(f"Image reference missing 'id' field: {img_ref}")
            continue

        if await validate_image_exists(image_id, db):
            valid_images.append(img_ref)
        else:
            invalid_image_ids.append(image_id)
            logger.info(f"Invalid image reference removed: {image_id}")

    return valid_images, invalid_image_ids


async def clean_question_images(question: Dict[str, Any], db) -> Tuple[Dict[str, Any], int]:
    """
    Clean orphaned image references from a question

    Args:
        question: Question document from database
        db: DatabaseManager instance

    Returns:
        Tuple of (cleaned_question, removed_count)
    """
    removed_count = 0

    # Clean 'images' field
    if question.get("images"):
        valid_images, invalid_ids = await validate_images_list(question["images"], db)
        removed_count += len(invalid_ids)
        question["images"] = valid_images

        if invalid_ids:
            logger.info(f"Removed {len(invalid_ids)} invalid images from question {question.get('id')}: {invalid_ids}")

    # Clean 'question_figures' field
    if question.get("question_figures"):
        valid_figures, invalid_ids = await validate_images_list(question["question_figures"], db)
        removed_count += len(invalid_ids)
        question["question_figures"] = valid_figures

        if invalid_ids:
            logger.info(f"Removed {len(invalid_ids)} invalid figures from question {question.get('id')}: {invalid_ids}")

    return question, removed_count


async def get_orphaned_images_in_question(question_id: str, db) -> List[str]:
    """
    Get list of orphaned image IDs in a specific question

    Args:
        question_id: Question ID to check
        db: DatabaseManager instance

    Returns:
        List of orphaned image IDs
    """
    orphaned = []

    question = await db.mongo_find_one("questions", {"id": question_id})
    if not question:
        return orphaned

    # Check images
    if question.get("images"):
        for img_ref in question["images"]:
            image_id = img_ref.get("id")
            if image_id and not await validate_image_exists(image_id, db):
                orphaned.append(image_id)

    # Check question_figures
    if question.get("question_figures"):
        for fig_ref in question["question_figures"]:
            figure_id = fig_ref.get("id")
            if figure_id and not await validate_image_exists(figure_id, db):
                orphaned.append(figure_id)

    return orphaned


async def get_orphaned_images_in_document(document_id: str, db) -> Dict[str, List[str]]:
    """
    Get all orphaned image references in a document

    Args:
        document_id: Document ID to check
        db: DatabaseManager instance

    Returns:
        Dictionary mapping question_id to list of orphaned image IDs
    """
    orphaned_by_question = {}

    questions = await db.mongo_find("questions", {"document_id": document_id})

    for question in questions:
        question_id = question.get("id")
        if not question_id:
            continue

        orphaned = await get_orphaned_images_in_question(question_id, db)
        if orphaned:
            orphaned_by_question[question_id] = orphaned

    return orphaned_by_question
