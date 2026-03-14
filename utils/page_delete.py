"""
Shared page-delete helper.

Cascade-deletes all data for a single page identified by
(pen_mac, book_type, page_number), regardless of whether a
note_classifications record exists.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


async def delete_page_by_identity(
    tenant_db,
    pen_mac: str,
    book_type: str,
    page_number: int,
    user_id: str,
    user_id_variants: List[str],
) -> Dict[str, Any]:
    """
    Delete all data for a single page identified by (pen_mac, book_type, page_number).

    Cascade (classification doc deleted last for recoverability):
    1. Find note_classifications doc (if exists) — grab thumbnail_url & _id
    2. Delete S3 thumbnail (if thumbnail_url found)
    3. Delete from strokes collection
    4. Delete from canvas_pages collection
    5. Delete from classification_queue (if exists)
    6. Delete the note_classifications doc itself (if found in step 1)

    Returns { deleted_strokes, deleted_canvas_pages, deleted_classification,
              had_data } where had_data is True when at least one document
              was actually removed.
    """
    pen_mac_upper = pen_mac.upper()

    # 1. Find classification doc (may not exist for unclassified pages)
    cls_doc = None
    thumbnail_url = None
    cls_query = {
        "user_id": {"$in": user_id_variants},
        "pen_mac": {"$regex": f"^{pen_mac}$", "$options": "i"},
        "book_type": book_type,
        "page_number": page_number,
    }
    cls_doc = await tenant_db["note_classifications"].find_one(cls_query)
    if cls_doc:
        thumbnail_url = cls_doc.get("thumbnail_url")

    # 2. Delete S3 thumbnail
    if thumbnail_url:
        try:
            from utils.s3_storage import delete_file
            await delete_file(thumbnail_url)
        except Exception as e:
            logger.warning(f"Failed to delete S3 thumbnail {thumbnail_url}: {e}")

    # 3. Delete strokes for this page
    from services.note_classification_service import _build_user_id_match
    user_match = _build_user_id_match(user_id)
    stroke_result = await tenant_db["strokes"].delete_many({
        "user_id": user_match,
        "pen_mac": {"$regex": f"^{pen_mac}$", "$options": "i"},
        "book_type": book_type,
        "page_number": page_number,
    })

    # 4. Delete canvas_pages document(s) for this page
    canvas_pages_deleted = 0
    try:
        cp_result = await tenant_db["canvas_pages"].delete_many({
            "user_id": {"$in": user_id_variants},
            "pen_mac": {"$regex": f"^{pen_mac}$", "$options": "i"},
            "book_type": book_type,
            "page_number": page_number,
        })
        canvas_pages_deleted = cp_result.deleted_count
    except Exception as e:
        logger.warning(f"Failed to delete canvas_pages for page {page_number}: {e}")

    # 5. Delete classification queue entry if exists
    await tenant_db["classification_queue"].delete_many({
        "user_id": {"$in": user_id_variants},
        "pen_mac": pen_mac_upper,
        "book_type": book_type,
        "page_number": page_number,
    })

    # 6. Delete the classification doc itself last (safe: leaf data already gone)
    deleted_classification = False
    if cls_doc:
        await tenant_db["note_classifications"].delete_one({"_id": cls_doc["_id"]})
        deleted_classification = True

    had_data = (
        deleted_classification
        or stroke_result.deleted_count > 0
        or canvas_pages_deleted > 0
    )

    return {
        "deleted_strokes": stroke_result.deleted_count,
        "deleted_canvas_pages": canvas_pages_deleted,
        "deleted_classification": deleted_classification,
        "had_data": had_data,
    }
