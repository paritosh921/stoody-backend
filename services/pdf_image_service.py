"""
Image processing helpers for PDF OCR extraction.
"""

import base64
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiofiles

from core.database import DatabaseManager
from utils.path_utils import get_relative_path
from utils.s3_storage import upload_file as s3_upload_file, is_s3_enabled

logger = logging.getLogger(__name__)


def split_composite_image(image_data: bytes, image_id: str) -> List[bytes]:
    """
    Detect if image contains multiple option figures (A, B, C, D) arranged horizontally or in grid
    and split them into individual images. Returns list of image bytes.
    """
    try:
        from PIL import Image
        import io
        import numpy as np

        # Load image
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size

        # Convert to grayscale for analysis
        gray_img = img.convert('L')
        img_array = np.array(gray_img)

        # Check aspect ratio - if very wide (width > 2.5 * height), likely horizontal arrangement
        # If roughly square/tall, likely vertical or grid arrangement
        aspect_ratio = width / height if height > 0 else 1

        logger.info("Analyzing image %s: size=%sx%s, aspect_ratio=%.2f", image_id, width, height, aspect_ratio)

        # Heuristic: If aspect ratio > 2.5, likely 4 figures arranged horizontally
        # If aspect ratio between 1.5 and 2.5, likely 2 figures side by side
        if aspect_ratio > 2.0:
            # Try splitting horizontally into 4 parts for (A) (B) (C) (D)
            num_splits = 4 if aspect_ratio > 2.5 else 2
            split_width = width // num_splits

            logger.info(
                "Image %s has wide aspect ratio %.2f, splitting into %s horizontal parts",
                image_id,
                aspect_ratio,
                num_splits
            )

            split_images = []
            for i in range(num_splits):
                left = i * split_width
                right = (i + 1) * split_width if i < num_splits - 1 else width
                cropped = img.crop((left, 0, right, height))

                # Convert back to bytes
                output = io.BytesIO()
                cropped.save(output, format='JPEG', quality=95)
                split_images.append(output.getvalue())

            logger.info("Successfully split %s into %s images", image_id, len(split_images))
            return split_images

        # If aspect ratio suggests grid (roughly square), try 2x2 split
        elif 0.8 <= aspect_ratio <= 1.5 and width > 300 and height > 300:
            logger.info("Image %s has grid-like aspect ratio %.2f, splitting into 2x2 grid", image_id, aspect_ratio)

            split_images = []
            mid_width = width // 2
            mid_height = height // 2

            # Top-left, top-right, bottom-left, bottom-right
            for row in range(2):
                for col in range(2):
                    left = col * mid_width
                    right = (col + 1) * mid_width if col == 0 else width
                    top = row * mid_height
                    bottom = (row + 1) * mid_height if row == 0 else height

                    cropped = img.crop((left, top, right, bottom))
                    output = io.BytesIO()
                    cropped.save(output, format='JPEG', quality=95)
                    split_images.append(output.getvalue())

            logger.info("Successfully split %s into %s grid images", image_id, len(split_images))
            return split_images

        # Not a composite image
        logger.info("Image %s does not appear to be a composite (aspect_ratio=%.2f)", image_id, aspect_ratio)
        return [image_data]

    except Exception as exc:
        logger.warning("Failed to analyze/split image %s: %s", image_id, str(exc))
        # Return original if splitting fails
        return [image_data]


async def save_image_to_disk(
    image_base64: str,
    image_id: str,
    pdf_filename: str,
    db: DatabaseManager,
    user_id: str,
    split_composite: bool = True,
    is_b2c: bool = False
) -> List[Dict[str, Any]]:
    """
    Save extracted image to disk and return metadata.
    If split_composite=True, detects and splits composite images with multiple option figures.
    Returns list of saved image metadata (1 item if not split, multiple if split).
    """
    try:
        logger.info("Saving image %s, base64 preview: %s...", image_id, image_base64[:100])

        # Strip data URI prefix if present (e.g., "data:image/png;base64,")
        if ',' in image_base64 and image_base64.startswith('data:'):
            logger.info("Stripping data URI prefix from %s", image_id)
            image_base64 = image_base64.split(',', 1)[1]

        # Decode base64 image
        image_data = base64.b64decode(image_base64)

        # Log decoded data info
        logger.info("Decoded %s bytes, first 16 bytes: %s", len(image_data), image_data[:16].hex())

        # Detect actual image format from magic bytes
        def detect_image_format(data: bytes) -> tuple[str, str]:
            """Detect image format from binary data, return (extension, content_type)"""
            if data.startswith(b'\xFF\xD8\xFF'):
                return 'jpeg', 'image/jpeg'
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'png', 'image/png'
            elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
                return 'gif', 'image/gif'
            elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:
                return 'webp', 'image/webp'
            else:
                # Default to PNG if unknown
                logger.warning(
                    "Unknown image format for %s, defaulting to PNG. First 16 bytes: %s",
                    image_id,
                    data[:16].hex()
                )
                return 'png', 'image/png'

        # Split composite image if enabled
        image_parts = split_composite_image(image_data, image_id) if split_composite else [image_data]
        was_split = len(image_parts) > 1

        # Define uploads directory structure (only create if NOT using S3)
        upload_dir = os.path.join(os.getcwd(), "uploads", "pdf_images", pdf_filename.replace('.pdf', ''))

        # Generate filename with correct extension based on actual format
        # Strip any existing extension from image_id
        base_image_id = image_id.split('.')[0] if '.' in image_id else image_id

        saved_images = []

        # ALWAYS save the original unsplit image first with base_image_id
        # This ensures question figures have access to the complete image
        original_detected_ext, original_content_type = detect_image_format(image_data)
        original_image_filename = f"{base_image_id}.{original_detected_ext}"
        original_file_path = os.path.join(upload_dir, original_image_filename)

        # Use S3 storage if enabled, otherwise save locally
        if is_s3_enabled():
            success, storage_path = await s3_upload_file(
                file_data=image_data,
                local_path=original_file_path,
                content_type=original_content_type
            )
            if success:
                logger.info("✅ Saved image to S3: %s", storage_path)
                original_relative_path = storage_path  # Store S3 path
            else:
                logger.warning("S3 upload failed, image not saved: %s", original_image_filename)
                original_relative_path = ""
        else:
            # Save locally (fallback)
            os.makedirs(upload_dir, exist_ok=True)
            async with aiofiles.open(original_file_path, "wb") as file_handle:
                await file_handle.write(image_data)
            logger.info("Saved original image locally: %s", original_image_filename)
            original_relative_path = get_relative_path(original_file_path)

        # Save original to database
        original_metadata = {
            "_id": base_image_id,
            "filename": original_image_filename,
            "original_filename": original_image_filename,
            "size": len(image_data),
            "content_type": original_content_type,
            "uploaded_by": user_id,
            "uploaded_at": datetime.utcnow(),
            "is_processed": True,
            "file_path": original_relative_path,
            "source_pdf": pdf_filename,
            "tags": ["pdf_extracted", "ocr", "original"],
            "was_split": was_split,
            "is_s3": is_s3_enabled()  # Track storage type
        }

        # Save to database (use update_one with upsert to handle re-processing)
        if is_b2c:
            await db.b2c_update_one("images", {"_id": base_image_id}, {"$set": original_metadata}, upsert=True)
        else:
            await db.mongo_update_one("images", {"_id": base_image_id}, {"$set": original_metadata}, upsert=True)

        saved_images.append({
            "id": base_image_id,
            "filename": original_image_filename,
            "path": original_relative_path,
            "url": f"/api/v1/images/{base_image_id}",
            "size": len(image_data),
            "is_original": True
        })

        # If the image was split, also save split parts with -A, -B, etc. suffixes
        if was_split:
            for idx, img_data in enumerate(image_parts):
                # Detect format for this part
                detected_ext, content_type = detect_image_format(img_data)

                # Create unique ID for each split part
                db_image_id = f"{base_image_id}-{chr(65 + idx)}"  # img-9-A, img-9-B, img-9-C, img-9-D
                image_filename = f"{db_image_id}.{detected_ext}"
                file_path = os.path.join(upload_dir, image_filename)

                # Use S3 storage if enabled, otherwise save locally
                if is_s3_enabled():
                    success, storage_path = await s3_upload_file(
                        file_data=img_data,
                        local_path=file_path,
                        content_type=content_type
                    )
                    if success:
                        logger.info("✅ Saved split part %s to S3: %s", idx + 1, storage_path)
                        relative_path = storage_path
                    else:
                        logger.warning("S3 upload failed for split part: %s", image_filename)
                        relative_path = ""
                else:
                    os.makedirs(upload_dir, exist_ok=True)
                    async with aiofiles.open(file_path, "wb") as file_handle:
                        await file_handle.write(img_data)
                    logger.info(
                        "Saved split part %s/%s locally: %s",
                        idx + 1,
                        len(image_parts),
                        image_filename
                    )
                    relative_path = get_relative_path(file_path)

                # Create image metadata for database
                image_metadata = {
                    "_id": db_image_id,
                    "filename": image_filename,
                    "original_filename": original_image_filename,
                    "size": len(img_data),
                    "content_type": content_type,
                    "uploaded_by": user_id,
                    "uploaded_at": datetime.utcnow(),
                    "is_processed": True,
                    "file_path": relative_path,
                    "source_pdf": pdf_filename,
                    "tags": ["pdf_extracted", "ocr", "split_composite"],
                    "parent_image_id": base_image_id,
                    "split_index": idx,
                    "is_s3": is_s3_enabled()
                }

                # Save to database
                if is_b2c:
                    await db.b2c_update_one("images", {"_id": db_image_id}, {"$set": image_metadata}, upsert=True)
                else:
                    await db.mongo_update_one("images", {"_id": db_image_id}, {"$set": image_metadata}, upsert=True)

                saved_images.append({
                    "id": db_image_id,
                    "filename": image_filename,
                    "path": relative_path,
                    "url": f"/api/v1/images/{db_image_id}",
                    "size": len(img_data),
                    "is_original": False
                })

        return saved_images

    except Exception as exc:
        logger.error("Failed to save image %s: %s", image_id, str(exc))
        return []
