"""Compatibility projection for durably stored image answer options."""

from __future__ import annotations

import base64
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

from utils.path_utils import get_absolute_path
from utils.s3_storage import download_file


logger = logging.getLogger(__name__)


def enhanced_option_image_id(option: Dict[str, Any]) -> str:
    """Return the storage id from current and legacy option shapes."""

    explicit = str(option.get("image_id") or "").strip()
    if explicit:
        return explicit
    content = str(option.get("content") or "").strip()
    if content.startswith("/api/v1/images/"):
        return content.rsplit("/", 1)[-1].strip()
    if content and not content.startswith(("data:", "/", "http://", "https://", "s3://")):
        # Older records stored the image id directly in ``content``.
        return content
    return str(option.get("id") or "").strip()


def _as_data_uri(value: Any, content_type: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("data:image/"):
        return raw
    return f"data:{content_type};base64,{raw}"


async def enrich_enhanced_option_images(
    options: Any,
    *,
    db: Any,
    is_b2c: bool,
) -> List[Dict[str, Any]]:
    """Return response-only image data while Mongo keeps durable references.

    Released web and mobile clients already understand data-image option
    content. The authoring store, however, must contain an image id/path rather
    than multi-megabyte inline data. This projection preserves both contracts
    without mutating the stored question document.
    """

    if not isinstance(options, list):
        return []
    projected = copy.deepcopy(options)
    for option in projected:
        if not isinstance(option, dict) or str(option.get("type") or "").lower() != "image":
            continue
        content = str(option.get("content") or "").strip()
        if content.startswith("data:image/"):
            continue
        image_id = enhanced_option_image_id(option)
        if not image_id:
            continue
        try:
            image_doc = (
                await db.b2c_find_one("images", {"_id": image_id})
                if is_b2c
                else await db.mongo_find_one("images", {"_id": image_id})
            )
            if not image_doc:
                continue
            content_type = str(image_doc.get("content_type") or "image/png")
            if not content_type.startswith("image/"):
                content_type = "image/png"
            data_uri = _as_data_uri(
                image_doc.get("base64Data") or image_doc.get("base64_data"),
                content_type,
            )
            if not data_uri:
                stored_path = image_doc.get("file_path") or image_doc.get("storage_path")
                if stored_path:
                    resolved_path: Any = stored_path
                    if not str(stored_path).startswith("s3://") and not Path(str(stored_path)).is_absolute():
                        resolved_path = get_absolute_path(str(stored_path))
                    image_bytes = await download_file(str(resolved_path))
                    if image_bytes:
                        data_uri = (
                            f"data:{content_type};base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        )
            if data_uri:
                option["content"] = data_uri
                option["image_id"] = image_id
                option.setdefault("url", f"/api/v1/images/{image_id}")
        except Exception as exc:
            logger.warning("Could not project image option %s for a client response: %s", image_id, exc)
    return projected
