"""Institution-owned content category validation and lookup helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


CONTENT_CATEGORY_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_CONTENT_CATEGORIES = 50
MAX_CONTENT_CATEGORY_NAME_LENGTH = 60


def normalize_content_categories(value: Any, *, strict: bool) -> List[Dict[str, Any]]:
    """Return a canonical category list without inventing institution labels."""
    if value is None:
        return []
    if not isinstance(value, list):
        if strict:
            raise ValueError("Content categories must be a list")
        return []
    if len(value) > MAX_CONTENT_CATEGORIES:
        raise ValueError(f"A school can configure at most {MAX_CONTENT_CATEGORIES} content categories")

    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_names = set()

    for raw in value:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        elif hasattr(raw, "dict"):
            raw = raw.dict()

        if not isinstance(raw, dict):
            if strict:
                raise ValueError("Every content category must be an object")
            continue

        category_id = str(raw.get("id") or "").strip().lower()
        name = " ".join(str(raw.get("name") or "").split())
        active = bool(raw.get("active", True))

        if not CONTENT_CATEGORY_ID_PATTERN.fullmatch(category_id):
            if strict:
                raise ValueError(
                    "Content category IDs must use lowercase letters, numbers, and hyphens"
                )
            continue
        if not name or len(name) > MAX_CONTENT_CATEGORY_NAME_LENGTH:
            if strict:
                raise ValueError(
                    f"Content category names must be 1-{MAX_CONTENT_CATEGORY_NAME_LENGTH} characters"
                )
            continue

        folded_name = name.casefold()
        if category_id in seen_ids:
            if strict:
                raise ValueError(f"Duplicate content category ID: {category_id}")
            continue
        if folded_name in seen_names:
            if strict:
                raise ValueError(f"Duplicate content category name: {name}")
            continue

        seen_ids.add(category_id)
        seen_names.add(folded_name)
        normalized.append({"id": category_id, "name": name, "active": active})

    return normalized


def find_content_category(
    categories: Iterable[Dict[str, Any]],
    category_id: Optional[str],
    *,
    require_active: bool,
) -> Optional[Dict[str, Any]]:
    normalized_id = str(category_id or "").strip().lower()
    if not normalized_id:
        return None

    for category in categories:
        if category.get("id") != normalized_id:
            continue
        if require_active and not category.get("active", True):
            raise ValueError("The selected content category is archived")
        return category
    raise ValueError("The selected content category is not configured for this school")

