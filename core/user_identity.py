"""
Shared user-identity resolution for canvas/note data.

The write path (strokes_async.py) stores ``user_id = username``, while JWT
tokens carry ``sub = ObjectId``.  These helpers centralise the mapping so
every read path queries with **all** known variants.
"""

from __future__ import annotations

from typing import Any, Dict, List


def canonical_canvas_user_id(current_user: Dict[str, Any]) -> str:
    """Canonical owner id for page data — username-style."""
    username = current_user.get("username")
    if username:
        return str(username)
    return str(current_user.get("user_id") or current_user.get("_id"))


def canvas_user_id_variants(current_user: Dict[str, Any]) -> List[str]:
    """All possible user_id values for querying user-owned canvas/note documents."""
    variants: set[str] = set()
    uid = current_user.get("user_id") or current_user.get("_id")
    if uid:
        variants.add(str(uid))
    username = current_user.get("username")
    if username:
        variants.add(str(username))
    return list(variants)
