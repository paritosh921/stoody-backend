from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def is_flag_resolved(flag: Any) -> bool:
    """Read canonical nested resolutions and older top-level mirrors safely."""

    if not isinstance(flag, dict):
        return False
    if bool(flag.get("resolved")):
        return True
    resolution = flag.get("resolution")
    return bool(isinstance(resolution, dict) and resolution.get("resolved"))


def resolve_flag(
    flag: Dict[str, Any],
    *,
    actor_id: str,
    resolved_at: datetime,
    reason: str,
    action: str = "approved_by_teacher",
) -> Dict[str, Any]:
    """Return a resolved flag using the canonical nested resolution contract."""

    resolved = dict(flag)
    existing_resolution = flag.get("resolution")
    resolution = (
        dict(existing_resolution)
        if isinstance(existing_resolution, dict)
        else {}
    )
    resolution.update(
        {
            "resolved": True,
            "resolution": action,
            "note": reason,
            "resolved_by": actor_id,
            "resolved_at": resolved_at,
        }
    )
    resolved["resolution"] = resolution
    resolved["resolved"] = True
    return resolved
