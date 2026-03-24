"""Miss indicator auto-detection. ZERO I/O — pure logic.

Determines per-question miss state based on stroke presence and
sync metadata. Teacher overrides are NOT handled here (they are
written by svc-score-engine).

States:
  answered           - strokes present in question region
  miss_no_strokes    - no strokes in region at all
  miss_sync_failure  - sync metadata incomplete for this pen
  miss_pen_inactive  - pen was connected but no writing detected
"""

from __future__ import annotations

from src.domain.models import (
    CanonicalPoint,
    MissAutoState,
    QuestionRegion,
    Stroke,
    SyncMetadata,
)


def _strokes_in_region(
    strokes: list[Stroke],
    region: QuestionRegion,
) -> list[Stroke]:
    """Return strokes that have at least one point inside the region."""
    result: list[Stroke] = []
    for stroke in strokes:
        for pt in stroke.points:
            if (
                region.x_min <= pt.x <= region.x_max
                and region.y_min <= pt.y <= region.y_max
            ):
                result.append(stroke)
                break
    return result


def _count_points_in_region(
    strokes: list[Stroke],
    region: QuestionRegion,
) -> int:
    """Count total points falling inside the region."""
    count = 0
    for stroke in strokes:
        for pt in stroke.points:
            if (
                region.x_min <= pt.x <= region.x_max
                and region.y_min <= pt.y <= region.y_max
            ):
                count += 1
    return count


def detect_miss_state(
    strokes: list[Stroke],
    region: QuestionRegion,
    sync_metadata: SyncMetadata | None,
) -> MissAutoState:
    """Detect the miss indicator auto-state for a single question region.

    Decision tree:
    1. If sync metadata is missing or sync was incomplete -> miss_sync_failure
    2. If pen was connected but no strokes on entire page -> miss_pen_inactive
    3. If strokes exist in this region -> answered
    4. Otherwise -> miss_no_strokes

    The caller is responsible for supplying only strokes belonging to the
    relevant page. This function checks region intersection within that page.
    """
    # Sync failure takes priority: we cannot trust stroke absence
    if sync_metadata is not None and not sync_metadata.sync_complete:
        return MissAutoState.MISS_SYNC_FAILURE

    if sync_metadata is None:
        return MissAutoState.MISS_SYNC_FAILURE

    # Pen was connected but produced zero strokes on this page
    if sync_metadata.pen_connected and len(strokes) == 0:
        return MissAutoState.MISS_PEN_INACTIVE

    # Check for strokes in the question region
    hits = _strokes_in_region(strokes, region)
    if hits:
        return MissAutoState.ANSWERED

    return MissAutoState.MISS_NO_STROKES


def detect_all_regions(
    strokes: list[Stroke],
    regions: list[QuestionRegion],
    sync_metadata: SyncMetadata | None,
) -> dict[str, MissAutoState]:
    """Detect miss state for every question region on a page.

    Returns a dict mapping question_id -> MissAutoState.
    """
    return {
        region.question_id: detect_miss_state(strokes, region, sync_metadata)
        for region in regions
    }
