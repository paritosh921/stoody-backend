"""Safe, answer-key-free orientation views for v15 evidence mapping.

The original stored page is always retained.  A strong sideways signal adds
clockwise 90 and 270 degree views of that *same physical page* so a visual
mapper can choose the readable candidate without a second document identity.
Coordinates emitted against an alternate view can be inverted to the original
stored-page frame through the pure transforms in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from services.page_orientation import detect_sideways_page


NORMALIZED_SCALE = 1000.0
_VALID_ROTATIONS = {0, 90, 270}


class OrientationViewError(ValueError):
    """Raised when an alternate view cannot be produced safely."""


@dataclass(frozen=True)
class OrientationView:
    """One visual candidate for one immutable physical page."""

    physical_page_number: int
    view_id: str
    alternate_of: str
    rotation_degrees_clockwise: int
    image_bytes: bytes
    width_px: int
    height_px: int
    coordinate_frame: Dict[str, Any]
    orientation_evidence: Dict[str, Any]
    is_original: bool

    def as_manifest(self) -> Dict[str, Any]:
        """Return metadata safe to persist beside the original asset."""

        return {
            "physical_page_number": self.physical_page_number,
            "view_id": self.view_id,
            "alternate_of": self.alternate_of,
            "rotation_degrees_clockwise": self.rotation_degrees_clockwise,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "is_original": self.is_original,
            "coordinate_frame": dict(self.coordinate_frame),
            "orientation_evidence": dict(self.orientation_evidence),
        }


def build_orientation_views(
    image_bytes: bytes,
    *,
    physical_page_number: int,
    width_px: Optional[int] = None,
    height_px: Optional[int] = None,
    line_ratio: float = 1.45,
    detector: Callable[..., Tuple[bool, Dict[str, Any]]] = detect_sideways_page,
) -> Tuple[OrientationView, ...]:
    """Build original plus bounded readable candidates for one stored page.

    A missing/weak detector signal is fail-closed and returns the original only.
    When the detector proves a page is sideways, both possible readable
    directions are exposed; the detector intentionally cannot choose clockwise
    versus counter-clockwise.  No answer content or answer key is inspected.
    """

    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        raise OrientationViewError("A non-empty original page image is required")
    try:
        page_number = int(physical_page_number)
    except (TypeError, ValueError) as exc:
        raise OrientationViewError("Physical page number must be positive") from exc
    if page_number <= 0:
        raise OrientationViewError("Physical page number must be positive")

    original_bytes = bytes(image_bytes)
    width, height = _resolve_dimensions(original_bytes, width_px, height_px)
    try:
        sideways, detector_evidence = detector(original_bytes, line_ratio=line_ratio)
    except Exception as exc:  # detector failures never trigger a blind rotation
        sideways = False
        detector_evidence = {
            "method": "line_projection",
            "reason": "detector_error",
            "error_type": type(exc).__name__,
        }
    evidence = dict(detector_evidence or {})
    evidence["sideways"] = bool(sideways)
    original_id = f"physical-page-{page_number}-original"
    frame_id = f"physical-page-{page_number}-original-frame"
    views = [
        _make_view(
            page_number=page_number,
            view_id=original_id,
            alternate_of=original_id,
            rotation=0,
            image_bytes=original_bytes,
            width=width,
            height=height,
            frame_id=frame_id,
            original_width=width,
            original_height=height,
            evidence=evidence,
            is_original=True,
        )
    ]
    if not sideways:
        return tuple(views)

    for rotation in (90, 270):
        rotated_bytes, rotated_width, rotated_height = _rotate_image(
            original_bytes, rotation, width, height
        )
        views.append(
            _make_view(
                page_number=page_number,
                view_id=f"physical-page-{page_number}-rotation-{rotation}",
                alternate_of=original_id,
                rotation=rotation,
                image_bytes=rotated_bytes,
                width=rotated_width,
                height=rotated_height,
                frame_id=frame_id,
                original_width=width,
                original_height=height,
                evidence=evidence,
                is_original=False,
            )
        )
    return tuple(views)


def view_point_to_original(
    x: float,
    y: float,
    *,
    rotation_degrees_clockwise: int,
) -> Tuple[float, float]:
    """Convert normalized-1000 view coordinates into original-page coordinates."""

    rotation = _normalise_rotation(rotation_degrees_clockwise)
    nx, ny = _normalise_point(x, y)
    if rotation == 0:
        return nx * NORMALIZED_SCALE, ny * NORMALIZED_SCALE
    if rotation == 90:
        return ny * NORMALIZED_SCALE, (1.0 - nx) * NORMALIZED_SCALE
    return (1.0 - ny) * NORMALIZED_SCALE, nx * NORMALIZED_SCALE


def original_point_to_view(
    x: float,
    y: float,
    *,
    rotation_degrees_clockwise: int,
) -> Tuple[float, float]:
    """Convert normalized-1000 original-page coordinates into a view."""

    rotation = _normalise_rotation(rotation_degrees_clockwise)
    nx, ny = _normalise_point(x, y)
    if rotation == 0:
        return nx * NORMALIZED_SCALE, ny * NORMALIZED_SCALE
    if rotation == 90:
        return (1.0 - ny) * NORMALIZED_SCALE, nx * NORMALIZED_SCALE
    return ny * NORMALIZED_SCALE, (1.0 - nx) * NORMALIZED_SCALE


def view_region_to_original(
    region: Mapping[str, Any],
    *,
    rotation_degrees_clockwise: int,
) -> Dict[str, Any]:
    """Invert a normalized-1000 rectangle while preserving audit metadata."""

    try:
        points = [
            view_point_to_original(region[key], region[other], rotation_degrees_clockwise=rotation_degrees_clockwise)
            for key, other in (("x_start", "y_start"), ("x_start", "y_end"), ("x_end", "y_start"), ("x_end", "y_end"))
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise OrientationViewError("A normalized region is missing finite coordinates") from exc
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    result = dict(region)
    result.update({
        "x_start": round(min(xs), 3),
        "y_start": round(min(ys), 3),
        "x_end": round(max(xs), 3),
        "y_end": round(max(ys), 3),
        "coordinate_space": "normalized_1000",
        "source_rotation_degrees_clockwise": _normalise_rotation(rotation_degrees_clockwise),
        "coordinate_transform": {
            "type": "normalized_1000_view_to_original",
            "rotation_degrees_clockwise": _normalise_rotation(rotation_degrees_clockwise),
            "invertible": True,
        },
    })
    return result


def _make_view(
    *,
    page_number: int,
    view_id: str,
    alternate_of: str,
    rotation: int,
    image_bytes: bytes,
    width: int,
    height: int,
    frame_id: str,
    original_width: int,
    original_height: int,
    evidence: Mapping[str, Any],
    is_original: bool,
) -> OrientationView:
    return OrientationView(
        physical_page_number=page_number,
        view_id=view_id,
        alternate_of=alternate_of,
        rotation_degrees_clockwise=rotation,
        image_bytes=image_bytes,
        width_px=width,
        height_px=height,
        coordinate_frame={
            "id": frame_id,
            "kind": "original_stored_page",
            "coordinate_space": "normalized_1000",
            "width_px": original_width,
            "height_px": original_height,
            "view_rotation_degrees_clockwise": rotation,
            "original_width_px": original_width,
            "original_height_px": original_height,
            "invertible": True,
        },
        orientation_evidence=dict(evidence),
        is_original=is_original,
    )


def _resolve_dimensions(image_bytes: bytes, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if width is not None and height is not None:
        try:
            resolved_width, resolved_height = int(width), int(height)
        except (TypeError, ValueError) as exc:
            raise OrientationViewError("Page dimensions must be positive integers") from exc
        if resolved_width > 0 and resolved_height > 0:
            return resolved_width, resolved_height
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            resolved_width, resolved_height = image.size
    except Exception as exc:
        raise OrientationViewError("Page dimensions are required for an unreadable image") from exc
    if resolved_width <= 0 or resolved_height <= 0:
        raise OrientationViewError("Page dimensions must be positive")
    return int(resolved_width), int(resolved_height)


def _rotate_image(image_bytes: bytes, rotation: int, width: int, height: int) -> Tuple[bytes, int, int]:
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            rotated = image.convert("RGB").rotate(-rotation, expand=True, fillcolor="white")
            output = BytesIO()
            rotated.save(output, format="PNG", optimize=True)
            return output.getvalue(), int(rotated.width), int(rotated.height)
    except Exception as exc:
        raise OrientationViewError("A sideways page could not be safely rotated") from exc


def _normalise_rotation(value: Any) -> int:
    try:
        rotation = int(value) % 360
    except (TypeError, ValueError) as exc:
        raise OrientationViewError("Rotation must be 0, 90, or 270 degrees clockwise") from exc
    if rotation not in _VALID_ROTATIONS:
        raise OrientationViewError("Rotation must be 0, 90, or 270 degrees clockwise")
    return rotation


def _normalise_point(x: Any, y: Any) -> Tuple[float, float]:
    try:
        nx, ny = float(x) / NORMALIZED_SCALE, float(y) / NORMALIZED_SCALE
    except (TypeError, ValueError) as exc:
        raise OrientationViewError("Coordinates must be numeric") from exc
    if not all(0.0 <= value <= 1.0 for value in (nx, ny)):
        raise OrientationViewError("Normalized coordinates must be between 0 and 1000")
    return nx, ny
