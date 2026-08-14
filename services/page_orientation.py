"""Deterministic orientation signals for photographed or rendered answer pages.

The detector deliberately answers only the safe question that pixels can prove:
whether long ruled-page lines are predominantly vertical.  It does not guess
whether clockwise or counter-clockwise rotation is correct.  Callers must expose
both readable candidates to the visual model and persist the selected transform
before interpreting any normalized coordinates.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def detect_sideways_page(
    image_bytes: bytes,
    *,
    line_ratio: float = 1.45,
) -> Tuple[bool, Dict[str, Any]]:
    """Return whether a ruled page is safely identifiable as sideways.

    OpenCV is optional in some worker deployments.  An unavailable detector is
    reported as evidence and resolves to ``False``; callers must never rotate a
    page on a missing or weak signal.
    """

    try:
        import cv2
        import numpy as np

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return False, {"method": "line_projection", "reason": "decode_failed"}

        height, width = image.shape[:2]
        largest = max(height, width)
        if largest > 1400:
            scale = 1400.0 / largest
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = image.shape[:2]

        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        longest = max(height, width)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(30, int(min(height, width) * 0.06)),
            minLineLength=max(60, int(longest * 0.22)),
            maxLineGap=max(12, int(longest * 0.025)),
        )
        horizontal_support = 0.0
        vertical_support = 0.0
        horizontal_count = 0
        vertical_count = 0
        if lines is not None:
            for raw_line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = (int(value) for value in raw_line)
                dx, dy = x2 - x1, y2 - y1
                length = float((dx * dx + dy * dy) ** 0.5)
                if length <= 0:
                    continue
                angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
                folded = min(angle, 180.0 - angle)
                if folded <= 12.0:
                    horizontal_support += length
                    horizontal_count += 1
                elif folded >= 78.0:
                    vertical_support += length
                    vertical_count += 1

        enough_signal = (
            vertical_count >= 2
            and vertical_support >= longest
            and (vertical_support + horizontal_support) >= longest * 1.5
        )
        sideways = bool(
            enough_signal
            and vertical_support > horizontal_support * max(1.15, float(line_ratio))
        )
        return sideways, {
            "method": "line_projection",
            "horizontal_support": round(horizontal_support, 1),
            "vertical_support": round(vertical_support, 1),
            "horizontal_lines": horizontal_count,
            "vertical_lines": vertical_count,
            "sideways": sideways,
        }
    except Exception as exc:
        return False, {
            "method": "line_projection",
            "reason": "detector_unavailable",
            "error_type": type(exc).__name__,
        }
