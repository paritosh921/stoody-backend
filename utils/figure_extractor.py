"""
Deterministic figure/diagram detection using OpenCV.

Standalone module — no FastAPI, database, or OpenAI dependencies.
Takes a page image (bytes), returns candidate figure regions with
cropped base64 thumbnails for downstream assignment.
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FigureCandidate:
    candidate_id: str              # e.g. "cand-p0-0"
    page_index: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    area: int
    confidence: float              # 0.0-1.0
    crop_b64: Optional[str] = None  # base64 PNG of cropped region
    label: str = ""


@dataclass
class PageFigureResult:
    page_index: int
    page_width: int
    page_height: int
    candidates: List[FigureCandidate] = field(default_factory=list)


def detect_figure_candidates(
    page_image_bytes: bytes,
    page_index: int,
    *,
    header_frac: float = 0.08,
    footer_frac: float = 0.08,
    min_area_frac: float = 0.02,
    max_area_frac: float = 0.50,
    min_aspect: float = 0.25,
    max_aspect: float = 4.0,
    merge_distance: int = 30,
    block_size: int = 31,
    constant_c: int = 10,
    close_kernel_size: int = 25,
    dilate_kernel_size: int = 8,
) -> PageFigureResult:
    """
    Detect figure candidate regions on a single page image.

    Tuned to reject text blocks / formula lines and only keep actual
    diagrams, circuits, graphs.  Key heuristics:
    - min 2% of page area (rejects small text clusters)
    - aspect ratio 0.25-4.0 (rejects wide text lines & narrow strips)
    - "full-width" text-band rejection (boxes >85% page width with
      aspect >2.5 are almost certainly text paragraphs)
    - large closing kernel (25x25) + dilate (8x8) so nearby strokes
      that form a single diagram get merged before contour detection

    Returns PageFigureResult with candidates sorted top-to-bottom.
    Candidates do NOT have crop_b64 set — call crop_candidates() after.
    """
    if not CV2_AVAILABLE:
        logger.warning("[FIGURE-DETECT] OpenCV not available — returning empty results")
        return PageFigureResult(page_index=page_index, page_width=0, page_height=0)

    # Decode image
    img_array = np.frombuffer(page_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"[FIGURE-DETECT] Page {page_index}: failed to decode image")
        return PageFigureResult(page_index=page_index, page_width=0, page_height=0)

    h, w = img.shape[:2]
    page_area = h * w
    result = PageFigureResult(page_index=page_index, page_width=w, page_height=h)

    # Exclusion zones
    header_y = int(h * header_frac)
    footer_y = int(h * (1.0 - footer_frac))

    # Grayscale + adaptive threshold (BINARY_INV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, constant_c,
    )

    # Morphological operations — aggressive to merge diagram strokes
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size)
    )
    thresh = cv2.dilate(thresh, dilate_kernel, iterations=1)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours into candidate boxes
    raw_boxes: List[Tuple[int, int, int, int, int, int]] = []  # (x1,y1,x2,y2, contour_area, bbox_area)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        contour_area = cv2.contourArea(cnt)
        bbox_area = bw * bh

        # Exclude header/footer
        center_y = (y1 + y2) // 2
        if center_y < header_y or center_y > footer_y:
            continue

        # Area filter
        area_frac = bbox_area / page_area
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue

        # Aspect ratio filter
        if bh == 0:
            continue
        aspect = bw / bh
        if aspect < min_aspect or aspect > max_aspect:
            continue

        # Reject full-width text bands: boxes that span >85% of page
        # width and are wider than tall (aspect > 2.5) are text paragraphs
        width_frac = bw / w
        if width_frac > 0.85 and aspect > 2.5:
            continue

        # Fill-ratio pre-filter: very sparse boxes (< 1% filled) are
        # likely noise from large morphological kernels
        if bbox_area > 0 and contour_area / bbox_area < 0.01:
            continue

        raw_boxes.append((x1, y1, x2, y2, contour_area, bbox_area))

    if not raw_boxes:
        print(f"[FIGURE-DETECT] Page {page_index}: no candidates after filtering ({len(contours)} contours checked)", flush=True)
        return result

    # Merge overlapping/nearby boxes
    merged = _merge_overlapping_boxes(
        [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in raw_boxes],
        merge_distance,
    )

    # Post-merge filtering: re-check merged boxes against the same rules
    final_merged = []
    for mx1, my1, mx2, my2 in merged:
        mbw = mx2 - mx1
        mbh = my2 - my1
        m_area = mbw * mbh
        m_area_frac = m_area / page_area
        if m_area_frac < min_area_frac or m_area_frac > max_area_frac:
            continue
        if mbh == 0:
            continue
        m_aspect = mbw / mbh
        if m_aspect < min_aspect or m_aspect > max_aspect:
            continue
        m_width_frac = mbw / w
        if m_width_frac > 0.85 and m_aspect > 2.5:
            continue
        final_merged.append((mx1, my1, mx2, my2))

    if not final_merged:
        print(f"[FIGURE-DETECT] Page {page_index}: no candidates after post-merge filtering", flush=True)
        return result

    # Build candidates with confidence scores
    for idx, (mx1, my1, mx2, my2) in enumerate(final_merged):
        m_bbox_area = (mx2 - mx1) * (my2 - my1)
        # Sum contour areas of raw boxes that fall within this merged box
        m_contour_area = 0
        for x1, y1, x2, y2, ca, _ in raw_boxes:
            if x1 >= mx1 and y1 >= my1 and x2 <= mx2 and y2 <= my2:
                m_contour_area += ca

        conf = _compute_confidence(
            (mx1, my1, mx2, my2), w, h, m_contour_area, m_bbox_area
        )
        cand = FigureCandidate(
            candidate_id=f"cand-p{page_index}-{idx}",
            page_index=page_index,
            bbox=(mx1, my1, mx2, my2),
            area=m_bbox_area,
            confidence=conf,
        )
        result.candidates.append(cand)

    # Sort top-to-bottom by y1
    result.candidates.sort(key=lambda c: c.bbox[1])

    print(
        f"[FIGURE-DETECT] Page {page_index}: {len(result.candidates)} candidates "
        f"(from {len(contours)} contours, {len(raw_boxes)} after filter, "
        f"{len(merged)} merged, {len(final_merged)} after post-merge)",
        flush=True
    )
    return result


def crop_candidates(
    page_image_bytes: bytes,
    result: PageFigureResult,
    padding_px: int = 8,
) -> None:
    """
    Crop each candidate from the page image and set crop_b64.
    Mutates candidates in-place.
    """
    if not CV2_AVAILABLE or not result.candidates:
        return

    img_array = np.frombuffer(page_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]
    for cand in result.candidates:
        x1, y1, x2, y2 = cand.bbox
        # Apply padding
        px1 = max(0, x1 - padding_px)
        py1 = max(0, y1 - padding_px)
        px2 = min(w, x2 + padding_px)
        py2 = min(h, y2 + padding_px)

        crop = img[py1:py2, px1:px2]
        success, buf = cv2.imencode(".png", crop)
        if success:
            cand.crop_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")


def _merge_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]],
    merge_distance: int,
) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that overlap or are within merge_distance pixels."""
    if not boxes:
        return []

    # Sort by y1 then x1
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[List[int]] = [list(sorted_boxes[0])]

    for bx1, by1, bx2, by2 in sorted_boxes[1:]:
        did_merge = False
        for m in merged:
            # Check if boxes overlap or are close enough
            if (bx1 <= m[2] + merge_distance and
                bx2 >= m[0] - merge_distance and
                by1 <= m[3] + merge_distance and
                by2 >= m[1] - merge_distance):
                # Expand merged box
                m[0] = min(m[0], bx1)
                m[1] = min(m[1], by1)
                m[2] = max(m[2], bx2)
                m[3] = max(m[3], by2)
                did_merge = True
                break
        if not did_merge:
            merged.append([bx1, by1, bx2, by2])

    return [(m[0], m[1], m[2], m[3]) for m in merged]


def _compute_confidence(
    bbox: Tuple[int, int, int, int],
    page_width: int,
    page_height: int,
    contour_area: int,
    bbox_area: int,
) -> float:
    """
    Heuristic confidence score for a figure candidate.

    Factors:
    - Fill ratio (contour area vs bbox area) — higher = more likely a real figure
    - Size — very small or very large boxes are less confident
    - Position — central regions slightly preferred
    """
    if bbox_area <= 0 or page_width <= 0 or page_height <= 0:
        return 0.0

    page_area = page_width * page_height

    # Fill ratio: how much of the bounding box is filled by contour ink
    fill_ratio = min(contour_area / bbox_area, 1.0)
    # Sweet spot: 0.05-0.6 fill ratio is typical for diagrams
    if fill_ratio < 0.02:
        fill_score = 0.2
    elif fill_ratio < 0.5:
        fill_score = 0.6 + fill_ratio * 0.6
    else:
        fill_score = 0.8

    # Size score: ideal figures are 2-30% of page area
    size_frac = bbox_area / page_area
    if size_frac < 0.01:
        size_score = 0.3
    elif size_frac < 0.02:
        size_score = 0.5
    elif size_frac <= 0.30:
        size_score = 0.9
    elif size_frac <= 0.50:
        size_score = 0.7
    else:
        size_score = 0.4

    # Position score: slightly prefer central horizontal position
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / page_width
    # Figures near center or slightly right are common
    if 0.15 <= cx <= 0.85:
        pos_score = 0.9
    else:
        pos_score = 0.6

    # Weighted combination
    confidence = fill_score * 0.4 + size_score * 0.4 + pos_score * 0.2
    return round(min(max(confidence, 0.0), 1.0), 3)
