"""Manual-region crop helper used before OCR provider calls."""

from __future__ import annotations

import base64
from typing import Any, Dict, List


class RegionCropService:
    """Crop a saved manual region into provider-ready PDF and local metadata."""

    def crop(
        self,
        *,
        pdf_content: bytes,
        page_number: int,
        bbox: Dict[str, float],
        region_id: str,
        region_scope: str,
    ) -> Dict[str, Any]:
        import fitz

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        try:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"Invalid page number {page_number}")

            page = doc[page_number - 1]
            page_rect = page.rect
            x0 = page_rect.width * (float(bbox["x"]) / 100)
            y0 = page_rect.height * (float(bbox["y"]) / 100)
            x1 = x0 + page_rect.width * (float(bbox["width"]) / 100)
            y1 = y0 + page_rect.height * (float(bbox["height"]) / 100)
            clip_rect = fitz.Rect(x0, y0, x1, y1)

            region_doc = fitz.open()
            try:
                region_width = x1 - x0
                region_height = y1 - y0
                new_page = region_doc.new_page(width=region_width, height=region_height)
                new_page.show_pdf_page(
                    fitz.Rect(0, 0, region_width, region_height),
                    doc,
                    page_number - 1,
                    clip=clip_rect,
                )
                region_pdf_bytes = region_doc.tobytes()
            finally:
                region_doc.close()

            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            region_png_base64 = base64.b64encode(pix.tobytes("png")).decode("ascii")

            return {
                "region_id": region_id,
                "region_scope": region_scope,
                "region_pdf_bytes": region_pdf_bytes,
                "region_png_base64": region_png_base64,
                "text_items": self._extract_text_items(page, clip_rect),
                "embedded_images": self._extract_intersecting_images(page, page_rect, clip_rect, page_number, region_id),
                "crop_metadata": {
                    "page": page_number,
                    "x": float(bbox["x"]),
                    "y": float(bbox["y"]),
                    "width": float(bbox["width"]),
                    "height": float(bbox["height"]),
                    "page_width": float(page_rect.width),
                    "page_height": float(page_rect.height),
                    "region_width": float(region_width),
                    "region_height": float(region_height),
                },
            }
        finally:
            doc.close()

    def _extract_text_items(self, page: Any, clip_rect: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        try:
            text_dict = page.get_text("dict", clip=clip_rect)
        except Exception:
            return items

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = str(span.get("text", "") or "").strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                    items.append(
                        {
                            "text": text,
                            "x": float(x0) - float(clip_rect.x0),
                            "y": float(y0) - float(clip_rect.y0),
                            "width": float(x1) - float(x0),
                            "height": float(y1) - float(y0),
                        }
                    )
        return items

    def _extract_intersecting_images(
        self,
        page: Any,
        page_rect: Any,
        clip_rect: Any,
        page_number: int,
        region_id: str,
    ) -> List[Dict[str, Any]]:
        import fitz

        embedded_images: List[Dict[str, Any]] = []
        try:
            image_infos = page.get_image_info(xrefs=True)
        except Exception:
            return embedded_images

        seen_regions = set()
        page_area = max(float(page_rect.get_area()), 1.0)
        image_matrix = fitz.Matrix(3.0, 3.0)
        for img_idx, info in enumerate(image_infos):
            bbox_value = info.get("bbox")
            if not bbox_value or len(bbox_value) != 4:
                continue
            try:
                img_rect = fitz.Rect(bbox_value)
                intersection = img_rect & clip_rect
            except Exception:
                continue
            if intersection.is_empty or intersection.get_area() <= 0:
                continue
            if intersection.width < 8.0 or intersection.height < 8.0:
                continue
            img_area = max(float(img_rect.get_area()), 1.0)
            if img_area / page_area > 0.75 or float(intersection.get_area()) / img_area < 0.10:
                continue
            region_key = tuple(round(float(v), 1) for v in (intersection.x0, intersection.y0, intersection.x1, intersection.y1))
            if region_key in seen_regions:
                continue
            seen_regions.add(region_key)
            try:
                pix = page.get_pixmap(matrix=image_matrix, clip=intersection)
                embedded_images.append(
                    {
                        "id": f"page-{page_number}-region-{region_id}-img-{img_idx}",
                        "base64": base64.b64encode(pix.tobytes("png")).decode("ascii"),
                        "top_left_x": int(round(intersection.x0 - clip_rect.x0)),
                        "top_left_y": int(round(intersection.y0 - clip_rect.y0)),
                        "bottom_right_x": int(round(intersection.x1 - clip_rect.x0)),
                        "bottom_right_y": int(round(intersection.y1 - clip_rect.y0)),
                        "source": "pymupdf_region_intersection",
                    }
                )
            except Exception:
                continue
        return embedded_images
