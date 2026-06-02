"""Full-document layout analysis with optional LiteParse and PyMuPDF fallback."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


QUESTION_ANCHOR_RE = re.compile(r"^\s*(?:Q(?:uestion)?\.?\s*)?(\d{1,3})[\.\)]\s+", re.IGNORECASE)
ANSWER_ANCHOR_RE = re.compile(
    r"^\s*(?:(?:ans(?:wer)?|sol(?:ution)?|exp(?:lanation)?)\s*[:\-]?\s*)?(\d{1,3})[\.\)]\s+",
    re.IGNORECASE,
)
OPTION_LABEL_RE = re.compile(r"^\s*[\(\[]?([a-dA-D])[\.\)]\s*(.*)$")
ANSWER_CUE_RE = re.compile(r"\b(ans(?:wer)?|sol(?:ution)?|exp(?:lanation)?|worked)\b", re.IGNORECASE)


class DocumentLayoutProvider:
    """Analyze full PDF layout without mutating OCR/document state."""

    async def analyze(
        self,
        *,
        pdf_bytes: bytes,
        document_id: str,
        mode: str,
        timeout_seconds: int = 20,
    ) -> Dict[str, Any]:
        if self._liteparse_enabled():
            liteparse_report = await self._try_liteparse(
                pdf_bytes=pdf_bytes,
                document_id=document_id,
                mode=mode,
                timeout_seconds=timeout_seconds,
            )
            if liteparse_report:
                return liteparse_report
        return self._pymupdf_report(pdf_bytes=pdf_bytes, document_id=document_id, mode=mode)

    def _liteparse_enabled(self) -> bool:
        return os.getenv("LITEPARSE_LAYOUT_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}

    async def _try_liteparse(
        self,
        *,
        pdf_bytes: bytes,
        document_id: str,
        mode: str,
        timeout_seconds: int,
    ) -> Optional[Dict[str, Any]]:
        liteparse = shutil.which("liteparse")
        if not liteparse:
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="stoody-layout-") as tmp:
                pdf_path = Path(tmp) / f"{document_id or 'document'}.pdf"
                pdf_path.write_bytes(pdf_bytes)
                proc = await asyncio.create_subprocess_exec(
                    liteparse,
                    "parse",
                    str(pdf_path),
                    "--output-format",
                    "json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
                if proc.returncode != 0 or not stdout:
                    return None
                parsed = json.loads(stdout.decode("utf-8", errors="ignore"))
                return self._coerce_liteparse_report(parsed, document_id=document_id, mode=mode)
        except Exception:
            return None

    def _coerce_liteparse_report(self, parsed: Dict[str, Any], *, document_id: str, mode: str) -> Dict[str, Any]:
        pages_in = parsed.get("pages") or []
        pages: List[Dict[str, Any]] = []
        document_risks: List[str] = []
        for idx, page in enumerate(pages_in):
            page_number = int(page.get("page") or page.get("page_number") or idx + 1)
            text = page.get("text") or page.get("markdown") or ""
            page_report = self._text_report_for_page(
                text=text,
                page_number=page_number,
                mode=mode,
                text_blocks=[],
                width=page.get("width"),
                height=page.get("height"),
            )
            pages.append(page_report)
            document_risks.extend(page_report.get("layout_risks", []))
        return {
            "provider": "liteparse",
            "fallback_provider": None,
            "document_id": document_id,
            "mode": mode,
            "page_count": len(pages),
            "has_text_layer": any(page.get("has_text_layer") for page in pages),
            "document_layout_risks": sorted(set(document_risks)),
            "pages": pages,
            "recommended_strategy": self._recommended_strategy(document_risks),
        }

    def _pymupdf_report(self, *, pdf_bytes: bytes, document_id: str, mode: str) -> Dict[str, Any]:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: List[Dict[str, Any]] = []
        document_risks: List[str] = []
        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text_blocks = self._page_text_blocks(page)
                text = "\n".join(block["text"] for block in text_blocks if block.get("text"))
                image_regions = [
                    {"xref": image[0], "page": page_index + 1}
                    for image in (page.get_images(full=True) or [])
                ]
                page_report = self._text_report_for_page(
                    text=text,
                    page_number=page_index + 1,
                    mode=mode,
                    text_blocks=text_blocks,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    image_regions=image_regions,
                )
                pages.append(page_report)
                document_risks.extend(page_report.get("layout_risks", []))
        finally:
            doc.close()

        return {
            "provider": "pymupdf",
            "fallback_provider": None,
            "document_id": document_id,
            "mode": mode,
            "page_count": len(pages),
            "has_text_layer": any(page.get("has_text_layer") for page in pages),
            "document_layout_risks": sorted(set(document_risks)),
            "pages": pages,
            "recommended_strategy": self._recommended_strategy(document_risks),
        }

    def _page_text_blocks(self, page: Any) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for block in page.get_text("blocks") or []:
            if len(block) < 5:
                continue
            text = str(block[4] or "").strip()
            if not text:
                continue
            blocks.append(
                {
                    "x": float(block[0]),
                    "y": float(block[1]),
                    "width": float(block[2]) - float(block[0]),
                    "height": float(block[3]) - float(block[1]),
                    "text": text,
                }
            )
        return sorted(blocks, key=lambda item: (item["y"], item["x"]))

    def _text_report_for_page(
        self,
        *,
        text: str,
        page_number: int,
        mode: str,
        text_blocks: List[Dict[str, Any]],
        width: Optional[float],
        height: Optional[float],
        image_regions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        lines = self._lines_from_blocks(text_blocks) if text_blocks else self._lines_from_text(text)
        has_text_layer = bool(lines)
        question_anchors: List[Dict[str, Any]] = []
        answer_anchors: List[Dict[str, Any]] = []
        option_risks: List[Dict[str, Any]] = []
        layout_risks: List[str] = []
        option_labels: List[str] = []
        label_only: List[str] = []
        unlabelled_before_label = 0

        for idx, line in enumerate(lines):
            line_text = str(line.get("text") or "").strip()
            q_match = QUESTION_ANCHOR_RE.match(line_text)
            if q_match:
                question_anchors.append({"number": q_match.group(1), "x": line.get("x"), "y": line.get("y")})
            a_match = ANSWER_ANCHOR_RE.match(line_text)
            answer_cue = self._answer_cue(line_text)
            if a_match and answer_cue:
                answer_anchors.append(
                    {
                        "number": a_match.group(1),
                        "x": line.get("x"),
                        "y": line.get("y"),
                        "cue": answer_cue,
                    }
                )
            option_match = OPTION_LABEL_RE.match(line_text)
            if option_match:
                label = option_match.group(1).lower()
                option_labels.append(label)
                if not option_match.group(2).strip():
                    label_only.append(label)
                    prev = lines[idx - 1] if idx > 0 else {}
                    if prev and not OPTION_LABEL_RE.match(str(prev.get("text") or "")):
                        unlabelled_before_label += 1

        unique_labels = sorted(set(option_labels))
        if len(unique_labels) >= 3 and label_only and unlabelled_before_label:
            layout_risks.append("staggered_options_possible")
            option_risks.append(
                {
                    "question_number": question_anchors[-1]["number"] if question_anchors else None,
                    "risk": "staggered_options_possible",
                    "labels_found": unique_labels,
                    "label_only_lines": label_only,
                }
            )
        if image_regions:
            layout_risks.append("formula_or_image_dependency")

        density = "none"
        if has_text_layer:
            chars = len(text or "")
            density = "low" if chars < 200 else ("high" if chars > 3000 else "normal")

        return {
            "page": page_number,
            "has_text_layer": has_text_layer,
            "text_density": density,
            "width": width,
            "height": height,
            "question_anchors": question_anchors[:100],
            "answer_anchors": answer_anchors[:100],
            "option_layout_risks": option_risks,
            "image_or_formula_regions": image_regions or [],
            "layout_risks": sorted(set(layout_risks)),
            "text_blocks": text_blocks[:120],
        }

    def _lines_from_blocks(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        for block in text_blocks:
            block_lines = str(block.get("text") or "").splitlines()
            for offset, text in enumerate(block_lines):
                stripped = text.strip()
                if not stripped:
                    continue
                lines.append(
                    {
                        "text": stripped,
                        "x": block.get("x"),
                        "y": float(block.get("y") or 0) + (offset * 12),
                    }
                )
        return lines

    def _lines_from_text(self, text: str) -> List[Dict[str, Any]]:
        return [
            {"text": line.strip(), "x": None, "y": idx}
            for idx, line in enumerate(str(text or "").splitlines())
            if line.strip()
        ]

    def _answer_cue(self, text: str) -> Optional[str]:
        match = ANSWER_CUE_RE.search(text or "")
        return match.group(1).lower() if match else None

    def _recommended_strategy(self, risks: List[str]) -> str:
        unique = set(risks or [])
        if "staggered_options_possible" in unique:
            return "full_ocr_with_layout_validation"
        if "formula_or_image_dependency" in unique:
            return "full_ocr_with_image_review"
        return "full_ocr"


def compact_layout_context(layout_report: Optional[Dict[str, Any]], *, max_pages: int = 20) -> Dict[str, Any]:
    """Return a parser-safe subset of a full layout report."""
    if not layout_report:
        return {}
    pages = []
    for page in (layout_report.get("pages") or [])[:max_pages]:
        pages.append(
            {
                "page": page.get("page"),
                "has_text_layer": page.get("has_text_layer"),
                "text_density": page.get("text_density"),
                "question_anchors": page.get("question_anchors", [])[:50],
                "answer_anchors": page.get("answer_anchors", [])[:50],
                "option_layout_risks": page.get("option_layout_risks", [])[:20],
                "layout_risks": page.get("layout_risks", []),
            }
        )
    return {
        "provider": layout_report.get("provider"),
        "mode": layout_report.get("mode"),
        "page_count": layout_report.get("page_count"),
        "has_text_layer": layout_report.get("has_text_layer"),
        "document_layout_risks": layout_report.get("document_layout_risks", []),
        "recommended_strategy": layout_report.get("recommended_strategy"),
        "pages": pages,
    }
