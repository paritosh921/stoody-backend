"""Vision-assisted answer-sheet mapping for difficult full-document OCR cases."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from services.ai_gateway_service import AIGatewayService, estimate_ocr_tokens, estimate_text_tokens
from utils.s3_storage import get_public_url, upload_file


class AnswerSheetVisionMapper:
    """Ask a vision model to resolve weak answer-block to question mappings."""

    def __init__(self, model: Optional[str] = None, max_pages: Optional[int] = None):
        self.model = model or os.getenv("ANSWER_MAPPING_VISION_MODEL", "gpt-5.4-mini")
        self.max_pages = max(1, int(max_pages or os.getenv("ANSWER_MAPPING_VISION_MAX_PAGES", "12")))
        self.auto_orientation_enabled = str(
            os.getenv("ANSWER_MAPPING_AUTO_ORIENTATION_ENABLED", "true")
        ).lower() not in {"0", "false", "no"}
        self.orientation_line_ratio = max(
            1.15,
            float(os.getenv("ANSWER_MAPPING_SIDEWAYS_LINE_RATIO", "1.45")),
        )
        self.orientation_recovery_max_pages = max(
            1,
            int(os.getenv("ANSWER_MAPPING_ORIENTATION_RECOVERY_MAX_PAGES", "6")),
        )

    async def extract_by_question(
        self,
        *,
        pdf_bytes: bytes,
        question_docs: List[Dict[str, Any]],
        answer_blocks: Optional[List[Dict[str, Any]]] = None,
        page_summaries: Optional[List[Dict[str, Any]]] = None,
        layout_report: Optional[Dict[str, Any]] = None,
        gateway_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract answer key + worked solution by anchoring on question_index."""
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            return {
                "used": False,
                "provider": "openai",
                "model": self.model,
                "mode": "question_anchored",
                "error": "OPENAI_API_KEY not configured",
                "mappings": [],
            }

        page_renders = self._render_relevant_pages(
            pdf_bytes,
            [],
            recover_unknown_orientation=not bool(answer_blocks),
        )
        if not page_renders:
            return {
                "used": False,
                "provider": "openai",
                "model": self.model,
                "mode": "question_anchored",
                "error": "No answer-sheet pages could be rendered",
                "mappings": [],
            }

        questions_by_index = {
            index: question
            for index, question in enumerate(question_docs or [], start=1)
        }
        prompt = self._build_question_anchored_prompt(
            question_docs=question_docs,
            page_summaries=page_summaries or [],
            layout_report=layout_report,
            render_manifest=self._render_manifest(page_renders),
        )

        async def _raw_call():
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=openai_key)
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for render in page_renders:
                content.append({"type": "text", "text": self._render_label(render)})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{render['b64']}",
                            "detail": "high",
                        },
                    }
                )
            return await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract teacher-uploaded answer keys and worked solutions from answer-sheet PDF images. "
                            "Use question_index as the primary anchor. Return strict JSON only."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=8192,
            )

        input_bytes = sum(int(render.get("bytes", 0) or 0) for render in page_renders)
        if gateway_context:
            gateway = AIGatewayService(
                gateway_context.get("db"),
                is_b2c=bool(gateway_context.get("is_b2c")),
            )
            response = await gateway.call(
                user_id=str(gateway_context.get("user_id") or "unknown"),
                tenant_id=gateway_context.get("tenant_id"),
                document_id=gateway_context.get("document_id"),
                region_id=gateway_context.get("region_id"),
                region_scope=gateway_context.get("region_scope") or "answer_question_anchored_vision",
                stage="answer_sheet_question_anchored_extraction",
                provider="openai",
                model=self.model,
                input_kind="answer_sheet_page_images",
                estimated_input_tokens=estimate_text_tokens(prompt)
                + estimate_ocr_tokens(image_bytes=input_bytes, page_count=len(page_renders)),
                estimated_output_tokens=4096,
                max_output_tokens=8192,
                input_units={"image_bytes": input_bytes, "page_count": len(page_renders)},
                call_fn=_raw_call,
            )
        else:
            response = await _raw_call()

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"solutions": [], "notes": ["vision_extractor_returned_invalid_json"]}

        raw_items = []
        if isinstance(parsed, dict):
            raw_items = parsed.get("solutions") or parsed.get("mappings") or []
        if not isinstance(raw_items, list):
            raw_items = []
        mappings = [
            self._normalise_question_anchored_mapping(item, questions_by_index)
            for item in raw_items
            if isinstance(item, dict)
        ]
        mappings = [mapping for mapping in mappings if mapping.get("question_id")]
        mappings = await self._attach_solution_image_crops(
            mappings=mappings,
            page_renders=page_renders,
            document_id=str((gateway_context or {}).get("document_id") or "answer-sheet"),
        )
        return {
            "used": True,
            "provider": "openai",
            "model": self.model,
            "mode": "question_anchored",
            "mappings": mappings,
            "notes": parsed.get("notes", []) if isinstance(parsed, dict) else [],
            "rendered_pages": self._rendered_page_indexes(page_renders),
            "render_orientations": self._render_manifest(page_renders),
            "orientation_recovery_used": any(
                int(render.get("rotation_degrees") or 0) != 0 for render in page_renders
            ),
        }

    async def map(
        self,
        *,
        pdf_bytes: bytes,
        question_docs: List[Dict[str, Any]],
        answer_blocks: List[Dict[str, Any]],
        candidate_mappings: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]] = None,
        reasons: Optional[List[str]] = None,
        gateway_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            return {
                "used": False,
                "provider": "openai",
                "model": self.model,
                "error": "OPENAI_API_KEY not configured",
                "mappings": [],
            }

        page_renders = self._render_relevant_pages(pdf_bytes, answer_blocks)
        if not page_renders:
            return {
                "used": False,
                "provider": "openai",
                "model": self.model,
                "error": "No answer-sheet pages could be rendered",
                "mappings": [],
            }

        prompt = self._build_prompt(
            question_docs=question_docs,
            answer_blocks=answer_blocks,
            candidate_mappings=candidate_mappings,
            layout_report=layout_report,
            reasons=reasons or [],
        )

        async def _raw_call():
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=openai_key)
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for render in page_renders:
                content.append({"type": "text", "text": self._render_label(render)})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{render['b64']}",
                            "detail": "high",
                        },
                    }
                )
            return await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You map uploaded answer-sheet worked solutions to already extracted "
                            "question IDs. Return strict JSON only. Do not invent missing answers."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
            )

        input_bytes = sum(int(render.get("bytes", 0) or 0) for render in page_renders)
        if gateway_context:
            gateway = AIGatewayService(
                gateway_context.get("db"),
                is_b2c=bool(gateway_context.get("is_b2c")),
            )
            response = await gateway.call(
                user_id=str(gateway_context.get("user_id") or "unknown"),
                tenant_id=gateway_context.get("tenant_id"),
                document_id=gateway_context.get("document_id"),
                region_id=gateway_context.get("region_id"),
                region_scope=gateway_context.get("region_scope") or "answer_mapping_vision",
                stage="answer_sheet_vision_mapping",
                provider="openai",
                model=self.model,
                input_kind="answer_sheet_page_images",
                estimated_input_tokens=estimate_text_tokens(prompt)
                + estimate_ocr_tokens(image_bytes=input_bytes, page_count=len(page_renders)),
                estimated_output_tokens=2048,
                max_output_tokens=4096,
                input_units={"image_bytes": input_bytes, "page_count": len(page_renders)},
                call_fn=_raw_call,
            )
        else:
            response = await _raw_call()

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"mappings": [], "notes": ["vision_mapper_returned_invalid_json"]}

        mappings = parsed.get("mappings") if isinstance(parsed, dict) else []
        if not isinstance(mappings, list):
            mappings = []
        return {
            "used": True,
            "provider": "openai",
            "model": self.model,
            "mappings": [self._normalise_mapping(mapping) for mapping in mappings if isinstance(mapping, dict)],
            "notes": parsed.get("notes", []) if isinstance(parsed, dict) else [],
            "rendered_pages": self._rendered_page_indexes(page_renders),
            "render_orientations": self._render_manifest(page_renders),
            "orientation_recovery_used": any(
                int(render.get("rotation_degrees") or 0) != 0 for render in page_renders
            ),
        }

    def _render_relevant_pages(
        self,
        pdf_bytes: bytes,
        answer_blocks: List[Dict[str, Any]],
        recover_unknown_orientation: bool = False,
    ) -> List[Dict[str, Any]]:
        try:
            import fitz
        except Exception:
            return []

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception:
            return []

        try:
            page_indexes = self._page_indexes(answer_blocks, len(doc))
            bounded_unknown_recovery = bool(
                recover_unknown_orientation
                and len(page_indexes) <= self.orientation_recovery_max_pages
            )
            renders: List[Dict[str, Any]] = []
            for page_index in page_indexes[: self.max_pages]:
                page = doc[page_index]
                mat = fitz.Matrix(180 / 72, 180 / 72)
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
                renders.extend(
                    self._orientation_aware_renders(
                        page_index=page_index,
                        image_bytes=image_bytes,
                        width=pix.width,
                        height=pix.height,
                        recover_unknown_orientation=bounded_unknown_recovery,
                    )
                )
            return renders
        finally:
            doc.close()

    def _orientation_aware_renders(
        self,
        *,
        page_index: int,
        image_bytes: bytes,
        width: int,
        height: int,
        recover_unknown_orientation: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return one normal render or two bounded sideways recovery candidates.

        Ruled notebook pages provide a strong local signal: in an upright page the
        long rules are horizontal, while a sideways camera/PDF upload turns them
        vertical. Direction cannot be established safely without reading the page,
        so suspicious pages are represented by clockwise 90 and 270 degree
        candidates in the *same* model request. This avoids another paid API call
        and lets the vision model choose the readable candidate.
        """
        sideways, evidence = self._detect_sideways_page(image_bytes)
        rotations = (
            [90, 270]
            if sideways
            else ([0, 90, 270] if recover_unknown_orientation else [0])
        )
        renders: List[Dict[str, Any]] = []
        for rotation in rotations:
            rotated_bytes, rotated_width, rotated_height = self._rotate_png(
                image_bytes=image_bytes,
                rotation_degrees=rotation,
                fallback_width=width,
                fallback_height=height,
            )
            renders.append(
                {
                    "page_index": page_index,
                    "rotation_degrees": rotation,
                    "orientation_recovery_candidate": sideways or recover_unknown_orientation,
                    "orientation_detection_uncertain": bool(
                        recover_unknown_orientation and not sideways
                    ),
                    "orientation_evidence": evidence,
                    "b64": base64.b64encode(rotated_bytes).decode("utf-8"),
                    "image_bytes": rotated_bytes,
                    "bytes": len(rotated_bytes),
                    "width": rotated_width,
                    "height": rotated_height,
                }
            )
        return renders

    def _detect_sideways_page(self, image_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
        if not self.auto_orientation_enabled:
            return False, {"method": "disabled"}
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
                # OpenCV returns either N x 1 x 4 or N x 4 depending on build.
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
                and vertical_support > horizontal_support * self.orientation_line_ratio
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

    def _rotate_png(
        self,
        *,
        image_bytes: bytes,
        rotation_degrees: int,
        fallback_width: int,
        fallback_height: int,
    ) -> Tuple[bytes, int, int]:
        rotation = self._normalise_rotation(rotation_degrees)
        if rotation == 0:
            return image_bytes, fallback_width, fallback_height
        try:
            from PIL import Image

            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            # PIL positive angles are counter-clockwise; the public contract is clockwise.
            rotated = image.rotate(-rotation, expand=True, fillcolor="white")
            output = BytesIO()
            rotated.save(output, format="PNG", optimize=True)
            return output.getvalue(), rotated.width, rotated.height
        except Exception:
            return image_bytes, fallback_width, fallback_height

    def _render_manifest(self, page_renders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "physical_page": int(render.get("page_index") or 0) + 1,
                "rotation_degrees_clockwise": int(render.get("rotation_degrees") or 0),
                "orientation_recovery_candidate": bool(
                    render.get("orientation_recovery_candidate")
                ),
            }
            for render in page_renders
        ]

    def _render_label(self, render: Dict[str, Any]) -> str:
        page_number = int(render.get("page_index") or 0) + 1
        rotation = int(render.get("rotation_degrees") or 0)
        if render.get("orientation_recovery_candidate"):
            return (
                f"Physical answer-sheet page {page_number}; orientation candidate "
                f"rotated {rotation} degrees clockwise. Another image may be the same "
                "physical page in the opposite orientation. Use only the readable candidate."
            )
        return f"Physical answer-sheet page {page_number}; rotation 0 degrees."

    def _rendered_page_indexes(self, page_renders: List[Dict[str, Any]]) -> List[int]:
        return sorted({int(render.get("page_index") or 0) for render in page_renders})

    def _page_indexes(self, answer_blocks: List[Dict[str, Any]], total_pages: int) -> List[int]:
        seen: set[int] = set()
        indexes: List[int] = []
        for block in answer_blocks or []:
            raw_page = block.get("page")
            try:
                page = int(raw_page)
            except (TypeError, ValueError):
                page = 0
            candidates = [page, page - 1] if page > 0 else [page]
            for candidate in candidates:
                if 0 <= candidate < total_pages and candidate not in seen:
                    seen.add(candidate)
                    indexes.append(candidate)
                    break
        if not indexes:
            indexes = list(range(min(total_pages, self.max_pages)))
        return indexes

    def _build_prompt(
        self,
        *,
        question_docs: List[Dict[str, Any]],
        answer_blocks: List[Dict[str, Any]],
        candidate_mappings: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]],
        reasons: List[str],
    ) -> str:
        questions_payload = []
        for index, question in enumerate(question_docs or [], start=1):
            questions_payload.append(
                {
                    "question_index": index,
                    "question_id": str(question.get("id") or question.get("question_id") or ""),
                    "text": str(question.get("text") or question.get("question_text") or "")[:900],
                    "correct_answer": question.get("correct_answer"),
                }
            )
        blocks_payload = []
        for index, block in enumerate(answer_blocks or [], start=1):
            blocks_payload.append(
                {
                    "answer_block_id": str(block.get("block_id") or block.get("id") or f"answer_block_{index}"),
                    "answer_index": index,
                    "number": block.get("number"),
                    "page": block.get("page"),
                    "text": str(block.get("text") or block.get("answer_text") or "")[:1400],
                    "confidence": block.get("confidence"),
                    "reasons": block.get("reasons") or [],
                }
            )
        candidates_payload = [
            {
                "question_id": mapping.get("question_id"),
                "answer_block_id": mapping.get("answer_block_id") or mapping.get("answer_region_id"),
                "strategy": mapping.get("mapping_strategy"),
                "confidence": mapping.get("confidence"),
                "manual_review_required": mapping.get("manual_review_required"),
                "reasons": mapping.get("mapping_reasons") or [],
            }
            for mapping in candidate_mappings or []
        ]
        layout_payload = {
            "page_count": (layout_report or {}).get("page_count"),
            "layout_risks": (
                (layout_report or {}).get("layout_risks", [])
                or (layout_report or {}).get("document_layout_risks", [])
            ),
            "answer_anchors": self._compact_answer_anchors(layout_report),
        }
        schema_hint = {
            "mappings": [
                {
                    "question_id": "saved question id",
                    "answer_block_id": "input answer block id",
                    "answer_item_id": "unique id for this extracted solution item",
                    "answer_number": "visible answer number or null",
                    "correct_answer": "A/B/C/D/E/F if visible, otherwise empty string",
                    "correct_answer_confidence": 0.0,
                    "final_answer_text": "visible final answer text if present",
                    "answer_text": "worked solution text only",
                    "source_rotation_degrees": "0, 90, or 270 from the readable orientation candidate",
                    "mapping_strategy": "gpt_vision_mapper",
                    "confidence": 0.0,
                    "manual_review_required": True,
                    "evidence": "short visual/text reason",
                    "notes": "optional warning",
                }
            ],
            "notes": [],
        }
        return (
            "Map answer-sheet worked solutions to the saved question IDs.\n"
            "Use the page images as the source of truth when OCR text or numbering is noisy.\n"
            "Rules:\n"
            "- Only map an answer to a question if there is visible evidence.\n"
            "- Keep answer_text to the worked solution/explanation, not copied question text.\n"
            "- Extract correct_answer only when the visible answer sheet clearly states an option label or final selected answer.\n"
            "- Leave correct_answer empty when the option label is inferred but not visible.\n"
            "- If one broad OCR block contains multiple worked solutions, return one mapping per question and give each one a distinct answer_item_id.\n"
            "- answer_item_id must be unique within this response. Use values like page_2_q_7 or answer_block_3_q_9.\n"
            "- When a physical page has orientation candidates, inspect both but use it only once; set source_rotation_degrees to the readable candidate.\n"
            "- If a mapping is uncertain, set manual_review_required=true and confidence below 0.75.\n"
            "- Do not invent missing answer numbers or missing solution text.\n\n"
            f"Reasons this vision check is needed:\n{json.dumps(reasons, ensure_ascii=False)}\n\n"
            f"Questions:\n{json.dumps(questions_payload, ensure_ascii=False)}\n\n"
            f"Extracted answer blocks:\n{json.dumps(blocks_payload, ensure_ascii=False)}\n\n"
            f"Deterministic candidate mappings:\n{json.dumps(candidates_payload, ensure_ascii=False)}\n\n"
            f"Layout summary:\n{json.dumps(layout_payload, ensure_ascii=False)}\n\n"
            f"Return JSON exactly in this shape:\n{json.dumps(schema_hint, ensure_ascii=False)}"
        )

    def _build_question_anchored_prompt(
        self,
        *,
        question_docs: List[Dict[str, Any]],
        page_summaries: List[Dict[str, Any]],
        layout_report: Optional[Dict[str, Any]],
        render_manifest: List[Dict[str, Any]],
    ) -> str:
        questions_payload = []
        for index, question in enumerate(question_docs or [], start=1):
            options = question.get("enhanced_options") or question.get("options") or []
            requires_option_label = self._question_requires_option_label(question)
            option_payload = []
            if options and isinstance(options[0], dict):
                for opt in options:
                    option_payload.append(
                        {
                            "label": opt.get("label"),
                            "text": str(opt.get("content") or opt.get("text") or "")[:500],
                        }
                    )
            else:
                for opt_index, opt in enumerate(options or []):
                    option_payload.append(
                        {
                            "label": chr(65 + opt_index),
                            "text": str(opt)[:500],
                        }
                    )
            questions_payload.append(
                {
                    "question_index": index,
                    "question_id": str(question.get("id") or question.get("question_id") or ""),
                    "text": str(question.get("text") or question.get("question_text") or "")[:1000],
                    "question_type": str(question.get("question_type") or "subjective"),
                    "answer_format": "option_label" if requires_option_label else "worked_solution",
                    "options": option_payload,
                }
            )
        ocr_payload = [
            {
                "page": int(page.get("index", page.get("page", 0)) or 0) + 1,
                "markdown": str(page.get("markdown") or "")[:2500],
            }
            for page in (page_summaries or [])[: self.max_pages]
        ]
        schema_hint = {
            "solutions": [
                {
                    "question_index": 1,
                    "question_id": "copy from input",
                    "visible_answer_number": "visible answer number on sheet, e.g. 1",
                    "correct_answer": "A/B/C/D/E/F for objective questions; empty for subjective questions",
                    "correct_answer_confidence": 0.0,
                    "answer_text": "teacher's worked solution/explanation for this question",
                    "final_answer_text": "visible final answer text if present",
                    "solution_image_notes": "describe any diagram/image/formula evidence in the solution",
                    "source_rotation_degrees": "0, 90, or 270 from the readable orientation candidate",
                    "solution_bbox": {
                        "page": 1,
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.0,
                        "height": 0.0,
                    },
                    "page_numbers": [1],
                    "confidence": 0.0,
                    "manual_review_required": False,
                    "evidence": "short reason from visible answer sheet",
                    "notes": "",
                }
            ],
            "missing_question_indexes": [],
            "notes": [],
        }
        layout_payload = {
            "page_count": (layout_report or {}).get("page_count"),
            "layout_risks": (
                (layout_report or {}).get("layout_risks", [])
                or (layout_report or {}).get("document_layout_risks", [])
            ),
        }
        return (
            "Extract the answer key and worked solution from the uploaded answer-sheet PDF images.\n"
            "The question paper has already been OCR'd. Use the provided question_index list as the source of truth.\n\n"
            "Rules:\n"
            "- For every question_index, find the matching answer/solution in the answer-sheet images.\n"
            "- Use question_index as the primary identifier. Do not reorder questions.\n"
            "- Images labelled as orientation candidates are alternate views of the same physical page, not additional pages. Read both candidates, use the upright one, and set source_rotation_degrees accordingly.\n"
            "- For answer_format=option_label, if the answer sheet writes Ans 1/2/3/4/5/6, convert it to A/B/C/D/E/F.\n"
            "- For answer_format=option_label, correct_answer must be the option label, not the option text.\n"
            "- For answer_format=worked_solution, leave correct_answer empty. A missing A-F label is expected and must not by itself require review.\n"
            "- answer_text must be the teacher/admin uploaded worked solution, not a new generated solution.\n"
            "- Preserve equations and final values in answer_text.\n"
            "- If the solution contains a diagram/image/formula that cannot be fully represented as text, describe it in solution_image_notes and mention the page.\n"
            "- If a visual crop is needed to preserve a diagram/table/formula, set solution_bbox to normalized page coordinates x/y/width/height around only that solution. Otherwise set solution_bbox to null.\n"
            "- Never use a full-page bbox unless the solution genuinely spans the full page.\n"
            "- If you cannot visibly locate a question's answer, omit that solution and include its index in missing_question_indexes.\n"
            "- Set manual_review_required=true for ambiguous, partial, or inferred answers.\n"
            "- Do not invent missing solutions or fill from your own knowledge.\n\n"
            f"Questions:\n{json.dumps(questions_payload, ensure_ascii=False)}\n\n"
            f"OCR text from answer sheet for cross-checking:\n{json.dumps(ocr_payload, ensure_ascii=False)}\n\n"
            f"Layout summary:\n{json.dumps(layout_payload, ensure_ascii=False)}\n\n"
            f"Image orientation manifest:\n{json.dumps(render_manifest, ensure_ascii=False)}\n\n"
            f"Return JSON exactly in this shape:\n{json.dumps(schema_hint, ensure_ascii=False)}"
        )

    def _compact_answer_anchors(self, layout_report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        for page in (layout_report or {}).get("pages", []) or []:
            page_number = page.get("page") or page.get("index")
            for anchor in page.get("answer_anchors", []) or []:
                anchors.append(
                    {
                        "page": page_number,
                        "number": anchor.get("number"),
                        "text": str(anchor.get("text") or "")[:120],
                    }
                )
        return anchors[:80]

    def _normalise_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        confidence = mapping.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "question_id": str(mapping.get("question_id") or "").strip(),
            "answer_block_id": str(mapping.get("answer_block_id") or mapping.get("answer_region_id") or "").strip(),
            "answer_item_id": str(mapping.get("answer_item_id") or "").strip(),
            "answer_number": mapping.get("answer_number"),
            "correct_answer": str(mapping.get("correct_answer") or "").strip().upper(),
            "correct_answer_confidence": self._confidence(mapping.get("correct_answer_confidence")),
            "final_answer_text": str(mapping.get("final_answer_text") or "").strip(),
            "answer_text": str(mapping.get("answer_text") or "").strip(),
            "source_rotation_degrees": self._normalise_rotation(
                mapping.get("source_rotation_degrees")
            ),
            "mapping_strategy": "gpt_vision_mapper",
            "confidence": max(0.0, min(1.0, confidence)),
            "manual_review_required": bool(mapping.get("manual_review_required")),
            "evidence": str(mapping.get("evidence") or "").strip(),
            "notes": str(mapping.get("notes") or "").strip(),
        }

    def _normalise_question_anchored_mapping(
        self,
        mapping: Dict[str, Any],
        questions_by_index: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        question_index = self._int(mapping.get("question_index"), 0)
        question = questions_by_index.get(question_index, {})
        question_id = str(question.get("id") or question.get("question_id") or "").strip()
        model_question_id = str(mapping.get("question_id") or "").strip()
        if not question_id and model_question_id:
            question_id = model_question_id
        confidence = self._confidence(mapping.get("confidence"))
        correct_answer_confidence = self._confidence(mapping.get("correct_answer_confidence"))
        correct_answer = self._normalise_answer_label(mapping.get("correct_answer"))
        answer_text = str(mapping.get("answer_text") or "").strip()
        requires_option_label = self._question_requires_option_label(question)
        image_notes = str(mapping.get("solution_image_notes") or "").strip()
        if image_notes and image_notes not in answer_text:
            answer_text = f"{answer_text}\n\nSolution visual notes: {image_notes}".strip()
        manual_review = bool(mapping.get("manual_review_required")) and (
            confidence < 0.9
            or (requires_option_label and correct_answer_confidence < 0.9)
        )
        if not answer_text or confidence < 0.75:
            manual_review = True
        if requires_option_label and (
            not correct_answer or correct_answer_confidence < 0.75
        ):
            manual_review = True
        return {
            "question_index": question_index,
            "question_id": question_id,
            "answer_block_id": f"question_anchored_q_{question_index}",
            "answer_item_id": f"question_anchored_q_{question_index}",
            "answer_number": mapping.get("visible_answer_number") or mapping.get("answer_number") or question_index,
            "correct_answer": correct_answer,
            "correct_answer_confidence": correct_answer_confidence,
            "question_type": str(question.get("question_type") or "subjective"),
            "requires_option_label": requires_option_label,
            "final_answer_text": str(mapping.get("final_answer_text") or "").strip(),
            "answer_text": answer_text,
            "mapping_strategy": "gpt_question_anchored",
            "confidence": confidence,
            "manual_review_required": manual_review,
            "evidence": str(mapping.get("evidence") or "").strip(),
            "notes": str(mapping.get("notes") or "").strip(),
            "page_numbers": mapping.get("page_numbers") if isinstance(mapping.get("page_numbers"), list) else [],
            "solution_image_notes": image_notes,
            "source_rotation_degrees": self._normalise_rotation(
                mapping.get("source_rotation_degrees")
            ),
            "solution_bbox": self._normalise_bbox(mapping.get("solution_bbox") or mapping.get("solution_image_bbox")),
        }

    def _question_requires_option_label(self, question: Dict[str, Any]) -> bool:
        options = question.get("enhanced_options") or question.get("options") or []
        if isinstance(options, list) and len(options) > 0:
            return True
        question_type = str(question.get("question_type") or "").strip().lower()
        return question_type in {
            "mcq",
            "multiple_choice",
            "multiple-choice",
            "objective",
            "single_choice",
            "single-choice",
        }

    def _confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _normalise_answer_label(self, value: Any) -> str:
        label = str(value or "").strip().upper()
        if label in {"A", "B", "C", "D", "E", "F"}:
            return label
        if label in {"1", "2", "3", "4", "5", "6"}:
            return chr(64 + int(label))
        return ""

    def _int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _normalise_rotation(self, value: Any) -> int:
        try:
            rotation = int(value or 0) % 360
        except (TypeError, ValueError):
            return 0
        return rotation if rotation in {0, 90, 180, 270} else 0

    def _normalise_bbox(self, value: Any) -> Optional[Dict[str, float]]:
        if not value:
            return None
        if isinstance(value, list) and len(value) >= 4:
            value = {
                "x": value[0],
                "y": value[1],
                "width": value[2],
                "height": value[3],
            }
        if not isinstance(value, dict):
            return None
        try:
            x = float(value.get("x"))
            y = float(value.get("y"))
            width = float(value.get("width"))
            height = float(value.get("height"))
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        bbox: Dict[str, float] = {
            "x": max(0.0, min(1.0, x)),
            "y": max(0.0, min(1.0, y)),
            "width": max(0.0, min(1.0, width)),
            "height": max(0.0, min(1.0, height)),
        }
        page = self._int(value.get("page"), 0)
        if page > 0:
            bbox["page"] = float(page)
        return bbox

    async def _attach_solution_image_crops(
        self,
        *,
        mappings: List[Dict[str, Any]],
        page_renders: List[Dict[str, Any]],
        document_id: str,
    ) -> List[Dict[str, Any]]:
        if not mappings or not page_renders:
            return mappings
        renders_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for render in page_renders:
            page_number = int(render.get("page_index") or 0) + 1
            renders_by_page.setdefault(page_number, []).append(render)
        updated: List[Dict[str, Any]] = []
        for mapping in mappings:
            bbox = mapping.get("solution_bbox")
            if not bbox:
                updated.append(mapping)
                continue
            page_number = int(bbox.get("page") or 0)
            if page_number <= 0:
                page_numbers = mapping.get("page_numbers") if isinstance(mapping.get("page_numbers"), list) else []
                page_number = self._int(page_numbers[0], 0) if page_numbers else 0
            page_candidates = renders_by_page.get(page_number, [])
            requested_rotation = self._normalise_rotation(
                mapping.get("source_rotation_degrees")
            )
            render = next(
                (
                    candidate
                    for candidate in page_candidates
                    if self._normalise_rotation(candidate.get("rotation_degrees"))
                    == requested_rotation
                ),
                page_candidates[0] if page_candidates else None,
            )
            if not render:
                updated.append(mapping)
                continue
            crop = self._crop_rendered_page(render, bbox)
            if not crop:
                updated.append(mapping)
                continue
            question_index = self._int(mapping.get("question_index"), 0)
            filename = f"q{question_index or 'unknown'}-p{page_number}.png"
            path = (
                "uploads/answer_solution_images/"
                f"{self._safe_storage_segment(document_id)}/"
                f"{self._safe_storage_segment(filename)}"
            )
            try:
                success, storage_path = await upload_file(
                    crop["bytes"],
                    path,
                    content_type="image/png",
                    metadata={
                        "document_id": document_id[:120],
                        "question_id": str(mapping.get("question_id") or "")[:120],
                        "source": "answer_sheet_solution_crop",
                    },
                )
            except Exception:
                success, storage_path = False, ""
            if not success or not storage_path:
                updated.append(mapping)
                continue
            image_record = {
                "storage_path": storage_path,
                "url": get_public_url(storage_path),
                "content_type": "image/png",
                "page_number": page_number,
                "bbox": crop["bbox"],
                "orientation_applied": self._normalise_rotation(
                    render.get("rotation_degrees")
                ),
                "source": "answer_sheet_solution_crop",
            }
            updated.append(
                {
                    **mapping,
                    "solution_images": [image_record],
                }
            )
        return updated

    def _crop_rendered_page(self, render: Dict[str, Any], bbox: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            from PIL import Image
        except Exception:
            return None
        image_bytes = render.get("image_bytes")
        if not image_bytes:
            return None
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None
        width, height = image.size
        x = float(bbox.get("x") or 0)
        y = float(bbox.get("y") or 0)
        box_width = float(bbox.get("width") or 0)
        box_height = float(bbox.get("height") or 0)
        if box_width <= 0 or box_height <= 0:
            return None
        if box_width >= 0.98 and box_height >= 0.98:
            return None
        margin = 0.015
        left = max(0, int((x - margin) * width))
        top = max(0, int((y - margin) * height))
        right = min(width, int((x + box_width + margin) * width))
        bottom = min(height, int((y + box_height + margin) * height))
        if right - left < 24 or bottom - top < 24:
            return None
        cropped = image.crop((left, top, right, bottom))
        output = BytesIO()
        cropped.save(output, format="PNG")
        return {
            "bytes": output.getvalue(),
            "bbox": {
                "x": round(left / width, 4),
                "y": round(top / height, 4),
                "width": round((right - left) / width, 4),
                "height": round((bottom - top) / height, 4),
            },
        }

    def _safe_storage_segment(self, value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-")
        return safe[:160] or "answer-sheet"
