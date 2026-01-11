"""
Question extraction logic for OCR Markdown output.
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from api.v1.pdf_schemas import ExtractedQuestion

logger = logging.getLogger(__name__)

# Match inline/display LaTeX blocks
_MATH_BLOCK_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([^)]*?\\\))")

# Characters typically present in a plain rendered math echo (no natural language)
_MATHY_TEXT_RE = re.compile(
    r"^[\s0-9A-Za-z_\^\-\+=*/()\[\]{}|.,:\xd7\xb7%\xb0<>\xb2\xb3\u2074\u2075\u2076\u2077\u2078\u2079\u2070\u207b\u207a\xb1\u221e\u2264\u2265\u2260\u2248\u221a\u2044~]*$"
)


def _dedupe_latex_echo(text: str) -> str:
    """If a string contains LaTeX and an immediate plain-text echo of the same formula, prefer LaTeX."""
    if not text:
        return text

    parts = _MATH_BLOCK_RE.split(text)
    if len(parts) == 1:
        return text

    latex_blocks = [p for p in parts if _MATH_BLOCK_RE.match(p)]
    non_latex = "".join(p for p in parts if not _MATH_BLOCK_RE.match(p)).strip()

    if latex_blocks and (not non_latex or _MATHY_TEXT_RE.match(non_latex)):
        return " ".join(latex_blocks).strip()

    if latex_blocks and any(
        char in non_latex
        for char in '\u2070\xb9\xb2\xb3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a\xb1\xd7\xf7\u221e\u2264\u2265\u2260\u2248\u221a\u2044'
    ):
        return " ".join(latex_blocks).strip()

    return text


def _clean_option_text(text: str) -> str:
    # Normalise whitespace
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    # Remove duplicated plain echo when LaTeX is present
    return _dedupe_latex_echo(cleaned)


def extract_questions_from_ocr(
    ocr_result: Dict[str, Any],
    subject: str,
    difficulty: str
) -> List[ExtractedQuestion]:
    """Extract questions from Mistral OCR result."""
    questions: List[ExtractedQuestion] = []

    pages = ocr_result.get("pages", [])

    for page in pages:
        markdown_content = page.get("markdown", "")

        logger.info(
            "Page %s markdown sample (first 1500 chars):\n%s",
            page.get("index", 0),
            markdown_content[:1500]
        )

        potential_questions = len(
            re.findall(r'(?:^|\n)(?:#{1,3}\s+)?Q\.?\s*\d+|(?:^|\n)\d+[\.\)]\s+', markdown_content, re.MULTILINE)
        )
        logger.info("Potential questions detected in markdown: %s", potential_questions)

        lines = markdown_content.split("\n")
        current_question = None
        current_question_text = ""
        current_options = []
        current_image_refs = []
        current_question_images = []
        current_option_images = []
        accumulating_option_idx: Optional[int] = None
        previous_line = ""

        option_label_pattern = re.compile(r'^\s*\(([A-Da-d]|[ivxIVX]+)\)\s*$')

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                previous_line = line_stripped
                continue

            if line_num < 10:
                logger.debug("Line %s: %s", line_num, line_stripped[:100])

            image_refs_raw = re.findall(r'!\[([^\]]*)\](?:\(([^)]+)\))?', line)
            image_refs = []
            for alt_text, url_text in image_refs_raw:
                img_ref = alt_text.strip() if alt_text.strip() else url_text.strip()
                if img_ref:
                    image_refs.append(img_ref)

            if image_refs:
                logger.info("📸 Line %s: Extracted %s image refs: %s", line_num, len(image_refs), image_refs)

            if image_refs and not current_question:
                pass
            elif image_refs and current_question:
                is_option_image = option_label_pattern.match(previous_line)

                for img_ref in image_refs:
                    if is_option_image:
                        if img_ref not in current_option_images:
                            current_option_images.append(img_ref)
                            logger.info("✓ Detected option image: %s (preceded by label: %s)", img_ref, previous_line.strip())
                        if img_ref not in current_image_refs:
                            current_image_refs.append(img_ref)
                    else:
                        if img_ref not in current_question_images:
                            current_question_images.append(img_ref)
                            logger.info("✓ Detected question figure: %s (no option label detected)", img_ref)
                        if img_ref not in current_image_refs:
                            current_image_refs.append(img_ref)

            previous_line = line_stripped

            line_without_heading = re.sub(r'^#+\s*', '', line_stripped)
            is_question = (
                line_stripped.endswith("?") or
                line_stripped.startswith(("Question", "Problem", "Q.", "Q ")) or
                re.match(r'^\d+[\.\)]\s+', line_stripped) or
                re.match(r'^Q\.?\s*\d+', line_stripped) or
                re.match(r'^Q\.?\s*\d+', line_without_heading) or
                (line_stripped and line_stripped[0].isdigit() and ('.' in line_stripped or ')' in line_stripped))
            )

            if is_question:
                logger.info("Detected new question starting: %s...", line_stripped[:80])
                if current_question:
                    total_images = len(current_image_refs)
                    num_option_images = len(current_option_images)
                    num_question_figures = len(current_question_images)
                    has_text_options = len(current_options) > 0 and any(opt.strip() for opt in current_options)

                    final_question_images = current_question_images.copy()
                    final_options = current_options
                    is_image_based_mcq = False

                    if num_option_images > 0:
                        valid_text_options = [opt for opt in current_options if opt.strip()]

                        if not valid_text_options:
                            logger.info(
                                "📊 Image-based MCQ: %s option images, %s question figures",
                                num_option_images,
                                num_question_figures
                            )
                            final_options = []
                            is_image_based_mcq = True
                        else:
                            logger.warning(
                                "⚠️ Mixed question: %s option images + %s text options",
                                num_option_images,
                                len(valid_text_options)
                            )
                            final_options = valid_text_options
                            is_image_based_mcq = False
                    elif total_images > 0 and not has_text_options:
                        if total_images >= 3:
                            logger.info(
                                "✅ Fallback RULE 1: %s images, no labels, no text → Treating as option images",
                                total_images
                            )
                            final_question_images = []
                            final_options = []
                            is_image_based_mcq = True
                        else:
                            logger.info("📊 %s question figures (no option labels detected)", total_images)
                    else:
                        logger.info(
                            "📝 Text-based question: %s text options, %s question figures",
                            len(current_options),
                            num_question_figures
                        )

                    logger.info(
                        "Extracted question: %s options, %s question figures, %s total images",
                        len(final_options),
                        len(final_question_images),
                        total_images
                    )
                    questions.append(ExtractedQuestion(
                        id=str(uuid.uuid4()),
                        text=current_question_text,
                        options=final_options,
                        metadata={
                            "subject": subject,
                            "difficulty": difficulty,
                            "page": page.get("index", 0),
                            "image_refs": current_image_refs,
                            "question_image_refs": final_question_images,
                            "is_image_based_mcq": is_image_based_mcq
                        }
                    ))

                current_question = line_without_heading
                current_question_text = line_without_heading
                current_options = []
                current_image_refs = []
                current_question_images = []
                current_option_images = []
                accumulating_option_idx = None

            elif current_question and not image_refs:
                option_match = re.match(r'^\s*(?:\(|\[)?([A-Fa-f])[\.|\)]\s*(.*)', line_stripped)
                if option_match:
                    option_label = option_match.group(1).upper()
                    option_text = option_match.group(2).strip()

                    cleaned = _clean_option_text(option_text) if option_text else f"Option {option_label}"
                    logger.debug(
                        "Detected text option %s: %s...",
                        option_label,
                        cleaned[:80] if cleaned else "(empty)"
                    )
                    current_options.append(cleaned)
                    accumulating_option_idx = len(current_options) - 1
                else:
                    if accumulating_option_idx is not None:
                        is_new_question_like = (
                            line_stripped.startswith(("Question", "Problem", "Q.", "Q ")) or
                            re.match(r'^\d+[\.\)]\s+', line_stripped) is not None or
                            re.match(r'^Q\.?\s*\d+', line_stripped) is not None
                        )
                        if not is_new_question_like:
                            addon = line_stripped
                            if not re.match(r'^(Answer|Solution)\b', addon, re.IGNORECASE):
                                merged = (current_options[accumulating_option_idx] + " " + addon).strip()
                                current_options[accumulating_option_idx] = _clean_option_text(merged)
                                logger.debug(
                                    "Appended continuation to option %s: %s...",
                                    accumulating_option_idx,
                                    current_options[accumulating_option_idx][:120]
                                )
                        else:
                            accumulating_option_idx = None

        if current_question:
            total_images = len(current_image_refs)
            num_option_images = len(current_option_images)
            num_question_figures = len(current_question_images)
            has_text_options = len(current_options) > 0 and any(opt.strip() for opt in current_options)

            final_question_images = current_question_images.copy()
            final_options = current_options
            is_image_based_mcq = False

            if num_option_images > 0:
                valid_text_options = [opt for opt in current_options if opt.strip()]

                if not valid_text_options:
                    logger.info(
                        "📊 Image-based MCQ (last): %s option images, %s question figures",
                        num_option_images,
                        num_question_figures
                    )
                    final_options = []
                    is_image_based_mcq = True
                else:
                    logger.warning(
                        "⚠️ Mixed question (last): %s option images + %s text options",
                        num_option_images,
                        len(valid_text_options)
                    )
                    final_options = valid_text_options
                    is_image_based_mcq = False
            elif total_images > 0 and not has_text_options:
                if total_images >= 3:
                    logger.info(
                        "✅ Fallback RULE 1 (last): %s images, no labels, no text → Treating as option images",
                        total_images
                    )
                    final_question_images = []
                    final_options = []
                    is_image_based_mcq = True
                else:
                    logger.info("📊 %s question figures (last, no option labels detected)", total_images)
            else:
                logger.info(
                    "📝 Text-based question (last): %s text options, %s question figures",
                    len(current_options),
                    num_question_figures
                )

            if final_options:
                final_options = [_clean_option_text(opt) for opt in final_options]

            logger.info(
                "Extracted last question: %s options, %s question figures, %s total images",
                len(final_options),
                len(final_question_images),
                total_images
            )
            questions.append(ExtractedQuestion(
                id=str(uuid.uuid4()),
                text=current_question_text,
                options=final_options,
                metadata={
                    "subject": subject,
                    "difficulty": difficulty,
                    "page": page.get("index", 0),
                    "image_refs": current_image_refs,
                    "question_image_refs": final_question_images,
                    "is_image_based_mcq": is_image_based_mcq
                }
            ))

    return questions
