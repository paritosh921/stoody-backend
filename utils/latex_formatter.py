"""
Utility functions for formatting LaTeX expressions in question text

This module ensures LaTeX content is properly wrapped with delimiters
for frontend rendering.
"""

import re
import logging

logger = logging.getLogger(__name__)


def format_latex_in_text(text: str) -> str:
    """
    Detect and wrap raw LaTeX expressions with proper delimiters.

    Converts:
        {\left(1 + \frac{a^{2}}{2}\right)}
    To:
        \({\left(1 + \frac{a^{2}}{2}\right)}\)

    Args:
        text: Raw text potentially containing LaTeX

    Returns:
        Text with LaTeX expressions properly delimited
    """
    if not text or not isinstance(text, str):
        return text

    # Skip if already has delimiters
    if '\\(' in text or '$$' in text or '$' in text:
        logger.debug("Text already has LaTeX delimiters, skipping formatting")
        return text

    # Patterns that indicate raw LaTeX content
    latex_patterns = [
        # {\left(...\right...)}
        (r'\{\\left\([^}]*\\right\)[^}]*\}', 'inline'),
        # {\frac{num}{denom}}
        (r'\{\\frac\{[^}]+\}\{[^}]+\}[^}]*\}', 'inline'),
        # \frac{num}{denom} without outer braces
        (r'\\frac\{[^}]+\}\{[^}]+\}', 'inline'),
        # \left...\right
        (r'\\left[\(\[\{][^\\]*\\right[\)\]\}]', 'inline'),
        # \sqrt{content}
        (r'\\sqrt\{[^}]+\}', 'inline'),
        # \mathrm{content} or \text{content}
        (r'\\(?:mathrm|text)\{[^}]+\}', 'inline'),
        # Variable with superscript: x^{2}
        (r'[A-Za-z_]\^\{[^}]+\}', 'inline'),
        # Variable with subscript: x_{0}
        (r'[A-Za-z_]_\{[^}]+\}', 'inline'),
        # Generic {content with \command}
        (r'\{[^}]*\\[a-z]+[^}]*\}', 'inline'),
    ]

    # Check if text contains raw LaTeX
    has_raw_latex = False
    for pattern, _ in latex_patterns:
        if re.search(pattern, text):
            has_raw_latex = True
            logger.info(f"Detected raw LaTeX pattern: {pattern}")
            break

    if not has_raw_latex:
        return text

    logger.info("Formatting raw LaTeX expressions in text")

    # Split text by existing delimited expressions to avoid double-wrapping
    # Preserve existing delimited math
    already_delimited_pattern = r'(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\\ [[\s\S]*?\\\]|\\\([^)]*?\\\)|!\[[^\]]*\]\([^)]*\))'
    segments = re.split(already_delimited_pattern, text)

    formatted_segments = []
    for i, segment in enumerate(segments):
        # Skip already delimited segments (odd indices from split)
        if i % 2 == 1:
            formatted_segments.append(segment)
            continue

        # Check if this segment has raw LaTeX
        segment_has_latex = False
        for pattern, _ in latex_patterns:
            if re.search(pattern, segment):
                segment_has_latex = True
                break

        if not segment_has_latex:
            formatted_segments.append(segment)
            continue

        # Split by common separators to wrap individual expressions
        # This prevents wrapping entire sentences
        parts = re.split(r'(Options:|will be|[.,:;]\s+|\n+)', segment)

        formatted_parts = []
        for j, part in enumerate(parts):
            # Keep separators as-is (odd indices)
            if j % 2 == 1:
                formatted_parts.append(part)
                continue

            # Check if this part has LaTeX
            part_has_latex = False
            for pattern, _ in latex_patterns:
                if re.search(pattern, part):
                    part_has_latex = True
                    break

            if part_has_latex and part.strip() and not part.strip().startswith('-'):
                # Wrap with inline math delimiters
                wrapped = f"\\({part.strip()}\\)"
                logger.debug(f"Wrapped LaTeX: {part.strip()[:50]}... -> {wrapped[:50]}...")
                formatted_parts.append(wrapped)
            else:
                formatted_parts.append(part)

        formatted_segments.append(''.join(formatted_parts))

    result = ''.join(formatted_segments)
    logger.info("LaTeX formatting complete")
    return result


def format_question_latex(question_dict: dict) -> dict:
    """
    Format LaTeX in all text fields of a question dictionary.

    Args:
        question_dict: Question data from MongoDB

    Returns:
        Question with formatted LaTeX expressions
    """
    if not question_dict:
        return question_dict

    # Format question text
    if 'text' in question_dict and question_dict['text']:
        question_dict['text'] = format_latex_in_text(question_dict['text'])

    # Format options if they exist
    if 'options' in question_dict and isinstance(question_dict['options'], list):
        question_dict['options'] = [
            format_latex_in_text(opt) if isinstance(opt, str) else opt
            for opt in question_dict['options']
        ]

    # Format enhanced options
    if 'enhancedOptions' in question_dict and isinstance(question_dict['enhancedOptions'], list):
        for opt in question_dict['enhancedOptions']:
            if isinstance(opt, dict) and opt.get('type') == 'text' and 'content' in opt:
                opt['content'] = format_latex_in_text(opt['content'])

    return question_dict
