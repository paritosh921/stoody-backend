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
    Wrap raw LaTeX or mhchem snippets in inline math delimiters without touching
    already-delimited math, code spans, markdown images, or Stoody image tokens.
    """
    if not text or not isinstance(text, str):
        return text

    text = text.replace("\\\\\\\\", "\\\\")
    protected_pattern = re.compile(
        r"(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|`[^`]*`|\[\[IMG:[^\]]+\]\]|!\[[^\]]*\]\([^)]+\))"
    )

    parts = []
    cursor = 0
    for match in protected_pattern.finditer(text):
        parts.append(_wrap_raw_latex_segment(text[cursor:match.start()]))
        parts.append(match.group(0))
        cursor = match.end()
    parts.append(_wrap_raw_latex_segment(text[cursor:]))
    return "".join(parts)


LATEX_COMMANDS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
    "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon",
    "phi", "varphi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
    "frac", "dfrac", "tfrac", "sqrt", "binom", "sum", "prod", "int", "oint",
    "partial", "nabla", "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln",
    "exp", "lim", "max", "min", "sup", "inf", "arg",
    "text", "textbf", "textit", "textrm", "mathrm", "mathbf", "mathit",
    "mathsf", "mathtt", "mathcal", "mathbb", "mathfrak", "bold", "boldsymbol",
    "hat", "bar", "vec", "dot", "ddot", "tilde", "overline", "underline",
    "overbrace", "underbrace", "overrightarrow", "overleftarrow",
    "left", "right", "middle", "big", "Big", "bigg", "Bigg",
    "quad", "qquad", "hspace", "vspace", "hfill", "vfill",
    "times", "div", "cdot", "pm", "mp", "leq", "geq", "neq", "approx",
    "equiv", "sim", "simeq", "cong", "propto", "perp", "parallel",
    "subset", "supset", "subseteq", "supseteq", "in", "notin", "ni",
    "cup", "cap", "setminus", "emptyset", "varnothing",
    "to", "rightarrow", "leftarrow", "leftrightarrow", "Rightarrow",
    "Leftarrow", "Leftrightarrow", "uparrow", "downarrow", "mapsto",
    "infty", "angle", "triangle", "square", "circ", "bullet", "star",
    "forall", "exists", "nexists", "therefore", "because",
    "ldots", "cdots", "vdots", "ddots", "prime", "degree",
    "ce", "pu",
}


def _wrap_raw_latex_segment(segment: str) -> str:
    result = []
    cursor = 0
    while cursor < len(segment):
        found = _find_next_raw_latex(segment, cursor)
        if not found:
            result.append(segment[cursor:])
            break
        start, content, length = found
        result.append(segment[cursor:start])
        result.append(f"${content}$")
        cursor = start + length
    return "".join(result)


def _find_next_raw_latex(text: str, from_index: int):
    for i in range(from_index, max(len(text) - 1, 0)):
        if text[i] != "\\" or not text[i + 1].isalpha():
            continue
        if text.startswith("\\(", i) or text.startswith("\\[", i):
            continue
        match = _match_raw_latex_expression(text, i)
        if match:
            content, length = match
            return i, content, length
    return None


def _match_raw_latex_expression(text: str, start: int):
    if start >= len(text) - 1 or text[start] != "\\" or not text[start + 1].isalpha():
        return None

    command_end = _read_command_end(text, start + 1)
    command = text[start + 1:command_end]
    if command not in LATEX_COMMANDS or command in {"right", "middle"}:
        return None

    if command == "left":
        return _match_left_right_expression(text, start)

    end = _consume_command_expression(text, start)
    if end <= command_end:
        return None
    end = _consume_connected_latex_tail(text, end)
    return text[start:end], end - start


def _read_command_end(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isalpha():
        cursor += 1
    return cursor


def _command_starts_at(text: str, index: int, command: str) -> bool:
    token = f"\\{command}"
    if not text.startswith(token, index):
        return False
    next_index = index + len(token)
    return next_index >= len(text) or not text[next_index].isalpha()


def _match_left_right_expression(text: str, start: int):
    cursor = _consume_latex_delimiter(text, start + len("\\left"))
    depth = 1
    while cursor < len(text):
        if _command_starts_at(text, cursor, "left"):
            depth += 1
            cursor = _consume_latex_delimiter(text, cursor + len("\\left"))
            continue
        if _command_starts_at(text, cursor, "right"):
            cursor = _consume_latex_delimiter(text, cursor + len("\\right"))
            depth -= 1
            if depth == 0:
                return text[start:cursor], cursor - start
            continue
        cursor += 1
    return None


def _consume_latex_delimiter(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor] == " ":
        cursor += 1
    if cursor >= len(text):
        return cursor
    if text[cursor] == "\\" and cursor + 1 < len(text) and text[cursor + 1] in "{}[]|":
        return cursor + 2
    return cursor + 1


def _consume_command_expression(text: str, start: int) -> int:
    cursor = _read_command_end(text, start + 1)
    while cursor < len(text) and text[cursor] == " ":
        cursor += 1
    while cursor < len(text) and text[cursor] == "[":
        close = _find_matching_pair(text, cursor, "[", "]")
        if close == -1:
            break
        cursor = close + 1
        while cursor < len(text) and text[cursor] == " ":
            cursor += 1
    while cursor < len(text) and text[cursor] == "{":
        close = _find_matching_pair(text, cursor, "{", "}")
        if close == -1:
            break
        cursor = close + 1
        while cursor < len(text) and text[cursor] == " ":
            cursor += 1
    return _consume_scripts(text, cursor)


def _consume_connected_latex_tail(text: str, cursor: int) -> int:
    while cursor < len(text):
        before = cursor
        char = text[cursor]
        if char == "\\":
            next_char = text[cursor + 1] if cursor + 1 < len(text) else ""
            if next_char in ",;:! ":
                cursor += 2
                continue
            if next_char.isalpha():
                match = _match_raw_latex_expression(text, cursor)
                if not match:
                    break
                _, length = match
                cursor += length
                continue
            break
        if char.isalnum():
            while cursor < len(text) and text[cursor].isalnum():
                cursor += 1
            cursor = _consume_scripts(text, cursor)
            continue
        if char == "{":
            close = _find_matching_pair(text, cursor, "{", "}")
            if close == -1:
                break
            cursor = close + 1
            continue
        if char in "^_+-=*/<>|":
            cursor += 1
            cursor = _consume_scripts(text, cursor)
            continue
        if char == "." and cursor > 0 and cursor + 1 < len(text) and text[cursor - 1].isdigit() and text[cursor + 1].isdigit():
            cursor += 1
            continue
        if char == " ":
            lookahead = cursor + 1
            if lookahead < len(text) and (text[lookahead] == "\\" or text[lookahead] in "+-=*/<>|"):
                cursor += 1
                continue
        if cursor == before:
            break
        break
    return cursor


def _consume_scripts(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor] in "^_":
        cursor += 1
        if cursor < len(text) and text[cursor] == "{":
            close = _find_matching_pair(text, cursor, "{", "}")
            if close == -1:
                break
            cursor = close + 1
        elif cursor < len(text) and re.match(r"[0-9a-zA-Z+\-]", text[cursor]):
            cursor += 1
    return cursor


def _find_matching_pair(text: str, open_index: int, open_char: str, close_char: str) -> int:
    depth = 1
    for i in range(open_index + 1, len(text)):
        if text[i] == open_char:
            depth += 1
        elif text[i] == close_char:
            depth -= 1
        if depth == 0:
            return i
    return -1


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
