"""Structural similarity: edit distance and same-error weighting.

ZERO I/O -- pure computation only.
"""

from __future__ import annotations


# ---- Levenshtein ---------------------------------------------------------- #


def _levenshtein_distance(s: str, t: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses the classic two-row dynamic programming approach for O(min(m,n))
    space.
    """
    if len(s) < len(t):
        return _levenshtein_distance(t, s)

    # s is the longer (or equal) string
    m, n = len(s), len(t)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def _normalized_edit_similarity(text_a: str, text_b: str) -> float:
    """Return 1 - (edit_distance / max_len), clamped to [0.0, 1.0].

    Identical strings -> 1.0, completely different -> near 0.0.
    Empty pair -> 0.0.
    """
    if not text_a and not text_b:
        return 0.0
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 0.0
    dist = _levenshtein_distance(text_a, text_b)
    return 1.0 - (dist / max_len)


# ---- Same-error weighting ------------------------------------------------- #


def _extract_errors(
    text: str,
    correct_answer: str | None,
) -> set[str]:
    """Identify tokens in *text* that differ from *correct_answer*.

    If no correct answer is provided, returns an empty set (no error
    analysis possible).  Each "error" is a lowercase token that appears
    in the student text but not in the correct answer.
    """
    if correct_answer is None:
        return set()
    student_tokens = set(text.lower().split())
    correct_tokens = set(correct_answer.lower().split())
    return student_tokens - correct_tokens


def _same_error_boost(
    text_a: str,
    text_b: str,
    correct_answer: str | None = None,
) -> float:
    """Return a 0.0-1.0 boost based on shared *wrong* tokens.

    Shared incorrect tokens (tokens absent from the correct answer)
    are a stronger plagiarism signal than shared correct tokens.
    Returns the Jaccard similarity of the two error sets.
    """
    errors_a = _extract_errors(text_a, correct_answer)
    errors_b = _extract_errors(text_b, correct_answer)

    if not errors_a and not errors_b:
        return 0.0

    intersection = errors_a & errors_b
    union = errors_a | errors_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


# ---- public API ----------------------------------------------------------- #

_ERROR_BOOST_WEIGHT = 0.3


def compute_structural_similarity(
    text_a: str,
    text_b: str,
    correct_answer: str | None = None,
) -> float:
    """Return structural similarity (0.0-1.0) combining edit distance
    and same-error weighting.

    Base score is the normalized Levenshtein similarity.  If a
    ``correct_answer`` is provided, identical mistakes in both texts
    boost the score (weighted at 30%).

    Parameters
    ----------
    text_a, text_b:
        Student answer texts to compare.
    correct_answer:
        Optional known-correct answer for same-error analysis.

    Returns
    -------
    float in [0.0, 1.0]
    """
    base = _normalized_edit_similarity(text_a, text_b)

    # Without a correct answer, error analysis is not possible —
    # return the raw edit similarity as the full score.
    if correct_answer is None:
        return base

    boost = _same_error_boost(text_a, text_b, correct_answer)

    # Weighted combination, capped at 1.0
    combined = base * (1.0 - _ERROR_BOOST_WEIGHT) + boost * _ERROR_BOOST_WEIGHT
    return min(combined, 1.0)
