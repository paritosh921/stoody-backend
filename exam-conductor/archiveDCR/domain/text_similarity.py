"""TF-IDF cosine similarity between answer text pairs.

ZERO I/O -- pure computation only.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ---- constants ------------------------------------------------------------ #

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "of", "in", "to",
        "for", "with", "on", "at", "from", "by", "as", "into", "through",
        "during", "before", "after", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no", "only", "own",
        "same", "than", "too", "very", "it", "its", "this", "that", "these",
        "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "his", "she", "her", "they", "them", "their", "what", "which", "who",
        "whom", "how", "when", "where", "why",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---- helpers -------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, remove stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    """Normalized term frequency: count / total tokens."""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def _inverse_document_frequency(
    doc_a_tokens: list[str],
    doc_b_tokens: list[str],
) -> dict[str, float]:
    """IDF across the two-document corpus.

    idf(t) = log(N / df(t)) where N=2 and df(t) is how many of the two
    documents contain term t.
    """
    vocab_a = set(doc_a_tokens)
    vocab_b = set(doc_b_tokens)
    all_terms = vocab_a | vocab_b
    idf: dict[str, float] = {}
    for term in all_terms:
        df = (1 if term in vocab_a else 0) + (1 if term in vocab_b else 0)
        # Smoothed IDF: log(1 + N/df) so shared terms still have positive weight.
        # Without smoothing, identical docs yield idf=log(2/2)=0 for all terms.
        idf[term] = math.log(1.0 + 2.0 / df)
    return idf


def _tfidf_vector(
    tf: dict[str, float],
    idf: dict[str, float],
) -> dict[str, float]:
    """TF-IDF weight per term."""
    return {term: tf.get(term, 0.0) * idf.get(term, 0.0) for term in idf}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---- public API ----------------------------------------------------------- #


def compute_tfidf_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity (0.0-1.0) of TF-IDF vectors for two texts.

    Returns 0.0 when either text is empty after tokenization.
    Returns 1.0 for identical (post-normalization) texts.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = _term_frequency(tokens_a)
    tf_b = _term_frequency(tokens_b)
    idf = _inverse_document_frequency(tokens_a, tokens_b)

    vec_a = _tfidf_vector(tf_a, idf)
    vec_b = _tfidf_vector(tf_b, idf)

    return _cosine(vec_a, vec_b)
