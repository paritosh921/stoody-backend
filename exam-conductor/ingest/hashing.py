"""
Content Hashing Utilities
=========================

SHA-256 based content hashing for canonical conducted-exam artifacts.

The content hash is the foundation of tamper-proof integrity
(TAMPER_PROOF_SPEC Layer 1).  It is computed once at ingest time and
stored alongside the artifact.  Downstream consumers can recompute the
hash to verify that the artifact has not been altered (TAMP-02).

Test coverage
-------------
- U-ING-02: content hash generated for conducted-exam artifact
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonicalize(obj: Any) -> str:
    """Produce a deterministic JSON string for hashing.

    Uses ``sort_keys=True`` and no extra whitespace so that logically
    equivalent payloads always produce the same digest.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_page_hash(
    *,
    page_number: int,
    raw_strokes: Optional[List[Dict[str, Any]]] = None,
    raw_image_ref: Optional[str] = None,
    asset_sha256: Optional[str] = None,
) -> str:
    """Compute SHA-256 hex digest for a single answer page.

    For pen-originated pages the hash covers the canonical stroke vectors.
    For camera-originated pages the preferred input is the SHA-256 digest of
    the actual image bytes.  ``raw_image_ref`` remains a legacy fallback for
    older capture clients that cannot yet provide a byte commitment.

    Parameters
    ----------
    page_number:
        1-based page number (included in hash to bind content to position).
    raw_strokes:
        Canonical stroke vector list (pen path).
    raw_image_ref:
        Opaque reference to a camera/scan image asset.
    asset_sha256:
        Lowercase SHA-256 digest of the actual immutable image bytes.

    Returns
    -------
    str
        Lowercase hex SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(f"page:{page_number}".encode("utf-8"))

    if raw_strokes is not None:
        h.update(b"strokes:")
        h.update(_canonicalize(raw_strokes).encode("utf-8"))

    normalized_asset_hash = str(asset_sha256 or "").strip().lower()
    if normalized_asset_hash:
        if not _SHA256_RE.fullmatch(normalized_asset_hash):
            raise ValueError("asset_sha256 must be a 64-character hexadecimal SHA-256 digest")
        h.update(b"image_sha256:")
        h.update(normalized_asset_hash.encode("ascii"))
    elif raw_image_ref is not None:
        # Backward compatibility only. New camera/PDF paths must pass the
        # scanner/object checksum so a mutable path cannot stand in for bytes.
        h.update(b"image_ref:")
        h.update(raw_image_ref.encode("utf-8"))

    return h.hexdigest()


def compute_content_hash(
    *,
    exam_id: str,
    student_id: str,
    page_hashes: List[str],
) -> str:
    """Compute a submission-level SHA-256 content hash.

    The submission hash is derived from the exam identity, student identity,
    and the ordered list of per-page hashes.  This makes the submission hash
    a Merkle-like commitment over all page content.

    Parameters
    ----------
    exam_id:
        Conducted exam identifier.
    student_id:
        Student identity.
    page_hashes:
        Ordered list of per-page SHA-256 hex digests (from ``compute_page_hash``).

    Returns
    -------
    str
        Lowercase hex SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(f"exam:{exam_id}".encode("utf-8"))
    h.update(f"student:{student_id}".encode("utf-8"))

    for idx, ph in enumerate(page_hashes):
        h.update(f"page_hash:{idx}:{ph}".encode("utf-8"))

    return h.hexdigest()
