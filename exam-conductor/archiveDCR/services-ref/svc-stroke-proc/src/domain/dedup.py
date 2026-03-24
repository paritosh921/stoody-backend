"""Idempotency / deduplication logic for stroke chunks.

ZERO I/O -- this module must never import asyncio, aiohttp, sqlalchemy,
nats, or any I/O library.

Idempotency key format: ``{exam_id}:{pen_mac}:{chunk_index}``
"""

from __future__ import annotations


def make_idempotency_key(
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
) -> str:
    """Build a deterministic idempotency key for a single chunk.

    Parameters
    ----------
    exam_id:
        UUID string identifying the exam session.
    pen_mac:
        MAC address of the pen (e.g. ``AA:BB:CC:DD:EE:FF``).
    chunk_index:
        Zero-based index of this chunk within the pen's upload.

    Returns
    -------
    Colon-separated key suitable for set lookup or DB constraint.
    """
    return f"{exam_id}:{pen_mac}:{chunk_index}"


def is_duplicate(key: str, seen_keys: set[str]) -> bool:
    """Check whether *key* has already been processed in this batch.

    Parameters
    ----------
    key:
        Idempotency key produced by :func:`make_idempotency_key`.
    seen_keys:
        Mutable set of keys already encountered. If *key* is new it is
        added to the set before returning ``False``.

    Returns
    -------
    ``True`` if *key* was already in *seen_keys* (duplicate);
    ``False`` otherwise (first occurrence, now recorded).
    """
    if key in seen_keys:
        return True
    seen_keys.add(key)
    return False


def filter_duplicates(
    keys: list[str],
    already_committed: set[str] | None = None,
) -> tuple[list[int], set[str]]:
    """Partition a list of idempotency keys into new vs duplicate.

    Parameters
    ----------
    keys:
        Ordered list of idempotency keys to evaluate.
    already_committed:
        Optional set of keys known to be committed in the database.
        If ``None``, only intra-batch dedup is performed.

    Returns
    -------
    Tuple of ``(new_indices, all_seen)`` where *new_indices* lists the
    positional indices of keys that are not duplicates, and *all_seen*
    is the union of *already_committed* and any new keys.
    """
    seen: set[str] = set(already_committed) if already_committed else set()
    new_indices: list[int] = []

    for idx, key in enumerate(keys):
        if not is_duplicate(key, seen):
            new_indices.append(idx)

    return new_indices, seen
