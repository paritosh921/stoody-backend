"""Unit tests for domain/dedup.py — idempotency key and duplicate filtering.

Test IDs: U-SPROC-01 through U-SPROC-08
Markers: unit (ZERO I/O)
"""

from __future__ import annotations

import pytest

from src.domain.dedup import (
    filter_duplicates,
    is_duplicate,
    make_idempotency_key,
)

# ---------------------------------------------------------------------------
# U-SPROC-01: Idempotency key format
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_idempotency_key_format():
    key = make_idempotency_key("exam-abc", "AA:BB:CC:DD:EE:FF", 7)
    assert key == "exam-abc:AA:BB:CC:DD:EE:FF:7"


@pytest.mark.unit
def test_idempotency_key_zero_index():
    key = make_idempotency_key("e1", "00:11:22:33:44:55", 0)
    assert key == "e1:00:11:22:33:44:55:0"


# ---------------------------------------------------------------------------
# U-SPROC-02: is_duplicate detects first occurrence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_duplicate_first_occurrence():
    seen: set[str] = set()
    assert is_duplicate("key-1", seen) is False
    assert "key-1" in seen


# ---------------------------------------------------------------------------
# U-SPROC-03: is_duplicate detects second occurrence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_duplicate_second_occurrence():
    seen: set[str] = {"key-1"}
    assert is_duplicate("key-1", seen) is True


# ---------------------------------------------------------------------------
# U-SPROC-04: is_duplicate handles distinct keys
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_duplicate_distinct_keys():
    seen: set[str] = set()
    assert is_duplicate("a", seen) is False
    assert is_duplicate("b", seen) is False
    assert is_duplicate("a", seen) is True
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# U-SPROC-05: filter_duplicates — all new
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_duplicates_all_new():
    keys = ["k1", "k2", "k3"]
    new_indices, seen = filter_duplicates(keys)
    assert new_indices == [0, 1, 2]
    assert seen == {"k1", "k2", "k3"}


# ---------------------------------------------------------------------------
# U-SPROC-06: filter_duplicates — intra-batch dedup
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_duplicates_intra_batch():
    keys = ["k1", "k2", "k1", "k3", "k2"]
    new_indices, _ = filter_duplicates(keys)
    assert new_indices == [0, 1, 3]


# ---------------------------------------------------------------------------
# U-SPROC-07: filter_duplicates — with already-committed set
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_duplicates_with_committed():
    keys = ["k1", "k2", "k3"]
    committed = {"k1", "k3"}
    new_indices, seen = filter_duplicates(keys, already_committed=committed)
    assert new_indices == [1]
    assert "k2" in seen


# ---------------------------------------------------------------------------
# U-SPROC-08: filter_duplicates — empty list
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_filter_duplicates_empty():
    new_indices, seen = filter_duplicates([])
    assert new_indices == []
    assert seen == set()
