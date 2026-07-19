"""
ExamPen Test Harness — Ingest substrate tests.

Test IDs covered:
    U-ING-01  Provenance fields (admin_id, student_id, pen_mac) are all present
    U-ING-02  Content hash is correct SHA-256
    I-ING-01  Write-once — second write with same content returns existing
    I-ING-02  Idempotent duplicate detection on (exam_id, student_id)

Spec authority: new-docs/architecture/DUAL_MODE_ARCHITECTURE.md section 3
Integrity:      new-docs/architecture/TAMPER_PROOF_SPEC.md Layer 1
Failure modes:  ING-01 (artifact loss), ING-02 (mis-attribution), ING-03 (duplicates)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exam-conductor")
if _EC_DIR not in sys.path:
    sys.path.insert(0, _EC_DIR)

from ingest.models import (
    AnswerPage,
    ArtifactSource,
    ConductedExamSubmission,
    IngestResult,
    PageRef,
    SubmissionStatus,
)
from ingest.hashing import compute_content_hash, compute_page_hash
from ingest.service import IngestService, _deterministic_submission_id
from ingest.repository import IngestRepository, ImmutableDocumentError


# ===========================================================================
# U-ING-01: Provenance fields are all present
# ===========================================================================


class TestUIng01:
    """U-ING-01: Conducted-exam artifact stored with admin_id, student_id,
    pen_mac, and timestamps."""

    def test_u_ing_01_submission_provenance_fields(self):
        """ConductedExamSubmission carries all provenance fields."""
        sub = ConductedExamSubmission(
            submission_id="sub-001",
            exam_id="exam-001",
            student_id="stu-001",
            admin_id="admin-001",
            source=ArtifactSource.BLE_PEN,
            pen_mac="AA:BB:CC:DD:EE:FF",
            page_count=3,
            content_hash="abc123",
        )
        assert sub.admin_id == "admin-001"
        assert sub.student_id == "stu-001"
        assert sub.pen_mac == "AA:BB:CC:DD:EE:FF"
        assert isinstance(sub.submitted_at, datetime)
        assert isinstance(sub.created_at, datetime)
        assert sub.exam_id == "exam-001"
        assert sub.source == ArtifactSource.BLE_PEN.value

    def test_u_ing_01_answer_page_provenance_fields(self):
        """AnswerPage carries provenance fields including pen_mac."""
        page = AnswerPage(
            page_id="page-001",
            submission_id="sub-001",
            exam_id="exam-001",
            student_id="stu-001",
            admin_id="admin-001",
            page_number=1,
            source=ArtifactSource.BLE_PEN,
            pen_mac="AA:BB:CC:DD:EE:FF",
            raw_strokes=[{"x": 0, "y": 0}],
            content_hash="def456",
        )
        assert page.admin_id == "admin-001"
        assert page.student_id == "stu-001"
        assert page.pen_mac == "AA:BB:CC:DD:EE:FF"
        assert page.page_number == 1
        assert page.source == ArtifactSource.BLE_PEN.value
        assert isinstance(page.created_at, datetime)

    def test_u_ing_01_immutable_flag_default_true(self):
        """Documents carry _immutable = True by default (TAMPER_PROOF Layer 1)."""
        sub = ConductedExamSubmission(
            submission_id="sub-001",
            exam_id="exam-001",
            student_id="stu-001",
            admin_id="admin-001",
            source=ArtifactSource.BLE_PEN,
            content_hash="abc",
        )
        assert sub.immutable is True

        page = AnswerPage(
            page_id="p-001",
            submission_id="sub-001",
            exam_id="exam-001",
            student_id="stu-001",
            admin_id="admin-001",
            page_number=1,
            source=ArtifactSource.BLE_PEN,
            content_hash="def",
        )
        assert page.immutable is True

    def test_u_ing_01_to_mongo_doc_uses_alias(self):
        """to_mongo_doc() serializes immutable under _immutable key."""
        sub = ConductedExamSubmission(
            submission_id="sub-001",
            exam_id="exam-001",
            student_id="stu-001",
            admin_id="admin-001",
            source=ArtifactSource.BLE_PEN,
            content_hash="abc",
        )
        doc = sub.to_mongo_doc()
        assert doc["_immutable"] is True
        assert "admin_id" in doc
        assert "student_id" in doc

    def test_u_ing_01_camera_source_pen_mac_optional(self):
        """Camera-originated submissions do not require pen_mac."""
        sub = ConductedExamSubmission(
            submission_id="sub-002",
            exam_id="exam-002",
            student_id="stu-002",
            admin_id="admin-002",
            source=ArtifactSource.CAMERA,
            pen_mac=None,
            content_hash="ghi",
        )
        assert sub.pen_mac is None
        assert sub.source == ArtifactSource.CAMERA.value


# ===========================================================================
# U-ING-02: Content hash is correct SHA-256
# ===========================================================================


class TestUIng02:
    """U-ING-02: Content hash generated for conducted-exam artifact."""

    def test_u_ing_02_page_hash_deterministic(self):
        """compute_page_hash returns consistent SHA-256 for same input."""
        strokes = [{"x": 10, "y": 20}]
        h1 = compute_page_hash(page_number=1, raw_strokes=strokes)
        h2 = compute_page_hash(page_number=1, raw_strokes=strokes)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex = 64 chars

    def test_u_ing_02_page_hash_differs_for_different_content(self):
        """Different strokes produce different hashes."""
        h1 = compute_page_hash(
            page_number=1, raw_strokes=[{"x": 0, "y": 0}]
        )
        h2 = compute_page_hash(
            page_number=1, raw_strokes=[{"x": 1, "y": 1}]
        )
        assert h1 != h2

    def test_u_ing_02_page_hash_differs_for_different_page_number(self):
        """Same strokes on different pages produce different hashes."""
        strokes = [{"x": 10, "y": 20}]
        h1 = compute_page_hash(page_number=1, raw_strokes=strokes)
        h2 = compute_page_hash(page_number=2, raw_strokes=strokes)
        assert h1 != h2

    def test_u_ing_02_page_hash_camera_path(self):
        """compute_page_hash works for camera-originated pages (image ref)."""
        h = compute_page_hash(page_number=1, raw_image_ref="s3://bucket/img1.jpg")
        assert len(h) == 64

    def test_u_ing_02_camera_hash_commits_to_image_bytes_not_storage_path(self):
        """Relocating an immutable object does not change its byte commitment."""
        digest = "ab" * 32
        first = compute_page_hash(
            page_number=1,
            raw_image_ref="s3://bucket/original/page.png",
            asset_sha256=digest,
        )
        relocated = compute_page_hash(
            page_number=1,
            raw_image_ref="s3://bucket/archive/page.png",
            asset_sha256=digest,
        )
        changed_bytes = compute_page_hash(
            page_number=1,
            raw_image_ref="s3://bucket/original/page.png",
            asset_sha256="cd" * 32,
        )

        assert first == relocated
        assert first != changed_bytes

    def test_u_ing_02_rejects_invalid_camera_byte_digest(self):
        with pytest.raises(ValueError, match="asset_sha256"):
            compute_page_hash(
                page_number=1,
                raw_image_ref="s3://bucket/page.png",
                asset_sha256="not-a-sha256",
            )

    def test_u_ing_02_content_hash_deterministic(self):
        """compute_content_hash returns consistent SHA-256 for same input."""
        page_hashes = ["aaa", "bbb"]
        h1 = compute_content_hash(
            exam_id="exam-001", student_id="stu-001", page_hashes=page_hashes
        )
        h2 = compute_content_hash(
            exam_id="exam-001", student_id="stu-001", page_hashes=page_hashes
        )
        assert h1 == h2
        assert len(h1) == 64

    def test_u_ing_02_content_hash_is_sha256(self):
        """Content hash is a valid hex SHA-256 digest."""
        h = compute_content_hash(
            exam_id="exam-001",
            student_id="stu-001",
            page_hashes=["abc"],
        )
        # Verify it can be decoded as hex
        bytes.fromhex(h)
        assert len(h) == 64

    def test_u_ing_02_content_hash_differs_for_different_student(self):
        """Different student_id produces different submission hash."""
        page_hashes = ["same_hash"]
        h1 = compute_content_hash(
            exam_id="exam-001", student_id="stu-001", page_hashes=page_hashes
        )
        h2 = compute_content_hash(
            exam_id="exam-001", student_id="stu-002", page_hashes=page_hashes
        )
        assert h1 != h2

    def test_u_ing_02_page_hash_matches_manual_sha256(self):
        """Verify the hash algorithm matches manual SHA-256 computation."""
        page_number = 1
        raw_strokes = [{"a": 1}]

        # Manual computation
        h = hashlib.sha256()
        h.update(f"page:{page_number}".encode("utf-8"))
        h.update(b"strokes:")
        canonical = json.dumps(raw_strokes, sort_keys=True, separators=(",", ":"), default=str)
        h.update(canonical.encode("utf-8"))
        expected = h.hexdigest()

        actual = compute_page_hash(page_number=page_number, raw_strokes=raw_strokes)
        assert actual == expected


# ===========================================================================
# I-ING-01: Write-once — second write with same content returns existing
# ===========================================================================


class TestIIng01:
    """I-ING-01: hub/upload path writes canonical artifact once."""

    def test_i_ing_01_write_once_first_insert(self):
        """First ingest creates the submission and answer pages."""
        async def _run():
            db = MagicMock()
            service = IngestService(db)
            repo = AsyncMock()
            service._repo = repo

            # Simulate successful first insert
            repo.insert_answer_pages_bulk = AsyncMock(
                return_value=(2, 0, ["page-1", "page-2"])
            )
            repo.insert_submission = AsyncMock(
                side_effect=lambda doc: (doc, False)
            )

            result = await service.ingest_submission(
                exam_id="exam-001",
                student_id="stu-001",
                admin_id="admin-001",
                source="ble_pen",
                pen_mac="AA:BB:CC:DD:EE:FF",
                pages=[
                    {"page_number": 1, "raw_strokes": [{"x": 0}]},
                    {"page_number": 2, "raw_strokes": [{"x": 1}]},
                ],
            )

            assert isinstance(result, IngestResult)
            assert result.already_existed is False
            assert result.page_count == 2
            assert len(result.content_hash) == 64
        asyncio.run(_run())

    def test_i_ing_01_pen_mac_required_for_ble_pen(self):
        """pen_mac is required when source is ble_pen (ING-02 provenance)."""
        async def _run():
            db = MagicMock()
            service = IngestService(db)

            with pytest.raises(ValueError, match="pen_mac is required"):
                await service.ingest_submission(
                    exam_id="exam-001",
                    student_id="stu-001",
                    admin_id="admin-001",
                    source="ble_pen",
                    pen_mac=None,
                    pages=[{"page_number": 1, "raw_strokes": []}],
                )
        asyncio.run(_run())


# ===========================================================================
# I-ING-02: Idempotent duplicate detection on (exam_id, student_id)
# ===========================================================================


class TestIIng02:
    """I-ING-02: duplicate upload is idempotent."""

    def test_i_ing_02_duplicate_returns_existing(self):
        """Second ingest with same content returns already_existed=True."""
        async def _run():
            db = MagicMock()
            service = IngestService(db)
            repo = AsyncMock()
            service._repo = repo

            content_hash = "abcdef0123456789" * 4  # 64-char fake hash

            # Simulate duplicate on insert_submission
            existing_doc = {
                "submission_id": "sub-existing",
                "content_hash": content_hash,
                "page_count": 1,
                "segmentation_status": "pending",
            }
            repo.insert_answer_pages_bulk = AsyncMock(return_value=(0, 1, []))
            repo.insert_submission = AsyncMock(
                return_value=(existing_doc, True)
            )

            result = await service.ingest_submission(
                exam_id="exam-001",
                student_id="stu-001",
                admin_id="admin-001",
                source="camera",
                pages=[{"page_number": 1, "raw_image_ref": "s3://img1.jpg"}],
            )

            assert result.already_existed is True
            assert result.submission_id == "sub-existing"
        asyncio.run(_run())

    def test_i_ing_02_deterministic_submission_id(self):
        """Same (exam_id, student_id) always produces the same submission_id."""
        id1 = _deterministic_submission_id("exam-001", "stu-001")
        id2 = _deterministic_submission_id("exam-001", "stu-001")
        assert id1 == id2
        assert len(id1) == 32

    def test_i_ing_02_different_pair_different_id(self):
        """Different (exam_id, student_id) produces different submission_id."""
        id1 = _deterministic_submission_id("exam-001", "stu-001")
        id2 = _deterministic_submission_id("exam-001", "stu-002")
        assert id1 != id2

    def test_i_ing_02_ingest_result_shape(self):
        """IngestResult has the expected fields."""
        result = IngestResult(
            submission_id="sub-001",
            content_hash="a" * 64,
            page_count=3,
            segmentation_status=SubmissionStatus.PENDING,
            already_existed=False,
        )
        assert result.submission_id == "sub-001"
        assert result.page_count == 3
        assert result.segmentation_status == SubmissionStatus.PENDING
        assert result.already_existed is False

    def test_submission_failure_rolls_back_only_pages_inserted_by_this_attempt(self):
        """A duplicate immutable page must survive compensating cleanup."""

        async def _run():
            db = MagicMock()
            service = IngestService(db)
            repo = AsyncMock()
            service._repo = repo
            repo.insert_answer_pages_bulk = AsyncMock(
                return_value=(1, 1, ["new-page-id"])
            )
            repo.insert_submission = AsyncMock(side_effect=RuntimeError("write failed"))

            with pytest.raises(RuntimeError, match="write failed"):
                await service.ingest_submission(
                    exam_id="exam-001",
                    student_id="stu-001",
                    admin_id="admin-001",
                    source="camera",
                    pages=[
                        {
                            "page_number": 1,
                            "raw_image_ref": "s3://copy/existing.png",
                            "content_hash": "11" * 32,
                        },
                        {
                            "page_number": 2,
                            "raw_image_ref": "s3://copy/new.png",
                            "content_hash": "22" * 32,
                        },
                    ],
                )

            repo.delete_answer_pages_by_ids.assert_awaited_once_with(
                ["new-page-id"]
            )

        asyncio.run(_run())
