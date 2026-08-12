import io
import asyncio
import re
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from starlette.requests import Request

from PIL import Image, ImageDraw
from unittest.mock import AsyncMock
from bson import ObjectId
import pytest

from api.v1 import credits_async
from services import student_credits


class FakeCursor:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def sort(self, *_args, **_kwargs) -> "FakeCursor":
        return self

    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        if length is None or length < 0:
            return list(self._rows)
        return list(self._rows[:length])


def _query_value_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        for op, op_value in expected.items():
            if op == "$in":
                if value not in set(op_value):
                    return False
            if op == "$nin":
                if value in set(op_value):
                    return False
            if op == "$ne":
                if value == op_value:
                    return False
            if op == "$exists":
                exists = value is not None and value != ""
                if bool(op_value) != exists:
                    return False
            if op == "$regex":
                options = expected.get("$options", "")
                flags = re.IGNORECASE if "i" in options else 0
                if not re.match(str(op_value), str(value or ""), flags):
                    return False
        return True
    return value == expected


def _match_query(document: Dict[str, Any], query: Dict[str, Any]) -> bool:
    for key, expected in query.items():
        if not _query_value_matches(document.get(key), expected):
            return False
    return True


class FakeCollection:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.find_queries: List[Dict[str, Any]] = []
        self.find_one_calls: List[Dict[str, Any]] = []

    async def find_one(self, query: Dict[str, Any], projection=None):
        self.find_one_calls.append({"query": query, "projection": projection})
        for doc in self.documents:
            if _match_query(doc, query):
                return dict(doc)
        return None

    def find(self, query: Dict[str, Any], projection=None):
        self.find_queries.append({"query": query, "projection": projection})
        return FakeCursor([dict(document) for document in self.documents if _match_query(document, query)])


class FakeDb:
    def __init__(self, collections: Dict[str, FakeCollection]):
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


def _png_bytes(width: int = 20, height: int = 20, color=(255, 255, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _readable_white_page_bytes() -> bytes:
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    for row in range(260, 1320, 70):
        draw.line((180, row, 1020, row + 8), fill=(25, 25, 25), width=8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeDbManager:
    def __init__(self, tenant_db: Any):
        self._tenant_db = tenant_db

    async def get_tenant_db(self, db_name: str):
        return self._tenant_db


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


@pytest.mark.asyncio
async def test_judge_stroke_source_uses_canvas_lookup_query_without_page_id():
    canvas_collection = FakeCollection([
        {
            "user_id": "stu-1",
            "copy_id": "copy-1",
            "book_type": "MS",
            "page_number": 4,
            "strokes": [],
        }
    ])
    db = FakeDb({"canvas_pages": canvas_collection})
    job = {
        "source_ref": {
            "user_id": "stu-1",
            "copy_id": "copy-1",
            "book_type": "ms",
            "page_number": 4,
            "source": "test",
            "pen_mac": "PEN-1",
        },
        "source_id": "canvas:stu-1:copy-1:MS:4",
    }
    policy = student_credits._normalise_policy({})

    result = await student_credits._judge_stroke_source(db, job, policy)

    assert result["decision"] == "rejected"
    query = canvas_collection.find_one_calls[0]["query"]
    assert "_id" not in query
    assert query["user_id"] == "stu-1"
    assert query["book_type"] == "MS"
    assert query["page_number"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path_length_mm", "expected_credits"),
    [(250.0, 1), (251.0, 2), (1251.0, 5)],
)
async def test_v2_stroke_award_uses_250mm_units_and_five_credit_cap(
    monkeypatch,
    path_length_mm: float,
    expected_credits: int,
):
    page = {
        "user_id": "stu-1",
        "copy_id": "copy-1",
        "book_type": "MS",
        "page_number": 4,
    }
    db = FakeDb({"canvas_pages": FakeCollection([page])})
    metrics = {
        "stroke_count": 8,
        "point_count": 64,
        "path_length_mm": path_length_mm,
        "coverage": 0.08,
        "repeat_ratio": 0.05,
        "deterministic_score": 0.92,
    }
    monkeypatch.setattr(student_credits, "compute_stroke_metrics", lambda _page: metrics)

    result = await student_credits._judge_stroke_source(
        db,
        {
            "source_ref": {
                "user_id": "stu-1",
                "copy_id": "copy-1",
                "book_type": "MS",
                "page_number": 4,
            },
            "source_id": "canvas:stu-1:copy-1:MS:4",
        },
        student_credits._normalise_policy({"semantic_judge_enabled": False}),
    )

    assert result["decision"] == "accepted"
    assert result["target_credits"] == expected_credits


@pytest.mark.asyncio
async def test_judge_photo_source_uses_private_answer_copy_prefix(monkeypatch):
    pages = FakeCollection([{"submission_id": "sub-1", "raw_image_ref": "private/exampen/student-answer-copies/page-1.png", "page_number": 1}])
    db = FakeDb({"evalpen_answer_pages": pages})
    job = {"source_ref": {"submission_id": "sub-1"}, "source_type": student_credits.SOURCE_PHOTO}
    policy = student_credits._normalise_policy({"semantic_judge_enabled": False})
    fake_download = AsyncMock(return_value=_png_bytes())
    monkeypatch.setattr(student_credits, "download_private_object", fake_download)

    result = await student_credits._judge_photo_source(db, job, policy)

    assert fake_download.call_count == 1
    assert fake_download.await_args.kwargs["allowed_key_prefix"] == "private/exampen/student-answer-copies"
    assert result["metrics"]["page_count"] == 1
    assert result["decision"] == "rejected"


def test_hard_gate_reasons_include_image_gate_and_canvas_source_flags():
    policy = student_credits._normalise_policy({
        "min_written_coverage": 0.05,
        "max_written_coverage": 0.60,
        "max_skew_angle": 10.0,
        "max_perspective_distortion": 0.10,
        "max_glare_ratio": 0.01,
        "max_overexposure_ratio": 0.10,
        "max_edge_clipping_ratio": 0.01,
    })
    reasons = student_credits._hard_gate_reasons(
        metrics={
            "width": 1000,
            "height": 1000,
            "blur_variance": 0.0,
            "ink_density": 0.002,
            "written_coverage_ratio": 0.0001,
            "skew_angle": 25.0,
            "perspective_distortion": 0.25,
            "glare_ratio": 0.05,
            "overexposure_ratio": 0.5,
            "edge_clipping_ratio": 0.20,
        },
        policy=policy,
        source_type=student_credits.SOURCE_PHOTO,
    )

    assert "insufficient_written_coverage" in reasons
    assert "image_skew_too_high" in reasons
    assert "perspective_distortion_too_high" in reasons
    assert "image_glare_excessive" in reasons
    assert "image_overexposed" in reasons
    assert "image_edge_clipping" in reasons
    assert not student_credits._is_canvas_student_source({"source": "camera", "pen_mac": "PEN"}, "canvas:stu:1:MS:1")


def test_readable_white_page_is_not_misclassified_as_glare_or_overexposure():
    metrics = student_credits.compute_image_metrics(_readable_white_page_bytes())
    reasons = student_credits._hard_gate_reasons(
        metrics,
        student_credits._normalise_policy({}),
        student_credits.SOURCE_PHOTO,
    )

    assert metrics["written_coverage_ratio"] > 0.008
    assert metrics["glare_ratio"] == 0.0
    assert metrics["overexposure_ratio"] == 0.0
    assert metrics["skew_angle"] <= 30.0
    assert "image_glare_excessive" not in reasons
    assert "image_overexposed" not in reasons
    assert "image_skew_too_high" not in reasons


@pytest.mark.asyncio
async def test_semantic_judge_uses_strict_responses_schema(monkeypatch):
    captured: Dict[str, Any] = {}

    class FakeGate:
        def __init__(self, _db):
            pass

        async def call(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                completion_status="completed",
                incomplete_reason=None,
                content=json.dumps(
                    {
                        "has_handwriting": True,
                        "legible": True,
                        "quality_score": 0.88,
                        "randomness_score": 0.05,
                        "reason_codes": [],
                    }
                ),
            )

    fake_module = types.ModuleType("llm_gate")
    fake_module.LLMGate = FakeGate
    monkeypatch.setitem(sys.modules, "llm_gate", fake_module)

    result = await student_credits._semantic_judgment(
        object(), _readable_white_page_bytes(), student_credits.SOURCE_PHOTO
    )

    assert result["has_handwriting"] is True
    assert "messages" not in captured
    assert "responses_input" in captured
    assert captured["responses_input"][0]["content"][1]["type"] == "input_image"
    assert captured["json_schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_judge_photo_source_rejects_multi_page_when_any_page_fails(monkeypatch):
    pages = FakeCollection([
        {"submission_id": "sub-2", "raw_image_ref": "private/exampen/student-answer-copies/page-1.png", "page_number": 1},
        {"submission_id": "sub-2", "raw_image_ref": "private/exampen/student-answer-copies/page-2.png", "page_number": 2},
    ])
    db = FakeDb({"evalpen_answer_pages": pages})

    metrics_by_page = [
        {
            "width": 1600,
            "height": 1200,
            "blur_variance": 100.0,
            "ink_density": 0.12,
            "deterministic_score": 0.85,
            "written_coverage_ratio": 0.08,
            "skew_angle": 3.0,
            "perspective_distortion": 0.02,
            "glare_ratio": 0.01,
            "overexposure_ratio": 0.03,
            "edge_clipping_ratio": 0.01,
        },
        {
            "width": 640,
            "height": 1200,
            "blur_variance": 100.0,
            "ink_density": 0.12,
            "deterministic_score": 0.85,
            "written_coverage_ratio": 0.08,
            "skew_angle": 3.0,
            "perspective_distortion": 0.02,
            "glare_ratio": 0.01,
            "overexposure_ratio": 0.03,
            "edge_clipping_ratio": 0.01,
        },
    ]
    call_count = {"value": 0}

    def fake_compute(_data: bytes):
        index = call_count["value"]
        call_count["value"] += 1
        return metrics_by_page[index]

    monkeypatch.setattr(student_credits, "compute_image_metrics", fake_compute)
    monkeypatch.setattr(student_credits, "download_private_object", AsyncMock(return_value=_png_bytes()))
    policy = student_credits._normalise_policy({
        "semantic_judge_enabled": False,
        "min_image_width": 800,
    })

    result = await student_credits._judge_photo_source(
        db,
        {"source_ref": {"submission_id": "sub-2"}},
        policy,
    )

    assert result["decision"] == "rejected"
    assert result["target_credits"] == 0
    assert result["metrics"]["accepted_pages"] == 1
    assert result["metrics"]["page_count"] == 2
    assert result["metrics"]["pages"][0]["accepted"] is True
    assert result["metrics"]["pages"][1]["accepted"] is False
    assert "photo_submission_requires_all_pages" in result["reason_codes"]


@pytest.mark.asyncio
async def test_v2_photo_award_is_one_per_page_capped_at_ten(monkeypatch):
    pages = FakeCollection([
        {
            "submission_id": "sub-v2",
            "raw_image_ref": f"private/exampen/student-answer-copies/page-{index}.png",
            "page_number": index,
        }
        for index in range(1, 12)
    ])
    db = FakeDb({"evalpen_answer_pages": pages})
    passing_metrics = {
        "width": 1600,
        "height": 2200,
        "blur_variance": 100.0,
        "ink_density": 0.12,
        "deterministic_score": 0.9,
        "written_coverage_ratio": 0.08,
        "skew_angle": 3.0,
        "perspective_distortion": 0.02,
        "glare_ratio": 0.01,
        "overexposure_ratio": 0.03,
        "edge_clipping_ratio": 0.01,
    }
    monkeypatch.setattr(student_credits, "compute_image_metrics", lambda _data: passing_metrics)
    monkeypatch.setattr(student_credits, "download_private_object", AsyncMock(return_value=_png_bytes()))

    result = await student_credits._judge_photo_source(
        db,
        {"source_ref": {"submission_id": "sub-v2"}},
        student_credits._normalise_policy({"semantic_judge_enabled": False}),
    )

    assert result["decision"] == "accepted"
    assert result["metrics"]["accepted_pages"] == 11
    assert result["target_credits"] == 10


def test_photo_source_descriptor_is_immutable_version_one():
    first = student_credits.photo_source_descriptor({"submission_id": "sub-immutable", "student_id": "STU-1"})
    replay = student_credits.photo_source_descriptor({"submission_id": "sub-immutable", "student_id": "STU-1"})

    assert first == replay
    assert first[1] == "1"


@pytest.mark.asyncio
async def test_credit_policy_update_and_response_accept_new_image_thresholds():
    payload = credits_async.CreditPolicyUpdate(
        min_written_coverage=0.002,
        max_written_coverage=0.75,
        max_skew_angle=22.0,
        max_perspective_distortion=0.3,
        max_glare_ratio=0.2,
        max_overexposure_ratio=0.55,
        max_edge_clipping_ratio=0.15,
    )
    assert payload.min_written_coverage == 0.002
    policy = student_credits._normalise_policy({
        "min_written_coverage": payload.min_written_coverage,
        "max_written_coverage": payload.max_written_coverage,
        "max_skew_angle": payload.max_skew_angle,
        "max_perspective_distortion": payload.max_perspective_distortion,
        "max_glare_ratio": payload.max_glare_ratio,
        "max_overexposure_ratio": payload.max_overexposure_ratio,
        "max_edge_clipping_ratio": payload.max_edge_clipping_ratio,
    })
    response = await credits_async._policy_dict_to_response(policy)
    assert response["min_written_coverage"] == 0.002
    assert response["max_written_coverage"] == 0.75
    assert response["max_skew_angle"] == 22.0
    assert response["max_perspective_distortion"] == 0.3
    assert response["max_glare_ratio"] == 0.2
    assert response["max_overexposure_ratio"] == 0.55
    assert response["max_edge_clipping_ratio"] == 0.15


@pytest.mark.asyncio
async def test_credit_job_enqueue_is_idempotent_and_respects_policy_cutoff():
    from datetime import timedelta
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_enqueue_test"]
    student = {"_id": ObjectId(), "student_id": "STU-1", "username": "student1"}
    policy = await student_credits.initialize_credit_policy(db, admin_id="admin-1")

    excluded = await student_credits.enqueue_credit_job(
        db,
        db_name="credit_enqueue_test",
        admin_id="admin-1",
        student=student,
        source_type=student_credits.SOURCE_STROKE,
        source_id="canvas:STU-1:copy:MS:1",
        source_version="1",
        group_key="stroke:STU-1:copy:MS:1",
        source_ref={"user_id": "STU-1", "copy_id": "copy", "book_type": "MS", "page_number": 1},
        source_completed_at=policy["earning_started_at"] - timedelta(seconds=1),
    )
    assert excluded is None

    first = await student_credits.enqueue_credit_job(
        db,
        db_name="credit_enqueue_test",
        admin_id="admin-1",
        student=student,
        source_type=student_credits.SOURCE_STROKE,
        source_id="canvas:STU-1:copy:MS:1",
        source_version="1",
        group_key="stroke:STU-1:copy:MS:1",
        source_ref={"user_id": "STU-1", "copy_id": "copy", "book_type": "MS", "page_number": 1},
    )
    second = await student_credits.enqueue_credit_job(
        db,
        db_name="credit_enqueue_test",
        admin_id="admin-1",
        student=student,
        source_type=student_credits.SOURCE_STROKE,
        source_id="canvas:STU-1:copy:MS:1",
        source_version="1",
        group_key="stroke:STU-1:copy:MS:1",
        source_ref={"user_id": "STU-1", "copy_id": "copy", "book_type": "MS", "page_number": 1},
    )

    assert first["job_id"] == second["job_id"]
    assert await db[student_credits.JOB_COLLECTION].count_documents({}) == 1


@pytest.mark.asyncio
async def test_policy_activation_cannot_overtake_an_enqueue_snapshot(monkeypatch):
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_policy_enqueue_lock_test"]
    await student_credits.update_credit_policy(
        db,
        {
            "stroke_credits_per_unit": 4,
            "max_stroke_credits_per_page": 40,
            "daily_credit_cap": 200,
        },
        admin_id="admin-1",
        db_name="credit_policy_enqueue_lock_test",
    )
    original_initialize_policy = student_credits.initialize_credit_policy
    enqueue_has_lock = asyncio.Event()
    allow_enqueue = asyncio.Event()

    async def delayed_initialize_policy(db_value, *, admin_id: str = ""):
        if admin_id == "enqueue-admin":
            enqueue_has_lock.set()
            await allow_enqueue.wait()
        return await original_initialize_policy(db_value, admin_id=admin_id)

    monkeypatch.setattr(student_credits, "initialize_credit_policy", delayed_initialize_policy)
    student = {"_id": ObjectId(), "student_id": "STU-LOCK", "username": "student-lock"}
    enqueue_task = asyncio.create_task(
        student_credits.enqueue_credit_job(
            db,
            db_name="credit_policy_enqueue_lock_test",
            admin_id="enqueue-admin",
            student=student,
            source_type=student_credits.SOURCE_STROKE,
            source_id="canvas:STU-LOCK:copy:MS:1",
            source_version="1",
            group_key="stroke:STU-LOCK:copy:MS:1",
            source_ref={"user_id": "STU-LOCK", "copy_id": "copy", "book_type": "MS", "page_number": 1},
        )
    )

    await enqueue_has_lock.wait()
    with pytest.raises(student_credits.CreditPolicyConflictError):
        await student_credits.activate_v2_credit_policy(
            db,
            admin_id="admin-1",
            db_name="credit_policy_enqueue_lock_test",
        )

    allow_enqueue.set()
    job = await enqueue_task
    assert job["policy_snapshot"]["stroke_credits_per_unit"] == 4
    assert job["policy_version"] == 2


@pytest.mark.parametrize(
    ("total", "tier_id"),
    [(0, "seed"), (99, "seed"), (100, "scribe"), (499, "scribe"), (500, "pathfinder"),
     (1499, "pathfinder"), (1500, "beacon"), (3999, "beacon"), (4000, "luminary")],
)
def test_v2_tier_boundaries(total: int, tier_id: str):
    assert student_credits.tier_for_credits(total, student_credits.DEFAULT_TIERS)["current"]["id"] == tier_id


@pytest.mark.asyncio
async def test_ledger_commit_is_idempotent_and_awards_only_cumulative_delta():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_ledger_test"]
    await student_credits.ensure_credit_indexes(db)
    student_record_id = str(ObjectId())
    policy = student_credits._normalise_policy({})

    def job(version: str, token: str) -> Dict[str, Any]:
        return {
            "job_id": f"job-{version}",
            "admin_id": "admin-1",
            "student_record_id": student_record_id,
            "student_id": "STU-1",
            "student_username": "student1",
            "source_type": student_credits.SOURCE_STROKE,
            "source_id": "canvas:STU-1:copy:MS:1",
            "source_version": version,
            "group_key": "stroke:STU-1:copy:MS:1",
            "policy_version": 1,
            "policy_snapshot": policy,
            "lease_token": token,
        }

    result_v1 = {
        "decision": "accepted",
        "quality_score": 0.9,
        "target_credits": 20,
        "reason_codes": [],
        "metrics": {},
        "semantic": {},
    }
    first = await student_credits._commit_judgment_and_ledger(db, job("1", "token-1"), result_v1)
    replay = await student_credits._commit_judgment_and_ledger(db, job("1", "token-1"), result_v1)
    result_v2 = {**result_v1, "target_credits": 40}
    second_version = await student_credits._commit_judgment_and_ledger(
        db, job("2", "token-2"), result_v2
    )

    assert first["award_delta"] == 20
    assert replay["award_delta"] == 20
    assert second_version["award_delta"] == 20
    assert await db[student_credits.JUDGMENT_COLLECTION].count_documents({}) == 2
    assert await db[student_credits.LEDGER_COLLECTION].count_documents({}) == 2
    ledger = await db[student_credits.LEDGER_COLLECTION].find({}).to_list(length=10)
    assert sum(int(row["delta"]) for row in ledger) == 40


@pytest.mark.asyncio
async def test_student_day_lock_is_held_during_cap_calculation_and_released_on_completion(monkeypatch):
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_student_day_lock_test"]
    await student_credits.ensure_credit_indexes(db)
    monkeypatch.setattr(student_credits, "utc_now", lambda: datetime(2026, 8, 12, 10, 0, 0, tzinfo=timezone.utc))
    student_record_id = str(ObjectId())
    policy = student_credits._normalise_policy({"daily_credit_cap": 10})
    result_payload = {
        "decision": "accepted",
        "quality_score": 0.9,
        "target_credits": 10,
        "reason_codes": [],
        "metrics": {},
        "semantic": {},
    }

    def job(source_id: str, group_key: str, source_version: str, token: str) -> Dict[str, Any]:
        return {
            "job_id": f"job-{source_id}",
            "admin_id": "admin-1",
            "student_record_id": student_record_id,
            "student_id": "STU-1",
            "student_username": "student1",
            "source_type": student_credits.SOURCE_STROKE,
            "source_id": source_id,
            "source_version": source_version,
            "group_key": group_key,
            "policy_version": 1,
            "policy_snapshot": policy,
            "lease_token": token,
        }

    first_job = job(
        "canvas:STU-1:copy-1:MS:1",
        "stroke:STU-1:copy-1:MS:1",
        "1",
        "token-1",
    )
    second_job = job(
        "canvas:STU-1:copy-2:MS:1",
        "stroke:STU-1:copy-2:MS:1",
        "1",
        "token-2",
    )

    holder_acquired = asyncio.Event()
    hold_release = asyncio.Event()
    original_acquire = student_credits._acquire_student_day_lock
    student_day_lock_key = student_credits._student_day_lock_key(student_record_id, datetime(2026, 8, 12, 10, 0, 0, tzinfo=timezone.utc))

    async def acquire_with_hold(db_obj: Any, lock_key: str, token: str, *, lease_seconds: int, now: Optional[datetime] = None) -> bool:
        acquired = await original_acquire(db_obj, lock_key, token, lease_seconds=lease_seconds, now=now)
        if acquired and token == "token-1":
            holder_acquired.set()
            await hold_release.wait()
        return acquired

    monkeypatch.setattr(student_credits, "_acquire_student_day_lock", acquire_with_hold)

    async def run_commit(job_value: Dict[str, Any]):
        try:
            result = await student_credits._commit_judgment_and_ledger(db, job_value, result_payload)
            return ("ok", result)
        except RuntimeError as exc:
            return ("err", str(exc))

    first_task = asyncio.create_task(run_commit(first_job))
    await holder_acquired.wait()
    second_task = asyncio.create_task(run_commit(second_job))
    second_status, second_error = await second_task
    assert second_status == "err"
    assert "student daily credit lock is busy" in second_error

    hold_release.set()
    first_status, first_result = await first_task
    assert first_status == "ok"
    assert first_result["award_delta"] == 10

    retry = await student_credits._commit_judgment_and_ledger(db, second_job, result_payload)
    assert retry["award_delta"] == 0

    student_locks = await db[student_credits.LOCK_COLLECTION].find({"_id": {"$in": [student_day_lock_key, "stroke:STU-1:copy-1:MS:1", "stroke:STU-1:copy-2:MS:1"]}}).to_list(length=10)
    assert len(student_locks) == 3
    assert all(lock["lease_token"] is None for lock in student_locks)

    ledger = await db[student_credits.LEDGER_COLLECTION].find({"student_record_id": student_record_id}).to_list(length=10)
    assert sum(int(row["delta"]) for row in ledger) == 10


@pytest.mark.asyncio
async def test_reconcile_credit_jobs_returns_repaired_zero_when_disabled():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_reconcile_disabled_test"]
    await student_credits.update_credit_policy(db, {"enabled": False}, admin_id="admin-1")
    stale = datetime.now(timezone.utc) - timedelta(minutes=5)
    await db[student_credits.JOB_COLLECTION].insert_one({
        "job_id": "stale-processing-job",
        "status": "processing",
        "lease_expires_at": stale - timedelta(minutes=2),
        "next_attempt_at": stale - timedelta(minutes=3),
    })
    result = await student_credits.reconcile_credit_jobs(db, db_name="credit_reconcile_disabled_test", dispatch=True)
    assert result == {"stale_recovered": 1, "dispatched": 0, "repaired": 0}


@pytest.mark.asyncio
async def test_leaderboard_uses_accepted_count_as_equal_credit_tiebreaker():
    from datetime import datetime, timedelta, timezone
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_leaderboard_test"]
    first_id = ObjectId()
    second_id = ObjectId()
    await db["students"].insert_many(
        [
            {"_id": first_id, "student_id": "STU-1", "first_name": "First"},
            {"_id": second_id, "student_id": "STU-2", "first_name": "Second"},
        ]
    )
    now = datetime.now(timezone.utc)
    await db[student_credits.LEDGER_COLLECTION].insert_many(
        [
            {"judgment_key": "first-1", "student_record_id": str(first_id), "delta": 10, "created_at": now},
            {"judgment_key": "second-1", "student_record_id": str(second_id), "delta": 10, "created_at": now - timedelta(seconds=5)},
        ]
    )
    await db[student_credits.JUDGMENT_COLLECTION].insert_many(
        [
            {"student_record_id": str(first_id), "decision": "accepted", "source_type": "stroke_page", "source_id": "first-1", "source_version": "1"},
            {"student_record_id": str(first_id), "decision": "accepted", "source_type": "stroke_page", "source_id": "first-2", "source_version": "1"},
            {"student_record_id": str(second_id), "decision": "accepted", "source_type": "stroke_page", "source_id": "second-1", "source_version": "1"},
        ]
    )

    rows = await student_credits.get_credit_leaderboard(db, limit=10)

    assert [row["student_id"] for row in rows] == ["STU-1", "STU-2"]
    assert [row["rank"] for row in rows] == [1, 2]


@pytest.mark.asyncio
async def test_default_credit_policy_is_v2_for_new_tenants():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_v2_default_policy_test"]
    policy = await student_credits.get_credit_policy(db, admin_id="admin-1")

    assert policy["stroke_mm_per_credit_unit"] == 250.0
    assert policy["stroke_credits_per_unit"] == 1
    assert policy["max_stroke_credits_per_page"] == 5
    assert policy["image_credits_per_page"] == 1
    assert policy["max_image_credits_per_submission"] == 10
    assert policy["daily_credit_cap"] == 100
    assert [tier["id"] for tier in policy["tiers"]] == ["seed", "scribe", "pathfinder", "beacon", "luminary"]
    assert [tier["min_credits"] for tier in policy["tiers"]] == [0, 100, 500, 1500, 4000]
    assert await db[student_credits.POLICY_COLLECTION].count_documents({}) == 0


@pytest.mark.asyncio
async def test_credit_summary_and_leaderboard_do_not_initialize_policy():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_read_only_policy_test"]
    student = {"_id": ObjectId(), "student_id": "STU-READ", "username": "reader"}

    summary = await student_credits.get_student_credit_summary(db, student)
    leaderboard = await student_credits.get_credit_leaderboard(db, limit=10)

    assert summary["policy_version"] == 1
    assert leaderboard == []
    assert await db[student_credits.POLICY_COLLECTION].count_documents({}) == 0


@pytest.mark.asyncio
async def test_update_credit_policy_rejects_invalid_tiers():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_invalid_policy_test"]
    await student_credits.ensure_credit_indexes(db)

    with pytest.raises(student_credits.CreditPolicyValidationError):
        await student_credits.update_credit_policy(
            db,
            {
                "tiers": [
                    {"id": "seed", "name": "Seed", "min_credits": 0, "accent": "#22A06B", "icon": "sprout"},
                    {"id": "seed", "name": "Duplicate", "min_credits": 100, "accent": "#2563EB", "icon": "pen-tool"},
                ],
                "min_written_coverage": 0.01,
                "max_written_coverage": 0.0,
            },
            admin_id="admin-1",
            db_name="credit_invalid_policy_test",
        )

    with pytest.raises(student_credits.CreditPolicyValidationError, match="strictly increasing order"):
        await student_credits.update_credit_policy(
            db,
            {
                "tiers": [
                    {"id": "seed", "name": "Seed", "min_credits": 0, "accent": "#22A06B", "icon": "sprout"},
                    {"id": "pathfinder", "name": "Pathfinder", "min_credits": 500, "accent": "#7C3AED", "icon": "compass"},
                    {"id": "scribe", "name": "Scribe", "min_credits": 100, "accent": "#2563EB", "icon": "pen-tool"},
                ]
            },
            admin_id="admin-1",
            db_name="credit_invalid_policy_test",
        )


@pytest.mark.asyncio
async def test_activate_v2_credit_policy_enforces_open_job_and_daily_cap_guards():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_activate_guard_test"]
    now = datetime.now(timezone.utc)
    await student_credits.update_credit_policy(
        db,
        {
            "stroke_acceptance_threshold": 0.83,
            "stroke_mm_per_credit_unit": 400.0,
            "stroke_credits_per_unit": 4,
            "max_stroke_credits_per_page": 40,
            "image_credits_per_page": 2,
            "max_image_credits_per_submission": 20,
            "daily_credit_cap": 200,
        },
        admin_id="admin-1",
        db_name="credit_activate_guard_test",
    )

    await db[student_credits.JOB_COLLECTION].insert_one(
        {
            "job_id": "open-job",
            "status": "pending",
            "attempts": 0,
            "source_type": student_credits.SOURCE_STROKE,
            "source_id": "canvas:user:copy:MS:1",
            "source_version": "1",
            "next_attempt_at": now - timedelta(minutes=1),
        }
    )

    with pytest.raises(student_credits.CreditPolicyConflictError):
        await student_credits.activate_v2_credit_policy(db, admin_id="admin-1", db_name="credit_activate_guard_test")

    await db[student_credits.JOB_COLLECTION].delete_many({})
    await db[student_credits.LEDGER_COLLECTION].insert_one(
        {
            "student_record_id": "stu-1",
            "delta": 150,
            "created_at": now,
        }
    )

    with pytest.raises(student_credits.CreditPolicyConflictError):
        await student_credits.activate_v2_credit_policy(db, admin_id="admin-1", db_name="credit_activate_guard_test")

    await db[student_credits.LEDGER_COLLECTION].delete_many({})
    await db[student_credits.LEDGER_COLLECTION].insert_one(
        {
            "student_record_id": "stu-historical",
            "delta": 500,
            "created_at": now - timedelta(days=1),
        }
    )
    policy, activated = await student_credits.activate_v2_credit_policy(db, admin_id="admin-1", db_name="credit_activate_guard_test")
    assert activated is True
    assert student_credits._is_v2_preset(policy)
    assert policy["stroke_acceptance_threshold"] == 0.83
    historical = await db[student_credits.LEDGER_COLLECTION].find_one({"student_record_id": "stu-historical"})
    assert historical["delta"] == 500
    recheck_policy, recheck_activated = await student_credits.activate_v2_credit_policy(
        db,
        admin_id="admin-1",
        db_name="credit_activate_guard_test",
    )
    assert recheck_activated is False
    assert recheck_policy["version"] == policy["version"]


@pytest.mark.asyncio
async def test_reconcile_missing_completion_becomes_terminal_after_bound_lookups(monkeypatch):
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_missing_completion_test"]
    monkeypatch.setattr(student_credits, "_reconcile_canvas_sources", AsyncMock(return_value=0))
    monkeypatch.setattr(student_credits, "_reconcile_photo_sources", AsyncMock(return_value=0))
    await student_credits.get_credit_policy(db, admin_id="admin-1")
    now = datetime.now(timezone.utc)
    await db[student_credits.JOB_COLLECTION].insert_one(
        {
            "job_id": "missing-completion-job",
            "status": "pending",
            "attempts": 0,
            "source_type": student_credits.SOURCE_STROKE,
            "source_id": "canvas:stu:copy:MS:1",
            "source_version": "1",
            "source_ref": {"user_id": "stu", "copy_id": "copy", "book_type": "MS", "page_number": 1},
            "group_key": "stroke:stu:copy:MS:1",
            "next_attempt_at": now - timedelta(minutes=1),
            "created_at": now - timedelta(minutes=1),
            "updated_at": now - timedelta(minutes=1),
        }
    )

    for _ in range(student_credits.MISSING_COMPLETION_LOOKUPS):
        result = await student_credits.reconcile_credit_jobs(db, db_name="credit_missing_completion_test")
        assert result["dispatched"] == 0
        assert result["repaired"] == 0

    job = await db[student_credits.JOB_COLLECTION].find_one({"job_id": "missing-completion-job"})
    assert job["status"] == "failed"
    assert job["decision"] == "missing_completion"
    assert job["last_error"] == student_credits._missing_completion_reason()
    assert job["missing_completion_lookups"] >= student_credits.MISSING_COMPLETION_LOOKUPS


@pytest.mark.asyncio
async def test_student_credit_summary_includes_policy_version_and_full_tiers():
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["credit_summary_payload_test"]
    now = datetime(2026, 8, 12, 10, 0, 0, tzinfo=timezone.utc)
    student = {"_id": ObjectId(), "student_id": "STU-10", "admin_id": "admin-1"}
    await db["students"].insert_one(student)
    await db[student_credits.LEDGER_COLLECTION].insert_many(
        [
            {"judgment_key": "summary-1", "student_record_id": str(student["_id"]), "delta": 120, "created_at": now},
            {"judgment_key": "summary-2", "student_record_id": str(student["_id"]), "delta": 30, "created_at": now},
        ]
    )

    summary = await student_credits.get_student_credit_summary(db, student)

    assert summary["total_credits"] == 150
    assert summary["policy_version"] == 1
    assert [tier["id"] for tier in summary["tiers"]] == ["seed", "scribe", "pathfinder", "beacon", "luminary"]
    assert [tier["min_credits"] for tier in summary["tiers"]] == [0, 100, 500, 1500, 4000]


@pytest.mark.asyncio
async def test_activate_v2_policy_endpoint_maps_conflict_to_http_409(monkeypatch):
    from mongomock_motor import AsyncMongoMockClient

    tenant_db = AsyncMongoMockClient()["credit_activate_endpoint_test"]
    policy = await student_credits.get_credit_policy(tenant_db, admin_id="admin-1")

    async def _fake_activate(_db, *, admin_id: str, db_name: str):
        raise student_credits.CreditPolicyConflictError("tenant conflict")

    monkeypatch.setattr(credits_async.student_credits, "activate_v2_credit_policy", _fake_activate)

    with pytest.raises(HTTPException) as exc:
        await credits_async.activate_v2_credit_policy(
            _request(),
            current_user={"user_type": "admin", "user_id": "admin-1", "db_name": "credit_activate_endpoint_test"},
            db=_FakeDbManager(tenant_db),
        )
    assert exc.value.status_code == 409

    async def _fake_activate_success(_db, *, admin_id: str, db_name: str):
        return policy, True

    monkeypatch.setattr(credits_async.student_credits, "activate_v2_credit_policy", _fake_activate_success)
    response = await credits_async.activate_v2_credit_policy(
        _request(),
        current_user={"user_type": "admin", "user_id": "admin-1", "db_name": "credit_activate_endpoint_test"},
        db=_FakeDbManager(tenant_db),
    )
    assert response.activated is True
    assert response.data.version == 1
