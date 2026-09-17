from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.v1.practice_async import EvaluateRequest, QuestionPageRefsModel
from api.v1.strokes_async import _merge_stroke_docs, _persist_canvas_page, CanvasPageUpsert
from services import practice_stroke_evidence as evidence
from services import practice_answer_lifecycle as lifecycle


def matches(doc, query):
    for key, value in query.items():
        if key == "$or":
            if not any(matches(doc, clause) for clause in value):
                return False
            continue
        actual = doc.get(key)
        if isinstance(value, dict):
            if "$in" in value and actual not in value["$in"]:
                return False
            if "$lt" in value and (actual is None or actual >= value["$lt"]):
                return False
            if "$elemMatch" in value and not any(matches(item, value["$elemMatch"]) for item in actual or []):
                return False
        elif actual != value:
            return False
    return True


class Cursor:
    def __init__(self, docs):
        self.docs = deepcopy(docs)

    def sort(self, *args):
        return self

    async def to_list(self, length):
        return self.docs[:length]


class Collection:
    def __init__(self, docs=()):
        self.docs = deepcopy(list(docs))
        self.interleave = None

    def find(self, query):
        return Cursor([d for d in self.docs if matches(d, query)])

    async def find_one(self, query):
        return deepcopy(next((d for d in self.docs if matches(d, query)), None))

    async def insert_one(self, doc):
        self.docs.append({"_id": str(len(self.docs)), **deepcopy(doc)})

    async def update_one(self, query, update, upsert=False):
        doc = next((d for d in self.docs if matches(d, query)), None)
        if doc is None and upsert:
            doc = {**query, **deepcopy(update.get("$setOnInsert", {}))}
            self.docs.append(doc)
        if doc is not None:
            doc.update(deepcopy(update.get("$set", {})))
            for key in update.get("$unset", {}):
                doc.pop(key, None)
        return SimpleNamespace(matched_count=int(doc is not None))

    async def find_one_and_update(self, query, update, upsert=False, **kwargs):
        await self.update_one(query, update, upsert)
        return await self.find_one({"_id": query["_id"]}) if upsert else next(
            (deepcopy(d) for d in self.docs if d.get("lease") == update.get("$set", {}).get("lease")), None)

    async def replace_one(self, query, doc):
        if self.interleave:
            callback, self.interleave = self.interleave, None
            callback(self.docs)
        for i, old in enumerate(self.docs):
            if matches(old, query):
                self.docs[i] = {"_id": old["_id"], **deepcopy(doc)}
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)


def stroke(id="s1", owner="attempt-1", question="q1", **kw):
    return {"id": id, "points": [[1, 2, 0, 128, .5, 0]],
            "processingVersion": "ble-canonical-v1", "sourceMode": "live",
            "practiceSessionId": owner, "questionId": question, "timestamp": 150, **kw}


def page(number, strokes, book="LS", user="alice"):
    return {"_id": f"{book}-{number}", "user_id": user, "copy_id": "copy-1", "book_type": book,
            "page_number": number, "version": 1, "strokes": strokes}


@pytest.fixture
def setup(monkeypatch):
    database = {name: Collection() for name in ["canvas_pages", "practice_drafts", "practice_answer_snapshots", "practice_answer_evaluations", "practice_attempts"]}
    for collection in database.values():
        collection.database = database

    async def collection(*args):
        return database["canvas_pages"]

    monkeypatch.setattr(evidence, "_canvas_collection", collection)
    monkeypatch.setattr(lifecycle, "_canvas_collection", collection)
    monkeypatch.setattr(evidence, "render_stroke_page", lambda value: repr(value).encode())
    return database, {"user_id": "alice", "db_name": "tenant-1"}


def refs(**kwargs):
    return QuestionPageRefsModel(copyId="copy-1", practiceSessionId="attempt-1", questionId="q1", **kwargs)


@pytest.mark.asyncio
async def test_cache_absent_server_pages_recovered_and_foreign_page_excluded(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(17, [stroke()]), page(32, [stroke("other", question="q2")]), page(36, [stroke("s2")])]
    prepared = await lifecycle.prepare_answer(user, None, refs(legacyPageNumbers=[17, 32, 36]))
    assert prepared["questionPageRefs"]["activePages"] == [17, 36]
    snapshot = await lifecycle.load_snapshot(user, None, prepared["snapshotId"], "q1", "attempt-1")
    assert len(snapshot["images"]) == 2


@pytest.mark.asyncio
async def test_full_identity_preserves_same_page_number_in_two_books(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(7, [stroke("a")], "LS"), page(7, [stroke("b")], "MS")]
    prepared = await lifecycle.prepare_answer(user, None, refs())
    assert {p["bookType"] for p in prepared["questionPageRefs"]["virtualPages"]} == {"LS", "MS"}
    database["canvas_pages"].docs.reverse()
    repeated = await lifecycle.prepare_answer(user, None, refs())
    assert repeated["snapshotId"] == prepared["snapshotId"]


@pytest.mark.asyncio
async def test_missing_expected_stroke_prevents_partial_submission(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(17, [stroke()])]
    with pytest.raises(evidence.PracticeStrokeEvidenceError, match="unacknowledged"):
        await lifecycle.prepare_answer(user, None, refs(virtualPages=[{"physicalPageNo": 17, "bookType": "LS", "expectedStrokeIds": ["s1", "not-uploaded"]}]))
    assert not database["practice_answer_snapshots"].docs


@pytest.mark.asyncio
async def test_missing_legacy_page_requires_reconciliation(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(17, [stroke()])]
    with pytest.raises(evidence.PracticeStrokeEvidenceError, match="Legacy page 32"):
        await lifecycle.prepare_answer(user, None, refs(legacyPageNumbers=[32]))


@pytest.mark.asyncio
async def test_ambiguous_legacy_book_does_not_guess(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(7, [stroke("a")], "LS"), page(7, [stroke("b")], "MS")]
    with pytest.raises(evidence.PracticeStrokeEvidenceError, match="multiple books"):
        await lifecycle.prepare_answer(user, None, refs(legacyPageNumbers=[7]))


@pytest.mark.asyncio
async def test_mixed_tagged_and_legacy_page_preserves_all_question_visit_windows(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(17, [
        stroke("tagged"), stroke("legacy-first", owner=None, question=None, timestamp=110),
        stroke("legacy-return", owner=None, question=None, timestamp=310),
        stroke("between-visits", owner=None, question=None, timestamp=250),
        stroke("different-question", question="q2", timestamp=110),
        stroke("different-attempt", owner="attempt-2", timestamp=110),
    ])]
    prepared = await lifecycle.prepare_answer(user, None, refs(
        legacyPageNumbers=[17], timeIntervals=[{"startTs": 100, "endTs": 200}, {"startTs": 300, "endTs": 400}],
    ))
    assert prepared["questionPageRefs"]["virtualPages"][0]["expectedStrokeIds"] == ["legacy-first", "legacy-return", "tagged"]
    repeated = await lifecycle.prepare_answer(user, None, QuestionPageRefsModel(**prepared["questionPageRefs"]))
    assert repeated["snapshotId"] == prepared["snapshotId"]


@pytest.mark.asyncio
async def test_random_pages_and_return_to_same_question_keep_old_and_new_owned_ink(setup):
    database, user = setup
    database["canvas_pages"].docs = [
        page(36, [stroke("continuation")]),
        page(17, [stroke("first-visit"), stroke("later-visit"), stroke("foreign", question="q2")]),
        page(32, [stroke("middle")]),
    ]
    prepared = await lifecycle.prepare_answer(user, None, refs())
    assert [(p["physicalPageNo"], p["expectedStrokeIds"]) for p in prepared["questionPageRefs"]["virtualPages"]] == [
        (17, ["first-visit", "later-visit"]), (32, ["middle"]), (36, ["continuation"]),
    ]


@pytest.mark.asyncio
async def test_snapshot_is_immutable_and_scoped_to_user_question_attempt(setup):
    database, user = setup
    database["canvas_pages"].docs = [page(17, [stroke()])]
    first = await lifecycle.prepare_answer(user, None, refs())
    original = await lifecycle.load_snapshot(user, None, first["snapshotId"], "q1", "attempt-1")
    database["canvas_pages"].docs[0]["strokes"].append(stroke("later"))
    second = await lifecycle.prepare_answer(user, None, refs())
    assert first["snapshotId"] != second["snapshotId"]
    assert (await lifecycle.load_snapshot(user, None, first["snapshotId"], "q1", "attempt-1"))["images"] == original["images"]
    for bad_user, question, attempt in [({"user_id": "bob"}, "q1", "attempt-1"), (user, "q2", "attempt-1"), (user, "q1", "attempt-2")]:
        with pytest.raises(HTTPException):
            await lifecycle.load_snapshot(bad_user, None, first["snapshotId"], question, attempt)


def test_duplicate_strokes_enrich_ownership_without_replacing_geometry():
    original = stroke(owner=None, question=None)
    incoming = stroke(points=[[999, 999]])
    merged, changed = _merge_stroke_docs([original], [incoming])
    assert changed == 1
    assert merged[0]["points"] == original["points"]
    assert merged[0]["practiceSessionId"] == "attempt-1"
    assert _merge_stroke_docs(merged, [original])[0] == merged
    with pytest.raises(HTTPException, match=""):
        _merge_stroke_docs(merged, [stroke(question="q2")])


def test_null_start_uses_timestamp_but_explicit_foreign_owner_never_does():
    args = dict(practice_session_id="attempt-1", question_id="q1", ordinal=None, start_ts=100, end_ts=200)
    assert evidence._stroke_matches_scope(stroke(owner=None, question=None, startedAt=None), **args)
    assert not evidence._stroke_matches_scope(stroke(question="q2", startedAt=None), **args)


@pytest.mark.asyncio
async def test_concurrent_page_save_retries_and_preserves_both_writers():
    coll = Collection([page(17, [stroke("old")])])
    def race(docs):
        docs[0]["strokes"].append(stroke("agent"))
        docs[0]["version"] += 1
    coll.interleave = race
    incoming = CanvasPageUpsert(book_type="LS", page_number=17, strokes=[stroke("browser", pageNumber=17, bookType="LS", startedAt=150, endedAt=160)])
    saved, changed, created = await _persist_canvas_page(coll, {"_id": "LS-17"}, "alice", None, incoming, datetime.now(timezone.utc), "copy-1")
    assert changed and not created
    assert {s["id"] for s in saved["strokes"]} == {"old", "agent", "browser"}
    assert saved["version"] == 3


@pytest.mark.asyncio
async def test_repeated_submit_reuses_result_and_concurrent_submit_is_rejected(setup):
    _, user = setup
    request = EvaluateRequest(questionId="q1", sessionId="attempt-1", snapshotId="snapshot")
    claim = await lifecycle.begin_evaluation(user, None, request)
    with pytest.raises(HTTPException) as caught:
        await lifecycle.begin_evaluation(user, None, request)
    assert caught.value.status_code == 409
    await lifecycle.finish_evaluation(claim, {"success": True, "evaluation": {"score": 1}})
    request.timeSpent = 999
    again = await lifecycle.begin_evaluation(user, None, request)
    assert again[3]["evaluation"]["score"] == 1
    request.answerText = "new answer"
    assert (await lifecycle.begin_evaluation(user, None, request))[2]


@pytest.mark.asyncio
async def test_failed_evaluation_can_retry_same_snapshot(setup):
    _, user = setup
    request = EvaluateRequest(questionId="q1", snapshotId="snapshot")
    claim = await lifecycle.begin_evaluation(user, None, request)
    await lifecycle.finish_evaluation(claim)
    retry = await lifecycle.begin_evaluation(user, None, request)
    assert retry[2] != claim[2]


@pytest.mark.asyncio
async def test_server_attempt_survives_missing_browser_session(setup):
    _, user = setup
    first = await lifecycle.resolve_attempt(user, None, "rotation", "original")
    again = await lifecycle.resolve_attempt(user, None, "rotation", "new-browser-generated-id")
    assert first == again == "original"


@pytest.mark.asyncio
async def test_device_with_pending_ink_resumes_its_original_attempt(setup):
    _, user = setup
    await lifecycle.resolve_attempt(user, None, "rotation", "other-device")
    assert await lifecycle.resolve_attempt(user, None, "rotation", "pending-device", True) == "pending-device"


@pytest.mark.asyncio
async def test_attachment_only_answer_can_prepare_without_inventing_notebook_evidence(setup):
    _, user = setup
    prepared = await lifecycle.prepare_answer(user, None, refs(allowEmptyAnswer=True))
    snapshot = await lifecycle.load_snapshot(user, None, prepared["snapshotId"], "q1", "attempt-1")
    assert snapshot["images"] == []
    with pytest.raises(evidence.PracticeStrokeEvidenceError):
        await lifecycle.prepare_answer(user, None, refs())


@pytest.mark.asyncio
async def test_history_projection_failure_does_not_repeat_evaluation(setup, monkeypatch):
    database, user = setup
    request = EvaluateRequest(questionId="q1", snapshotId="snapshot")
    claim = await lifecycle.begin_evaluation(user, None, request)
    real_update = database["practice_attempts"].update_one
    async def fail(*args, **kwargs):
        raise RuntimeError("temporary database failure")
    monkeypatch.setattr(database["practice_attempts"], "update_one", fail)
    with pytest.raises(RuntimeError):
        await lifecycle.finish_evaluation(claim, {"success": True, "evaluation": {"score": 1}}, {"question_id": "q1"})
    await lifecycle.finish_evaluation(claim)  # exception cleanup must not erase the committed result
    monkeypatch.setattr(database["practice_attempts"], "update_one", real_update)
    retry = await lifecycle.begin_evaluation(user, None, request)
    assert retry[3]["evaluation"]["score"] == 1
    assert len(database["practice_attempts"].docs) == 1


@pytest.mark.asyncio
async def test_prepare_and_evaluate_http_contract_reuses_owned_snapshot(setup, monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from api.v1 import practice_async as api

    database, user = setup
    database["canvas_pages"].docs = [page(17, [stroke()])]
    app = FastAPI()
    app.include_router(api.router, prefix="/api/v1/practice")
    app.dependency_overrides[api.get_current_user] = lambda: user
    app.dependency_overrides[api.get_database] = lambda: None
    monkeypatch.setattr(api.limiter, "enabled", False)

    async def question(*args, **kwargs):
        return {"question_text": "Test question", "correct_answer": "1", "question_type": "subjective"}

    async def no_images(*args, **kwargs):
        return []

    monkeypatch.setattr(api, "_load_question_doc", question)
    monkeypatch.setattr(api, "_figure_images_base64", no_images)
    monkeypatch.setattr(api, "_option_images_base64", no_images)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        prepared = await client.post("/api/v1/practice/answers/prepare", json=refs().model_dump())
        assert prepared.status_code == 200, prepared.text
        body = {"questionId": "q1", "sessionId": "attempt-1", "snapshotId": prepared.json()["snapshotId"]}
        claim = await lifecycle.begin_evaluation(user, None, EvaluateRequest(**body))
        await lifecycle.finish_evaluation(claim, {"success": True, "evaluation": {"score": .75}})
        # Later page changes cannot alter the prepared answer or force a new
        # evaluation. The HTTP request must retain the snapshotId field.
        database["canvas_pages"].docs.clear()
        response = await client.post("/api/v1/practice/evaluate", json=body)
        assert response.status_code == 200, response.text
        assert response.json()["evaluation"]["score"] == .75
        body["sessionId"] = "another-attempt"
        rejected = await client.post("/api/v1/practice/evaluate", json=body)
        assert rejected.status_code == 409
