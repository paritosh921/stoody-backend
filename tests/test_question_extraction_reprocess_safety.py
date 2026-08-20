from types import SimpleNamespace

import pytest
from starlette.requests import Request


class _FakeDocumentDb:
    def __init__(self):
        self.updates = []

    async def mongo_find_one(self, collection, query):
        assert collection == "documents"
        return {
            "document_id": query["document_id"],
            "document_type": "Test Series",
            "filename": "english.pdf",
            "file_path": "s3://private/english.pdf",
            "ocr_status": "completed",
        }

    async def mongo_update_one(self, collection, query, update):
        self.updates.append((collection, query, update))
        return True

    async def mongo_delete_many(self, *_args, **_kwargs):
        raise AssertionError("reprocess start must not delete the current extraction")


class _FakeCache:
    async def set(self, *_args, **_kwargs):
        return True


@pytest.mark.asyncio
async def test_reprocess_keeps_current_questions_until_replacement_pipeline_succeeds(monkeypatch):
    from api.v1 import pdf_async

    db = _FakeDocumentDb()
    app = SimpleNamespace(state=SimpleNamespace(ocr_semaphore=None, ocr_tasks={}))
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/documents/doc-1/process-ocr",
        "headers": [],
        "app": app,
    })

    async def _download(_path):
        return b"valid-pdf-bytes"

    async def _pipeline(**kwargs):
        result = dict(kwargs["processing_result"])
        result.update({"status": "completed", "progress": 100})
        return pdf_async.PDFProcessingResult(**result)

    monkeypatch.setattr(pdf_async, "download_file", _download)
    monkeypatch.setattr(pdf_async, "run_document_ocr_pipeline", _pipeline)

    handler = getattr(pdf_async.process_document_ocr, "__wrapped__", pdf_async.process_document_ocr)
    result = await handler(
        request=request,
        document_id="doc-1",
        async_mode=False,
        current_user={"user_id": "admin-1", "user_type": "admin"},
        db=db,
        cache=_FakeCache(),
    )

    assert result.status == "completed"
    assert db.updates[0][2]["$set"]["ocr_status"] == "processing"
    assert db.updates[0][2]["$set"]["ocr_error"] is None
