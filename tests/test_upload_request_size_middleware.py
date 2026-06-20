import pytest

from fastapi import FastAPI, Request

from core.upload_security.routes import resolve_upload_policy_for_route, UPLOAD_ROUTE_POLICY_MAP
from middleware.request_size_limit import RequestSizeLimitMiddleware


def test_route_policy_map_covers_specific_upload_paths():
    assert resolve_upload_policy_for_route("POST", "/api/v1/debugger/upload").policy_id == "debugger_document"
    assert resolve_upload_policy_for_route("POST", "/api/debugger/upload").policy_id == "debugger_document"
    assert (
        resolve_upload_policy_for_route("POST", "/api/v1/ingest/strokes/exam-1/AA:BB/complete").policy_id
        == "hub_stroke_finalize"
    )
    assert (
        resolve_upload_policy_for_route("POST", "/api/v1/ingest/strokes/exam-1/AA:BB").policy_id
        == "hub_stroke_chunk"
    )
    assert (
        resolve_upload_policy_for_route("POST", "/api/v1/hubs/hub-1/data/upload").policy_id
        == "hub_raw_data_batch"
    )
    assert resolve_upload_policy_for_route("POST", "/api/v1/hubs/hub-1/commands/pending") is None


def test_route_policy_map_has_unique_method_path_pairs():
    keys = [(entry.method, entry.path_template) for entry in UPLOAD_ROUTE_POLICY_MAP]
    assert len(keys) == len(set(keys))


def make_app(max_body_bytes=8):
    app = FastAPI()
    app.add_middleware(RequestSizeLimitMiddleware, default_max_body_bytes=max_body_bytes)

    @app.post("/limited")
    async def limited(request: Request):
        return {"size": len(await request.body())}

    return app


async def call_app(app, body_chunks, headers=None):
    messages = [
        {
            "type": "http.request",
            "body": body,
            "more_body": index < len(body_chunks) - 1,
        }
        for index, body in enumerate(body_chunks)
    ]
    sent = []

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/limited",
            "headers": headers or [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "http_version": "1.1",
        },
        receive,
        send,
    )
    return sent


@pytest.mark.asyncio
async def test_content_length_rejection_before_handler():
    sent = await call_app(
        make_app(max_body_bytes=4),
        [b"abcde"],
        headers=[(b"content-length", b"5")],
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413


@pytest.mark.asyncio
async def test_allowed_body_passes_through():
    sent = await call_app(
        make_app(max_body_bytes=8),
        [b"abcd"],
        headers=[(b"content-length", b"4")],
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    assert start["status"] == 200
    assert body == b'{"size":4}'


@pytest.mark.asyncio
async def test_chunked_rejection_without_content_length():
    sent = await call_app(make_app(max_body_bytes=4), [b"abc", b"de"])

    start = next(message for message in sent if message["type"] == "http.response.start")
    assert start["status"] == 413
