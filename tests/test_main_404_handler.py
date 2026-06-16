import asyncio
import json
from types import SimpleNamespace

from fastapi import HTTPException


def test_404_handler_preserves_domain_error_message():
    from main_async import not_found_handler

    response = asyncio.run(
        not_found_handler(
            SimpleNamespace(url=SimpleNamespace(path="/api/v1/auth/student/password-reset/request")),
            HTTPException(status_code=404, detail="No records found"),
        )
    )

    body = json.loads(response.body)
    assert body["message"] == "No records found"
    assert body["error"] == "Not found"


def test_404_handler_uses_endpoint_message_for_missing_routes():
    from main_async import not_found_handler

    response = asyncio.run(
        not_found_handler(
            SimpleNamespace(url=SimpleNamespace(path="/api/v1/missing")),
            HTTPException(status_code=404, detail="Not Found"),
        )
    )

    body = json.loads(response.body)
    assert body["message"] == "The requested endpoint does not exist"
