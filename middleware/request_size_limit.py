"""ASGI request body limit middleware for upload routes."""

from __future__ import annotations

import json
from typing import Awaitable, Callable

from core.upload_security.routes import resolve_upload_policy_for_route


class RequestBodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(
        self,
        app,
        *,
        default_max_body_bytes: int = 64 * 1024 * 1024,
        enabled: bool = True,
    ) -> None:
        self.app = app
        self.default_max_body_bytes = default_max_body_bytes
        self.enabled = enabled

    async def __call__(self, scope, receive: Callable[[], Awaitable[dict]], send: Callable[[dict], Awaitable[None]]):
        if not self.enabled or scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        path = scope.get("path", "")
        route_policy = resolve_upload_policy_for_route(method, path)
        max_body_bytes = route_policy.request_limit_bytes if route_policy else self.default_max_body_bytes

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length.decode("ascii")) > max_body_bytes:
                    await self._send_413(send, max_body_bytes)
                    return
            except ValueError:
                await self._send_413(send, max_body_bytes)
                return

        consumed = 0

        async def limited_receive() -> dict:
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body") or b"")
                if consumed > max_body_bytes:
                    raise RequestBodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._send_413(send, max_body_bytes)

    @staticmethod
    async def _send_413(send: Callable[[dict], Awaitable[None]], max_body_bytes: int) -> None:
        body = json.dumps(
            {
                "detail": "Request body too large",
                "max_size_bytes": max_body_bytes,
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
