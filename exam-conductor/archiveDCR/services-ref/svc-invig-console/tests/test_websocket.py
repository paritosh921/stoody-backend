"""Integration tests for the WebSocket endpoint.

Test IDs: I-INVIG-01 through I-INVIG-06
Validation level: L4 (integration test with TestClient, mocked infra)

Uses FastAPI TestClient with mocked JWKS, NATS, and exam-orch to verify
WebSocket authentication, subscription, message format, and disconnect.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.domain.status_aggregator import (
    DashboardSnapshot,
    WifiStatus,
    SyncProgress,
    UploadProgress,
)
from src.events.hub_relay import HubRelay
from src.routes.sessions import router as sessions_router
from src.routes.websocket import router as ws_router


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_MOCK_USER_CLAIMS = {
    "sub": "user-123",
    "tenant_id": "tenant-abc",
    "role": "tutor",
    "name": "Test Invigilator",
    "email": "test@example.com",
}


def _build_test_app(
    hub_relay: HubRelay | None = None,
    exam_client: Any = None,
) -> FastAPI:
    """Build a minimal test app with mocked state."""
    app = FastAPI()
    app.include_router(sessions_router, prefix="/api/v1/invigilator")
    app.include_router(ws_router, prefix="/api/v1/invigilator")

    # Mock hub relay
    if hub_relay is None:
        hub_relay = MagicMock(spec=HubRelay)
        hub_relay.get_latest = MagicMock(return_value={})
        hub_relay.register_listener = AsyncMock()
        hub_relay.unregister_listener = AsyncMock()

    # Mock exam client
    if exam_client is None:
        exam_client = AsyncMock()
        exam_client.get_session = AsyncMock(return_value={
            "exam_id": "exam-001",
            "state": "in_progress",
            "timer_remaining_sec": 3600,
            "upload_status": "pending",
        })
        exam_client.list_active_sessions = AsyncMock(return_value=[])

    app.state.hub_relay = hub_relay
    app.state.exam_client = exam_client
    return app


def _mock_validate_token(token: str) -> Any:
    """Return a mock ExamPenUser for any token starting with 'valid-'."""
    if not token.startswith("valid-"):
        raise Exception("Invalid token")

    from exampen_common.auth import ExamPenUser
    return ExamPenUser(
        user_id="user-123",
        tenant_id="tenant-abc",
        stoody_role="tutor",
        exampen_roles=["teacher"],
        name="Test Invigilator",
        email="test@example.com",
    )


# ---------------------------------------------------------------------------
# I-INVIG-01: WebSocket rejects unauthenticated connections
# ---------------------------------------------------------------------------


def test_ws_rejects_no_auth():
    """I-INVIG-01: WS closes with 4001 when no auth is provided."""
    app = _build_test_app()
    client = TestClient(app)

    with client.websocket_connect("/api/v1/invigilator/ws") as ws:
        # Send a non-auth message
        ws.send_text(json.dumps({"type": "subscribe", "exam_id": "exam-001"}))
        # Server should close the connection
        with pytest.raises(Exception):
            ws.receive_json()


# ---------------------------------------------------------------------------
# I-INVIG-02: WebSocket authenticates via query param
# ---------------------------------------------------------------------------


def test_ws_auth_via_query_param():
    """I-INVIG-02: WS authenticates when valid token in query param."""
    app = _build_test_app()
    client = TestClient(app)

    with patch(
        "src.routes.websocket.validate_token",
        side_effect=_mock_validate_token,
    ):
        with client.websocket_connect(
            "/api/v1/invigilator/ws?token=valid-token-123"
        ) as ws:
            # Send subscribe message
            ws.send_text(json.dumps({
                "type": "subscribe",
                "exam_id": "exam-001",
            }))
            # Should receive subscription confirmation
            msg = ws.receive_json()
            assert msg["event_type"] == "subscribed"
            assert msg["payload"]["exam_id"] == "exam-001"


# ---------------------------------------------------------------------------
# I-INVIG-03: WebSocket authenticates via first message
# ---------------------------------------------------------------------------


def test_ws_auth_via_first_message():
    """I-INVIG-03: WS authenticates when token sent in first message."""
    app = _build_test_app()
    client = TestClient(app)

    with patch(
        "src.routes.websocket.validate_token",
        side_effect=_mock_validate_token,
    ):
        with client.websocket_connect("/api/v1/invigilator/ws") as ws:
            # Send auth message
            ws.send_text(json.dumps({
                "type": "auth",
                "token": "valid-token-456",
            }))
            # Send subscribe
            ws.send_text(json.dumps({
                "type": "subscribe",
                "exam_id": "exam-001",
            }))
            # Should receive subscription confirmation
            msg = ws.receive_json()
            assert msg["event_type"] == "subscribed"
            assert msg["payload"]["exam_id"] == "exam-001"


# ---------------------------------------------------------------------------
# I-INVIG-04: WebSocket sends snapshot after subscribe
# ---------------------------------------------------------------------------


def test_ws_sends_snapshot_after_subscribe():
    """I-INVIG-04: After subscribe, WS pushes a session.snapshot."""
    hub_relay = MagicMock(spec=HubRelay)
    hub_relay.get_latest = MagicMock(return_value={
        "exam_id": "exam-001",
        "pens": [
            {"pen_mac": "AA:BB", "sync_status": "syncing", "total_chunks": 5},
        ],
        "dongles": [
            {"dongle_mac": "DD:01", "status": "healthy", "connected_pens": 1},
        ],
        "wifi": {"connected": True, "ssid": "ExamNet"},
        "upload": {"status": "in_progress", "total_chunks": 10, "acked_chunks": 3},
        "timer": {"remaining_sec": 2700},
    })
    hub_relay.register_listener = AsyncMock()
    hub_relay.unregister_listener = AsyncMock()

    app = _build_test_app(hub_relay=hub_relay)
    client = TestClient(app)

    with patch(
        "src.routes.websocket.validate_token",
        side_effect=_mock_validate_token,
    ):
        with client.websocket_connect(
            "/api/v1/invigilator/ws?token=valid-token"
        ) as ws:
            ws.send_text(json.dumps({
                "type": "subscribe",
                "exam_id": "exam-001",
            }))
            # First message: subscribed confirmation
            msg1 = ws.receive_json()
            assert msg1["event_type"] == "subscribed"

            # Second message: first snapshot push
            msg2 = ws.receive_json()
            assert msg2["event_type"] == "session.snapshot"
            payload = msg2["payload"]
            assert payload["exam_id"] == "exam-001"
            assert payload["timer_remaining_sec"] == 2700
            assert len(payload["pens"]) == 1
            assert len(payload["dongles"]) == 1
            assert payload["wifi"]["connected"] is True
            assert payload["upload_progress"]["acked_chunks"] == 3


# ---------------------------------------------------------------------------
# I-INVIG-05: Snapshot envelope matches OpenAPI WebSocketEnvelope schema
# ---------------------------------------------------------------------------


def test_snapshot_envelope_schema():
    """I-INVIG-05: Snapshot messages match {event_type, payload} shape."""
    app = _build_test_app()
    client = TestClient(app)

    with patch(
        "src.routes.websocket.validate_token",
        side_effect=_mock_validate_token,
    ):
        with client.websocket_connect(
            "/api/v1/invigilator/ws?token=valid-token"
        ) as ws:
            ws.send_text(json.dumps({
                "type": "subscribe",
                "exam_id": "exam-001",
            }))
            # Skip subscribed message
            ws.receive_json()
            # Receive snapshot
            msg = ws.receive_json()

            # Validate envelope shape per WebSocketEnvelope schema
            assert "event_type" in msg
            assert "payload" in msg
            assert msg["event_type"] in (
                "session.snapshot",
                "sync.progress",
                "dongle.health",
                "upload.progress",
            )
            assert isinstance(msg["payload"], dict)


# ---------------------------------------------------------------------------
# I-INVIG-06: Invalid token rejects WebSocket
# ---------------------------------------------------------------------------


def test_ws_rejects_invalid_token():
    """I-INVIG-06: WS closes when an invalid token is provided."""
    app = _build_test_app()
    client = TestClient(app)

    with patch(
        "src.routes.websocket.validate_token",
        side_effect=_mock_validate_token,
    ):
        with client.websocket_connect("/api/v1/invigilator/ws") as ws:
            # Send auth with invalid token (does not start with 'valid-')
            ws.send_text(json.dumps({
                "type": "auth",
                "token": "bad-token",
            }))
            # Server should close the connection
            with pytest.raises(Exception):
                ws.receive_json()
