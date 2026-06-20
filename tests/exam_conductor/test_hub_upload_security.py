import pytest
from pydantic import ValidationError

from api.v1.hub_ops_async import HubDataUploadRequest, HubRawSessionUpload
from core.upload_security.routes import resolve_upload_policy_for_route


def test_hub_raw_data_upload_route_uses_structured_policy():
    route = resolve_upload_policy_for_route("POST", "/api/v1/hubs/hub-1/data/upload")

    assert route.policy_id == "hub_raw_data_batch"
    assert resolve_upload_policy_for_route("POST", "/api/v1/hubs/hub-1/commands/pending") is None


def test_hub_raw_data_rejects_oversized_frame_json():
    oversized_frame = {"payload": "x" * (8 * 1024 + 1)}

    with pytest.raises(ValidationError):
        HubRawSessionUpload(
            raw_session_key="session-1",
            session_id="session-1",
            frame_count=1,
            frames=[oversized_frame],
        )


def test_hub_raw_data_rejects_too_many_sessions():
    session = {
        "raw_session_key": "session",
        "session_id": "session",
        "frame_count": 0,
        "frames": [],
    }

    with pytest.raises(ValidationError):
        HubDataUploadRequest(sessions=[{**session, "raw_session_key": f"session-{i}"} for i in range(21)])
