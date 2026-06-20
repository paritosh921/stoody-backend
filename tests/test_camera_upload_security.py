from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_camera_upload_route_uses_camera_answer_image_policy():
    route = resolve_upload_policy_for_route("POST", "/api/v1/ingest/camera/exam-1/student-1/1")
    policy = get_upload_policy(route.policy_id)

    assert route.policy_id == "camera_answer_image"
    assert policy.allowed_extensions == ("jpg", "jpeg", "png")
    assert policy.allowed_magic_types == ("jpeg", "png")
    assert policy.max_image_pixels == 25_000_000
