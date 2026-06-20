from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_bulk_upload_routes_use_spreadsheet_policies():
    expectations = {
        "/api/v1/admin/students/bulk/preview": "bulk_students",
        "/api/v1/admin/students/bulk/import": "bulk_students",
        "/api/v1/tutor/tutors/bulk/preview": "bulk_tutors",
        "/api/v1/tutor/tutors/bulk/import": "bulk_tutors",
        "/api/v1/admin/timetable/bulk-upload/preview": "bulk_timetable",
        "/api/v1/admin/timetable/bulk-upload/import": "bulk_timetable",
    }

    for path, policy_id in expectations.items():
        route = resolve_upload_policy_for_route("POST", path)
        policy = get_upload_policy(policy_id)
        assert route.policy_id == policy_id
        assert set(policy.allowed_magic_types) == {"csv", "zip", "ole"}
        assert policy.max_rows is not None
        assert policy.max_columns is not None
