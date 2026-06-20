from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_image_logo_and_manual_question_routes_have_image_magic_policies():
    expectations = {
        "/api/v1/images/upload": "generic_image_upload",
        "/api/v1/admin/settings/logo": "school_logo",
        "/api/v1/pdf/questions": "manual_question_image",
    }

    for path, policy_id in expectations.items():
        route = resolve_upload_policy_for_route("POST", path)
        policy = get_upload_policy(policy_id)
        assert route.policy_id == policy_id
        assert {"png", "jpeg"}.issubset(set(policy.allowed_magic_types))
        assert policy.max_image_pixels is not None


def test_registration_and_support_attachment_routes_are_policy_mapped():
    expectations = {
        "/api/v1/auth/admin/register": "registration_document",
        "/auth/admin/register": "registration_document",
        "/api/v1/auth/admin/registration-status-message": "registration_reply_attachment",
        "/api/v1/admin/superadmin-messages": "support_message_attachment",
        "/api/v1/superadmin/tenants/tenant-1/messages": "support_message_attachment",
        "/api/v1/desktop-diagnostics/upload": "desktop_diagnostics_zip",
        "/api/v1/desktop-bug-reports/submit": "desktop_bug_image",
        "/api/v1/teaching-materials/upload": "teaching_material",
    }

    for path, policy_id in expectations.items():
        assert resolve_upload_policy_for_route("POST", path).policy_id == policy_id
