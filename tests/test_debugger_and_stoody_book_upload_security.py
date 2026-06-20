from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_debugger_upload_is_mapped_for_v1_and_legacy_routes():
    assert resolve_upload_policy_for_route("POST", "/api/v1/debugger/upload").policy_id == "debugger_document"
    assert resolve_upload_policy_for_route("POST", "/api/debugger/upload").policy_id == "debugger_document"


def test_stoody_book_pdf_requires_real_pdf_magic():
    route = resolve_upload_policy_for_route("POST", "/api/v1/stoody-book/sessions/session-1/pdfs")
    policy = get_upload_policy(route.policy_id)

    assert route.policy_id == "stoody_book_pdf"
    assert policy.allowed_extensions == ("pdf",)
    assert policy.allowed_magic_types == ("pdf",)
