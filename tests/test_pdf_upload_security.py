from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_pdf_upload_routes_use_pdf_and_template_policies():
    upload_route = resolve_upload_policy_for_route("POST", "/api/v1/pdf/upload")
    template_route = resolve_upload_policy_for_route("POST", "/api/v1/pdf/documents/doc-1/upload-template")
    direct_ocr_route = resolve_upload_policy_for_route("POST", "/api/v1/pdf/direct-ocr")

    assert upload_route.policy_id == "pdf_document"
    assert upload_route.field_policies == {
        "exam_template": "exam_template_file",
        "answer_sheet": "answer_sheet_pdf",
    }
    assert template_route.policy_id == "exam_template_file"
    assert direct_ocr_route.policy_id == "direct_ocr_pdf"


def test_pdf_policies_require_real_pdf_magic():
    for policy_id in ("pdf_document", "answer_sheet_pdf", "direct_ocr_pdf"):
        policy = get_upload_policy(policy_id)
        assert policy.allowed_extensions == ("pdf",)
        assert "pdf" in policy.allowed_magic_types
        assert policy.max_pdf_pages is not None


def test_dcr_template_derivative_uses_private_derived_storage():
    import inspect

    from api.v1.pdf_async import _store_exam_template_file

    source = inspect.getsource(_store_exam_template_file)
    assert "settings.UPLOAD_DERIVED_PREFIX" in source
    assert "PrivateUploadStorage" in source
    assert "uploads/documents/templates" not in source
