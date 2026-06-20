from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_tally_question_source_preview_uses_pdf_policy():
    route = resolve_upload_policy_for_route("POST", "/api/v1/exam-tally/question-source/preview")
    policy = get_upload_policy(route.policy_id)

    assert route.policy_id == "tally_question_source_pdf"
    assert policy.allowed_extensions == ("pdf",)
    assert policy.allowed_magic_types == ("pdf",)
    assert policy.max_pdf_pages == 100
