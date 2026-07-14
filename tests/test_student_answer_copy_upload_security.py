from core.upload_security.policies import get_upload_policy
from core.upload_security.routes import resolve_upload_policy_for_route


def test_student_answer_copy_route_uses_aggregate_and_field_specific_policies():
    route = resolve_upload_policy_for_route(
        "POST",
        "/api/v1/student/exams/exam-1/answer-copy",
    )

    assert route is not None
    assert route.policy_id == "student_answer_copy_upload"
    assert route.field_policies == {
        "pages": "student_answer_copy_image",
        "answer_pdf": "student_answer_copy_pdf",
    }
    assert get_upload_policy(route.field_policies["pages"]).allowed_magic_types == ("jpeg", "png")
    assert get_upload_policy(route.field_policies["answer_pdf"]).allowed_magic_types == ("pdf",)
