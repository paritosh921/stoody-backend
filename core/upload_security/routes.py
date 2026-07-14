"""Upload route-to-policy mapping used by middleware and coverage tests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .policies import get_upload_policy


@dataclass(frozen=True)
class UploadRoutePolicy:
    method: str
    path_template: str
    policy_id: str
    owner_note: str
    field_policies: dict[str, str] = field(default_factory=dict)

    @property
    def request_limit_bytes(self) -> int:
        policy = get_upload_policy(self.policy_id)
        return policy.max_total_size_bytes or policy.max_size_bytes


def _route(method: str, path_template: str, policy_id: str, owner_note: str, **field_policies: str) -> UploadRoutePolicy:
    return UploadRoutePolicy(
        method=method.upper(),
        path_template=path_template,
        policy_id=policy_id,
        owner_note=owner_note,
        field_policies=field_policies,
    )


UPLOAD_ROUTE_POLICY_MAP: tuple[UploadRoutePolicy, ...] = (
    _route("POST", "/api/v1/auth/admin/register", "registration_document", "Admin registration documents"),
    _route("POST", "/auth/admin/register", "registration_document", "Legacy admin registration documents"),
    _route(
        "POST",
        "/api/v1/auth/admin/registration-status-message",
        "registration_reply_attachment",
        "Registration status reply attachments",
    ),
    _route(
        "POST",
        "/auth/admin/registration-status-message",
        "registration_reply_attachment",
        "Legacy registration status reply attachments",
    ),
    _route("POST", "/api/v1/admin/superadmin-messages", "support_message_attachment", "Admin support messages"),
    _route(
        "POST",
        "/api/v1/superadmin/tenants/{tenant_id}/messages",
        "support_message_attachment",
        "Superadmin tenant messages",
    ),
    _route(
        "POST",
        "/api/v1/pdf/upload",
        "pdf_document",
        "PDF upload with field-specific answer/template policies",
        exam_template="exam_template_file",
        answer_sheet="answer_sheet_pdf",
    ),
    _route(
        "POST",
        "/api/v1/pdf/documents/{document_id}/upload-template",
        "exam_template_file",
        "Document exam template upload",
    ),
    _route("POST", "/api/v1/pdf/direct-ocr", "direct_ocr_pdf", "Direct PDF OCR"),
    _route(
        "POST",
        "/api/v1/exam-tally/question-source/preview",
        "tally_question_source_pdf",
        "Tally question source preview",
    ),
    _route("POST", "/api/v1/pdf/questions", "manual_question_image", "Manual question and option images"),
    _route("POST", "/api/v1/images/upload", "generic_image_upload", "Generic image upload"),
    _route("POST", "/api/v1/admin/settings/logo", "school_logo", "School logo"),
    _route("POST", "/api/v1/teaching-materials/upload", "teaching_material", "Teaching material upload"),
    _route("POST", "/api/v1/desktop-diagnostics/upload", "desktop_diagnostics_zip", "Desktop diagnostics ZIP"),
    _route("POST", "/api/v1/desktop-bug-reports/submit", "desktop_bug_image", "Desktop bug report images"),
    _route("POST", "/api/v1/admin/students/bulk/preview", "bulk_students", "Student bulk preview"),
    _route("POST", "/api/v1/admin/students/bulk/import", "bulk_students", "Student bulk import"),
    _route("POST", "/api/v1/tutor/tutors/bulk/preview", "bulk_tutors", "Tutor bulk preview"),
    _route("POST", "/api/v1/tutor/tutors/bulk/import", "bulk_tutors", "Tutor bulk import"),
    _route("POST", "/api/v1/admin/timetable/bulk-upload/preview", "bulk_timetable", "Timetable bulk preview"),
    _route("POST", "/api/v1/admin/timetable/bulk-upload/import", "bulk_timetable", "Timetable bulk import"),
    _route(
        "POST",
        "/api/v1/ingest/camera/{exam_id}/{student_id}/{page_num}",
        "camera_answer_image",
        "Mobile camera answer image",
    ),
    _route(
        "POST",
        "/api/v1/student/exams/{exam_id}/answer-copy",
        "student_answer_copy_upload",
        "Student-authenticated PCR answer-copy submission",
        pages="student_answer_copy_image",
        answer_pdf="student_answer_copy_pdf",
    ),
    _route("POST", "/api/v1/debugger/upload", "debugger_document", "Authenticated debugger RAG document"),
    _route("POST", "/api/debugger/upload", "debugger_document", "Legacy authenticated debugger RAG document"),
    _route(
        "POST",
        "/api/v1/stoody-book/sessions/{session_id}/pdfs",
        "stoody_book_pdf",
        "Stoody Book session PDF",
    ),
    _route(
        "POST",
        "/api/v1/ingest/strokes/{exam_id}/{pen_mac}/complete",
        "hub_stroke_finalize",
        "Stroke upload finalization",
    ),
    _route(
        "POST",
        "/api/v1/ingest/strokes/{exam_id}/{pen_mac}",
        "hub_stroke_chunk",
        "Stroke upload chunk",
    ),
    _route("POST", "/api/v1/hubs/{hub_id}/data/upload", "hub_raw_data_batch", "Hub raw data upload"),
)


def _template_to_regex(path_template: str) -> re.Pattern[str]:
    escaped = re.escape(path_template)
    pattern = re.sub(r"\\\{[^/]+\\\}", r"[^/]+", escaped)
    return re.compile(f"^{pattern}$")


_COMPILED_ROUTES = tuple(
    (entry, _template_to_regex(entry.path_template), entry.path_template.count("{"), len(entry.path_template))
    for entry in UPLOAD_ROUTE_POLICY_MAP
)


def resolve_upload_policy_for_route(method: str, path: str) -> UploadRoutePolicy | None:
    method = method.upper()
    matches: list[tuple[UploadRoutePolicy, int, int]] = []
    for entry, regex, parameter_count, template_length in _COMPILED_ROUTES:
        if entry.method != method:
            continue
        if regex.match(path):
            matches.append((entry, parameter_count, template_length))
    if not matches:
        return None
    # Prefer exact/static and longer templates, so /complete beats the chunk route.
    matches.sort(key=lambda item: (item[1], -item[2]))
    return matches[0][0]
