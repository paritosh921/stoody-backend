from unittest.mock import AsyncMock, patch

import pytest
from bson import ObjectId
from fastapi import HTTPException
from mongomock_motor import AsyncMongoMockClient

from services.exampen_eligibility import resolve_exam_students, student_exam_visibility


async def setup_exam():
    db = AsyncMongoMockClient()['tenant']
    owner = ObjectId()
    exam = {'exam_id': 'exam', 'prepared_document_id': 'paper', 'admin_id': str(owner),
            'exam_type': 'pcr', 'capture_mode': 'camera', 'roster': ['old'],
            'lifecycle_state': 'uploading', 'answer_copy_upload_state': 'open',
            'student_self_submission_enabled': True}
    await db.documents.insert_one({'document_id': 'paper', 'admin_id': owner, 'standard': '11'})
    await db.exampen_exams.insert_one(dict(exam))
    await db.students.insert_one({'student_id': 'old', 'admin_id': owner, 'grade': '11',
                                 'section': 'A', 'is_active': True})
    return db, exam, owner


@pytest.mark.asyncio
async def test_new_student_visible_and_uploadable_without_any_backfill():
    db, exam, owner = await setup_exam()
    assert (await resolve_exam_students(db, exam))['roster'] == ['old']
    await db.students.insert_one({'student_id': 'new', 'admin_id': owner, 'grade': '11', 'is_active': True})
    resolved = await resolve_exam_students(db, exam)
    assert set(resolved['roster']) == {'old', 'new'}
    assert (await db.exampen_exams.find_one({}))['roster'] == ['old']
    from api.v1.camera_upload_async import _require_camera_upload_context
    from api.v1.evalpen_student_submission_async import _get_student_exam_or_404, _student_upload_availability
    with patch('api.v1.exam_orch_async._require_tutor_visibility', AsyncMock()):
        staff = await _require_camera_upload_context(db, db=None, exam_id='exam', student_id='new',
                                                    current_user={'user_type': 'admin'})
    assert 'new' in staff['roster']
    student = await _get_student_exam_or_404(db, exam_id='exam', student_id='new')
    assert _student_upload_availability(student, 'new')[0]
    assert await db.exampen_exams.count_documents(await student_exam_visibility(db, 'new')) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [{'grade': '12'}, {'is_active': False}, {'admin_id': ObjectId()}])
async def test_ineligible_student_rejected_even_if_saved_roster_includes_them(change):
    db, exam, owner = await setup_exam()
    await db.students.update_one({'student_id': 'old'}, {'$set': change})
    assert (await resolve_exam_students(db, exam))['roster'] == []
    from api.v1.camera_upload_async import _require_camera_upload_context
    with patch('api.v1.exam_orch_async._require_tutor_visibility', AsyncMock()):
        with pytest.raises(HTTPException) as err:
            await _require_camera_upload_context(db, db=None, exam_id='exam', student_id='old', current_user={'user_type': 'admin'})
    assert err.value.status_code == 404


@pytest.mark.asyncio
async def test_section_and_missing_paper_fail_closed():
    db, exam, owner = await setup_exam()
    await db.documents.update_one({}, {'$set': {'section': 'B'}})
    assert (await resolve_exam_students(db, exam))['roster'] == []
    await db.documents.delete_many({})
    assert (await resolve_exam_students(db, exam))['roster'] == []


@pytest.mark.asyncio
async def test_existing_copy_history_survives_class_change_but_cannot_be_uploaded_again():
    db, exam, owner = await setup_exam()
    await db.evalpen_submissions.insert_one({'exam_id': 'exam', 'student_id': 'old', 'submission_id': 'copy'})
    await db.students.update_one({}, {'$set': {'grade': '12'}})
    from api.v1.evalpen_student_submission_async import _get_student_exam_or_404, _student_upload_availability
    visible = await _get_student_exam_or_404(db, exam_id='exam', student_id='old')
    assert not _student_upload_availability(visible, 'old')[0]
    assert await db.exampen_exams.count_documents(await student_exam_visibility(db, 'old')) == 1


@pytest.mark.asyncio
async def test_new_student_does_not_override_closed_uploads():
    db, exam, owner = await setup_exam()
    await db.exampen_exams.update_one({}, {'$set': {'answer_copy_upload_state': 'closed'}})
    from api.v1.camera_upload_async import _require_camera_upload_context
    with patch('api.v1.exam_orch_async._require_tutor_visibility', AsyncMock()):
        with pytest.raises(HTTPException) as err:
            await _require_camera_upload_context(db, db=None, exam_id='exam', student_id='old', current_user={'user_type': 'admin'})
    assert err.value.status_code == 409


@pytest.mark.asyncio
async def test_pen_admission_unchanged():
    db, exam, owner = await setup_exam()
    exam['capture_mode'] = 'pen'
    assert await resolve_exam_students(db, exam) == exam


@pytest.mark.asyncio
async def test_completion_depends_on_submitted_copies_after_close():
    db, exam, owner = await setup_exam()
    from services.exampen_workflow import _maybe_mark_exam_ready_for_review
    await db.evalpen_submissions.insert_one({'exam_id': 'exam', 'student_id': 'late', 'submission_id': 'copy'})
    await db.exampen_processing_jobs.insert_one({'submission_id': 'copy', 'status': 'completed'})
    await _maybe_mark_exam_ready_for_review(db, 'exam')
    assert (await db.exampen_exams.find_one({}))['lifecycle_state'] == 'uploading'
    await db.exampen_exams.update_one({}, {'$set': {'answer_copy_upload_state': 'closed'}})
    await _maybe_mark_exam_ready_for_review(db, 'exam')
    assert (await db.exampen_exams.find_one({}))['lifecycle_state'] == 'ready_for_eval'


@pytest.mark.asyncio
async def test_workspace_list_includes_new_students_and_preserves_departed_submitters():
    db, exam, owner = await setup_exam()
    await db.students.insert_one({'student_id': 'new', 'admin_id': owner, 'grade': '11', 'is_active': True})
    await db.students.update_one({'student_id': 'old'}, {'$set': {'grade': '12'}})
    await db.evalpen_submissions.insert_one({'exam_id': 'exam', 'student_id': 'old',
        'submission_id': 'copy', 'publication_status': 'published', 'source': 'camera'})
    from api.v1.evalpen_review_async import get_exam_roster
    with patch('api.v1.evalpen_review_async._get_tenant_db', AsyncMock(return_value=db)), \
         patch('api.v1.evalpen_review_async._get_tutor_scoped_student_ids', AsyncMock(return_value=None)):
        response = await get_exam_roster('exam', current_user={'user_type': 'admin'}, db=None)
    assert {r.student_id for r in response.expected_students} == {'new', 'old'}
    assert next(r for r in response.expected_students if r.student_id == 'old').status == 'published'


@pytest.mark.asyncio
async def test_scoped_teacher_can_upload_only_their_students():
    db, exam, owner = await setup_exam()
    from api.v1.camera_upload_async import _require_camera_upload_context
    with patch('api.v1.exam_orch_async._require_tutor_visibility', AsyncMock()), \
         patch('api.v1.evalpen_review_async._has_full_exam_access', return_value=False), \
         patch('api.v1.evalpen_review_async._get_tutor_scoped_student_ids', AsyncMock(return_value=[])):
        with pytest.raises(HTTPException) as err:
            await _require_camera_upload_context(db, db=None, exam_id='exam', student_id='old', current_user={'user_type':'tutor'})
    assert err.value.status_code == 403


@pytest.mark.asyncio
async def test_assigned_teacher_upload_accepts_new_class_student():
    db, exam, owner = await setup_exam()
    await db.exampen_exams.update_one({}, {'$set': {'teacher_ids': ['teacher']}})
    await db.students.insert_one({'student_id': 'new', 'admin_id': owner, 'grade': '11', 'is_active': True})
    from api.v1.camera_upload_async import _require_camera_upload_context
    with patch('api.v1.exam_orch_async._require_tutor_visibility', AsyncMock()):
        result = await _require_camera_upload_context(db, db=None, exam_id='exam', student_id='new',
            current_user={'user_type': 'tutor', 'tutor_id': 'teacher'})
    assert 'new' in result['roster']
