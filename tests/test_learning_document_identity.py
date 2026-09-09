from unittest.mock import AsyncMock, patch

import pytest
from bson import ObjectId
from fastapi import HTTPException
from starlette.requests import Request

from api.v1 import learning_async as learning


ADMIN = ObjectId()
STUDENT = ObjectId()
DOCUMENT = {
    '_id': ObjectId(), 'document_id': 'BM001', 'admin_id': ADMIN,
    'document_type': 'Chapter Notes', 'title': 'Properties of Bulk Matter',
    'standard': '11', 'subject': 'Physics', 'course_plan': 'CBSE',
    'file_path': 's3://test/private/notes/bm001.pdf',
}


def database(document=DOCUMENT, *, grade='11', subjects=None, plans=None):
    db = AsyncMock()
    async def find(collection, query):
        if collection == 'students':
            return {'grade': grade, 'subjects': subjects if subjects is not None else ['Physics'],
                    'plan_types': plans if plans is not None else ['CBSE']}
        if query.get('admin_id') != ADMIN:
            return None
        if query.get('document_id') == document['document_id'] or query.get('_id') == document['_id']:
            return document
        return None
    db.mongo_find_one.side_effect = find
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize('identifier', ['BM001', str(DOCUMENT['_id'])])
async def test_public_and_legacy_ids_serve_same_authorized_s3_pdf(identifier):
    db = database()
    user = {'user_id': str(STUDENT), 'user_type': 'student', 'admin_id': str(ADMIN)}
    request = Request({'type': 'http', 'headers': []})
    with patch('api.v1.questions_async.get_admin_id_from_user', return_value=str(ADMIN)), \
         patch.object(learning, 's3_download_file', AsyncMock(return_value=b'%PDF-test')) as read:
        response = await learning.get_chapter_pdf(identifier, request, user, db)
        metadata = await learning.get_document_metadata(identifier, user, db)
    assert response.status_code == 200
    assert response.body == b'%PDF-test'
    assert response.headers['cache-control'] == 'private, no-store'
    assert metadata['data']['title'] == DOCUMENT['title']
    read.assert_awaited_once_with(DOCUMENT['file_path'])


@pytest.mark.asyncio
async def test_public_ids_do_not_cross_admin_scope():
    assert await learning._find_learning_document(database(), 'BM001', admin_id=ObjectId()) is None


@pytest.mark.asyncio
async def test_unknown_non_object_id_returns_404_not_500():
    with patch('api.v1.questions_async.get_admin_id_from_user', return_value=str(ADMIN)):
        with pytest.raises(HTTPException) as error:
            await learning.get_chapter_pdf('unknown-chapter', Request({'type': 'http', 'headers': []}),
                                           {'user_type': 'admin'}, database())
    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize('profile', [{'grade': '12'}, {'subjects': ['Chemistry']}, {'plans': ['JEE']}])
async def test_student_access_rules_apply_before_s3_read(profile):
    with patch('api.v1.questions_async.get_admin_id_from_user', return_value=str(ADMIN)), \
         patch.object(learning, 's3_download_file', AsyncMock()) as read:
        with pytest.raises(HTTPException) as error:
            await learning.get_chapter_pdf('BM001', Request({'type': 'http', 'headers': []}),
                {'user_type': 'student', 'user_id': str(STUDENT)}, database(**profile))
    assert error.value.status_code == 403
    read.assert_not_awaited()


@pytest.mark.asyncio
async def test_b2c_lookup_uses_only_b2c_database():
    db = AsyncMock()
    db.b2c_find_one.return_value = DOCUMENT
    assert await learning._find_learning_document(db, 'BM001', is_b2c=True) == DOCUMENT
    db.b2c_find_one.assert_awaited_once_with('documents', {'document_id': 'BM001'})
    db.mongo_find_one.assert_not_awaited()
