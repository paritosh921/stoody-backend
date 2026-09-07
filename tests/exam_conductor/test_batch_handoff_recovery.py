"""Regression tests for an imported mapping with no durable grading child."""
from unittest.mock import AsyncMock, patch

import pytest
from mongomock_motor import AsyncMongoMockClient
from services import exampen_openai_batch as batch


async def seed(db, *, cached=True):
    await db[batch.BATCH_GROUPS_COLLECTION].insert_one({
        'batch_group_id': 'group', 'status': 'completed', 'job_ids': ['job'],
        'completed_count': 0,
    })
    await db[batch.PROCESSING_JOBS_COLLECTION].insert_one({
        'job_id': 'job', 'status': 'importing_batch', 'grading_generation': 1,
    })
    await db[batch.BATCH_PARTS_COLLECTION].insert_one({
        'local_part_id': 'part', 'batch_group_id': 'group', 'status': 'imported',
        'provider_batch_id': 'provider',
        'provider_state': {'status': 'completed', 'output_file_id': 'output'},
    })
    item = {
        'custom_id': 'parent', 'local_part_id': 'part', 'batch_group_id': 'group',
        'job_id': 'job', 'job_generation': 1, 'import_status': 'waiting_recovery',
        'recorded_call_indexes': [0],
    }
    if cached:
        item['provider_result'] = {'response': {'status_code': 200, 'body': {'id': 'response'}}}
    await db[batch.BATCH_ITEMS_COLLECTION].insert_one(item)


@pytest.mark.asyncio
async def test_orphan_replays_cached_output_and_creates_only_one_child():
    db = AsyncMongoMockClient()['test']
    await seed(db)
    client = AsyncMock()
    importer = AsyncMock(return_value={'parent_custom_id': 'parent', 'recorded_call_indexes': [0]})

    async def persist_child(*args, **kwargs):
        await db[batch.BATCH_ITEMS_COLLECTION].insert_one({
            'custom_id': 'child', 'parent_custom_id': 'parent', 'batch_group_id': 'group',
            'import_status': 'pending',
        })
        return 1

    with patch.object(batch, 'OpenAIBatchClient', return_value=client), \
         patch.object(batch, '_import_item', importer), \
         patch.object(batch, '_create_provider_parts', side_effect=persist_child):
        await batch.reconcile_economy_batches(db)
        await batch.reconcile_economy_batches(db)
    assert importer.await_count == 1
    assert importer.call_args.kwargs['item']['recorded_call_indexes'] == [0]
    client.file_content.assert_not_awaited()
    assert await db[batch.BATCH_ITEMS_COLLECTION].count_documents({'parent_custom_id': 'parent'}) == 1
    assert (await db[batch.BATCH_GROUPS_COLLECTION].find_one({}))['status'] == 'importing'


@pytest.mark.asyncio
async def test_refused_handoff_keeps_parent_replayable_and_output_retained():
    db = AsyncMongoMockClient()['test']
    await seed(db)
    client = AsyncMock()
    with patch.object(batch, 'OpenAIBatchClient', return_value=client), \
         patch.object(batch, '_import_item', AsyncMock(return_value={'parent_custom_id': 'parent'})), \
         patch.object(batch, '_create_provider_parts', AsyncMock(return_value=0)):
        await batch.reconcile_economy_batches(db)
    assert (await db[batch.BATCH_PARTS_COLLECTION].find_one({}))['status'] == batch.PART_IMPORTING_STATUS
    client.delete_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_deleted_output_becomes_actionable_failure():
    db = AsyncMongoMockClient()['test']
    await seed(db, cached=False)
    client = AsyncMock()
    client.file_content.side_effect = RuntimeError('No such File object (HTTP 404)')
    with patch.object(batch, 'OpenAIBatchClient', return_value=client):
        await batch.reconcile_economy_batches(db)
    job = await db[batch.PROCESSING_JOBS_COLLECTION].find_one({})
    assert job['status'] == 'batch_failed'
    assert 'fresh economy check' in job['last_error']
    assert (await db[batch.BATCH_GROUPS_COLLECTION].find_one({}))['status'] == 'completed_with_errors'
    client.create_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_generation_is_not_reopened_by_old_orphan():
    db = AsyncMongoMockClient()['test']
    await seed(db)
    await db[batch.PROCESSING_JOBS_COLLECTION].update_one({}, {'$set': {'grading_generation': 2}})
    await batch._recover_orphaned_handoffs(db)
    assert (await db[batch.BATCH_PARTS_COLLECTION].find_one({}))['status'] == 'imported'


@pytest.mark.asyncio
async def test_worker_crash_after_mapping_replays_after_lease_expiry():
    db = AsyncMongoMockClient()['test']
    await seed(db)
    client = AsyncMock()
    importer = AsyncMock(return_value={'parent_custom_id': 'parent', 'recorded_call_indexes': [0]})
    with patch.object(batch, 'OpenAIBatchClient', return_value=client), \
         patch.object(batch, '_import_item', importer), \
         patch.object(batch, '_create_provider_parts', AsyncMock(side_effect=RuntimeError('worker interrupted'))):
        with pytest.raises(RuntimeError, match='worker interrupted'):
            await batch.reconcile_economy_batches(db)
    parent = await db[batch.BATCH_ITEMS_COLLECTION].find_one({'custom_id': 'parent'})
    assert parent['recorded_call_indexes'] == [0]
    assert parent['provider_result']['response']['body']['id'] == 'response'
    client.delete_file.assert_not_awaited()
    await db[batch.BATCH_PARTS_COLLECTION].update_one({}, {'$set': {'import_lease_expires_at': batch._now()}})
    with patch.object(batch, 'OpenAIBatchClient', return_value=client), \
         patch.object(batch, '_import_item', importer), \
         patch.object(batch, '_create_provider_parts', AsyncMock(return_value=0)):
        await batch.reconcile_economy_batches(db)
    assert importer.await_count == 2
    client.file_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_cannot_complete_with_missing_job():
    db = AsyncMongoMockClient()['test']
    await db[batch.BATCH_GROUPS_COLLECTION].insert_one({
        'batch_group_id': 'group', 'status': 'importing', 'job_ids': ['missing'],
    })
    await batch.reconcile_economy_batches(db)
    assert (await db[batch.BATCH_GROUPS_COLLECTION].find_one({}))['status'] == 'importing'
