from __future__ import annotations

import pytest
from bson import ObjectId

from api.v1.notices_async import AudienceSpec, _resolve_recipients


class _DatabaseAdapter:
    def __init__(self, database):
        self.database = database

    async def mongo_find_one(self, collection, query, projection=None):
        return await self.database[collection].find_one(query, projection)

    async def mongo_find(
        self,
        collection,
        query,
        projection=None,
        sort=None,
        skip=0,
        limit=0,
    ):
        cursor = self.database[collection].find(query, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list(length=limit or 5000)


async def _seed_tutor_roster(database):
    admin_id = ObjectId()
    tutor_oid = ObjectId()
    await database["tutors"].insert_one(
        {
            "_id": tutor_oid,
            "tutor_id": "TUT-1",
            "created_by": admin_id,
            "teaching_assignments": [
                {"standard": "6", "sections": ["A"]},
                {"standard": "7", "sections": ["B"]},
            ],
        }
    )
    roster = []
    for grade, section in (("6", "A"), ("6", "B"), ("7", "A"), ("7", "B")):
        document = {
            "_id": ObjectId(),
            "student_id": f"STU-{grade}{section}",
            "name": f"Student {grade}{section}",
            "grade": grade,
            "section": section,
            "admin_id": admin_id,
        }
        roster.append(document)
    await database["students"].insert_many(roster)
    current_user = {
        "user_type": "tutor",
        "user_id": str(tutor_oid),
        "tutor_id": "TUT-1",
        "admin_id": str(admin_id),
    }
    return current_user, roster


@pytest.mark.asyncio
async def test_class_pairs_do_not_expand_to_grade_section_cross_product():
    database = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["notices"]
    current_user, roster = await _seed_tutor_roster(database)
    db = _DatabaseAdapter(database)

    students, tutors = await _resolve_recipients(
        db,
        AudienceSpec(
            type="class",
            recipient_type="student",
            class_pairs=[
                {"grade": "6", "section": "A"},
                {"grade": "7", "section": "B"},
            ],
        ),
        current_user,
    )

    expected = {
        str(student["_id"])
        for student in roster
        if (student["grade"], student["section"]) in {("6", "A"), ("7", "B")}
    }
    assert set(students) == expected
    assert tutors == []


@pytest.mark.asyncio
async def test_individual_business_ids_are_canonicalized_for_notifications():
    database = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["notices"]
    current_user, roster = await _seed_tutor_roster(database)
    db = _DatabaseAdapter(database)
    target = roster[0]

    students, _ = await _resolve_recipients(
        db,
        AudienceSpec(
            type="individual",
            recipient_type="student",
            recipient_ids=[target["student_id"]],
        ),
        current_user,
    )

    assert students == [str(target["_id"])]


@pytest.mark.asyncio
async def test_admin_class_pairs_target_only_teachers_of_exact_classes():
    database = pytest.importorskip("mongomock_motor").AsyncMongoMockClient()["notices"]
    matching_id = ObjectId()
    other_id = ObjectId()
    await database["tutors"].insert_many(
        [
            {
                "_id": matching_id,
                "teaching_assignments": [{"standard": "6", "sections": ["A"]}],
            },
            {
                "_id": other_id,
                "teaching_assignments": [{"standard": "6", "sections": ["B"]}],
            },
        ]
    )

    _, tutors = await _resolve_recipients(
        _DatabaseAdapter(database),
        AudienceSpec(
            type="class",
            recipient_type="tutor",
            class_pairs=[{"grade": "6", "section": "A"}],
        ),
        {"user_type": "admin", "user_id": str(ObjectId())},
    )

    assert tutors == [str(matching_id)]
