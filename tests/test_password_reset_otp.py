import asyncio
from datetime import datetime, timedelta

from bson import ObjectId


def test_otp_record_hashes_code_and_tracks_role_scope():
    from core.password_reset_otp import PasswordResetOtpManager

    now = datetime(2026, 6, 17, 10, 0, 0)
    manager = PasswordResetOtpManager(length=6, expire_minutes=10)
    created = manager.create_otp_record(
        user_id="student-1",
        email="student@example.com",
        role="student",
        tenant_id="ABCD-1234",
        now=now,
        otp="123456",
    )

    record = created["record"]
    assert created["otp"] == "123456"
    assert record["role"] == "student"
    assert record["tenant_id"] == "ABCD-1234"
    assert record["email"] == "student@example.com"
    assert record["otp_hash"] != "123456"
    assert record["expires_at"] == now + timedelta(minutes=10)
    assert record["attempts"] == 0
    assert record["used"] is False
    assert manager.verify_otp("123456", record["otp_hash"])


def test_wrong_expired_used_and_attempt_exhausted_otps_are_rejected():
    from core.password_reset_otp import PasswordResetOtpManager

    now = datetime(2026, 6, 17, 10, 0, 0)
    manager = PasswordResetOtpManager(length=6, expire_minutes=10, max_attempts=3)
    record = manager.create_otp_record(
        user_id="tutor-1",
        email="tutor@example.com",
        role="tutor",
        tenant_id="ABCD-1234",
        now=now,
        otp="654321",
    )["record"]

    assert manager.validate_record(record, "000000", now=now) == (False, "invalid_otp")
    assert manager.validate_record({**record, "used": True}, "654321", now=now) == (False, "used")
    assert manager.validate_record(record, "654321", now=now + timedelta(minutes=11)) == (False, "expired")
    assert manager.validate_record({**record, "attempts": 3}, "654321", now=now) == (False, "attempts_exhausted")
    assert manager.validate_record(record, "654321", now=now) == (True, "valid")


def test_student_request_searches_students_by_username_and_email_and_sends_to_stored_email(monkeypatch):
    from api.v1 import auth_async

    asyncio.run(_student_request_searches_students_by_username_and_email_and_sends_to_stored_email(monkeypatch, auth_async))


async def _student_request_searches_students_by_username_and_email_and_sends_to_stored_email(monkeypatch, auth_async):
    class FakeCollection:
        def __init__(self, name, docs):
            self.name = name
            self.docs = docs
            self.find_one_calls = []
            self.inserted = []

        async def find_one(self, query, *args, **kwargs):
            self.find_one_calls.append(query)
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        async def insert_one(self, doc):
            self.inserted.append(doc)

        async def count_documents(self, query):
            return 0

    class FakeTenantDb(dict):
        def __init__(self):
            super().__init__(
                students=FakeCollection(
                    "students",
                    [{
                        "_id": ObjectId(),
                        "username_lower": "student1",
                        "username": "Student1",
                        "email": "stored-student@example.com",
                        "is_active": True,
                    }],
                ),
                tutors=FakeCollection("tutors", [{"email": "student1@example.com", "is_active": True}]),
                admins=FakeCollection("admins", [{"email": "student1@example.com", "is_active": True}]),
                password_reset_otps=FakeCollection("password_reset_otps", []),
            )

    sent = []
    tenant_db = FakeTenantDb()

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    async def fake_send_password_reset_otp(*, to_email, otp, username, role, expire_minutes):
        sent.append({"to_email": to_email, "otp": otp, "username": username, "role": role, "expire_minutes": expire_minutes})
        return True

    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
    monkeypatch.setattr(auth_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    response = await auth_async.request_student_password_reset_otp.__wrapped__(
        request=object(),
        reset_data=auth_async.StudentPasswordResetOtpRequest(
            tenant_id="ABCD-1234",
            username="Student1",
            email="stored-student@example.com",
        ),
        db=object(),
    )

    assert response["success"] is True
    assert tenant_db["students"].find_one_calls
    assert tenant_db["tutors"].find_one_calls == []
    assert tenant_db["admins"].find_one_calls == []
    assert tenant_db["students"].find_one_calls == [
        {"email": "stored-student@example.com", "username_lower": "student1", "is_active": True}
    ]
    assert sent[0]["to_email"] == "stored-student@example.com"
    assert sent[0]["role"] == "student"
    assert tenant_db["password_reset_otps"].inserted[0]["role"] == "student"


def test_student_request_respects_existing_otp_cooldown(monkeypatch):
    from api.v1 import auth_async
    from core.password_reset_otp import PasswordResetOtpManager

    asyncio.run(_student_request_respects_existing_otp_cooldown(monkeypatch, auth_async, PasswordResetOtpManager))


async def _student_request_respects_existing_otp_cooldown(monkeypatch, auth_async, PasswordResetOtpManager):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs
            self.inserted = []

        async def find_one(self, query, *args, **kwargs):
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        async def insert_one(self, doc):
            self.inserted.append(doc)

        async def count_documents(self, query):
            return 1

    student_id = ObjectId()
    student = {
        "_id": student_id,
        "username_lower": "student1",
        "email": "stored-student@example.com",
        "is_active": True,
    }
    existing = PasswordResetOtpManager().create_otp_record(
        user_id=str(student_id),
        email="stored-student@example.com",
        role="student",
        tenant_id="ABCD-1234",
        otp="123456",
    )["record"]
    tenant_db = {
        "students": FakeCollection([student]),
        "password_reset_otps": FakeCollection([existing]),
    }
    sent = []

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    async def fake_send_password_reset_otp(*, to_email, otp, username, role, expire_minutes):
        sent.append(to_email)
        return True

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
    monkeypatch.setattr(auth_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    response = await auth_async.request_student_password_reset_otp.__wrapped__(
        request=object(),
        reset_data=auth_async.StudentPasswordResetOtpRequest(
            tenant_id="ABCD-1234",
            username="student1",
            email="stored-student@example.com",
        ),
        db=object(),
    )

    assert response["success"] is True
    assert response["cooldown_seconds"] > 0
    assert response["attempts_remaining"] == 3
    assert sent == []
    assert tenant_db["password_reset_otps"].inserted == []


def test_student_request_blocks_when_latest_otp_is_locked(monkeypatch):
    from api.v1 import auth_async
    from core.password_reset_otp import PasswordResetOtpManager

    asyncio.run(_student_request_blocks_when_latest_otp_is_locked(monkeypatch, auth_async, PasswordResetOtpManager))


async def _student_request_blocks_when_latest_otp_is_locked(monkeypatch, auth_async, PasswordResetOtpManager):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs
            self.inserted = []

        async def find_one(self, query, *args, **kwargs):
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        async def insert_one(self, doc):
            self.inserted.append(doc)

        async def count_documents(self, query):
            return 0

    student_id = ObjectId()
    student = {
        "_id": student_id,
        "username_lower": "student1",
        "email": "stored-student@example.com",
        "is_active": True,
    }
    existing = PasswordResetOtpManager(max_attempts=3).create_otp_record(
        user_id=str(student_id),
        email="stored-student@example.com",
        role="student",
        tenant_id="ABCD-1234",
        otp="123456",
    )["record"]
    existing["attempts"] = 3
    existing["locked_until"] = datetime.utcnow() + timedelta(hours=23)
    tenant_db = {
        "students": FakeCollection([student]),
        "password_reset_otps": FakeCollection([existing]),
    }
    sent = []

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    async def fake_send_password_reset_otp(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
    monkeypatch.setattr(auth_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    response = await auth_async.request_student_password_reset_otp.__wrapped__(
        request=object(),
        reset_data=auth_async.StudentPasswordResetOtpRequest(
            tenant_id="ABCD-1234",
            username="student1",
            email="stored-student@example.com",
        ),
        db=object(),
    )

    assert response["success"] is False
    assert response["message"] == "Too many incorrect codes. Try after 24 hours."
    assert response["attempts_remaining"] == 0
    assert response["locked_until"]
    assert sent == []
    assert tenant_db["password_reset_otps"].inserted == []


def test_student_request_no_records_found_does_not_send_email(monkeypatch):
    from fastapi import HTTPException
    from api.v1 import auth_async

    async def run():
        class FakeCollection:
            def __init__(self):
                self.find_one_calls = []
                self.inserted = []

            async def find_one(self, query, *args, **kwargs):
                self.find_one_calls.append(query)
                return None

            async def insert_one(self, doc):
                self.inserted.append(doc)

        tenant_db = {
            "students": FakeCollection(),
            "password_reset_otps": FakeCollection(),
        }
        sent = []

        async def fake_resolve_tenant(*_args, **_kwargs):
            return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

        async def fake_get_tenant_db(*_args, **_kwargs):
            return tenant_db

        async def fake_send_password_reset_otp(**kwargs):
            sent.append(kwargs)
            return True

        monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)
        monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
        monkeypatch.setattr(auth_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

        try:
            await auth_async.request_student_password_reset_otp.__wrapped__(
                request=object(),
                reset_data=auth_async.StudentPasswordResetOtpRequest(
                    tenant_id="ABCD-1234",
                    username="wrong-student",
                    email="wrong@example.com",
                ),
                db=object(),
            )
            assert False, "expected HTTPException"
        except HTTPException as exc:
            assert exc.status_code == 404
            assert exc.detail == "No records found"

        assert sent == []
        assert tenant_db["password_reset_otps"].inserted == []

    asyncio.run(run())


def test_tutor_request_searches_tutors_only(monkeypatch):
    from api.v1 import auth_async

    asyncio.run(_role_email_request_searches_only_target_collection(monkeypatch, auth_async, role="tutor"))


def test_admin_request_searches_admins_only(monkeypatch):
    from api.v1 import auth_async

    asyncio.run(_role_email_request_searches_only_target_collection(monkeypatch, auth_async, role="admin"))


def test_role_request_no_records_found_does_not_send_email(monkeypatch):
    from fastapi import HTTPException
    from api.v1 import auth_async

    asyncio.run(_role_request_no_records_found_does_not_send_email(monkeypatch, auth_async, role="tutor", HTTPException=HTTPException))


async def _role_request_no_records_found_does_not_send_email(monkeypatch, auth_async, role, HTTPException):
    class FakeCollection:
        def __init__(self):
            self.inserted = []

        async def find_one(self, query, *args, **kwargs):
            return None

        async def insert_one(self, doc):
            self.inserted.append(doc)

    tenant_db = {
        "tutors": FakeCollection(),
        "admins": FakeCollection(),
        "password_reset_otps": FakeCollection(),
    }
    sent = []

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    async def fake_send_password_reset_otp(**kwargs):
        sent.append(kwargs)
        return True

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
    monkeypatch.setattr(auth_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    try:
        await auth_async.request_tutor_password_reset_otp.__wrapped__(
            object(),
            auth_async.RolePasswordResetOtpRequest(
                tenant_id="ABCD-1234",
                username="wrong-tutor",
                email="wrong@example.com",
            ),
            object(),
        )
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "No records found"

    assert sent == []
    assert tenant_db["password_reset_otps"].inserted == []


async def _role_email_request_searches_only_target_collection(monkeypatch, auth_async, role):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs
            self.find_one_calls = []
            self.inserted = []

        async def find_one(self, query, *args, **kwargs):
            self.find_one_calls.append(query)
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        async def insert_one(self, doc):
            self.inserted.append(doc)

        async def count_documents(self, query):
            return 0

    target_doc = {
        "_id": ObjectId(),
        "email": f"{role}@example.com",
        "username": f"{role}1",
        "username_lower": f"{role}1",
        "full_name": f"{role.title()} User",
        "is_active": True,
    }
    tenant_db = {
        "students": FakeCollection([{"email": f"{role}@example.com", "is_active": True}]),
        "tutors": FakeCollection([target_doc] if role == "tutor" else []),
        "admins": FakeCollection([target_doc] if role == "admin" else []),
        "password_reset_otps": FakeCollection([]),
    }
    sent = []

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    async def fake_send_password_reset_otp(*, to_email, otp, username, role, expire_minutes):
        sent.append({"to_email": to_email, "role": role})
        return True

    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
    monkeypatch.setattr(auth_async, "send_password_reset_otp_email", fake_send_password_reset_otp)

    request_model = auth_async.RolePasswordResetOtpRequest(
        tenant_id="ABCD-1234",
        username=f"{role}1",
        email=f"{role}@example.com",
    )
    if role == "tutor":
        response = await auth_async.request_tutor_password_reset_otp.__wrapped__(object(), request_model, object())
    else:
        response = await auth_async.request_admin_password_reset_otp.__wrapped__(object(), request_model, object())

    assert response["success"] is True
    assert tenant_db["students"].find_one_calls == []
    assert bool(tenant_db["tutors"].find_one_calls) is (role == "tutor")
    assert bool(tenant_db["admins"].find_one_calls) is (role == "admin")
    assert sent[0]["to_email"] == f"{role}@example.com"
    assert sent[0]["role"] == role


def test_auth_router_exposes_role_scoped_otp_paths_and_removes_generic_email_link_paths():
    from api.v1 import auth_async

    paths = {route.path for route in auth_async.router.routes}

    assert "/student/password-reset/request" in paths
    assert "/student/password-reset/complete" in paths
    assert "/tutor/password-reset/request" in paths
    assert "/tutor/password-reset/complete" in paths
    assert "/admin/password-reset/request" in paths
    assert "/admin/password-reset/complete" in paths

    assert "/password-reset/request" not in paths
    assert "/password-reset/verify" not in paths
    assert "/password-reset/complete" not in paths


def test_student_complete_resets_only_student_password_and_consumes_otp(monkeypatch):
    from api.v1 import auth_async
    from core.auth import AuthManager
    from core.password_reset_otp import PasswordResetOtpManager

    asyncio.run(_student_complete_resets_only_student_password_and_consumes_otp(monkeypatch, auth_async, AuthManager, PasswordResetOtpManager))


async def _student_complete_resets_only_student_password_and_consumes_otp(monkeypatch, auth_async, AuthManager, PasswordResetOtpManager):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs
            self.update_calls = []

        async def find_one(self, query, *args, **kwargs):
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        async def update_one(self, query, update):
            self.update_calls.append((query, update))
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    if "$set" in update:
                        doc.update(update["$set"])
                    if "$inc" in update:
                        for key, value in update["$inc"].items():
                            doc[key] = doc.get(key, 0) + value
                    return

    student_id = ObjectId()
    student = {
        "_id": student_id,
        "username_lower": "student1",
        "email": "student@example.com",
        "is_active": True,
        "password_hash": "old",
    }
    otp_record = PasswordResetOtpManager().create_otp_record(
        user_id=str(student_id),
        email="student@example.com",
        role="student",
        tenant_id="ABCD-1234",
        otp="123456",
    )["record"]
    otp_record["_id"] = ObjectId()
    tenant_db = {
        "students": FakeCollection([student]),
        "password_reset_otps": FakeCollection([otp_record]),
    }

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)
    auth_manager = AuthManager()

    response = await auth_async.complete_student_password_reset_otp.__wrapped__(
        object(),
        auth_async.StudentPasswordResetOtpCompleteRequest(
            tenant_id="ABCD-1234",
            username="student1",
            email="student@example.com",
            otp="123456",
            new_password="new-password-123",
        ),
        object(),
        auth_manager,
    )

    assert response["success"] is True
    assert auth_manager.verify_password("new-password-123", student["password_hash"])
    assert student["password_reset_requested"] is False
    assert student["requires_password_change"] is False
    assert otp_record["used"] is True


def test_student_complete_locks_after_third_invalid_otp(monkeypatch):
    from fastapi import HTTPException
    from api.v1 import auth_async
    from core.auth import AuthManager
    from core.password_reset_otp import PasswordResetOtpManager

    asyncio.run(_student_complete_locks_after_third_invalid_otp(monkeypatch, auth_async, AuthManager, PasswordResetOtpManager, HTTPException))


async def _student_complete_locks_after_third_invalid_otp(monkeypatch, auth_async, AuthManager, PasswordResetOtpManager, HTTPException):
    class FakeCollection:
        def __init__(self, docs):
            self.docs = docs

        async def find_one(self, query, *args, **kwargs):
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    return doc
            return None

        async def update_one(self, query, update):
            for doc in self.docs:
                if all(doc.get(k) == v for k, v in query.items()):
                    if "$inc" in update:
                        for key, value in update["$inc"].items():
                            doc[key] = doc.get(key, 0) + value
                    if "$set" in update:
                        doc.update(update["$set"])

    student_id = ObjectId()
    student = {
        "_id": student_id,
        "username_lower": "student1",
        "email": "student@example.com",
        "is_active": True,
        "password_hash": "old",
    }
    otp_record = PasswordResetOtpManager(max_attempts=3).create_otp_record(
        user_id=str(student_id),
        email="student@example.com",
        role="student",
        tenant_id="ABCD-1234",
        otp="123456",
    )["record"]
    otp_record["_id"] = ObjectId()
    otp_record["attempts"] = 2
    tenant_db = {
        "students": FakeCollection([student]),
        "password_reset_otps": FakeCollection([otp_record]),
    }

    async def fake_resolve_tenant(*_args, **_kwargs):
        return {"tenant_id": "ABCD-1234", "db_name": "tenant"}

    async def fake_get_tenant_db(*_args, **_kwargs):
        return tenant_db

    monkeypatch.setattr(auth_async, "_resolve_tenant_for_auth", fake_resolve_tenant)
    monkeypatch.setattr(auth_async, "_get_tenant_db_or_503", fake_get_tenant_db)

    try:
        await auth_async.complete_student_password_reset_otp.__wrapped__(
            object(),
            auth_async.StudentPasswordResetOtpCompleteRequest(
                tenant_id="ABCD-1234",
                username="student1",
                email="student@example.com",
                otp="000000",
                new_password="new-password-123",
            ),
            object(),
            AuthManager(),
        )
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 429
        assert exc.detail["message"] == "Too many incorrect codes. Try after 24 hours."
        assert exc.detail["attempts_remaining"] == 0
        assert exc.detail["locked_until"]

    assert otp_record["attempts"] == 3
    assert otp_record["locked_until"] > datetime.utcnow()
